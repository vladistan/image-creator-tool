"""Image-to-text style extraction and a provider-agnostic style library.

`extract_style` asks a vision LLM (OpenAI GPT-4o or Gemini Vision) to describe
the *visual style* of a reference image as a concise, comma-separated descriptor
suitable for prepending to any text-to-image generation prompt. The style library
persists those descriptors as plain-text files under the config directory so they
can be reused across providers via the `generate --style <name>` flag.

This module is the domain layer — it never imports typer. CLI concerns live in
`commands/style.py`, which converts the domain exceptions raised here into exit
codes.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sentry_sdk
import structlog

from image_creator_tool.config import CONFIG_DIR
from image_creator_tool.errors import PermanentAPIError, TransientAPIError

if TYPE_CHECKING:
    from collections.abc import Sequence

log = structlog.get_logger()

STYLES_DIR = CONFIG_DIR / "styles"

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_DEFAULT_VISION_MODELS = {"openai": "gpt-4o", "gemini": "gemini-2.0-flash"}

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_BACKOFF_S = 2.0

_STYLE_INSTRUCTION = (
    "You are an expert art director. Analyze the provided image and describe ONLY its "
    "visual style — medium, colour palette, lighting, texture, mood, and artistic "
    "technique. Respond with a single concise, comma-separated list of style descriptors "
    "suitable for prepending to a text-to-image generation prompt. Do NOT describe the "
    "subject matter or objects in the image. Example: 'oil painting, warm sunset palette, "
    "impressionist brushwork, soft diffused lighting'. Respond with the descriptors only, "
    "no preamble and no sentences."
)

_MEMBER_STYLE_INSTRUCTION = (
    "You are an expert art director. Analyze this image and describe ONLY its visual "
    "style — medium, and specifically NAME every distinct colour you see (dominant AND "
    "accent colours, e.g. 'deep violet, orange, magenta, neon green'), plus lighting, "
    "texture (including grain), flatness vs rendering, use of negative space, and mood. "
    "Be specific about the palette; do not omit accent colours. Ignore subject matter. "
    "Respond with a single concise, comma-separated list of style descriptors. "
    "Descriptors only, no preamble, no sentences."
)

_COMBINE_INSTRUCTION = (
    "You are an expert art director. Below are style descriptions, each extracted from "
    "one image of a single cohesive style set:\n{descriptions}\n\nMerge them into ONE "
    "comma-separated style descriptor that captures the COMPLETE shared style. Preserve "
    "the UNION of every distinct colour named across the descriptions — do NOT drop "
    "accent colours or collapse the palette to only the two most common hues. Also keep "
    "the shared medium, texture/grain, flatness, negative space, lighting, and mood. "
    "Respond with the merged descriptor only — no preamble, no sentences, no explanation."
)

_ASSESS_INSTRUCTION = (
    "You are a meticulous art director scoring STYLE fidelity. The first {n} image(s) are "
    "the TARGET STYLE reference set. The LAST image is a CANDIDATE generated from the "
    "style descriptor: '{style}'. Judge ONLY how well the candidate matches the reference "
    "set's VISUAL STYLE (palette discipline, flatness vs rendering, texture/grain, "
    "negative space, linework, mood) — IGNORE differences in subject matter. Respond with "
    "ONLY a JSON object, no prose, of the form: "
    '{{"score": <integer 0-100>, "critique": "<one sentence: what diverges from the '
    'target style>", "suggestions": "<concrete style-descriptor changes to close the '
    'gap>"}}.'
)

_REFINE_INSTRUCTION = (
    "You are an expert art director refining a text-to-image STYLE descriptor. Current "
    "descriptor:\n'{style}'\n\nCritiques of images generated with it (how they diverged "
    "from the target style):\n{critiques}\n\nRewrite the descriptor to close the gap: "
    "tighten the colour palette if it drifted too broad/saturated, adjust flatness, "
    "texture, negative space, and mood, and remove loaded terms that push the model the "
    "wrong way. Respond with ONLY the improved comma-separated style descriptor — no "
    "preamble, no sentences, no explanation."
)


# --- Vision LLM style extraction (Step 3.1) ---------------------------------


def extract_style(
    image_path: str | Path,
    *,
    provider: str = "openai",
    vision_model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Extract a prompt-ready style descriptor from a reference image.

    Args:
        image_path: Path to the reference image.
        provider: Vision backend, "openai" (GPT-4o) or "gemini" (Gemini Vision).
        vision_model: Override the default vision model for the provider.
        api_key: Explicit API key; falls back to the provider's env var.

    Returns:
        A concise, comma-separated style description (e.g.
        "oil painting, warm tones, impressionist brushwork").

    Raises:
        PermanentAPIError: Missing image, unknown provider, missing key, or an
            empty model response.
        TransientAPIError: The vision API kept failing after retries.
    """
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise PermanentAPIError(f"Reference image not found: {path}")

    model = vision_model or _DEFAULT_VISION_MODELS.get(provider)
    if model is None:
        raise PermanentAPIError(
            f"Unknown vision provider '{provider}'. Supported: "
            f"{sorted(_DEFAULT_VISION_MODELS)}"
        )

    with sentry_sdk.start_transaction(op="style.extract", name="extract_style") as txn:
        txn.set_tag("provider", provider)
        txn.set_tag("vision_model", model)
        raw = _vision_chat([_encode_image(path)], _STYLE_INSTRUCTION, provider, model, api_key)

    style = _clean_style_text(raw)
    if not style:
        raise PermanentAPIError("Vision model returned an empty style description")
    return style


def extract_style_group(
    image_paths: Sequence[str | Path],
    *,
    provider: str = "openai",
    vision_model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Extract ONE unified style descriptor common to a set of reference images.

    Sends all images in a single vision request so the model describes the shared
    style (including deliberately withheld constraints like a limited palette),
    not any single image's quirks.

    Raises:
        PermanentAPIError: No images given, a missing file, unknown provider,
            missing key, or an empty model response.
        TransientAPIError: The vision API kept failing after retries.
    """
    if not image_paths:
        raise PermanentAPIError("extract_style_group requires at least one image")
    paths = [Path(p).expanduser() for p in image_paths]
    for p in paths:
        if not p.is_file():
            raise PermanentAPIError(f"Reference image not found: {p}")

    model = vision_model or _DEFAULT_VISION_MODELS.get(provider)
    if model is None:
        raise PermanentAPIError(
            f"Unknown vision provider '{provider}'. Supported: {sorted(_DEFAULT_VISION_MODELS)}"
        )

    with sentry_sdk.start_transaction(op="style.extract_group", name="extract_style_group") as txn:
        txn.set_tag("provider", provider)
        txn.set_tag("vision_model", model)
        txn.set_data("image_count", len(paths))
        # Per-image extraction preserves each image's full palette; a single
        # all-at-once call summarizes to the common denominator and drops accents.
        per_image = []
        for p in paths:
            raw = _vision_chat(
                [_encode_image(p)], _MEMBER_STYLE_INSTRUCTION, provider, model, api_key
            )
            cleaned = _clean_style_text(raw)
            if cleaned:
                per_image.append(cleaned)
        if not per_image:
            raise PermanentAPIError("Vision model returned no per-image style descriptions")
        if len(per_image) == 1:
            return per_image[0]
        return combine_style_descriptions(
            per_image, provider=provider, vision_model=model, api_key=api_key
        )


def combine_style_descriptions(
    descriptions: list[str],
    *,
    provider: str = "openai",
    vision_model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Merge per-image style descriptions into one, preserving the palette union.

    Raises:
        PermanentAPIError: No descriptions, unknown provider, missing key, or an
            empty response.
        TransientAPIError: The API kept failing after retries.
    """
    if not descriptions:
        raise PermanentAPIError("combine_style_descriptions requires at least one description")
    model = vision_model or _DEFAULT_VISION_MODELS.get(provider)
    if model is None:
        raise PermanentAPIError(
            f"Unknown vision provider '{provider}'. Supported: {sorted(_DEFAULT_VISION_MODELS)}"
        )
    joined = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(descriptions))
    instruction = _COMBINE_INSTRUCTION.format(descriptions=joined)
    raw = _vision_chat([], instruction, provider, model, api_key)
    merged = _clean_style_text(raw)
    if not merged:
        raise PermanentAPIError("Combine step returned an empty style description")
    return merged


def assess_style_fidelity(
    source_paths: Sequence[str | Path],
    candidate_path: str | Path,
    style: str,
    *,
    provider: str = "openai",
    vision_model: str | None = None,
    api_key: str | None = None,
) -> tuple[int, str]:
    """Score how well `candidate_path` matches the source set's visual style.

    Returns (score 0-100, critique-with-suggestions). The vision model compares
    the candidate against the reference set, judging style only (not subject).

    Raises:
        PermanentAPIError: Missing files, unknown provider, missing key, or a
            malformed model response.
        TransientAPIError: The vision API kept failing after retries.
    """
    if not source_paths:
        raise PermanentAPIError("assess_style_fidelity requires at least one source image")
    sources = [Path(p).expanduser() for p in source_paths]
    candidate = Path(candidate_path).expanduser()
    for p in [*sources, candidate]:
        if not p.is_file():
            raise PermanentAPIError(f"Image not found: {p}")

    model = vision_model or _DEFAULT_VISION_MODELS.get(provider)
    if model is None:
        raise PermanentAPIError(
            f"Unknown vision provider '{provider}'. Supported: {sorted(_DEFAULT_VISION_MODELS)}"
        )

    candidate_enc = _encode_image(candidate)
    instruction = _ASSESS_INSTRUCTION.format(n=1, style=style)
    with sentry_sdk.start_transaction(op="style.assess", name="assess_style_fidelity") as txn:
        txn.set_tag("provider", provider)
        txn.set_tag("vision_model", model)
        txn.set_data("source_count", len(sources))
        # One call per source image: pairwise scoring is more discriminating and
        # yields source-specific critiques a single blended call would wash out.
        scored = []
        for src in sources:
            raw = _vision_chat(
                [_encode_image(src), candidate_enc], instruction, provider, model, api_key
            )
            scored.append(_parse_assessment(raw))

    scores = [s for s, _ in scored]
    mean_score = round(sum(scores) / len(scores))
    critiques = [c for _, c in scored if c]
    note = " | ".join(dict.fromkeys(critiques))  # dedupe, preserve order
    return mean_score, note


def refine_style_text(
    style: str,
    critiques: list[str],
    *,
    provider: str = "openai",
    vision_model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Rewrite a style descriptor to address critiques (text-only LLM call).

    Raises:
        PermanentAPIError: Unknown provider, missing key, or an empty response.
        TransientAPIError: The API kept failing after retries.
    """
    model = vision_model or _DEFAULT_VISION_MODELS.get(provider)
    if model is None:
        raise PermanentAPIError(
            f"Unknown vision provider '{provider}'. Supported: {sorted(_DEFAULT_VISION_MODELS)}"
        )

    joined = "\n".join(f"- {c}" for c in critiques if c) or "- (no specific critique)"
    instruction = _REFINE_INSTRUCTION.format(style=style, critiques=joined)
    with sentry_sdk.start_transaction(op="style.refine", name="refine_style_text") as txn:
        txn.set_tag("provider", provider)
        txn.set_tag("vision_model", model)
        raw = _vision_chat([], instruction, provider, model, api_key)

    refined = _clean_style_text(raw)
    if not refined:
        raise PermanentAPIError("Refinement model returned an empty style description")
    return refined


def _parse_assessment(raw: str) -> tuple[int, str]:
    """Parse the assessor's JSON response into (score, critique+suggestions)."""
    try:
        data = json.loads(_clean_style_text(raw))
    except (json.JSONDecodeError, ValueError) as e:
        raise PermanentAPIError(f"Assessor returned non-JSON response: {raw[:200]}") from e
    try:
        score = int(data["score"])
    except (KeyError, TypeError, ValueError) as e:
        raise PermanentAPIError(f"Assessment missing integer 'score': {raw[:200]}") from e
    score = max(0, min(100, score))
    critique = str(data.get("critique", "")).strip()
    suggestions = str(data.get("suggestions", "")).strip()
    note = " ".join(part for part in [critique, suggestions] if part)
    return score, note


def _encode_image(path: Path) -> tuple[str, str]:
    """Return (base64-encoded bytes, MIME type) for the image at `path`."""
    b64 = base64.b64encode(path.read_bytes()).decode()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return b64, mime


def _clean_style_text(text: str) -> str:
    """Strip markdown fences and surrounding whitespace from a model response."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    return cleaned


def _resolve_key(provider: str, api_key: str | None) -> str:
    """Resolve the vision API key from the explicit arg or the provider env var."""
    env_var = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    key = api_key or os.environ.get(env_var)
    if not key:
        raise PermanentAPIError(
            f"{provider} vision requires an API key. Set {env_var} or pass api_key."
        )
    return key


def _vision_chat(
    images: list[tuple[str, str]],
    instruction: str,
    provider: str,
    model: str,
    api_key: str | None,
) -> str:
    """Send `instruction` plus zero or more (base64, mime) images to a vision LLM.

    An empty image list yields a text-only call (used by style refinement).
    """
    if provider == "openai":
        return _chat_openai(images, instruction, model, api_key)
    return _chat_gemini(images, instruction, model, api_key)


def _chat_openai(
    images: list[tuple[str, str]], instruction: str, model: str, api_key: str | None
) -> str:
    """Call the OpenAI chat completions endpoint with text + N images; return the text."""
    key = _resolve_key("openai", api_key)
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    for b64, mime in images:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 400,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    data = _post_with_retry(
        _OPENAI_URL, payload, headers, span_data={"provider": "openai", "model": model}
    )

    choices = data.get("choices", [])
    if not choices:
        raise PermanentAPIError(f"No choices in vision response: {data}")
    return str(choices[0].get("message", {}).get("content", ""))


def _chat_gemini(
    images: list[tuple[str, str]], instruction: str, model: str, api_key: str | None
) -> str:
    """Call the Gemini multimodal endpoint with text + N images; return the text."""
    key = _resolve_key("gemini", api_key)
    parts: list[dict[str, Any]] = [{"text": instruction}]
    for b64, mime in images:
        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
    payload = {"contents": [{"parts": parts}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    data = _post_with_retry(
        _GEMINI_URL.format(model=model),
        payload,
        headers,
        span_data={"provider": "gemini", "model": model},
    )

    candidates = data.get("candidates", [])
    if not candidates:
        feedback = data.get("promptFeedback", {})
        raise PermanentAPIError(f"No candidates in vision response. Feedback: {feedback}")
    parts_out = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts_out)


def _post_with_retry(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    span_data: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST JSON with exponential backoff retry on transient failures."""
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return _http_post_json(url, payload, headers, span_data=span_data)
        except TransientAPIError as e:
            last_err = e
            if attempt == _MAX_RETRIES - 1:
                break
            delay = _BASE_BACKOFF_S * (2**attempt)
            context = span_data or {}
            log.warning(
                "transient vision API error, retrying",
                attempt=attempt + 1,
                delay_s=delay,
                provider=context.get("provider"),
                model=context.get("model"),
                error=str(e)[:120],
            )
            if attempt >= 1:
                sentry_sdk.capture_message(
                    f"style extraction retry attempt {attempt + 1} "
                    f"({context.get('provider')}/{context.get('model')})",
                    level="warning",
                )
            time.sleep(delay)
    raise TransientAPIError(f"Gave up after {_MAX_RETRIES} retries. Last error: {last_err}")


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int = 120,
    *,
    span_data: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Single JSON POST, classifying HTTP failures as transient or permanent."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with sentry_sdk.start_span(op="http.client", description="style.extract") as span:
        for key, value in (span_data or {}).items():
            span.set_data(key, value)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return dict(json.loads(resp.read()))
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code in _RETRYABLE_STATUS:
                raise TransientAPIError(f"HTTP {e.code}: {body[:400]}") from e
            raise PermanentAPIError(f"HTTP {e.code}: {body[:400]}") from e
        except urllib.error.URLError as e:
            raise TransientAPIError(f"Network error: {e}") from e


# --- Style library (Step 3.2) -----------------------------------------------


def save_style(name: str, description: str, *, overwrite: bool = False) -> Path:
    """Persist a style description to `<styles-dir>/<slug>.txt`.

    Raises:
        PermanentAPIError: The name slugifies to empty, or a style with the same
            slug already exists and `overwrite` is False.
    """
    path = _style_path(name)
    if path.exists() and not overwrite:
        raise PermanentAPIError(
            f"Style '{path.stem}' already exists. Pass overwrite=True to replace it."
        )
    STYLES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(description.strip() + "\n")
    return path


def load_style(name: str) -> str:
    """Return the saved description for `name`.

    Raises:
        PermanentAPIError: No saved style matches the slugified name.
    """
    path = _style_path(name)
    if not path.is_file():
        raise PermanentAPIError(
            f"Style '{path.stem}' not found. Available: {list_styles()}"
        )
    return path.read_text().strip()


def list_styles() -> list[str]:
    """Return the sorted slugs of all saved styles."""
    if not STYLES_DIR.is_dir():
        return []
    return sorted(p.stem for p in STYLES_DIR.glob("*.txt"))


def delete_style(name: str) -> None:
    """Remove a saved style.

    Raises:
        PermanentAPIError: No saved style matches the slugified name.
    """
    path = _style_path(name)
    if not path.is_file():
        raise PermanentAPIError(
            f"Style '{path.stem}' not found. Available: {list_styles()}"
        )
    path.unlink()


def _style_path(name: str) -> Path:
    """Map a user-supplied style name to its on-disk `.txt` path."""
    return STYLES_DIR / f"{_slugify(name)}.txt"


def _slugify(name: str) -> str:
    """Lowercase to an alphanumeric-with-hyphens slug; reject empty results."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise PermanentAPIError(f"Invalid style name: {name!r}")
    return slug
