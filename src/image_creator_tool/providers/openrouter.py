"""OpenRouter image generation provider via chat completions API.

Uses modalities: ["text", "image"] to request image output.
Images returned in message.images field (not content).
Authenticates via api_key in config profile or OPENROUTER_API_KEY env var.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

import sentry_sdk
import structlog

from image_creator_tool.errors import PermanentAPIError, TransientAPIError
from image_creator_tool.providers.base import GenerationParams, Provider

log = structlog.get_logger()

_API_BASE = "https://openrouter.ai/api/v1"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_BACKOFF_S = 2.0
_REQUEST_TIMEOUT_S = 300
_DOWNLOAD_TIMEOUT_S = 60
_ERROR_TRUNCATE_LEN = 120
_BODY_TRUNCATE_LEN = 400
_CONTENT_TRUNCATE_LEN = 200


class OpenRouterProvider(Provider):
    """OpenRouter provider using chat completions with image modality.

    API key from: config profile api_key > OPENROUTER_API_KEY env var.
    """

    name: ClassVar[str] = "openrouter"
    default_model: ClassVar[str] = "google/gemini-2.5-flash-image"
    MODELS: ClassVar[dict[str, str]] = {
        # Gemini
        "flash": "google/gemini-2.5-flash-image",
        "pro": "google/gemini-3-pro-image-preview",
        "flash-3.1": "google/gemini-3.1-flash-image-preview",
        # OpenAI
        "gpt-5": "openai/gpt-5-image",
        "gpt-5-mini": "openai/gpt-5-image-mini",
        "gpt-5.4": "openai/gpt-5.4-image-2",
        # FLUX (Black Forest Labs)
        "flux-max": "black-forest-labs/flux.2-max",
        "flux-klein": "black-forest-labs/flux.2-klein-4b",
        # Recraft
        "recraft-pro": "recraft/recraft-v4.1-pro",
        "recraft-vector": "recraft/recraft-v4.1-pro-vector",
        "recraft-utility": "recraft/recraft-v4.1-utility-pro",
        # Grok
        "grok": "x-ai/grok-imagine-image-quality",
        # Seedream
        "seedream": "bytedance-seed/seedream-4.5",
        # Riverflow
        "riverflow-pro": "sourceful/riverflow-v2-pro",
        "riverflow-fast": "sourceful/riverflow-v2-fast",
    }

    # Models that are image-only (don't need modalities param)
    _IMAGE_ONLY_MODELS: ClassVar[set[str]] = {
        "black-forest-labs/flux.2-max",
        "black-forest-labs/flux.2-klein-4b",
        "recraft/recraft-v4.1-pro",
        "recraft/recraft-v4.1-pro-vector",
        "recraft/recraft-v4.1-vector",
        "recraft/recraft-v4.1-utility-pro",
        "recraft/recraft-v4.1-utility",
        "x-ai/grok-imagine-image-quality",
        "bytedance-seed/seedream-4.5",
        "sourceful/riverflow-v2-pro",
        "sourceful/riverflow-v2-fast",
        "sourceful/riverflow-v2-standard-preview",
    }

    def __init__(self, api_key: str = "", **_kwargs: str) -> None:
        self._api_key = api_key

    def get_api_key(self) -> str | None:
        """Get API key from config, then fall back to env var."""
        return self._api_key or os.environ.get("OPENROUTER_API_KEY")

    def generate(self, prompt: str, *, params: GenerationParams) -> bytes:
        """Generate or edit an image via OpenRouter chat completions API.

        All OpenRouter image models accept image input via multimodal content
        parts alongside text prompts, enabling edit mode on any model.
        """
        api_key = self.get_api_key()
        if not api_key:
            raise PermanentAPIError(
                "OpenRouter API key not set. Add api_key to [profile.openrouter] "
                "in config.toml or set OPENROUTER_API_KEY env var"
            )
        resolved_model = self.resolve_model(params.model)
        # Collect all input images (edit source + references)
        input_images: list[Path] = []
        if params.edit_source is not None:
            input_images.append(params.edit_source)
        if params.reference_images:
            input_images.extend(params.reference_images)
        return self._call_with_retry(prompt, resolved_model, api_key, images=input_images)

    def _call_with_retry(
        self,
        prompt: str,
        model: str,
        api_key: str,
        images: list[Path] | None = None,
    ) -> bytes:
        """Execute with retry on transient failures."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._call_once(prompt, model, api_key, images=images)
            except TransientAPIError as e:
                last_err = e
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_BACKOFF_S * (2**attempt)
                log.warning(
                    "transient error, retrying",
                    provider="openrouter",
                    attempt=attempt + 1,
                    delay_s=delay,
                    error=str(e)[:_ERROR_TRUNCATE_LEN],
                )
                if attempt >= 1:
                    sentry_sdk.capture_message(
                        f"{self.name} provider retry attempt {attempt + 1}",
                        level="warning",
                    )
                time.sleep(delay)
            except PermanentAPIError:
                raise
        raise TransientAPIError(f"Gave up after {_MAX_RETRIES} retries. Last error: {last_err}")

    def _call_once(
        self,
        prompt: str,
        model: str,
        api_key: str,
        images: list[Path] | None = None,
    ) -> bytes:
        """Single chat completion call with image output.

        When images are provided, builds a multimodal content array with
        image_url parts followed by the text prompt.
        """
        # Build message content
        if images:
            content: list[dict[str, Any]] = []
            for img_path in images:
                img_bytes = img_path.read_bytes()
                mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
                b64 = base64.b64encode(img_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            content.append({"type": "text", "text": prompt})
            message_content: str | list[dict[str, Any]] = content
        else:
            message_content = prompt

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": message_content}],
        }
        # Multimodal models (Gemini, GPT) need explicit modalities param
        # Image-only models (FLUX, Recraft, etc.) don't support it
        if model not in self._IMAGE_ONLY_MODELS:
            payload["modalities"] = ["text", "image"]

        req = urllib.request.Request(
            f"{_API_BASE}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/vladistan/image-creator-tool",
                "X-Title": "image-creator-tool",
            },
        )
        with sentry_sdk.start_span(op="http.client", description=f"{self.name}.generate") as span:
            span.set_data("model", model)
            try:
                with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                    data = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code in _RETRYABLE_STATUS:
                    raise TransientAPIError(f"HTTP {e.code}: {body[:_BODY_TRUNCATE_LEN]}") from e
                raise PermanentAPIError(f"HTTP {e.code}: {body[:_BODY_TRUNCATE_LEN]}") from e
            except urllib.error.URLError as e:
                raise TransientAPIError(f"Network error: {e}") from e

        return self._extract_image(data)

    def _extract_image(self, data: dict[str, Any]) -> bytes:
        """Extract image from message.images field.

        OpenRouter returns images as a separate field on the message object,
        not inline in content. Format: images[0].image_url.url (data URI).
        """
        choices = data.get("choices", [])
        if not choices:
            raise PermanentAPIError(f"No choices in response: {data}")

        message = choices[0].get("message", {})
        images = message.get("images", [])

        if not images:
            # Fallback: check content for inline images
            content = message.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            b64 = url.split(",", 1)[1]
                            return base64.b64decode(b64)
            raise PermanentAPIError(
                f"No images in response. Content: {str(content)[:_CONTENT_TRUNCATE_LEN]}"
            )

        # Extract from images array
        img = images[0]
        if isinstance(img, dict):
            url = img.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                b64 = url.split(",", 1)[1]
                return base64.b64decode(b64)
            if url.startswith("http"):
                try:
                    with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
                        return bytes(resp.read())
                except urllib.error.URLError as e:
                    raise TransientAPIError(f"Failed to download image: {e}") from e

        raise PermanentAPIError(f"Could not extract image from: {str(img)[:_CONTENT_TRUNCATE_LEN]}")
