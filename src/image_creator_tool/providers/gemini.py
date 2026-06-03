"""Google Gemini image generation provider.

Implements the Provider interface using the Gemini generativelanguage API
with exponential backoff retry for transient failures.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path  # noqa: TC003
from typing import Any, ClassVar

import sentry_sdk
import structlog

from image_creator_tool.errors import PermanentAPIError, TransientAPIError
from image_creator_tool.providers.base import GenerationParams, Provider

log = structlog.get_logger()

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_BASE_BACKOFF_S = 2.0


class GeminiProvider(Provider):
    """Google Gemini image generation provider.

    Supports models: flash (3.1), pro (3), flash-2.5.
    Authenticates via GEMINI_API_KEY environment variable.
    """

    name: ClassVar[str] = "gemini"
    default_model: ClassVar[str] = "gemini-3.1-flash-image-preview"
    MODELS: ClassVar[dict[str, str]] = {
        "flash": "gemini-3.1-flash-image-preview",
        "pro": "gemini-3-pro-image-preview",
        "flash-2.5": "gemini-2.5-flash-image",
    }

    def get_api_key(self) -> str | None:
        """Resolve API key from GEMINI_API_KEY environment variable."""
        return os.environ.get("GEMINI_API_KEY") or None

    def generate(self, prompt: str, *, params: GenerationParams) -> bytes:
        """Generate an image via Gemini API with automatic retry on transient failures."""
        api_key = self.get_api_key()
        if not api_key:
            raise PermanentAPIError("GEMINI_API_KEY environment variable not set")
        resolved_model = self.resolve_model(params.model)
        return self._call_with_retry(
            prompt, resolved_model, api_key,
            params.edit_source, params.reference_images or None, seed=params.seed,
        )

    def _call_with_retry(
        self,
        prompt: str,
        model: str,
        api_key: str,
        edit_source: Path | None = None,
        reference_images: list[Path] | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Execute API call with exponential backoff retry on transient errors."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._call_once(prompt, model, api_key, edit_source, reference_images, seed)
            except TransientAPIError as e:
                last_err = e
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_BACKOFF_S * (2**attempt)
                log.warning(
                    "transient API error, retrying",
                    attempt=attempt + 1,
                    max_retries=_MAX_RETRIES,
                    delay_s=delay,
                    error=str(e)[:120],
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
        edit_source: Path | None,
        reference_images: list[Path] | None,
        seed: int | None = None,
    ) -> bytes:
        """Single API call to Gemini generateContent endpoint."""
        parts: list[dict[str, Any]] = [{"text": prompt}]
        if edit_source:
            parts.append(self._image_part(edit_source))
        parts.extend(self._image_part(ref) for ref in reference_images or [])

        gen_config: dict[str, Any] = {"responseModalities": ["TEXT", "IMAGE"]}
        if seed is not None:
            gen_config["seed"] = seed

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": gen_config,
        }
        req = urllib.request.Request(
            API_URL.format(model=model),
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
        )
        with sentry_sdk.start_span(op="http.client", description=f"{self.name}.generate") as span:
            span.set_data("model", model)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code in _RETRYABLE_STATUS:
                    raise TransientAPIError(f"HTTP {e.code}: {body[:400]}") from e
                raise PermanentAPIError(f"HTTP {e.code}: {body[:400]}") from e
            except urllib.error.URLError as e:
                raise TransientAPIError(f"Network error: {e}") from e

        candidates = data.get("candidates", [])
        if not candidates:
            feedback = data.get("promptFeedback", {})
            raise PermanentAPIError(f"No candidates. Feedback: {feedback}")

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")
        for part in candidate.get("content", {}).get("parts", []):
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"])

        text_parts = [p.get("text", "") for p in candidate.get("content", {}).get("parts", [])]
        msg = f"No image. finishReason={finish_reason}. Text: {' '.join(text_parts)[:300]}"
        if finish_reason in {"SAFETY", "RECITATION", "PROHIBITED_CONTENT"}:
            raise PermanentAPIError(msg)
        raise TransientAPIError(msg)

    @staticmethod
    def _image_part(path: Path) -> dict[str, Any]:
        """Encode an image file as a base64 inline data part for the API."""
        with path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return {"inlineData": {"mimeType": mime, "data": b64}}
