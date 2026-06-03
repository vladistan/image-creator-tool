"""OpenAI direct image generation provider.

Uses the /v1/images/generations endpoint with OPENAI_API_KEY.
Supports gpt-image-2, gpt-image-1.5, gpt-image-1, gpt-image-1-mini.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

import sentry_sdk
import structlog

from image_creator_tool.errors import PermanentAPIError, TransientAPIError
from image_creator_tool.providers.base import GenerationParams, Provider

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger()

_API_URL = "https://api.openai.com/v1/images/generations"
_API_EDITS_URL = "https://api.openai.com/v1/images/edits"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_BACKOFF_S = 2.0


class OpenAIProvider(Provider):
    """OpenAI direct provider using /v1/images/generations.

    API key from: config profile api_key > OPENAI_API_KEY env var.
    """

    name: ClassVar[str] = "openai"
    default_model: ClassVar[str] = "gpt-image-2"
    MODELS: ClassVar[dict[str, str]] = {
        "gpt-image-2": "gpt-image-2",
        "gpt-image-1.5": "gpt-image-1.5",
        "gpt-image-1": "gpt-image-1",
        "gpt-image-mini": "gpt-image-1-mini",
        "chatgpt": "chatgpt-image-latest",
    }

    def __init__(self, api_key: str = "", **_kwargs: str) -> None:
        self._api_key = api_key

    def get_api_key(self) -> str | None:
        """Get API key from config, then fall back to env var."""
        return self._api_key or os.environ.get("OPENAI_API_KEY")

    def generate(self, prompt: str, *, params: GenerationParams) -> bytes:
        """Generate or edit an image via OpenAI images API."""
        api_key = self.get_api_key()
        if not api_key:
            raise PermanentAPIError(
                "OpenAI API key not set. Add api_key to [profile.openai] "
                "in config.toml or set OPENAI_API_KEY env var"
            )
        resolved_model = self.resolve_model(params.model)
        if params.edit_source is not None:
            return self._edit_with_retry(
                prompt, resolved_model, api_key, params.edit_source, size=params.size,
            )
        return self._call_with_retry(
            prompt, resolved_model, api_key, size=params.size, quality=params.quality,
        )

    def _call_with_retry(
        self,
        prompt: str,
        model: str,
        api_key: str,
        size: str | None = None,
        quality: str | None = None,
    ) -> bytes:
        """Execute with retry on transient failures."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._call_once(prompt, model, api_key, size=size, quality=quality)
            except TransientAPIError as e:
                last_err = e
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_BACKOFF_S * (2**attempt)
                log.warning(
                    "transient error, retrying",
                    provider="openai",
                    attempt=attempt + 1,
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
        size: str | None = None,
        quality: str | None = None,
    ) -> bytes:
        """Single call to OpenAI images/generations endpoint."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size or "1024x1024",
        }
        if quality:
            payload["quality"] = quality

        req = urllib.request.Request(
            _API_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with sentry_sdk.start_span(op="http.client", description=f"{self.name}.generate") as span:
            span.set_data("model", model)
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code in _RETRYABLE_STATUS:
                    raise TransientAPIError(f"HTTP {e.code}: {body[:400]}") from e
                raise PermanentAPIError(f"HTTP {e.code}: {body[:400]}") from e
            except urllib.error.URLError as e:
                raise TransientAPIError(f"Network error: {e}") from e

        images = data.get("data", [])
        if not images:
            raise PermanentAPIError(f"No images in response: {data}")

        img = images[0]
        if "b64_json" in img:
            return base64.b64decode(img["b64_json"])
        if "url" in img:
            try:
                with urllib.request.urlopen(img["url"], timeout=60) as resp:
                    return bytes(resp.read())
            except urllib.error.URLError as e:
                raise TransientAPIError(f"Failed to download image: {e}") from e

        raise PermanentAPIError(f"No image data in response: {list(img.keys())}")

    def _edit_with_retry(
        self,
        prompt: str,
        model: str,
        api_key: str,
        edit_source: Path,
        size: str | None = None,
    ) -> bytes:
        """Execute edit request with retry on transient failures."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._edit_once(prompt, model, api_key, edit_source, size=size)
            except TransientAPIError as e:
                last_err = e
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_BACKOFF_S * (2**attempt)
                log.warning(
                    "transient error retrying edit",
                    provider="openai",
                    attempt=attempt + 1,
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
        raise TransientAPIError(
            f"Edit gave up after {_MAX_RETRIES} retries. Last error: {last_err}"
        )

    def _edit_once(
        self,
        prompt: str,
        model: str,
        api_key: str,
        edit_source: Path,
        size: str | None = None,
    ) -> bytes:
        """Single call to OpenAI images/edits endpoint using multipart form data."""
        boundary = f"----{uuid.uuid4().hex}"
        body = io.BytesIO()

        def _field(name: str, value: str) -> None:
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.write(f"{value}\r\n".encode())

        def _file_field(name: str, filename: str, data: bytes, content_type: str) -> None:
            body.write(f"--{boundary}\r\n".encode())
            body.write(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            )
            body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
            body.write(data)
            body.write(b"\r\n")

        img_bytes = edit_source.read_bytes()
        mime = "image/png" if edit_source.suffix.lower() == ".png" else "image/jpeg"
        _file_field("image[]", edit_source.name, img_bytes, mime)
        _field("model", model)
        _field("prompt", prompt)
        _field("n", "1")
        if size:
            _field("size", size)
        body.write(f"--{boundary}--\r\n".encode())

        req = urllib.request.Request(
            _API_EDITS_URL,
            data=body.getvalue(),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with sentry_sdk.start_span(op="http.client", description=f"{self.name}.edit") as span:
            span.set_data("model", model)
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data: dict[str, Any] = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                body_text = e.read().decode(errors="replace")
                if e.code in _RETRYABLE_STATUS:
                    raise TransientAPIError(f"HTTP {e.code}: {body_text[:400]}") from e
                raise PermanentAPIError(f"HTTP {e.code}: {body_text[:400]}") from e
            except urllib.error.URLError as e:
                raise TransientAPIError(f"Network error: {e}") from e

        images = data.get("data", [])
        if not images:
            raise PermanentAPIError(f"No images in edit response: {data}")
        img = images[0]
        if "b64_json" in img:
            return base64.b64decode(img["b64_json"])
        if "url" in img:
            try:
                with urllib.request.urlopen(img["url"], timeout=60) as resp:
                    return bytes(resp.read())
            except urllib.error.URLError as e:
                raise TransientAPIError(f"Failed to download edited image: {e}") from e
        raise PermanentAPIError(f"No image data in edit response: {list(img.keys())}")
