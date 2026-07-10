"""LiteLLM image generation provider.

Routes image generation through litellm.image_generation(), giving access to
any image backend LiteLLM supports (OpenAI DALL-E, Stability, Bedrock, Vertex,
etc.) via LiteLLM's provider-prefix model convention (e.g. "dall-e-3",
"stability/stable-diffusion-xl-1024-v1-0"). API keys are resolved by LiteLLM
from its usual per-provider env vars; an explicit config api_key overrides them.

litellm is an optional dependency, imported lazily so the module loads without
it. Install with: uv pip install 'image-creator-tool[litellm]'.
"""

from __future__ import annotations

import base64
import time
import urllib.error
import urllib.request
from typing import Any, ClassVar

import sentry_sdk
import structlog

from image_creator_tool.errors import PermanentAPIError, TransientAPIError
from image_creator_tool.providers.base import GenerationParams, Provider

log = structlog.get_logger()

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_BACKOFF_S = 2.0
_DOWNLOAD_TIMEOUT_S = 60
_ERROR_TRUNCATE_LEN = 120
_BODY_TRUNCATE_LEN = 400
_HTTP_CLIENT_ERROR_START = 400
_HTTP_SERVER_ERROR_START = 500


def _load_litellm() -> Any:
    """Import litellm lazily, mapping a missing install to a permanent error."""
    try:
        import litellm  # noqa: PLC0415
    except ImportError as e:
        raise PermanentAPIError(
            "litellm not installed. Install with: uv pip install 'image-creator-tool[litellm]'"
        ) from e
    return litellm


class LiteLLMProvider(Provider):
    """LiteLLM provider routing through litellm.image_generation().

    Model aliases resolve to LiteLLM provider-prefixed model strings; unknown
    names pass through unchanged so any LiteLLM-supported model can be used.
    """

    name: ClassVar[str] = "litellm"
    default_model: ClassVar[str] = "dall-e-3"
    MODELS: ClassVar[dict[str, str]] = {
        "dall-e-3": "dall-e-3",
        "dall-e-2": "dall-e-2",
        "gpt-image-1": "gpt-image-1",
        "stable-diffusion-xl": "stability/stable-diffusion-xl-1024-v1-0",
        "stable-diffusion-3": "stability/sd3-large",
    }

    def __init__(self, api_key: str = "", **_kwargs: str) -> None:
        self._api_key = api_key

    def get_api_key(self) -> str | None:
        """Return the explicit config override, if any.

        LiteLLM otherwise resolves keys from its per-provider env vars, so a
        missing override here is not itself an error.
        """
        return self._api_key or None

    def generate(self, prompt: str, *, params: GenerationParams) -> bytes:
        """Generate an image via litellm.image_generation() with retry."""
        resolved_model = self.resolve_model(params.model)
        if params.edit_source is not None:
            log.warning("edit mode not supported by litellm provider, ignored")
        if params.reference_images:
            log.warning("reference images not supported by litellm provider, ignored")
        return self._call_with_retry(prompt, resolved_model, size=params.size)

    def _call_with_retry(self, prompt: str, model: str, size: str | None = None) -> bytes:
        """Execute with exponential backoff retry on transient failures."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._call_once(prompt, model, size=size)
            except TransientAPIError as e:
                last_err = e
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_BACKOFF_S * (2**attempt)
                log.warning(
                    "transient error, retrying",
                    provider="litellm",
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

    def _call_once(self, prompt: str, model: str, size: str | None = None) -> bytes:
        """Single litellm.image_generation() call, mapping SDK errors to domain errors."""
        litellm = _load_litellm()
        kwargs: dict[str, Any] = {"model": model, "prompt": prompt, "n": 1}
        if size:
            kwargs["size"] = size
        if self._api_key:
            kwargs["api_key"] = self._api_key

        with sentry_sdk.start_span(op="litellm.image", description=f"{self.name}.generate") as span:
            span.set_data("model", model)
            try:
                response = litellm.image_generation(**kwargs)
            except Exception as e:  # adapter boundary: map SDK errors to domain errors
                raise self._map_error(e) from e

        return self._extract_image(response)

    @staticmethod
    def _map_error(exc: Exception) -> PermanentAPIError | TransientAPIError:
        """Classify a LiteLLM SDK exception as transient or permanent.

        4xx client errors (except rate-limit 429) are permanent; retryable
        statuses, server errors, and status-less connection/timeout errors are
        transient.
        """
        status = getattr(exc, "status_code", None)
        detail = str(exc)[:_BODY_TRUNCATE_LEN]
        is_client_error = (
            isinstance(status, int)
            and _HTTP_CLIENT_ERROR_START <= status < _HTTP_SERVER_ERROR_START
        )
        if is_client_error and status not in _RETRYABLE_STATUS:
            return PermanentAPIError(f"litellm image_generation failed (HTTP {status}): {detail}")
        return TransientAPIError(f"litellm image_generation failed: {detail}")

    def _extract_image(self, response: Any) -> bytes:
        """Extract raw image bytes from a LiteLLM ImageResponse (b64 or URL)."""
        data = _get(response, "data")
        if not data:
            raise PermanentAPIError("No image data in litellm response")
        item = data[0]

        b64 = _get(item, "b64_json")
        if b64:
            return base64.b64decode(b64)

        url = _get(item, "url")
        if url:
            try:
                with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
                    return bytes(resp.read())
            except urllib.error.URLError as e:
                raise TransientAPIError(f"Failed to download image: {e}") from e

        raise PermanentAPIError("litellm response item has neither b64_json nor url")


def _get(obj: Any, key: str) -> Any:
    """Read `key` from a LiteLLM response object via attribute or mapping access."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
