"""Azure OpenAI image generation provider.

Uses the openai SDK's AzureOpenAI client against a DALL-E deployment. Requires
endpoint, API key, and deployment name from config (api_key in the provider
profile) or the standard Azure env vars: AZURE_OPENAI_ENDPOINT,
AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT. When a source image is supplied,
the Azure DALL-E edit endpoint is used instead of generation.

openai is an optional dependency, imported lazily so the module loads without
it. Install with: uv pip install 'image-creator-tool[azure]'.
"""

from __future__ import annotations

import base64
import os
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, ClassVar

import sentry_sdk
import structlog

from image_creator_tool.errors import PermanentAPIError, TransientAPIError
from image_creator_tool.providers.base import GenerationParams, Provider

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger()

_API_VERSION = "2024-10-21"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_BACKOFF_S = 2.0
_DOWNLOAD_TIMEOUT_S = 60
_ERROR_TRUNCATE_LEN = 120
_BODY_TRUNCATE_LEN = 400
_HTTP_CLIENT_ERROR_START = 400
_HTTP_SERVER_ERROR_START = 500

def _load_azure_client_cls() -> Any:
    """Import the openai SDK's AzureOpenAI client lazily."""
    try:
        from openai import AzureOpenAI  # noqa: PLC0415
    except ImportError as e:
        raise PermanentAPIError(
            "openai SDK not installed. Install with: uv pip install 'image-creator-tool[azure]'"
        ) from e
    return AzureOpenAI


class AzureOpenAIProvider(Provider):
    """Azure OpenAI provider using the AzureOpenAI client for DALL-E deployments.

    Config resolution (config profile value > env var):
    - endpoint:   AZURE_OPENAI_ENDPOINT
    - api_key:    AZURE_OPENAI_API_KEY
    - deployment: AZURE_OPENAI_DEPLOYMENT

    The deployment name is the effective model, so MODELS is empty and
    resolve_model() passes names through unchanged.
    """

    name: ClassVar[str] = "azure-openai"
    default_model: ClassVar[str] = "dall-e-3"
    MODELS: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "",
        deployment: str = "",
        **_kwargs: str,
    ) -> None:
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self._endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self._deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        if not (self._api_key and self._endpoint and self._deployment):
            raise PermanentAPIError(
                "Azure OpenAI not configured. Set AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT (or the equivalent "
                "[profile.azure-openai] config keys)"
            )

    def get_api_key(self) -> str | None:
        """Return the Azure API key (config override or AZURE_OPENAI_API_KEY)."""
        return self._api_key or None

    def _client(self) -> Any:
        """Construct an AzureOpenAI client from resolved config."""
        azure_openai_cls = _load_azure_client_cls()
        return azure_openai_cls(
            api_key=self._api_key,
            azure_endpoint=self._endpoint,
            api_version=_API_VERSION,
        )

    def generate(self, prompt: str, *, params: GenerationParams) -> bytes:
        """Generate or edit an image via the Azure DALL-E deployment with retry."""
        if params.reference_images:
            log.warning("reference images not supported by azure-openai provider, ignored")
        return self._call_with_retry(prompt, size=params.size, edit_source=params.edit_source)

    def _call_with_retry(
        self,
        prompt: str,
        size: str | None = None,
        edit_source: Path | None = None,
    ) -> bytes:
        """Execute with exponential backoff retry on transient failures."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._call_once(prompt, size=size, edit_source=edit_source)
            except TransientAPIError as e:
                last_err = e
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_BACKOFF_S * (2**attempt)
                log.warning(
                    "transient error, retrying",
                    provider="azure-openai",
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
        size: str | None = None,
        edit_source: Path | None = None,
    ) -> bytes:
        """Single generate/edit call, mapping SDK errors to domain errors."""
        client = self._client()
        with sentry_sdk.start_span(op="openai.image", description=f"{self.name}.generate") as span:
            span.set_data("deployment", self._deployment)
            try:
                if edit_source is not None:
                    with edit_source.open("rb") as image_file:
                        response = client.images.edit(
                            model=self._deployment,
                            image=image_file,
                            prompt=prompt,
                            n=1,
                            size=size or "1024x1024",
                        )
                else:
                    response = client.images.generate(
                        model=self._deployment,
                        prompt=prompt,
                        n=1,
                        size=size or "1024x1024",
                    )
            except Exception as e:  # adapter boundary: map SDK errors to domain errors
                raise self._map_error(e) from e

        return self._extract_image(response)

    @staticmethod
    def _map_error(exc: Exception) -> PermanentAPIError | TransientAPIError:
        """Classify an openai SDK exception as transient or permanent by status code."""
        status = getattr(exc, "status_code", None)
        detail = str(exc)[:_BODY_TRUNCATE_LEN]
        is_client_error = (
            isinstance(status, int)
            and _HTTP_CLIENT_ERROR_START <= status < _HTTP_SERVER_ERROR_START
        )
        if is_client_error and status not in _RETRYABLE_STATUS:
            return PermanentAPIError(f"Azure OpenAI request failed (HTTP {status}): {detail}")
        return TransientAPIError(f"Azure OpenAI request failed: {detail}")

    def _extract_image(self, response: Any) -> bytes:
        """Extract raw image bytes from an openai ImagesResponse (b64 or URL)."""
        data = _get(response, "data")
        if not data:
            raise PermanentAPIError("No image data in Azure OpenAI response")
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

        raise PermanentAPIError("Azure OpenAI response item has neither b64_json nor url")


def _get(obj: Any, key: str) -> Any:
    """Read `key` from an openai response object via attribute or mapping access."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
