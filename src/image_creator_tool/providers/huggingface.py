"""HuggingFace image generation provider.

Routes text-to-image generation through the huggingface_hub InferenceClient with
`provider="auto"`, so each model is dispatched to whichever inference backend
serves it (hf-inference, fal-ai, replicate, together, ...), billed via the
HuggingFace token. Authenticates via api_key in config profile or the HF_TOKEN
env var. Unrecognized model names pass through unchanged, so any HuggingFace
text-to-image repo or custom fine-tune slug can be used directly. Retries
transient failures (rate limits, server errors) with exponential backoff,
mirroring the other providers.

huggingface_hub is an optional dependency, imported lazily so the module loads
without it. Install with: uv pip install 'image-creator-tool[huggingface]'.
"""

from __future__ import annotations

import io
import os
import time
from typing import Any, ClassVar

import sentry_sdk
import structlog

from image_creator_tool.errors import PermanentAPIError, TransientAPIError
from image_creator_tool.providers.base import GenerationParams, Provider

log = structlog.get_logger()

_DEFAULT_HF_PROVIDER = "auto"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_BACKOFF_S = 2.0
_REQUEST_TIMEOUT_S = 300
_ERROR_TRUNCATE_LEN = 120
_BODY_TRUNCATE_LEN = 400
_HTTP_CLIENT_ERROR_START = 400
_HTTP_SERVER_ERROR_START = 500


def _load_inference_client_cls() -> Any:
    """Import the huggingface_hub InferenceClient lazily."""
    try:
        from huggingface_hub import InferenceClient  # noqa: PLC0415
    except ImportError as e:
        raise PermanentAPIError(
            "huggingface_hub not installed. Install with: "
            "uv pip install 'image-creator-tool[huggingface]'"
        ) from e
    return InferenceClient


class HuggingFaceProvider(Provider):
    """HuggingFace provider using the huggingface_hub InferenceClient.

    Dispatches via provider="auto", so any model with a live inference-provider
    mapping is reachable, not just the hf-inference backend. API key from: config
    profile api_key > HF_TOKEN env var. Unrecognized model names pass through
    unchanged, so any HuggingFace text-to-image repo or custom fine-tune slug
    (e.g. "my-org/my-finetune") can be used directly.
    """

    name: ClassVar[str] = "huggingface"
    default_model: ClassVar[str] = "flux-schnell"
    MODELS: ClassVar[dict[str, str]] = {
        # FLUX
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        # Stable Diffusion XL
        "sdxl-turbo": "stabilityai/sdxl-turbo",
        "stable-diffusion-xl": "stabilityai/stable-diffusion-xl-base-1.0",
        # Stable Diffusion 3.5
        "sd-3.5-large": "stabilityai/stable-diffusion-3.5-large",
    }

    def __init__(
        self,
        api_key: str = "",
        hf_provider: str = _DEFAULT_HF_PROVIDER,
        **_kwargs: str,
    ) -> None:
        self._api_key = api_key
        self._hf_provider = hf_provider or _DEFAULT_HF_PROVIDER

    def get_api_key(self) -> str | None:
        """Get API key from config, then fall back to HF_TOKEN env var."""
        return self._api_key or os.environ.get("HF_TOKEN")

    def generate(self, prompt: str, *, params: GenerationParams) -> bytes:
        """Generate an image via the huggingface_hub InferenceClient.

        Text-to-image only; edit sources and reference images are not supported
        and are ignored with a warning.
        """
        api_key = self.get_api_key()
        if not api_key:
            raise PermanentAPIError(
                "HuggingFace API key not set. Add api_key to [profile.huggingface] "
                "in config.toml or set HF_TOKEN env var"
            )
        resolved_model = self.resolve_model(params.model)

        if params.edit_source is not None:
            log.warning("edit mode not supported by huggingface provider, ignored")
        if params.reference_images:
            log.warning("reference images not supported by huggingface provider, ignored")

        return self._call_with_retry(
            prompt, resolved_model, api_key, size=params.size, seed=params.seed,
        )

    def _call_with_retry(
        self,
        prompt: str,
        model: str,
        api_key: str,
        size: str | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Execute with exponential backoff retry on transient failures."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._call_once(prompt, model, api_key, size=size, seed=seed)
            except TransientAPIError as e:
                last_err = e
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_BACKOFF_S * (2**attempt)
                log.warning(
                    "transient error, retrying",
                    provider="huggingface",
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
        size: str | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Single inference call via InferenceClient; returns PNG-encoded bytes."""
        client_cls = _load_inference_client_cls()
        kwargs: dict[str, Any] = {"model": model}
        if size and "x" in size.lower():
            try:
                w, h = (int(v) for v in size.lower().split("x", 1))
                kwargs["width"] = w
                kwargs["height"] = h
            except ValueError:
                pass
        if seed is not None:
            kwargs["seed"] = seed

        with sentry_sdk.start_span(op="http.client", description=f"{self.name}.generate") as span:
            span.set_data("model", model)
            span.set_data("hf_provider", self._hf_provider)
            try:
                client = client_cls(
                    provider=self._hf_provider,
                    api_key=api_key,
                    timeout=_REQUEST_TIMEOUT_S,
                )
                image = client.text_to_image(prompt, **kwargs)
            except Exception as e:  # adapter boundary: map SDK errors to domain errors
                raise self._map_error(e) from e

        return self._encode_png(image)

    @staticmethod
    def _map_error(exc: Exception) -> PermanentAPIError | TransientAPIError:
        """Classify a huggingface_hub exception as transient or permanent.

        4xx client errors (except rate-limit 429) are permanent; retryable
        statuses, server errors, and status-less connection/timeout errors are
        transient.
        """
        status = _status_of(exc)
        detail = str(exc)[:_BODY_TRUNCATE_LEN]
        is_client_error = (
            isinstance(status, int)
            and _HTTP_CLIENT_ERROR_START <= status < _HTTP_SERVER_ERROR_START
        )
        if is_client_error and status not in _RETRYABLE_STATUS:
            return PermanentAPIError(f"HuggingFace request failed (HTTP {status}): {detail}")
        return TransientAPIError(f"HuggingFace request failed: {detail}")

    @staticmethod
    def _encode_png(image: Any) -> bytes:
        """Encode a PIL image returned by InferenceClient as PNG bytes."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


def _status_of(exc: Exception) -> int | None:
    """Extract an HTTP status code from a huggingface_hub error, if present."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    fallback = getattr(exc, "status_code", None)
    return fallback if isinstance(fallback, int) else None
