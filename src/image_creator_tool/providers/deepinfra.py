"""DeepInfra image generation provider.

Uses the /v1/inference/ endpoint for image generation models.
Authenticates via api_key in config profile or DEEPINFRA_API_KEY env var.
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

_API_BASE = "https://api.deepinfra.com/v1/inference"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_BACKOFF_S = 2.0
_REQUEST_TIMEOUT_S = 300
_DOWNLOAD_TIMEOUT_S = 60
_ERROR_TRUNCATE_LEN = 120
_BODY_TRUNCATE_LEN = 400
_SEEDREAM_DEFAULT_SIZE = 1920


class DeepInfraProvider(Provider):
    """DeepInfra provider using the inference API for image generation.

    API key from: config profile api_key > DEEPINFRA_API_KEY env var.
    """

    name: ClassVar[str] = "deepinfra"
    default_model: ClassVar[str] = "black-forest-labs/FLUX-2-dev"
    MODELS: ClassVar[dict[str, str]] = {
        # FLUX — generation
        "flux-2-dev": "black-forest-labs/FLUX-2-dev",
        "flux-2-pro": "black-forest-labs/FLUX-2-pro",
        "flux-2-max": "black-forest-labs/FLUX-2-max",
        "flux-2-klein": "black-forest-labs/FLUX-2-klein-9b",
        "flux-schnell": "black-forest-labs/FLUX-1-schnell",
        # FLUX — editing
        "flux-kontext": "black-forest-labs/FLUX.1-Kontext-dev",
        # Seedream — generation
        "seedream-4": "ByteDance/Seedream-4",
        "seedream-4.5": "ByteDance/Seedream-4.5",
        # Bria — generation
        "bria": "Bria/Bria-3.2",
        "bria-vector": "Bria/Bria-3.2-vector",
        # Bria — editing (sync-capable only; enhance/gen-fill/erase/erase-fg are async-only)
        "bria-remove-bg": "Bria/remove_background",
        # Qwen — generation + editing
        "qwen-image": "Qwen/Qwen-Image-Max",
        "qwen-edit": "Qwen/Qwen-Image-Edit",
        "qwen-edit-max": "Qwen/Qwen-Image-Edit-Max",
        # Wan — generation + editing
        "wan": "Wan-AI/Wan2.6-T2I",
        "wan-edit": "Wan-AI/Wan2.7-Image-Edit",
        # Google — generation only (no edit support on DeepInfra)
        "gemini-pro": "google/gemini-3-pro-image",
    }

    # Models that accept image input for editing (field name varies)
    _EDIT_MODELS: ClassVar[dict[str, str]] = {
        # model_id → image field name
        "black-forest-labs/FLUX.1-Kontext-dev": "image",
        "Qwen/Qwen-Image-Edit": "image",
        "Bria/remove_background": "image",
    }

    # Models that use image_urls format (data URI list) instead of raw base64
    _IMAGE_URL_MODELS: ClassVar[set[str]] = {
        "Wan-AI/Wan2.7-Image-Edit",
        "Wan-AI/Wan2.6-Image-Edit",
        "Qwen/Qwen-Image-Edit-Max",
    }

    def __init__(self, api_key: str = "", **_kwargs: str) -> None:
        self._api_key = api_key

    def get_api_key(self) -> str | None:
        """Get API key from config, then fall back to env var."""
        return self._api_key or os.environ.get("DEEPINFRA_API_KEY")

    def generate(self, prompt: str, *, params: GenerationParams) -> bytes:
        """Generate or edit an image via DeepInfra inference API.

        Edit-capable models (FLUX Kontext, Qwen-Edit, Wan-Edit, Bria)
        accept a source image alongside the prompt. Non-edit models ignore
        edit_source with a warning.
        """
        api_key = self.get_api_key()
        if not api_key:
            raise PermanentAPIError(
                "DeepInfra API key not set. Add api_key to [profile.deepinfra] "
                "in config.toml or set DEEPINFRA_API_KEY env var"
            )
        resolved_model = self.resolve_model(params.model)

        # Check if model supports editing
        supports_edit = (
            resolved_model in self._EDIT_MODELS
            or resolved_model in self._IMAGE_URL_MODELS
        )

        if params.edit_source is not None and not supports_edit:
            log.warning(
                "edit mode not supported by this deepinfra model",
                model=resolved_model,
            )
        if params.reference_images and not supports_edit:
            log.warning("reference images not supported by this deepinfra model, ignored")

        return self._call_with_retry(
            prompt, resolved_model, api_key,
            size=params.size, seed=params.seed,
            edit_source=params.edit_source if supports_edit else None,
        )

    def _call_with_retry(
        self,
        prompt: str,
        model: str,
        api_key: str,
        size: str | None = None,
        seed: int | None = None,
        edit_source: Path | None = None,
    ) -> bytes:
        """Execute with retry on transient failures."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._call_once(
                    prompt, model, api_key, size=size, seed=seed, edit_source=edit_source,
                )
            except TransientAPIError as e:
                last_err = e
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_BACKOFF_S * (2**attempt)
                log.warning(
                    "transient error, retrying",
                    provider="deepinfra",
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

    # Models that require explicit dimensions (min pixel count)
    _NEEDS_SIZE: ClassVar[set[str]] = {"ByteDance/Seedream-4", "ByteDance/Seedream-4.5"}

    def _call_once(
        self,
        prompt: str,
        model: str,
        api_key: str,
        size: str | None = None,
        seed: int | None = None,
        edit_source: Path | None = None,
    ) -> bytes:
        """Single inference call to DeepInfra."""
        payload: dict[str, Any] = {"prompt": prompt}

        # Attach source image for edit-capable models
        if edit_source is not None:
            img_b64 = base64.b64encode(edit_source.read_bytes()).decode()
            if model in self._IMAGE_URL_MODELS:
                mime = "image/png" if edit_source.suffix.lower() == ".png" else "image/jpeg"
                payload["image_urls"] = [f"data:{mime};base64,{img_b64}"]
            elif model in self._EDIT_MODELS:
                payload[self._EDIT_MODELS[model]] = img_b64

        # Parse size if provided (e.g. "1024x1024", "1920x1920")
        if size and "x" in size:
            try:
                w, h = (int(v) for v in size.lower().split("x", 1))
                payload["width"] = w
                payload["height"] = h
            except ValueError:
                pass
        elif model in self._NEEDS_SIZE:
            payload["width"] = _SEEDREAM_DEFAULT_SIZE
            payload["height"] = _SEEDREAM_DEFAULT_SIZE
        if seed is not None:
            payload["seed"] = seed

        req = urllib.request.Request(
            f"{_API_BASE}/{model}",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
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
        """Extract image bytes from inference response.

        Handles three formats:
        - data URI: data:image/png;base64,<data>
        - URL: https://... (downloaded)
        - raw base64 string
        """
        # Some models return "images" list, others return "image_url" string
        images = data.get("images", [])
        if not images and "image_url" in data:
            images = [data["image_url"]]
        if not images:
            raise PermanentAPIError(f"No images in response: {list(data.keys())}")

        img_data = images[0]
        if not isinstance(img_data, str):
            raise PermanentAPIError(f"Unexpected image format: {type(img_data)}")

        # data URI format
        if img_data.startswith("data:"):
            b64 = img_data.split(",", 1)[1]
            return base64.b64decode(b64)

        # URL format — download the image
        if img_data.startswith("http"):
            try:
                with urllib.request.urlopen(img_data, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
                    return bytes(resp.read())
            except urllib.error.URLError as e:
                raise TransientAPIError(f"Failed to download image: {e}") from e

        # Raw base64
        return base64.b64decode(img_data)
