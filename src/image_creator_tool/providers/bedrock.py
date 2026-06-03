"""Amazon Bedrock image generation provider.

Uses boto3 with AWS credentials (SSO profiles or IAM) to call Bedrock Runtime.
Supports Stability AI models (Ultra, Core, SD3.5) for both generation and editing.
"""

from __future__ import annotations

import base64
import json
import time
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar

import boto3
import sentry_sdk
import structlog
from PIL import Image

from image_creator_tool.errors import PermanentAPIError, TransientAPIError
from image_creator_tool.providers.base import GenerationParams, Provider

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger()

_MAX_RETRIES = 3
_BASE_BACKOFF_S = 2.0

# Default image-to-image strength (0 = identical to source, 1 = ignore source).
_DEFAULT_EDIT_STRENGTH = 0.65

# Pixel threshold above which we prefer compressed formats to stay under 16MB.
_LARGE_PIXEL_COUNT = 2_000_000

# Stability editing models — each operation has a dedicated model via inference profiles.
_EDIT_OP_MODELS: dict[str, str] = {
    "remove-bg": "us.stability.stable-image-remove-background-v1:0",
    "upscale-fast": "us.stability.stable-fast-upscale-v1:0",
    "upscale-conservative": "us.stability.stable-conservative-upscale-v1:0",
    "upscale-creative": "us.stability.stable-creative-upscale-v1:0",
    "erase": "us.stability.stable-image-erase-object-v1:0",
    "inpaint": "us.stability.stable-image-inpaint-v1:0",
    "outpaint": "us.stability.stable-outpaint-v1:0",
    "search-replace": "us.stability.stable-image-search-replace-v1:0",
    "style-transfer": "us.stability.stable-style-transfer-v1:0",
    "style-guide": "us.stability.stable-image-style-guide-v1:0",
}

# Edit ops that need alpha channel output (must use png or webp).
_ALPHA_REQUIRED_OPS = {"remove-bg"}

# Default pixel extension for outpaint in all directions.
_OUTPAINT_EXTEND_PX = 200


class BedrockProvider(Provider):
    """Amazon Bedrock provider using boto3 for Stability AI models.

    Supports text-to-image generation, image-to-image editing, and
    specialised editing operations (remove-bg, upscale, inpaint, etc.)
    via dedicated Stability inference profiles.

    Uses AWS credential chain (SSO profiles, env vars, IAM roles).
    Config: aws_profile, aws_region in profile section.
    """

    name: ClassVar[str] = "bedrock"
    default_model: ClassVar[str] = "stability.stable-image-ultra-v1:1"
    MODELS: ClassVar[dict[str, str]] = {
        "ultra": "stability.stable-image-ultra-v1:1",
        "core": "stability.stable-image-core-v1:1",
        "sd3.5": "stability.sd3-5-large-v1:0",
    }

    def __init__(
        self, aws_profile: str = "", aws_region: str = "us-west-2", **_kwargs: str
    ) -> None:
        session_kwargs: dict[str, str] = {"region_name": aws_region}
        if aws_profile:
            session_kwargs["profile_name"] = aws_profile
        session = boto3.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime")

    def get_api_key(self) -> str | None:
        """Not used — Bedrock uses AWS credentials."""
        return None

    def generate(self, prompt: str, *, params: GenerationParams) -> bytes:
        """Generate or edit an image via Bedrock Runtime.

        Supports:
        - text-to-image generation (default)
        - image-to-image editing (when edit_source provided, no edit_op)
        - specialised edit operations (when edit_op specified): remove-bg,
          upscale-fast, upscale-conservative, upscale-creative, erase,
          inpaint, outpaint, search-replace, style-transfer, style-guide
        """
        if params.edit_op and params.edit_op in _EDIT_OP_MODELS:
            return self._call_edit_op(
                prompt, params.edit_source, params.edit_op,
                reference_images=params.reference_images or None,
                search_prompt=params.search_prompt,
                mask=params.mask, seed=params.seed,
            )
        if params.reference_images and not params.edit_op:
            log.warning("reference images not supported by bedrock provider, ignored")
        resolved_model = self.resolve_model(params.model)
        return self._call_with_retry(
            prompt, resolved_model,
            edit_source=params.edit_source,
            aspect_ratio=params.aspect_ratio,
            seed=params.seed,
        )

    def _call_with_retry(
        self,
        prompt: str,
        model: str,
        edit_source: Path | None = None,
        aspect_ratio: str | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Execute with retry on transient failures."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._call_once(
                    prompt, model, edit_source=edit_source,
                    aspect_ratio=aspect_ratio, seed=seed,
                )
            except TransientAPIError as e:
                last_err = e
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_BACKOFF_S * (2**attempt)
                log.warning(
                    "transient error, retrying",
                    provider="bedrock",
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
        edit_source: Path | None = None,
        aspect_ratio: str | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Single Bedrock invoke-model call.

        Switches between text-to-image and image-to-image based on edit_source.
        """
        if edit_source is not None:
            img_b64 = base64.b64encode(edit_source.read_bytes()).decode()
            payload: dict[str, Any] = {
                "prompt": prompt,
                "mode": "image-to-image",
                "image": img_b64,
                "strength": _DEFAULT_EDIT_STRENGTH,
                "output_format": "png",
            }
        else:
            payload = {
                "prompt": prompt,
                "mode": "text-to-image",
                "aspect_ratio": aspect_ratio or "1:1",
                "output_format": "png",
            }
        if seed is not None:
            payload["seed"] = seed

        with sentry_sdk.start_span(op="http.client", description=f"{self.name}.generate") as span:
            span.set_data("model", model)
            try:
                response = self._client.invoke_model(
                    modelId=model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(payload),
                )
            except self._client.exceptions.ThrottlingException as e:
                raise TransientAPIError(f"Bedrock throttled: {e}") from e
            except self._client.exceptions.ServiceUnavailableException as e:
                raise TransientAPIError(f"Bedrock unavailable: {e}") from e
            except Exception as e:
                err_name = type(e).__name__
                if "Throttl" in err_name or "ServiceUnavailable" in err_name:
                    raise TransientAPIError(f"Bedrock transient: {e}") from e
                raise PermanentAPIError(f"Bedrock error: {e}") from e

            data = json.loads(response["body"].read())

        if "images" in data:
            return base64.b64decode(data["images"][0])
        if "image" in data:
            return base64.b64decode(data["image"])

        raise PermanentAPIError(f"No image in Bedrock response: {list(data.keys())}")

    # ------------------------------------------------------------------
    # Specialised edit-op helpers
    # ------------------------------------------------------------------

    def _get_output_format(self, edit_op: str, raw_bytes: bytes) -> str:
        """Choose output format to stay under 16MB response limit."""
        img = Image.open(BytesIO(raw_bytes))
        pixel_count = img.size[0] * img.size[1]
        img.close()
        if edit_op in _ALPHA_REQUIRED_OPS:
            return "webp" if pixel_count > _LARGE_PIXEL_COUNT else "png"
        return "jpeg" if pixel_count > _LARGE_PIXEL_COUNT else "png"

    def _payload_simple_image(
        self, img_b64: str, out_fmt: str, **_kw: Any,
    ) -> dict[str, Any]:
        return {"output_format": out_fmt, "image": img_b64}

    def _payload_upscale(
        self, img_b64: str, out_fmt: str, *, prompt: str = "", **_kw: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"output_format": "jpeg", "image": img_b64}
        if prompt:
            payload["prompt"] = prompt
        return payload

    def _payload_erase(
        self, img_b64: str, out_fmt: str, *, mask: Path | None = None, **_kw: Any,
    ) -> dict[str, Any]:
        if not mask:
            raise PermanentAPIError("erase requires --mask <mask.png> (B&W image)")
        return {
            "output_format": out_fmt, "image": img_b64,
            "mask": base64.b64encode(mask.read_bytes()).decode(),
        }

    def _payload_inpaint(
        self, img_b64: str, out_fmt: str, *, prompt: str = "", mask: Path | None = None, **_kw: Any,
    ) -> dict[str, Any]:
        if not mask:
            raise PermanentAPIError("inpaint requires --mask <mask.png> (B&W image)")
        return {
            "output_format": out_fmt, "image": img_b64, "prompt": prompt,
            "mask": base64.b64encode(mask.read_bytes()).decode(),
        }

    def _payload_outpaint(
        self, img_b64: str, out_fmt: str, *, prompt: str = "", **_kw: Any,
    ) -> dict[str, Any]:
        return {
            "output_format": out_fmt, "image": img_b64, "prompt": prompt,
            "left": _OUTPAINT_EXTEND_PX, "right": _OUTPAINT_EXTEND_PX,
            "up": _OUTPAINT_EXTEND_PX, "down": _OUTPAINT_EXTEND_PX,
        }

    def _payload_search_replace(
        self, img_b64: str, out_fmt: str, *, prompt: str = "", search_prompt: str | None = None,
        **_kw: Any,
    ) -> dict[str, Any]:
        if not search_prompt:
            raise PermanentAPIError("search-replace requires --search <what to find>")
        return {
            "output_format": out_fmt, "image": img_b64,
            "prompt": prompt, "search_prompt": search_prompt,
        }

    def _payload_style_transfer(
        self, img_b64: str, out_fmt: str, *, prompt: str = "",
        reference_images: list[Path] | None = None, **_kw: Any,
    ) -> dict[str, Any]:
        if not reference_images:
            raise PermanentAPIError("style-transfer requires --style-ref <style image>")
        return {
            "output_format": out_fmt,
            "init_image": img_b64,
            "style_image": base64.b64encode(reference_images[0].read_bytes()).decode(),
            "prompt": prompt or "",
        }

    def _payload_style_guide(
        self, img_b64: str, out_fmt: str, *, prompt: str = "", **_kw: Any,
    ) -> dict[str, Any]:
        return {"output_format": out_fmt, "image": img_b64, "prompt": prompt}

    _EDIT_OP_BUILDERS: ClassVar[dict[str, Any]] = {
        "remove-bg": _payload_simple_image,
        "upscale-fast": _payload_simple_image,
        "upscale-conservative": _payload_upscale,
        "upscale-creative": _payload_upscale,
        "erase": _payload_erase,
        "inpaint": _payload_inpaint,
        "outpaint": _payload_outpaint,
        "search-replace": _payload_search_replace,
        "style-transfer": _payload_style_transfer,
        "style-guide": _payload_style_guide,
    }

    def _build_edit_op_payload(
        self,
        edit_op: str,
        img_b64: str,
        out_fmt: str,
        prompt: str,
        reference_images: list[Path] | None,
        search_prompt: str | None,
        mask: Path | None,
    ) -> dict[str, Any]:
        """Build the Stability API payload for a specialised edit operation."""
        builder = self._EDIT_OP_BUILDERS.get(edit_op)
        if not builder:
            raise PermanentAPIError(f"Unknown edit-op: {edit_op}")
        result: dict[str, Any] = builder(
            self, img_b64, out_fmt,
            prompt=prompt, reference_images=reference_images,
            search_prompt=search_prompt, mask=mask,
        )
        return result

    def _invoke_edit_model(self, model_id: str, payload: dict[str, Any], edit_op: str) -> bytes:
        """Invoke a Stability edit model and extract the image bytes."""
        try:
            response = self._client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
        except self._client.exceptions.ThrottlingException as e:
            raise TransientAPIError(f"Bedrock throttled: {e}") from e
        except self._client.exceptions.ServiceUnavailableException as e:
            raise TransientAPIError(f"Bedrock unavailable: {e}") from e
        except Exception as e:
            err_name = type(e).__name__
            if "Throttl" in err_name or "ServiceUnavailable" in err_name:
                raise TransientAPIError(f"Bedrock transient: {e}") from e
            raise PermanentAPIError(f"Bedrock edit-op '{edit_op}' error: {e}") from e

        data = json.loads(response["body"].read())
        if "images" in data:
            return base64.b64decode(data["images"][0])
        if "image" in data:
            return base64.b64decode(data["image"])
        raise PermanentAPIError(f"No image in edit-op response: {list(data.keys())}")

    def _call_edit_op(
        self,
        prompt: str,
        edit_source: Path | None,
        edit_op: str,
        *,
        reference_images: list[Path] | None = None,
        search_prompt: str | None = None,
        mask: Path | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Execute a specialised Stability editing operation."""
        if edit_source is None:
            raise PermanentAPIError(f"edit-op '{edit_op}' requires --edit <source image>")

        raw_bytes = edit_source.read_bytes()
        img_b64 = base64.b64encode(raw_bytes).decode()
        model_id = _EDIT_OP_MODELS[edit_op]
        out_fmt = self._get_output_format(edit_op, raw_bytes)

        payload = self._build_edit_op_payload(
            edit_op, img_b64, out_fmt, prompt, reference_images, search_prompt, mask
        )
        if seed is not None:
            payload["seed"] = seed

        return self._invoke_edit_model(model_id, payload, edit_op)
