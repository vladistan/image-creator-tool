"""Google Vertex AI image generation provider using Application Default Credentials.

Uses ADC (gcloud auth application-default login) instead of API keys.
Same generateContent API format as the direct Gemini provider, but routed
through Vertex AI endpoints with OAuth2 bearer token auth.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

import sentry_sdk
import structlog
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest

from image_creator_tool.errors import PermanentAPIError, TransientAPIError
from image_creator_tool.providers.base import GenerationParams, Provider

log = structlog.get_logger()

_GEMINI_URL = (
    "https://aiplatform.googleapis.com/v1/projects/{project}"
    "/locations/global/publishers/google/models/{model}:generateContent"
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_BASE_BACKOFF_S = 2.0


class VertexProvider(Provider):
    """Google Vertex AI provider using Application Default Credentials.

    Requires:
    - gcloud auth application-default login (or service account)
    - IMAGE_CREATOR_GCP_PROJECT env var or config setting

    The Gemini generateContent endpoint is global, so no region is needed.
    """

    name: ClassVar[str] = "vertex"
    default_model: ClassVar[str] = "gemini-2.5-flash-image"
    MODELS: ClassVar[dict[str, str]] = {
        "flash": "gemini-2.5-flash-image",
        "pro": "gemini-3-pro-image-preview",
        "flash-3.1": "gemini-3.1-flash-image-preview",
    }

    def __init__(self, project: str = "") -> None:
        self._project = project
        self._credentials: Any = None

    def get_api_key(self) -> str | None:
        """Not used — Vertex AI uses ADC bearer tokens instead."""
        return None

    def _get_access_token(self) -> str:
        """Get a valid OAuth2 access token via Application Default Credentials."""
        if self._credentials is None:
            self._credentials, _ = google_auth_default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self._credentials.valid:
            self._credentials.refresh(GoogleAuthRequest())
        token: str = self._credentials.token
        return token

    def generate(self, prompt: str, *, params: GenerationParams) -> bytes:
        """Generate an image via Vertex AI with ADC authentication."""
        if not self._project:
            raise PermanentAPIError(
                "GCP project not configured. Set IMAGE_CREATOR_GCP_PROJECT env var "
                "or gcp_project in config.toml"
            )
        resolved_model = self.resolve_model(params.model)
        return self._call_with_retry(
            prompt, resolved_model, params.edit_source, params.reference_images or None,
            seed=params.seed,
        )

    def _call_with_retry(
        self,
        prompt: str,
        model: str,
        edit_source: Path | None = None,
        reference_images: list[Path] | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Execute API call with exponential backoff retry."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._call_once(
                    prompt, model, edit_source, reference_images,
                    seed=seed,
                )
            except TransientAPIError as e:
                last_err = e
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_BACKOFF_S * (2**attempt)
                log.warning(
                    "transient API error, retrying",
                    provider="vertex",
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
        edit_source: Path | None,
        reference_images: list[Path] | None,
        seed: int | None = None,
    ) -> bytes:
        """Single API call — always routes to the Gemini generateContent endpoint."""
        return self._call_gemini(prompt, model, edit_source, reference_images, seed=seed)

    def _call_gemini(
        self,
        prompt: str,
        model: str,
        edit_source: Path | None,
        reference_images: list[Path] | None,
        seed: int | None = None,
    ) -> bytes:
        """Call Gemini generateContent endpoint."""
        access_token = self._get_access_token()

        parts: list[dict[str, Any]] = [{"text": prompt}]
        if edit_source:
            parts.append(self._image_part(edit_source))
        parts.extend(self._image_part(ref) for ref in reference_images or [])

        gen_config: dict[str, Any] = {"responseModalities": ["TEXT", "IMAGE"]}
        if seed is not None:
            gen_config["seed"] = seed

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": gen_config,
        }

        url = _GEMINI_URL.format(project=self._project, model=model)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
                "x-goog-user-project": self._project,
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
        """Encode an image file as a base64 inline data part."""
        with path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return {"inlineData": {"mimeType": mime, "data": b64}}
