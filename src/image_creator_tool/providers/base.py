"""Abstract base class for image generation providers.

Defines the contract that all provider implementations must fulfill,
enabling the generation layer to remain provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 — used at runtime by GenerationParams dataclass
from typing import ClassVar


@dataclass(frozen=True)
class GenerationParams:
    """Bundled parameters for image generation calls."""

    model: str
    reference_images: list[Path] = field(default_factory=list)
    edit_source: Path | None = None
    size: str | None = None
    quality: str | None = None
    aspect_ratio: str | None = None
    seed: int | None = None
    edit_op: str | None = None
    search_prompt: str | None = None
    mask: Path | None = None


class Provider(ABC):
    """Abstract image generation provider.

    Subclasses must define class variables (name, default_model, MODELS)
    and implement the generate() and get_api_key() methods.
    """

    name: ClassVar[str]
    default_model: ClassVar[str]
    MODELS: ClassVar[dict[str, str]]

    @abstractmethod
    def generate(self, prompt: str, *, params: GenerationParams) -> bytes:
        """Generate an image from a text prompt.

        Args:
            prompt: The text prompt describing the desired image.
            params: Bundled generation parameters (model, size, seed, etc.).

        Returns:
            Raw image bytes (PNG format).

        Raises:
            TransientAPIError: Retryable failure (rate limit, server error).
            PermanentAPIError: Non-retryable failure (auth, safety block).
        """

    @abstractmethod
    def get_api_key(self) -> str | None:
        """Retrieve the API key for this provider.

        Returns None if no key is available (not configured).
        """

    def resolve_model(self, model: str) -> str:
        """Resolve a model alias to its full API model ID.

        Passes through unrecognized names unchanged (assumed to be full IDs).
        """
        return self.MODELS.get(model, model)
