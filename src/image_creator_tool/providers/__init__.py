"""Provider registry and factory for image generation backends.

Supports multiple AI image generation providers through a common interface.
Registered: azure-openai, bedrock, deepinfra, gemini, huggingface, litellm,
openai, openrouter, vertex.
"""

from __future__ import annotations

from image_creator_tool.providers.azure_openai import AzureOpenAIProvider
from image_creator_tool.providers.base import Provider
from image_creator_tool.providers.bedrock import BedrockProvider
from image_creator_tool.providers.deepinfra import DeepInfraProvider
from image_creator_tool.providers.gemini import GeminiProvider
from image_creator_tool.providers.huggingface import HuggingFaceProvider
from image_creator_tool.providers.litellm import LiteLLMProvider
from image_creator_tool.providers.openai import OpenAIProvider
from image_creator_tool.providers.openrouter import OpenRouterProvider
from image_creator_tool.providers.vertex import VertexProvider

REGISTRY: dict[str, type[Provider]] = {
    "azure-openai": AzureOpenAIProvider,
    "bedrock": BedrockProvider,
    "deepinfra": DeepInfraProvider,
    "gemini": GeminiProvider,
    "huggingface": HuggingFaceProvider,
    "litellm": LiteLLMProvider,
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "vertex": VertexProvider,
}


def get_provider(name: str, **kwargs: str) -> Provider:
    """Instantiate a provider by registered name.

    Keyword args are passed to the provider constructor (e.g. project, region for vertex).
    Raises KeyError if the provider name is not in the registry.
    """
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(f"Unknown provider '{name}'. Available: {available}")
    return REGISTRY[name](**kwargs)


def list_providers() -> list[str]:
    """Return sorted list of registered provider names."""
    return sorted(REGISTRY.keys())


__all__ = ["Provider", "get_provider", "list_providers"]
