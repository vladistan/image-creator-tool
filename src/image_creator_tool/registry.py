"""Global model registry with auto-provider resolution.

Builds an index of all model aliases across all providers, enabling
auto-detection of which provider serves a given model name.
"""

from __future__ import annotations

from image_creator_tool.providers import REGISTRY

_DEFAULT_PREFERENCE = ["vertex", "deepinfra", "openrouter", "openai", "bedrock", "gemini"]


def build_model_index() -> dict[str, list[str]]:
    """Map every model alias → list of provider names that serve it.

    Scans all registered providers' MODELS dicts. A model alias that appears
    in multiple providers will have multiple entries in the list.
    """
    index: dict[str, list[str]] = {}
    for provider_name, provider_cls in REGISTRY.items():
        for alias in provider_cls.MODELS:
            index.setdefault(alias, []).append(provider_name)
    return index


def resolve_model(
    model: str, preference: list[str] | None = None
) -> tuple[str, str]:
    """Resolve a model alias to (provider_name, full_model_id).

    For models available on multiple providers, uses the preference list
    to pick the best provider. Unknown models raise KeyError.

    Returns:
        Tuple of (provider_name, resolved_model_id)
    """
    pref = preference or _DEFAULT_PREFERENCE
    index = build_model_index()

    if model in index:
        providers = index[model]
        # Pick by preference order
        for p in pref:
            if p in providers:
                return p, REGISTRY[p].MODELS[model]
        # Fallback to first available
        provider_name = providers[0]
        return provider_name, REGISTRY[provider_name].MODELS[model]

    # Not an alias — check if it's a full model ID in any provider
    for provider_name, provider_cls in REGISTRY.items():
        if model in provider_cls.MODELS.values():
            return provider_name, model

    raise KeyError(
        f"Unknown model '{model}'. Not found as alias or full ID in any provider. "
        f"Available aliases: {sorted(index.keys())}"
    )


def list_all_models() -> dict[str, list[str]]:
    """Return all model aliases grouped by provider.

    Returns dict: provider_name → list of aliases.
    """
    result: dict[str, list[str]] = {}
    for provider_name, provider_cls in REGISTRY.items():
        result[provider_name] = sorted(provider_cls.MODELS.keys())
    return result
