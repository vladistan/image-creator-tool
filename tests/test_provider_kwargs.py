"""Regression tests for per-provider api_key scoping in provider-kwargs builders.

Guards against the cross-provider credential leak where the active/default
profile's api_key (e.g. an OpenRouter key) was handed to an unrelated provider
selected via `-p` (e.g. huggingface), shadowing that provider's own env-var
credential resolution.
"""

from image_creator_tool.cli import _build_cli_provider_kwargs
from image_creator_tool.config import build_provider_kwargs

_LEAK_KEY = "sk-or-should-not-leak"  # pragma: allowlist secret
_HF_KEY = "hf-profile-key"  # pragma: allowlist secret


def _patch_profiles(monkeypatch, profiles):
    monkeypatch.setattr("image_creator_tool.config._load_profiles_raw", lambda: profiles)


def test_build_kwargs_drops_mismatched_default_profile_key(monkeypatch):
    _patch_profiles(monkeypatch, {"openrouter": {"provider": "openrouter", "api_key": _LEAK_KEY}})
    config = {"default_provider": "openrouter", "api_key": _LEAK_KEY}
    kwargs = build_provider_kwargs("huggingface", config)
    assert "api_key" not in kwargs


def test_build_kwargs_uses_matching_provider_profile_key(monkeypatch):
    _patch_profiles(
        monkeypatch,
        {
            "openrouter": {"provider": "openrouter", "api_key": _LEAK_KEY},
            "hf-primary": {"provider": "huggingface", "api_key": _HF_KEY},
        },
    )
    config = {"default_provider": "openrouter", "api_key": _LEAK_KEY}
    kwargs = build_provider_kwargs("huggingface", config)
    assert kwargs["api_key"] == _HF_KEY


def test_build_kwargs_passes_active_default_key_when_provider_matches(monkeypatch):
    _patch_profiles(monkeypatch, {"openrouter": {"provider": "openrouter", "api_key": _LEAK_KEY}})
    config = {"default_provider": "openrouter", "api_key": _LEAK_KEY}
    kwargs = build_provider_kwargs("openrouter", config)
    assert kwargs["api_key"] == _LEAK_KEY


def test_cli_kwargs_drops_mismatched_default_profile_key():
    config = {"default_provider": "openrouter", "api_key": _LEAK_KEY}
    kwargs = _build_cli_provider_kwargs("huggingface", config)
    assert "api_key" not in kwargs


def test_cli_kwargs_passes_key_when_provider_matches():
    config = {"default_provider": "openrouter", "api_key": _LEAK_KEY}
    kwargs = _build_cli_provider_kwargs("openrouter", config)
    assert kwargs["api_key"] == _LEAK_KEY
