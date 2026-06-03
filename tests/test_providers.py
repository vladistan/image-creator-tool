"""Tests for the provider registry and GeminiProvider."""

import pytest

from image_creator_tool.errors import ImageCreatorError, PermanentAPIError, TransientAPIError
from image_creator_tool.providers import get_provider, list_providers
from image_creator_tool.providers.gemini import GeminiProvider


def test_list_providers_includes_gemini():
    assert "gemini" in list_providers()


def test_get_provider_returns_gemini():
    provider = get_provider("gemini")
    assert isinstance(provider, GeminiProvider)


def test_unknown_provider_raises():
    with pytest.raises(KeyError, match="Unknown provider"):
        get_provider("nonexistent")


def test_gemini_provider_name():
    p = GeminiProvider()
    assert p.name == "gemini"


def test_gemini_provider_default_model():
    p = GeminiProvider()
    assert p.default_model == "gemini-3.1-flash-image-preview"


def test_gemini_provider_model_aliases():
    p = GeminiProvider()
    assert "flash" in p.MODELS
    assert "pro" in p.MODELS
    assert "flash-2.5" in p.MODELS


def test_gemini_provider_resolve_model_alias():
    p = GeminiProvider()
    assert p.resolve_model("flash") == "gemini-3.1-flash-image-preview"
    assert p.resolve_model("pro") == "gemini-3-pro-image-preview"


def test_gemini_provider_resolve_model_passthrough():
    p = GeminiProvider()
    full_id = "gemini-custom-model-v1"
    assert p.resolve_model(full_id) == full_id


def test_transient_is_image_creator_error():
    err = TransientAPIError("rate limited")
    assert isinstance(err, TransientAPIError)
    assert isinstance(err, ImageCreatorError)


def test_permanent_is_image_creator_error():
    err = PermanentAPIError("content blocked")
    assert isinstance(err, PermanentAPIError)
    assert isinstance(err, ImageCreatorError)
