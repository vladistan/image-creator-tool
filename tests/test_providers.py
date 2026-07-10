"""Tests for the provider registry and GeminiProvider."""

from pathlib import Path

import pytest

from image_creator_tool.errors import ImageCreatorError, PermanentAPIError, TransientAPIError
from image_creator_tool.providers import get_provider, list_providers
from image_creator_tool.providers import vertex as vertex_mod
from image_creator_tool.providers.gemini import GeminiProvider
from image_creator_tool.providers.vertex import VertexProvider


def test_list_providers_includes_gemini():
    assert "gemini" in list_providers()


def test_get_provider_returns_gemini():
    provider = get_provider("gemini")
    assert isinstance(provider, GeminiProvider)


def test_unknown_provider_raises():
    with pytest.raises(KeyError, match="Unknown provider"):
        get_provider("nonexistent")


@pytest.mark.parametrize("provider_name", ["huggingface", "litellm", "azure-openai"])
def test_phase6_providers_registered(provider_name):
    assert provider_name in list_providers()


def test_get_provider_returns_huggingface():
    from image_creator_tool.providers.huggingface import HuggingFaceProvider  # noqa: PLC0415

    assert isinstance(get_provider("huggingface"), HuggingFaceProvider)


def test_get_provider_returns_litellm():
    from image_creator_tool.providers.litellm import LiteLLMProvider  # noqa: PLC0415

    assert isinstance(get_provider("litellm"), LiteLLMProvider)


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


# Vertex model registry migration regression tests (T300: Imagen 4 sunset removal).


def test_vertex_models_have_no_imagen_aliases():
    p = VertexProvider(project="test-project")
    assert "imagen" not in p.MODELS
    assert "imagen-ultra" not in p.MODELS
    assert "imagen-fast" not in p.MODELS


def test_vertex_resolve_flash_31_still_maps_to_gemini():
    p = VertexProvider(project="test-project")
    assert p.resolve_model("flash-3.1") == "gemini-3.1-flash-image-preview"


def test_vertex_stale_imagen_alias_is_not_silently_remapped():
    # A stale --model imagen must pass through unchanged (honest failure downstream),
    # never silently redirect to a Gemini model.
    p = VertexProvider(project="test-project")
    assert p.resolve_model("imagen") == "imagen"


def test_vertex_has_no_imagen_symbols_in_source():
    source = Path(vertex_mod.__file__).read_text()
    assert "imagen" not in source.lower()
    assert not hasattr(VertexProvider, "_call_imagen")


def test_vertex_call_once_routes_to_gemini_unconditionally(monkeypatch):
    p = VertexProvider(project="test-project")
    captured = {}

    def fake_gemini(prompt, model, edit_source, reference_images, seed=None):
        captured["model"] = model
        return b"image-bytes"

    monkeypatch.setattr(p, "_call_gemini", fake_gemini)
    result = p._call_once("a prompt", "any-model-id", None, None)
    assert result == b"image-bytes"
    assert captured["model"] == "any-model-id"


def test_transient_is_image_creator_error():
    err = TransientAPIError("rate limited")
    assert isinstance(err, TransientAPIError)
    assert isinstance(err, ImageCreatorError)


def test_permanent_is_image_creator_error():
    err = PermanentAPIError("content blocked")
    assert isinstance(err, PermanentAPIError)
    assert isinstance(err, ImageCreatorError)
