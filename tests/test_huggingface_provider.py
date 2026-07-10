"""Tests for the HuggingFace image generation provider (mocked InferenceClient)."""

import pytest
from PIL import Image

from image_creator_tool.errors import PermanentAPIError, TransientAPIError
from image_creator_tool.providers import huggingface as hf_mod
from image_creator_tool.providers.base import GenerationParams
from image_creator_tool.providers.huggingface import HuggingFaceProvider

_SLEEP = "image_creator_tool.providers.huggingface.time.sleep"
_TOKEN = "hf-test-token"  # pragma: allowlist secret

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _StatusError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.response = _FakeResponse(status_code)


def _small_image():
    return Image.new("RGB", (4, 4), (255, 0, 0))


def _install(monkeypatch, results):
    """Patch the loader with a client whose text_to_image consumes `results`.

    Each result is either an Exception (raised) or a PIL image (returned). A new
    client is constructed per call, so results/records live in the closure to
    survive across retry attempts.
    """
    seq = list(results)
    record = {"calls": [], "init_kwargs": None}

    class _Client:
        def __init__(self, **kwargs):
            record["init_kwargs"] = kwargs

        def text_to_image(self, prompt, **kwargs):
            record["calls"].append({"prompt": prompt, **kwargs})
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    monkeypatch.setattr(hf_mod, "_load_inference_client_cls", lambda: _Client)
    return record


def _params(model="flux-schnell", size=None, seed=None, edit_source=None, reference_images=None):
    return GenerationParams(
        model=model,
        size=size,
        seed=seed,
        edit_source=edit_source,
        reference_images=reference_images or [],
    )


def test_provider_name():
    assert HuggingFaceProvider().name == "huggingface"


def test_default_model():
    assert HuggingFaceProvider().default_model == "flux-schnell"


def test_model_aliases_present():
    p = HuggingFaceProvider()
    assert "flux-schnell" in p.MODELS
    assert "sdxl-turbo" in p.MODELS
    assert "stable-diffusion-xl" in p.MODELS


def test_resolve_model_alias():
    p = HuggingFaceProvider()
    assert p.resolve_model("flux-schnell") == "black-forest-labs/FLUX.1-schnell"


def test_resolve_model_passthrough():
    p = HuggingFaceProvider()
    assert p.resolve_model("my-org/my-finetune") == "my-org/my-finetune"


def test_missing_api_key_raises_permanent(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(PermanentAPIError, match="API key not set"):
        HuggingFaceProvider().generate("a cat", params=_params())


def test_generate_returns_png_bytes(monkeypatch):
    _install(monkeypatch, [_small_image()])
    result = HuggingFaceProvider(api_key=_TOKEN).generate("a cat", params=_params())
    assert result[:8] == _PNG_SIGNATURE


def test_client_constructed_with_auto_provider_and_key(monkeypatch):
    record = _install(monkeypatch, [_small_image()])
    HuggingFaceProvider(api_key=_TOKEN).generate("a cat", params=_params())
    assert record["init_kwargs"]["provider"] == "auto"
    assert record["init_kwargs"]["api_key"] == _TOKEN


def test_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-token")  # pragma: allowlist secret
    record = _install(monkeypatch, [_small_image()])
    HuggingFaceProvider().generate("a cat", params=_params())
    assert record["init_kwargs"]["api_key"] == "env-token"  # pragma: allowlist secret


def test_model_alias_resolved_before_call(monkeypatch):
    record = _install(monkeypatch, [_small_image()])
    HuggingFaceProvider(api_key=_TOKEN).generate("a cat", params=_params(model="flux-schnell"))
    assert record["calls"][0]["model"] == "black-forest-labs/FLUX.1-schnell"


def test_custom_slug_passthrough(monkeypatch):
    record = _install(monkeypatch, [_small_image()])
    HuggingFaceProvider(api_key=_TOKEN).generate("a cat", params=_params(model="krea/Krea-2-Turbo"))
    assert record["calls"][0]["model"] == "krea/Krea-2-Turbo"


def test_size_forwarded_as_width_height(monkeypatch):
    record = _install(monkeypatch, [_small_image()])
    HuggingFaceProvider(api_key=_TOKEN).generate("a cat", params=_params(size="512x768"))
    assert record["calls"][0]["width"] == 512
    assert record["calls"][0]["height"] == 768


def test_seed_forwarded(monkeypatch):
    record = _install(monkeypatch, [_small_image()])
    HuggingFaceProvider(api_key=_TOKEN).generate("a cat", params=_params(seed=42))
    assert record["calls"][0]["seed"] == 42


def test_edit_source_ignored_with_warning(monkeypatch, tmp_path):
    source = tmp_path / "src.png"
    source.write_bytes(b"data")
    _install(monkeypatch, [_small_image()])
    result = HuggingFaceProvider(api_key=_TOKEN).generate(
        "a cat", params=_params(edit_source=source)
    )
    assert result[:8] == _PNG_SIGNATURE


def test_4xx_maps_to_permanent(monkeypatch):
    _install(monkeypatch, [_StatusError("bad request", 400)])
    with pytest.raises(PermanentAPIError, match="400"):
        HuggingFaceProvider(api_key=_TOKEN).generate("a cat", params=_params())


def test_rate_limit_retries_then_raises_transient(monkeypatch):
    monkeypatch.setattr(_SLEEP, lambda _s: None)
    _install(monkeypatch, [_StatusError("rate", 429)] * 3)
    with pytest.raises(TransientAPIError, match="Gave up"):
        HuggingFaceProvider(api_key=_TOKEN).generate("a cat", params=_params())


def test_server_error_is_transient(monkeypatch):
    monkeypatch.setattr(_SLEEP, lambda _s: None)
    _install(monkeypatch, [_StatusError("boom", 503)] * 3)
    with pytest.raises(TransientAPIError):
        HuggingFaceProvider(api_key=_TOKEN).generate("a cat", params=_params())


def test_transient_then_success(monkeypatch):
    monkeypatch.setattr(_SLEEP, lambda _s: None)
    record = _install(monkeypatch, [_StatusError("temporary", 500), _small_image()])
    result = HuggingFaceProvider(api_key=_TOKEN).generate("a cat", params=_params())
    assert result[:8] == _PNG_SIGNATURE
    assert len(record["calls"]) == 2


def test_client_construction_error_is_mapped(monkeypatch):
    class _BadClient:
        def __init__(self, **kwargs):
            raise _StatusError("bad provider", 400)

    monkeypatch.setattr(hf_mod, "_load_inference_client_cls", lambda: _BadClient)
    with pytest.raises(PermanentAPIError, match="400"):
        HuggingFaceProvider(api_key=_TOKEN).generate("a cat", params=_params())
