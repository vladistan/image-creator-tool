"""Tests for the LiteLLM image generation provider (mocked SDK)."""

import base64

import pytest

from image_creator_tool.errors import PermanentAPIError, TransientAPIError
from image_creator_tool.providers import litellm as litellm_mod
from image_creator_tool.providers.base import GenerationParams
from image_creator_tool.providers.litellm import LiteLLMProvider

_SLEEP = "image_creator_tool.providers.litellm.time.sleep"


class _ImageObject:
    def __init__(self, b64_json=None, url=None):
        self.b64_json = b64_json
        self.url = url


class _ImageResponse:
    def __init__(self, items):
        self.data = items


class _FakeLiteLLM:
    """Stand-in litellm module: records kwargs, returns or raises as configured."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []

    def image_generation(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class _StatusError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def _install(monkeypatch, fake):
    monkeypatch.setattr(litellm_mod, "_load_litellm", lambda: fake)
    return fake


def _params(model="dall-e-3", size=None):
    return GenerationParams(model=model, size=size)


def _b64_response(raw=b"img-bytes"):
    return _ImageResponse([_ImageObject(b64_json=base64.b64encode(raw).decode())])


def test_provider_name():
    assert LiteLLMProvider().name == "litellm"


def test_default_model():
    assert LiteLLMProvider().default_model == "dall-e-3"


def test_resolve_model_alias():
    p = LiteLLMProvider()
    assert p.resolve_model("stable-diffusion-xl") == "stability/stable-diffusion-xl-1024-v1-0"


def test_resolve_model_passthrough():
    p = LiteLLMProvider()
    assert p.resolve_model("replicate/some/model") == "replicate/some/model"


def test_generate_returns_decoded_b64(monkeypatch):
    fake = _install(monkeypatch, _FakeLiteLLM(response=_b64_response(b"hello-png")))
    result = LiteLLMProvider().generate("a cat", params=_params())
    assert result == b"hello-png"
    assert fake.calls[0]["model"] == "dall-e-3"
    assert fake.calls[0]["prompt"] == "a cat"


def test_generate_passes_resolved_model_and_size(monkeypatch):
    fake = _install(monkeypatch, _FakeLiteLLM(response=_b64_response()))
    LiteLLMProvider().generate("x", params=_params("stable-diffusion-xl", size="1024x1024"))
    assert fake.calls[0]["model"] == "stability/stable-diffusion-xl-1024-v1-0"
    assert fake.calls[0]["size"] == "1024x1024"


def test_generate_forwards_config_api_key(monkeypatch):
    fake = _install(monkeypatch, _FakeLiteLLM(response=_b64_response()))
    LiteLLMProvider(api_key="sk-cfg").generate("x", params=_params())  # pragma: allowlist secret
    assert fake.calls[0]["api_key"] == "sk-cfg"  # pragma: allowlist secret


def test_generate_omits_api_key_when_unset(monkeypatch):
    fake = _install(monkeypatch, _FakeLiteLLM(response=_b64_response()))
    LiteLLMProvider().generate("x", params=_params())
    assert "api_key" not in fake.calls[0]


def test_generate_downloads_url_when_no_b64(monkeypatch):
    response = _ImageResponse([_ImageObject(url="https://example.com/img.png")])
    _install(monkeypatch, _FakeLiteLLM(response=response))

    def fake_urlopen(url, timeout=None):
        class _Resp:
            def read(self):
                return b"downloaded"

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        return _Resp()

    monkeypatch.setattr(
        "image_creator_tool.providers.litellm.urllib.request.urlopen", fake_urlopen
    )
    result = LiteLLMProvider().generate("x", params=_params())
    assert result == b"downloaded"


def test_4xx_error_maps_to_permanent(monkeypatch):
    _install(monkeypatch, _FakeLiteLLM(error=_StatusError("bad request", 400)))
    with pytest.raises(PermanentAPIError, match="400"):
        LiteLLMProvider().generate("x", params=_params())


def test_auth_error_maps_to_permanent(monkeypatch):
    _install(monkeypatch, _FakeLiteLLM(error=_StatusError("unauthorized", 401)))
    with pytest.raises(PermanentAPIError, match="401"):
        LiteLLMProvider().generate("x", params=_params())


def test_rate_limit_retries_then_raises_transient(monkeypatch):
    monkeypatch.setattr(_SLEEP, lambda _s: None)
    _install(monkeypatch, _FakeLiteLLM(error=_StatusError("rate limited", 429)))
    with pytest.raises(TransientAPIError, match="Gave up"):
        LiteLLMProvider().generate("x", params=_params())


def test_server_error_is_transient(monkeypatch):
    monkeypatch.setattr(_SLEEP, lambda _s: None)
    _install(monkeypatch, _FakeLiteLLM(error=_StatusError("boom", 500)))
    with pytest.raises(TransientAPIError):
        LiteLLMProvider().generate("x", params=_params())


def test_statusless_error_is_transient(monkeypatch):
    monkeypatch.setattr(_SLEEP, lambda _s: None)
    _install(monkeypatch, _FakeLiteLLM(error=RuntimeError("connection reset")))
    with pytest.raises(TransientAPIError):
        LiteLLMProvider().generate("x", params=_params())


def test_transient_then_success(monkeypatch):
    monkeypatch.setattr(_SLEEP, lambda _s: None)

    class _Flaky(_FakeLiteLLM):
        def image_generation(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise _StatusError("temporary", 503)
            return _b64_response(b"ok")

    fake = _install(monkeypatch, _Flaky())
    result = LiteLLMProvider().generate("x", params=_params())
    assert result == b"ok"
    assert len(fake.calls) == 2


def test_empty_data_raises_permanent(monkeypatch):
    _install(monkeypatch, _FakeLiteLLM(response=_ImageResponse([])))
    with pytest.raises(PermanentAPIError, match="No image data"):
        LiteLLMProvider().generate("x", params=_params())


def test_missing_litellm_raises_permanent(monkeypatch):
    def boom():
        raise PermanentAPIError("litellm not installed. Install with: uv pip install ...")

    monkeypatch.setattr(litellm_mod, "_load_litellm", boom)
    with pytest.raises(PermanentAPIError, match="not installed"):
        LiteLLMProvider().generate("x", params=_params())
