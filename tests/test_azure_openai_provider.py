"""Tests for the Azure OpenAI image generation provider (mocked SDK)."""

import base64

import pytest

from image_creator_tool.errors import PermanentAPIError, TransientAPIError
from image_creator_tool.providers import azure_openai as azure_mod
from image_creator_tool.providers.azure_openai import AzureOpenAIProvider
from image_creator_tool.providers.base import GenerationParams

_SLEEP = "image_creator_tool.providers.azure_openai.time.sleep"

_ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
    "AZURE_OPENAI_API_KEY": "azure-key",  # pragma: allowlist secret
    "AZURE_OPENAI_DEPLOYMENT": "my-dalle",
}


class _ImageObject:
    def __init__(self, b64_json=None, url=None):
        self.b64_json = b64_json
        self.url = url


class _ImagesResponse:
    def __init__(self, items):
        self.data = items


class _StatusError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class _FakeImages:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.generate_calls = []
        self.edit_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response

    def edit(self, **kwargs):
        self.edit_calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class _FakeClient:
    def __init__(self, images):
        self.images = images
        self.init_kwargs = None


def _install(monkeypatch, images):
    """Patch the SDK loader to return a client class capturing init kwargs."""
    created = {}

    class _ClientCls:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs
            self.images = images

    monkeypatch.setattr(azure_mod, "_load_azure_client_cls", lambda: _ClientCls)
    return created


def _configured_env(monkeypatch):
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)


def _params(size=None, edit_source=None):
    return GenerationParams(model="dall-e-3", size=size, edit_source=edit_source)


def _b64_response(raw=b"img-bytes"):
    return _ImagesResponse([_ImageObject(b64_json=base64.b64encode(raw).decode())])


def test_provider_name(monkeypatch):
    _configured_env(monkeypatch)
    assert AzureOpenAIProvider().name == "azure-openai"


def test_missing_config_raises_permanent_at_init(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(PermanentAPIError, match="not configured"):
        AzureOpenAIProvider()


def test_partial_config_raises_permanent_at_init(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")  # pragma: allowlist secret
    with pytest.raises(PermanentAPIError, match="not configured"):
        AzureOpenAIProvider()


def test_explicit_config_overrides_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    provider = AzureOpenAIProvider(
        api_key="cfg-key", endpoint="https://cfg.example.com", deployment="cfg-deploy",
# pragma: allowlist secret
    )
    assert provider.get_api_key() == "cfg-key"  # pragma: allowlist secret


def test_generate_returns_decoded_b64(monkeypatch):
    _configured_env(monkeypatch)
    images = _FakeImages(response=_b64_response(b"hello-png"))
    _install(monkeypatch, images)
    result = AzureOpenAIProvider().generate("a cat", params=_params())
    assert result == b"hello-png"
    assert images.generate_calls[0]["model"] == "my-dalle"
    assert images.generate_calls[0]["prompt"] == "a cat"


def test_client_constructed_with_endpoint_and_key(monkeypatch):
    _configured_env(monkeypatch)
    created = _install(monkeypatch, _FakeImages(response=_b64_response()))
    AzureOpenAIProvider().generate("x", params=_params())
    assert created["kwargs"]["azure_endpoint"] == "https://example.openai.azure.com"
    assert created["kwargs"]["api_key"] == "azure-key"  # pragma: allowlist secret


def test_generate_forwards_size(monkeypatch):
    _configured_env(monkeypatch)
    images = _FakeImages(response=_b64_response())
    _install(monkeypatch, images)
    AzureOpenAIProvider().generate("x", params=_params(size="512x512"))
    assert images.generate_calls[0]["size"] == "512x512"


def test_edit_source_routes_to_edit_endpoint(monkeypatch, tmp_path):
    _configured_env(monkeypatch)
    source = tmp_path / "src.png"
    source.write_bytes(b"\x89PNG-source")
    images = _FakeImages(response=_b64_response(b"edited"))
    _install(monkeypatch, images)

    result = AzureOpenAIProvider().generate("make it blue", params=_params(edit_source=source))

    assert result == b"edited"
    assert len(images.edit_calls) == 1
    assert images.generate_calls == []
    assert images.edit_calls[0]["prompt"] == "make it blue"


def test_generate_downloads_url_when_no_b64(monkeypatch):
    _configured_env(monkeypatch)
    response = _ImagesResponse([_ImageObject(url="https://example.com/img.png")])
    _install(monkeypatch, _FakeImages(response=response))

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
        "image_creator_tool.providers.azure_openai.urllib.request.urlopen", fake_urlopen
    )
    result = AzureOpenAIProvider().generate("x", params=_params())
    assert result == b"downloaded"


def test_4xx_error_maps_to_permanent(monkeypatch):
    _configured_env(monkeypatch)
    _install(monkeypatch, _FakeImages(error=_StatusError("bad request", 400)))
    with pytest.raises(PermanentAPIError, match="400"):
        AzureOpenAIProvider().generate("x", params=_params())


def test_rate_limit_retries_then_raises_transient(monkeypatch):
    _configured_env(monkeypatch)
    monkeypatch.setattr(_SLEEP, lambda _s: None)
    _install(monkeypatch, _FakeImages(error=_StatusError("rate limited", 429)))
    with pytest.raises(TransientAPIError, match="Gave up"):
        AzureOpenAIProvider().generate("x", params=_params())


def test_server_error_is_transient(monkeypatch):
    _configured_env(monkeypatch)
    monkeypatch.setattr(_SLEEP, lambda _s: None)
    _install(monkeypatch, _FakeImages(error=_StatusError("boom", 503)))
    with pytest.raises(TransientAPIError):
        AzureOpenAIProvider().generate("x", params=_params())


def test_transient_then_success(monkeypatch):
    _configured_env(monkeypatch)
    monkeypatch.setattr(_SLEEP, lambda _s: None)

    class _FlakyImages(_FakeImages):
        def generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            if len(self.generate_calls) == 1:
                raise _StatusError("temporary", 500)
            return _b64_response(b"ok")

    images = _FlakyImages()
    _install(monkeypatch, images)
    result = AzureOpenAIProvider().generate("x", params=_params())
    assert result == b"ok"
    assert len(images.generate_calls) == 2


def test_empty_data_raises_permanent(monkeypatch):
    _configured_env(monkeypatch)
    _install(monkeypatch, _FakeImages(response=_ImagesResponse([])))
    with pytest.raises(PermanentAPIError, match="No image data"):
        AzureOpenAIProvider().generate("x", params=_params())
