"""Tests for style extraction (vision LLM) and the style library."""

import base64

import pytest

from image_creator_tool import style as style_lib
from image_creator_tool.errors import PermanentAPIError, TransientAPIError


@pytest.fixture
def styles_dir(tmp_path, monkeypatch):
    """Redirect the style library to a temp directory."""
    target = tmp_path / "styles"
    monkeypatch.setattr(style_lib, "STYLES_DIR", target)
    return target


@pytest.fixture
def sample_image(tmp_path):
    """Create a tiny PNG file on disk and return its path."""
    path = tmp_path / "ref.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")
    return path


def _openai_response(text):
    return {"choices": [{"message": {"content": text}}]}


def _gemini_response(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


# --- extract_style (Step 3.1) ---


def test_extract_style_openai_returns_description(monkeypatch, sample_image):
    monkeypatch.setattr(
        style_lib, "_http_post_json",
        lambda *a, **k: _openai_response("oil painting, warm tones"),
    )
    result = style_lib.extract_style(sample_image, provider="openai", api_key="k")
    assert result == "oil painting, warm tones"


def test_extract_style_openai_encodes_image_as_data_uri(monkeypatch, sample_image):
    captured = {}

    def fake_post(url, payload, headers, timeout=120, span_data=None):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return _openai_response("style")

    monkeypatch.setattr(style_lib, "_http_post_json", fake_post)
    style_lib.extract_style(
        sample_image, provider="openai", api_key="secret"  # pragma: allowlist secret
    )

    assert captured["url"] == style_lib._OPENAI_URL
    assert captured["headers"]["Authorization"] == "Bearer secret"
    content = captured["payload"]["messages"][0]["content"]
    image_url = content[1]["image_url"]["url"]
    expected_b64 = base64.b64encode(sample_image.read_bytes()).decode()
    assert image_url == f"data:image/png;base64,{expected_b64}"


def test_extract_style_gemini_uses_generatecontent_endpoint(monkeypatch, sample_image):
    captured = {}

    def fake_post(url, payload, headers, timeout=120, span_data=None):
        captured["url"] = url
        captured["headers"] = headers
        return _gemini_response("cyberpunk, neon")

    monkeypatch.setattr(style_lib, "_http_post_json", fake_post)
    result = style_lib.extract_style(sample_image, provider="gemini", api_key="gk")

    assert result == "cyberpunk, neon"
    assert "generateContent" in captured["url"]
    assert "gemini-2.0-flash" in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "gk"


def test_extract_style_strips_markdown_fences(monkeypatch, sample_image):
    monkeypatch.setattr(
        style_lib, "_http_post_json",
        lambda *a, **k: _openai_response("```\nwatercolour, pastel palette\n```"),
    )
    result = style_lib.extract_style(sample_image, provider="openai", api_key="k")
    assert result == "watercolour, pastel palette"


def test_extract_style_empty_response_raises(monkeypatch, sample_image):
    monkeypatch.setattr(
        style_lib, "_http_post_json",
        lambda *a, **k: _openai_response("   "),
    )
    with pytest.raises(PermanentAPIError, match="empty style"):
        style_lib.extract_style(sample_image, provider="openai", api_key="k")


def test_extract_style_missing_image_raises(tmp_path):
    with pytest.raises(PermanentAPIError, match="not found"):
        style_lib.extract_style(tmp_path / "nope.png", provider="openai", api_key="k")


def test_extract_style_unknown_provider_raises(sample_image):
    with pytest.raises(PermanentAPIError, match="Unknown vision provider"):
        style_lib.extract_style(sample_image, provider="dalle", api_key="k")


def test_extract_style_missing_key_raises(monkeypatch, sample_image):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(PermanentAPIError, match="requires an API key"):
        style_lib.extract_style(sample_image, provider="openai")


def test_extract_style_transient_error_propagates(monkeypatch, sample_image):
    def always_transient(url, payload, headers, timeout=120, span_data=None):
        raise TransientAPIError("boom")

    monkeypatch.setattr(style_lib, "_http_post_json", always_transient)
    monkeypatch.setattr(style_lib.time, "sleep", lambda _s: None)
    with pytest.raises(TransientAPIError, match="Gave up"):
        style_lib.extract_style(sample_image, provider="openai", api_key="k")


# --- style library (Step 3.2) ---


def test_save_and_load_style_roundtrip(styles_dir):
    style_lib.save_style("My Style", "oil painting, warm tones")
    assert style_lib.load_style("My Style") == "oil painting, warm tones"


def test_save_style_slugifies_name(styles_dir):
    path = style_lib.save_style("Vintage Film!", "grainy, faded")
    assert path.name == "vintage-film.txt"


def test_load_style_accepts_unslugified_name(styles_dir):
    style_lib.save_style("vintage-film", "grainy")
    assert style_lib.load_style("Vintage Film") == "grainy"


def test_list_styles_returns_sorted_slugs(styles_dir):
    style_lib.save_style("beta", "b")
    style_lib.save_style("alpha", "a")
    assert style_lib.list_styles() == ["alpha", "beta"]


def test_list_styles_empty_when_no_dir(styles_dir):
    assert style_lib.list_styles() == []


def test_delete_style_removes_file(styles_dir):
    style_lib.save_style("temp", "x")
    style_lib.delete_style("temp")
    assert style_lib.list_styles() == []


def test_delete_missing_style_raises(styles_dir):
    with pytest.raises(PermanentAPIError, match="not found"):
        style_lib.delete_style("ghost")


def test_load_missing_style_raises(styles_dir):
    with pytest.raises(PermanentAPIError, match="not found"):
        style_lib.load_style("ghost")


def test_save_duplicate_without_overwrite_raises(styles_dir):
    style_lib.save_style("dup", "first")
    with pytest.raises(PermanentAPIError, match="already exists"):
        style_lib.save_style("dup", "second")


def test_save_duplicate_with_overwrite_succeeds(styles_dir):
    style_lib.save_style("dup", "first")
    style_lib.save_style("dup", "second", overwrite=True)
    assert style_lib.load_style("dup") == "second"


def test_save_style_empty_name_raises(styles_dir):
    with pytest.raises(PermanentAPIError, match="Invalid style name"):
        style_lib.save_style("!!!", "desc")
