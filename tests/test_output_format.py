"""Tests for deterministic output format normalization (Phase 12).

Covers the `normalize_image_bytes` helper, `generate_once` normalization of
provider bytes, and the `--format` CLI flag validation/marshalling.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from image_creator_tool.cli import app
from image_creator_tool.generation_core import generate_once
from image_creator_tool.imaging import normalize_image_bytes
from image_creator_tool.provenance import ProvenanceRecord, embed_exif_metadata


def _image_bytes(fmt: str, mode: str = "RGB", color: tuple[int, ...] = (10, 20, 30)) -> bytes:
    """Encode a tiny solid-colour image in ``fmt`` (Pillow save format)."""
    buffer = BytesIO()
    Image.new(mode, (8, 8), color).save(buffer, format=fmt)
    return buffer.getvalue()


def _detect(data: bytes) -> str:
    """Return Pillow's detected format string for ``data``."""
    with Image.open(BytesIO(data)) as img:
        return img.format or ""


# --- normalize_image_bytes --------------------------------------------------


@pytest.mark.parametrize("target,expected", [("png", "PNG"), ("webp", "WEBP"), ("jpg", "JPEG")])
def test_normalize_from_webp(target, expected):
    """A webp-native payload re-encodes to each supported target format."""
    out = normalize_image_bytes(_image_bytes("WEBP"), target)
    assert _detect(out) == expected


@pytest.mark.parametrize("fmt,pil", [("PNG", "PNG"), ("WEBP", "WEBP"), ("JPEG", "JPEG")])
def test_normalize_round_trips(fmt, pil):
    """png/webp/jpg each survive a normalize round-trip to their own format."""
    target = {"PNG": "png", "WEBP": "webp", "JPEG": "jpg"}[fmt]
    out = normalize_image_bytes(_image_bytes(fmt), target)
    assert _detect(out) == pil


def test_normalize_jpeg_flattens_alpha():
    """RGBA input flattens onto white when normalized to JPEG (no alpha channel)."""
    out = normalize_image_bytes(_image_bytes("PNG", mode="RGBA", color=(10, 20, 30, 0)), "jpg")
    with Image.open(BytesIO(out)) as img:
        assert img.mode == "RGB"


def test_normalize_unknown_format_raises():
    with pytest.raises(ValueError, match="unsupported output format"):
        normalize_image_bytes(_image_bytes("PNG"), "tiff")


# --- generate_once normalization --------------------------------------------


class _FakeProvider:
    """Minimal provider returning fixed bytes, for generate_once tests."""

    name = "fake"

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def generate(self, prompt: str, *, params) -> bytes:
        return self._payload


def test_generate_once_webp_provider_normalized_to_png(tmp_path):
    """A webp-emitting provider + default png yields an embeddable .png file."""
    provider = _FakeProvider(_image_bytes("WEBP"))
    out = tmp_path / "barn.png"

    result = generate_once(
        "a barn", out, provider, "fake-model", explicit_path=True, output_format="png"
    )

    assert result.output_path.suffix == ".png"
    assert _detect(result.output_path.read_bytes()) == "PNG"

    rec = ProvenanceRecord(
        prompt="a barn",
        model="fake-model",
        provider="fake",
        output_path=str(result.output_path),
        timestamp="2026-07-11T00:00:00",
    )
    # The whole point of Phase 12: embedding now succeeds instead of the ".webp
    # unsupported format" skip.
    assert embed_exif_metadata(result.output_path, rec) is True


@pytest.mark.parametrize("fmt,suffix,pil", [("png", ".png", "PNG"), ("jpg", ".jpg", "JPEG")])
def test_generate_once_honours_requested_format(tmp_path, fmt, suffix, pil):
    """generate_once persists the requested format even from webp provider bytes."""
    provider = _FakeProvider(_image_bytes("WEBP"))
    out = tmp_path / "img.bin"

    result = generate_once(
        "x", out, provider, "fake-model", explicit_path=False, output_format=fmt
    )

    assert result.output_path.suffix == suffix
    assert _detect(result.output_path.read_bytes()) == pil


def test_generate_once_no_format_preserves_detection(tmp_path):
    """Without output_format, provider bytes are written as-detected (webp → .webp)."""
    provider = _FakeProvider(_image_bytes("WEBP"))
    out = tmp_path / "img.png"

    result = generate_once("x", out, provider, "fake-model", explicit_path=False)

    assert result.output_path.suffix == ".webp"


# --- CLI --format flag ------------------------------------------------------


def _capture_generate_args(monkeypatch):
    """Replace cli.generate with a capturing stub; return the mutable holder."""
    holder: dict = {}

    def _capture(args, config, provider):
        holder["args"] = args
        return []

    monkeypatch.setattr("image_creator_tool.cli.generate", _capture)
    return holder


def test_format_defaults_to_png(runner, monkeypatch):
    holder = _capture_generate_args(monkeypatch)
    result = runner.invoke(app, ["generate", "a robot", "--provider", "gemini"])
    assert result.exit_code == 0
    assert holder["args"].output_format == "png"


@pytest.mark.parametrize("value", ["webp", "jpg", "PNG"])
def test_format_explicit_value_marshalled(runner, monkeypatch, value):
    holder = _capture_generate_args(monkeypatch)
    result = runner.invoke(
        app, ["generate", "a robot", "--provider", "gemini", "--format", value]
    )
    assert result.exit_code == 0
    assert holder["args"].output_format == value.lower()


def test_format_invalid_is_usage_error(runner, monkeypatch):
    _capture_generate_args(monkeypatch)
    result = runner.invoke(
        app, ["generate", "a robot", "--provider", "gemini", "--format", "tiff"]
    )
    assert result.exit_code == 2
    assert "unsupported --format" in result.output
