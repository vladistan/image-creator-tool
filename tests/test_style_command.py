"""Tests for the `style` command group and `generate --style` integration."""

import pytest

from image_creator_tool import style as style_lib
from image_creator_tool.cli import app


@pytest.fixture
def styles_dir(tmp_path, monkeypatch):
    """Redirect the style library to a temp directory."""
    target = tmp_path / "styles"
    monkeypatch.setattr(style_lib, "STYLES_DIR", target)
    return target


def test_style_save_and_list(runner, styles_dir):
    save = runner.invoke(app, ["style", "save", "noir", "high contrast, moody shadows"])
    assert save.exit_code == 0

    listing = runner.invoke(app, ["style", "list"])
    assert listing.exit_code == 0
    assert "noir" in listing.stdout


def test_style_show(runner, styles_dir):
    runner.invoke(app, ["style", "save", "noir", "high contrast, moody shadows"])
    result = runner.invoke(app, ["style", "show", "noir"])
    assert result.exit_code == 0
    assert "high contrast, moody shadows" in result.stdout


def test_style_delete(runner, styles_dir):
    runner.invoke(app, ["style", "save", "temp", "x"])
    result = runner.invoke(app, ["style", "delete", "temp"])
    assert result.exit_code == 0
    assert style_lib.list_styles() == []


def test_style_show_missing_exits_nonzero(runner, styles_dir):
    result = runner.invoke(app, ["style", "show", "ghost"])
    assert result.exit_code != 0


def test_style_list_empty(runner, styles_dir):
    result = runner.invoke(app, ["style", "list"])
    assert result.exit_code == 0
    assert "No saved styles" in result.stdout


def test_style_extract_prints_and_saves(runner, styles_dir, tmp_path, monkeypatch):
    image = tmp_path / "ref.png"
    image.write_bytes(b"fake")
    monkeypatch.setattr(
        style_lib, "extract_style",
        lambda img, provider="openai", vision_model=None: "impressionist, soft light",
    )
    result = runner.invoke(app, ["style", "extract", str(image), "--save", "impr"])
    assert result.exit_code == 0
    assert "impressionist, soft light" in result.stdout
    assert style_lib.load_style("impr") == "impressionist, soft light"


def test_generate_style_prepends_description(runner, styles_dir):
    style_lib.save_style("noir", "high contrast, moody shadows")
    result = runner.invoke(
        app,
        ["generate", "a farmhouse", "--style", "noir", "--provider", "gemini", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "high contrast, moody shadows, a farmhouse" in result.stdout


def test_generate_unknown_style_exits_nonzero(runner, styles_dir):
    result = runner.invoke(
        app,
        ["generate", "a farmhouse", "--style", "ghost", "--provider", "gemini", "--dry-run"],
    )
    assert result.exit_code != 0
