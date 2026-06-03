"""Tests for imaging utilities — contact sheet configurability."""

import pytest
from PIL import Image

from image_creator_tool.imaging import make_contact_sheet, make_labeled_contact_sheet


def _make_test_image(path, color, size=(100, 100)):
    """Write a small solid-colour PNG to disk."""
    img = Image.new("RGB", size, color)
    img.save(path)
    return path


def test_make_contact_sheet_default_cols(tmp_path):
    imgs = [
        _make_test_image(tmp_path / f"img{i}.png", (i * 40, 0, 0))
        for i in range(4)
    ]
    out = tmp_path / "sheet.png"
    result = make_contact_sheet(imgs, out)
    assert result == out
    assert out.exists()
    with Image.open(out) as sheet:
        # 4 images in 2 cols → 2 rows
        assert sheet.width > 0
        assert sheet.height > 0


def test_make_contact_sheet_custom_cols(tmp_path):
    imgs = [_make_test_image(tmp_path / f"img{i}.png", (0, i * 40, 0)) for i in range(4)]
    out = tmp_path / "sheet_1col.png"
    result = make_contact_sheet(imgs, out, cols=1)
    assert result.exists()
    with Image.open(result) as sheet:
        # 1 col → narrow sheet
        assert sheet.width < sheet.height


def test_make_contact_sheet_custom_cell_width(tmp_path):
    imgs = [_make_test_image(tmp_path / f"img{i}.png", (0, 0, i * 60)) for i in range(2)]
    out_small = tmp_path / "small.png"
    out_large = tmp_path / "large.png"
    make_contact_sheet(imgs, out_small, cols=2, cell_width=200)
    make_contact_sheet(imgs, out_large, cols=2, cell_width=600)
    with Image.open(out_small) as small_sheet, Image.open(out_large) as large_sheet:
        assert large_sheet.width > small_sheet.width


def test_make_contact_sheet_custom_bg_color(tmp_path):
    imgs = [_make_test_image(tmp_path / f"img{i}.png", (128, 128, 128)) for i in range(2)]
    out = tmp_path / "red_bg.png"
    make_contact_sheet(imgs, out, cols=2, cell_width=50, spacing=5, bg_color=(200, 0, 0))
    assert out.exists()
    with Image.open(out) as sheet:
        # Top-left corner should be approximately red (bg_color)
        pixel = sheet.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        assert pixel[0] > 150  # red component dominant


def test_make_contact_sheet_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="No images"):
        make_contact_sheet([], tmp_path / "empty.png")


def test_make_labeled_contact_sheet_basic(tmp_path):
    cells = [
        (_make_test_image(tmp_path / f"img{i}.png", (i * 50, i * 30, 0)), f"label_{i}")
        for i in range(4)
    ]
    out = tmp_path / "labeled.png"
    result = make_labeled_contact_sheet(cells, out, cols=2, title="Test Sheet")
    assert result == out
    assert out.exists()


def test_make_labeled_contact_sheet_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="No cells"):
        make_labeled_contact_sheet([], tmp_path / "empty.png")
