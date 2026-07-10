"""Tests for imaging utilities — contact sheet configurability and badges."""

import pytest
from PIL import Image, ImageDraw, ImageFont

from image_creator_tool import imaging
from image_creator_tool.imaging import (
    _BADGE_INSET,
    _LABEL_HEIGHT,
    _draw_badge,
    _load_badge_font,
    make_contact_sheet,
    make_labeled_contact_sheet,
)

# Non-red fill so the only red-dominant pixels come from badges.
_CELL_COLOR = (0, 0, 200)


def _make_test_image(path, color, size=(100, 100)):
    """Write a small solid-colour PNG to disk."""
    img = Image.new("RGB", size, color)
    img.save(path)
    return path


def _is_red(px):
    return px[0] > 150 and px[1] < 60 and px[2] < 60


def _count_red(sheet):
    return sum(1 for px in sheet.getdata() if _is_red(px))


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


# --- Badge geometry helpers (mirror the drawing math in imaging.py) ---


def _plain_badge_probe(i, cols, cell_width, spacing, radius):
    """Red-interior point near the top of cell i's badge (above the glyph)."""
    col = i % cols
    row = i // cols
    x = spacing + col * (cell_width + spacing)
    y = spacing + row * (cell_width + spacing)
    return (x + _BADGE_INSET + radius, y + _BADGE_INSET + 4)


def _labeled_badge_probe(i, cols, cell_width, spacing, radius, title=True):
    """Red-interior point for cell i's badge on a labeled sheet."""
    col = i % cols
    row = i // cols
    title_h = (_LABEL_HEIGHT + spacing) if title else 0
    total_cell_h = cell_width + _LABEL_HEIGHT
    x = spacing + col * (cell_width + spacing)
    y = title_h + spacing + row * (total_cell_h + spacing)
    return (x + _BADGE_INSET + radius, y + _LABEL_HEIGHT + _BADGE_INSET + 4)


# --- Step 1.1: badge helpers ---


def test_draw_badge_red_fill_and_white_glyph():
    canvas = Image.new("RGB", (80, 80), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    _draw_badge(draw, 5, 5, 1, 30)
    # Interior red point near the top of the circle, above the centered glyph
    assert _is_red(canvas.getpixel((5 + 30, 5 + 4)))
    # The number is drawn in white somewhere inside the badge
    whites = [px for px in canvas.getdata() if px[0] > 200 and px[1] > 200 and px[2] > 200]
    assert whites, "number glyph not drawn"


def test_load_badge_font_fallback_on_typeerror(monkeypatch):
    real = ImageFont.load_default

    def fake(*args, **kwargs):
        if "size" in kwargs:
            raise TypeError("size not supported on this Pillow")
        return real()

    monkeypatch.setattr(ImageFont, "load_default", fake)
    font = _load_badge_font(30)
    assert font is not None
    # Badge still renders through the fallback branch without raising
    canvas = Image.new("RGB", (80, 80), (0, 0, 0))
    _draw_badge(ImageDraw.Draw(canvas), 5, 5, 2, 30)
    assert _count_red(canvas) > 0


def test_draw_badge_distinct_numbers_render_differently():
    def glyph_pixels(n):
        canvas = Image.new("RGB", (80, 80), (0, 0, 0))
        _draw_badge(ImageDraw.Draw(canvas), 5, 5, n, 30)
        return frozenset(
            i
            for i, px in enumerate(canvas.getdata())
            if px[0] > 200 and px[1] > 200 and px[2] > 200
        )

    assert glyph_pixels(1) != glyph_pixels(8)


# --- Step 1.2: make_contact_sheet badges ---


def test_contact_sheet_badges_present_and_counted(tmp_path):
    imgs = [_make_test_image(tmp_path / f"i{i}.png", _CELL_COLOR) for i in range(4)]
    out = tmp_path / "sheet.png"
    make_contact_sheet(imgs, out, cols=2, cell_width=80, spacing=10)
    radius = min(imaging._BADGE_RADIUS, (80 - 2 * _BADGE_INSET) // 2)
    with Image.open(out) as sheet:
        for i in range(4):
            px = sheet.getpixel(_plain_badge_probe(i, 2, 80, 10, radius))
            assert _is_red(px), f"cell {i} missing badge at expected location"


def test_contact_sheet_no_badges_when_disabled(tmp_path):
    imgs = [_make_test_image(tmp_path / f"i{i}.png", _CELL_COLOR) for i in range(4)]
    out = tmp_path / "sheet.png"
    make_contact_sheet(imgs, out, cols=2, cell_width=80, spacing=10, badges=False)
    with Image.open(out) as sheet:
        assert _count_red(sheet) == 0


def test_contact_sheet_badge_radius_clamped_to_cell(tmp_path):
    imgs = [_make_test_image(tmp_path / f"i{i}.png", _CELL_COLOR) for i in range(2)]
    out = tmp_path / "sheet.png"
    # Oversized radius on a tiny cell must clamp, render, and not raise
    make_contact_sheet(imgs, out, cols=2, cell_width=20, spacing=4, badge_radius=30)
    with Image.open(out) as sheet:
        assert _count_red(sheet) > 0


def test_contact_sheet_badge_radius_zero_guard(tmp_path):
    imgs = [_make_test_image(tmp_path / f"i{i}.png", _CELL_COLOR) for i in range(2)]
    out = tmp_path / "sheet.png"
    make_contact_sheet(imgs, out, cols=2, cell_width=80, badge_radius=0)
    with Image.open(out) as sheet:
        assert _count_red(sheet) == 0


# --- Step 1.3: make_labeled_contact_sheet badges ---


def test_labeled_sheet_badges_present(tmp_path):
    cells = [
        (_make_test_image(tmp_path / f"i{i}.png", _CELL_COLOR), f"label_{i}")
        for i in range(4)
    ]
    out = tmp_path / "labeled.png"
    make_labeled_contact_sheet(cells, out, cols=2, title="T", cell_width=80, spacing=10)
    radius = min(imaging._BADGE_RADIUS, (80 - 2 * _BADGE_INSET) // 2)
    with Image.open(out) as sheet:
        for i in range(4):
            px = sheet.getpixel(_labeled_badge_probe(i, 2, 80, 10, radius))
            assert _is_red(px), f"cell {i} missing badge on labeled sheet"


def test_labeled_sheet_no_badges_when_disabled(tmp_path):
    cells = [
        (_make_test_image(tmp_path / f"i{i}.png", _CELL_COLOR), f"label_{i}")
        for i in range(4)
    ]
    out = tmp_path / "labeled.png"
    make_labeled_contact_sheet(cells, out, cols=2, title="T", cell_width=80, badges=False)
    with Image.open(out) as sheet:
        assert _count_red(sheet) == 0


def test_labeled_sheet_badge_stays_below_label_strip(tmp_path):
    cells = [
        (_make_test_image(tmp_path / f"i{i}.png", _CELL_COLOR), f"label_{i}")
        for i in range(2)
    ]
    out = tmp_path / "labeled.png"
    make_labeled_contact_sheet(cells, out, cols=2, title="T", cell_width=80, spacing=10)
    # Cell 0's label strip must contain no badge red (badge sits below it)
    y0 = (_LABEL_HEIGHT + 10) + 10
    with Image.open(out) as sheet:
        for yy in range(y0, y0 + _LABEL_HEIGHT):
            for xx in range(10, 90):
                assert not _is_red(sheet.getpixel((xx, yy))), "badge leaked into label strip"
