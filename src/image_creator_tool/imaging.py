"""Image post-processing operations using Pillow.

Provides platform-specific resizing/cropping and contact sheet generation
for multi-variant outputs. SVG files rendered via CairoSVG.
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import TYPE_CHECKING, Any

import cairosvg
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Resampling

if TYPE_CHECKING:
    from pathlib import Path


def _open_image(path: Path) -> Image.Image:
    """Open an image file, rendering SVG to PNG via CairoSVG if needed."""
    if path.suffix.lower() == ".svg":
        png_bytes = cairosvg.svg2png(url=str(path), output_width=1024)
        return Image.open(BytesIO(png_bytes))
    return Image.open(path)


SUPPORTED_OUTPUT_FORMATS = ("png", "webp", "jpg")
_PIL_SAVE_FORMATS = {"png": "PNG", "webp": "WEBP", "jpg": "JPEG", "jpeg": "JPEG"}


def normalize_image_bytes(image_bytes: bytes, target_format: str) -> bytes:
    """Re-encode raster ``image_bytes`` to ``target_format`` (png/webp/jpg) via Pillow.

    Guarantees the persisted bytes match the requested format regardless of what
    the provider returned, so downstream metadata/EXIF embedding always has a known,
    embeddable format to write into. JPEG carries no alpha channel, so images with
    transparency are flattened onto a white background. Raises ``ValueError`` for an
    unknown target format so callers surface an honest error rather than mis-saving.
    """
    pil_format = _PIL_SAVE_FORMATS.get(target_format.lower())
    if pil_format is None:
        raise ValueError(f"unsupported output format: {target_format}")
    img: Image.Image = Image.open(BytesIO(image_bytes))
    if pil_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flattened = Image.new("RGB", img.size, (255, 255, 255))
        flattened.paste(img, mask=img.split()[-1])
        img = flattened
    buffer = BytesIO()
    img.save(buffer, format=pil_format)
    return buffer.getvalue()

_CONTACT_BG_COLOR = (15, 15, 15)  # #0f0f0f
_CONTACT_CELL_WIDTH = 600
_CONTACT_SPACING = 10

# Numbered-badge styling for contact-sheet cells (T86).
_BADGE_RADIUS = 30
_BADGE_FILL = (221, 17, 17)  # #d11
_BADGE_TEXT = (255, 255, 255)
_BADGE_INSET = 6


def _load_badge_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return PIL's built-in default font at ``size``.

    Falls back to the no-argument default font on older Pillow builds where
    ``load_default`` does not accept a ``size`` keyword.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _center_badge_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw ``text`` centered within ``box`` in the badge text colour."""
    x0, y0, x1, y1 = box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top
    tx = x0 + (x1 - x0 - text_w) / 2 - left
    ty = y0 + (y1 - y0 - text_h) / 2 - top
    draw.text((tx, ty), text, fill=_BADGE_TEXT, font=font)


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    number: int,
    radius: int,
) -> None:
    """Draw a filled red circle with a centered white ``number``.

    The circle spans ``(x, y)`` to ``(x + 2*radius, y + 2*radius)``.
    """
    box = (x, y, x + 2 * radius, y + 2 * radius)
    draw.ellipse(box, fill=_BADGE_FILL)
    font = _load_badge_font(radius)
    _center_badge_text(draw, box, str(number), font)


def apply_platform_fit(
    image_path: Path,
    platform: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Resize and crop image to platform target dimensions.

    Uses center-gravity cropping to fill the target aspect ratio without
    distortion or letterboxing (equivalent to ImageMagick's -resize WxH^ -extent WxH).
    """
    width = platform["width"]
    height = platform["height"]
    dest = output_path or image_path

    with Image.open(image_path) as img:
        # Calculate scale to fill target (cover, not contain)
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)

        # Resize to fill
        resized = img.resize((new_w, new_h), Resampling.LANCZOS)

        # Center crop to exact target
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        cropped = resized.crop((left, top, left + width, top + height))

        cropped.save(dest)

    return dest


def make_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    cols: int = 2,
    cell_width: int = _CONTACT_CELL_WIDTH,
    spacing: int = _CONTACT_SPACING,
    bg_color: tuple[int, int, int] = _CONTACT_BG_COLOR,
    badges: bool = True,
    badge_radius: int = _BADGE_RADIUS,
) -> Path:
    """Assemble a contact sheet montage from multiple images.

    Creates a grid layout with dark background, useful for comparing
    variants side-by-side.

    Args:
        image_paths: Images to arrange in the grid.
        output_path: Where to save the resulting sheet.
        cols: Number of columns in the grid.
        cell_width: Width in pixels of each cell (images are scaled to fit).
        spacing: Pixel gap between cells and around the border.
        bg_color: RGB background colour tuple.
        badges: Draw a red numbered badge (1..N) on each cell.
        badge_radius: Badge circle radius in pixels (clamped to fit the cell).
    """
    if not image_paths:
        raise ValueError("No images for contact sheet")

    # Load and resize all images to uniform cell width
    cells: list[Image.Image] = []
    cell_height = 0
    for path in image_paths:
        img = _open_image(path)
        # Scale to cell width, maintaining aspect ratio
        scale = cell_width / img.width
        new_h = int(img.height * scale)
        cell = img.resize((cell_width, new_h), Resampling.LANCZOS)
        cells.append(cell)
        cell_height = max(cell_height, new_h)

    # Calculate grid dimensions
    rows = (len(cells) + cols - 1) // cols
    grid_w = cols * cell_width + (cols + 1) * spacing
    grid_h = rows * cell_height + (rows + 1) * spacing

    # Compose grid
    sheet = Image.new("RGB", (grid_w, grid_h), bg_color)
    draw = ImageDraw.Draw(sheet)
    badge_r = min(badge_radius, (cell_width - 2 * _BADGE_INSET) // 2)
    for i, cell in enumerate(cells):
        col = i % cols
        row = i // cols
        x = spacing + col * (cell_width + spacing)
        y = spacing + row * (cell_height + spacing)
        sheet.paste(cell, (x, y))
        if badges and badge_r > 0:
            _draw_badge(draw, x + _BADGE_INSET, y + _BADGE_INSET, i + 1, badge_r)

    sheet.save(output_path)

    # Clean up opened images
    for cell in cells:
        cell.close()

    return output_path


_LABEL_HEIGHT = 24
_LABEL_COLOR = (255, 255, 255)
_LABEL_BG = (30, 30, 30)

# Panel-strip styling: bold frames + gutters distinguish an assembled comic strip
# from a contact sheet (which carries numbered badges instead).
_STRIP_BG_COLOR = (255, 255, 255)
_STRIP_BORDER_COLOR = (0, 0, 0)
_STRIP_GUTTER = 16
_STRIP_BORDER = 6
_STRIP_CAPTION_FONT_SIZE = 16


def make_labeled_contact_sheet(  # noqa: PLR0913 — signature fixed by design (labeled + badge params)
    cells: list[tuple[Path, str]],
    output_path: Path,
    cols: int = 0,
    title: str = "",
    cell_width: int = _CONTACT_CELL_WIDTH,
    spacing: int = _CONTACT_SPACING,
    bg_color: tuple[int, int, int] = _CONTACT_BG_COLOR,
    badges: bool = True,
    badge_radius: int = _BADGE_RADIUS,
) -> Path:
    """Assemble a labeled contact sheet for multi-model comparison.

    Each cell is an (image_path, label) tuple where label identifies
    the provider/model used. Labels are rendered above each image cell.

    Args:
        cells: List of (image_path, label_text) tuples.
        output_path: Where to save the contact sheet.
        cols: Grid columns. 0 = auto (square root of cell count).
        cell_width: Width in pixels of each cell.
        spacing: Pixel gap between cells and around the border.
        bg_color: RGB background colour tuple.
        badges: Draw a red numbered badge (1..N) on each cell's thumbnail.
        badge_radius: Badge circle radius in pixels (clamped to fit the cell).
    """

    if not cells:
        raise ValueError("No cells for labeled contact sheet")

    if cols <= 0:
        cols = max(1, math.ceil(math.sqrt(len(cells))))

    # Load and resize all images
    images: list[tuple[Image.Image, str]] = []
    cell_height = 0
    for path, label in cells:
        img = _open_image(path)
        scale = cell_width / img.width
        new_h = int(img.height * scale)
        cell = img.resize((cell_width, new_h), Resampling.LANCZOS)
        images.append((cell, label))
        cell_height = max(cell_height, new_h)

    # Grid dimensions (with label height added per row, plus title if present)
    rows = (len(images) + cols - 1) // cols
    total_cell_h = cell_height + _LABEL_HEIGHT
    title_h = _LABEL_HEIGHT + spacing if title else 0
    grid_w = cols * cell_width + (cols + 1) * spacing
    grid_h = rows * total_cell_h + (rows + 1) * spacing + title_h

    # Compose grid
    sheet = Image.new("RGB", (grid_w, grid_h), bg_color)
    draw = ImageDraw.Draw(sheet)

    # Draw title at top if provided
    if title:
        draw.rectangle([0, 0, grid_w, _LABEL_HEIGHT], fill=_LABEL_BG)
        # Truncate title if too long for grid width
        _max_title_len = 100
        display_title = title if len(title) < _max_title_len else title[:97] + "..."
        draw.text((_CONTACT_SPACING, 4), f"Prompt: {display_title}", fill=_LABEL_COLOR)

    badge_r = min(badge_radius, (cell_width - 2 * _BADGE_INSET) // 2)
    for i, (cell, label) in enumerate(images):
        col = i % cols
        row = i // cols
        x = spacing + col * (cell_width + spacing)
        y = title_h + spacing + row * (total_cell_h + spacing)

        # Draw label background + text
        draw.rectangle([x, y, x + cell_width, y + _LABEL_HEIGHT], fill=_LABEL_BG)
        draw.text((x + 4, y + 4), label, fill=_LABEL_COLOR)

        # Paste image below label
        sheet.paste(cell, (x, y + _LABEL_HEIGHT))

        # Draw numbered badge on the thumbnail region (below the label strip)
        if badges and badge_r > 0:
            _draw_badge(
                draw, x + _BADGE_INSET, y + _LABEL_HEIGHT + _BADGE_INSET, i + 1, badge_r
            )

    sheet.save(output_path)

    for cell, _ in images:
        cell.close()

    return output_path


def make_panel_strip(  # noqa: PLR0913 — layout/border/caption knobs are the command's surface
    panels: list[tuple[Path, str]],
    output_path: Path,
    rows: int = 1,
    cols: int = 0,
    gutter: int = _STRIP_GUTTER,
    border: int = _STRIP_BORDER,
    bg_color: tuple[int, int, int] = _STRIP_BG_COLOR,
    border_color: tuple[int, int, int] = _STRIP_BORDER_COLOR,
    cell_width: int = _CONTACT_CELL_WIDTH,
    captions: bool = False,
) -> Path:
    """Compose @index panels into a bordered comic strip.

    Each panel is framed with a bold ``border`` and separated by ``gutter`` on a solid
    background — deliberately distinct from a contact sheet (no numbered badges). Panels lay
    out left-to-right across ``cols`` columns; ``cols=0`` derives the column count from
    ``rows`` so the default (``rows=1``) yields a single horizontal strip. When ``captions``
    is set, each panel's caption text is rendered in a bar beneath its image.

    Args:
        panels: ``(image_path, caption)`` tuples; caption is ignored unless ``captions`` is set.
        output_path: Where to save the assembled strip.
        rows: Row count used to derive columns when ``cols`` is 0.
        cols: Explicit column count; 0 = derive from ``rows``.
        gutter: Pixel gap between panels and around the border.
        border: Panel frame thickness in pixels.
        bg_color: RGB background colour behind the panels.
        border_color: RGB colour of each panel's frame.
        cell_width: Width in pixels each panel image is scaled to.
        captions: Render a caption bar beneath each panel.
    """
    if not panels:
        raise ValueError("No panels for strip")

    count = len(panels)
    if cols <= 0:
        cols = math.ceil(count / rows) if rows > 0 else count
    grid_rows = math.ceil(count / cols)

    loaded: list[tuple[Image.Image, str]] = []
    cell_height = 0
    for path, caption in panels:
        img = _open_image(path)
        scale = cell_width / img.width
        new_h = int(img.height * scale)
        cell = img.resize((cell_width, new_h), Resampling.LANCZOS)
        loaded.append((cell, caption))
        cell_height = max(cell_height, new_h)

    caption_h = _LABEL_HEIGHT if captions else 0
    tile_w = cell_width + 2 * border
    tile_h = cell_height + caption_h + 2 * border
    grid_w = cols * tile_w + (cols + 1) * gutter
    grid_h = grid_rows * tile_h + (grid_rows + 1) * gutter

    sheet = Image.new("RGB", (grid_w, grid_h), bg_color)
    draw = ImageDraw.Draw(sheet)
    font = _load_badge_font(_STRIP_CAPTION_FONT_SIZE)
    for i, (cell, caption) in enumerate(loaded):
        col = i % cols
        row = i // cols
        x = gutter + col * (tile_w + gutter)
        y = gutter + row * (tile_h + gutter)
        draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], fill=border_color)
        sheet.paste(cell, (x + border, y + border))
        if captions:
            cap_y = y + border + cell_height
            draw.rectangle(
                [x + border, cap_y, x + border + cell_width, cap_y + caption_h], fill=_LABEL_BG
            )
            draw.text((x + border + 4, cap_y + 4), caption, fill=_LABEL_COLOR, font=font)

    sheet.save(output_path)

    for cell, _ in loaded:
        cell.close()

    return output_path
