"""Image post-processing operations using Pillow.

Provides platform-specific resizing/cropping and contact sheet generation
for multi-variant outputs. SVG files rendered via CairoSVG.
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import TYPE_CHECKING, Any

import cairosvg
from PIL import Image, ImageDraw
from PIL.Image import Resampling

if TYPE_CHECKING:
    from pathlib import Path


def _open_image(path: Path) -> Image.Image:
    """Open an image file, rendering SVG to PNG via CairoSVG if needed."""
    if path.suffix.lower() == ".svg":
        png_bytes = cairosvg.svg2png(url=str(path), output_width=1024)
        return Image.open(BytesIO(png_bytes))
    return Image.open(path)

_CONTACT_BG_COLOR = (15, 15, 15)  # #0f0f0f
_CONTACT_CELL_WIDTH = 600
_CONTACT_SPACING = 10


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
    for i, cell in enumerate(cells):
        col = i % cols
        row = i // cols
        x = spacing + col * (cell_width + spacing)
        y = spacing + row * (cell_height + spacing)
        sheet.paste(cell, (x, y))

    sheet.save(output_path)

    # Clean up opened images
    for cell in cells:
        cell.close()

    return output_path


_LABEL_HEIGHT = 24
_LABEL_COLOR = (255, 255, 255)
_LABEL_BG = (30, 30, 30)


def make_labeled_contact_sheet(
    cells: list[tuple[Path, str]],
    output_path: Path,
    cols: int = 0,
    title: str = "",
    cell_width: int = _CONTACT_CELL_WIDTH,
    spacing: int = _CONTACT_SPACING,
    bg_color: tuple[int, int, int] = _CONTACT_BG_COLOR,
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

    sheet.save(output_path)

    for cell, _ in images:
        cell.close()

    return output_path
