"""Typer command to assemble @index panels into a comic strip.

``image-creator-tool strip @p1 @p2 @p3 out.png`` resolves each @index (imported or generated)
to a panel and composes a single bordered strip — the assembly step of the image-to-comic
pipeline. Unlike ``contact-sheet`` (a labeled selection grid with numbered badges), a strip is
the finished artifact: bold panel borders, gutters, and optional per-panel caption bars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import sentry_sdk
import typer

from image_creator_tool import indexer, provenance
from image_creator_tool.config import load_settings
from image_creator_tool.errors import PermanentAPIError
from image_creator_tool.imaging import make_panel_strip

_CAPTION_MAX = 48


def _caption_for(index: str, image_path: Path) -> str:
    """Build a panel caption from its PROV subject/prompt, falling back to the @index."""
    sidecar = provenance.sidecar_path_for(image_path)
    record = provenance.load_record(sidecar) if sidecar.is_file() else None
    text = (record.subject or record.prompt) if record else ""
    if not text:
        return f"@{index}"
    if len(text) > _CAPTION_MAX:
        text = text[: _CAPTION_MAX - 1] + "…"
    return text


def strip(
    panels: Annotated[
        list[str], typer.Argument(help="Panel indices in order, e.g. @A @B @C")
    ],
    output: Annotated[
        Path | None, typer.Argument(help="Output path (default: ./strip.png)")
    ] = None,
    rows: Annotated[
        int, typer.Option("--rows", help="Row count (columns derived when --cols=0)")
    ] = 1,
    cols: Annotated[int, typer.Option("--cols", help="Column count (0 = derive from --rows)")] = 0,
    gutter: Annotated[int, typer.Option("--gutter", help="Gap between panels in pixels")] = 16,
    border: Annotated[int, typer.Option("--border", help="Panel frame thickness in pixels")] = 6,
    caption: Annotated[
        bool, typer.Option("--caption", help="Render a caption bar beneath each panel")
    ] = False,
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", help="Directory to scan for indices")
    ] = None,
) -> None:
    """Assemble @index panels into a single bordered comic strip."""
    search_dir = Path(output_dir) if output_dir is not None else load_settings().output_dir

    with sentry_sdk.start_transaction(op="strip", name="strip") as txn:
        txn.set_tag("panel_count", len(panels))
        cells: list[tuple[Path, str]] = []
        for raw in panels:
            with sentry_sdk.start_span(op="index.resolve", description="resolve_index"):
                image_path = indexer.resolve_index(raw, search_dir)
            if not image_path.exists():
                norm = raw.lstrip("@").upper()
                raise PermanentAPIError(
                    f"Panel @{norm} is indexed but missing on disk: {image_path}"
                )
            label = _caption_for(raw.lstrip("@").upper(), image_path) if caption else ""
            cells.append((image_path, label))

        out_path = output or Path("strip.png")
        with sentry_sdk.start_span(op="strip.render", description="make_panel_strip"):
            result = make_panel_strip(
                cells,
                out_path,
                rows=rows,
                cols=cols,
                gutter=gutter,
                border=border,
                captions=caption,
            )

    typer.echo(f"Strip: {result} ({len(cells)} panels)")
