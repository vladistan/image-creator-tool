"""Typer command for resolving short image indices to paths and metadata.

Usage:
    image-creator lookup INDEX     Show the path and provenance for one index.
    image-creator lookup --list    List recent indices with their images.

Indices are minted during generation (see `indexer.py`) and joined here with the
Phase 4 PROV records for a rich, human-readable view.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import sentry_sdk
import typer
from rich.console import Console
from rich.table import Table

from image_creator_tool import indexer, provenance
from image_creator_tool.config import load_settings
from image_creator_tool.errors import PermanentAPIError

if TYPE_CHECKING:
    from image_creator_tool.provenance import ProvenanceRecord


def _load_record(image_path: Path) -> ProvenanceRecord | None:
    """Load the PROV record for an image, or None when it has no sidecar.

    A missing sidecar is an expected state (images generated with `--no-metadata`
    still get an index), so it maps to None. A sidecar that exists but is corrupt
    is a real fault and is left to propagate.
    """
    sidecar = provenance.sidecar_path_for(image_path)
    if not sidecar.is_file():
        return None
    return provenance.load_record(sidecar)


def lookup(
    index: Annotated[
        str | None, typer.Argument(help="Short image index to resolve (e.g. IISDSXS3)")
    ] = None,
    list_all: Annotated[
        bool, typer.Option("--list", help="List recent indices with their images")
    ] = False,
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", help="Directory to scan (default: config)")
    ] = None,
) -> None:
    """Resolve a short image index to its path and metadata."""
    # Path(...) at runtime (not just in annotations) so the pathlib import stays
    # available for typer's option parsing.
    search_dir = Path(output_dir) if output_dir is not None else load_settings().output_dir

    with sentry_sdk.start_transaction(op="lookup", name="lookup") as txn:
        txn.set_tag("output_dir", str(search_dir))
        txn.set_tag("mode", "list" if list_all else "single")
        if list_all:
            txn.set_data("index_count", _list_indices(search_dir))
            return
        if not index:
            raise PermanentAPIError("Provide an INDEX to look up, or use --list.")
        _show_index(index, search_dir)


def _show_index(index: str, search_dir: Path) -> None:
    """Print the image path and provenance metadata for a single index."""
    with sentry_sdk.start_span(op="index.resolve", description="resolve_index"):
        image_path = indexer.resolve_index(index, search_dir)
    typer.echo(f"Index:     {index.lstrip('@').upper()}")
    typer.echo(f"Image:     {image_path}")
    typer.echo(f"Exists:    {'yes' if image_path.exists() else 'no (file moved or deleted)'}")

    with sentry_sdk.start_span(op="prov.load", description="load_record"):
        record = _load_record(image_path)
    if record is None:
        typer.echo("Metadata:  (no provenance record found)")
        return
    typer.echo(f"Provider:  {record.provider}")
    typer.echo(f"Model:     {record.model}")
    typer.echo(f"Timestamp: {record.timestamp}")
    typer.echo(f"Seed:      {record.seed if record.seed is not None else '(none)'}")
    typer.echo(f"Prompt:    {record.prompt}")


def _list_indices(search_dir: Path) -> int:
    """Print a table of recent indices with their images; return the entry count."""
    with sentry_sdk.start_span(op="index.scan", description="list_index_entries"):
        entries = indexer.list_index_entries(search_dir)
    if not entries:
        typer.echo(f"No image indices found in {search_dir}.")
        return 0

    console = Console()
    table = Table(title=f"Image indices ({len(entries)})")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Image", style="green", no_wrap=True)
    table.add_column("Prompt", style="white")
    for index, image_path in entries:
        record = _load_record(image_path)
        table.add_row(index, image_path.name, _truncate(record.prompt) if record else "")
    console.print(table)
    return len(entries)


def _truncate(text: str, width: int = 48) -> str:
    """Shorten a prompt for single-line table display."""
    return text if len(text) <= width else text[: width - 1] + "…"
