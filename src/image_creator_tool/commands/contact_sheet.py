"""Typer command to assemble a contact sheet from already-generated images.

Unlike the sweep engine (which builds a sheet as a byproduct of generation), this
command composes a labeled contact sheet from images that already exist, referenced
by their short `@index` — so results from separate runs can be combined without
regenerating anything. Indices and PROV records come from Phases 5 and 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import sentry_sdk
import typer

from image_creator_tool import indexer, provenance
from image_creator_tool.config import load_settings
from image_creator_tool.errors import PermanentAPIError
from image_creator_tool.imaging import make_labeled_contact_sheet

_PROMPT_LABEL_MAX = 40


def _label_for(index: str, image_path: Path, mode: str) -> str:
    """Build a cell label from the index and, for model/prompt modes, its PROV record."""
    if mode == "none":
        return ""
    if mode == "index":
        return f"@{index}"

    sidecar = provenance.sidecar_path_for(image_path)
    record = provenance.load_record(sidecar) if sidecar.is_file() else None
    if mode == "model":
        model = record.model if record else "?"
        return f"@{index} {model}"
    if mode == "prompt":
        text = (record.subject or record.prompt) if record else ""
        if len(text) > _PROMPT_LABEL_MAX:
            text = text[: _PROMPT_LABEL_MAX - 1] + "…"
        return f"@{index} {text}".rstrip()
    return f"@{index}"


def _looks_like_path(arg: str) -> bool:
    """Decide whether a positional arg is a raw file path rather than an @index code.

    An ``@``-prefix is always an index (handled by the caller). Otherwise the arg is a raw
    path when it names an existing file, carries a path separator, or has a file extension;
    a bare token with none of these falls through to index resolution so plain index codes
    keep working.
    """
    if Path(arg).is_file():
        return True
    if "/" in arg or "\\" in arg:
        return True
    return bool(Path(arg).suffix)


def _resolve_cell(arg: str, search_dir: Path, label_mode: str) -> tuple[Path, str]:
    """Resolve one positional arg to an ``(image_path, label)`` contact-sheet cell.

    Raw paths are ephemeral — no import, no index minted, no store write — and are labeled
    with the bare filename minus its extension. @index args resolve through the store and
    keep their configured label mode.

    Raises:
        PermanentAPIError: A raw path does not exist, or an @index cannot be resolved.
    """
    if not arg.startswith("@") and _looks_like_path(arg):
        path = Path(arg)
        if not path.is_file():
            raise PermanentAPIError(f"Contact-sheet source not found: {path}")
        return path, path.stem

    image_path = indexer.resolve_index(arg, search_dir)
    if not image_path.exists():
        norm = arg.lstrip("@").upper()
        raise PermanentAPIError(
            f"Image for @{norm} is indexed but missing on disk: {image_path}"
        )
    return image_path, _label_for(arg.lstrip("@").upper(), image_path, label_mode)


def contact_sheet(
    indices: Annotated[
        list[str],
        typer.Argument(
            help="Image indices (@X7TYJYD3) and/or raw file paths to include; mixed lists allowed"
        ),
    ],
    output: Annotated[
        Path | None, typer.Argument(help="Output path (default: ./contact-sheet.png)")
    ] = None,
    cols: Annotated[int, typer.Option("--cols", help="Grid columns (0 = auto)")] = 0,
    title: Annotated[str | None, typer.Option("--title", help="Sheet title")] = None,
    label: Annotated[
        str, typer.Option("--label", help="Cell label: model | index | prompt | none")
    ] = "model",
    no_badges: Annotated[
        bool, typer.Option("--no-badges", help="Disable numbered badges")
    ] = False,
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", help="Directory to scan for indices")
    ] = None,
) -> None:
    """Assemble a labeled contact sheet from already-generated images by their @index."""
    search_dir = Path(output_dir) if output_dir is not None else load_settings().output_dir

    with sentry_sdk.start_transaction(op="contact_sheet", name="contact_sheet") as txn:
        txn.set_tag("index_count", len(indices))
        txn.set_tag("label_mode", label)
        cells: list[tuple[Path, str]] = []
        for raw in indices:
            with sentry_sdk.start_span(op="cell.resolve", description="resolve_cell"):
                cells.append(_resolve_cell(raw, search_dir, label))

        out_path = output or Path("contact-sheet.png")
        with sentry_sdk.start_span(op="sheet.render", description="make_labeled_contact_sheet"):
            result = make_labeled_contact_sheet(
                cells, out_path, cols=cols, title=title or "", badges=not no_badges
            )

    typer.echo(f"Contact sheet: {result} ({len(cells)} images)")
