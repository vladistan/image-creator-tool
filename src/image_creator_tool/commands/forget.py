"""Typer command to forget imported reference images and prune discovery leftovers.

``image-creator-tool forget @X @Y ...`` removes imported images (copy, PROV sidecar, and
index entry) — the reclaim side of ``import``. ``--prune`` instead drops every imported
image no retained generation still references, and ``--dry-run`` previews that set without
deleting. Only imported images are ever touched; a generated @index is refused, not removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import sentry_sdk
import typer

from image_creator_tool import refimages
from image_creator_tool.config import load_settings, reference_images_dir
from image_creator_tool.errors import PermanentAPIError


def forget(
    indices: Annotated[
        list[str] | None, typer.Argument(help="Imported image indices to forget, e.g. @X7TYJYD3")
    ] = None,
    prune: Annotated[
        bool, typer.Option("--prune", help="Forget imported images no generation references")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="With --prune, preview the candidates without deleting"),
    ] = False,
    output_dir: Annotated[
        str | None, typer.Option("--output-dir", help="Directory to scan for indices")
    ] = None,
) -> None:
    """Forget imported reference images by @index, or prune unreferenced imports."""
    settings = load_settings()
    # Mirror import's scoping so prune scans the same store the images were written to.
    if output_dir is not None:
        search_dir = output_dir
        ref_dir = str(Path(output_dir) / "ref-images")
    else:
        search_dir = str(settings.output_dir)
        ref_dir = str(reference_images_dir(settings))

    with sentry_sdk.start_transaction(op="forget", name="forget") as txn:
        txn.set_tag("mode", "prune" if prune else "explicit")
        if prune:
            _prune(ref_dir, search_dir, dry_run=dry_run)
            return
        if not indices:
            raise PermanentAPIError("Provide an @index to forget, or use --prune.")
        _forget_explicit(indices, search_dir)


def _forget_explicit(indices: list[str], search_dir: str) -> None:
    """Forget each explicitly listed @index, reporting removals and refusals."""
    for raw in indices:
        with sentry_sdk.start_span(op="forget.index", description="forget_image"):
            result = refimages.forget_image(raw, search_dir)
        if result.removed:
            typer.echo(f"Forgot @{result.index}: {result.image_path}")
        else:
            typer.echo(
                f"Skipped @{result.index}: not an imported image (only imports can be forgotten)"
            )


def _prune(ref_dir: str, search_dir: str, *, dry_run: bool) -> None:
    """Drop (or, with dry_run, preview) imported images no generation references."""
    candidates = refimages.collect_unreferenced(ref_dir, search_dir)
    if not candidates:
        typer.echo("No unreferenced imported images to prune.")
        return
    for index, image_path in candidates:
        if dry_run:
            typer.echo(f"Would prune @{index}: {image_path}")
        else:
            refimages.forget_image(index, search_dir)
            typer.echo(f"Pruned @{index}: {image_path}")
    verb = "would be pruned" if dry_run else "pruned"
    typer.echo(f"{len(candidates)} imported image(s) {verb}.")
