"""Typer command to import external images onto the uniform @index path.

``image-creator-tool import <file...>`` copies each file into the reference-image store,
mints an @index for it, and records imported provenance (origin + source) so the file can
be used anywhere an @index is accepted — ``contact-sheet``, ``prov show``, ``strip``, and
``generate --ref``. Re-importing identical bytes is idempotent (see ``refimages.py``).

The module is named ``import_images`` because ``import`` is a reserved keyword; the command
itself is still registered as ``import`` in ``cli.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import sentry_sdk
import typer

from image_creator_tool import refimages
from image_creator_tool.config import load_settings, reference_images_dir


def import_images(
    files: Annotated[
        list[str], typer.Argument(help="Image files to import into the reference store")
    ],
    output_dir: Annotated[
        str | None, typer.Option("--output-dir", help="Directory to scan for indices")
    ] = None,
) -> None:
    """Import external images into the reference store and print an @index for each."""
    settings = load_settings()
    # Keep the reference store inside the search scope: an explicit --output-dir gets its own
    # <dir>/ref-images so the freshly minted @index resolves there; otherwise use the config.
    if output_dir is not None:
        search_dir = output_dir
        ref_dir = Path(output_dir) / "ref-images"
    else:
        search_dir = str(settings.output_dir)
        ref_dir = reference_images_dir(settings)

    with sentry_sdk.start_transaction(op="import", name="import") as txn:
        txn.set_tag("file_count", len(files))
        for source in files:
            with sentry_sdk.start_span(op="import.file", description="import_image"):
                result = refimages.import_image(source, ref_dir, search_dir)
            suffix = " (existing)" if result.is_duplicate else ""
            typer.echo(f"@{result.index}  {source}{suffix}")
