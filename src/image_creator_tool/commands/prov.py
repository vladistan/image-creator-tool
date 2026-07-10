"""Typer command group for inspecting image provenance records.

Subcommands:
    list    Scan the output directory for `.prov.json` sidecars (filterable).
    show    Print the full provenance record for a specific image.
    export  Merge all provenance under a directory and export as PROV-N.

Records are produced automatically during generation (see `provenance.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import sentry_sdk
import typer
from rich.console import Console
from rich.table import Table

from image_creator_tool import provenance
from image_creator_tool.config import load_settings
from image_creator_tool.errors import PermanentAPIError

prov_app = typer.Typer(
    help="Inspect W3C PROV provenance records for generated images.",
    no_args_is_help=True,
)


def _resolve_output_dir(output_dir: Path | None) -> Path:
    """Fall back to the configured output directory when none is given."""
    return output_dir or load_settings().output_dir


@prov_app.command("list")
def list_cmd(
    date: Annotated[
        str | None, typer.Option("--date", help="Filter to a generation date (YYYY-MM-DD)")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Filter to a specific model")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Filter to a specific provider")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", help="Directory to scan (default: config)")
    ] = None,
) -> None:
    """List provenance records in the output directory, with optional filters."""
    with sentry_sdk.start_transaction(op="prov.list", name="prov_list") as txn:
        directory = _resolve_output_dir(output_dir)
        txn.set_tag("output_dir", str(directory))
        txn.set_data("filters", {"date": date, "model": model, "provider": provider})
        with sentry_sdk.start_span(op="prov.scan", description="scan_provenance"):
            records = provenance.scan_provenance(
                directory, date=date, model=model, provider=provider
            )
        txn.set_data("record_count", len(records))

    if not records:
        typer.echo(f"No provenance records found in {directory}.")
        return

    console = Console()
    table = Table(title=f"Provenance ({len(records)} records)")
    table.add_column("Image", style="cyan", no_wrap=True)
    table.add_column("Provider", style="green")
    table.add_column("Model", style="magenta")
    table.add_column("Timestamp", style="yellow")
    table.add_column("Prompt", style="white")
    for _sidecar, record in records:
        table.add_row(
            Path(record.output_path).name,
            record.provider,
            record.model,
            record.timestamp,
            _truncate(record.prompt),
        )
    console.print(table)


@prov_app.command("show")
def show_cmd(
    image_path: Annotated[
        str, typer.Argument(help="Image (or .prov.json sidecar) to describe")
    ],
) -> None:
    """Print the full provenance record for a specific image."""
    with sentry_sdk.start_transaction(op="prov.show", name="prov_show") as txn:
        target = Path(image_path)
        sidecar = target if target.suffix == ".json" else provenance.sidecar_path_for(target)
        txn.set_tag("sidecar", str(sidecar))
        with sentry_sdk.start_span(op="prov.load", description="load_record"):
            record = provenance.load_record(sidecar)

    typer.echo(f"Image:     {record.output_path}")
    typer.echo(f"Provider:  {record.provider}")
    typer.echo(f"Model:     {record.model}")
    typer.echo(f"Timestamp: {record.timestamp}")
    typer.echo(f"Seed:      {record.seed if record.seed is not None else '(none)'}")
    typer.echo(f"Prompt:    {record.prompt}")
    if record.parameters:
        typer.echo("Parameters:")
        for key, value in sorted(record.parameters.items()):
            typer.echo(f"  {key}: {value}")
    typer.echo("\nPROV-N:")
    typer.echo(record.to_prov_n())


@prov_app.command("export")
def export_cmd(
    output_dir: Annotated[
        Path, typer.Argument(help="Directory of provenance sidecars to export")
    ],
    format: Annotated[
        str, typer.Option("--format", help="Export format (prov-n)")
    ] = "prov-n",
) -> None:
    """Merge all provenance under a directory and export it as PROV-N."""
    if format != "prov-n":
        raise PermanentAPIError(f"Unsupported export format '{format}'. Supported: prov-n")
    with sentry_sdk.start_transaction(op="prov.export", name="prov_export") as txn:
        txn.set_tag("output_dir", str(output_dir))
        txn.set_tag("format", format)
        with sentry_sdk.start_span(op="prov.export", description="export_prov_n"):
            document = provenance.export_prov_n(output_dir)
    typer.echo(document)


def _truncate(text: str, width: int = 48) -> str:
    """Shorten a prompt for single-line table display."""
    return text if len(text) <= width else text[: width - 1] + "…"
