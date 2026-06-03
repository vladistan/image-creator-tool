"""History browsing command for image-creator-tool.

Displays past generation records from the persistent history log,
with optional project filtering.
"""

from __future__ import annotations

import json

import typer

from image_creator_tool.history import HISTORY_FILE


def history(
    n: int = typer.Option(20, "-n", help="Number of entries to display"),
    project: str | None = typer.Option(None, "--project", help="Filter by project name"),
) -> None:
    """Show generation history."""
    if not HISTORY_FILE.exists():
        typer.echo("No history yet.")
        return

    with HISTORY_FILE.open() as f:
        lines = f.readlines()
    entries = [json.loads(ln) for ln in lines]

    if project:
        entries = [e for e in entries if e.get("project") == project]
    entries = entries[-n:]

    for e in entries:
        ts = e.get("timestamp", "?")[:19].replace("T", " ")
        preset = e.get("preset") or "-"
        subject = e.get("subject", "")[:60]
        typer.echo(f"{ts}  {preset:14s}  {subject}")
        typer.echo(f"              → {e.get('output_path', '?')}")
