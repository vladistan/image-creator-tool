"""Regenerate last run command for image-creator-tool.

Replays the most recent generation with the same parameters,
optionally overriding the variant count.
"""

from __future__ import annotations

from types import SimpleNamespace

import sentry_sdk
import typer

from image_creator_tool.config import get_config_dict
from image_creator_tool.generation import generate
from image_creator_tool.history import load_last_run
from image_creator_tool.presets import load_platforms, load_presets
from image_creator_tool.providers import get_provider


def again(
    n: int = typer.Option(0, "--n", help="Override variant count (0 = use last run's value)"),
) -> None:
    """Regenerate the last image with the same parameters."""
    last = load_last_run()
    if not last:
        typer.echo("Error: no previous run found. Generate an image first.", err=True)
        raise typer.Exit(code=1)

    provider_name = last.get("provider", "gemini")
    provider = get_provider(provider_name)

    args = SimpleNamespace(
        prompt=last["subject"],
        output=None,
        preset=last.get("preset"),
        platform=last.get("platform"),
        model=last.get("model"),
        edit=last.get("edit_source"),
        reference=last.get("reference", []),
        project=last.get("project"),
        n=n if n else last.get("n", 1),
        no_metadata=False,
        dry_run=False,
        presets=load_presets(),
        platforms=load_platforms(),
    )
    typer.echo(f"↻ Regenerating: {last['subject']}")
    with sentry_sdk.start_transaction(op="cli", name="again"):
        generate(args, get_config_dict(), provider)
