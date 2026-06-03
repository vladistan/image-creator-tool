"""Listing commands for presets, platforms, and providers.

Provides discovery of available style presets, platform sizing options,
and registered image generation providers.
"""

from __future__ import annotations

from typing import Any

import typer

from image_creator_tool.presets import load_platforms, load_presets
from image_creator_tool.providers import list_providers


def list_presets_cmd() -> None:
    """List available style presets."""
    presets: dict[str, Any] = load_presets()
    if not presets:
        typer.echo("No presets found.")
        return
    for name, conf in presets.items():
        typer.echo(f"  {name:16s} {conf.get('description', '')}")


def list_platforms_cmd() -> None:
    """List available platform sizing presets."""
    platforms: dict[str, Any] = load_platforms()
    if not platforms:
        typer.echo("No platforms found.")
        return
    for name, conf in platforms.items():
        desc = conf.get("description", "")
        size = f"{conf.get('width', '?')}x{conf.get('height', '?')}"
        typer.echo(f"  {name:16s} {size:12s} {desc}")


def list_providers_cmd() -> None:
    """List registered image generation providers."""
    for name in list_providers():
        typer.echo(f"  {name}")
