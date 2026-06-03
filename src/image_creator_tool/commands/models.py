"""List all available models with providers and capabilities."""

from __future__ import annotations

import json as json_lib
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from image_creator_tool.providers import REGISTRY
from image_creator_tool.providers.bedrock import _EDIT_OP_MODELS
from image_creator_tool.providers.deepinfra import DeepInfraProvider


def _build_model_catalog() -> list[dict[str, Any]]:
    """Build a catalog of all models with provider and capability info."""
    catalog: list[dict[str, Any]] = []

    for provider_name, provider_cls in REGISTRY.items():
        for alias, full_id in provider_cls.MODELS.items():
            caps = _get_capabilities(provider_name, full_id, alias)
            catalog.append({
                "alias": alias,
                "model_id": full_id,
                "provider": provider_name,
                "capabilities": caps,
            })

    # Add Bedrock edit-op models (not in MODELS dict but available)
    for op_name, model_id in _EDIT_OP_MODELS.items():
        catalog.append({
            "alias": op_name,
            "model_id": model_id,
            "provider": "bedrock",
            "capabilities": ["edit"],
        })

    return catalog


def _get_capabilities(provider_name: str, full_id: str, alias: str) -> list[str]:
    """Determine capabilities for a model based on provider and model type."""
    caps: list[str] = []

    # Generation capability
    if provider_name == "bedrock":
        # Bedrock SD3.5 does both gen + edit; ultra/core gen only
        if "sd3" in full_id:
            caps.extend(["generate", "edit"])
        else:
            caps.append("generate")
    elif provider_name == "deepinfra":
        # Check if model supports editing
        if (
            full_id in DeepInfraProvider._EDIT_MODELS
            or full_id in DeepInfraProvider._IMAGE_URL_MODELS
        ):
            caps.append("edit")
            # Some edit models also generate (qwen-image, etc.)
            if "Edit" not in full_id and "remove" not in full_id and "erase" not in full_id:
                caps.append("generate")
        else:
            caps.append("generate")
    elif provider_name in ("gemini", "vertex"):
        # All Gemini/Vertex models support both gen + edit (image as inline part)
        caps.extend(["generate", "edit"])
    elif provider_name == "openai":
        # OpenAI supports both gen + edit
        caps.extend(["generate", "edit"])
    elif provider_name == "openrouter":
        # All OpenRouter image models support both gen + edit via multimodal content
        caps.extend(["generate", "edit"])

    return caps


def list_models_cmd(
    provider: str | None = typer.Option(None, "--provider", "-p", help="Filter by provider"),
    capability: str | None = typer.Option(
        None, "--cap", "-c", help="Filter by capability (generate, edit)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List all available models with providers and capabilities."""
    catalog = _build_model_catalog()

    # Apply filters
    if provider:
        catalog = [m for m in catalog if m["provider"] == provider]
    if capability:
        catalog = [m for m in catalog if capability in m["capabilities"]]

    if not catalog:
        typer.echo("No models match the filter criteria.")
        return

    if json_output:
        typer.echo(json_lib.dumps(catalog, indent=2))
        return

    # Rich table output
    console = Console()
    table = Table(title="Available Models")
    table.add_column("Alias", style="cyan", no_wrap=True)
    table.add_column("Provider", style="magenta", no_wrap=True)
    table.add_column("Capabilities", style="green")
    table.add_column("Model ID", style="dim")

    # Sort by provider then alias
    catalog.sort(key=lambda m: (m["provider"], m["alias"]))

    for m in catalog:
        caps_str = ", ".join(m["capabilities"])
        table.add_row(m["alias"], m["provider"], caps_str, m["model_id"])

    console.print(table)
