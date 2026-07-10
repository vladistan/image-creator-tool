"""Image Creator Tool CLI — multi-provider AI image generation.

Typer-based CLI with subcommands for generation, preset listing,
history browsing, and provider management.
"""

from __future__ import annotations

import subprocess
import sys
from enum import IntEnum
from types import SimpleNamespace
from typing import Annotated, Any

import sentry_sdk
import typer

from image_creator_tool import __version__
from image_creator_tool.commands.again import again
from image_creator_tool.commands.contact_sheet import contact_sheet
from image_creator_tool.commands.gallery import gallery
from image_creator_tool.commands.history import history
from image_creator_tool.commands.init_cmd import init
from image_creator_tool.commands.lookup import lookup
from image_creator_tool.commands.models import list_models_cmd
from image_creator_tool.commands.presets import (
    list_platforms_cmd,
    list_presets_cmd,
    list_providers_cmd,
)
from image_creator_tool.commands.prov import prov_app
from image_creator_tool.commands.style import style_app
from image_creator_tool.config import get_config_dict, list_profiles, load_settings
from image_creator_tool.errors import ImageCreatorError
from image_creator_tool.generation import generate
from image_creator_tool.indexer import expand_reference
from image_creator_tool.monitoring import setup_logging, setup_sentry
from image_creator_tool.presets import load_platforms, load_presets
from image_creator_tool.providers import get_provider, list_providers
from image_creator_tool.redaction import sanitize_error
from image_creator_tool.style import load_style


class ExitCode(IntEnum):
    """Standard exit codes for image-creator-tool."""

    SUCCESS = 0
    GENERAL_ERROR = 1
    USAGE_ERROR = 2
    INPUT_ERROR = 3
    OUTPUT_ERROR = 4
    NETWORK_ERROR = 5
    TIMEOUT = 6


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"image-creator-tool {__version__}")
        raise typer.Exit()


app = typer.Typer(
    help="Image Creator Tool — multi-provider AI image generation CLI",
    no_args_is_help=True,
)


@app.callback()
def _global_options(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-V",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Image Creator Tool — multi-provider AI image generation CLI."""


# --- Register subcommands ---

app.command("init")(init)
app.command("again")(again)
app.command("lookup")(lookup)
app.command("contact-sheet")(contact_sheet)
app.command("history")(history)
app.command("gallery")(gallery)
app.command("list-models")(list_models_cmd)
app.command("list-presets")(list_presets_cmd)
app.command("list-platforms")(list_platforms_cmd)
app.command("list-providers")(list_providers_cmd)
app.add_typer(style_app, name="style")
app.add_typer(prov_app, name="prov")


@app.command("list-profiles")
def list_profiles_cmd() -> None:
    """List available configuration profiles."""
    settings = load_settings()
    profiles = list_profiles()
    if not profiles:
        typer.echo("No profiles configured. Add [profile.*] sections to config.toml")
        return
    for name in profiles:
        marker = " (default)" if name == settings.default_profile else ""
        typer.echo(f"  {name}{marker}")


def _build_cli_provider_kwargs(effective_provider: str, config: dict[str, Any]) -> dict[str, str]:
    """Build provider-specific constructor kwargs from config.

    The active profile's `api_key` is a per-provider secret, so it is only
    applied when that profile targets `effective_provider` (config's
    `default_provider` reflects the active profile's provider). A CLI `-p`
    override to a different provider drops the key, letting that provider fall
    back to its own env-var credential resolution instead of receiving a
    mismatched key.
    """
    kwargs: dict[str, str] = {}
    if effective_provider == "vertex":
        if "gcp_project" in config:
            kwargs["project"] = config["gcp_project"]
    elif effective_provider == "bedrock":
        if "aws_profile" in config:
            kwargs["aws_profile"] = config["aws_profile"]
        if "aws_region" in config:
            kwargs["aws_region"] = config["aws_region"]
    if "api_key" in config and config.get("default_provider") == effective_provider:
        kwargs["api_key"] = config["api_key"]
    return kwargs


def _open_result(results: list[Any], is_multi_model: bool, n: int) -> None:
    """Open generated image(s) in default viewer."""
    if is_multi_model or n > 1:
        out_dir = results[0].output_path.parent
        sheets = (
            list(out_dir.glob("*comparison*"))
            + list(out_dir.glob("*contact*"))
            + list(out_dir.glob("*sweep*"))
        )
        if sheets:
            newest = max(sheets, key=lambda p: p.stat().st_mtime)
            subprocess.run(["open", str(newest)], check=False)
        else:
            subprocess.run(["open", str(results[0].output_path)], check=False)
    else:
        subprocess.run(["open", str(results[0].output_path)], check=False)


# --- Main generation command ---


@app.command("generate")
def generate_cmd(  # noqa: PLR0913
    prompt: Annotated[str, typer.Argument(help="Subject text or edit instruction")],
    output: Annotated[str | None, typer.Argument(help="Output path (auto if omitted)")] = None,
    provider_name: Annotated[
        str | None, typer.Option("--provider", "-p", help="Image generation provider")
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name/alias; comma-separated for sweep"),
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option("--preset", help="Style preset name; comma-separated for sweep"),
    ] = None,
    platform: Annotated[
        str | None,
        typer.Option("--platform", help="Platform sizing preset; comma-separated for sweep"),
    ] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="Project name (groups outputs)")
    ] = None,
    edit: Annotated[
        str | None, typer.Option("--edit", help="Source image for edit mode")
    ] = None,
    edit_op: Annotated[
        str | None,
        typer.Option(
            "--edit-op",
            help="Edit operation: remove-bg, upscale-fast, upscale-conservative, "
            "upscale-creative, erase, inpaint, outpaint, search-replace, "
            "style-transfer, style-guide",
        ),
    ] = None,
    search: Annotated[
        str | None,
        typer.Option("--search", help="Search prompt for search-replace edit operation"),
    ] = None,
    mask: Annotated[
        str | None,
        typer.Option("--mask", help="Mask image (B&W PNG) for erase/inpaint operations"),
    ] = None,
    ref: Annotated[
        list[str] | None, typer.Option("--ref", help="Reference image (repeatable)")
    ] = None,
    style_ref: Annotated[
        list[str] | None,
        typer.Option("--style-ref", help="Style reference image (repeatable)"),
    ] = None,
    style: Annotated[
        str | None,
        typer.Option("--style", help="Saved style name to prepend to the prompt"),
    ] = None,
    insert_object: Annotated[
        list[str] | None,
        typer.Option("--insert-object", help="Object insertion reference image (repeatable)"),
    ] = None,
    n: Annotated[int, typer.Option("--n", help="Number of variants")] = 1,
    seed: Annotated[
        str | None, typer.Option("--seed", help="Random seed (or comma-separated for sweep)")
    ] = None,
    aspect: Annotated[
        str | None,
        typer.Option("--aspect", help="Aspect ratio hint e.g. 16:9, 1:1 (provider-dependent)"),
    ] = None,
    size: Annotated[
        str | None,
        typer.Option("--size", help="Output size hint e.g. 1024x1024, 2k (provider-dependent)"),
    ] = None,
    quality: Annotated[
        str | None,
        typer.Option("--quality", help="Quality hint e.g. standard, hd (provider-dependent)"),
    ] = None,
    contact_cols: Annotated[
        int | None,
        typer.Option("--contact-cols", help="Contact sheet column count (default 2)"),
    ] = None,
    contact_cell_width: Annotated[
        int | None,
        typer.Option("--contact-cell-width", help="Contact sheet cell width in pixels"),
    ] = None,
    contact_bg: Annotated[
        str | None,
        typer.Option("--contact-bg", help="Contact sheet background colour (#rrggbb or r,g,b)"),
    ] = None,
    no_badges: Annotated[
        bool,
        typer.Option("--no-badges", help="Disable numbered badges on contact sheets"),
    ] = False,
    contact_badge_radius: Annotated[
        int | None,
        typer.Option("--contact-badge-radius", help="Badge radius in pixels (default 30)"),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show prompt without calling API")
    ] = False,
    no_metadata: Annotated[
        bool, typer.Option("--no-metadata", help="Skip sidecar JSON")
    ] = False,
    open_result: Annotated[
        bool, typer.Option("--open", "-O", help="Open generated image in default viewer")
    ] = False,
    profile: Annotated[
        str | None, typer.Option("--profile", "-P", help="Config profile to use")
    ] = None,
) -> None:
    """Generate an image from a text prompt."""
    # Resolve config with profile overlay
    config = get_config_dict(profile_name=profile)

    # Multi-model mode: skip single-provider validation (registry handles it)
    is_multi_model = model is not None and "," in model

    # Use provider from config if not explicitly passed on CLI
    effective_provider = (
        provider_name if provider_name else config.get("default_provider", "gemini")
    )

    # For multi-model, we don't need a single provider — each model resolves its own
    if not is_multi_model:
        # Validate provider
        available = list_providers()
        if effective_provider not in available:
            typer.echo(
                f"Error: unknown provider '{effective_provider}'. Available: {available}",
                err=True,
            )
            raise typer.Exit(code=ExitCode.USAGE_ERROR)

    # Pass provider-specific config
    provider_kwargs = _build_cli_provider_kwargs(effective_provider, config)

    if is_multi_model:
        provider = get_provider("gemini")  # placeholder — multi-model resolves per-model
    else:
        provider = get_provider(effective_provider, **provider_kwargs)

    # Validate model if specified
    if model and model not in provider.MODELS and model != provider.resolve_model(model):
        typer.echo(
            f"Warning: model '{model}' not in known aliases {list(provider.MODELS.keys())}."
            " Passing as full model ID.",
            err=True,
        )

    # Resolve saved style name to its description text (raises if unknown).
    style_description = load_style(style) if style else None

    # Expand any `@INDEX` short-index references in path options to full paths.
    search_dir = config["output_dir"]
    edit = expand_reference(edit, search_dir) if edit else edit
    mask = expand_reference(mask, search_dir) if mask else mask
    ref = [expand_reference(r, search_dir) for r in ref] if ref else ref
    style_ref = [expand_reference(r, search_dir) for r in style_ref] if style_ref else style_ref
    insert_object = (
        [expand_reference(r, search_dir) for r in insert_object] if insert_object else insert_object
    )

    args = SimpleNamespace(
        prompt=prompt,
        output=output,
        preset=preset,
        style=style_description,
        platform=platform,
        model=model,
        edit=edit,
        edit_op=edit_op,
        search_prompt=search,
        mask=mask,
        reference=ref or [],
        style_refs=style_ref or [],
        object_refs=insert_object or [],
        project=project,
        n=n,
        seed=seed,
        aspect=aspect,
        size=size,
        quality=quality,
        contact_cols=contact_cols,
        contact_cell_width=contact_cell_width,
        contact_bg=contact_bg,
        badges=not no_badges,
        contact_badge_radius=contact_badge_radius,
        dry_run=dry_run,
        no_metadata=no_metadata,
        presets=load_presets(),
        platforms=load_platforms(),
    )
    results = generate(args, config, provider)

    # Open result in default viewer if requested
    if open_result and results:
        _open_result(results, is_multi_model, n)


def main() -> None:
    """Entry point with exception handling."""
    setup_logging()
    setup_sentry()

    try:
        app()
    except ImageCreatorError as e:
        typer.echo(f"Error: {sanitize_error(str(e))}", err=True)
        sentry_sdk.flush(timeout=2)
        sys.exit(ExitCode.GENERAL_ERROR)
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.", err=True)
        sys.exit(130)
    except Exception as e:
        # Last-resort guard for the CLI boundary: an unexpected exception's raw
        # string can embed a provider key; redact it before echoing so no secret
        # escapes even on the unhandled path.
        typer.echo(f"Error: {sanitize_error(str(e))}", err=True)
        sentry_sdk.flush(timeout=2)
        sys.exit(ExitCode.GENERAL_ERROR)
    finally:
        sentry_sdk.flush(timeout=2)
