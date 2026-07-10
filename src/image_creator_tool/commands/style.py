"""Typer command group for the image-to-text style library.

Subcommands:
    extract  Analyze a reference image and print its visual style (optionally save).
    save     Persist a style description under a name.
    list     Show all saved styles.
    show     Print a saved style's description.
    delete   Remove a saved style.

The saved descriptions are consumed by `generate --style <name>`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from image_creator_tool import style as style_lib

if TYPE_CHECKING:
    from pathlib import Path

style_app = typer.Typer(
    help="Manage reusable style descriptions extracted from reference images.",
    no_args_is_help=True,
)


@style_app.command("extract")
def extract_cmd(
    images: Annotated[
        list[str],
        typer.Argument(help="Reference image(s) to analyze; multiple = one group style"),
    ],
    provider: Annotated[
        str, typer.Option("--provider", "-p", help="Vision provider: openai or gemini")
    ] = "openai",
    vision_model: Annotated[
        str | None, typer.Option("--model", "-m", help="Override the default vision model")
    ] = None,
    save: Annotated[
        str | None, typer.Option("--save", help="Save the extracted style under this name")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing saved style")
    ] = False,
) -> None:
    """Extract a prompt-ready style from one image, or a unified style from a set."""
    if len(images) == 1:
        description = style_lib.extract_style(
            images[0], provider=provider, vision_model=vision_model
        )
    else:
        description = style_lib.extract_style_group(
            images, provider=provider, vision_model=vision_model
        )
    typer.echo(description)
    if save:
        path = style_lib.save_style(save, description, overwrite=overwrite)
        typer.echo(f"Saved style '{path.stem}' -> {path}", err=True)


_DEFAULT_REFINE_MODELS = "flash-3.1,flux-max,sd-3.5-large,flux-2-pro,ultra,core"


@style_app.command("refine")
def refine_cmd(  # noqa: PLR0913 — user-facing loop knobs; each maps to one option
    subject: Annotated[str, typer.Argument(help="Subject prompt to generate while refining")],
    source: Annotated[
        list[str],
        typer.Option("--source", "-s", help="Target-style reference image(s) (repeatable)"),
    ],
    models: Annotated[
        str, typer.Option("--models", "-m", help="Comma-separated models to generate across")
    ] = _DEFAULT_REFINE_MODELS,
    iterations: Annotated[
        int, typer.Option("--iterations", "-n", help="Max refinement passes (capped at 4)")
    ] = 4,
    provider: Annotated[
        str, typer.Option("--provider", "-p", help="Vision provider for extract/assess")
    ] = "openai",
    vision_model: Annotated[
        str | None, typer.Option("--vision-model", help="Override the vision model")
    ] = None,
    threshold: Annotated[
        float, typer.Option("--threshold", help="Stop early at this mean fidelity score")
    ] = 85.0,
    start_style: Annotated[
        str | None,
        typer.Option("--start-style", help="Saved style to start from (else extract from sources)"),
    ] = None,
    save: Annotated[
        str | None, typer.Option("--save", help="Save the best refined style under this name")
    ] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="Project name (groups outputs)")
    ] = None,
) -> None:
    """Iteratively refine a style toward a reference set via generate-assess-rewrite.

    Extracts a starting style from the source set (or --start-style), then loops up
    to `iterations` (max 4): generate `subject` across `models`, score each result
    against the sources with a vision model, and rewrite the style from the
    critiques. Stops early once the mean fidelity score reaches `threshold`.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    from image_creator_tool.config import get_config_dict  # noqa: PLC0415
    from image_creator_tool.generation import generate  # noqa: PLC0415
    from image_creator_tool.presets import load_platforms, load_presets  # noqa: PLC0415
    from image_creator_tool.providers import get_provider  # noqa: PLC0415
    from image_creator_tool.refine import refine_style_loop  # noqa: PLC0415

    config = get_config_dict()
    if start_style:
        initial = style_lib.load_style(start_style)
    else:
        typer.echo(f"Extracting group style from {len(source)} source image(s)…", err=True)
        initial = style_lib.extract_style_group(
            source, provider=provider, vision_model=vision_model
        )
    typer.echo(f"Initial style: {initial}", err=True)

    placeholder = get_provider("gemini")  # sweep resolves each model's provider per-cell
    presets, platforms = load_presets(), load_platforms()
    state = {"style": initial}

    def generate_fn(style_text: str) -> list[Path]:
        state["style"] = style_text
        args = SimpleNamespace(
            prompt=subject, output=None, preset=None, style=style_text, platform=None,
            model=models, edit=None, edit_op=None, search_prompt=None, mask=None,
            reference=[], style_refs=[], object_refs=[], project=project, n=1, seed=None,
            aspect=None, size=None, quality=None, contact_cols=None, contact_cell_width=None,
            contact_bg=None, badges=True, contact_badge_radius=None, dry_run=False,
            no_metadata=False, presets=presets, platforms=platforms,
        )
        results = generate(args, config, placeholder)
        return [r.output_path for r in results]

    def assess_fn(candidate: Path) -> tuple[int, str]:
        return style_lib.assess_style_fidelity(
            source, candidate, state["style"], provider=provider, vision_model=vision_model
        )

    def refine_fn(current: str, critiques: list[str]) -> str:
        return style_lib.refine_style_text(
            current, critiques, provider=provider, vision_model=vision_model
        )

    result = refine_style_loop(
        initial_style=initial,
        generate_fn=generate_fn,
        assess_fn=assess_fn,
        refine_fn=refine_fn,
        max_iterations=iterations,
        threshold=threshold,
    )

    typer.echo("")
    for rec in result.iterations:
        typer.echo(f"  iter {rec.iteration}: score {rec.score:.1f} — {rec.style}")
    status = "converged" if result.converged else "budget exhausted"
    typer.echo(f"\nBest style (score {result.best_score:.1f}, {status}):")
    typer.echo(result.best_style)
    if save:
        path = style_lib.save_style(save, result.best_style, overwrite=True)
        typer.echo(f"Saved style '{path.stem}' -> {path}", err=True)


@style_app.command("save")
def save_cmd(
    name: Annotated[str, typer.Argument(help="Name for the style")],
    description: Annotated[str, typer.Argument(help="Style description text")],
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing saved style")
    ] = False,
) -> None:
    """Save a style description under a name."""
    path = style_lib.save_style(name, description, overwrite=overwrite)
    typer.echo(f"Saved style '{path.stem}' -> {path}")


@style_app.command("list")
def list_cmd() -> None:
    """List all saved styles."""
    names = style_lib.list_styles()
    if not names:
        typer.echo("No saved styles. Create one with 'style extract <image> --save <name>'.")
        return
    console = Console()
    table = Table(title="Saved Styles")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    for name in names:
        table.add_row(name, style_lib.load_style(name))
    console.print(table)


@style_app.command("show")
def show_cmd(
    name: Annotated[str, typer.Argument(help="Name of the saved style")],
) -> None:
    """Print a saved style's description."""
    typer.echo(style_lib.load_style(name))


@style_app.command("delete")
def delete_cmd(
    name: Annotated[str, typer.Argument(help="Name of the saved style")],
) -> None:
    """Delete a saved style."""
    style_lib.delete_style(name)
    typer.echo(f"Deleted style '{name}'.")
