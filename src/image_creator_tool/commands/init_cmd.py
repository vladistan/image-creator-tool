"""Onboarding wizard command for image-creator-tool.

Checks dependencies, API key availability, and guides the user
through initial configuration setup.
"""

from __future__ import annotations

import shutil
import tomllib

import typer

from image_creator_tool.config import CONFIG_DIR, CONFIG_FILE
from image_creator_tool.providers import get_provider


def init() -> None:
    """Run the onboarding wizard — check dependencies and configure defaults."""
    typer.echo("Image Creator Tool — Onboarding\n")

    # Check binary dependencies
    deps = {
        "magick": "brew install imagemagick",
    }
    missing = [d for d in deps if shutil.which(d) is None]
    if missing:
        typer.echo("Missing dependencies:")
        for d in missing:
            typer.echo(f"  {d}  — install with: {deps[d]}")
        typer.echo()
    else:
        typer.echo("✓ Dependencies (imagemagick) installed\n")

    # Check API key
    provider = get_provider("gemini")
    key = provider.get_api_key()
    if key:
        typer.echo("✓ GEMINI_API_KEY accessible")
    else:
        typer.echo("✗ GEMINI_API_KEY not found.")
        typer.echo("  Set it: export GEMINI_API_KEY=AIza... in your shell rc")
    typer.echo()

    # Load existing config if present
    raw_data: dict[str, object] = {}
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("rb") as f:
                raw_data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            typer.echo(f"Warning: could not read existing config: {e}", err=True)

    # Extract only the flat base settings (strings) for interactive editing
    base_config: dict[str, str] = {
        k: str(v) for k, v in raw_data.items() if isinstance(v, str)
    }

    def ask(key: str, prompt: str, default: str = "") -> str:
        current = base_config.get(key, default)
        shown = f" [{current}]" if current else ""
        val = input(f"{prompt}{shown}: ").strip()
        return val or current

    base_config["default_model"] = ask(
        "default_model",
        "Default model (flash / pro / flash-2.5)",
        "flash",
    )
    base_config["default_platform"] = ask(
        "default_platform",
        "Default platform (youtube / slides / blog / square / none)",
        "",
    )
    base_config["default_project"] = ask(
        "default_project",
        "Default project name (optional, e.g. 'blog')",
        "",
    )
    out_dir = ask(
        "output_dir",
        "Default output directory",
        "~/.local/share/image-creator-tool/outputs",
    )
    base_config["output_dir"] = out_dir

    # Write base settings, preserving [profile.*] sections from existing file
    base_config = {k: v for k, v in base_config.items() if v}

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f'{k} = "{v}"' for k, v in sorted(base_config.items())]

    # Preserve profile table sections if they exist
    profiles = raw_data.get("profile")
    if isinstance(profiles, dict):
        lines.append("")
        for prof_name, prof_data in profiles.items():
            if not isinstance(prof_data, dict):
                continue
            lines.append(f"[profile.{prof_name}]")
            for pk, pv in prof_data.items():
                lines.append(f'{pk} = "{pv}"')
            lines.append("")

    CONFIG_FILE.write_text("\n".join(lines) + "\n")
    typer.echo(f"\n✓ Config saved to {CONFIG_FILE}")
    typer.echo("\nTry: image-creator-tool generate --preset editorial 'a robot'")
