"""CLI integration tests using typer CliRunner."""

from image_creator_tool import __version__
from image_creator_tool.cli import app


def test_help_exits_zero(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Image Creator Tool" in result.stdout


def test_version_flag(runner):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_args_shows_help(runner):
    result = runner.invoke(app, [])
    # no_args_is_help shows help text (exit code varies by typer version)
    assert "Commands" in result.stdout or "Usage" in result.stdout


def test_list_presets(runner):
    result = runner.invoke(app, ["list-presets"])
    assert result.exit_code == 0
    assert "editorial" in result.stdout
    assert "blueprint" in result.stdout


def test_list_platforms(runner):
    result = runner.invoke(app, ["list-platforms"])
    assert result.exit_code == 0
    assert "youtube" in result.stdout
    assert "1280x720" in result.stdout


def test_list_providers(runner):
    result = runner.invoke(app, ["list-providers"])
    assert result.exit_code == 0
    assert "gemini" in result.stdout


def test_dry_run_composes_prompt(runner):
    result = runner.invoke(
        app, ["generate", "a robot", "--dry-run", "--preset", "editorial",
        "--provider", "gemini"]
    )
    assert result.exit_code == 0
    assert "Model:" in result.stdout
    assert "Preset: editorial" in result.stdout
    assert "a robot" in result.stdout


def test_dry_run_with_platform(runner):
    result = runner.invoke(
        app,
        ["generate", "sunset", "--dry-run", "--platform", "youtube", "--provider", "gemini"],
    )
    assert result.exit_code == 0
    assert "Platform: youtube" in result.stdout


def test_invalid_provider(runner):
    result = runner.invoke(
        app, ["generate", "test", "--provider", "nonexistent", "--dry-run"]
    )
    assert result.exit_code != 0


def test_generate_help(runner):
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--provider" in result.stdout
    assert "--preset" in result.stdout
    assert "--dry-run" in result.stdout
