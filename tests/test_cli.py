"""CLI integration tests using typer CliRunner."""

from image_creator_tool import __version__
from image_creator_tool.cli import app
from image_creator_tool.indexer import register_index


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


def _capture_generate_args(monkeypatch):
    """Replace cli.generate with a capturing stub; return the mutable holder."""
    holder = {}

    def _capture(args, config, provider):
        holder["args"] = args
        return []

    monkeypatch.setattr("image_creator_tool.cli.generate", _capture)
    return holder


def test_generate_badge_defaults_marshalled(runner, monkeypatch):
    holder = _capture_generate_args(monkeypatch)
    result = runner.invoke(app, ["generate", "a robot", "--provider", "gemini"])
    assert result.exit_code == 0
    assert holder["args"].badges is True
    assert holder["args"].contact_badge_radius is None


def test_generate_no_badges_flag_marshalled(runner, monkeypatch):
    holder = _capture_generate_args(monkeypatch)
    result = runner.invoke(app, ["generate", "a robot", "--provider", "gemini", "--no-badges"])
    assert result.exit_code == 0
    assert holder["args"].badges is False


def test_generate_badge_radius_flag_marshalled(runner, monkeypatch):
    holder = _capture_generate_args(monkeypatch)
    result = runner.invoke(
        app, ["generate", "a robot", "--provider", "gemini", "--contact-badge-radius", "40"]
    )
    assert result.exit_code == 0
    assert holder["args"].contact_badge_radius == 40


def test_generate_resolves_index_reference_in_edit(runner, monkeypatch, tmp_path):
    """`--edit @INDEX` is expanded to the indexed image's path before generation."""
    image = tmp_path / "barn.png"
    image.write_bytes(b"barn")
    index = register_index(image, key="entity-barn")
    monkeypatch.setenv("IMAGE_CREATOR_OUTPUT_DIR", str(tmp_path))

    holder = _capture_generate_args(monkeypatch)
    result = runner.invoke(
        app, ["generate", "make it snowy", "--provider", "gemini", "--edit", f"@{index}"]
    )
    assert result.exit_code == 0
    assert holder["args"].edit == str(image)


def test_generate_invalid_index_reference_errors(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_CREATOR_OUTPUT_DIR", str(tmp_path))
    _capture_generate_args(monkeypatch)
    result = runner.invoke(
        app, ["generate", "make it snowy", "--provider", "gemini", "--edit", "@ZZZZZZZZ"]
    )
    assert result.exit_code != 0
