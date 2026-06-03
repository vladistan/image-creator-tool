# Contributing to image-creator-tool

## Development Setup

```bash
uv sync
```

## Running Tests

```bash
uv run pytest -v
uv run pytest --cov=src/image_creator_tool --cov-report=term-missing
```

## Linting & Type Checking

```bash
uv run ruff check .
uv run ruff check --fix .   # Auto-fix
uv run mypy src/
```

## Code Style

- Python 3.13+, strict mypy
- ruff for linting and formatting
- structlog for diagnostic logging (not print)
- typer for CLI (not argparse)
- Type hints on all public functions
- Docstrings that explain *why*, not just *what*

## Architecture

```
src/image_creator_tool/
├── __init__.py          # Package + version
├── cli.py               # Typer app, commands, entry point
├── config.py            # pydantic-settings, TOML config
├── presets.py           # Preset/platform loading + merge
├── generation.py        # Orchestration (provider-agnostic)
├── imaging.py           # ImageMagick wrappers
├── history.py           # Output paths, sidecars, history
├── errors.py            # Exception hierarchy
├── monitoring.py        # structlog + Sentry setup
├── commands/            # CLI subcommand modules
│   ├── init_cmd.py
│   ├── again.py
│   ├── history.py
│   └── presets.py
├── providers/           # Image generation backends
│   ├── base.py          # Provider ABC
│   └── gemini.py        # Google Gemini implementation
└── data/                # Bundled YAML data
    ├── presets.yaml
    └── platforms.yaml
```

## Adding a New Provider

1. Create `providers/your_provider.py` implementing `Provider` ABC
2. Register in `providers/__init__.py` REGISTRY
3. Add tests in `tests/test_providers.py`

## Adding a Preset

Add to `src/image_creator_tool/data/presets.yaml`:

```yaml
my-preset:
  description: "Short description of the style"
  prompt: "prompt template with {subject} placeholder"
```

The `{subject}` placeholder is required and validated on load.
