# image-creator-tool

Multi-provider AI image generation CLI with style presets, platform sizing, and variant generation.

## Features

- **Multi-provider architecture** — currently supports Google Gemini; extensible to other providers
- **Style presets** — apply reusable prompt templates (editorial, blueprint, ink, risograph, etc.)
- **Platform sizing** — auto-resize/crop to target dimensions (YouTube, slides, blog, X, Instagram)
- **Variant generation** — generate N variants in parallel with contact sheet
- **Edit mode** — modify existing images with text instructions
- **Reference images** — anchor style/aesthetic from example images
- **History & replay** — browse past generations, replay the last one with `again`
- **Structured logging** — structured output via structlog for diagnostics
- **Sentry integration** — optional error tracking

## Installation

```bash
cd tools/image-creator-tool
uv sync
```

## Usage

```bash
# Generate an image
uv run image-creator-tool generate "a robot playing chess" --preset editorial

# Dry run (see composed prompt without calling API)
uv run image-creator-tool generate "sunset" --dry-run --preset grain --platform youtube

# List available options
uv run image-creator-tool list-presets
uv run image-creator-tool list-platforms
uv run image-creator-tool list-providers

# Regenerate last image
uv run image-creator-tool again

# View history
uv run image-creator-tool history -n 10

# Show version
uv run image-creator-tool --version
```

## Configuration

Configuration file: `~/.config/image-creator-tool/config.toml`

```toml
default_profile = "vertex-work"
sentry_dsn = ""  # optional

[profile.vertex-work]
provider = "vertex"
gcp_project = "my-project"
default_model = "flash"

[profile.deepinfra]
provider = "deepinfra"
api_key = "your-key-here"  # pragma: allowlist secret
default_model = "flux-2-dev"

[profile.openrouter]
provider = "openrouter"
api_key = "sk-or-v1-..."  # pragma: allowlist secret

[profile.bedrock]
provider = "bedrock"
aws_profile = "my-aws-profile"
aws_region = "us-west-2"

[profile.openai]
provider = "openai"
api_key = "sk-..."  # pragma: allowlist secret
```

Switch profiles via `--profile` / `-P` flag: `image-creator-tool generate "subject" -P deepinfra`

Environment variables (override config): `IMAGE_CREATOR_DEFAULT_PROVIDER`, `IMAGE_CREATOR_OUTPUT_DIR`, etc.

See `config.example.toml` for full documentation.

## Custom Presets & Platforms

Add your own presets at `~/.config/image-creator-tool/presets.yaml`:

```yaml
my-style:
  description: "My custom style"
  prompt: "{subject} in my unique artistic style"
```

Custom platforms at `~/.config/image-creator-tool/platforms.yaml`:

```yaml
my-size:
  description: "My custom size"
  width: 1600
  height: 1200
```

User presets/platforms merge with (and can override) the bundled defaults.

## Requirements

- Python >= 3.13
- `GEMINI_API_KEY` environment variable (or SOPS-encrypted secrets)
- ImageMagick 7 (`magick` command) for platform resizing and contact sheets

## Development

```bash
uv sync
uv run pytest -v
uv run ruff check .
uv run mypy src/
```

## Acknowledgments

This tool is an adaptation of the **Nano Banana** image generation skill by [Gleb Kalinin](https://github.com/glebis), originally published in the [claude-skills](https://github.com/glebis/claude-skills) repository. The core concepts of multi-provider image generation, style presets, and the CLI workflow all originate from that work.

## License

MIT
