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
cd image-creator-tool
uv sync
```

## Usage

```bash
# Generate an image
uv run image-creator-tool generate "a robot playing chess" --preset editorial

# Dry run (see composed prompt without calling API)
uv run image-creator-tool generate "sunset" --dry-run --preset grain --platform youtube

# Deterministic output format — provider bytes are normalized to --format
# (png default, or webp/jpg) so the file extension never depends on which
# provider/model served the request. Default png also guarantees metadata/EXIF
# embedding succeeds (no "unsupported format" skip on webp-native providers).
uv run image-creator-tool generate "a robot" --format png

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

## @index Workflow: inspiration → selection → strip

Generated images and imported/web-sourced images share one short-index (`@index`) system, so
`contact-sheet`, `lookup`, `prov`, and `strip` all speak the same references. A typical
image-to-comic flow:

```bash
# 1. Inspiration: build a throwaway selection sheet straight from raw files (no import).
#    Raw paths are ephemeral — no @index minted, nothing written to the store — and each
#    cell is labeled with the filename minus its extension. @index and raw paths can mix.
uv run image-creator-tool contact-sheet ./downloads/cat-a.png ./downloads/cat-b.png @GEN12345 sheet.png

# 2. Import the keepers: copies into ref_images_dir, mints an @index, records provenance.
#    De-dupe is by image *content* (not URL): re-importing the same bytes returns the same
#    @index and appends the new source to that record's origin list — no duplicate.
uv run image-creator-tool import ./downloads/cat-a.png ./downloads/cat-b.png
# → @LN27KFOD  ./downloads/cat-a.png
# → @S5DJQU76  ./downloads/cat-b.png

# 3. Imported indices work everywhere a generated @index does:
uv run image-creator-tool lookup @LN27KFOD
uv run image-creator-tool prov show @LN27KFOD          # origin=imported, traces back to source
uv run image-creator-tool generate "a cat astronaut" --ref @LN27KFOD

# 4. Assemble panels into a comic strip (bordered, guttered; no numbered badges):
uv run image-creator-tool strip @LN27KFOD @S5DJQU76 @GEN12345 strip.png --caption
uv run image-creator-tool strip @A @B @C @D grid.png --cols 2 --gutter 20 --border 8

# 5. Reclaim discovery leftovers — forget picked-over imports, or prune the unreferenced:
uv run image-creator-tool forget @S5DJQU76               # removes copy + sidecar + index entry
uv run image-creator-tool forget --prune --dry-run       # preview imports no generation references
uv run image-creator-tool forget --prune                 # delete them
```

Notes:

- `import`/`forget` operate only on imported images. `forget` refuses a generated @index
  (leaving it intact), and both exit non-zero on a missing source or unknown @index.
- `strip` and `contact-sheet` exit non-zero on any unresolvable panel/index or missing raw path.
- Imported files live in `ref_images_dir` (default `<output_dir>/ref-images`), which is covered
  by the recursive index scan. See Configuration below.

**Troubleshooting:** `contact-sheet` (and `strip`) return a non-zero exit code on an unresolved
`@index`. When scripting, do not mask `$?` behind a pipe — e.g. `... | sed ...` or `... | head`
reports the exit status of the *last* command in the pipe, not the tool. Capture the tool's own
exit code before piping its output.

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

Reference-image store: `ref_images_dir` (default `<output_dir>/ref-images`) holds images
brought in via `import`; it is created on first import and included in the recursive index scan.

Environment variables (override config): `IMAGE_CREATOR_DEFAULT_PROVIDER`, `IMAGE_CREATOR_OUTPUT_DIR`, `IMAGE_CREATOR_REF_IMAGES_DIR`, etc.

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
