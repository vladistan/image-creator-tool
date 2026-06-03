---
name: image-creator
description: Multi-provider AI image generation and editing CLI. Supports Vertex (Gemini/Imagen), DeepInfra (FLUX/Qwen/Wan/Seedream), OpenRouter (Grok/Recraft/Riverflow/GPT), Bedrock (Stability AI), and OpenAI. Features include style presets, platform sizing, multi-model sweeps, image editing (20+ models), upscaling, background removal, outpainting, style transfer, reference images, history/gallery, and parameter sweep engine.
---

# Image Creator Tool — Multi-Provider AI Image Generation & Editing

Generate and edit images from text prompts across 50+ models on 5 providers.

## When to Use

- Generate images from text descriptions
- Edit existing images with text instructions (`--edit`)
- Style transfer between images
- Upscale low-resolution images
- Remove backgrounds
- Outpaint / extend images
- Compare models side-by-side (multi-model sweep)
- Create illustrations for presentations, articles, thumbnails, social posts
- Batch parameter sweeps across models × presets × platforms × seeds

## Quick Start

```bash
# Simple generation (uses default profile — Vertex)
image-creator-tool generate "a minimalist rocket illustration"

# With style preset + platform sizing
image-creator-tool generate "interconnected nodes" --preset editorial --platform youtube

# Generate 4 variants + contact sheet
image-creator-tool generate "a crystal" --preset wireframe --n 4

# Edit existing image
image-creator-tool generate "make the sky purple" --edit ./photo.png

# Multi-model comparison sweep
image-creator-tool generate "a cat on a roof" --model flash,seedream,grok,recraft-pro

# Parameter sweep (4 dimensions → labeled contact sheets)
image-creator-tool generate "a chicken" --model seedream,grok --preset editorial,ink --platform square,blog --seed 42,99

# Style reference
image-creator-tool generate "a new landscape" --style-ref ./style-guide.png

# Upscale (Bedrock)
image-creator-tool generate "enhance detail" --edit ./small.jpg --edit-op upscale-creative --profile bedrock

# Remove background (Bedrock or DeepInfra)
image-creator-tool generate "" --edit ./photo.jpg --edit-op remove-bg --profile bedrock

# Search & replace (Bedrock)
image-creator-tool generate "a tropical beach" --edit ./photo.jpg --edit-op search-replace --search "the background" --profile bedrock

# Outpaint (extend image)
image-creator-tool generate "extend the landscape" --edit ./photo.jpg --edit-op outpaint --profile bedrock

# Re-roll last prompt
image-creator-tool again

# Browse history
image-creator-tool history -n 20

# Visual gallery (opens in browser)
image-creator-tool gallery --all --gap 10

# List all models with capabilities
image-creator-tool list-models --cap edit
```

## Providers

Supports 5 providers via config profiles (`--profile` / `-P`):

- **Vertex** — Google Cloud ADC (Gemini + Imagen models)
- **DeepInfra** — API key (FLUX, Qwen, Wan, Seedream, Bria)
- **OpenRouter** — API key (Grok, Recraft, Riverflow, GPT, Gemini)
- **Bedrock** — AWS SSO (Stability AI generation + editing)
- **OpenAI** — API key (GPT-Image models)

## Models

```bash
image-creator-tool list-models              # all models
image-creator-tool list-models --cap edit   # edit-capable only
image-creator-tool list-models -p deepinfra # by provider
image-creator-tool list-models --json       # machine-readable
```

### Key Models by Task

| Task | Best Models | Provider |
|------|------------|----------|
| General generation | flash, seedream-4.5, flux-2-dev | vertex, deepinfra |
| General editing | flux-kontext, qwen-edit, grok | deepinfra, openrouter |
| Style transfer | flux-kontext, seedream | deepinfra, openrouter |
| Upscale (quality) | qwen-edit | deepinfra |
| Upscale (resolution 14×) | upscale-creative | bedrock |
| Remove background | remove-bg, bria-remove-bg | bedrock, deepinfra |
| Outpaint | outpaint | bedrock |
| Search & replace | search-replace | bedrock |

## Style Presets

```bash
image-creator-tool list-presets
```

| Preset | Style |
|--------|-------|
| `editorial` | Thin lines on black, muted palette, technical diagram |
| `blueprint` | White/cyan on dark navy, engineering drawing |
| `ink` | Japanese sumi-e ink wash, organic brushstrokes |
| `risograph` | Flat colors, grain, zine aesthetic |
| `wireframe` | 3D wireframe mesh, glowing edges |
| `constellation` | Star map dots connected by faint lines |
| `brutalist` | Bold shapes, thick borders, flat colors |
| `grain` | Film grain photo, high ISO, warm cinematic |

## Platform Presets

```bash
image-creator-tool list-platforms
```

| Platform | Size | Use |
|----------|------|-----|
| `youtube` | 1280×720 | YouTube thumbnail |
| `youtube-short` | 1080×1920 | Shorts cover |
| `slides` | 1920×1080 | Presentations |
| `blog` | 1200×630 | Blog hero / social preview |
| `x` | 1600×900 | X/Twitter |
| `square` | 1080×1080 | Instagram/LinkedIn |
| `story` | 1080×1920 | Stories/TikTok |
| `pinterest` | 1000×1500 | Pinterest pin |

## Edit Operations (Bedrock)

Specialized image editing via `--edit-op`:

| Operation | Description | Extra Flags |
|-----------|-------------|-------------|
| `upscale-fast` | 4× upscale, fast | — |
| `upscale-conservative` | ~14× upscale, faithful | prompt optional |
| `upscale-creative` | ~14× upscale, enhanced | prompt optional |
| `remove-bg` | Remove background (transparent) | — |
| `outpaint` | Extend image edges | prompt |
| `search-replace` | Find & replace elements | `--search "..."` |
| `style-transfer` | Apply style from reference | `--style-ref img` |
| `style-guide` | Apply style from text | prompt |
| `erase` | Remove object (needs mask) | `--mask mask.png` |
| `inpaint` | Fill area (needs mask) | `--mask mask.png` + prompt |

## Parameter Sweep Engine

Comma-separated values in any dimension trigger cross-product generation:

```bash
# 2×2 grid: models as rows, presets as cols
image-creator-tool generate "subject" --model flash,grok --preset editorial,ink

# 3×8×2×2 = 96 cells across 4 sheets
image-creator-tool generate "subject" --model a,b,c --preset all8 --platform sq,blog --seed 42,99
```

Layout: first two dimensions → rows × cols; additional → separate sheets.

## Contact Sheet Configuration

```bash
image-creator-tool generate "subject" --n 4 \
  --contact-cols 4 --contact-cell-width 300 --contact-bg "#1a1a1a"
```

## Gallery

```bash
image-creator-tool gallery              # last 100 images, browser
image-creator-tool gallery --all        # entire history
image-creator-tool gallery --gap 10     # 10-min gap = new section
image-creator-tool gallery --since 2026-05-25
image-creator-tool gallery --model grok --preset editorial
```

## Output & History

- Outputs: `~/.local/share/image-creator-tool/outputs/<project?>/<timestamp>-<slug>-<hash>.png`
- History: `~/.config/image-creator-tool/history.jsonl`
- Last run: `~/.config/image-creator-tool/last.json` (for `again`)
- Sidecar JSON metadata alongside each generated image

