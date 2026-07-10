---
name: image-creator
description: Multi-provider AI image generation and editing CLI. Supports Vertex (Gemini/Imagen), DeepInfra (FLUX/Qwen/Wan/Seedream), OpenRouter (Grok/Recraft/Riverflow/GPT), Bedrock (Stability AI), OpenAI, and HuggingFace (auto-routed InferenceClient, hf-inference/fal-ai/replicate). Features include image-to-text style extraction (single image or multi-image group), closed-loop style refinement (generate → vision-assess vs reference → rewrite → repeat), style presets, platform sizing, cross-provider multi-model sweeps, image editing (20+ models), upscaling, background removal, outpainting, style transfer, reference images, history/gallery, provenance/short-index tracking, and a parameter sweep engine.
---

# Image Creator Tool — Multi-Provider AI Image Generation & Editing

Generate and edit images from text prompts across 50+ models on 6 providers.

## When to Use

- Generate images from text descriptions
- Edit existing images with text instructions (`--edit`)
- Style transfer between images
- **Lift a style from reference image(s)** and reuse it (`style extract` → `--style`)
- **Match a target style faithfully** via the closed-loop refiner (`style refine`)
- Upscale low-resolution images
- Remove backgrounds
- Outpaint / extend images
- Compare models side-by-side (multi-model / cross-provider sweep)
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

Providers via config profiles (`--profile` / `-P`) or per-model registry resolution:

- **Vertex** — Google Cloud ADC (Gemini + Imagen models)
- **DeepInfra** — API key (FLUX, Qwen, Wan, Seedream, Bria)
- **OpenRouter** — API key (Grok, Recraft, Riverflow, GPT, Gemini)
- **Bedrock** — AWS SSO (Stability AI generation + editing)
- **OpenAI** — API key (GPT-Image models)
- **HuggingFace** — `HF_TOKEN`; routes via `huggingface_hub` InferenceClient with
  `provider="auto"`, so a model is dispatched to whichever inference backend serves it
  (hf-inference / fal-ai / replicate / …). Aliases (`flux-schnell`, `sdxl-turbo`,
  `stable-diffusion-xl`) plus **any custom repo slug** (`krea/Krea-2-Turbo`,
  `black-forest-labs/FLUX.1-Krea-dev`) — a slug works only if some provider serves it
  live. Requires the `[huggingface]` extra.
- **LiteLLM**, **Azure OpenAI** — registered (`litellm`, `azure-openai`); optional
  `[litellm]` / `[azure]` extras. Code complete but not yet live-validated.

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

## Custom Styles — Extract & Refine (from reference images)

Two style mechanisms: **`--preset`** names one of the built-in templates above;
**`--style`** names a *custom* style you saved (in `~/.config/image-creator-tool/styles/`),
usually lifted from reference images. `--preset` = curated/built-in; `--style` = your own.
They can be combined.

### Extract a style from reference image(s)

```bash
# From one image
image-creator-tool style extract ./ref.png --save my-style

# From a SET of images → ONE unified "group" style. Each image is extracted
# separately, then an LLM merges them preserving the UNION of the palette, so
# accent colours survive (a single all-at-once call collapses to the two
# dominant hues and drops accents).
image-creator-tool style extract a.png b.png c.png --save brand-style

image-creator-tool style list          # saved styles
image-creator-tool style show my-style
image-creator-tool style delete my-style
```

Vision provider: `--provider openai` (default, needs `OPENAI_API_KEY`) or `gemini`.

### Apply a saved style

```bash
image-creator-tool generate "a fox reading a book" --style brand-style
```

### Closed-loop style refinement (`style refine`)

Iteratively converge a style toward a reference set: extract → generate across
several models → score each result against the sources with a vision model
(per-source pairwise) → rewrite the style from the critiques → repeat. Bounded to
≤4 iterations, early-stop once mean fidelity clears `--threshold`.

```bash
image-creator-tool style refine "a bear drinking espresso" \
  -s ref1.png -s ref2.png -s ref3.png \
  -m flux-max,flash-3.1,sd-3.5-large \
  --iterations 4 --threshold 90 --save bear-style
```

- `-s/--source` (repeatable): target-style reference images
- `-m/--models`: comma-separated models generated across each iteration (per-source
  assessment means cost ≈ models × sources × iterations vision calls — keep the set small)
- `-n/--iterations` (max 4), `--threshold` (early stop), `--start-style` (start from a saved style)

### Style-fidelity tips

- **Text-prefix `--style` is lossy** — models default to *more* (saturation, clutter).
  Group extraction preserves accent colours the single-call summary drops.
- **Pin the background/negative space explicitly.** Models drift to bright fills; if the
  target has dark negative space, say so ("solid near-black background, generous black
  negative space"). Some models (e.g. `sd-3.5-large`) still resist background control.
- **`style refine` corrects drift automatically** by scoring against the sources and
  rewriting — watch the score climb across iterations (e.g. 72.7 → 76.0 → 81.3).

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
| `outpaint` | Extend image edges | prompt (use `"."` for model-driven) |
| `search-replace` | Find & replace elements | `--search "..."` |
| `style-transfer` | Apply style from reference | `--style-ref img` |
| `style-guide` | Apply style from text | prompt |
| `erase` | Remove object (needs mask) | `--mask mask.png` |
| `inpaint` | Fill area (needs mask) | `--mask mask.png` + prompt |

## Gotchas & Tips

- **Edit mode always requires a prompt.** A truly empty `""` is rejected for `--edit`/`--edit-op`. To let the model invent surroundings (outpaint) or self-direct an edit, pass a minimal token like `"."`.
- **Bedrock edit ops return JPEG.** Even if you request a `.png` output path, Bedrock (Stability) operations save as `.jpg` and print a warning. Name the output `.jpg` to avoid surprises.
- **Iterative outpaint = progressive zoom-out.** Feed each outpaint's output back in as the next `--edit` source to recursively expand the scene. With a minimal `"."` prompt the model continues existing textures conservatively; with a descriptive prompt it adds new subjects each pass.
- **Cross-provider sweeps work.** A single `--model a,b,c` spanning *different providers* resolves each model to its own provider from the registry (the `-p`/profile is ignored for multi-model) and gives each provider its own credentials — e.g. `--model flash-3.1,flux-max,sd-3.5-large,flux-2-pro,ultra` sweeps Vertex + OpenRouter + HuggingFace + DeepInfra + Bedrock in one run. Failed cells are isolated (skipped with an error) while the rest still render. Only registered *aliases* resolve per-provider; unknown custom slugs fall back to the passed provider, so keep custom slugs to single-model runs.
- **General models don't truly upscale.** `flash` (Vertex) and `gpt-image-*` (OpenAI) regenerate at the target size, inventing fine detail/text rather than recovering it. For faithful enlargement use Bedrock `upscale-*` ops; for closest-to-source content, Vertex `flash` edit; for most polished look, gpt-image-2 (slow, ~60–85s).
- **SVG/vector output:** `recraft-vector` (OpenRouter) emits true geometric vector paths (great for low-poly). `bria-vector` (DeepInfra) generates a raster then traces it with VTracer (posterized cut-paper look). Render `.svg` previews with `rsvg-convert -w 600 file.svg -o out.png`.

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

