"""Image generation orchestration layer.

Thin orchestrator that routes generation requests to the appropriate path:
sweep engine (multi-dimension), multi-model comparison, or single generation.
Core logic lives in generation_core.py.
"""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, Any

import sentry_sdk
import structlog

from image_creator_tool.config import build_provider_kwargs, load_settings
from image_creator_tool.errors import ImageCreatorError
from image_creator_tool.generation_core import (
    GenerationResult,
    _generate_inner,
    compose_prompt,
    generate_once,
)
from image_creator_tool.history import resolve_output_path
from image_creator_tool.imaging import make_labeled_contact_sheet
from image_creator_tool.providers import get_provider
from image_creator_tool.registry import resolve_model
from image_creator_tool.sweep import is_sweep, run_sweep

if TYPE_CHECKING:
    from image_creator_tool.providers.base import Provider

log = structlog.get_logger()

# Re-export for backward compatibility
__all__ = ["GenerationResult", "compose_prompt", "generate", "generate_once"]


def generate(args: Any, config: dict[str, Any], provider: Provider) -> list[GenerationResult]:
    """Main generation orchestrator — handles variants, metadata, and history.

    Coordinates the full generation flow: prompt composition, parallel variant
    generation, contact sheet assembly, sidecar metadata, and history logging.
    Routes to the sweep engine when any dimension has comma-separated values.
    Wrapped in a Sentry transaction for performance tracing.
    """
    model_str = args.model or ""

    # Multi-dim sweep: any comma in model/preset/platform/seed
    if is_sweep(args):
        with sentry_sdk.start_transaction(op="image.sweep", name="sweep") as txn:
            txn.set_tag("provider", provider.name)
            txn.set_tag("model", model_str)
            txn.set_tag("dry_run", str(args.dry_run))
            return run_sweep(args, config, provider)

    # Legacy multi-model path (kept for backward compat when only model has commas)
    is_multi_model = "," in model_str

    n = max(1, args.n) if not args.dry_run else 0
    with sentry_sdk.start_transaction(
        op="image.generate",
        name="generate",
    ) as txn:
        txn.set_tag("provider", provider.name)
        txn.set_tag("model", model_str or provider.default_model)
        txn.set_tag("variants", str(n))
        txn.set_tag("preset", args.preset or "none")
        txn.set_tag("dry_run", str(args.dry_run))
        txn.set_tag("multi_model", str(is_multi_model))

        if is_multi_model:
            return _generate_multi_model(args, config)
        return _generate_inner(args, config, provider)


def _generate_multi_model(args: Any, config: dict[str, Any]) -> list[GenerationResult]:
    """Generate images across multiple models for comparison.

    Each model is auto-resolved to its provider. Results are assembled
    into a labeled contact sheet showing provider/model per cell.
    """
    settings = load_settings()
    preference = settings.provider_preference

    model_names = [m.strip() for m in args.model.split(",")]
    presets: dict[str, Any] = args.presets if hasattr(args, "presets") else {}
    preset_name = args.preset or config.get("default_preset")

    # Compose prompt once
    prompt = args.prompt
    if preset_name:
        prompt = compose_prompt(args.prompt, preset_name, presets)

    # Resolve output base path
    project = args.project or config.get("default_project")
    base_output = resolve_output_path(None, args.prompt, project, config)

    # Resolve each model to its provider
    resolved: list[tuple[str, str, str]] = []  # (alias, provider_name, full_model_id)
    for model_name in model_names:
        try:
            prov_name, full_id = resolve_model(model_name, preference)
            resolved.append((model_name, prov_name, full_id))
        except KeyError as e:
            print(f"  ⚠ Skipping unknown model '{model_name}': {e}")

    if not resolved:
        raise ImageCreatorError("Error: no valid models to generate with")

    if args.dry_run:
        print(f"Multi-model comparison ({len(resolved)} models):")
        for alias, prov, full_id in resolved:
            print(f"  {alias} → {prov}/{full_id}")
        print(f"Preset: {preset_name or '(none)'}")
        print(f"Prompt:\n  {prompt}")
        return []

    # Generate one image per model (parallel)
    outputs: list[GenerationResult] = []

    def _gen_model(item: tuple[str, str, str, int]) -> GenerationResult | None:
        alias, prov_name, full_id, idx = item
        provider = get_provider(prov_name, **build_provider_kwargs(prov_name, config))
        name = f"{base_output.stem}-{idx + 1:02d}-{alias}{base_output.suffix}"
        out_path = base_output.with_name(name)
        print(f"[{idx + 1}/{len(resolved)}] {prov_name}/{alias} → {out_path.name}")
        try:
            return generate_once(
                prompt,
                out_path,
                provider,
                full_id,
                explicit_path=False,
            )
        except Exception as e:
            log.error("model generation failed", model=alias, provider=prov_name, error=str(e))
            return None

    items = [(alias, prov, full_id, i) for i, (alias, prov, full_id) in enumerate(resolved)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(items), 4)) as pool:
        outputs = [r for r in pool.map(_gen_model, items) if r is not None]

    if not outputs:
        raise ImageCreatorError("All model generations failed.")

    # Build labeled contact sheet
    cells = [
        (result.output_path, f"{resolved[i][1]}/{resolved[i][0]}")
        for i, result in enumerate(outputs)
    ]
    contact_path = base_output.with_name(f"{base_output.stem}-comparison.png")
    title = args.prompt
    if preset_name:
        title = f"{args.prompt} [preset: {preset_name}]"
    make_labeled_contact_sheet(cells, contact_path, cols=len(resolved), title=title)
    print(f"Comparison sheet: {contact_path}")

    for result in outputs:
        print(f"✓ {result.output_path} ({result.duration_s}s)")

    return outputs
