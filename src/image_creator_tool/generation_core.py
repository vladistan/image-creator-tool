"""Core image generation logic — single-image generation, prompt composition, and inner loop.

Extracted from generation.py to break the circular dependency with sweep.py.
This module contains no imports from sweep.py.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sentry_sdk
import structlog

from image_creator_tool.config import build_provider_kwargs, load_settings
from image_creator_tool.errors import ImageCreatorError, PermanentAPIError
from image_creator_tool.history import (
    append_history,
    resolve_output_path,
    save_last_run,
    write_sidecar,
)
from image_creator_tool.imaging import apply_platform_fit, make_contact_sheet
from image_creator_tool.indexer import register_index
from image_creator_tool.provenance import (
    ProvenanceRecord,
    embed_exif_metadata,
    write_provenance_sidecar,
)
from image_creator_tool.providers import get_provider
from image_creator_tool.providers.base import GenerationParams
from image_creator_tool.registry import resolve_model

if TYPE_CHECKING:
    from image_creator_tool.providers.base import Provider

log = structlog.get_logger()


@dataclass
class GenerationResult:
    """Result of a single image generation call."""

    output_path: Path
    prompt: str
    model: str
    preset: str | None
    platform: str | None
    edit_source: str | None
    timestamp: str
    duration_s: float
    reference: list[str] = field(default_factory=list)


def compose_prompt(subject: str, preset_name: str | None, presets: dict[str, Any]) -> str:
    """Apply a style preset template to the user's subject text.

    Returns the raw subject if no preset is specified.
    Raises PermanentAPIError if the preset name is not found.
    """
    if not preset_name:
        return subject
    if preset_name not in presets:
        raise PermanentAPIError(
            f"Error: preset '{preset_name}' not found. Available: {list(presets.keys())}"
        )
    template: str = presets[preset_name]["prompt"]
    return template.replace("{subject}", subject)


def _detect_format(data: bytes) -> str:
    """Detect image format from file magic bytes.

    Returns the canonical extension (without dot): png, jpeg, svg, webp, gif.
    """
    if data[:4] == b"\x89PNG":
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data.lstrip()[:5] in (b"<?xml", b"<svg "):
        return "svg"
    return "png"  # default fallback


def _fix_output_extension(
    output_path: Path, detected_format: str, *, explicit_path: bool = False
) -> Path:
    """Adjust output path extension to match detected format.

    Only warns if the user explicitly specified an output path with a mismatched extension.
    Auto-generated paths are silently corrected.
    """
    expected_ext = f".{detected_format}"
    if detected_format == "jpeg":
        expected_ext = ".jpg"

    current_ext = output_path.suffix.lower()
    if current_ext == expected_ext:
        return output_path
    if current_ext in (".jpg", ".jpeg") and detected_format == "jpeg":
        return output_path

    # Extension mismatch — warn only if user chose the path
    if explicit_path:
        print(
            f"  ⚠ Output format is {detected_format.upper()}, "
            f"saving as {expected_ext} instead of {output_path.suffix}"
        )
    new_path = output_path.with_suffix(expected_ext)
    return new_path


def generate_once(  # noqa: PLR0913
    prompt: str,
    output_path: Path,
    provider: Provider,
    model: str,
    *,
    edit_source: Path | None = None,
    reference_images: list[Path] | None = None,
    platform: dict[str, Any] | None = None,
    explicit_path: bool = False,
    size: str | None = None,
    quality: str | None = None,
    aspect_ratio: str | None = None,
    seed: int | None = None,
    edit_op: str | None = None,
    search_prompt: str | None = None,
    mask: Path | None = None,
) -> GenerationResult:
    """Generate a single image through the provider and write to disk.

    Detects output format from response bytes and adjusts file extension.
    Handles platform fitting as a post-processing step (skipped for SVG).
    """
    start = time.time()
    with sentry_sdk.start_span(op="ai.generate", description=f"{provider.name}/{model}") as span:
        span.set_data("provider", provider.name)
        span.set_data("model", model)
        span.set_data("prompt_length", len(prompt))
        gen_params = GenerationParams(
            model=model,
            edit_source=edit_source,
            reference_images=reference_images or [],
            size=size,
            quality=quality,
            aspect_ratio=aspect_ratio,
            seed=seed,
            edit_op=edit_op,
            search_prompt=search_prompt,
            mask=mask,
        )
        img_bytes = provider.generate(prompt, params=gen_params)
        span.set_data("response_bytes", len(img_bytes))

    # Detect actual format and fix extension if needed
    detected = _detect_format(img_bytes)
    output_path = _fix_output_extension(output_path, detected, explicit_path=explicit_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(img_bytes)

    # Platform fitting only applies to raster images
    if platform and detected != "svg":
        apply_platform_fit(output_path, platform)

    return GenerationResult(
        output_path=output_path,
        prompt=prompt,
        model=model,
        preset=None,
        platform=None,
        edit_source=str(edit_source) if edit_source else None,
        timestamp=datetime.now().isoformat(),
        duration_s=round(time.time() - start, 2),
        reference=[str(r) for r in (reference_images or [])],
    )


def _validate_references(args: Any) -> list[Path]:
    """Validate and resolve all reference image paths (regular, style, object)."""
    reference_paths: list[Path] = []
    for ref in getattr(args, "reference", None) or []:
        p = Path(ref).expanduser().resolve()
        if not p.exists():
            raise PermanentAPIError(f"Error: reference image not found: {p}")
        reference_paths.append(p)
    for ref in getattr(args, "style_refs", None) or []:
        p = Path(ref).expanduser().resolve()
        if not p.exists():
            raise PermanentAPIError(f"Error: style-ref image not found: {p}")
        reference_paths.append(p)
    for ref in getattr(args, "object_refs", None) or []:
        p = Path(ref).expanduser().resolve()
        if not p.exists():
            raise PermanentAPIError(f"Error: insert-object image not found: {p}")
        reference_paths.append(p)
    return reference_paths


# Edit operations/models that don't require a prompt
_NO_PROMPT_OPS = {
    "remove-bg", "upscale-fast", "upscale-conservative", "upscale-creative", "erase",
}
_NO_PROMPT_MODELS = {"bria-remove-bg", "bria-erase-fg"}


def _compose_generation_prompt(
    args: Any,
    config: dict[str, Any],
    presets: dict[str, Any],
    edit_source: Path | None,
) -> tuple[str, str | None]:
    """Handle prompt composition with preset and semantic hints.

    Returns (composed_prompt, preset_name).
    """
    style_refs = getattr(args, "style_refs", None) or []
    object_refs = getattr(args, "object_refs", None) or []
    preset_name = args.preset or config.get("default_preset")

    if edit_source:
        edit_op_val = getattr(args, "edit_op", None)
        model_name = getattr(args, "model", None) or ""
        needs_prompt = edit_op_val not in _NO_PROMPT_OPS and model_name not in _NO_PROMPT_MODELS
        if not args.prompt and needs_prompt:
            raise PermanentAPIError("Error: edit mode requires a prompt (the instruction)")
        prompt = args.prompt or ""
        if preset_name:
            prompt = compose_prompt(prompt, preset_name, presets)
    else:
        if not args.prompt:
            raise PermanentAPIError("Error: prompt is required")
        prompt = compose_prompt(args.prompt, preset_name, presets)

    # Prepend a saved style description (from `--style <name>`) if present.
    style_description = getattr(args, "style", None)
    if style_description:
        prompt = f"{style_description}, {prompt}"

    # Prepend semantic context for reference images
    if style_refs:
        prompt = f"Use the provided image(s) as a style reference. {prompt}"
    if object_refs:
        prompt = (
            f"Include the object shown in the provided reference image(s) in the scene. {prompt}"
        )
    return prompt, preset_name


def _resolve_platform(
    args: Any, config: dict[str, Any], platforms: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve platform configuration from args/config. Returns (platform_conf, platform_name)."""
    platform_name = args.platform or config.get("default_platform")
    if not platform_name:
        return None, None
    if platform_name not in platforms:
        raise PermanentAPIError(
            f"Error: platform '{platform_name}' not found. Available: {list(platforms.keys())}"
        )
    return platforms[platform_name], platform_name


def _resolve_generation_model(
    args: Any, config: dict[str, Any], provider: Provider
) -> tuple[Provider, str]:
    """Resolve model alias with registry fallback. Returns (provider, full_model_id)."""
    model_alias = args.model or config.get("default_model", provider.default_model)
    model = provider.resolve_model(model_alias)
    if (
        model == model_alias
        and model_alias not in provider.MODELS
        and model_alias not in provider.MODELS.values()
    ):
        return _route_via_registry(model_alias, provider, config)
    return provider, model


def _route_via_registry(
    model_alias: str, provider: Provider, config: dict[str, Any]
) -> tuple[Provider, str]:
    """Resolve a model alias unknown to `provider` against the global registry.

    Returns the (provider, full_model_id) that actually serves the alias. If the
    alias is unknown to every registered provider, returns the passed provider and
    alias unchanged so the provider can surface its own informative error.
    """
    try:
        preference = load_settings().provider_preference
        prov_name, full_model_id = resolve_model(model_alias, preference)
    except KeyError:
        return provider, model_alias

    if prov_name == provider.name:
        return provider, full_model_id

    log.info("auto-routed to provider via registry", model=model_alias, provider=prov_name)
    return get_provider(prov_name, **build_provider_kwargs(prov_name, config)), full_model_id


def _parse_seed(args: Any) -> int | None:
    """Parse seed from args, coercing string to int if needed."""
    seed_val: int | None = getattr(args, "seed", None)
    if seed_val is not None and not isinstance(seed_val, int):
        try:  # type: ignore[unreachable]
            seed_val = int(seed_val)
        except (ValueError, TypeError):
            seed_val = None
    return seed_val


def _parse_contact_bg_color(contact_bg: str | None) -> tuple[int, int, int]:
    """Parse contact sheet background color from '#rrggbb' or 'r,g,b' format."""
    if not contact_bg:
        return (15, 15, 15)
    try:
        if contact_bg.startswith("#"):
            h = contact_bg.lstrip("#")
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        if "," in contact_bg:
            r, g, b = (int(v.strip()) for v in contact_bg.split(",", 2))
            return (r, g, b)
    except (ValueError, IndexError):
        pass
    return (15, 15, 15)


def _write_variant_contact_sheet(
    outputs: list[GenerationResult],
    primary_output: Path,
    contact_cols: int,
    contact_cell_width: int,
    contact_bg_color: tuple[int, int, int],
    badges: bool,
    badge_radius: int,
) -> None:
    """Compose the multi-variant contact sheet and print its summary line."""
    contact_path = primary_output.with_name(f"{primary_output.stem}-contact.png")
    with sentry_sdk.start_span(op="image.contact_sheet", description=f"{len(outputs)} variants"):
        make_contact_sheet(
            [o.output_path for o in outputs],
            contact_path,
            cols=contact_cols,
            cell_width=contact_cell_width,
            bg_color=contact_bg_color,
            badges=badges,
            badge_radius=badge_radius,
        )
    numbered_suffix = ", numbered" if badges else ""
    print(f"Contact sheet: {contact_path} ({len(outputs)} images{numbered_suffix})")


def _record_provenance(
    result: GenerationResult,
    provider_name: str,
    seed: int | None,
    params: dict[str, Any],
    subject: str = "",
) -> str | None:
    """Write a PROV sidecar, embed metadata, and mint a short index for an image.

    ``subject`` is the raw prompt before preset composition; recording it
    alongside the composed ``result.prompt`` makes the (subject, preset) pairing
    explicit in provenance.

    Provenance and indexing are best-effort enrichment: a failure here (e.g. an
    image format that resists EXIF embedding) must never fail the generation
    itself, so any error is logged and reported to Sentry rather than propagated.

    Returns the short index assigned to the image, or None if recording failed.
    """
    extra = {"preset": result.preset, "platform": result.platform}
    record = ProvenanceRecord(
        prompt=result.prompt,
        model=result.model,
        provider=provider_name,
        output_path=str(result.output_path),
        timestamp=result.timestamp,
        seed=seed,
        parameters={**params, **{k: v for k, v in extra.items() if v is not None}},
        subject=subject,
    )
    try:
        write_provenance_sidecar(record, result.output_path, overwrite=True)
        embed_exif_metadata(result.output_path, record)
        return register_index(result.output_path, record.entity_id)
    except Exception as e:
        log.warning("provenance recording failed", path=str(result.output_path), error=str(e))
        sentry_sdk.capture_exception(e)
        return None


def _persist_generation(
    outputs: list[GenerationResult],
    args: Any,
    provider: Provider,
    seed_val: int | None,
    project: str | None,
    size: str | None,
    quality: str | None,
    aspect: str | None,
) -> dict[Path, str | None]:
    """Write per-image sidecar/provenance/history and return each output's short index.

    Skips metadata writes when ``--no-metadata`` is set; those images still appear in
    history but carry no index (mapped to None).
    """
    no_metadata = getattr(args, "no_metadata", False)
    gen_params = {
        k: v
        for k, v in {"size": size, "quality": quality, "aspect": aspect}.items()
        if v is not None
    }
    indices: dict[Path, str | None] = {}
    for result in outputs:
        entry = asdict(result)
        entry["output_path"] = str(result.output_path)
        entry["provider"] = provider.name
        index = None
        if not no_metadata:
            write_sidecar(result.output_path, entry)
            index = _record_provenance(
                result, provider.name, seed_val, gen_params, subject=args.prompt
            )
        indices[result.output_path] = index
        append_history({**entry, "subject": args.prompt, "project": project, "index": index})
    return indices


def _generate_inner(
    args: Any, config: dict[str, Any], provider: Provider
) -> list[GenerationResult]:
    """Inner generation logic — prompt composition, variant generation, metadata."""
    presets: dict[str, Any] = args.presets if hasattr(args, "presets") else {}
    platforms: dict[str, Any] = args.platforms if hasattr(args, "platforms") else {}

    edit_source = Path(args.edit).expanduser().resolve() if args.edit else None
    if edit_source and not edit_source.exists():
        raise PermanentAPIError(f"Error: edit source not found: {edit_source}")

    reference_paths = _validate_references(args)
    prompt, preset_name = _compose_generation_prompt(args, config, presets, edit_source)
    platform_conf, platform_name = _resolve_platform(args, config, platforms)
    provider, model = _resolve_generation_model(args, config, provider)

    # Generation control params
    size: str | None = getattr(args, "size", None) or config.get("default_size")
    quality: str | None = getattr(args, "quality", None) or config.get("default_quality")
    aspect: str | None = getattr(args, "aspect", None) or config.get("default_aspect")
    seed_val = _parse_seed(args)

    # Contact sheet config
    contact_cols: int = int(getattr(args, "contact_cols", None) or config.get("contact_cols", 2))
    contact_cell_width: int = int(
        getattr(args, "contact_cell_width", None) or config.get("contact_cell_width", 600)
    )
    contact_bg: str | None = getattr(args, "contact_bg", None) or config.get("contact_bg")
    contact_bg_color = _parse_contact_bg_color(contact_bg)
    badges: bool = getattr(args, "badges", True)
    badge_radius: int = int(
        getattr(args, "contact_badge_radius", None) or config.get("contact_badge_radius", 30)
    )

    # Dry run
    if args.dry_run:
        print(f"Model: {model}")
        print(f"Preset: {preset_name or '(none)'}")
        print(f"Platform: {platform_name or '(none)'}")
        if size:
            print(f"Size: {size}")
        if quality:
            print(f"Quality: {quality}")
        if aspect:
            print(f"Aspect: {aspect}")
        if seed_val is not None:
            print(f"Seed: {seed_val}")
        if edit_source:
            print(f"Edit source: {edit_source}")
        if reference_paths:
            print(f"References: {', '.join(str(r) for r in reference_paths)}")
        print(f"Prompt:\n  {prompt}")
        return []

    # Output path resolution
    project = args.project or config.get("default_project")
    primary_output = resolve_output_path(args.output, args.prompt, project, config)

    n = max(1, args.n)
    outputs: list[GenerationResult] = []

    def _job(i: int) -> GenerationResult | None:
        if n == 1:
            out = primary_output
        else:
            name = f"{primary_output.stem}-{i + 1:02d}{primary_output.suffix}"
            out = primary_output.with_name(name)
        print(f"[{i + 1}/{n}] Generating → {out.name}")
        edit_op: str | None = getattr(args, "edit_op", None)
        search_prompt_val: str | None = getattr(args, "search_prompt", None)
        mask_path: Path | None = None
        mask_str = getattr(args, "mask", None)
        if mask_str:
            mask_path = Path(mask_str).expanduser().resolve()
            if not mask_path.exists():
                log.error("mask file not found", path=str(mask_path))
                return None

        try:
            result = generate_once(
                prompt, out, provider, model,
                edit_source=edit_source,
                reference_images=reference_paths,
                platform=platform_conf,
                explicit_path=bool(args.output),
                size=size, quality=quality, aspect_ratio=aspect,
                seed=seed_val, edit_op=edit_op,
                search_prompt=search_prompt_val, mask=mask_path,
            )
        except Exception as e:
            log.error("variant generation failed", variant=i + 1, error=str(e))
            return None
        result.preset = preset_name
        result.platform = platform_name
        return result

    if n == 1:
        one = _job(0)
        if one is None:
            raise ImageCreatorError("Generation failed.")
        outputs.append(one)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
            outputs = [r for r in pool.map(_job, range(n)) if r is not None]
        if not outputs:
            raise ImageCreatorError("All variants failed.")

    # Contact sheet for multi-variant runs
    if n > 1:
        _write_variant_contact_sheet(
            outputs,
            primary_output,
            contact_cols,
            contact_cell_width,
            contact_bg_color,
            badges,
            badge_radius,
        )

    # Metadata and history
    indices = _persist_generation(
        outputs, args, provider, seed_val, project, size, quality, aspect
    )

    save_last_run(
        {
            "subject": args.prompt,
            "prompt": prompt,
            "preset": preset_name,
            "platform": platform_name,
            "model": model,
            "edit_source": str(edit_source) if edit_source else None,
            "reference": [str(r) for r in reference_paths],
            "project": project,
            "n": n,
            "output": str(primary_output),
            "size": size,
            "quality": quality,
            "aspect": aspect,
            "seed": seed_val,
        }
    )

    for result in outputs:
        index = indices.get(result.output_path)
        ref = f" [@{index}]" if index else ""
        print(f"✓ {result.output_path}{ref} ({result.duration_s}s)")

    return outputs
