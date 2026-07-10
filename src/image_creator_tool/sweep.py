"""Parameter sweep engine for cross-product image generation.

Detects comma-separated values in any supported dimension (model, preset,
platform, seed) and generates the full cross-product grid, assembling results
into labeled contact sheets.

Layout rules:
- First two sweep dimensions → rows x cols
- Additional dimensions → separate sheet per combination
- `--n > 1` (when other dims present) → variant columns appended

Usage example (called from generation.py):
    from image_creator_tool.sweep import is_sweep, run_sweep
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path  # noqa: TC003
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import structlog

from image_creator_tool.config import build_provider_kwargs, load_settings
from image_creator_tool.errors import ImageCreatorError
from image_creator_tool.generation_core import GenerationResult, _generate_inner
from image_creator_tool.history import resolve_output_path
from image_creator_tool.imaging import make_labeled_contact_sheet
from image_creator_tool.indexer import index_for
from image_creator_tool.providers import get_provider
from image_creator_tool.registry import resolve_model

if TYPE_CHECKING:
    from image_creator_tool.providers.base import Provider

log = structlog.get_logger()


@dataclass
class SweepDimension:
    """One axis of the sweep grid."""

    name: str
    values: list[str]


@dataclass
class SweepCell:
    """One (params, label) cell in the sweep grid, populated after generation."""

    params: dict[str, Any]
    label: str
    result: GenerationResult | None = field(default=None)


def _parse_dims(args: Any) -> list[SweepDimension]:
    """Extract all sweep dimensions from generation args.

    Only dimensions with comma-separated values (or n>1 when other dims exist)
    are included. Preserves declaration order; n-variants appended last.
    """
    dims: list[SweepDimension] = []

    for attr, name in (
        ("model", "model"),
        ("preset", "preset"),
        ("platform", "platform"),
        ("seed", "seed"),
    ):
        val = getattr(args, attr, None)
        if val and isinstance(val, str) and "," in val:
            dims.append(SweepDimension(name=name, values=[v.strip() for v in val.split(",")]))

    # n > 1 is only a sweep dimension when there are other sweep dims
    n_val = max(1, getattr(args, "n", 1) or 1)
    if n_val > 1 and dims:
        dims.append(SweepDimension(name="variant", values=[str(i + 1) for i in range(n_val)]))

    return dims


def is_sweep(args: Any) -> bool:
    """Return True if args describe a sweep (any dimension has comma-separated values)."""
    return bool(_parse_dims(args))


def _make_cell_label(combo: dict[str, str]) -> str:
    """Join dimension values with ' / ' for use as a contact-sheet cell label."""
    return " / ".join(str(v) for v in combo.values())


def _apply_combo(base_args: Any, combo: dict[str, str]) -> SimpleNamespace:
    """Override base_args fields with a single sweep-dimension combination.

    Variant entries are intentionally ignored — each variant cell uses the same
    model/preset/platform as its row/col position.
    """
    overrides: dict[str, Any] = {
        "model": getattr(base_args, "model", None),
        "preset": getattr(base_args, "preset", None),
        "platform": getattr(base_args, "platform", None),
        "seed": getattr(base_args, "seed", None),
        "n": 1,  # variants are already unrolled into individual cells
    }
    for dim_name, val in combo.items():
        if dim_name == "variant":
            # variant = independent run (same params, different seed offset)
            pass
        else:
            overrides[dim_name] = val

    return SimpleNamespace(
        prompt=base_args.prompt,
        output=None,
        preset=overrides["preset"],
        platform=overrides["platform"],
        model=overrides["model"],
        edit=getattr(base_args, "edit", None),
        reference=getattr(base_args, "reference", []),
        style_refs=getattr(base_args, "style_refs", []),
        object_refs=getattr(base_args, "object_refs", []),
        project=getattr(base_args, "project", None),
        n=1,
        dry_run=getattr(base_args, "dry_run", False),
        no_metadata=getattr(base_args, "no_metadata", False),
        presets=getattr(base_args, "presets", {}),
        platforms=getattr(base_args, "platforms", {}),
        seed=overrides["seed"],
        size=getattr(base_args, "size", None),
        quality=getattr(base_args, "quality", None),
        aspect=getattr(base_args, "aspect", None),
        # Carry through prompt-shaping / edit inputs that _generate_inner reads;
        # omitting these silently drops --style (and edit ops) from every cell.
        style=getattr(base_args, "style", None),
        edit_op=getattr(base_args, "edit_op", None),
        search_prompt=getattr(base_args, "search_prompt", None),
        mask=getattr(base_args, "mask", None),
    )


def run_sweep(
    args: Any,
    config: dict[str, Any],
    provider: Provider,
) -> list[GenerationResult]:
    """Execute the full parameter sweep and assemble contact sheet(s).

    Returns the list of all successful GenerationResult objects.
    """
    dims = _parse_dims(args)
    if not dims:
        return _generate_inner(args, config, provider)

    # Layout: first dim = rows, second = cols, rest = sheet-level dims
    row_dim = dims[0]
    col_dim = dims[1] if len(dims) >= 2 else None  # noqa: PLR2004
    sheet_dims = dims[2:] if len(dims) > 2 else []  # noqa: PLR2004

    # Build sheet-level combinations (outer loop)
    if sheet_dims:
        sheet_combos: list[dict[str, str]] = [
            dict(zip([d.name for d in sheet_dims], vals, strict=True))
            for vals in product(*[d.values for d in sheet_dims])
        ]
    else:
        sheet_combos = [{}]

    # Build row x col grid combinations (inner loop)
    if col_dim:
        grid_combos: list[dict[str, str]] = [
            {row_dim.name: rv, col_dim.name: cv}
            for rv in row_dim.values
            for cv in col_dim.values
        ]
        n_cols = len(col_dim.values)
    else:
        grid_combos = [{row_dim.name: rv} for rv in row_dim.values]
        n_cols = len(row_dim.values)

    all_results: list[GenerationResult] = []

    # Contact-sheet badge config (threaded from CLI, mirroring generation_core).
    badges: bool = getattr(args, "badges", True)
    badge_radius: int = int(
        getattr(args, "contact_badge_radius", None) or config.get("contact_badge_radius", 30)
    )

    project = args.project or config.get("default_project")
    base_output = resolve_output_path(None, args.prompt, project, config)

    total_sheets = len(sheet_combos)
    total_cells = len(grid_combos)

    if args.dry_run:
        print(
            f"Parameter sweep: {len(dims)} dimension(s), "
            f"{total_sheets} sheet(s), {total_cells} cell(s) per sheet"
        )
        for i, dim in enumerate(dims):
            role = "rows" if i == 0 else ("cols" if i == 1 else "sheets")
            print(f"  [{role}] {dim.name}: {', '.join(dim.values)}")
        print(f"Total generations: {total_sheets * total_cells}")
        return []

    for sheet_idx, sheet_combo in enumerate(sheet_combos):
        sheet_label = _make_cell_label(sheet_combo) if sheet_combo else ""

        cells_for_sheet: list[tuple[Path, str]] = []

        for cell_idx, grid_combo in enumerate(grid_combos):
            full_combo = {**sheet_combo, **grid_combo}
            cell_args = _apply_combo(args, full_combo)
            cell_label = _make_cell_label(grid_combo)

            # Build unique output path for this cell
            label_slug = cell_label.replace(" / ", "_").replace(" ", "-")
            sheet_slug = sheet_label.replace(" / ", "_").replace(" ", "-") if sheet_label else ""
            suffix_parts = [p for p in [sheet_slug, label_slug] if p]
            suffix = ("-" + "-".join(suffix_parts)) if suffix_parts else f"-{cell_idx + 1:02d}"
            cell_out = base_output.with_name(f"{base_output.stem}{suffix}{base_output.suffix}")

            print(
                f"[{sheet_idx + 1}/{total_sheets}]"
                f"[{cell_idx + 1}/{total_cells}] "
                f"{cell_label} → {cell_out.name}"
            )

            # Resolve provider for this cell's model
            cell_model = cell_args.model or config.get("default_model") or provider.default_model
            settings_pref: list[str] = []
            try:
                settings_pref = load_settings().provider_preference
            except Exception as exc:
                # Degraded operation: proceed without preference ordering
                log.warning("could not load provider preference", error=str(exc))

            cell_provider = provider
            full_model_id = cell_model
            try:
                prov_name, full_model_id = resolve_model(cell_model, settings_pref)
                cell_provider = get_provider(prov_name, **build_provider_kwargs(prov_name, config))
            except KeyError:
                # Unknown model alias — fall back to the provider passed on the CLI
                # (the -p value), keeping the original model name for passthrough.
                log.warning(
                    "unknown model alias in sweep, using the provider passed on the command line",
                    model=cell_model,
                )
                full_model_id = cell_model

            cell_args.model = full_model_id

            try:
                results = _generate_inner(cell_args, config, cell_provider)
                if results:
                    all_results.extend(results)
                    # Annotate the cell with the short index minted during
                    # _generate_inner's provenance step, when one was assigned.
                    cell_index = index_for(results[0].output_path)
                    sheet_label_text = f"{cell_label}  @{cell_index}" if cell_index else cell_label
                    cells_for_sheet.append((results[0].output_path, sheet_label_text))
            except Exception as e:
                log.error("sweep cell failed", combo=full_combo, error=str(e))
                print(f"  ✗ {cell_label}: {e}")

        if not cells_for_sheet:
            print(f"  ✗ No successful results for sheet {sheet_idx + 1}")
            continue

        # Assemble contact sheet for this sheet combo
        sheet_name_parts = [base_output.stem, "sweep"]
        if sheet_label:
            sheet_name_parts.append(sheet_label.replace(" / ", "_").replace(" ", "-"))
        if total_sheets > 1:
            sheet_name_parts.append(f"s{sheet_idx + 1}")
        contact_name = "-".join(sheet_name_parts) + ".png"
        contact_path = base_output.with_name(contact_name)

        title = args.prompt
        if sheet_label:
            title = f"{args.prompt} [{sheet_label}]"

        make_labeled_contact_sheet(
            cells_for_sheet,
            contact_path,
            cols=n_cols,
            title=title,
            badges=badges,
            badge_radius=badge_radius,
        )
        numbered_suffix = ", numbered" if badges else ""
        print(f"Sweep sheet: {contact_path} ({len(cells_for_sheet)} images{numbered_suffix})")

    if not all_results:
        raise ImageCreatorError("All sweep cells failed.")

    return all_results
