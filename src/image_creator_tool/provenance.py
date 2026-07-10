"""W3C PROV provenance records for generated images.

Every generated image can carry a machine-readable provenance record describing
how it was made: the prompt, model, provider, seed, generation parameters, and
timestamp. Records serialize to the W3C PROV-JSON / PROV-N interchange formats
via the `prov` library, are written as `<image-stem>.prov.json` sidecar files,
and are optionally embedded into the image itself (JPEG EXIF or PNG tEXt chunks).

These records are the source data consumed by the Phase 5 short-index companion.

This module is the domain layer — it never imports typer. CLI concerns live in
`commands/prov.py`, which converts the domain exceptions raised here into exit
codes.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog
from PIL import Image, PngImagePlugin
from prov.model import ProvDocument

from image_creator_tool.errors import PermanentAPIError

if TYPE_CHECKING:
    from types import ModuleType

log = structlog.get_logger()

# W3C PROV namespace for this tool's domain terms.
PROV_NAMESPACE_URI = "https://github.com/vladistan/image-creator-tool/ns#"
PROV_NAMESPACE_PREFIX = "imgc"

_SIDECAR_SUFFIX = ".prov.json"

# Stable UUID namespace so entity/activity IDs are deterministic across runs.
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, PROV_NAMESPACE_URI)


@dataclass
class ProvenanceRecord:
    """A W3C PROV-compliant description of a single image generation.

    Fields map onto PROV concepts: the image is an *entity*, the generation is an
    *activity* bounded by `timestamp`, and the provider+model pair is the
    *software agent* that produced it.
    """

    prompt: str
    model: str
    provider: str
    output_path: str
    timestamp: str
    seed: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    subject: str = ""
    """Raw prompt before preset composition; empty when no preset was applied."""

    @property
    def entity_id(self) -> str:
        """Qualified PROV entity ID for the generated image.

        Derived from the output path *and* the timestamp so that regenerating to
        the same path yields a distinct provenance entity rather than colliding.
        """
        raw = f"{self.output_path}@{self.timestamp}"
        return f"{PROV_NAMESPACE_PREFIX}:image-{uuid.uuid5(_ID_NAMESPACE, raw)}"

    @property
    def activity_id(self) -> str:
        """Activity ID for the generation event, derived from the entity ID.

        Deriving it from the (timestamp-sensitive) entity ID keeps every serialization
        of one record pointing at the same activity while still distinguishing separate
        regenerations to the same path.
        """
        return f"{PROV_NAMESPACE_PREFIX}:generation-{uuid.uuid5(_ID_NAMESPACE, self.entity_id)}"

    @property
    def agent_id(self) -> str:
        """Agent ID keyed on provider+model, so identical generators collapse to one agent.

        Unlike the entity/activity IDs this deliberately omits the timestamp: two images
        from the same provider/model share a single PROV agent across records.
        """
        return f"{PROV_NAMESPACE_PREFIX}:{_slug(self.provider)}-{_slug(self.model)}"

    def to_prov_document(self) -> ProvDocument:
        """Build a PROV document capturing entity, activity, agent, and relations."""
        doc = ProvDocument()
        doc.add_namespace(PROV_NAMESPACE_PREFIX, PROV_NAMESPACE_URI)

        moment = _parse_iso(self.timestamp)
        agent_attrs: dict[str, Any] = {
            "prov:type": "prov:SoftwareAgent",
            f"{PROV_NAMESPACE_PREFIX}:provider": self.provider,
            f"{PROV_NAMESPACE_PREFIX}:model": self.model,
        }
        # `.items()` (Iterable[tuple[str, Any]]) satisfies prov's qualified-name arg type.
        entity = doc.entity(self.entity_id, list(self._entity_attributes().items()))
        activity = doc.activity(self.activity_id, moment, moment)
        agent = doc.agent(self.agent_id, list(agent_attrs.items()))
        doc.wasGeneratedBy(entity, activity, moment)
        doc.wasAssociatedWith(activity, agent)
        doc.wasAttributedTo(entity, agent)
        return doc

    def to_prov_json(self) -> str:
        """Render PROV-JSON — the machine-readable form written to the `.prov.json` sidecar."""
        return str(self.to_prov_document().serialize(format="json"))

    def to_prov_n(self) -> str:
        """Render PROV-N — the human-readable notation shown by `prov show`/`prov export`."""
        return str(self.to_prov_document().serialize(format="provn"))

    def _entity_attributes(self) -> dict[str, Any]:
        """Build the entity's PROV attributes, omitting seed/parameters when unset.

        Absent optional fields are dropped rather than encoded as null so the sidecar
        distinguishes "no seed recorded" from "seed was zero/empty".
        """
        attrs: dict[str, Any] = {
            "prov:type": f"{PROV_NAMESPACE_PREFIX}:GeneratedImage",
            f"{PROV_NAMESPACE_PREFIX}:prompt": self.prompt,
            f"{PROV_NAMESPACE_PREFIX}:model": self.model,
            f"{PROV_NAMESPACE_PREFIX}:provider": self.provider,
            f"{PROV_NAMESPACE_PREFIX}:output_path": self.output_path,
            f"{PROV_NAMESPACE_PREFIX}:timestamp": self.timestamp,
        }
        if self.subject:
            attrs[f"{PROV_NAMESPACE_PREFIX}:subject"] = self.subject
        if self.seed is not None:
            attrs[f"{PROV_NAMESPACE_PREFIX}:seed"] = self.seed
        if self.parameters:
            attrs[f"{PROV_NAMESPACE_PREFIX}:parameters"] = json.dumps(
                self.parameters, sort_keys=True, default=str
            )
        return attrs


# --- Sidecar writer (Step 4.2) ----------------------------------------------


def sidecar_path_for(image_path: str | Path) -> Path:
    """Return the `<image-stem>.prov.json` sidecar path for an image."""
    path = Path(image_path)
    return path.with_name(path.stem + _SIDECAR_SUFFIX)


def write_provenance_sidecar(
    record: ProvenanceRecord, image_path: str | Path, *, overwrite: bool = False
) -> Path:
    """Write a PROV-JSON sidecar next to `image_path`.

    Raises:
        PermanentAPIError: A sidecar already exists and `overwrite` is False.
    """
    sidecar = sidecar_path_for(image_path)
    if sidecar.exists() and not overwrite:
        raise PermanentAPIError(
            f"Provenance sidecar already exists: {sidecar}. Pass overwrite=True to replace it."
        )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(record.to_prov_json())
    return sidecar


def load_record(sidecar_path: str | Path) -> ProvenanceRecord:
    """Reconstruct a `ProvenanceRecord` from a PROV-JSON sidecar.

    Raises:
        PermanentAPIError: The sidecar is missing or has no PROV entity.
    """
    path = Path(sidecar_path)
    if not path.is_file():
        raise PermanentAPIError(f"Provenance sidecar not found: {path}")

    raw = json.loads(path.read_text())
    entities = raw.get("entity", {})
    if not entities:
        raise PermanentAPIError(f"No provenance entity in sidecar: {path}")

    attrs = next(iter(entities.values()))
    seed_raw = _attr(attrs, "seed")
    params_raw = _attr(attrs, "parameters")
    return ProvenanceRecord(
        prompt=_attr(attrs, "prompt") or "",
        model=_attr(attrs, "model") or "",
        provider=_attr(attrs, "provider") or "",
        output_path=_attr(attrs, "output_path") or "",
        timestamp=_attr(attrs, "timestamp") or "",
        seed=int(seed_raw) if seed_raw is not None else None,
        parameters=json.loads(params_raw) if params_raw else {},
        subject=_attr(attrs, "subject") or "",
    )


def scan_provenance(
    output_dir: str | Path,
    *,
    date: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> list[tuple[Path, ProvenanceRecord]]:
    """Load every provenance sidecar under `output_dir`, applying optional filters.

    Args:
        output_dir: Directory to scan for `*.prov.json` sidecars (non-recursive).
        date: Keep records whose ISO timestamp starts with this `YYYY-MM-DD`.
        model: Keep records whose model equals this value.
        provider: Keep records whose provider equals this value.

    Returns:
        `(sidecar_path, record)` pairs sorted by timestamp.
    """
    directory = Path(output_dir)
    if not directory.is_dir():
        return []

    matches: list[tuple[Path, ProvenanceRecord]] = []
    for sidecar in sorted(directory.glob(f"*{_SIDECAR_SUFFIX}")):
        record = load_record(sidecar)
        if date and not record.timestamp.startswith(date):
            continue
        if model and record.model != model:
            continue
        if provider and record.provider != provider:
            continue
        matches.append((sidecar, record))
    matches.sort(key=lambda pair: pair[1].timestamp)
    return matches


def export_prov_n(output_dir: str | Path) -> str:
    """Merge every provenance sidecar under `output_dir` into one PROV-N document."""
    combined = ProvDocument()
    combined.add_namespace(PROV_NAMESPACE_PREFIX, PROV_NAMESPACE_URI)
    for sidecar, _record in scan_provenance(output_dir):
        doc = ProvDocument.deserialize(content=sidecar.read_text(), format="json")
        combined.update(doc)
    return str(combined.serialize(format="provn"))


# --- EXIF / tEXt embedding (Step 4.3) ---------------------------------------


def embed_exif_metadata(image_path: str | Path, record: ProvenanceRecord) -> bool:
    """Embed provenance into the image's own metadata (best effort).

    JPEG images use EXIF via the optional `piexif` dependency; PNG images use
    tEXt chunks via Pillow. Returns True when metadata was embedded, False when
    skipped (unsupported format or `piexif` not installed).
    """
    path = Path(image_path)
    suffix = path.suffix.lower()
    software = f"{record.provider}/{record.model}"

    if suffix in {".jpg", ".jpeg"}:
        return _embed_jpeg_exif(path, record.prompt, software)
    if suffix == ".png":
        return _embed_png_text(path, record, software)

    log.info("exif embedding skipped: unsupported format", path=str(path), suffix=suffix)
    return False


def _embed_jpeg_exif(path: Path, description: str, software: str) -> bool:
    """Write ImageDescription/Software EXIF tags into a JPEG via piexif."""
    piexif = _load_piexif()
    if piexif is None:
        log.info("piexif not installed; skipping JPEG EXIF embedding", path=str(path))
        return False

    exif_dict = piexif.load(str(path))
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.Software] = software.encode("utf-8")
    piexif.insert(piexif.dump(exif_dict), str(path))
    return True


def _embed_png_text(path: Path, record: ProvenanceRecord, software: str) -> bool:
    """Write provenance into PNG tEXt chunks via Pillow."""
    image = Image.open(path)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Description", record.prompt)
    meta.add_text("Software", software)
    meta.add_text("provider", record.provider)
    meta.add_text("model", record.model)
    image.save(path, pnginfo=meta)
    return True


def _load_piexif() -> ModuleType | None:
    """Import `piexif` lazily so JPEG EXIF embedding stays an opt-in feature.

    `piexif` lives in the `exif` optional-dependency group; deferring the import to
    call time lets the tool run (and PNG embedding work) without it, returning None so
    callers can no-op gracefully rather than crashing on a missing import.
    """
    try:
        import piexif  # noqa: PLC0415
    except ImportError:
        return None
    return cast("ModuleType", piexif)


# --- Helpers ----------------------------------------------------------------


def _attr(attrs: dict[str, Any], key: str) -> Any:
    """Read an `imgc:<key>` PROV-JSON attribute, unwrapping typed literals."""
    value = attrs.get(f"{PROV_NAMESPACE_PREFIX}:{key}")
    if isinstance(value, dict):
        return value.get("$")
    return value


def _parse_iso(timestamp: str) -> datetime:
    """Parse the record's ISO-8601 timestamp into the datetime PROV needs.

    The timestamp bounds the generation activity, so a malformed value cannot be
    silently defaulted away: it would produce a provenance record that misrepresents
    when the image was created. Surface it as a domain error instead.
    """
    try:
        return datetime.fromisoformat(timestamp)
    except ValueError as exc:
        msg = f"Provenance record has an invalid timestamp: {timestamp!r}"
        raise PermanentAPIError(msg) from exc


def _slug(text: str) -> str:
    """Lowercase to an alphanumeric-with-hyphens slug for use in a PROV local name."""
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-") or "unknown"
