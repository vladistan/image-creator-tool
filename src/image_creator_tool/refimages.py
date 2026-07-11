"""Reference-image store: import external images onto the uniform @index path.

External/web-sourced images enter the tool here. :func:`import_image` copies a file
into ``ref_images_dir``, mints an @index (the same short-index system generated images
use), and writes a lightweight PROV sidecar marking ``origin=imported`` with the source
URL/path — so a final piece's provenance can trace a reference back to where it came from.

De-duplication keys on the image *bytes*, not the URL: the copied file is named by its
content hash, so re-importing the same image (even fetched under a different URL) collapses
onto the existing @index and merely appends the new source to that record's origin list.

:func:`forget_image` and :func:`collect_unreferenced` are the reclaim side — remove a picked
image's copy + sidecar + index entry, or list imported images no retained generation still
references so a caller can prune discovery leftovers.

Domain layer: this module never imports typer and never writes to stdout. CLI concerns
(argument parsing, exit codes, echo) live in ``commands/import_images.py`` and
``commands/forget.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from image_creator_tool import indexer, provenance
from image_creator_tool.errors import PermanentAPIError
from image_creator_tool.provenance import ProvenanceRecord

_IMPORTED_ORIGIN = "imported"


@dataclass
class ImportResult:
    """One source file's import outcome; ``is_duplicate`` tells a fresh mint from a reuse."""

    index: str
    image_path: Path
    is_duplicate: bool
    """True when the bytes were already in the store and the existing @index was reused."""


@dataclass
class ForgetResult:
    """One @index's forget outcome; ``removed`` reports whether it was an imported image."""

    index: str
    image_path: Path
    removed: bool
    """False when the index resolved to a non-imported (generated) image and was left intact."""


def _content_hash(data: bytes) -> str:
    """Return a short hex digest of image bytes, used as the de-dupe key and file stem."""
    return hashlib.sha256(data).hexdigest()[:32]


def import_image(source: str | Path, ref_dir: str | Path, search_dir: str | Path) -> ImportResult:
    """Copy ``source`` into ``ref_dir``, mint an @index, and record imported provenance.

    Idempotent on content: the destination file is named by the bytes' content hash, so a
    repeat import of the same image reuses the existing @index (appending ``source`` to the
    record's origin list) instead of minting a duplicate.

    Raises:
        PermanentAPIError: ``source`` does not exist or is not a file.
    """
    src = Path(source)
    if not src.is_file():
        raise PermanentAPIError(f"Import source not found: {src}")

    ref_path = Path(ref_dir)
    ref_path.mkdir(parents=True, exist_ok=True)

    data = src.read_bytes()
    digest = _content_hash(data)
    dest = ref_path / f"{digest}{src.suffix.lower()}"

    if dest.is_file():
        index = indexer.index_for(dest) or indexer.register_index(dest, key=digest)
        _append_source(dest, str(src))
        return ImportResult(index=index, image_path=dest, is_duplicate=True)

    dest.write_bytes(data)
    index = indexer.register_index(dest, key=digest)
    record = ProvenanceRecord(
        prompt="",
        model="",
        provider=_IMPORTED_ORIGIN,
        output_path=str(dest),
        timestamp=datetime.now().astimezone().isoformat(),
        origin=_IMPORTED_ORIGIN,
        sources=[str(src)],
    )
    provenance.write_provenance_sidecar(record, dest, overwrite=True)
    return ImportResult(index=index, image_path=dest, is_duplicate=False)


def _append_source(image_path: Path, source: str) -> None:
    """Add ``source`` to an imported image's origin list, de-duplicating URLs/paths."""
    sidecar = provenance.sidecar_path_for(image_path)
    if not sidecar.is_file():
        return
    record = provenance.load_record(sidecar)
    if source in record.sources:
        return
    record.sources.append(source)
    provenance.write_provenance_sidecar(record, image_path, overwrite=True)


def forget_image(index: str, search_dir: str | Path) -> ForgetResult:
    """Remove the imported image at ``index``: its file, PROV sidecar, and index entry.

    Only imported images (``origin=imported``) are removed; a generated image is left fully
    intact and returned with ``removed=False`` so the caller can report the refusal. Removes
    the index-map entry last so a partial failure still leaves the entry resolvable.

    Raises:
        PermanentAPIError: ``index`` is not registered under ``search_dir``.
    """
    image_path = indexer.resolve_index(index, search_dir)
    normalized = index.lstrip("@").upper()

    if not _is_imported(image_path):
        return ForgetResult(index=normalized, image_path=image_path, removed=False)

    sidecar = provenance.sidecar_path_for(image_path)
    sidecar.unlink(missing_ok=True)
    image_path.unlink(missing_ok=True)
    indexer.deregister_index(index, search_dir)
    return ForgetResult(index=normalized, image_path=image_path, removed=True)


def collect_unreferenced(ref_dir: str | Path, search_dir: str | Path) -> list[tuple[str, Path]]:
    """Return ``(index, path)`` for imported images no retained generation references.

    An imported image is *referenced* when a generated image's provenance lists its path in
    ``parameters['reference']`` (populated when it was passed via ``--ref``/``--style-ref``/
    ``--insert-object``). Everything imported but never referenced is a prune candidate — the
    discovery leftovers a caller imported to compare and then did not pick.
    """
    referenced = _referenced_paths(search_dir)
    candidates: list[tuple[str, Path]] = []
    for index, image_path in indexer.list_index_entries(ref_dir):
        if not _is_imported(image_path):
            continue
        if str(image_path) in referenced or image_path.name in referenced:
            continue
        candidates.append((index, image_path))
    return candidates


def _referenced_paths(search_dir: str | Path) -> set[str]:
    """Collect every reference-image path (and its bare name) cited by generated records.

    Walks ``search_dir`` recursively (generated records live in per-project subdirectories,
    which the non-recursive ``scan_provenance`` would miss) so a referenced import is not
    misreported as prunable.
    """
    referenced: set[str] = set()
    for sidecar in Path(search_dir).rglob(f"*{provenance.SIDECAR_SUFFIX}"):
        record = provenance.load_record(sidecar)
        if record.origin == _IMPORTED_ORIGIN:
            continue
        for ref in record.parameters.get("reference", []):
            referenced.add(str(ref))
            referenced.add(Path(str(ref)).name)
    return referenced


def _is_imported(image_path: Path) -> bool:
    """True when the image carries an ``origin=imported`` provenance sidecar."""
    sidecar = provenance.sidecar_path_for(image_path)
    if not sidecar.is_file():
        return False
    return provenance.load_record(sidecar).origin == _IMPORTED_ORIGIN
