"""Short alphanumeric indices for quick reference to generated images.

Every generated image can be tagged with a compact identifier (e.g. ``IISDSXS3``)
so follow-up commands can refer to it as ``@IISDSXS3`` instead of a long
timestamped path. The index is the Phase 5 companion to the Phase 4 PROV records:
it is derived deterministically from the image's PROV entity id and persisted in a
per-directory index file so it survives across sessions.

Schema
------
* **Alphabet**: RFC 4648 Base32 (``A`` to ``Z`` and ``2`` to ``7``) - uppercase, no
  ``0/1/8/9`` so the identifier is unambiguous when read aloud or transcribed.
* **Length**: :data:`INDEX_LENGTH` characters by default (extended only on the rare
  in-directory collision).
* **Derivation**: ``base32(sha256(key))[:length]`` where ``key`` is the PROV entity
  id of the image. Deriving from the entity id — itself keyed on ``output_path`` +
  ``timestamp`` — makes the index deterministic and one-to-one with a provenance
  record while still distinguishing regenerations to the same path.
* **Uniqueness**: guaranteed *within an output directory*. The mapping is stored in
  a :data:`INDEX_FILE_NAME` file next to the images; on the (hash-)improbable event
  of two distinct images colliding, the newcomer's index is lengthened until unique.

This module is the domain layer — it never imports typer and never logs to stdout
(its return values feed the ``lookup`` command's output). CLI concerns live in
``commands/lookup.py``; provenance metadata is joined in there, not here.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from image_creator_tool.errors import PermanentAPIError

# Per-directory file mapping short index -> image filename (relative to the dir).
INDEX_FILE_NAME = ".image-index.json"

# Default identifier length in Base32 characters (40 bits of the digest).
INDEX_LENGTH = 8

# Longest identifier we will grow to while resolving a collision before giving up.
_MAX_INDEX_LENGTH = 52  # sha256 is 256 bits => 52 Base32 chars without padding


def compute_index(key: str, length: int = INDEX_LENGTH) -> str:
    """Derive a Base32 short index of ``length`` chars from an arbitrary key string.

    The same key always yields the same index, so callers key on the PROV entity id
    to bind the index to a provenance record.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    encoded = base64.b32encode(digest).decode("ascii").rstrip("=")
    return encoded[:length]


def index_file_for(output_dir: str | Path) -> Path:
    """Locate the per-directory `.image-index.json` that maps indices to filenames."""
    return Path(output_dir) / INDEX_FILE_NAME


def load_index_map(output_dir: str | Path) -> dict[str, str]:
    """Load the ``index -> filename`` map for a directory (empty if none)."""
    path = index_file_for(output_dir)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    # A wrong-shape file (e.g. a JSON list) is treated as empty rather than
    # crashing generation; genuinely malformed JSON still surfaces from json.loads.
    return data if isinstance(data, dict) else {}


def index_for(image_path: str | Path) -> str | None:
    """Return the short index registered for ``image_path``, or None if unindexed.

    Reverse lookup within the image's own directory index; used to annotate
    contact-sheet cells with the index minted for each thumbnail.
    """
    path = Path(image_path)
    for index, filename in load_index_map(path.parent).items():
        if filename == path.name:
            return index
    return None


def register_index(image_path: str | Path, key: str) -> str:
    """Assign and persist a short index for ``image_path``, keyed on ``key``.

    Idempotent: re-registering an already-indexed image returns its existing index
    rather than minting a new one. On the improbable collision with a *different*
    image already holding the derived index, the new index is lengthened until unique.

    Returns the short index string.
    """
    path = Path(image_path)
    directory = path.parent
    filename = path.name
    mapping = load_index_map(directory)

    for existing_index, existing_name in mapping.items():
        if existing_name == filename:
            return existing_index

    index = compute_index(key)
    length = INDEX_LENGTH
    while index in mapping and mapping[index] != filename and length < _MAX_INDEX_LENGTH:
        length += 1
        index = compute_index(key, length)

    mapping[index] = filename
    directory.mkdir(parents=True, exist_ok=True)
    index_file_for(directory).write_text(json.dumps(mapping, indent=2, sort_keys=True))
    return index


def deregister_index(index: str, search_dir: str | Path) -> Path | None:
    """Remove ``index`` from whichever per-directory map holds it; return its image path.

    Scans ``search_dir`` recursively for the index file containing ``index`` (matching
    :func:`resolve_index`'s lookup), drops that entry, and rewrites the map. Returns the
    image path that was mapped, or ``None`` when the index is not registered anywhere.
    Used by ``forget`` to retire an imported image's registration alongside its file.
    """
    normalized = index.lstrip("@").upper()
    for index_file in Path(search_dir).rglob(INDEX_FILE_NAME):
        mapping = load_index_map(index_file.parent)
        if normalized in mapping:
            image_path = index_file.parent / mapping.pop(normalized)
            index_file.write_text(json.dumps(mapping, indent=2, sort_keys=True))
            return image_path
    return None


def resolve_index(index: str, search_dir: str | Path) -> Path:
    """Resolve a short index to its image path by scanning ``search_dir`` recursively.

    Raises:
        PermanentAPIError: No index file under ``search_dir`` contains ``index``.
    """
    normalized = index.lstrip("@").upper()
    for index_file in Path(search_dir).rglob(INDEX_FILE_NAME):
        mapping = load_index_map(index_file.parent)
        if normalized in mapping:
            return index_file.parent / mapping[normalized]
    raise PermanentAPIError(
        f"Unknown image index: @{normalized}. Run 'image-creator lookup --list' to see indices."
    )


def expand_reference(token: str, search_dir: str | Path) -> str:
    """Expand an ``@INDEX`` token to a full image path; pass other tokens through.

    Used by the CLI so path-accepting options (``--edit``, ``--ref``, ``--mask``, …)
    accept a short index in place of a filesystem path.
    """
    if not token.startswith("@"):
        return token
    return str(resolve_index(token, search_dir))


def list_index_entries(search_dir: str | Path) -> list[tuple[str, Path]]:
    """Return ``(index, image_path)`` pairs under ``search_dir``, newest image first.

    Entries whose image file no longer exists are still returned (sorted last) so a
    stale index is visible rather than silently dropped.
    """
    entries: list[tuple[str, Path]] = []
    for index_file in Path(search_dir).rglob(INDEX_FILE_NAME):
        for index, filename in load_index_map(index_file.parent).items():
            entries.append((index, index_file.parent / filename))

    def _recency(entry: tuple[str, Path]) -> float:
        _index, image_path = entry
        return image_path.stat().st_mtime if image_path.exists() else 0.0

    entries.sort(key=_recency, reverse=True)
    return entries
