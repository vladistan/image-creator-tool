"""Tests for the short-index domain layer (Phase 5, Steps 5.1 to 5.3).

Covers deterministic index derivation and schema (5.1), generation/persistence
across sessions (5.2), and index resolution for `@INDEX` references (5.3).
"""

import os

import pytest

from image_creator_tool import indexer
from image_creator_tool.errors import PermanentAPIError
from image_creator_tool.indexer import (
    INDEX_LENGTH,
    compute_index,
    expand_reference,
    list_index_entries,
    load_index_map,
    register_index,
    resolve_index,
)

_BASE32_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")


# --- Step 5.1: Schema + deterministic derivation ----------------------------


def test_index_uses_base32_alphabet_and_length():
    index = compute_index("imgc:image-abc123")
    assert len(index) == INDEX_LENGTH
    assert set(index) <= _BASE32_ALPHABET


def test_index_is_deterministic_for_same_key():
    assert compute_index("imgc:image-abc") == compute_index("imgc:image-abc")


def test_index_differs_for_different_keys():
    assert compute_index("imgc:image-abc") != compute_index("imgc:image-xyz")


# --- Step 5.2: Generation + persistence -------------------------------------


def test_register_assigns_unique_index_per_image(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    index_a = register_index(a, key="entity-a")
    index_b = register_index(b, key="entity-b")

    assert index_a != index_b
    assert set(index_a) <= _BASE32_ALPHABET


def test_register_persists_to_index_file(tmp_path):
    image = tmp_path / "barn.png"
    image.write_bytes(b"barn")
    index = register_index(image, key="entity-barn")

    # Persisted in a sidecar index file in the output directory.
    assert indexer.index_file_for(tmp_path).is_file()
    assert load_index_map(tmp_path)[index] == "barn.png"


def test_index_survives_across_sessions(tmp_path):
    image = tmp_path / "barn.png"
    image.write_bytes(b"barn")
    index = register_index(image, key="entity-barn")

    # A fresh read (simulating a new process) still resolves the index.
    assert resolve_index(index, tmp_path) == image


def test_index_for_reverse_lookup(tmp_path):
    image = tmp_path / "barn.png"
    image.write_bytes(b"barn")
    index = register_index(image, key="entity-barn")

    assert indexer.index_for(image) == index


def test_index_for_returns_none_when_unindexed(tmp_path):
    image = tmp_path / "unlisted.png"
    image.write_bytes(b"x")
    assert indexer.index_for(image) is None


def test_register_is_idempotent_for_same_image(tmp_path):
    image = tmp_path / "barn.png"
    image.write_bytes(b"barn")
    first = register_index(image, key="entity-barn")
    second = register_index(image, key="entity-barn")
    assert first == second
    assert len(load_index_map(tmp_path)) == 1


def test_colliding_index_is_lengthened(tmp_path, monkeypatch):
    """Two different images whose derived index collides get distinct indices."""
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    # Force both keys to derive the same short index at the default length.
    real_compute = indexer.compute_index

    def _fake_compute(key: str, length: int = INDEX_LENGTH) -> str:
        if length == INDEX_LENGTH:
            return "AAAAAAAA"
        return real_compute(key, length)

    monkeypatch.setattr(indexer, "compute_index", _fake_compute)

    index_a = register_index(a, key="entity-a")
    index_b = register_index(b, key="entity-b")

    assert index_a != index_b
    assert index_a == "AAAAAAAA"
    assert len(index_b) > INDEX_LENGTH


# --- Step 5.3: Resolution for @INDEX references -----------------------------


def test_resolve_index_finds_image_in_nested_dirs(tmp_path):
    nested = tmp_path / "project-x"
    nested.mkdir()
    image = nested / "barn.png"
    image.write_bytes(b"barn")
    index = register_index(image, key="entity-barn")

    assert resolve_index(index, tmp_path) == image


def test_resolve_index_is_case_insensitive_and_strips_at(tmp_path):
    image = tmp_path / "barn.png"
    image.write_bytes(b"barn")
    index = register_index(image, key="entity-barn")

    assert resolve_index(f"@{index.lower()}", tmp_path) == image


def test_resolve_unknown_index_raises(tmp_path):
    with pytest.raises(PermanentAPIError, match="Unknown image index"):
        resolve_index("ZZZZZZZZ", tmp_path)


def test_expand_reference_passes_through_plain_paths(tmp_path):
    assert expand_reference("/some/path.png", tmp_path) == "/some/path.png"


def test_expand_reference_resolves_at_index(tmp_path):
    image = tmp_path / "barn.png"
    image.write_bytes(b"barn")
    index = register_index(image, key="entity-barn")

    assert expand_reference(f"@{index}", tmp_path) == str(image)


def test_list_entries_sorted_newest_first(tmp_path):
    old = tmp_path / "old.png"
    new = tmp_path / "new.png"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    register_index(old, key="entity-old")
    register_index(new, key="entity-new")

    entries = list_index_entries(tmp_path)
    assert [p.name for _index, p in entries] == ["new.png", "old.png"]
