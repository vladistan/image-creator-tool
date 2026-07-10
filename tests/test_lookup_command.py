"""Tests for the `lookup` command (Phase 5, Step 5.4)."""

import pytest

from image_creator_tool.cli import app
from image_creator_tool.indexer import register_index
from image_creator_tool.provenance import ProvenanceRecord, write_provenance_sidecar


@pytest.fixture
def indexed_dir(tmp_path):
    """An output directory with one indexed image plus its provenance record."""
    image = tmp_path / "barn.png"
    image.write_bytes(b"barn")
    record = ProvenanceRecord(
        prompt="a red barn at sunset",
        model="gemini-3.1-flash-image-preview",
        provider="gemini",
        output_path=str(image),
        timestamp="2026-07-07T09:00:00",
        seed=7,
    )
    write_provenance_sidecar(record, image)
    index = register_index(image, key=record.entity_id)
    return tmp_path, index, image


def test_lookup_single_shows_path_and_metadata(runner, indexed_dir):
    output_dir, index, image = indexed_dir
    result = runner.invoke(app, ["lookup", index, "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert str(image) in result.stdout
    assert "gemini" in result.stdout
    assert "a red barn at sunset" in result.stdout


def test_lookup_list_shows_indices(runner, indexed_dir):
    output_dir, index, _image = indexed_dir
    result = runner.invoke(app, ["lookup", "--list", "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert index in result.stdout
    assert "barn.png" in result.stdout


def test_lookup_unknown_index_exits_nonzero(runner, tmp_path):
    result = runner.invoke(app, ["lookup", "ZZZZZZZZ", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_lookup_list_empty_dir(runner, tmp_path):
    result = runner.invoke(app, ["lookup", "--list", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No image indices" in result.stdout


def test_lookup_without_index_or_list_errors(runner, tmp_path):
    result = runner.invoke(app, ["lookup", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0
