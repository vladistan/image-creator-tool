"""Tests for the `prov` command group (Phase 4, Step 4.4)."""

from pathlib import Path

import pytest

from image_creator_tool.cli import app
from image_creator_tool.indexer import register_index
from image_creator_tool.provenance import ProvenanceRecord, write_provenance_sidecar


@pytest.fixture
def outputs_dir(tmp_path, monkeypatch):
    """A populated output directory with two provenance sidecars."""
    target = tmp_path / "outputs"
    target.mkdir()

    barn = target / "barn.png"
    lake = target / "lake.png"
    write_provenance_sidecar(
        ProvenanceRecord(
            prompt="a red barn",
            model="gemini-3.1-flash-image-preview",
            provider="gemini",
            output_path=str(barn),
            timestamp="2026-07-07T09:00:00",
            seed=1,
        ),
        barn,
    )
    write_provenance_sidecar(
        ProvenanceRecord(
            prompt="a blue lake",
            model="dall-e-3",
            provider="openai",
            output_path=str(lake),
            timestamp="2026-07-08T10:00:00",
            seed=2,
        ),
        lake,
    )
    return target


def test_prov_list_shows_all_records(runner, outputs_dir):
    result = runner.invoke(app, ["prov", "list", "--output-dir", str(outputs_dir)])
    assert result.exit_code == 0
    assert "gemini" in result.stdout
    assert "openai" in result.stdout


def test_prov_list_filters_by_provider(runner, outputs_dir):
    result = runner.invoke(
        app, ["prov", "list", "--output-dir", str(outputs_dir), "--provider", "openai"]
    )
    assert result.exit_code == 0
    assert "openai" in result.stdout
    assert "gemini" not in result.stdout


def test_prov_list_filters_by_date(runner, outputs_dir):
    result = runner.invoke(
        app, ["prov", "list", "--output-dir", str(outputs_dir), "--date", "2026-07-07"]
    )
    assert result.exit_code == 0
    assert "a red barn" in result.stdout or "barn" in result.stdout
    assert "lake" not in result.stdout


def test_prov_list_filters_by_model(runner, outputs_dir):
    result = runner.invoke(
        app, ["prov", "list", "--output-dir", str(outputs_dir), "--model", "dall-e-3"]
    )
    assert result.exit_code == 0
    assert "openai" in result.stdout
    assert "gemini" not in result.stdout


def test_prov_list_empty_dir(runner, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["prov", "list", "--output-dir", str(empty)])
    assert result.exit_code == 0
    assert "No provenance" in result.stdout


def test_prov_show_displays_record(runner, outputs_dir):
    barn = outputs_dir / "barn.png"
    result = runner.invoke(app, ["prov", "show", str(barn)])
    assert result.exit_code == 0
    assert "a red barn" in result.stdout
    assert "gemini" in result.stdout


def test_prov_show_missing_exits_nonzero(runner, tmp_path):
    result = runner.invoke(app, ["prov", "show", str(tmp_path / "ghost.png")])
    assert result.exit_code != 0


def test_prov_export_prov_n(runner, outputs_dir):
    result = runner.invoke(
        app, ["prov", "export", str(outputs_dir), "--format", "prov-n"]
    )
    assert result.exit_code == 0
    assert result.stdout.lstrip().startswith("document")
    assert "gemini" in result.stdout
    assert "openai" in result.stdout


def test_prov_export_unknown_format_exits_nonzero(runner, outputs_dir):
    result = runner.invoke(
        app, ["prov", "export", str(outputs_dir), "--format", "yaml"]
    )
    assert result.exit_code != 0


def test_prov_show_accepts_sidecar_path_directly(runner, outputs_dir):
    sidecar = outputs_dir / "barn.prov.json"
    result = runner.invoke(app, ["prov", "show", str(Path(sidecar))])
    assert result.exit_code == 0
    assert "a red barn" in result.stdout


def test_prov_show_accepts_index(runner, outputs_dir):
    barn = outputs_dir / "barn.png"
    barn.write_bytes(b"fake-png")
    index = register_index(barn, key="entity-barn")

    result = runner.invoke(
        app, ["prov", "show", f"@{index}", "--output-dir", str(outputs_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "a red barn" in result.stdout


def test_prov_show_unknown_index_exits_nonzero(runner, outputs_dir):
    result = runner.invoke(
        app, ["prov", "show", "@ZZZZZZZZ", "--output-dir", str(outputs_dir)]
    )
    assert result.exit_code != 0
