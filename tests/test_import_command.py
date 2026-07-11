"""Tests for the `import`/`forget` reference-image commands (Phase 11)."""

from PIL import Image

from image_creator_tool.cli import app
from image_creator_tool.indexer import register_index


def _make_source(tmp_path, name, color=(200, 60, 60)):
    path = tmp_path / name
    Image.new("RGB", (12, 12), color).save(path)
    return path


def _import(runner, tmp_path, *sources):
    args = ["import", *[str(s) for s in sources], "--output-dir", str(tmp_path)]
    return runner.invoke(app, args)


def _index_from(output):
    return output.split()[0].lstrip("@")


def test_import_prints_index_and_copies_into_ref_dir(runner, tmp_path):
    src = _make_source(tmp_path, "cat.png")
    result = _import(runner, tmp_path, src)
    assert result.exit_code == 0, result.output
    assert result.output.startswith("@")
    ref_dir = tmp_path / "ref-images"
    assert ref_dir.is_dir()
    assert list(ref_dir.glob("*.png"))


def test_import_round_trips_via_lookup_contact_sheet_and_prov(runner, tmp_path):
    src = _make_source(tmp_path, "cat.png")
    index = _index_from(_import(runner, tmp_path, src).output)

    lookup = runner.invoke(app, ["lookup", index, "--output-dir", str(tmp_path)])
    assert lookup.exit_code == 0, lookup.output

    sheet = runner.invoke(
        app, ["contact-sheet", f"@{index}", str(tmp_path / "s.png"), "--output-dir", str(tmp_path)]
    )
    assert sheet.exit_code == 0, sheet.output

    prov = runner.invoke(app, ["prov", "show", f"@{index}", "--output-dir", str(tmp_path)])
    assert prov.exit_code == 0, prov.output
    assert "imported" in prov.output


def test_import_dedupes_identical_bytes(runner, tmp_path):
    src = _make_source(tmp_path, "cat.png")
    first = _import(runner, tmp_path, src)
    second = _import(runner, tmp_path, src)

    assert _index_from(first.output) == _index_from(second.output)
    assert "existing" in second.output
    assert len(list((tmp_path / "ref-images").glob("*.png"))) == 1


def test_import_dedupe_appends_source_for_different_path(runner, tmp_path):
    src_a = _make_source(tmp_path, "cat.png")
    copy = tmp_path / "same-cat.png"
    copy.write_bytes(src_a.read_bytes())

    index = _index_from(_import(runner, tmp_path, src_a).output)
    _import(runner, tmp_path, copy)

    prov = runner.invoke(app, ["prov", "show", f"@{index}", "--output-dir", str(tmp_path)])
    assert "cat.png" in prov.output
    assert "same-cat.png" in prov.output


def test_import_missing_source_exits_nonzero(runner, tmp_path):
    result = _import(runner, tmp_path, tmp_path / "ghost.png")
    assert result.exit_code != 0


def test_forget_removes_imported_image(runner, tmp_path):
    src = _make_source(tmp_path, "cat.png")
    index = _index_from(_import(runner, tmp_path, src).output)
    copied = next((tmp_path / "ref-images").glob("*.png"))
    sidecar = copied.with_name(copied.stem + ".prov.json")
    assert sidecar.is_file()

    result = runner.invoke(app, ["forget", f"@{index}", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert not copied.exists()
    assert not sidecar.exists()

    lookup = runner.invoke(app, ["lookup", index, "--output-dir", str(tmp_path)])
    assert lookup.exit_code != 0


def test_forget_refuses_non_imported_index(runner, tmp_path):
    generated = tmp_path / "gen.png"
    Image.new("RGB", (8, 8), (10, 200, 10)).save(generated)
    index = register_index(generated, key="entity-gen")

    result = runner.invoke(app, ["forget", f"@{index}", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "not an imported image" in result.output
    assert generated.exists()


def test_forget_unknown_index_exits_nonzero(runner, tmp_path):
    result = runner.invoke(app, ["forget", "@ZZZZZZZZ", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_prune_dry_run_previews_unreferenced(runner, tmp_path):
    src = _make_source(tmp_path, "cat.png")
    index = _index_from(_import(runner, tmp_path, src).output)

    result = runner.invoke(app, ["forget", "--prune", "--dry-run", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert index in result.output
    assert "would be pruned" in result.output
    assert list((tmp_path / "ref-images").glob("*.png"))
