"""Tests for the `contact-sheet` command (assemble a sheet from image indices)."""

from PIL import Image

from image_creator_tool.cli import app
from image_creator_tool.indexer import register_index


def _make_indexed_image(tmp_path, name):
    path = tmp_path / name
    Image.new("RGB", (8, 8), (200, 60, 60)).save(path)
    return register_index(path, key=f"entity-{name}")


def test_contact_sheet_from_indices(runner, tmp_path):
    id_a = _make_indexed_image(tmp_path, "a.png")
    id_b = _make_indexed_image(tmp_path, "b.png")
    out = tmp_path / "sheet.png"

    result = runner.invoke(
        app,
        [
            "contact-sheet", f"@{id_a}", f"@{id_b}", str(out),
            "--output-dir", str(tmp_path), "--label", "index",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "2 images" in result.output


def test_contact_sheet_resolves_across_subdirs(runner, tmp_path):
    sub1 = tmp_path / "run1"
    sub2 = tmp_path / "run2"
    sub1.mkdir()
    sub2.mkdir()
    id_a = _make_indexed_image(sub1, "a.png")
    id_b = _make_indexed_image(sub2, "b.png")
    out = tmp_path / "sheet.png"

    result = runner.invoke(
        app,
        ["contact-sheet", f"@{id_a}", f"@{id_b}", str(out), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_contact_sheet_unknown_index_errors(runner, tmp_path):
    out = tmp_path / "sheet.png"
    result = runner.invoke(
        app, ["contact-sheet", "@ZZZZZZZZ", str(out), "--output-dir", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert not out.exists()


def _make_raw_image(tmp_path, name):
    path = tmp_path / name
    Image.new("RGB", (8, 8), (60, 160, 60)).save(path)
    return path


def test_contact_sheet_from_raw_paths(runner, tmp_path):
    a = _make_raw_image(tmp_path, "web-cat.png")
    b = _make_raw_image(tmp_path, "web-dog.png")
    out = tmp_path / "sheet.png"

    result = runner.invoke(
        app, ["contact-sheet", str(a), str(b), str(out), "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "2 images" in result.output


def test_contact_sheet_mixes_index_and_raw_path(runner, tmp_path):
    idx = _make_indexed_image(tmp_path, "gen.png")
    raw = _make_raw_image(tmp_path, "web-cat.png")
    out = tmp_path / "sheet.png"

    result = runner.invoke(
        app,
        ["contact-sheet", f"@{idx}", str(raw), str(out), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_contact_sheet_raw_missing_path_errors(runner, tmp_path):
    out = tmp_path / "sheet.png"
    result = runner.invoke(
        app,
        ["contact-sheet", str(tmp_path / "ghost.png"), str(out), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert not out.exists()


def test_contact_sheet_raw_label_is_filename_stem(runner, tmp_path, monkeypatch):
    raw = _make_raw_image(tmp_path, "sunset-beach.png")
    captured = {}

    def _fake_sheet(cells, output_path, **kwargs):
        captured["cells"] = cells
        output_path.write_bytes(b"x")
        return output_path

    monkeypatch.setattr(
        "image_creator_tool.commands.contact_sheet.make_labeled_contact_sheet", _fake_sheet
    )
    result = runner.invoke(
        app, ["contact-sheet", str(raw), str(tmp_path / "s.png"), "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert captured["cells"][0][1] == "sunset-beach"
