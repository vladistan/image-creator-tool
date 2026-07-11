"""Tests for the `strip` command (assemble @index panels into a comic strip)."""

from PIL import Image

from image_creator_tool.cli import app
from image_creator_tool.indexer import register_index


def _make_indexed_image(tmp_path, name, size=(40, 30)):
    path = tmp_path / name
    Image.new("RGB", size, (120, 120, 200)).save(path)
    return register_index(path, key=f"entity-{name}")


def test_strip_from_indices_produces_bordered_image(runner, tmp_path):
    ids = [_make_indexed_image(tmp_path, f"p{i}.png") for i in range(3)]
    out = tmp_path / "strip.png"

    result = runner.invoke(
        app,
        ["strip", *[f"@{i}" for i in ids], str(out), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "3 panels" in result.output


def test_strip_layout_options_change_dimensions(runner, tmp_path):
    ids = [_make_indexed_image(tmp_path, f"p{i}.png") for i in range(4)]
    row_out = tmp_path / "row.png"
    grid_out = tmp_path / "grid.png"

    runner.invoke(app, ["strip", *[f"@{i}" for i in ids], str(row_out),
                        "--output-dir", str(tmp_path)])
    runner.invoke(app, ["strip", *[f"@{i}" for i in ids], str(grid_out),
                        "--cols", "2", "--output-dir", str(tmp_path)])

    with Image.open(row_out) as row, Image.open(grid_out) as grid:
        # A single row is wider and shorter than a 2-column grid of the same panels.
        assert row.width > grid.width
        assert grid.height > row.height


def test_strip_gutter_and_border_widen_output(runner, tmp_path):
    ids = [_make_indexed_image(tmp_path, f"p{i}.png") for i in range(2)]
    thin = tmp_path / "thin.png"
    thick = tmp_path / "thick.png"

    runner.invoke(app, ["strip", *[f"@{i}" for i in ids], str(thin),
                        "--gutter", "4", "--border", "2", "--output-dir", str(tmp_path)])
    runner.invoke(app, ["strip", *[f"@{i}" for i in ids], str(thick),
                        "--gutter", "40", "--border", "20", "--output-dir", str(tmp_path)])

    with Image.open(thin) as a, Image.open(thick) as b:
        assert b.width > a.width


def test_strip_captions_add_height(runner, tmp_path):
    ids = [_make_indexed_image(tmp_path, f"p{i}.png") for i in range(2)]
    plain = tmp_path / "plain.png"
    captioned = tmp_path / "cap.png"

    runner.invoke(app, ["strip", *[f"@{i}" for i in ids], str(plain),
                        "--output-dir", str(tmp_path)])
    result = runner.invoke(app, ["strip", *[f"@{i}" for i in ids], str(captioned),
                                 "--caption", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output

    with Image.open(plain) as a, Image.open(captioned) as b:
        assert b.height > a.height


def test_strip_unresolvable_panel_exits_nonzero(runner, tmp_path):
    good = _make_indexed_image(tmp_path, "p0.png")
    out = tmp_path / "strip.png"
    result = runner.invoke(
        app, ["strip", f"@{good}", "@ZZZZZZZZ", str(out), "--output-dir", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert not out.exists()
