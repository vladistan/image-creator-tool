"""Integration tests: badge flags thread from args into the sheet builders.

Covers generation_core (multi-variant contact sheet) and sweep (labeled sheet)
pass-through plus the numbered/non-numbered output lines.
"""

from types import SimpleNamespace

from image_creator_tool import generation_core, sweep
from image_creator_tool.generation_core import GenerationResult


def _fake_result(path):
    return GenerationResult(
        output_path=path,
        prompt="a barn",
        model="m",
        preset=None,
        platform=None,
        edit_source=None,
        timestamp="2026-07-07T00:00:00",
        duration_s=0.0,
    )


def _gen_args(**kwargs):
    defaults = {
        "prompt": "a barn",
        "output": None,
        "preset": None,
        "platform": None,
        "model": None,
        "edit": None,
        "edit_op": None,
        "search_prompt": None,
        "mask": None,
        "reference": [],
        "style_refs": [],
        "object_refs": [],
        "project": None,
        "n": 2,
        "seed": None,
        "aspect": None,
        "size": None,
        "quality": None,
        "contact_cols": None,
        "contact_cell_width": None,
        "contact_bg": None,
        "badges": True,
        "contact_badge_radius": None,
        "dry_run": False,
        "no_metadata": True,
        "presets": {},
        "platforms": {},
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _patch_generation_core(monkeypatch, captured, tmp_path):
    provider = SimpleNamespace(name="fake")
    monkeypatch.setattr(
        generation_core, "_resolve_generation_model", lambda a, c, p: (provider, "m")
    )
    monkeypatch.setattr(
        generation_core,
        "generate_once",
        lambda *a, **k: _fake_result(a[1]),
    )
    monkeypatch.setattr(generation_core, "append_history", lambda *a, **k: None)

    def _capture_sheet(paths, out, **kwargs):
        captured.update(kwargs)
        captured["n"] = len(paths)
        return out

    monkeypatch.setattr(generation_core, "make_contact_sheet", _capture_sheet)
    return provider


def test_generation_core_threads_badges_and_prints_numbered(monkeypatch, tmp_path, capsys):
    captured = {}
    provider = _patch_generation_core(monkeypatch, captured, tmp_path)
    config = {"output_dir": str(tmp_path)}
    generation_core._generate_inner(_gen_args(), config, provider)
    assert captured["badges"] is True
    assert captured["badge_radius"] == 30
    out = capsys.readouterr().out
    assert "Contact sheet:" in out
    assert "(2 images, numbered)" in out


def test_generation_core_no_badges_omits_numbered(monkeypatch, tmp_path, capsys):
    captured = {}
    provider = _patch_generation_core(monkeypatch, captured, tmp_path)
    config = {"output_dir": str(tmp_path)}
    generation_core._generate_inner(_gen_args(badges=False), config, provider)
    assert captured["badges"] is False
    out = capsys.readouterr().out
    assert "(2 images)" in out
    assert "numbered" not in out


def test_generation_core_badge_radius_override(monkeypatch, tmp_path):
    captured = {}
    provider = _patch_generation_core(monkeypatch, captured, tmp_path)
    config = {"output_dir": str(tmp_path)}
    generation_core._generate_inner(_gen_args(contact_badge_radius=40), config, provider)
    assert captured["badge_radius"] == 40


def _sweep_args(**kwargs):
    defaults = {
        "model": "alpha,beta",
        "preset": None,
        "platform": None,
        "seed": None,
        "n": 1,
        "dry_run": False,
        "prompt": "a barn",
        "project": None,
        "badges": True,
        "contact_badge_radius": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _patch_sweep(monkeypatch, captured, tmp_path):
    monkeypatch.setattr(sweep, "resolve_output_path", lambda *a, **k: tmp_path / "out.png")
    monkeypatch.setattr(
        sweep, "_generate_inner", lambda *a, **k: [_fake_result(tmp_path / "cell.png")]
    )

    def _capture_sheet(cells, out, **kwargs):
        captured.update(kwargs)
        captured["n"] = len(cells)
        return out

    monkeypatch.setattr(sweep, "make_labeled_contact_sheet", _capture_sheet)


def test_sweep_threads_badges_and_prints_numbered(monkeypatch, tmp_path, capsys):
    captured = {}
    _patch_sweep(monkeypatch, captured, tmp_path)
    provider = SimpleNamespace(default_model="model-x")
    sweep.run_sweep(_sweep_args(), {}, provider)
    assert captured["badges"] is True
    assert captured["badge_radius"] == 30
    out = capsys.readouterr().out
    assert "Sweep sheet:" in out
    assert ", numbered" in out


def test_sweep_no_badges_and_radius_override(monkeypatch, tmp_path, capsys):
    captured = {}
    _patch_sweep(monkeypatch, captured, tmp_path)
    provider = SimpleNamespace(default_model="model-x")
    sweep.run_sweep(_sweep_args(badges=False, contact_badge_radius=40), {}, provider)
    assert captured["badges"] is False
    assert captured["badge_radius"] == 40
    out = capsys.readouterr().out
    assert "numbered" not in out
