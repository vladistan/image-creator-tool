"""Tests for the parameter sweep engine."""

from types import SimpleNamespace

from image_creator_tool.sweep import _parse_dims, is_sweep


def _args(**kwargs):
    defaults = {
        "model": None, "preset": None, "platform": None, "seed": None, "n": 1, "dry_run": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_no_sweep_when_no_commas():
    assert not is_sweep(_args(model="flash", preset="editorial"))


def test_sweep_detected_for_model_commas():
    assert is_sweep(_args(model="flash,flux"))


def test_sweep_detected_for_preset_commas():
    assert is_sweep(_args(preset="editorial,ink"))


def test_sweep_detected_for_platform_commas():
    assert is_sweep(_args(platform="youtube,square"))


def test_sweep_detected_for_seed_commas():
    assert is_sweep(_args(seed="42,99"))


def test_parse_dims_model_only():
    dims = _parse_dims(_args(model="flash,flux,ultra"))
    assert len(dims) == 1
    assert dims[0].name == "model"
    assert dims[0].values == ["flash", "flux", "ultra"]


def test_parse_dims_model_and_preset():
    dims = _parse_dims(_args(model="flash,flux", preset="editorial,ink"))
    names = [d.name for d in dims]
    assert "model" in names
    assert "preset" in names
    assert len(dims) == 2


def test_parse_dims_n_only_no_sweep():
    # n > 1 alone does NOT create a sweep
    dims = _parse_dims(_args(n=4))
    assert len(dims) == 0


def test_parse_dims_n_added_when_other_dims_present():
    # n > 1 WITH other dims → variant dimension appended
    dims = _parse_dims(_args(model="flash,flux", n=3))
    names = [d.name for d in dims]
    assert "variant" in names
    assert len(dims) == 2


def test_parse_dims_n_not_added_when_n_is_1():
    dims = _parse_dims(_args(model="flash,flux", n=1))
    names = [d.name for d in dims]
    assert "variant" not in names


def test_parse_dims_values_stripped():
    dims = _parse_dims(_args(model=" flash , flux "))
    assert dims[0].values == ["flash", "flux"]
