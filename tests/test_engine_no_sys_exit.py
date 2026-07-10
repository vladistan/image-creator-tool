"""Regression + guard tests: engine surfaces failures as domain errors, never sys.exit.

Only ``cli.py`` owns process exit codes. Engine modules (sweep, generation_core,
imaging) must raise ``ImageCreatorError`` on failure so the CLI can map it.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from image_creator_tool import generation_core, imaging, sweep
from image_creator_tool.errors import ImageCreatorError


def _sweep_args(**kwargs):
    defaults = {
        "model": "alpha,beta",
        "preset": None,
        "platform": None,
        "seed": None,
        "n": 1,
        "dry_run": False,
        "prompt": "a test prompt",
        "project": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_run_sweep_all_cells_fail_raises_domain_error(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "resolve_output_path", lambda *a, **k: tmp_path / "out.png")

    def _boom(*args, **kwargs):
        raise RuntimeError("cell generation failed")

    monkeypatch.setattr(sweep, "_generate_inner", _boom)

    provider = SimpleNamespace(default_model="model-x")
    with pytest.raises(ImageCreatorError, match="All sweep cells failed"):
        sweep.run_sweep(_sweep_args(), {}, provider)


def test_engine_modules_do_not_reference_sys_exit():
    for module in (sweep, generation_core, imaging):
        src = Path(inspect.getfile(module)).read_text()
        assert "sys.exit" not in src, f"{module.__name__} must not call sys.exit"
