"""Regression tests for per-cell arg construction in sweeps.

Guards that _apply_combo carries prompt-shaping / edit inputs (notably --style)
into each sweep cell; omitting them silently drops the style from every cell.
"""

from types import SimpleNamespace

from image_creator_tool import sweep


def _base_args(**overrides):
    base = {
        "prompt": "a corgi riding a skateboard",
        "style": "bold flat-vector neon illustration",
        "edit_op": "inpaint",
        "search_prompt": "the sky",
        "mask": "/tmp/mask.png",
        "preset": None,
        "platform": None,
        "model": "flash-3.1",
        "seed": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_apply_combo_carries_style():
    result = sweep._apply_combo(_base_args(), {"model": "flux-max"})
    assert result.style == "bold flat-vector neon illustration"


def test_apply_combo_carries_edit_fields():
    result = sweep._apply_combo(_base_args(), {"model": "flux-max"})
    assert result.edit_op == "inpaint"
    assert result.search_prompt == "the sky"
    assert result.mask == "/tmp/mask.png"


def test_apply_combo_applies_model_override():
    result = sweep._apply_combo(_base_args(), {"model": "flux-max"})
    assert result.model == "flux-max"


def test_apply_combo_style_absent_defaults_none():
    args = _base_args()
    del args.style
    result = sweep._apply_combo(args, {"model": "flux-max"})
    assert result.style is None
