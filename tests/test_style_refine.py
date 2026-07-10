"""Tests for per-image group extraction, per-source assessment, and rewriting.

Group extraction extracts each image separately then LLM-merges (union of the
palette); assessment scores the candidate against each source individually.
"""

import pytest

from image_creator_tool import style as style_lib
from image_creator_tool.errors import PermanentAPIError


def _img(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG-fake")
    return p


def _patch_chat(monkeypatch, response=None, *, on_call=None):
    """Patch _vision_chat; record every call and return `response` or on_call(...)."""
    calls = []

    def fake_chat(images, instruction, provider, model, api_key):
        calls.append({"images": images, "instruction": instruction, "provider": provider})
        if on_call is not None:
            return on_call(images, instruction)
        return response

    monkeypatch.setattr(style_lib, "_vision_chat", fake_chat)
    return calls


# --- Group extraction: per-image then combine --------------------------------


def test_extract_style_group_per_image_then_combines(monkeypatch, tmp_path):
    imgs = [_img(tmp_path, f"{i}.png") for i in range(3)]
    calls = _patch_chat(
        monkeypatch, on_call=lambda images, _i: "member desc" if images else "MERGED STYLE"
    )
    style = style_lib.extract_style_group([str(p) for p in imgs])

    assert style == "MERGED STYLE"
    member_calls = [c for c in calls if c["images"]]
    combine_calls = [c for c in calls if not c["images"]]
    assert len(member_calls) == 3  # one vision call per image
    assert all(len(c["images"]) == 1 for c in member_calls)
    assert len(combine_calls) == 1  # single text-only merge


def test_extract_style_group_single_image_skips_combine(monkeypatch, tmp_path):
    img = _img(tmp_path, "0.png")
    calls = _patch_chat(monkeypatch, on_call=lambda images, _i: "only-desc" if images else "MERGED")
    style = style_lib.extract_style_group([str(img)])
    assert style == "only-desc"
    assert all(c["images"] for c in calls)  # no combine call


def test_extract_style_group_empty_raises(monkeypatch):
    _patch_chat(monkeypatch, response="x")
    with pytest.raises(PermanentAPIError, match="at least one image"):
        style_lib.extract_style_group([])


def test_extract_style_group_missing_file_raises(monkeypatch, tmp_path):
    _patch_chat(monkeypatch, response="x")
    with pytest.raises(PermanentAPIError, match="not found"):
        style_lib.extract_style_group([str(tmp_path / "nope.png")])


def test_combine_preserves_via_text_only_call(monkeypatch):
    calls = _patch_chat(monkeypatch, response="violet, orange, magenta, neon green, flat")
    merged = style_lib.combine_style_descriptions(["a, violet", "b, neon green"])
    assert "neon green" in merged
    assert calls[0]["images"] == []  # text-only


def test_combine_empty_raises(monkeypatch):
    _patch_chat(monkeypatch, response="x")
    with pytest.raises(PermanentAPIError, match="at least one description"):
        style_lib.combine_style_descriptions([])


# --- Per-source assessment ---------------------------------------------------


def test_assess_scores_each_source_and_averages(monkeypatch, tmp_path):
    srcs = [_img(tmp_path, f"s{i}.png") for i in range(2)]
    cand = _img(tmp_path, "cand.png")
    scores = iter(['{"score": 60, "critique": "a"}', '{"score": 80, "critique": "b"}'])
    calls = _patch_chat(monkeypatch, on_call=lambda _im, _i: next(scores))

    score, note = style_lib.assess_style_fidelity([str(s) for s in srcs], str(cand), "style")

    assert score == 70  # mean of 60 and 80
    assert "a" in note and "b" in note
    assert len(calls) == 2  # one call per source
    assert all(len(c["images"]) == 2 for c in calls)  # each: [source, candidate]


def test_assess_single_source_returns_its_score(monkeypatch, tmp_path):
    src = _img(tmp_path, "src.png")
    cand = _img(tmp_path, "cand.png")
    _patch_chat(
        monkeypatch,
        response='{"score": 72, "critique": "too saturated", "suggestions": "narrow palette"}',
    )
    score, note = style_lib.assess_style_fidelity([str(src)], str(cand), "style")
    assert score == 72
    assert "too saturated" in note
    assert "narrow palette" in note


def test_assess_clamps_score(monkeypatch, tmp_path):
    src = _img(tmp_path, "src.png")
    cand = _img(tmp_path, "cand.png")
    _patch_chat(monkeypatch, response='{"score": 250, "critique": "c"}')
    score, _ = style_lib.assess_style_fidelity([str(src)], str(cand), "style")
    assert score == 100


def test_assess_non_json_raises(monkeypatch, tmp_path):
    src = _img(tmp_path, "src.png")
    cand = _img(tmp_path, "cand.png")
    _patch_chat(monkeypatch, response="not json")
    with pytest.raises(PermanentAPIError, match="non-JSON"):
        style_lib.assess_style_fidelity([str(src)], str(cand), "style")


def test_assess_missing_score_raises(monkeypatch, tmp_path):
    src = _img(tmp_path, "src.png")
    cand = _img(tmp_path, "cand.png")
    _patch_chat(monkeypatch, response='{"critique": "c"}')
    with pytest.raises(PermanentAPIError, match="score"):
        style_lib.assess_style_fidelity([str(src)], str(cand), "style")


# --- Style rewriting (text-only) ---------------------------------------------


def test_refine_style_text_returns_cleaned(monkeypatch):
    _patch_chat(monkeypatch, response="```\nflat vector, restrained violet palette\n```")
    out = style_lib.refine_style_text("old style", ["too saturated"])
    assert out == "flat vector, restrained violet palette"


def test_refine_style_text_is_text_only(monkeypatch):
    calls = _patch_chat(monkeypatch, response="refined")
    style_lib.refine_style_text("old", ["c1", "c2"])
    assert calls[-1]["images"] == []
