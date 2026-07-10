"""Tests for the bounded style-refinement loop (injected generate/assess/refine)."""

from pathlib import Path

from image_creator_tool.refine import refine_style_loop


def _gen(_style):
    return [Path("/tmp/a.png"), Path("/tmp/b.png")]


def test_converges_early_when_threshold_met():
    calls = {"refine": 0}

    def refine(style, _critiques):
        calls["refine"] += 1
        return style + "+"

    result = refine_style_loop(
        initial_style="base",
        generate_fn=_gen,
        assess_fn=lambda _img: (90, "good"),
        refine_fn=refine,
        max_iterations=4,
        threshold=85,
    )
    assert result.converged is True
    assert len(result.iterations) == 1
    assert calls["refine"] == 0  # no rewrite needed after convergence
    assert result.best_score == 90


def test_runs_full_budget_and_keeps_best():
    scores = iter([40, 70, 50, 60])  # 8 assessments (2 imgs x 4 iters), pairs -> 40,50,60...

    def assess(_img):
        return next(scores), "critique"

    # per-iteration means: (40,70)=55, (50,60)=55 -> need distinct; use single-image gen
    result = refine_style_loop(
        initial_style="base",
        generate_fn=lambda _s: [Path("/tmp/a.png")],
        assess_fn=assess,
        refine_fn=lambda s, _c: s + "+",
        max_iterations=4,
        threshold=95,
    )
    assert result.converged is False
    assert len(result.iterations) == 4
    # scores per iter: 40, 70, 50, 60 -> best is iter 2 (70)
    assert result.best_score == 70
    assert result.best_style == "base+"  # style used in iter 2 = one refine from base


def test_iterations_capped_at_four():
    result = refine_style_loop(
        initial_style="base",
        generate_fn=lambda _s: [Path("/tmp/a.png")],
        assess_fn=lambda _img: (10, "bad"),
        refine_fn=lambda s, _c: s,
        max_iterations=99,
        threshold=95,
    )
    assert len(result.iterations) == 4


def test_refine_receives_prior_critiques():
    seen = {}

    def refine(style, critiques):
        seen["critiques"] = critiques
        return style + "+"

    refine_style_loop(
        initial_style="base",
        generate_fn=lambda _s: [Path("/tmp/a.png")],
        assess_fn=lambda _img: (10, "too saturated"),
        refine_fn=refine,
        max_iterations=2,
        threshold=95,
    )
    assert seen["critiques"] == ["too saturated"]


def test_empty_generation_scores_zero_and_continues():
    result = refine_style_loop(
        initial_style="base",
        generate_fn=lambda _s: [],
        assess_fn=lambda _img: (100, "unused"),
        refine_fn=lambda s, _c: s,
        max_iterations=2,
        threshold=50,
    )
    assert result.best_score == 0.0
    assert len(result.iterations) == 2
