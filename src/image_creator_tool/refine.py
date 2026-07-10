"""Bounded closed-loop style refinement.

Ties together generation, vision assessment, and style rewriting into a loop that
converges a style descriptor toward a reference set: generate images with the
current style across several models, score each against the source, and rewrite
the style from the critiques — repeating until a fidelity threshold is met or the
iteration budget is spent.

Domain layer only: the generate / assess / refine steps are injected as callables
so the loop is unit-testable without real API calls. The CLI (`commands/style.py`)
wires the real sweep generation and vision functions in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

log = structlog.get_logger()

_MAX_ITERATIONS_CAP = 4


@dataclass
class IterationRecord:
    """One refinement pass: the style tried and how its images scored."""

    iteration: int
    style: str
    score: float
    images: list[Path] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)


@dataclass
class RefineResult:
    """Outcome of a refinement loop."""

    best_style: str
    best_score: float
    iterations: list[IterationRecord]
    converged: bool


def refine_style_loop(
    *,
    initial_style: str,
    generate_fn: Callable[[str], list[Path]],
    assess_fn: Callable[[Path], tuple[int, str]],
    refine_fn: Callable[[str, list[str]], str],
    max_iterations: int = _MAX_ITERATIONS_CAP,
    threshold: float = 85.0,
) -> RefineResult:
    """Iteratively refine `initial_style` toward the assessor's target.

    Args:
        initial_style: Starting style descriptor (e.g. from group extraction).
        generate_fn: Given a style, generate candidate images (across models) and
            return their paths.
        assess_fn: Given a candidate image, return (score 0-100, critique).
        refine_fn: Given the current style and the round's critiques, return a
            rewritten style.
        max_iterations: Hard cap on passes (clamped to [1, 4]).
        threshold: Stop early once a round's mean score reaches this.

    Returns:
        RefineResult with the best-scoring style, its score, per-iteration history,
        and whether the threshold was met.
    """
    iterations = max(1, min(_MAX_ITERATIONS_CAP, max_iterations))
    style = initial_style
    records: list[IterationRecord] = []
    best: IterationRecord | None = None
    converged = False

    for i in range(iterations):
        images = generate_fn(style)
        assessments = [assess_fn(img) for img in images]
        scores = [s for s, _ in assessments]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        critiques = [c for _, c in assessments if c]

        record = IterationRecord(
            iteration=i + 1,
            style=style,
            score=mean_score,
            images=images,
            critiques=critiques,
        )
        records.append(record)
        log.info(
            "refine iteration complete",
            iteration=i + 1,
            score=round(mean_score, 1),
            images=len(images),
        )

        if best is None or mean_score > best.score:
            best = record

        if mean_score >= threshold:
            converged = True
            break

        if i < iterations - 1:
            style = refine_fn(style, critiques)

    assert best is not None  # loop runs at least once
    return RefineResult(
        best_style=best.style,
        best_score=best.score,
        iterations=records,
        converged=converged,
    )
