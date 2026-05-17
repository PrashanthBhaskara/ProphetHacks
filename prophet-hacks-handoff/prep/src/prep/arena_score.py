"""Prophet Arena event-level scoring helpers.

The research page describes leaderboard "Brier Score" as
1 - classical Brier, while the developer page and local CLI use classical
Brier where lower is better. Report both names to avoid ambiguity.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence


def _clip_prob(value: float | int | str | None) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError):
        p = 0.0
    if math.isnan(p) or math.isinf(p):
        return 0.0
    return max(0.0, min(1.0, p))


def normalize_prediction(
    probabilities: Mapping[str, float],
    outcomes: Sequence[str],
) -> dict[str, float]:
    """Project a prediction onto the event outcomes and normalize if possible."""
    projected = {outcome: _clip_prob(probabilities.get(outcome)) for outcome in outcomes}
    total = sum(projected.values())
    if total <= 0:
        return projected
    return {outcome: value / total for outcome, value in projected.items()}


def event_brier_classical(
    probabilities: Mapping[str, float],
    actuals: Mapping[str, int | float | bool],
    outcomes: Sequence[str],
    *,
    normalize: bool = True,
) -> float:
    """Classical event Brier, averaged across outcomes.

    Averaging across labels makes binary coherent forecasts match the older
    scalar score: p_yes=0.7, outcome YES -> 0.09.
    """
    if not outcomes:
        return float("nan")
    probs = normalize_prediction(probabilities, outcomes) if normalize else {
        outcome: _clip_prob(probabilities.get(outcome)) for outcome in outcomes
    }
    total = 0.0
    for outcome in outcomes:
        y = 1.0 if bool(actuals.get(outcome, 0)) else 0.0
        total += (probs.get(outcome, 0.0) - y) ** 2
    return total / len(outcomes)


def event_brier_score(
    probabilities: Mapping[str, float],
    actuals: Mapping[str, int | float | bool],
    outcomes: Sequence[str],
    *,
    normalize: bool = True,
) -> float:
    """Leaderboard-style score from the research page: 1 - classical Brier."""
    return 1.0 - event_brier_classical(probabilities, actuals, outcomes, normalize=normalize)


def should_normalize_actuals(actuals: Mapping[str, int | float | bool]) -> bool:
    """Normalize predictions only for one-hot outcome rows.

    The current developer docs say each event resolves to one label, but the
    in-repo subset_1200 file contains older multi-market rows where multiple
    component markets can resolve YES. Those are best scored as independent
    probability labels rather than forcing the prediction to sum to one.
    """
    return sum(1 for value in actuals.values() if bool(value)) == 1


@dataclass
class ScoreSummary:
    n: int
    classical_brier: float
    arena_brier_score: float
    binary_n: int
    nonbinary_n: int
    exclusive_n: int
    multilabel_n: int
    classical_brier_ci: tuple[float, float] | None = None
    arena_brier_score_ci: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "classical_brier_lower_is_better": self.classical_brier,
            "arena_brier_score_higher_is_better": self.arena_brier_score,
            "binary_n": self.binary_n,
            "nonbinary_n": self.nonbinary_n,
            "exclusive_n": self.exclusive_n,
            "multilabel_n": self.multilabel_n,
            "classical_brier_ci": self.classical_brier_ci,
            "arena_brier_score_ci": self.arena_brier_score_ci,
        }


def summarize_event_scores(
    event_scores: Sequence[float],
    *,
    outcome_counts: Sequence[int],
    exclusive_flags: Sequence[bool],
    bootstrap_resamples: int = 0,
    seed: int = 42,
) -> ScoreSummary:
    scores = [float(s) for s in event_scores if not math.isnan(float(s))]
    if not scores:
        return ScoreSummary(
            n=0,
            classical_brier=float("nan"),
            arena_brier_score=float("nan"),
            binary_n=0,
            nonbinary_n=0,
            exclusive_n=0,
            multilabel_n=0,
        )

    classical = sum(scores) / len(scores)
    classical_ci = None
    arena_ci = None
    if bootstrap_resamples > 0 and len(scores) > 1:
        rng = random.Random(seed)
        boot = []
        n = len(scores)
        for _ in range(bootstrap_resamples):
            sample = [scores[rng.randrange(n)] for _ in range(n)]
            boot.append(sum(sample) / n)
        boot.sort()
        lo = boot[int(0.025 * len(boot))]
        hi = boot[min(len(boot) - 1, int(0.975 * len(boot)))]
        classical_ci = (lo, hi)
        arena_ci = (1.0 - hi, 1.0 - lo)

    binary_n = sum(1 for count in outcome_counts if count == 2)
    exclusive_n = sum(1 for flag in exclusive_flags if flag)
    return ScoreSummary(
        n=len(scores),
        classical_brier=classical,
        arena_brier_score=1.0 - classical,
        binary_n=binary_n,
        nonbinary_n=len(scores) - binary_n,
        exclusive_n=exclusive_n,
        multilabel_n=len(scores) - exclusive_n,
        classical_brier_ci=classical_ci,
        arena_brier_score_ci=arena_ci,
    )
