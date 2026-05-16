"""Scoring functions used in the Prophet Arena paper (§3.1.1, §3.1.2).

Brier = mean squared error between predicted p_yes and binary outcome.
ECE   = expected calibration error, binned. Measures whether predictions
        of "X%" actually happen X% of the time.
"""

from __future__ import annotations

from typing import Sequence


def brier(p_yes: Sequence[float], outcomes: Sequence[int]) -> float:
    if len(p_yes) != len(outcomes):
        raise ValueError("p_yes and outcomes length mismatch")
    if not p_yes:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(p_yes, outcomes)) / len(p_yes)


def ece(p_yes: Sequence[float], outcomes: Sequence[int], n_bins: int = 10) -> float:
    if len(p_yes) != len(outcomes):
        raise ValueError("p_yes and outcomes length mismatch")
    n = len(p_yes)
    if n == 0:
        return float("nan")

    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in zip(p_yes, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, o))

    total = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        avg_o = sum(o for _, o in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(avg_p - avg_o)
    return total
