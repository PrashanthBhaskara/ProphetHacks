"""Forecasting metric suite for the Prophet Hacks trading track.

The goal here is *not* to optimize a single number — it's to identify
which forecasting metric most reliably predicts realized trading P&L.
The paper (§B.4, §C.1) proves Brier and trading return diverge, so we
need a richer toolkit before trusting any one metric.

Metric tiers
------------
Tier 1 — proper scoring rules (penalize miscalibration):
    brier, log_loss, spherical_score

Tier 2 — decomposition (what *kind* of error):
    reliability, resolution, uncertainty  (Murphy decomposition of Brier)
    ece (uniform), ece_adaptive, mce
    sharpness, platt_slope, platt_intercept

Tier 3 — discrimination + trading-relevant (should predict P&L):
    auc_roc, auc_pr, accuracy_50
    direction_vs_market, signed_edge
    brier_skill_score (vs market, vs always_half)

Tier 4 — robustness:
    bootstrap_ci

All metrics accept Sequence[float] for p_yes (0–1) and Sequence[int]
for outcomes (0/1). Market price q is required only by the trading-
relevant metrics.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Tier 1 — proper scoring rules
# ---------------------------------------------------------------------------


def brier(p_yes: Sequence[float], outcomes: Sequence[int]) -> float:
    if len(p_yes) != len(outcomes):
        raise ValueError("p_yes and outcomes length mismatch")
    if not p_yes:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(p_yes, outcomes)) / len(p_yes)


def log_loss(p_yes: Sequence[float], outcomes: Sequence[int], eps: float = 1e-9) -> float:
    """Negative log-likelihood. Lower is better, perfect=0, unbounded above.
    Punishes confident-wrong much harder than Brier."""
    if not p_yes:
        return float("nan")
    total = 0.0
    for p, o in zip(p_yes, outcomes):
        p = min(1 - eps, max(eps, p))
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / len(p_yes)


def spherical_score(p_yes: Sequence[float], outcomes: Sequence[int]) -> float:
    """Spherical proper scoring rule. Higher is better, perfect=1.

    For binary outcome y and prediction p:
        score = (p if y=1 else (1-p)) / sqrt(p^2 + (1-p)^2)

    Less harsh than log-loss on extreme miscalibration."""
    if not p_yes:
        return float("nan")
    total = 0.0
    for p, o in zip(p_yes, outcomes):
        denom = math.sqrt(p * p + (1 - p) * (1 - p))
        if denom == 0:
            continue
        total += (p if o == 1 else (1 - p)) / denom
    return total / len(p_yes)


# ---------------------------------------------------------------------------
# Tier 2 — calibration, sharpness, decomposition
# ---------------------------------------------------------------------------


def ece(p_yes: Sequence[float], outcomes: Sequence[int], n_bins: int = 10) -> float:
    """Expected Calibration Error with uniform-width bins. Original ECE."""
    if not p_yes:
        return float("nan")
    n = len(p_yes)
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


def ece_adaptive(p_yes: Sequence[float], outcomes: Sequence[int], n_bins: int = 10) -> float:
    """ECE with equal-count (quantile) bins. More robust than fixed-width
    when predictions are concentrated near 0 or 1 (typical for our data:
    market prices on near-deterministic Crypto markets cluster at extremes).
    """
    n = len(p_yes)
    if n == 0:
        return float("nan")
    pairs = sorted(zip(p_yes, outcomes), key=lambda x: x[0])
    bin_size = max(1, n // n_bins)
    total = 0.0
    for i in range(0, n, bin_size):
        bucket = pairs[i : i + bin_size]
        if not bucket:
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        avg_o = sum(o for _, o in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(avg_p - avg_o)
    return total


def mce(p_yes: Sequence[float], outcomes: Sequence[int], n_bins: int = 10) -> float:
    """Maximum Calibration Error — worst single bin. Catches local
    miscalibration that ECE averages away."""
    if not p_yes:
        return float("nan")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in zip(p_yes, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, o))
    worst = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        avg_o = sum(o for _, o in bucket) / len(bucket)
        worst = max(worst, abs(avg_p - avg_o))
    return worst


def murphy_decomposition(
    p_yes: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> dict[str, float]:
    """Brier = reliability - resolution + uncertainty.

    - reliability: weighted mean squared distance between predicted prob
      and observed frequency in each bin. LOWER is better (calibration).
    - resolution: weighted mean squared distance of bin frequencies from
      overall base rate. HIGHER is better (predictions actually distinguish).
    - uncertainty: variance of outcomes; a property of the data, not the
      forecaster (it's a constant for a given test set).
    """
    n = len(p_yes)
    if n == 0:
        return {"reliability": float("nan"), "resolution": float("nan"), "uncertainty": float("nan")}
    base_rate = sum(outcomes) / n
    uncertainty = base_rate * (1 - base_rate)

    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in zip(p_yes, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, o))

    reliability = 0.0
    resolution = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        avg_o = sum(o for _, o in bucket) / len(bucket)
        w = len(bucket) / n
        reliability += w * (avg_p - avg_o) ** 2
        resolution += w * (avg_o - base_rate) ** 2
    return {"reliability": reliability, "resolution": resolution, "uncertainty": uncertainty}


def sharpness(p_yes: Sequence[float]) -> float:
    """Mean distance of predictions from 0.5. Pure confidence measure,
    independent of accuracy. Higher = more decisive.

    Sharpness alone is meaningless — a coin-flipper can have high
    sharpness. Pair with reliability / Brier to interpret."""
    if not p_yes:
        return float("nan")
    return sum(abs(p - 0.5) for p in p_yes) / len(p_yes)


def platt_fit(p_yes: Sequence[float], outcomes: Sequence[int]) -> dict[str, float]:
    """Fit a logistic recalibration: y ~ sigmoid(a*logit(p) + b).

    A perfectly calibrated forecaster has slope=1, intercept=0.
    - slope > 1: predictions are too conservative (need extremization)
    - slope < 1: predictions are overconfident (need shrinkage toward 0.5)
    - intercept > 0: systematic YES bias

    Returns slope, intercept, plus the Brier achievable after applying
    this recalibration ("post_platt_brier"). The gap between Brier and
    post_platt_brier shows how much your forecaster could improve with
    a 2-parameter rescale alone.
    """
    n = len(p_yes)
    if n < 10:
        return {"slope": float("nan"), "intercept": float("nan"), "post_platt_brier": float("nan")}

    p = np.clip(np.asarray(p_yes, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(outcomes, dtype=float)
    logit_p = np.log(p / (1 - p))

    # IRLS with L2 regularization on (a, b) — without it, perfectly
    # misaligned data (e.g. inverse_market predictor) drives the
    # logistic to ±infinity. The prior pulls the fit to a sensible scale.
    # Step damping keeps each Newton update bounded.
    L2 = 1e-3
    a, b = 1.0, 0.0
    for _ in range(50):
        z = np.clip(a * logit_p + b, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-z))
        W = mu * (1 - mu)
        W = np.clip(W, 1e-6, None)
        r = (y - mu)
        g_a = -np.sum(r * logit_p) + L2 * (a - 1.0)
        g_b = -np.sum(r) + L2 * b
        H_aa = np.sum(W * logit_p * logit_p) + L2
        H_ab = np.sum(W * logit_p)
        H_bb = np.sum(W) + L2
        det = H_aa * H_bb - H_ab * H_ab
        if abs(det) < 1e-12:
            break
        d_a = (H_bb * g_a - H_ab * g_b) / det
        d_b = (-H_ab * g_a + H_aa * g_b) / det
        # damp updates to keep them bounded
        step = max(1.0, abs(d_a), abs(d_b))
        d_a /= step
        d_b /= step
        a -= d_a
        b -= d_b
        if abs(d_a) + abs(d_b) < 1e-7:
            break

    z = np.clip(a * logit_p + b, -30.0, 30.0)
    mu = 1.0 / (1.0 + np.exp(-z))
    post_brier = float(np.mean((mu - y) ** 2))
    return {"slope": float(a), "intercept": float(b), "post_platt_brier": post_brier}


# ---------------------------------------------------------------------------
# Tier 3 — discrimination + trading-relevant
# ---------------------------------------------------------------------------


def auc_roc(p_yes: Sequence[float], outcomes: Sequence[int]) -> float:
    """Area under ROC. Scale-invariant — doesn't care about calibration,
    only about whether YES predictions rank above NO predictions.
    0.5 = random, 1.0 = perfect."""
    n = len(p_yes)
    if n == 0:
        return float("nan")
    pos = [p for p, o in zip(p_yes, outcomes) if o == 1]
    neg = [p for p, o in zip(p_yes, outcomes) if o == 0]
    if not pos or not neg:
        return float("nan")
    # Mann-Whitney U formulation
    ranks = {p: r for r, p in enumerate(sorted(set(p_yes)), start=1)}
    # Handle ties by averaging ranks within tied groups
    arr = np.asarray(sorted(p_yes))
    rank_lookup: dict[float, float] = {}
    i = 0
    while i < len(arr):
        j = i
        while j < len(arr) and arr[j] == arr[i]:
            j += 1
        avg = (i + j + 1) / 2  # ranks are 1-indexed
        for k in range(i, j):
            rank_lookup[float(arr[k])] = avg
        i = j
    sum_ranks_pos = sum(rank_lookup[float(p)] for p in pos)
    n_pos = len(pos)
    n_neg = len(neg)
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return U / (n_pos * n_neg)


def auc_pr(p_yes: Sequence[float], outcomes: Sequence[int]) -> float:
    """Area under Precision-Recall curve. Sensitive to class imbalance
    (our data is 29% YES — AUC-PR penalises errors on the minority class
    more directly than ROC does)."""
    n = len(p_yes)
    if n == 0:
        return float("nan")
    pairs = sorted(zip(p_yes, outcomes), key=lambda x: -x[0])
    total_pos = sum(outcomes)
    if total_pos == 0:
        return float("nan")
    tp = 0
    fp = 0
    last_recall = 0.0
    auc = 0.0
    for i, (_, o) in enumerate(pairs, start=1):
        if o == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / total_pos
        auc += precision * (recall - last_recall)
        last_recall = recall
    return auc


def accuracy_at_threshold(p_yes: Sequence[float], outcomes: Sequence[int], threshold: float = 0.5) -> float:
    """Fraction of predictions on the correct side of `threshold`."""
    if not p_yes:
        return float("nan")
    correct = sum(1 for p, o in zip(p_yes, outcomes) if (p >= threshold) == (o == 1))
    return correct / len(p_yes)


def direction_vs_market(
    p_yes: Sequence[float],
    outcomes: Sequence[int],
    market_q: Sequence[float],
    *,
    deadband: float = 0.0,
) -> float:
    """Fraction of markets where the forecaster picks the right side of `q`.

    Per §B.4 of the paper: this is the metric that actually determines
    trading P&L. A forecaster with worse Brier but better direction
    will make more money than one with the reverse.

    Definition:
        pick_yes = p_yes > q + deadband
        pick_no  = p_yes < q - deadband
        else: abstain (excluded from numerator)

    Score = fraction-correct over the non-abstain trades.
    `deadband` simulates a strategy's spread filter — a wider deadband
    means we only count predictions far from the market.
    """
    correct = 0
    n_decisions = 0
    for p, o, q in zip(p_yes, outcomes, market_q):
        if p > q + deadband:
            n_decisions += 1
            if o == 1:
                correct += 1
        elif p < q - deadband:
            n_decisions += 1
            if o == 0:
                correct += 1
    if n_decisions == 0:
        return float("nan")
    return correct / n_decisions


def signed_edge(
    p_yes: Sequence[float],
    outcomes: Sequence[int],
    market_q: Sequence[float],
) -> float:
    """Mean of (p - q) * (y - q) across markets.

    This is *the* trading-success surrogate without spreads/fees:
    positive iff the forecaster is on average on the right side of `q`,
    and magnitudes are weighted by how confidently they disagree with
    the market AND how surprising the resolution was. Equivalent to
    expected P&L per unit bet in an ideal-spread world.

    For comparison: a forecaster that just copies the market scores 0.
    A directionally-correct forecaster scores positive; an
    inverse-market forecaster scores negative."""
    if not p_yes:
        return float("nan")
    total = 0.0
    for p, o, q in zip(p_yes, outcomes, market_q):
        total += (p - q) * (o - q)
    return total / len(p_yes)


def brier_skill_score(
    p_yes: Sequence[float],
    outcomes: Sequence[int],
    reference_p: Sequence[float],
) -> float:
    """BSS = 1 - Brier(model) / Brier(reference).
    - BSS > 0: model beats reference (e.g., market)
    - BSS = 0: tied with reference
    - BSS < 0: worse than reference

    The "skill" normalisation makes Brier comparable across categories
    with different base rates (Sports Brier 0.18 vs Crypto Brier 0.02
    aren't comparable as-is)."""
    b_m = brier(p_yes, outcomes)
    b_r = brier(reference_p, outcomes)
    if b_r == 0 or math.isnan(b_r) or math.isnan(b_m):
        return float("nan")
    return 1 - b_m / b_r


# ---------------------------------------------------------------------------
# Tier 4 — robustness
# ---------------------------------------------------------------------------


def bootstrap_ci(
    metric_fn,
    *args,
    n_resamples: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap a (lower, upper) confidence interval for any metric_fn
    that takes positional Sequence args of equal length. Generic enough
    to wrap brier, log_loss, signed_edge, etc.
    """
    rng = random.Random(seed)
    n = len(args[0])
    if n == 0:
        return (float("nan"), float("nan"))
    vals: list[float] = []
    indices_pool = list(range(n))
    for _ in range(n_resamples):
        idx = [rng.choice(indices_pool) for _ in range(n)]
        resampled = tuple([a[i] for i in idx] for a in args)
        try:
            v = metric_fn(*resampled)
            if not math.isnan(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    lo = vals[int((1 - confidence) / 2 * len(vals))]
    hi = vals[int((1 + confidence) / 2 * len(vals)) - 1]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Convenience: full report
# ---------------------------------------------------------------------------


def full_report(
    p_yes: Sequence[float],
    outcomes: Sequence[int],
    market_q: Sequence[float] | None = None,
) -> dict:
    """Compute every metric in one pass.

    If `market_q` is provided, also reports trading-relevant metrics
    (signed_edge, direction_vs_market, BSS_vs_market). Else returns
    them as None.
    """
    report: dict = {
        "n": len(p_yes),
        "base_rate": sum(outcomes) / len(outcomes) if outcomes else float("nan"),
        # Tier 1
        "brier": brier(p_yes, outcomes),
        "log_loss": log_loss(p_yes, outcomes),
        "spherical": spherical_score(p_yes, outcomes),
        # Tier 2
        "ece": ece(p_yes, outcomes),
        "ece_adaptive": ece_adaptive(p_yes, outcomes),
        "mce": mce(p_yes, outcomes),
        "sharpness": sharpness(p_yes),
    }
    report.update(murphy_decomposition(p_yes, outcomes))
    report.update(platt_fit(p_yes, outcomes))
    # Tier 3 (discrimination)
    report["auc_roc"] = auc_roc(p_yes, outcomes)
    report["auc_pr"] = auc_pr(p_yes, outcomes)
    report["accuracy_50"] = accuracy_at_threshold(p_yes, outcomes, 0.5)

    if market_q is not None and len(market_q) == len(p_yes):
        report["direction_vs_market"] = direction_vs_market(p_yes, outcomes, market_q)
        report["direction_vs_market_deadband_0.05"] = direction_vs_market(p_yes, outcomes, market_q, deadband=0.05)
        report["signed_edge"] = signed_edge(p_yes, outcomes, market_q)
        report["bss_vs_market"] = brier_skill_score(p_yes, outcomes, market_q)
    half = [0.5] * len(p_yes)
    report["bss_vs_half"] = brier_skill_score(p_yes, outcomes, half)
    return report
