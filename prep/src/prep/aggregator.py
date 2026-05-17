"""Ensemble aggregator for the Prophet Hacks forecast track.

Combines per-model probability forecasts into a single calibrated p_yes
per market. Pipeline (per `prep/PLAN.md` Phase 3):

    per-model {ticker: p_yes}
        ↓ logit-pool average (optional per-model weights)
        ↓ isotonic recalibration (fit on holdout)
        ↓ market shrinkage (blend toward Kalshi price)
        ↓ extreme shrinkage (pull toward market at q ∈ [0,0.10] ∪ [0.90,1])
    final p_yes ∈ [0.01, 0.99]

All pure-stdlib — no scipy/sklearn dependency.

Logit-pool > arithmetic mean for ensembling probabilities: it's the
correct independent-evidence aggregation under a noisy-classifier model
and behaves well at extremes (averaging 0.99 and 0.99 in logit space
stays near 0.99, while arithmetic does too).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

CLAMP_LOW = 0.01
CLAMP_HIGH = 0.99


def _clamp(p: float) -> float:
    return max(CLAMP_LOW, min(CLAMP_HIGH, p))


def logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def logit_pool(preds: list[float], weights: list[float] | None = None) -> float:
    """Weighted geometric-mean-in-odds-space. Robust at extremes."""
    if not preds:
        raise ValueError("empty preds")
    if weights is None:
        weights = [1.0] * len(preds)
    if len(weights) != len(preds):
        raise ValueError("preds/weights length mismatch")
    w_sum = sum(weights)
    if w_sum <= 0:
        raise ValueError("weights sum to zero")
    s = sum(w * logit(p) for w, p in zip(weights, preds)) / w_sum
    return _clamp(sigmoid(s))


def arithmetic_pool(preds: list[float], weights: list[float] | None = None) -> float:
    """Weighted arithmetic mean — kept for ablation comparison."""
    if not preds:
        raise ValueError("empty preds")
    if weights is None:
        weights = [1.0] * len(preds)
    w_sum = sum(weights)
    return _clamp(sum(w * p for w, p in zip(weights, preds)) / w_sum)


# ---------------------------------------------------------------------------
# Isotonic calibration via PAVA (Pool Adjacent Violators)
# ---------------------------------------------------------------------------

@dataclass
class IsotonicCalibrator:
    """Monotone step function fit to (pred, outcome) pairs.

    `xs` and `ys` are the breakpoints after PAVA pooling. `transform`
    does piecewise-constant interpolation.
    """
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)

    def transform(self, p: float) -> float:
        if not self.xs:
            return _clamp(p)  # identity if not fit
        if p <= self.xs[0]:
            return _clamp(self.ys[0])
        if p >= self.xs[-1]:
            return _clamp(self.ys[-1])
        # Linear interpolation between adjacent breakpoints.
        for i in range(1, len(self.xs)):
            if p <= self.xs[i]:
                x0, x1 = self.xs[i - 1], self.xs[i]
                y0, y1 = self.ys[i - 1], self.ys[i]
                if x1 == x0:
                    return _clamp(y1)
                t = (p - x0) / (x1 - x0)
                return _clamp(y0 + t * (y1 - y0))
        return _clamp(self.ys[-1])


def fit_isotonic(preds: list[float], outcomes: list[int]) -> IsotonicCalibrator:
    """Pool-Adjacent-Violators isotonic regression.

    Sort by predicted probability, then iteratively pool adjacent
    violators (where mean is non-monotone) until output is monotone.
    """
    if not preds:
        return IsotonicCalibrator()
    if len(preds) != len(outcomes):
        raise ValueError("preds/outcomes length mismatch")

    paired = sorted(zip(preds, outcomes), key=lambda t: t[0])
    xs = [p for p, _ in paired]
    ys = [float(o) for _, o in paired]
    weights = [1.0] * len(ys)

    # Pool adjacent violators.
    i = 0
    while i < len(ys) - 1:
        if ys[i] > ys[i + 1]:
            new_w = weights[i] + weights[i + 1]
            new_y = (weights[i] * ys[i] + weights[i + 1] * ys[i + 1]) / new_w
            # collapse i+1 into i
            ys[i] = new_y
            weights[i] = new_w
            del ys[i + 1]
            del weights[i + 1]
            del xs[i + 1]
            # back up to recheck monotonicity with predecessor
            if i > 0:
                i -= 1
        else:
            i += 1

    return IsotonicCalibrator(xs=xs, ys=ys)


# ---------------------------------------------------------------------------
# Market shrinkage
# ---------------------------------------------------------------------------

def market_shrink(p_ens: float, p_market: float, alpha: float) -> float:
    """Linear blend toward market price. alpha ∈ [0, 1]; 0=no shrink."""
    alpha = max(0.0, min(1.0, alpha))
    return _clamp((1 - alpha) * p_ens + alpha * p_market)


def extreme_shrink(
    p_ens: float,
    p_market: float,
    *,
    threshold: float = 0.10,
    strength: float = 0.7,
) -> float:
    """When market is at an extreme (≤threshold or ≥1-threshold), pull
    the ensemble toward the market with the given strength.

    Paper §4.2.3 / Fig 6: LLMs are systematically miscalibrated at
    market extremes (too conservative). Shrinking toward q in those
    regions is a known win.
    """
    if p_market <= threshold or p_market >= 1 - threshold:
        return _clamp((1 - strength) * p_ens + strength * p_market)
    return p_ens


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

@dataclass
class AggregatorConfig:
    pool: str = "logit"  # "logit" | "arithmetic"
    model_weights: dict[str, float] | None = None  # name -> weight; uniform if None
    # Per-category overrides. Keyed by category name; value is the same
    # dict shape as model_weights. A category not present here uses the
    # global `model_weights` (or uniform). Markets without a known
    # category also use the global weights.
    model_weights_per_category: dict[str, dict[str, float]] | None = None
    isotonic: IsotonicCalibrator | None = None
    market_alpha: float = 0.0  # uniform market shrinkage
    # Per-category market_alpha overrides. Same fallback semantics as
    # model_weights_per_category.
    market_alpha_per_category: dict[str, float] | None = None
    extreme_shrink_threshold: float = 0.0  # 0 disables; e.g. 0.10
    extreme_shrink_strength: float = 0.7

    def weights_for(self, category: str | None) -> dict[str, float] | None:
        if category and self.model_weights_per_category:
            if category in self.model_weights_per_category:
                return self.model_weights_per_category[category]
        return self.model_weights

    def market_alpha_for(self, category: str | None) -> float:
        if category and self.market_alpha_per_category:
            if category in self.market_alpha_per_category:
                return self.market_alpha_per_category[category]
        return self.market_alpha


def aggregate_one(
    preds_by_model: dict[str, float],
    p_market: float | None,
    config: AggregatorConfig,
    category: str | None = None,
) -> float:
    """Aggregate one market's per-model predictions to a single p_yes.

    If `category` is given and the config has per-category overrides,
    those take precedence over global weights / market_alpha.
    """
    if not preds_by_model:
        # No model has a prediction — fall back to market or 0.5.
        return _clamp(p_market if p_market is not None else 0.5)

    names = list(preds_by_model.keys())
    preds = [preds_by_model[n] for n in names]

    weights_dict = config.weights_for(category)
    if weights_dict:
        weights = [weights_dict.get(n, 1.0) for n in names]
        # Filter zero-weight models out; logit_pool requires positive weights.
        filtered = [(p, w) for p, w in zip(preds, weights) if w > 0]
        if filtered:
            preds = [p for p, _ in filtered]
            weights = [w for _, w in filtered]
        else:
            # All models zero-weighted for this category — fall back to market.
            return _clamp(p_market if p_market is not None else 0.5)
    else:
        weights = None

    if config.pool == "logit":
        p = logit_pool(preds, weights)
    else:
        p = arithmetic_pool(preds, weights)

    if config.isotonic is not None:
        p = config.isotonic.transform(p)

    if p_market is not None:
        if config.extreme_shrink_threshold > 0:
            p = extreme_shrink(
                p, p_market,
                threshold=config.extreme_shrink_threshold,
                strength=config.extreme_shrink_strength,
            )
        market_alpha = config.market_alpha_for(category)
        if market_alpha > 0:
            p = market_shrink(p, p_market, market_alpha)

    return _clamp(p)


def aggregate_all(
    predictions: dict[str, dict[str, float]],  # model_name -> {ticker: p_yes}
    market_prices: dict[str, float] | None = None,  # ticker -> market p_yes
    config: AggregatorConfig | None = None,
    categories: dict[str, str] | None = None,  # ticker -> category (for per-cat configs)
) -> dict[str, float]:
    """Aggregate over the union of all tickers seen across models.

    `categories` is required when the config has any per-category
    overrides (model_weights_per_category, market_alpha_per_category).
    Without it, all tickers use the global settings.
    """
    config = config or AggregatorConfig()
    all_tickers: set[str] = set()
    for per_ticker in predictions.values():
        all_tickers.update(per_ticker.keys())

    out: dict[str, float] = {}
    for ticker in all_tickers:
        preds_here = {
            name: per_ticker[ticker]
            for name, per_ticker in predictions.items()
            if ticker in per_ticker
        }
        p_market = (market_prices or {}).get(ticker)
        category = (categories or {}).get(ticker)
        out[ticker] = aggregate_one(preds_here, p_market, config, category=category)
    return out


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_predictions_jsonl(path: str | Path) -> dict[str, float]:
    """Load `{market_ticker, p_yes, ...}` jsonl into ticker -> p_yes."""
    out: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["market_ticker"]] = float(row["p_yes"])
    return out


def load_outcomes_jsonl(path: str | Path) -> dict[str, int]:
    """Load `{market_ticker, outcome}` jsonl into ticker -> outcome."""
    out: dict[str, int] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "outcome" in row:
            out[row["market_ticker"]] = int(row["outcome"])
    return out
