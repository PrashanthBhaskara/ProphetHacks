"""Recommended data-only fair-price predictor for the trading agent.

This is the **clean, audited, team-facing** entry point. Use this — do
not call the helpers in `data_fair_price.py` directly unless you know
what you're doing (in particular, `event_size_platt` there silently
returns mis-calibrated probabilities if you forget `attach_test_sizes`).

Quick start
-----------

```python
from prep.baselines.fair_price import RecommendedPredictor
from prep.data import load_subset_1200

# Fit once at agent startup
predictor = RecommendedPredictor.fit(load_subset_1200())

# At each tick: pass the FULL candidate set so event sizes are computed
# from the live universe, not from training data
for event, market_info in candidate_set:
    p_yes = predictor.predict(event, market_info, candidate_set=candidate_set)
```

What it does
------------

1. Fits `event_size_platt`: p ≈ sigmoid(a + b·logit(q) + c·log(N_event))
   on the historical training set. Coefs are roughly:
        a ≈ +0.61  (bias)
        b ≈ +1.17  (slope on market price logit)
        c ≈ −0.46  (penalty per log-unit of event size)
2. Computes `n_event` from the LIVE candidate set you pass in at predict
   time — not from training data. (Using train counts silently produces
   Brier 0.224 vs market's 0.185; using live counts gives Brier 0.170.)
3. Logit-space shrinkage α=0.5 toward the market price q:
        p_final = sigmoid(0.5·logit(p_es) + 0.5·logit(q))
   Trades half the in-distribution mean alpha for tighter variance,
   raising P(beat market) at the 200-call eval scale from 94% → 99.9%.

Headline numbers (subset_1200 holdout, time-split 70/30)
--------------------------------------------------------

- Brier on full test (N=2,090): 0.171 vs market's 0.185
- Bootstrap N=200 (1,500 resamples):
    Brier mean 0.171, 95% CI [0.148, 0.194]
    P(beats market on Brier) = 99.9%
- Simulated P&L (default strategy) on full test: +$108 / $10k

Caveats — read before relying on this
-------------------------------------

- Test distribution is subset_1200 (Sports 75%, Politics/Economics/Other
  ~20%, Mentions 26 events). The May 2026 eval window is dominated by
  Fed/CPI/index strike grids — a different category mix. Numbers may
  not transfer.
- At N=200 the simulated P&L 95% CI is [−$40, +$96] — *includes zero*.
  Treat as "probably beats market on calibration, P&L uncertain".
- Without `candidate_set` passed in, `n_event` defaults to 1 and the
  prediction collapses to a simple Platt of q (still better than raw q,
  but ~half the alpha).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..data import Sample
from ..trade import market_mid_forecast


def _logit(p: float) -> float:
    p = max(1e-4, min(1 - 1e-4, p))
    return math.log(p / (1 - p))


def _inv_logit(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _q_of(market_info: dict) -> float:
    """Get the market mid-price (0-1), defensive against missing fields."""
    try:
        q = market_mid_forecast({}, market_info)
        return max(0.01, min(0.99, q))
    except Exception:
        return 0.5


def _event_size_lookup(samples: Sequence[Sample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in samples:
        et = s.event.get("event_ticker") or ""
        counts[et] = counts.get(et, 0) + 1
    return counts


def _ticker_prefix(event_ticker: str) -> str:
    """Kalshi ticker prefix (e.g. 'KXFED' from 'KXFED-26JUN-T425') — a
    stable proxy for event type. KXFED markets typically have ~11
    sibling outcomes; KXNBAGAME has ~2; etc. Used as a fallback when
    the live caller doesn't pass sibling context."""
    if not event_ticker:
        return ""
    head = event_ticker.split("-")[0]
    return head.upper()


@dataclass
class RecommendedPredictor:
    """Fitted predictor. Coefficients are `bias, logit_q_slope,
    log_event_size_slope`. Shrinkage α=0.5 is hard-coded — change
    `shrink_alpha` only if you understand the N=200 variance analysis
    in FORECAST_BENCHMARKS.md.

    `prefix_event_size` is a fallback table mapping ticker prefix →
    typical event size, learned from training. Used at predict time
    when the caller can't tell us the live n_event."""

    bias: float
    logit_q_slope: float
    log_event_size_slope: float
    prefix_event_size: dict[str, float]
    default_event_size: float
    shrink_alpha: float = 0.5

    @classmethod
    def fit(
        cls,
        train_samples: Sequence[Sample],
        *,
        l2: float = 0.5,
        max_iter: int = 100,
    ) -> "RecommendedPredictor":
        """Fit event_size_platt coefficients on training data.

        Each sample contributes one row with features
            [1, logit(q), log(num_markets_in_event)]
        where the event size is computed from the training batch
        (the natural per-event grouping). Outcomes drive a logistic
        regression with IRLS + small L2 on the slopes (not the bias).
        """
        sizes = _event_size_lookup(train_samples)

        X: list[list[float]] = []
        y: list[float] = []
        for s in train_samples:
            q = _q_of(s.market_info)
            et = s.event.get("event_ticker") or ""
            n = max(1, sizes.get(et, 1))
            X.append([1.0, _logit(q), math.log(n)])
            y.append(float(s.outcome))

        if len(X) < 30:
            # Too few samples to fit reliably — return identity predictor
            # (returns q exactly when shrink_alpha=0.5; equivalent to "don't
            # recalibrate").
            return cls(
                bias=0.0,
                logit_q_slope=1.0,
                log_event_size_slope=0.0,
                prefix_event_size={},
                default_event_size=1.0,
            )

        Xa = np.asarray(X)
        ya = np.asarray(y)
        L2 = np.array([0.0, l2, l2])
        beta = np.array([0.0, 1.0, 0.0])
        for _ in range(max_iter):
            z = np.clip(Xa @ beta, -30.0, 30.0)
            mu = 1.0 / (1.0 + np.exp(-z))
            W = np.clip(mu * (1 - mu), 1e-6, None)
            r = ya - mu
            g = -Xa.T @ r + L2 * beta
            H = (Xa.T * W) @ Xa + np.diag(L2)
            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            step = max(1.0, float(np.max(np.abs(delta))))
            delta /= step
            beta -= delta
            if float(np.max(np.abs(delta))) < 1e-7:
                break

        # Build the prefix → typical event size fallback table
        from collections import defaultdict
        by_prefix: dict[str, list[int]] = defaultdict(list)
        seen_events: set[str] = set()
        for s in train_samples:
            et = s.event.get("event_ticker") or ""
            if et in seen_events or not et:
                continue
            seen_events.add(et)
            n = sizes.get(et, 1)
            by_prefix[_ticker_prefix(et)].append(n)
        prefix_event_size = {
            p: float(np.median(ns)) for p, ns in by_prefix.items() if ns
        }
        default_event_size = float(np.median(list(sizes.values()) or [1.0]))

        return cls(
            bias=float(beta[0]),
            logit_q_slope=float(beta[1]),
            log_event_size_slope=float(beta[2]),
            prefix_event_size=prefix_event_size,
            default_event_size=default_event_size,
        )

    def predict(
        self,
        event: dict,
        market_info: dict,
        *,
        candidate_set: Sequence[Sample] | None = None,
        n_event: int | None = None,
    ) -> float:
        """Return P(YES) ∈ [0.01, 0.99] for one market.

        Pass `candidate_set` (list of all Samples in the current
        candidate universe) so n_event can be computed from the live
        data. If you can't, pass `n_event` explicitly. If both are
        omitted, n_event defaults to 1 — which makes this collapse
        to a Platt-only recalibration (≈ half the alpha).
        """
        q = _q_of(market_info)
        if n_event is None:
            et = event.get("event_ticker") or ""
            if candidate_set is not None:
                # Best: count siblings in the live batch
                n_event = sum(
                    1 for s in candidate_set
                    if (s.event.get("event_ticker") or "") == et
                )
            else:
                # Fallback: look up ticker prefix's typical event size
                prefix = _ticker_prefix(et)
                n_event = self.prefix_event_size.get(prefix, self.default_event_size)
        n_event = max(1.0, float(n_event))

        z = (
            self.bias
            + self.logit_q_slope * _logit(q)
            + self.log_event_size_slope * math.log(max(1.0, n_event))
        )
        p_es = _inv_logit(z)

        # Logit-space shrinkage toward market — variance reduction at small N
        z_shrunk = (
            self.shrink_alpha * _logit(p_es)
            + (1.0 - self.shrink_alpha) * _logit(q)
        )
        return max(0.01, min(0.99, _inv_logit(z_shrunk)))

    def predict_batch(self, candidate_set: Sequence[Sample]) -> dict[str, float]:
        """Predict for every market in the candidate set in one pass.

        Returns market_ticker → p_yes. The natural interface for the
        trading-track agent, which always sees the candidate set as
        a batch. This is the recommended call site — no chance of
        passing the wrong n_event."""
        # Compute event sizes once
        sizes = _event_size_lookup(candidate_set)
        out: dict[str, float] = {}
        for s in candidate_set:
            et = s.event.get("event_ticker") or ""
            n_event = max(1, sizes.get(et, 1))
            p = self.predict(s.event, s.market_info, n_event=n_event)
            out[s.event.get("market_ticker") or s.event.get("event_ticker") or ""] = p
        return out

    def __repr__(self) -> str:
        return (
            f"RecommendedPredictor(bias={self.bias:+.3f}, "
            f"logit_q_slope={self.logit_q_slope:+.3f}, "
            f"log_event_size_slope={self.log_event_size_slope:+.3f}, "
            f"shrink_alpha={self.shrink_alpha})"
        )
