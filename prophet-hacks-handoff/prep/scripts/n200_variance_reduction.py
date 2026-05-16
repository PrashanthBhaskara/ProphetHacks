"""Variance-reduction analysis at N=200.

We know event_size_platt has mean Brier Δ ≈ −0.016 vs market with
95% CI [−0.034, +0.003] at N=200. The signal is small enough that
variance reduction probably beats chasing marginal improvements.

Two approaches:
  A) Shrinkage: α · model + (1-α) · q. Trades mean alpha for lower variance.
  B) Stacked ensemble: average of multiple recalibrators.

For each, bootstrap N=200 and report:
  - Mean Brier
  - 95% CI
  - P(better than market on Brier)
  - P(positive P&L_default)

We want maximum P(better than market) AT N=200, not the highest in-distribution mean.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from prep.baselines.data_fair_price import (  # noqa: E402
    fit_decile_isotonic,
    fit_event_size_platt,
    fit_hierarchical_platt,
    fit_mean_bias,
    fit_platt_market,
)
from prep.data import load_subset_1200  # noqa: E402
from prep.trade import backtest, default_strategy, market_mid_forecast  # noqa: E402

# Reuse the time_split helper from n200_bootstrap
sys.path.insert(0, str(Path(__file__).resolve().parent))
from n200_bootstrap import time_split  # noqa: E402


def q_only(event, market_info):
    return max(0.01, min(0.99, market_mid_forecast(event, market_info)))


def shrink_predictor(base_fn, alpha):
    """alpha * base + (1 - alpha) * q."""
    def pred(event, market_info):
        try:
            p_base = base_fn(event, market_info)
            if isinstance(p_base, dict):
                p_base = p_base["p_yes"]
            p_base = max(0.01, min(0.99, float(p_base)))
        except Exception:
            p_base = 0.5
        q = q_only(event, market_info)
        return alpha * p_base + (1 - alpha) * q
    return pred


def ensemble_predictor(base_fns, weights=None):
    """Weighted average of multiple base predictors."""
    if weights is None:
        weights = [1.0 / len(base_fns)] * len(base_fns)
    def pred(event, market_info):
        ps = []
        for fn in base_fns:
            try:
                p = fn(event, market_info)
                if isinstance(p, dict):
                    p = p["p_yes"]
                ps.append(max(0.01, min(0.99, float(p))))
            except Exception:
                ps.append(0.5)
        return sum(w * p for w, p in zip(weights, ps))
    return pred


def brier(preds, outcomes):
    return float(np.mean((np.asarray(preds) - np.asarray(outcomes)) ** 2))


def evaluate(test, predictors):
    """For each predictor, compute Brier and P&L on this set."""
    results = {}
    for name, fn in predictors.items():
        preds, outs = [], []
        for s in test:
            try:
                p = fn(s.event, s.market_info)
                if isinstance(p, dict):
                    p = p["p_yes"]
                preds.append(max(0.01, min(0.99, float(p))))
                outs.append(int(s.outcome))
            except Exception:
                continue
        b = brier(preds, outs)
        pnl = backtest(test, forecast_fn=fn, strategy=default_strategy)["total_pnl"]
        results[name] = {"brier": b, "pnl": pnl}
    return results


def main():
    train, test = time_split(load_subset_1200())
    print(f"Train: {len(train):,}   Test: {len(test):,}")

    # Fit each baseline on train
    platt = fit_platt_market(train)
    event_size = fit_event_size_platt(train)
    if hasattr(event_size, "attach_test_sizes"):
        event_size.attach_test_sizes(test)
    mean_bias = fit_mean_bias(train)
    decile = fit_decile_isotonic(train)
    hier = fit_hierarchical_platt(train)

    def event_size_pred(e, m): return event_size(e, m)["p_yes"]
    def platt_pred(e, m): return platt(e, m)["p_yes"]
    def mean_bias_pred(e, m): return mean_bias(e, m)["p_yes"]
    def decile_pred(e, m): return decile(e, m)["p_yes"]
    def hier_pred(e, m): return hier(e, m)["p_yes"]

    # Build a battery of variants
    predictors = {"market": q_only}

    # Shrinkage of event_size_platt at several α levels
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        predictors[f"event_size_α={alpha:.2f}"] = shrink_predictor(event_size_pred, alpha)

    # Shrinkage of platt at several α levels
    for alpha in [0.5, 1.0]:
        predictors[f"platt_α={alpha:.2f}"] = shrink_predictor(platt_pred, alpha)

    # Simple equal-weight ensemble of all 4 fitted baselines (each fully weighted)
    predictors["ensemble_eq4"] = ensemble_predictor(
        [event_size_pred, platt_pred, mean_bias_pred, decile_pred])

    # Ensemble + market, equal weights
    predictors["ensemble_eq4+market"] = ensemble_predictor(
        [event_size_pred, platt_pred, mean_bias_pred, decile_pred, q_only])

    # The strongest single model + market average (50/50)
    predictors["0.5·event_size + 0.5·market"] = ensemble_predictor(
        [event_size_pred, q_only], weights=[0.5, 0.5])

    # Bootstrap N=200
    N = 200
    N_BOOT = 1500
    print(f"\nBootstrap {N_BOOT} subsamples of N={N}...")
    boot = {name: {"brier": [], "pnl": []} for name in predictors}
    rng = random.Random(42)
    for _ in range(N_BOOT):
        sub = [rng.choice(test) for _ in range(N)]
        res = evaluate(sub, predictors)
        for name, r in res.items():
            boot[name]["brier"].append(r["brier"])
            boot[name]["pnl"].append(r["pnl"])

    market_brier = np.array(boot["market"]["brier"])
    market_pnl = np.array(boot["market"]["pnl"])

    print(f"\n  {'Predictor':32s}  {'Brier (mean ± CI)':>26}  {'P(better)':>10}  "
          f"{'P&L_default ($)':>22}  {'P(>0)':>7}")
    print("  " + "-" * 105)
    for name in predictors:
        b_arr = np.array(boot[name]["brier"])
        p_arr = np.array(boot[name]["pnl"])
        b_lo, b_hi = np.percentile(b_arr, [2.5, 97.5])
        p_lo, p_hi = np.percentile(p_arr, [2.5, 97.5])
        p_better = (b_arr < market_brier).mean()
        p_pnl_pos = (p_arr > 0).mean()
        print(f"  {name:32s}  {b_arr.mean():.4f} [{b_lo:.3f},{b_hi:.3f}]  {p_better:>10.3f}  "
              f"{p_arr.mean():+6.1f} [{p_lo:+5.0f},{p_hi:+5.0f}]  {p_pnl_pos:>7.3f}")


if __name__ == "__main__":
    main()
