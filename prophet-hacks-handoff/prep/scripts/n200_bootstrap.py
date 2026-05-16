"""Bootstrap N=200 to estimate detectable differences at eval scale.

The judges call our endpoint at most 200 times in 2 weeks. With N=200,
sampling variance can swamp any calibration improvement. This script
answers:

  Q1: 95% CI on Brier and P&L for each baseline at N=200
  Q2: What is the minimum Brier improvement detectable at this scale?
  Q3: At N=200, is event_size_platt distinguishable from just-using-q?

Methodology: resample N=200 markets with replacement from the
subset_1200 test split, recompute Brier & P&L for each baseline, repeat
2000 times. Print mean + 95% percentile CIs.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from prep.baselines.data_fair_price import (  # noqa: E402
    fit_event_size_platt,
    fit_mean_bias,
    fit_platt_market,
)
from prep.data import Sample, load_subset_1200  # noqa: E402
from prep.trade import (  # noqa: E402
    backtest,
    default_strategy,
    default_tight_band_strategy,
    market_mid_forecast,
)


def time_split(samples, train_frac=0.7):
    """Replicate the time-split used in the rest of the analysis."""
    import pandas as pd
    from ast import literal_eval
    csv = Path(__file__).resolve().parents[1] / "data" / "external" / "subset_1200.csv"
    df = pd.read_csv(csv).sort_values("snapshot_time").reset_index(drop=True)
    cut = int(len(df) * train_frac)

    def to_samples(d):
        out = []
        for _, row in d.iterrows():
            try:
                outcomes = literal_eval(row["market_outcome"]) or {}
                market_data = literal_eval(row["market_data"]) or {}
            except Exception:
                continue
            for market_name, outcome in outcomes.items():
                md = market_data.get(market_name) or {}
                event = {
                    "event_ticker": row["event_ticker"],
                    "market_ticker": f"{row['event_ticker']}-{market_name.replace(' ', '_')}",
                    "title": row.get("title") or "",
                    "subtitle": market_name,
                    "category": row.get("category") or "Other",
                    "close_time": row.get("close_time") or "",
                    "rules": row.get("rules") or None,
                    "description": None,
                }
                out.append(Sample(event=event, market_info=md, outcome=int(outcome)))
        return out

    return to_samples(df.iloc[:cut]), to_samples(df.iloc[cut:])


def brier(preds, outcomes):
    return float(np.mean((np.asarray(preds) - np.asarray(outcomes)) ** 2))


def q_only(event, market_info):
    return max(0.01, min(0.99, market_mid_forecast(event, market_info)))


def evaluate_subsample(test_subsample, predictors):
    """Per-baseline: Brier, P&L_default, P&L_tight_band on this subsample."""
    results = {}
    for name, fn in predictors.items():
        preds = []
        outcomes = []
        for s in test_subsample:
            try:
                p = fn(s.event, s.market_info)
                if isinstance(p, dict):
                    p = p["p_yes"]
                preds.append(max(0.01, min(0.99, float(p))))
                outcomes.append(int(s.outcome))
            except Exception:
                continue
        b = brier(preds, outcomes)
        pnl_def = backtest(test_subsample, forecast_fn=fn, strategy=default_strategy)["total_pnl"]
        pnl_tb = backtest(test_subsample, forecast_fn=fn, strategy=default_tight_band_strategy)["total_pnl"]
        results[name] = {"brier": b, "pnl_default": pnl_def, "pnl_tight_band": pnl_tb}
    return results


def main():
    train, test = time_split(load_subset_1200())
    print(f"Train: {len(train):,}   Test: {len(test):,}")

    # Fit data baselines on train
    platt = fit_platt_market(train)
    event_size = fit_event_size_platt(train)
    mean_bias = fit_mean_bias(train)

    # Predictors (raw functions for backtest)
    def platt_pred(e, m): return platt(e, m)["p_yes"]
    def event_size_pred(e, m): return event_size(e, m)["p_yes"]
    def mean_bias_pred(e, m): return mean_bias(e, m)["p_yes"]

    predictors = {
        "market (just q)":   q_only,
        "mean_bias_market":  mean_bias_pred,
        "platt_market":      platt_pred,
        "event_size_platt":  event_size_pred,
    }

    # On the FULL test split
    print("\n=== Full test split (N=2,090) ===")
    full = evaluate_subsample(test, predictors)
    for name, r in full.items():
        print(f"  {name:22s}  Brier {r['brier']:.4f}  "
              f"P&L_default ${r['pnl_default']:+,.2f}  "
              f"P&L_tight_band ${r['pnl_tight_band']:+,.2f}")

    # Bootstrap N=200 subsamples
    N = 200
    N_BOOT = 2000
    print(f"\n=== Bootstrap {N_BOOT} subsamples of size N={N} from the test split ===")
    rng = random.Random(42)
    boot = {name: {"brier": [], "pnl_default": [], "pnl_tight_band": []} for name in predictors}
    test_event_size._attach(test, event_size)  # ensure event_size_platt has test sizes

    for b in range(N_BOOT):
        sub = [rng.choice(test) for _ in range(N)]
        res = evaluate_subsample(sub, predictors)
        for name, r in res.items():
            for metric in ("brier", "pnl_default", "pnl_tight_band"):
                boot[name][metric].append(r[metric])

    print(f"\n  {'Baseline':22s}  {'Brier (mean ± 95% CI)':>28}  "
          f"{'P&L_default ($)':>26}  {'P&L_tight_band ($)':>26}")
    for name in predictors:
        b_arr = np.array(boot[name]["brier"])
        p_def_arr = np.array(boot[name]["pnl_default"])
        p_tb_arr = np.array(boot[name]["pnl_tight_band"])
        b_lo, b_hi = np.percentile(b_arr, [2.5, 97.5])
        p_def_lo, p_def_hi = np.percentile(p_def_arr, [2.5, 97.5])
        p_tb_lo, p_tb_hi = np.percentile(p_tb_arr, [2.5, 97.5])
        print(f"  {name:22s}  "
              f"{b_arr.mean():.4f} [{b_lo:.3f},{b_hi:.3f}]   "
              f"{p_def_arr.mean():+7.1f} [{p_def_lo:+5.0f},{p_def_hi:+5.0f}]   "
              f"{p_tb_arr.mean():+7.1f} [{p_tb_lo:+5.0f},{p_tb_hi:+5.0f}]")

    # PAIRED comparisons: is event_size_platt vs q distinguishable at N=200?
    print(f"\n=== Paired differences (event_size_platt − market) per bootstrap sample ===")
    diff_b = np.array(boot["event_size_platt"]["brier"]) - np.array(boot["market (just q)"]["brier"])
    diff_p = np.array(boot["event_size_platt"]["pnl_default"]) - np.array(boot["market (just q)"]["pnl_default"])
    print(f"  Brier:        mean Δ {diff_b.mean():+.4f}   "
          f"95% CI [{np.percentile(diff_b, 2.5):+.4f}, {np.percentile(diff_b, 97.5):+.4f}]   "
          f"P(Δ < 0): {(diff_b < 0).mean():.3f}  (lower is better)")
    print(f"  P&L_default:  mean Δ {diff_p.mean():+.1f}   "
          f"95% CI [{np.percentile(diff_p, 2.5):+.0f}, {np.percentile(diff_p, 97.5):+.0f}]   "
          f"P(Δ > 0): {(diff_p > 0).mean():.3f}  (higher is better)")


def _attach(test, event_size):
    if hasattr(event_size, "attach_test_sizes"):
        event_size.attach_test_sizes(test)
test_event_size = type("T", (), {"_attach": staticmethod(_attach)})


if __name__ == "__main__":
    main()
