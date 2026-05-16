"""Walk-forward cross-validation of the data-fair-price baselines.

The +$108 result from `forecast_benchmarks.py --fair-price-split` is one
test on one split. This script answers: is that number robust, or did
we luck out on the 70/30 cut?

Procedure
---------
1. Sort subset_1200 submissions by snapshot_time.
2. Define K time-ordered folds. Each fold uses the FIRST i/K of data
   as training and the next 1/K as test. (Walk-forward — never train
   on data later than the test set.)
3. For each fold, fit every data-fair-price baseline and report:
     - test Brier, log_loss, BSS_vs_market
     - test P&L under tight_band, default, min_edge
4. Print mean ± std across folds. Also print fold-by-fold P&L so we
   can spot any single-fold flukes.

Usage:
    python scripts/walk_forward_cv.py
    python scripts/walk_forward_cv.py --folds 5
    python scripts/walk_forward_cv.py --folds 5 --min-train 0.4
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from ast import literal_eval
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from prep.baselines.data_fair_price import (  # noqa: E402
    fit_decile_isotonic,
    fit_event_size_platt,
    fit_event_size_platt_v2,
    fit_gated_platt,
    fit_hierarchical_platt,
    fit_mean_bias,
    fit_multi_feature,
    fit_platt_market,
    fit_platt_max_pnl,
)
from prep.data import Sample  # noqa: E402
from prep.score import brier, log_loss  # noqa: E402
from prep.trade import (  # noqa: E402
    backtest,
    default_min_edge_strategy,
    default_strategy,
    default_tight_band_strategy,
    market_mid_forecast,
)


def _submission_to_samples(row) -> list[Sample]:
    try:
        outcomes = literal_eval(row["market_outcome"]) or {}
        market_data = literal_eval(row["market_data"]) or {}
    except Exception:
        return []
    out: list[Sample] = []
    for market_name, outcome in outcomes.items():
        md = market_data.get(market_name) or {}
        event = {
            "event_ticker": row["event_ticker"],
            "market_ticker": f"{row['event_ticker']}-{market_name.replace(' ', '_')}",
            "title": row.get("title") or "",
            "subtitle": market_name,
            "description": None,
            "category": row.get("category") or "Other",
            "rules": row.get("rules") or None,
            "close_time": row.get("close_time") or "",
        }
        out.append(Sample(event=event, market_info=md, outcome=int(outcome)))
    return out


def load_sorted_submissions() -> list[list[Sample]]:
    """One inner list per submission, time-ordered. Splitting by
    submission keeps related markets of one event together (no leakage)."""
    csv = Path(__file__).resolve().parents[1] / "data" / "external" / "subset_1200.csv"
    df = pd.read_csv(csv).sort_values("snapshot_time").reset_index(drop=True)
    return [_submission_to_samples(row) for _, row in df.iterrows()]


def fold_indices(n: int, n_folds: int, min_train_frac: float = 0.3):
    """Walk-forward folds. Each fold tests on a non-overlapping 1/K slice,
    training on everything before it. The first fold needs at least
    `min_train_frac` of the data as training history."""
    start_test = max(int(n * min_train_frac), n // n_folds)
    test_size = (n - start_test) // n_folds
    for i in range(n_folds):
        tr_end = start_test + i * test_size
        te_end = min(n, tr_end + test_size)
        if te_end <= tr_end:
            continue
        yield i, tr_end, te_end


def _market_q_list(samples):
    qs = []
    for s in samples:
        try:
            qs.append(max(0.01, min(0.99, market_mid_forecast(s.event, s.market_info))))
        except Exception:
            qs.append(0.5)
    return qs


def evaluate(predict_fn, test_samples):
    """Compute Brier, log_loss, BSS_vs_market, and per-strategy P&L."""
    p_yes = []
    outcomes = []
    market_q = []
    for s in test_samples:
        try:
            p = predict_fn(s.event, s.market_info)
        except Exception:
            continue
        try:
            q = market_mid_forecast(s.event, s.market_info)
        except Exception:
            continue
        p_yes.append(max(0.01, min(0.99, float(p))))
        outcomes.append(int(s.outcome))
        market_q.append(max(0.01, min(0.99, q)))

    b = brier(p_yes, outcomes)
    ll = log_loss(p_yes, outcomes)
    b_mkt = brier(market_q, outcomes)
    bss_mkt = 1 - b / b_mkt if b_mkt > 0 else float("nan")

    res = {"brier": b, "log_loss": ll, "bss_vs_market": bss_mkt, "n": len(p_yes)}
    for strat_name, strat in (
        ("tight_band", default_tight_band_strategy),
        ("default", default_strategy),
        ("min_edge", default_min_edge_strategy),
    ):
        r = backtest(test_samples, forecast_fn=predict_fn, strategy=strat)
        res[f"pnl_{strat_name}"] = r["total_pnl"]
        res[f"trades_{strat_name}"] = r["n_trades"]
    return res


FITTERS = {
    "dfp_mean_bias":          fit_mean_bias,
    "dfp_platt_market":       fit_platt_market,
    "dfp_decile_isotonic":    fit_decile_isotonic,
    "dfp_multi_feature":      fit_multi_feature,
    "dfp_platt_max_pnl":      fit_platt_max_pnl,
    "dfp_event_size_platt":   fit_event_size_platt,    # NEW
    "dfp_event_size_v2":      fit_event_size_platt_v2,  # NEW (with interaction)
    "dfp_hierarchical_platt": fit_hierarchical_platt,  # NEW
    "dfp_gated_platt":        fit_gated_platt,         # NEW
}


def _market_predict(e, m):
    return max(0.01, min(0.99, market_mid_forecast(e, m)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train", type=float, default=0.30,
                        help="first fold needs at least this fraction as training history")
    args = parser.parse_args()

    submissions = load_sorted_submissions()
    n = len(submissions)
    print(f"Loaded {n} submissions (time-ordered). Running {args.folds} walk-forward folds.")

    fold_results: list[dict] = []
    for i, tr_end, te_end in fold_indices(n, args.folds, args.min_train):
        train_subs = submissions[:tr_end]
        test_subs = submissions[tr_end:te_end]
        train = [s for subs in train_subs for s in subs]
        test = [s for subs in test_subs for s in subs]
        if not train or not test:
            continue

        print(f"\nFold {i+1}: train submissions [0, {tr_end}) → "
              f"{len(train):,} markets;  "
              f"test [{tr_end}, {te_end}) → {len(test):,} markets")

        # Anchor: raw market (for BSS reference + comparison)
        fold = {"fold": i + 1, "n_train": len(train), "n_test": len(test)}
        fold["market"] = evaluate(_market_predict, test)

        for name, fitter in FITTERS.items():
            try:
                model = fitter(train)
                # The event-size predictor needs test-set event counts
                if hasattr(model, "attach_test_sizes"):
                    model.attach_test_sizes(test)
                pred = lambda e, m, _f=model: _f(e, m)["p_yes"]
                fold[name] = evaluate(pred, test)
            except Exception as exc:
                import traceback; traceback.print_exc()
                fold[name] = {"error": str(exc)}

        fold_results.append(fold)

    # Aggregate
    print("\n" + "=" * 110)
    print("WALK-FORWARD CV — mean P&L (and Brier) across folds")
    print("=" * 110)
    print(f"  {'Forecaster':22s}  {'tight_band $':>20}  {'default $':>20}  "
          f"{'min_edge $':>20}  {'Brier':>14}")
    print(f"  {'-'*22}  {'-'*20}  {'-'*20}  {'-'*20}  {'-'*14}")

    forecasters = ["market", *FITTERS.keys()]
    for name in forecasters:
        per_strat = {strat: [] for strat in ("tight_band", "default", "min_edge")}
        briers = []
        for fold in fold_results:
            f = fold.get(name)
            if not f or "error" in f:
                continue
            for strat in per_strat:
                per_strat[strat].append(f.get(f"pnl_{strat}", float("nan")))
            briers.append(f["brier"])

        def stats(vals):
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                return float("nan"), float("nan")
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return m, sd

        ms = {strat: stats(per_strat[strat]) for strat in per_strat}
        bm, bs = stats(briers)

        def fmt(m, s):
            if math.isnan(m):
                return "  n/a"
            return f"{m:>+10.2f} ± {s:>6.2f}"

        print(f"  {name:22s}  {fmt(*ms['tight_band']):>20}  {fmt(*ms['default']):>20}  "
              f"{fmt(*ms['min_edge']):>20}  {bm:>6.4f} ± {bs:>5.4f}")

    # Fold-by-fold detail for the leaderboard contenders
    print()
    print("=" * 110)
    print("FOLD-BY-FOLD detail (default strategy P&L)")
    print("=" * 110)
    hdr = f"  {'Forecaster':22s} "
    for fold in fold_results:
        hdr += f" {('F'+str(fold['fold'])):>12s}"
    hdr += f"  {'mean':>12s}  {'min':>10}"
    print(hdr)
    for name in forecasters:
        line = f"  {name:22s} "
        vals = []
        for fold in fold_results:
            f = fold.get(name)
            v = f.get("pnl_default", float("nan")) if f and "error" not in f else float("nan")
            vals.append(v)
            line += f"  {v:>+10.2f}"
        valid = [v for v in vals if not math.isnan(v)]
        if valid:
            line += f"  {statistics.mean(valid):>+10.2f}"
            line += f"  {min(valid):>+10.2f}"
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
