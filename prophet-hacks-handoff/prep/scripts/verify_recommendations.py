"""Verification script — reproduces every headline number in
FORECAST_BENCHMARKS.md so teammates can audit the recommendation.

Runtime: ~30 seconds (data is in the repo; no API calls).

Outputs:
  1. Coefficients of the fitted RecommendedPredictor
  2. Full-test Brier under each of the 3 call modes (A/B/C)
  3. N=200 bootstrap statistics + P(beats market) under each mode
  4. Noise floor estimate

If any of these don't match the doc, please flag it.
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from n200_bootstrap import time_split  # noqa: E402
from prep.baselines.fair_price import RecommendedPredictor  # noqa: E402
from prep.data import load_subset_1200  # noqa: E402
from prep.trade import market_mid_forecast  # noqa: E402


def q_only(event, market_info):
    return max(0.01, min(0.99, market_mid_forecast(event, market_info)))


def brier(preds, outs):
    return float(np.mean((np.asarray(preds) - np.asarray(outs)) ** 2))


def main():
    train, test = time_split(load_subset_1200())
    predictor = RecommendedPredictor.fit(train)

    print("=" * 72)
    print("VERIFICATION — reproduces FORECAST_BENCHMARKS.md headline numbers")
    print("=" * 72)
    print(f"\nTrain: {len(train):,} markets  Test: {len(test):,} markets")
    print(f"YES rate (train): {sum(s.outcome for s in train)/len(train):.3f}")
    print(f"YES rate (test):  {sum(s.outcome for s in test)/len(test):.3f}")

    print(f"\nFitted RecommendedPredictor:")
    print(f"  bias                 = {predictor.bias:+.4f}")
    print(f"  logit_q_slope        = {predictor.logit_q_slope:+.4f}")
    print(f"  log_event_size_slope = {predictor.log_event_size_slope:+.4f}")
    print(f"  shrink_alpha         = {predictor.shrink_alpha}")
    print(f"  prefix table size    = {len(predictor.prefix_event_size)} entries")
    print(f"  default event size   = {predictor.default_event_size:.1f}")

    # === Full-test Brier under three modes ===
    print(f"\n{'Mode':40s}  {'Brier':>7}")
    print("-" * 50)
    p_market = [q_only(s.event, s.market_info) for s in test]
    y_test = [s.outcome for s in test]
    print(f"  {'market (just q)':<38s}  {brier(p_market, y_test):>7.4f}")

    # Mode A: no sibling context (uses prefix fallback)
    p_a = [predictor.predict(s.event, s.market_info) for s in test]
    print(f"  {'A — no sibling context':<38s}  {brier(p_a, y_test):>7.4f}")

    # Mode B: predict_batch on the entire test set (sibling groups intact)
    preds_b = predictor.predict_batch(test)
    p_b = [preds_b.get(s.event.get('market_ticker', ''), 0.5) for s in test]
    print(f"  {'B — sibling-group batch':<38s}  {brier(p_b, y_test):>7.4f}")

    # Mode C: full universe sibling counts known
    full_sizes = defaultdict(int)
    for s in test:
        full_sizes[s.event.get('event_ticker', '')] += 1
    p_c = [predictor.predict(s.event, s.market_info,
                             n_event=full_sizes[s.event.get('event_ticker', '')])
           for s in test]
    print(f"  {'C — full universe cached':<38s}  {brier(p_c, y_test):>7.4f}")

    # === N=200 bootstrap ===
    print(f"\nN=200 bootstrap (1500 resamples) — P(better than market):")
    print(f"  {'Mode':40s}  {'mean Brier':>10}  {'95% CI':>20}  {'P(better)':>10}")
    print("  " + "-" * 86)

    N, NBOOT = 200, 1500
    by_event_test = defaultdict(list)
    for s in test:
        by_event_test[s.event.get('event_ticker', '')].append(s)
    events_list = list(by_event_test.values())

    for mode_name, sampler in [
        ("market (baseline)", "market"),
        ("A — no sibling context", "A"),
        ("B — sibling-group batch", "B"),
        ("C — full universe cached", "C"),
    ]:
        rng = random.Random(42)
        briers = []
        market_briers = []
        for _ in range(NBOOT):
            if sampler in ("market", "A", "C"):
                sub = [rng.choice(test) for _ in range(N)]
            else:  # B: resample at event level
                sub = []
                while len(sub) < N:
                    sub.extend(rng.choice(events_list))
                sub = sub[:N]

            y_sub = [s.outcome for s in sub]
            m_brier = brier([q_only(s.event, s.market_info) for s in sub], y_sub)
            market_briers.append(m_brier)

            if sampler == "market":
                briers.append(m_brier)
            elif sampler == "A":
                p = [predictor.predict(s.event, s.market_info) for s in sub]
                briers.append(brier(p, y_sub))
            elif sampler == "B":
                preds = predictor.predict_batch(sub)
                p = [preds.get(s.event.get('market_ticker', ''), 0.5) for s in sub]
                briers.append(brier(p, y_sub))
            elif sampler == "C":
                p = [predictor.predict(s.event, s.market_info,
                                       n_event=full_sizes[s.event.get('event_ticker', '')])
                     for s in sub]
                briers.append(brier(p, y_sub))

        a = np.array(briers)
        m = np.array(market_briers)
        lo, hi = np.percentile(a, [2.5, 97.5])
        if sampler == "market":
            p_better = "—"
        else:
            p_better = f"{(a < m).mean():.3f}"
        print(f"  {mode_name:40s}  {a.mean():>10.4f}  [{lo:.4f}, {hi:.4f}]  {p_better:>10}")

    # === Variant comparison (all on the same train/test split) ===
    print(f"\nAll fitted variants on the SAME train/test split (N_test={len(test):,}):")
    print(f"  {'Variant':36s}  {'Brier':>7}")
    print("  " + "-" * 46)
    print(f"  {'just q (market)':36s}  {brier(p_market, y_test):>7.4f}")

    from prep.baselines.data_fair_price import (
        fit_beta_calibration,
        fit_category_platt,
        fit_decile_isotonic,
        fit_hierarchical_platt,
        fit_mean_bias,
        fit_multi_feature,
        fit_platt_market,
        fit_platt_max_pnl,
    )

    def score(predict_fn):
        ps = [max(0.01, min(0.99, predict_fn(s.event, s.market_info))) for s in test]
        return brier(ps, y_test)

    fitted = [
        ("mean_bias",        fit_mean_bias(train)),
        ("platt_market",     fit_platt_market(train)),
        ("beta_calibration", fit_beta_calibration(train)),
        ("decile_isotonic",  fit_decile_isotonic(train)),
        ("multi_feature",    fit_multi_feature(train)),
        ("category_platt",   fit_category_platt(train)),
        ("hierarchical_platt", fit_hierarchical_platt(train)),
        ("platt_max_pnl",    fit_platt_max_pnl(train)),
    ]
    for name, model in fitted:
        b = score(lambda e, m, _M=model: _M(e, m)["p_yes"])
        print(f"  {name:36s}  {b:>7.4f}")

    print(f"  {'event_size_platt (no shrinkage)':36s}  ", end="")
    es_no_shrink_predictor = RecommendedPredictor(
        bias=predictor.bias,
        logit_q_slope=predictor.logit_q_slope,
        log_event_size_slope=predictor.log_event_size_slope,
        prefix_event_size=predictor.prefix_event_size,
        default_event_size=predictor.default_event_size,
        shrink_alpha=1.0,  # no shrinkage
    )
    preds_es = es_no_shrink_predictor.predict_batch(test)
    p_es_no_shrink = [preds_es.get(s.event.get('market_ticker', ''), 0.5) for s in test]
    print(f"{brier(p_es_no_shrink, y_test):>7.4f}")
    print(f"  {'event_size_platt + shrink α=0.5 (this)':36s}  {brier(p_b, y_test):>7.4f}")

    # === Noise floor ===
    print(f"\nNoise floor at N=200:")
    rng = random.Random(42)
    market_briers = []
    for _ in range(NBOOT):
        sub = [rng.choice(test) for _ in range(N)]
        market_briers.append(brier([q_only(s.event, s.market_info) for s in sub],
                                   [s.outcome for s in sub]))
    sd = np.std(market_briers)
    print(f"  empirical σ(market Brier across resamples) = {sd:.4f}")
    print(f"  95% CI half-width                          = {(np.percentile(market_briers,97.5)-np.percentile(market_briers,2.5))/2:.4f}")
    print(f"  → min detectable Brier difference at N=200 ≈ {2*sd:.3f}")


if __name__ == "__main__":
    main()
