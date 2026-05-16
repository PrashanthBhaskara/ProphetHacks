"""Sum-constrained predictor.

For each event of size N, the historical sum of YES outcomes in
subset_1200 is systematically LESS than the sum of market prices. This
script:

  1. Learns expected_sum_yes per event size from train
  2. At predict time, gets all markets in an event, predicts each via
     event_size_platt, then rescales so the sum matches expected_sum_yes
  3. Compares to event_size_platt (no constraint) and market

Bootstrap N=200 at the END to see if the gain survives at eval scale.
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
from prep.baselines.data_fair_price import fit_event_size_platt, fit_platt_market  # noqa: E402
from prep.data import Sample, load_subset_1200  # noqa: E402
from prep.trade import backtest, default_strategy, market_mid_forecast  # noqa: E402


def q_only(e, m): return max(0.01, min(0.99, market_mid_forecast(e, m)))


def learn_expected_sum_by_size(train_samples):
    """For each event in train, sum_yes / event_size. Then bucket by size
    and return a lookup function size → expected_sum_per_size."""
    by_event = defaultdict(list)
    for s in train_samples:
        by_event[s.event["event_ticker"]].append(s)

    by_size_outcome = defaultdict(list)  # size → list of sum_yes
    for ev, samples in by_event.items():
        size = len(samples)
        sum_y = sum(s.outcome for s in samples)
        by_size_outcome[size].append(sum_y)

    # Smooth using nearest-size pooling
    size_to_expected = {}
    for size, sums in by_size_outcome.items():
        size_to_expected[size] = float(np.mean(sums))

    def lookup(size):
        if size in size_to_expected:
            return size_to_expected[size]
        # Pool: find nearest size in keys
        if not size_to_expected:
            return size * 0.5
        closest = min(size_to_expected.keys(), key=lambda k: abs(k - size))
        return size_to_expected[closest] * (size / closest)

    return lookup, size_to_expected


def sum_constrained_predict_event(event_samples, base_pred, expected_sum, clip=(0.3, 1.5)):
    """Given all sibling markets of one event + a base predictor, rescale
    so sum matches expected. Each market gets p_i = p_base_i * scale.
    Returns dict market_ticker → p_yes (clipped 0.01..0.99)."""
    base = []
    for s in event_samples:
        try:
            p = base_pred(s.event, s.market_info)
            if isinstance(p, dict):
                p = p["p_yes"]
            base.append(max(0.001, min(0.999, float(p))))
        except Exception:
            base.append(0.5)
    cur_sum = sum(base)
    if cur_sum <= 0:
        return {s.event["market_ticker"]: 0.5 for s in event_samples}
    scale = expected_sum / cur_sum
    scale = max(clip[0], min(clip[1], scale))
    out = {}
    for s, p in zip(event_samples, base):
        out[s.event["market_ticker"]] = max(0.01, min(0.99, p * scale))
    return out


def evaluate(samples, predict_dict):
    """predict_dict: ticker → p_yes."""
    preds, outs = [], []
    for s in samples:
        t = s.event["market_ticker"]
        if t in predict_dict:
            preds.append(predict_dict[t])
            outs.append(s.outcome)
    if not preds:
        return float("nan"), 0
    b = float(np.mean((np.asarray(preds) - np.asarray(outs)) ** 2))
    return b, len(preds)


def main():
    train, test = time_split(load_subset_1200())
    print(f"Train: {len(train):,}   Test: {len(test):,}")

    # Fit event_size_platt + look up expected sum per event size
    es = fit_event_size_platt(train)
    if hasattr(es, "attach_test_sizes"):
        es.attach_test_sizes(test)
    platt = fit_platt_market(train)
    lookup, table = learn_expected_sum_by_size(train)

    print("\nExpected sum_yes per event size (train):")
    for sz in sorted(table.keys())[:20]:
        print(f"  size {sz}: {table[sz]:.2f}")

    # Build prediction dicts on full test
    # 1) market
    pred_market = {s.event["market_ticker"]: q_only(s.event, s.market_info) for s in test}

    # 2) event_size_platt (no constraint)
    pred_es = {}
    for s in test:
        try:
            pred_es[s.event["market_ticker"]] = max(0.01, min(0.99, es(s.event, s.market_info)["p_yes"]))
        except Exception:
            pred_es[s.event["market_ticker"]] = 0.5

    # 3) sum-constrained event_size_platt
    by_event_test = defaultdict(list)
    for s in test:
        by_event_test[s.event["event_ticker"]].append(s)
    pred_sum = {}
    for ev, samples in by_event_test.items():
        size = len(samples)
        expected = lookup(size)
        d = sum_constrained_predict_event(samples, es, expected)
        pred_sum.update(d)

    # 4) sum-constrained PLATT (using simple platt, not event_size)
    pred_sum_platt = {}
    for ev, samples in by_event_test.items():
        size = len(samples)
        expected = lookup(size)
        d = sum_constrained_predict_event(samples, platt, expected)
        pred_sum_platt.update(d)

    # 5) sum-constrained MARKET (just the q's, rescaled to expected sum)
    pred_sum_market = {}
    for ev, samples in by_event_test.items():
        size = len(samples)
        expected = lookup(size)
        d = sum_constrained_predict_event(samples, lambda e, m: q_only(e, m), expected)
        pred_sum_market.update(d)

    # 6) shrunk 0.5·sum_constrained + 0.5·market
    pred_shrunk_sum = {}
    for t in pred_sum:
        if t in pred_market:
            pred_shrunk_sum[t] = 0.5 * pred_sum[t] + 0.5 * pred_market[t]

    # Full-test Brier
    print("\n=== Full test (N={}) ===".format(len(test)))
    for name, preds in [
        ("market", pred_market),
        ("event_size_platt", pred_es),
        ("0.5·event_size + 0.5·market", {t: 0.5*pred_es[t] + 0.5*pred_market[t] for t in pred_es}),
        ("sum_constrained · MARKET", pred_sum_market),
        ("sum_constrained · platt", pred_sum_platt),
        ("sum_constrained · event_size", pred_sum),
        ("0.5·sum_constrained_es + 0.5·market", pred_shrunk_sum),
    ]:
        b, n = evaluate(test, preds)
        print(f"  {name:46s}  Brier {b:.4f}  (n={n})")

    # Bootstrap N=200 on the best variants
    print(f"\n=== Bootstrap N=200, 1500 subsamples ===")
    predictors = {
        "market":                              pred_market,
        "event_size_platt":                    pred_es,
        "0.5·event_size + 0.5·market":         {t: 0.5*pred_es[t] + 0.5*pred_market[t] for t in pred_es},
        "sum_constrained · MARKET":            pred_sum_market,
        "sum_constrained · event_size":        pred_sum,
        "0.5·sum_constrained + 0.5·market":    pred_shrunk_sum,
    }
    N, N_BOOT = 200, 1500
    rng = random.Random(42)
    boot = {name: [] for name in predictors}
    for _ in range(N_BOOT):
        sub = [rng.choice(test) for _ in range(N)]
        for name, preds in predictors.items():
            b, _ = evaluate(sub, preds)
            boot[name].append(b)

    market_boot = np.array(boot["market"])
    print(f"\n  {'Predictor':46s}  {'Brier mean ± 95% CI':>26}  {'P(better)':>10}")
    for name in predictors:
        arr = np.array(boot[name])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p_better = (arr < market_boot).mean()
        print(f"  {name:46s}  {arr.mean():.4f} [{lo:.3f},{hi:.3f}]   {p_better:>10.3f}")


if __name__ == "__main__":
    main()
