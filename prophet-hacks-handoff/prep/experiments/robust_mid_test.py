"""Test a robust mid that uses the bid-side as a fallback when asks are
degenerate. If the bid-vs-ask Brier gap is purely a "stale-ask" artifact,
a targeted robust mid should capture most of the gain without claiming
bids are universally better than asks.

Also: check whether the subset where the team's RecommendedPredictor
would naturally exclude / clip markets coincides with the markets that
drive the gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from price_formula_diagnose import (
    load_rows, brier, clamp, f_current_mid, f_bid_only_mid, f_four_price_mid,
)


def f_robust_mid(q):
    """Use current_mid except when asks are pinned at extremes (uninformative)
    OR yes_ask + no_ask is way above 1 (huge spread = MMs gave up); fall
    back to bid-side. Purely additive — common case unchanged."""
    ya = q.get("yes_ask")
    na = q.get("no_ask")
    yb = q.get("yes_bid")
    nb = q.get("no_bid")
    ask_mid = f_current_mid(q)
    bid_mid = f_bid_only_mid(q)
    # Degenerate-ask test: both pinned at 1.0 (or both at 0.0, or huge implied spread)
    if ya is not None and na is not None:
        ask_spread = ya + na  # should be ~1 for a healthy market; far above = wide
        both_pinned_high = ya >= 0.99 and na >= 0.99
        both_pinned_low = ya <= 0.02 and na <= 0.02
        huge_spread = ask_spread > 1.50  # implies wide bid-ask gap on either side
        if both_pinned_high or both_pinned_low or huge_spread:
            if bid_mid is not None:
                return bid_mid
    return ask_mid if ask_mid is not None else bid_mid


def f_four_price_with_floor(q):
    """four_price_mid except when only one side has data."""
    return f_four_price_mid(q) or f_current_mid(q) or f_bid_only_mid(q)


def main():
    here = Path(__file__).resolve().parent.parent
    rows = load_rows(here / "data" / "external" / "subset_1200.csv")
    print(f"Loaded {len(rows)} markets\n")

    formulas = {
        "current_mid (baseline)": f_current_mid,
        "bid_only_mid": f_bid_only_mid,
        "four_price_mid": f_four_price_mid,
        "robust_mid (ask + degeneracy fallback)": f_robust_mid,
        "four_price_with_floor": f_four_price_with_floor,
    }

    print("=" * 75)
    print("Full subset_1200 (N=6380)")
    print("=" * 75)
    baseline = None
    for name, fn in formulas.items():
        outs, preds = [], []
        for r in rows:
            p = fn(r)
            if p is not None:
                preds.append(p); outs.append(r["outcome"])
        b = brier(preds, outs)
        if baseline is None: baseline = b
        print(f"  {name:<40} Brier={b:.4f}  N={len(preds)}  Δ={b-baseline:+.4f}")
    print()

    # The 75% subset where arb is tight
    tight = []
    for r in rows:
        if all(r[k] is not None for k in ("yes_ask", "yes_bid", "no_ask", "no_bid")):
            if abs(r["yes_ask"] + r["no_bid"] - 1) < 0.01 and abs(r["yes_bid"] + r["no_ask"] - 1) < 0.01:
                tight.append(r)
    print("=" * 75)
    print(f"Tight-arb subset only (N={len(tight)}, 75.5%)")
    print("=" * 75)
    baseline = None
    for name, fn in formulas.items():
        outs, preds = [], []
        for r in tight:
            p = fn(r)
            if p is not None:
                preds.append(p); outs.append(r["outcome"])
        b = brier(preds, outs)
        if baseline is None: baseline = b
        print(f"  {name:<40} Brier={b:.4f}  N={len(preds)}  Δ={b-baseline:+.4f}")
    print()

    # The 25% subset where arb is broken
    broken = [r for r in rows if not (
        all(r[k] is not None for k in ("yes_ask", "yes_bid", "no_ask", "no_bid"))
        and abs(r["yes_ask"] + r["no_bid"] - 1) < 0.01
        and abs(r["yes_bid"] + r["no_ask"] - 1) < 0.01
    )]
    print("=" * 75)
    print(f"Broken-arb subset only (N={len(broken)}, 24.5%) — where the gap lives")
    print("=" * 75)
    baseline = None
    for name, fn in formulas.items():
        outs, preds = [], []
        for r in broken:
            p = fn(r)
            if p is not None:
                preds.append(p); outs.append(r["outcome"])
        b = brier(preds, outs)
        if baseline is None: baseline = b
        print(f"  {name:<40} Brier={b:.4f}  N={len(preds)}  Δ={b-baseline:+.4f}")
    print()

    # How many markets does robust_mid actually trigger the fallback on?
    triggered = 0
    for r in rows:
        ask_mid = f_current_mid(r)
        robust = f_robust_mid(r)
        if ask_mid is not None and robust is not None and abs(ask_mid - robust) > 0.001:
            triggered += 1
    print(f"robust_mid changes the answer on {triggered}/{len(rows)} markets ({100*triggered/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
