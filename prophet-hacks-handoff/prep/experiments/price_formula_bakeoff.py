"""Empirical bake-off of price formulas for the council anchor.

The current `KalshiQuote.market_mid` uses only the asks:
    p = (yes_ask + (1 - no_ask)) / 2

This script tests several alternatives on subset_1200 (the only dataset
that carries all four bid/ask prices per market). Each formula produces a
P(YES) per market; we score it with Brier on the resolved outcomes.

Caveat: subset_1200 is acknowledged contaminated in the team's memory.
Live/post-cutoff data (eval_pack_live_clean.jsonl, athetus_live/*.parquet)
omits NO-side bids, so the four-price variants can only be evaluated here.
Findings are advisory; differences within the 0.011 noise floor from
FORECAST_BENCHMARKS.md are not actionable.

Run:
    cd prep && python experiments/price_formula_bakeoff.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


def to_dollar(x: float | None) -> float | None:
    if x is None:
        return None
    return float(x) / 100.0 if x > 1.0 else float(x)


def brier(probs: list[float], outcomes: list[int]) -> float:
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / max(1, len(probs))


def clamp(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, p))


# ── Price formulas. Each takes a dict with yes_bid/yes_ask/no_bid/no_ask
# in dollars and returns P(YES) ∈ [0.01, 0.99], or None if insufficient data.

def f_current_mid(q: dict) -> float | None:
    """The team's current market_mid: average of YES-ask and (1 - NO-ask)."""
    ya, na = q["yes_ask"], q["no_ask"]
    if ya is None or na is None:
        return None
    return clamp((ya + (1.0 - na)) / 2.0)


def f_four_price_mid(q: dict) -> float | None:
    """Symmetric average over all four price levels."""
    yb, ya, nb, na = q["yes_bid"], q["yes_ask"], q["no_bid"], q["no_ask"]
    if any(x is None for x in (yb, ya, nb, na)):
        return None
    yes_side = (yb + ya) / 2.0
    no_side = 1.0 - (nb + na) / 2.0
    return clamp((yes_side + no_side) / 2.0)


def f_yes_side_mid(q: dict) -> float | None:
    """Use only the YES-side bid/ask midpoint."""
    yb, ya = q["yes_bid"], q["yes_ask"]
    if yb is None or ya is None:
        return None
    return clamp((yb + ya) / 2.0)


def f_no_side_mid(q: dict) -> float | None:
    """Use only the NO-side bid/ask midpoint, complemented."""
    nb, na = q["no_bid"], q["no_ask"]
    if nb is None or na is None:
        return None
    return clamp(1.0 - (nb + na) / 2.0)


def f_bid_only_mid(q: dict) -> float | None:
    """The mirror of current_mid using bids instead of asks (sanity check)."""
    yb, nb = q["yes_bid"], q["no_bid"]
    if yb is None or nb is None:
        return None
    return clamp((yb + (1.0 - nb)) / 2.0)


def f_conservative_ask(q: dict) -> float | None:
    """Pessimistic: use YES-ask for YES, NO-ask for NO. Implies p > 0.5 only
    when buying YES is cheaper than buying NO."""
    ya, na = q["yes_ask"], q["no_ask"]
    if ya is None or na is None:
        return None
    # Just yes_ask, treating ask as the "cost" of YES exposure
    return clamp(ya)


def f_asymmetric_2070(q: dict) -> float | None:
    """20% bid + 80% ask weight on each side."""
    yb, ya, nb, na = q["yes_bid"], q["yes_ask"], q["no_bid"], q["no_ask"]
    if any(x is None for x in (yb, ya, nb, na)):
        return None
    yes_side = 0.2 * yb + 0.8 * ya
    no_side = 1.0 - (0.2 * nb + 0.8 * na)
    return clamp((yes_side + no_side) / 2.0)


def f_asymmetric_8020(q: dict) -> float | None:
    """80% bid + 20% ask weight on each side."""
    yb, ya, nb, na = q["yes_bid"], q["yes_ask"], q["no_bid"], q["no_ask"]
    if any(x is None for x in (yb, ya, nb, na)):
        return None
    yes_side = 0.8 * yb + 0.2 * ya
    no_side = 1.0 - (0.8 * nb + 0.2 * na)
    return clamp((yes_side + no_side) / 2.0)


def f_spread_aware(q: dict) -> float | None:
    """If yes-side spread tighter than no-side, trust yes-side more."""
    yb, ya, nb, na = q["yes_bid"], q["yes_ask"], q["no_bid"], q["no_ask"]
    if any(x is None for x in (yb, ya, nb, na)):
        return None
    yes_spread = max(1e-4, ya - yb)
    no_spread = max(1e-4, na - nb)
    # Weight tighter-spread side more
    yes_weight = 1.0 / yes_spread
    no_weight = 1.0 / no_spread
    total = yes_weight + no_weight
    yes_side = (yb + ya) / 2.0
    no_side = 1.0 - (nb + na) / 2.0
    return clamp((yes_weight * yes_side + no_weight * no_side) / total)


def f_liquidity_naive(q: dict) -> float | None:
    """Liquidity isn't side-split; just falls back to four_price_mid.
    Included to confirm the placeholder doesn't accidentally help."""
    return f_four_price_mid(q)


FORMULAS = {
    "current_mid (asks only)": f_current_mid,
    "four_price_mid (symmetric)": f_four_price_mid,
    "yes_side_mid": f_yes_side_mid,
    "no_side_mid": f_no_side_mid,
    "bid_only_mid": f_bid_only_mid,
    "conservative_yes_ask": f_conservative_ask,
    "asym_20bid_80ask": f_asymmetric_2070,
    "asym_80bid_20ask": f_asymmetric_8020,
    "spread_aware": f_spread_aware,
}


def load_subset_1200_markets(path: Path) -> list[dict]:
    """Flatten subset_1200 events into per-market rows with bid/ask and outcome."""
    df = pd.read_csv(path)
    rows: list[dict] = []
    for _, ev in df.iterrows():
        try:
            mdata = json.loads(ev["market_data"])
            mout = json.loads(ev["market_outcome"])
        except (json.JSONDecodeError, TypeError):
            continue
        category = ev.get("category", "Other")
        event_ticker = ev["event_ticker"]
        for market_name, prices in mdata.items():
            if market_name not in mout:
                continue
            outcome = int(mout[market_name])
            row = {
                "event_ticker": event_ticker,
                "market_name": market_name,
                "category": category,
                "yes_bid": to_dollar(prices.get("yes_bid")),
                "yes_ask": to_dollar(prices.get("yes_ask")),
                "no_bid": to_dollar(prices.get("no_bid")),
                "no_ask": to_dollar(prices.get("no_ask")),
                "liquidity": prices.get("liquidity"),
                "outcome": outcome,
            }
            rows.append(row)
    return rows


def evaluate_formula(rows: list[dict], formula) -> tuple[float, int]:
    preds, outs = [], []
    for r in rows:
        p = formula(r)
        if p is None:
            continue
        preds.append(p)
        outs.append(r["outcome"])
    if not preds:
        return float("nan"), 0
    return brier(preds, outs), len(preds)


def bootstrap_se(rows: list[dict], formula, n_resamples: int = 500, sample_size: int = 200) -> float:
    """Empirical SE of Brier at N=sample_size, matching the team's noise-floor convention."""
    import random
    rng = random.Random(42)
    valid = [r for r in rows if formula(r) is not None]
    if len(valid) < sample_size:
        return float("nan")
    briers = []
    for _ in range(n_resamples):
        sample = rng.choices(valid, k=sample_size)
        preds = [formula(r) for r in sample]
        outs = [r["outcome"] for r in sample]
        briers.append(brier(preds, outs))
    mean = sum(briers) / len(briers)
    var = sum((b - mean) ** 2 for b in briers) / max(1, len(briers) - 1)
    return math.sqrt(var)


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    subset_path = here / "data" / "external" / "subset_1200.csv"
    print(f"Loading {subset_path}")
    rows = load_subset_1200_markets(subset_path)
    print(f"Loaded {len(rows)} markets across {len(set(r['event_ticker'] for r in rows))} events")
    print()

    # Aggregate by category for the breakdown
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_category[r["category"] or "Other"].append(r)

    # Overall bake-off
    print("=" * 75)
    print(f"{'Formula':<32} {'Brier':>8} {'N':>6} {'SE@200':>8} {'Δ_vs_current':>14}")
    print("=" * 75)
    baseline = None
    results = {}
    for name, fn in FORMULAS.items():
        b, n = evaluate_formula(rows, fn)
        se = bootstrap_se(rows, fn)
        results[name] = (b, n, se)
        if baseline is None:
            baseline = b
        delta = b - baseline
        marker = ""
        if abs(delta) > 0.02:
            marker = " ★" if delta < 0 else " ✗"
        elif abs(delta) > 0.011:
            marker = " ~"
        print(f"{name:<32} {b:>8.4f} {n:>6} {se:>8.4f} {delta:>+14.4f}{marker}")
    print()
    print("Legend: ★ improvement > noise floor; ~ within noise; ✗ regression > noise")
    print()

    # Per-category breakdown for top formulas
    print("=" * 75)
    print("Per-category Brier (top 4 formulas by aggregate)")
    print("=" * 75)
    top4 = sorted(results.items(), key=lambda kv: kv[1][0])[:4]
    cats = sorted(by_category.keys(), key=lambda c: -len(by_category[c]))
    header = f"{'Category':<20} {'N':>6} " + " ".join(f"{name.split(' ')[0][:14]:>14}" for name, _ in top4)
    print(header)
    print("-" * len(header))
    for cat in cats:
        cat_rows = by_category[cat]
        if len(cat_rows) < 20:
            continue
        line = f"{cat:<20} {len(cat_rows):>6} "
        for name, _ in top4:
            b, _ = evaluate_formula(cat_rows, FORMULAS[name])
            line += f"{b:>14.4f} "
        print(line)


if __name__ == "__main__":
    main()
