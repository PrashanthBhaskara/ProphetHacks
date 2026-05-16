"""Residual analysis of the winning data-fair-price baseline.

After CV, `dfp_platt_market` is our bar: +$110 ± $100 across 5 folds,
positive in all 5. This script breaks down WHERE that P&L came from
on the holdout test set:
    - per category
    - per market-price band (q ∈ [0,0.2], [0.2,0.4], ...)
    - per spread band (tight / mid / wide)

The output is a "loss map": which buckets did Platt lose money on?
Those buckets are exactly where the LLM agent needs to add information
the recalibration can't see. If a bucket already prints money, the LLM
should DEFER to the baseline there.

Usage:
    python scripts/residual_analysis.py
    python scripts/residual_analysis.py --strategy default
"""

from __future__ import annotations

import argparse
import sys
from ast import literal_eval
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from prep.baselines.data_fair_price import fit_platt_market  # noqa: E402
from prep.data import Sample  # noqa: E402
from prep.trade import (  # noqa: E402
    backtest,
    default_min_edge_strategy,
    default_strategy,
    default_tight_band_strategy,
    market_mid_forecast,
)


def _to_samples(row):
    try:
        outcomes = literal_eval(row["market_outcome"]) or {}
        market_data = literal_eval(row["market_data"]) or {}
    except Exception:
        return []
    out = []
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


def load_time_split(train_frac: float = 0.7):
    csv = Path(__file__).resolve().parents[1] / "data" / "external" / "subset_1200.csv"
    df = pd.read_csv(csv).sort_values("snapshot_time").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    train = [s for _, r in df.iloc[:cut].iterrows() for s in _to_samples(r)]
    test = [s for _, r in df.iloc[cut:].iterrows() for s in _to_samples(r)]
    return train, test


STRATEGIES = {
    "tight_band": default_tight_band_strategy,
    "default":    default_strategy,
    "min_edge":   default_min_edge_strategy,
}


def _q_band(q: float) -> str:
    edges = [0.0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0]
    labels = ["[0.00,0.10]", "[0.10,0.25]", "[0.25,0.40]",
              "[0.40,0.60]", "[0.60,0.75]", "[0.75,0.90]", "[0.90,1.00]"]
    for i in range(len(labels)):
        if q <= edges[i + 1]:
            return labels[i]
    return labels[-1]


def _spread_band(sp: float) -> str:
    if sp < 0.97:
        return "tight  (<0.97)"
    if sp < 1.02:
        return "mid   (0.97–1.02)"
    if sp < 1.10:
        return "wide  (1.02–1.10)"
    return "vwide (>1.10)"


def bucketed_pnl(trades, key_fn):
    by: dict[str, dict] = {}
    for t in trades:
        k = key_fn(t)
        d = by.setdefault(k, {"n": 0, "pnl": 0.0, "wins": 0})
        d["n"] += 1
        d["pnl"] += t.pnl
        d["wins"] += 1 if t.pnl > 0 else 0
    return by


def print_bucket(title, buckets, ordering=None):
    print()
    print(f"  {title}")
    print(f"    {'bucket':28s}  {'trades':>7}  {'P&L':>10}  {'$/trade':>9}  {'win%':>6}")
    keys = ordering or sorted(buckets.keys())
    total_pnl = 0.0
    total_n = 0
    for k in keys:
        if k not in buckets:
            continue
        d = buckets[k]
        per = d["pnl"] / d["n"] if d["n"] else 0.0
        wr = 100 * d["wins"] / d["n"] if d["n"] else 0.0
        flag = " <"  if d["pnl"] < 0 else "  "
        print(f"    {k:28s}  {d['n']:>7,}  ${d['pnl']:>+9,.2f}  ${per:>+8.2f}  {wr:>5.1f}%{flag}")
        total_pnl += d["pnl"]
        total_n += d["n"]
    if total_n:
        print(f"    {'TOTAL':28s}  {total_n:>7,}  ${total_pnl:>+9,.2f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="default")
    args = parser.parse_args()

    train, test = load_time_split(0.7)
    print(f"Train: {len(train):,} markets   Test: {len(test):,} markets")

    platt = fit_platt_market(train)
    print(f"Platt fit: slope={platt.slope:.3f}, intercept={platt.intercept:.3f}\n")

    pred = lambda e, m: platt(e, m)["p_yes"]
    res = backtest(test, forecast_fn=pred, strategy=STRATEGIES[args.strategy])
    print(f"Test P&L ({args.strategy}): ${res['total_pnl']:+,.2f} "
          f"over {res['n_trades']:,} trades")

    trades = res["trades"]

    # Bucket 1: by category
    by_cat = bucketed_pnl(trades, lambda t: t.category)
    print_bucket("By category", by_cat)

    # Bucket 2: by entry-price band (the price the agent paid)
    by_q = bucketed_pnl(trades, lambda t: _q_band(t.price))
    print_bucket("By entry price band (yes_ask or no_ask)", by_q,
                 ordering=["[0.00,0.10]", "[0.10,0.25]", "[0.25,0.40]",
                          "[0.40,0.60]", "[0.60,0.75]", "[0.75,0.90]", "[0.90,1.00]"])

    # Bucket 3: by side
    by_side = bucketed_pnl(trades, lambda t: t.side.upper())
    print_bucket("By side (YES vs NO)", by_side)

    # Bucket 4: by (category × side) — surfaces "Sports YES loses, Sports NO wins"
    by_cat_side = bucketed_pnl(trades, lambda t: f"{t.category} / {t.side.upper()}")
    print_bucket("By category × side", by_cat_side)

    # Negative-P&L buckets are the LLM's opportunity. Positive-P&L
    # buckets are where the data baseline already wins — the LLM
    # should DEFER there (just use Platt) rather than overriding.
    print()
    print("=" * 110)
    print("Where the LLM can add alpha (negative Platt P&L buckets above):")
    losers = [k for k, v in by_cat_side.items() if v["pnl"] < 0]
    winners = [k for k, v in by_cat_side.items() if v["pnl"] > 0]
    print(f"  - LLM should override Platt on: {', '.join(losers) if losers else '(none — defer everywhere)'}")
    print(f"  - LLM should defer to Platt on: {', '.join(winners) if winners else '(none — Platt only loses)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
