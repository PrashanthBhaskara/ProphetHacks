"""Diagnostic suite for the suspicious bid-vs-ask Brier gap.

The initial bake-off found ~0.035 Brier improvement from bid-based formulas
over the current ask-based market_mid. This is 2.5x what the team's
validated `RecommendedPredictor` achieves — suspicious.

Possible explanations:
  H1. Real signal — bids react faster than asks; team missed this axis.
  H2. Contamination — subset_1200 captured bids at a different (later) time
      than asks, leaking post-snapshot price action into one side.
  H3. Time-to-close artifact — bid advantage concentrated in near-close
      markets where settlement is partially known.
  H4. A handful of extreme events (e.g., one strike grid) dominate.
  H5. Stale asks: MMs cancel asks slower than they refresh bids.
  H6. The team's reported 0.185 baseline uses a different q formula than
      what we're calling current_mid, making our baselines incomparable.

This script runs the diagnostics needed to distinguish them. No code change
should follow unless the finding survives every test below.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


# ── Reuse the same primitives as the bake-off ─────────────────────────────

def to_dollar(x):
    if x is None:
        return None
    return float(x) / 100.0 if x > 1.0 else float(x)


def clamp(p, lo=0.01, hi=0.99):
    return max(lo, min(hi, p))


def brier(probs, outcomes):
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / max(1, len(probs))


def f_current_mid(q):
    ya, na = q["yes_ask"], q["no_ask"]
    if ya is None or na is None:
        return None
    return clamp((ya + (1.0 - na)) / 2.0)


def f_bid_only_mid(q):
    yb, nb = q["yes_bid"], q["no_bid"]
    if yb is None or nb is None:
        return None
    return clamp((yb + (1.0 - nb)) / 2.0)


def f_four_price_mid(q):
    yb, ya, nb, na = q["yes_bid"], q["yes_ask"], q["no_bid"], q["no_ask"]
    if any(x is None for x in (yb, ya, nb, na)):
        return None
    return clamp(((yb + ya) / 2.0 + 1.0 - (nb + na) / 2.0) / 2.0)


def parse_iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def load_rows(path):
    """Load subset_1200 with per-market timing fields preserved."""
    df = pd.read_csv(path)
    out = []
    for _, ev in df.iterrows():
        try:
            mdata = json.loads(ev["market_data"])
            mout = json.loads(ev["market_outcome"])
        except Exception:
            continue
        snap = parse_iso(ev.get("snapshot_time"))
        close = parse_iso(ev.get("close_time"))
        hours_to_close = None
        if snap and close:
            hours_to_close = max(0.0, (close - snap).total_seconds() / 3600.0)
        for name, prices in mdata.items():
            if name not in mout:
                continue
            out.append({
                "event_ticker": ev["event_ticker"],
                "market_name": name,
                "category": ev.get("category", "Other"),
                "snapshot_time": snap,
                "close_time": close,
                "hours_to_close": hours_to_close,
                "yes_bid": to_dollar(prices.get("yes_bid")),
                "yes_ask": to_dollar(prices.get("yes_ask")),
                "no_bid": to_dollar(prices.get("no_bid")),
                "no_ask": to_dollar(prices.get("no_ask")),
                "liquidity": prices.get("liquidity"),
                "outcome": int(mout[name]),
            })
    return out


# ── Diagnostic 1: arbitrage relations ─────────────────────────────────────

def diagnose_arb(rows):
    """If yes_ask + no_bid ≈ 1 always, bid-only and ask-only would give
    the same answer (both reduce to the yes-side mid). Departures from
    the arb relation are where the formulas can disagree."""
    print("=" * 70)
    print("DIAGNOSTIC 1 — arbitrage relations on the bid/ask quotes")
    print("=" * 70)
    yapnb_dev = []  # yes_ask + no_bid - 1
    ybpna_dev = []  # yes_bid + no_ask - 1
    yapnap = []     # yes_ask + no_ask - 1 (spread proxy; should be ≥ 0)
    ybpnbp = []     # yes_bid + no_bid - 1 (should be ≤ 0)
    for r in rows:
        if all(r[k] is not None for k in ("yes_ask", "yes_bid", "no_ask", "no_bid")):
            yapnb_dev.append(r["yes_ask"] + r["no_bid"] - 1.0)
            ybpna_dev.append(r["yes_bid"] + r["no_ask"] - 1.0)
            yapnap.append(r["yes_ask"] + r["no_ask"] - 1.0)
            ybpnbp.append(r["yes_bid"] + r["no_bid"] - 1.0)
    def stats(xs):
        xs_sorted = sorted(xs)
        n = len(xs)
        return {
            "mean": statistics.mean(xs),
            "median": xs_sorted[n // 2],
            "p10": xs_sorted[n // 10],
            "p90": xs_sorted[9 * n // 10],
            "abs_mean": statistics.mean(abs(x) for x in xs),
        }
    print(f"N = {len(yapnb_dev)}")
    print(f"\n  yes_ask + no_bid - 1 (should be 0 in perfect arb):")
    s = stats(yapnb_dev); print(f"    mean={s['mean']:+.4f}  median={s['median']:+.4f}  p10={s['p10']:+.4f}  p90={s['p90']:+.4f}  |mean|={s['abs_mean']:.4f}")
    print(f"\n  yes_bid + no_ask - 1 (should be 0 in perfect arb):")
    s = stats(ybpna_dev); print(f"    mean={s['mean']:+.4f}  median={s['median']:+.4f}  p10={s['p10']:+.4f}  p90={s['p90']:+.4f}  |mean|={s['abs_mean']:.4f}")
    print(f"\n  yes_ask + no_ask - 1 (spread; should be ≥ 0):")
    s = stats(yapnap); print(f"    mean={s['mean']:+.4f}  median={s['median']:+.4f}  p10={s['p10']:+.4f}  p90={s['p90']:+.4f}")
    print(f"\n  yes_bid + no_bid - 1 (should be ≤ 0):")
    s = stats(ybpnbp); print(f"    mean={s['mean']:+.4f}  median={s['median']:+.4f}  p10={s['p10']:+.4f}  p90={s['p90']:+.4f}")
    print()
    # Subset where arb is tight
    tight = [r for r in rows if all(r[k] is not None for k in ("yes_ask", "yes_bid", "no_ask", "no_bid"))
             and abs(r["yes_ask"] + r["no_bid"] - 1) < 0.01
             and abs(r["yes_bid"] + r["no_ask"] - 1) < 0.01]
    print(f"  Subset with TIGHT arb (both sums within 0.01 of 1): N = {len(tight)} ({100*len(tight)/max(1,len(yapnb_dev)):.1f}%)")
    if tight:
        b_curr = brier([f_current_mid(r) for r in tight], [r["outcome"] for r in tight])
        b_bid = brier([f_bid_only_mid(r) for r in tight], [r["outcome"] for r in tight])
        print(f"    Brier current_mid: {b_curr:.4f}    Brier bid_only: {b_bid:.4f}    Δ: {b_bid - b_curr:+.4f}")
    print()


# ── Diagnostic 2: time-to-close strata ────────────────────────────────────

def diagnose_horizon(rows):
    print("=" * 70)
    print("DIAGNOSTIC 2 — Brier by time-to-close")
    print("=" * 70)
    buckets = [
        ("< 1h", lambda h: h is not None and h < 1),
        ("1-6h", lambda h: h is not None and 1 <= h < 6),
        ("6-24h", lambda h: h is not None and 6 <= h < 24),
        ("1-7d", lambda h: h is not None and 24 <= h < 24 * 7),
        ("7-30d", lambda h: h is not None and 24 * 7 <= h < 24 * 30),
        ("> 30d", lambda h: h is not None and h >= 24 * 30),
        ("missing", lambda h: h is None),
    ]
    print(f"{'Bucket':<10} {'N':>6} {'current':>9} {'bid_only':>9} {'four_price':>11} {'Δ_bid_vs_curr':>14}")
    print("-" * 70)
    for label, pred in buckets:
        sub = [r for r in rows if pred(r["hours_to_close"])]
        if len(sub) < 50:
            continue
        b_curr, b_bid, b_four = [], [], []
        outs = [r["outcome"] for r in sub]
        for r in sub:
            p1 = f_current_mid(r); p2 = f_bid_only_mid(r); p3 = f_four_price_mid(r)
            if p1 is not None: b_curr.append(p1)
            if p2 is not None: b_bid.append(p2)
            if p3 is not None: b_four.append(p3)
        bs1 = brier(b_curr, outs[:len(b_curr)])
        bs2 = brier(b_bid, outs[:len(b_bid)])
        bs3 = brier(b_four, outs[:len(b_four)])
        print(f"{label:<10} {len(sub):>6} {bs1:>9.4f} {bs2:>9.4f} {bs3:>11.4f} {bs2-bs1:>+14.4f}")
    print()


# ── Diagnostic 3: time-split holdout (latest 30% by snapshot_time) ────────

def diagnose_holdout(rows):
    print("=" * 70)
    print("DIAGNOSTIC 3 — latest-30%-by-snapshot-time holdout (team convention)")
    print("=" * 70)
    timed = [r for r in rows if r["snapshot_time"] is not None]
    timed.sort(key=lambda r: r["snapshot_time"])
    split = int(0.7 * len(timed))
    holdout = timed[split:]
    print(f"  Full set: {len(rows)}    Timed: {len(timed)}    Holdout (last 30%): {len(holdout)}")
    print(f"  Holdout time range: {holdout[0]['snapshot_time']} → {holdout[-1]['snapshot_time']}")
    print()
    formulas = {"current_mid": f_current_mid, "bid_only_mid": f_bid_only_mid, "four_price_mid": f_four_price_mid}
    print(f"{'Formula':<20} {'Brier_holdout':>14} {'N':>6}")
    print("-" * 50)
    for name, fn in formulas.items():
        outs, preds = [], []
        for r in holdout:
            p = fn(r)
            if p is not None:
                preds.append(p); outs.append(r["outcome"])
        b = brier(preds, outs) if preds else float("nan")
        print(f"{name:<20} {b:>14.4f} {len(preds):>6}")
    print()


# ── Diagnostic 4: leave-one-event-out — does any event dominate? ──────────

def diagnose_event_concentration(rows):
    print("=" * 70)
    print("DIAGNOSTIC 4 — per-event contribution to the bid vs ask gap")
    print("=" * 70)
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event_ticker"]].append(r)
    # For each event, compute (Brier_current - Brier_bid) — positive means bid wins
    gaps = []
    for evt, ev_rows in by_event.items():
        outs = [r["outcome"] for r in ev_rows]
        preds_c = [f_current_mid(r) for r in ev_rows]
        preds_b = [f_bid_only_mid(r) for r in ev_rows]
        if any(p is None for p in preds_c + preds_b):
            continue
        b_c = brier(preds_c, outs)
        b_b = brier(preds_b, outs)
        gaps.append((b_c - b_b, len(ev_rows), evt))
    gaps.sort(reverse=True)
    print(f"  Total events with complete data: {len(gaps)}")
    print(f"  Mean gap (current - bid) per event: {statistics.mean(g[0] for g in gaps):+.4f}")
    print(f"  Median gap: {sorted(g[0] for g in gaps)[len(gaps)//2]:+.4f}")
    print(f"  Top 10 events where bid_only wins by largest margin:")
    for gap, n, evt in gaps[:10]:
        print(f"    {evt:<40} N={n:>3}  gap=+{gap:.4f}")
    print(f"  Top 10 events where current_mid wins by largest margin:")
    for gap, n, evt in gaps[-10:]:
        print(f"    {evt:<40} N={n:>3}  gap={gap:+.4f}")
    # If we remove top-K events, does the gap collapse?
    print()
    print("  Sensitivity: aggregate Brier gap after removing top-K bid-winning events")
    all_outs = [r["outcome"] for r in rows]
    all_preds_c = [f_current_mid(r) for r in rows]
    all_preds_b = [f_bid_only_mid(r) for r in rows]
    valid_idx = [i for i in range(len(rows)) if all_preds_c[i] is not None and all_preds_b[i] is not None]
    print(f"  K=0:    Brier(current)={brier([all_preds_c[i] for i in valid_idx], [all_outs[i] for i in valid_idx]):.4f}  Brier(bid)={brier([all_preds_b[i] for i in valid_idx], [all_outs[i] for i in valid_idx]):.4f}")
    for K in (5, 10, 25, 50):
        excluded = set(g[2] for g in gaps[:K])
        idx = [i for i in valid_idx if rows[i]["event_ticker"] not in excluded]
        b_c = brier([all_preds_c[i] for i in idx], [all_outs[i] for i in idx])
        b_b = brier([all_preds_b[i] for i in idx], [all_outs[i] for i in idx])
        print(f"  K={K:<3}: Brier(current)={b_c:.4f}  Brier(bid)={b_b:.4f}  gap={b_c-b_b:+.4f}")
    print()


# ── Diagnostic 5: stale-quote test ────────────────────────────────────────

def diagnose_extreme_prices(rows):
    """If asks are stale on near-resolved markets (e.g., price has moved
    deep in-the-money but asks haven't refreshed), the bid would carry
    the new info first. Check whether the gap is concentrated in
    near-0 or near-1 markets."""
    print("=" * 70)
    print("DIAGNOSTIC 5 — Brier gap by current_mid value (price-strata)")
    print("=" * 70)
    bands = [(0.00, 0.10), (0.10, 0.25), (0.25, 0.40), (0.40, 0.60),
             (0.60, 0.75), (0.75, 0.90), (0.90, 1.00)]
    print(f"{'Band':<14} {'N':>6} {'current':>9} {'bid_only':>9} {'gap':>9}")
    print("-" * 60)
    for lo, hi in bands:
        sub = []
        for r in rows:
            p = f_current_mid(r)
            if p is None: continue
            if lo <= p < hi:
                sub.append(r)
        if len(sub) < 20:
            continue
        outs = [r["outcome"] for r in sub]
        b_c = brier([f_current_mid(r) for r in sub], outs)
        b_b_preds, b_b_outs = [], []
        for r in sub:
            p = f_bid_only_mid(r)
            if p is not None:
                b_b_preds.append(p); b_b_outs.append(r["outcome"])
        b_b = brier(b_b_preds, b_b_outs) if b_b_preds else float("nan")
        print(f"[{lo:.2f},{hi:.2f}) {len(sub):>6} {b_c:>9.4f} {b_b:>9.4f} {b_c - b_b:>+9.4f}")
    print()


# ── Diagnostic 6: cross-check vs team's RecommendedPredictor ──────────────

def diagnose_vs_recommended(rows, repo_root):
    """If we plug bid_only_mid into RecommendedPredictor as the q input,
    does the Brier improvement compound or wash out? If the team's
    pipeline gets to 0.171 with ask-based q, what does it get with bid-based q?"""
    print("=" * 70)
    print("DIAGNOSTIC 6 — RecommendedPredictor baselines for sanity calibration")
    print("=" * 70)
    # We don't have the team's RecommendedPredictor wired up in this worktree,
    # but we can compute the headline numbers with simple substitutes.
    # Headline expectation per FORECAST_BENCHMARKS.md: market q baseline ≈ 0.185 on
    # holdout. If our current_mid on holdout ≈ 0.165, our baselines are
    # already ~0.02 different from the team's — meaning q formulas differ.
    print("  (RecommendedPredictor not loaded in this worktree; relying on")
    print("   the Diagnostic 3 holdout numbers above for comparison to the")
    print("   team's FORECAST_BENCHMARKS.md baseline 0.185.)")
    print()


def main():
    here = Path(__file__).resolve().parent.parent
    rows = load_rows(here / "data" / "external" / "subset_1200.csv")
    print(f"Loaded {len(rows)} markets across {len(set(r['event_ticker'] for r in rows))} events\n")
    diagnose_arb(rows)
    diagnose_horizon(rows)
    diagnose_holdout(rows)
    diagnose_event_concentration(rows)
    diagnose_extreme_prices(rows)
    diagnose_vs_recommended(rows, here.parent)


if __name__ == "__main__":
    main()
