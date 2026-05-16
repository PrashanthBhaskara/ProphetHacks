"""Ingest the thomaswmitch HF Kalshi trades dataset into our eval_pack
format with rich trajectory data.

Source datasets (downloaded under data/external/):
  - kalshi_trades_*.parquet (5M trades, May–Jul 2025, schema:
    trade_id, ticker, count, created_time, yes_price, no_price, taker_side)
  - kalshi_markets.parquet  (10k markets, metadata + outcome)

Output:
  - data/eval_pack_hf.jsonl — same shape as eval_pack.jsonl but with
    trajectory built from ACTUAL trades (not bid/ask snapshots) and
    outcomes from the markets metadata
  - data/outcomes_hf.jsonl  — outcomes log for HF markets (separate from
    our Kalshi-API outcomes to keep provenance clear)

Trajectory is bucketed: one price point per (ticker, hour) using the
last trade in that hour. Keeps eval_pack readable for ~10k markets.

Usage:
    python scripts/ingest_hf_trades.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PREP_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = PREP_ROOT / "data" / "external"
PACK_HF = PREP_ROOT / "data" / "eval_pack_hf.jsonl"
OUTCOMES_HF = PREP_ROOT / "data" / "outcomes_hf.jsonl"


def _outcome_from_result(result: str | None) -> int | None:
    if not result:
        return None
    r = str(result).lower()
    if r in ("yes", "true", "y"):
        return 1
    if r in ("no", "false", "n"):
        return 0
    return None


# Use the authoritative Kalshi /series category lookup we already saved.
SERIES_CATS = json.loads((PREP_ROOT / "data" / "kalshi_series_categories.json").read_text())


def _category(event_ticker: str) -> str:
    if not event_ticker:
        return "Other"
    p = event_ticker.split("-")[0].upper()
    return SERIES_CATS.get(p, "Other")


def main() -> int:
    print("Loading markets metadata...")
    markets_df = pd.read_parquet(EXTERNAL / "kalshi_markets.parquet")
    print(f"  {len(markets_df):,} markets")

    # Index by ticker; keep only fields we need
    markets_by_ticker = {}
    for _, m in markets_df.iterrows():
        ticker = m["ticker"]
        outcome = _outcome_from_result(m.get("result"))
        if outcome is None or m.get("status") != "finalized":
            continue
        markets_by_ticker[ticker] = {
            "event_ticker": m.get("event_ticker") or "",
            "title": m.get("title") or "",
            "subtitle": m.get("subtitle") or m.get("yes_sub_title") or None,
            "rules": m.get("rules_primary") or None,
            "close_time": str(m.get("close_time") or ""),
            "outcome": outcome,
            "result": m.get("result"),
            "settled_at": str(m.get("expiration_time") or m.get("close_time") or ""),
            "open_time": str(m.get("open_time") or ""),
            "volume": int(m["volume"]) if pd.notna(m.get("volume")) else None,
            "liquidity_dollars": float(m["liquidity_dollars"]) if pd.notna(m.get("liquidity_dollars")) else None,
        }
    print(f"  {len(markets_by_ticker):,} resolved markets eligible")

    print("Loading trades...")
    trades = pd.concat([
        pd.read_parquet(EXTERNAL / "kalshi_trades_0.parquet"),
        pd.read_parquet(EXTERNAL / "kalshi_trades_1.parquet"),
    ], ignore_index=True)
    print(f"  {len(trades):,} trades")

    # Parse timestamps; bucket by hour
    trades["t"] = pd.to_datetime(trades["created_time"], format="ISO8601")
    trades["hour_bucket"] = trades["t"].dt.floor("h")
    # Filter to markets we have metadata for
    trades = trades[trades["ticker"].isin(markets_by_ticker)]
    print(f"  {len(trades):,} trades match resolved markets")

    # For each (ticker, hour), take last trade as the "snapshot"
    print("Building trajectories...")
    trades = trades.sort_values("t")
    snapshots = trades.groupby(["ticker", "hour_bucket"]).agg({
        "yes_price": "last",
        "no_price": "last",
        "t": "last",
    }).reset_index()

    # Group by ticker into trajectory lists
    by_ticker: dict = {}
    for ticker, group in snapshots.groupby("ticker"):
        traj = []
        for _, r in group.iterrows():
            traj.append({
                "t": str(r["t"]),
                "yes_ask": float(r["yes_price"]) / 100,
                "no_ask": float(r["no_price"]) / 100,
                "last_price": float(r["yes_price"]) / 100,
            })
        by_ticker[ticker] = traj
    print(f"  {len(by_ticker):,} markets with trajectory data")

    # Write eval pack + outcomes
    PACK_HF.parent.mkdir(parents=True, exist_ok=True)
    with PACK_HF.open("w") as fh_pack, OUTCOMES_HF.open("w") as fh_out:
        n_written = 0
        for ticker, m in markets_by_ticker.items():
            traj = by_ticker.get(ticker)
            if not traj:
                continue
            event = {
                "event_ticker": m["event_ticker"],
                "market_ticker": ticker,
                "title": m["title"],
                "subtitle": m["subtitle"],
                "description": None,
                "category": _category(m["event_ticker"]),
                "rules": m["rules"],
                "close_time": m["close_time"],
            }
            fh_pack.write(json.dumps({
                "event": event,
                "snapshots": traj,
                "outcome": m["outcome"],
                "result": m["result"],
                "settled_at": m["settled_at"],
                "_source": "hf:thomaswmitch/kalshi-prediction-markets",
                "volume": m["volume"],
                "liquidity_dollars": m["liquidity_dollars"],
            }) + "\n")
            fh_out.write(json.dumps({
                "market_ticker": ticker,
                "event_ticker": m["event_ticker"],
                "result": m["result"],
                "outcome": m["outcome"],
                "settled_at": m["settled_at"],
                "_source": "hf",
            }) + "\n")
            n_written += 1

    print()
    print(f"Wrote {n_written:,} markets to {PACK_HF.name}")
    print(f"Wrote {n_written:,} outcomes to {OUTCOMES_HF.name}")

    # Category breakdown
    from collections import Counter
    cats = Counter()
    traj_lens = []
    for line in PACK_HF.read_text().splitlines():
        r = json.loads(line)
        cats[r["event"]["category"]] += 1
        traj_lens.append(len(r["snapshots"]))
    print()
    print(f"Categories: {dict(cats.most_common())}")
    print(f"Trajectory length: min={min(traj_lens)}, "
          f"median={sorted(traj_lens)[len(traj_lens)//2]}, max={max(traj_lens)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
