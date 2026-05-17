"""Sample pre-resolution market-time points from the NonBinaryMarkets dataset.

Outputs JSONL in the same shape as `prep_handoff/Kalshitopvolmarkets/samples/llm_calls_*.jsonl`
so eval_2026_sample.py and other downstream scripts work unchanged.

For each binary component market in NonBinaryMarkets:
  1. Load its OHLCV time series (1-minute candles)
  2. Pick a random minute that's at least min-time-to-close-minutes before close
  3. Use the yes_bid/yes_ask at that minute as the snapshot price
  4. Record outcome (result: yes/no) + sibling info (event_ticker)

Usage:
    python prep/scripts/sample_nonbinary_2026.py \\
        --data-dir prep_handoff/NonBinaryMarkets \\
        --samples-per-ticker 1 \\
        --out-jsonl prep_handoff/NonBinaryMarkets/samples/llm_calls_nbm_x1_seed42.jsonl
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc


def parse_iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_component_markets(data_dir: Path) -> list[dict]:
    """Load all resolved binary component markets across all weeks."""
    out = []
    for f in sorted((data_dir / "markets").glob("*_component_markets.jsonl")):
        for line in f.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("market_type") == "binary" and r.get("result") in ("yes", "no"):
                # Attach the week from filename for OHLCV lookup
                r["_week"] = f.name.split("_")[0]
                out.append(r)
    return out


def load_ohlcv(data_dir: Path, week: str, ticker: str) -> list[dict]:
    """Load OHLCV time series for one ticker. Returns list of dict rows."""
    f = data_dir / "ohlcv" / "period_1m" / f"week={week}" / f"{ticker}.csv.gz"
    if not f.exists():
        return []
    rows = []
    with gzip.open(f, "rt") as fp:
        for row in csv.DictReader(fp):
            rows.append(row)
    return rows


def sample_one(market: dict, ohlcv: list[dict], rng: random.Random, min_mtc_min: float) -> dict | None:
    """Pick one snapshot for this market, at random pre-close time with valid prices."""
    close_time = parse_iso(market.get("close_time"))
    if not close_time:
        return None

    # Filter rows: must have valid yes_ask + yes_bid and be at least min_mtc_min before close
    eligible = []
    for r in ohlcv:
        ts = parse_iso(r.get("end_period_time"))
        if not ts:
            continue
        mtc = (close_time - ts).total_seconds() / 60
        if mtc < min_mtc_min:
            continue
        ya = parse_float(r.get("yes_ask_close"))
        yb = parse_float(r.get("yes_bid_close"))
        if ya is None or yb is None:
            continue
        if not (0 < ya <= 1 and 0 <= yb <= 1):
            continue
        eligible.append((r, ts, mtc, ya, yb))

    if not eligible:
        return None

    row, ts, mtc, ya, yb = rng.choice(eligible)
    mid = (ya + yb) / 2

    return {
        "ticker": market["ticker"],
        "event": {
            "event_ticker": market.get("event_ticker", ""),
            "market_ticker": market["ticker"],
            "title": market.get("title", ""),
            "subtitle": market.get("subtitle"),
            "description": None,
            "category": _classify_category(market["ticker"]),
            "rules": market.get("rules_primary"),
            "close_time": str(market.get("close_time")),
        },
        "market_packet": {
            "kalshi": {
                "yes_ask": ya, "yes_bid": yb,
                "no_ask": max(0.0, min(1.0, 1.0 - yb)),
                "no_bid": max(0.0, min(1.0, 1.0 - ya)),
                "last_price": parse_float(row.get("price_close")),
                "snapshot_time": str(ts),
                "volume": parse_float(row.get("volume")) or 0.0,
                "open_interest": parse_float(row.get("open_interest")) or 0.0,
            },
        },
        "quote": {
            "yes_ask": ya, "yes_bid": yb,
            "no_ask": 1.0 - yb, "no_bid": 1.0 - ya,
            "market_mid": mid,
            "spread": max(0.0, ya - yb),
        },
        "series_ticker": market["ticker"].split("-")[0],
        "minutes_to_close": mtc,
        "outcome_yes": 1 if market["result"] == "yes" else 0,
        "week": market["_week"],
        "market_result": market["result"],
    }


SPORTS_SERIES_PREFIXES = (
    "KXNBA", "KXNCAAMB", "KXNCAAWB", "KXNFL", "KXNHL", "KXMLB", "KXATP", "KXWTA",
    "KXEPL", "KXUCL", "KXUFC", "KXIPL", "KXAFCONGAME", "KXLALIGA", "KXBUNDESLIGA",
    "KXMARMAD", "KXPGATOUR", "KXLPGA", "KXNCAAFGAME", "KXNCAAFSPREAD",
    "KXSERIEA", "KXLIGUE1", "KXT20", "KXTESTCRICKET", "KXNBASERIES",
)


def _classify_category(ticker: str) -> str:
    if any(ticker.startswith(p) for p in SPORTS_SERIES_PREFIXES):
        return "Sports"
    return "Other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("prep_handoff/NonBinaryMarkets"))
    parser.add_argument("--samples-per-ticker", type=int, default=1,
                        help="How many snapshot times to sample per market (default 1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-time-to-close-minutes", type=float, default=30.0)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--max-markets", type=int, default=None,
                        help="Cap the number of markets sampled (for quick tests).")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    markets = load_component_markets(args.data_dir)
    print(f"Loaded {len(markets)} binary component markets")
    if args.max_markets:
        rng.shuffle(markets)
        markets = markets[:args.max_markets]
        print(f"  capped to {len(markets)}")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out = args.out_jsonl.open("w", buffering=1)
    n_written = 0; n_skipped = 0
    for i, m in enumerate(markets):
        ohlcv = load_ohlcv(args.data_dir, m["_week"], m["ticker"])
        if not ohlcv:
            n_skipped += 1
            continue
        for _ in range(args.samples_per_ticker):
            rec = sample_one(m, ohlcv, rng, args.min_time_to_close_minutes)
            if rec is None:
                n_skipped += 1
                continue
            out.write(json.dumps(rec, default=str) + "\n")
            n_written += 1
        if (i + 1) % max(1, len(markets) // 20) == 0:
            print(f"  processed {i+1}/{len(markets)}, written={n_written}, skipped={n_skipped}")
    out.close()
    print(f"\nDone: {n_written} samples written, {n_skipped} skipped → {args.out_jsonl}")


if __name__ == "__main__":
    sys.exit(main())
