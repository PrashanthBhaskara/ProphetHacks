"""Bulk backfill of historical resolved Kalshi markets.

Our regular snapshot+resolve pipeline only captures markets we caught while
open. This script grabs *every* settled binary Kalshi market from the past
N days that we don't already have an outcome for. The trade-off vs the
regular pipeline:

  +  Massively more data (potentially 100k+ markets across N days)
  -  No price trajectory (single closing snapshot per market)
  -  Captures only `last_price` / `yes_ask` / `no_ask` at scan time
     (which for settled markets is essentially the resolution value)

Despite the trajectory limitation, this is great for:
  - Larger sample size for strategy backtesting
  - Per-category breakdown with more statistical power
  - Validating that our forecasting/trading strategy works across a
    distribution wider than just the last 5 days

Usage:
    python scripts/backfill_kalshi.py                  # default last 60 days
    python scripts/backfill_kalshi.py --days 30
    python scripts/backfill_kalshi.py --days 180 --max-markets 50000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.kalshi import _get  # noqa: E402

PREP_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = PREP_ROOT / "data" / "snapshots"
BACKFILL_ROOT = PREP_ROOT / "data" / "backfill"
OUTCOMES_PATH = PREP_ROOT / "data" / "outcomes.jsonl"


def _is_simple_binary(m: dict) -> bool:
    if m.get("mve_collection_ticker"):
        return False
    has_dollars = m.get("yes_ask_dollars") is not None and m.get("no_ask_dollars") is not None
    has_cents = m.get("yes_ask") is not None and m.get("no_ask") is not None
    return has_dollars or has_cents


def _outcome_from_result(result: str | None) -> int | None:
    if not result:
        return None
    r = result.lower()
    if r in ("yes", "true", "y"):
        return 1
    if r in ("no", "false", "n"):
        return 0
    return None


def _existing_tickers() -> set[str]:
    """All tickers we already have an outcome for — skip these to stay idempotent."""
    if not OUTCOMES_PATH.exists():
        return set()
    out: set[str] = set()
    for line in OUTCOMES_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get("market_ticker"):
                out.add(row["market_ticker"])
        except Exception:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60,
                        help="how far back to scan settled markets")
    parser.add_argument("--max-markets", type=int, default=None,
                        help="cap the total markets fetched (safety valve)")
    args = parser.parse_args()

    now_ts = int(time.time())
    min_close_ts = now_ts - args.days * 24 * 3600
    max_close_ts = now_ts

    BACKFILL_ROOT.mkdir(parents=True, exist_ok=True)
    OUTCOMES_PATH.parent.mkdir(parents=True, exist_ok=True)

    skip = _existing_tickers()
    print(f"Backfill window: {datetime.fromtimestamp(min_close_ts, tz=timezone.utc).date()} "
          f"to {datetime.fromtimestamp(max_close_ts, tz=timezone.utc).date()} ({args.days} days)")
    print(f"Already-resolved tickers (will skip): {len(skip)}")

    # Stash a snapshot of the historical markets as a single dated dir so
    # the existing consolidate.py picks them up automatically.
    snap_dir = SNAPSHOT_ROOT / f"BACKFILL-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_time = datetime.now(timezone.utc).isoformat()

    by_event: dict[str, list[dict]] = {}
    new_outcomes = 0
    scanned = 0
    cursor: str | None = None

    with OUTCOMES_PATH.open("a") as outfh:
        while True:
            params: dict = {
                "status": "settled",
                "limit": 200,
                "min_close_ts": min_close_ts,
                "max_close_ts": max_close_ts,
            }
            if cursor:
                params["cursor"] = cursor
            data = _get("/markets", params)
            markets = data.get("markets", [])

            for m in markets:
                scanned += 1
                ticker = m.get("ticker")
                if not ticker or ticker in skip:
                    continue
                if not _is_simple_binary(m):
                    continue
                outcome = _outcome_from_result(m.get("result"))
                if outcome is None:
                    continue

                # Append outcome
                outfh.write(json.dumps({
                    "market_ticker": ticker,
                    "event_ticker": m.get("event_ticker"),
                    "result": m.get("result"),
                    "outcome": outcome,
                    "settled_at": m.get("expiration_time") or m.get("close_time"),
                }) + "\n")
                skip.add(ticker)
                new_outcomes += 1

                # Stash as a fake "snapshot" so consolidate sees the price.
                by_event.setdefault(m["event_ticker"], []).append(m)

                if args.max_markets and new_outcomes >= args.max_markets:
                    break

            outfh.flush()
            if scanned % 1000 == 0 or not markets:
                print(f"  scanned {scanned:,} settled markets, new outcomes: {new_outcomes:,}")

            cursor = data.get("cursor") or None
            if not cursor or (args.max_markets and new_outcomes >= args.max_markets):
                break
            time.sleep(0.1)

    # Write snapshot files (one per event) so consolidate.py picks them up.
    for event_ticker, event_markets in by_event.items():
        safe = event_ticker.replace("/", "_")
        (snap_dir / f"{safe}.json").write_text(json.dumps({
            "event_ticker": event_ticker,
            "snapshot_time": snap_time,
            "markets": event_markets,
        }, indent=2, default=str))

    (snap_dir / "_meta.json").write_text(json.dumps({
        "snapshot_time": snap_time,
        "kind": "backfill",
        "window_days": args.days,
        "event_count": len(by_event),
        "market_count": new_outcomes,
    }, indent=2))

    print()
    print(f"Done. Scanned {scanned:,} settled markets.")
    print(f"New outcomes added: {new_outcomes:,}  (across {len(by_event):,} events)")
    print(f"Backfill snapshot written to: {snap_dir}")
    print()
    print("Next: re-run scripts/consolidate.py to rebuild eval_pack.jsonl with the new data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
