"""Targeted backfill for under-represented Kalshi categories.

The general backfill_kalshi.py scans by close-time across all settled
markets — fine for volume but skews toward whatever is highest-frequency
(crypto strikes, sports). Categories like Politics, Elections, World,
and Entertainment are heavily under-represented in our eval pack.

This script walks settled events in specific categories, then pulls
markets per event. Much more API calls per record, but targets the
data gap directly.

Usage:
    python scripts/backfill_categories.py
    python scripts/backfill_categories.py --categories Politics,Elections
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
OUTCOMES_PATH = PREP_ROOT / "data" / "outcomes.jsonl"

DEFAULT_CATEGORIES = [
    "Politics", "Elections", "Entertainment", "World",
    "Science and Technology", "Health", "Companies", "Mentions",
    "Commodities", "Social", "Transportation",
]


def _is_simple_binary(m: dict) -> bool:
    if m.get("mve_collection_ticker"):
        return False
    return (m.get("yes_ask_dollars") is not None and m.get("no_ask_dollars") is not None) or \
           (m.get("yes_ask") is not None and m.get("no_ask") is not None)


def _outcome_from_result(result):
    if not result:
        return None
    r = result.lower()
    if r in ("yes", "true", "y"):
        return 1
    if r in ("no", "false", "n"):
        return 0
    return None


def _existing_tickers() -> set[str]:
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


def _iter_events_by_category(category: str, status: str = "settled", pause: float = 0.15):
    cursor: str | None = None
    while True:
        params: dict = {"category": category, "status": status, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = _get("/events", params)
        for e in data.get("events", []):
            yield e
        cursor = data.get("cursor") or None
        if not cursor:
            return
        time.sleep(pause)


def _markets_for_event(event_ticker: str) -> list[dict]:
    data = _get("/markets", {"event_ticker": event_ticker, "limit": 200})
    return data.get("markets", []) or []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                        help="comma-separated category list")
    parser.add_argument("--max-events-per-category", type=int, default=2000)
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    skip = _existing_tickers()
    print(f"Categories to backfill: {categories}")
    print(f"Already-resolved tickers (will skip): {len(skip):,}")

    snap_dir = SNAPSHOT_ROOT / f"CATBACKFILL-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_time = datetime.now(timezone.utc).isoformat()

    grand_added = 0
    with OUTCOMES_PATH.open("a") as outfh:
        for cat in categories:
            cat_added = 0
            event_count = 0
            print(f"\n=== {cat} ===")
            for event in _iter_events_by_category(cat):
                event_count += 1
                if event_count > args.max_events_per_category:
                    print(f"  hit cap of {args.max_events_per_category} events for {cat}")
                    break
                event_ticker = event.get("event_ticker")
                if not event_ticker:
                    continue
                try:
                    markets = _markets_for_event(event_ticker)
                except Exception as e:
                    print(f"  [warn] {event_ticker}: {e}")
                    continue

                event_market_records = []
                for m in markets:
                    ticker = m.get("ticker")
                    if not ticker or ticker in skip:
                        continue
                    if not _is_simple_binary(m):
                        continue
                    outcome = _outcome_from_result(m.get("result"))
                    if outcome is None:
                        continue
                    outfh.write(json.dumps({
                        "market_ticker": ticker,
                        "event_ticker": event_ticker,
                        "result": m.get("result"),
                        "outcome": outcome,
                        "settled_at": m.get("expiration_time") or m.get("close_time"),
                    }) + "\n")
                    skip.add(ticker)
                    cat_added += 1
                    grand_added += 1
                    event_market_records.append(m)

                if event_market_records:
                    safe = event_ticker.replace("/", "_")
                    (snap_dir / f"{safe}.json").write_text(json.dumps({
                        "event_ticker": event_ticker,
                        "snapshot_time": snap_time,
                        "markets": event_market_records,
                    }, indent=2, default=str))

                if event_count % 25 == 0:
                    outfh.flush()
                    print(f"  events={event_count:<5} new outcomes={cat_added}")
                time.sleep(0.08)
            print(f"  {cat} complete: {cat_added:,} new outcomes from {event_count} events")

    (snap_dir / "_meta.json").write_text(json.dumps({
        "snapshot_time": snap_time,
        "kind": "category-backfill",
        "categories": categories,
        "market_count": grand_added,
    }, indent=2))

    print(f"\nDone. Total new outcomes: {grand_added:,}")
    print(f"Snapshot dir: {snap_dir}")
    print("Next: re-run scripts/consolidate.py to update eval_pack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
