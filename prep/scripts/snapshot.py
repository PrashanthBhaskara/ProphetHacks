"""Snapshot open Kalshi markets to disk for later backtesting.

Run this on a schedule (every 6–12 hours is plenty) for the days leading
into the hackathon. Each invocation creates a new timestamped directory
under prep/data/snapshots/.

After events resolve, run scripts/resolve.py to attach outcomes — then
the local snapshots become Sample data the harness can score against.

Usage:
    python scripts/snapshot.py                         # next 7 days, all categories
    python scripts/snapshot.py --window-days 3
    python scripts/snapshot.py --exclude-mve           # default ON
    python scripts/snapshot.py --keep-mve              # include multi-leg combos
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.kalshi import list_markets  # noqa: E402

PREP_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = PREP_ROOT / "data" / "snapshots"


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _is_simple_binary(m: dict) -> bool:
    if m.get("mve_collection_ticker"):
        return False
    has_dollars = m.get("yes_ask_dollars") is not None and m.get("no_ask_dollars") is not None
    has_cents = m.get("yes_ask") is not None and m.get("no_ask") is not None
    return has_dollars or has_cents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-days", type=int, default=5,
                        help="snapshot markets closing within N days from now (default 5: hackathon is May 16)")
    parser.add_argument("--keep-mve", action="store_true",
                        help="keep multi-leg combo markets (default: drop)")
    parser.add_argument("--out", default=None,
                        help="override output dir (default: prep/data/snapshots/<timestamp>)")
    args = parser.parse_args()

    now = int(time.time())
    max_close_ts = now + args.window_days * 24 * 3600

    out_dir = Path(args.out) if args.out else SNAPSHOT_ROOT / _utc_now_str()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching open markets closing within {args.window_days} days...")
    markets = list_markets(status="open", limit=200, max_close_ts=max_close_ts)
    print(f"  raw: {len(markets)} markets")

    if not args.keep_mve:
        markets = [m for m in markets if _is_simple_binary(m)]
        print(f"  after dropping MVE combos: {len(markets)} markets")

    # group by event for slightly easier downstream loading
    by_event: dict[str, list[dict]] = {}
    for m in markets:
        by_event.setdefault(m["event_ticker"], []).append(m)

    meta = {
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
        "window_days": args.window_days,
        "event_count": len(by_event),
        "market_count": len(markets),
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))

    for event_ticker, event_markets in by_event.items():
        safe = event_ticker.replace("/", "_")
        (out_dir / f"{safe}.json").write_text(json.dumps({
            "event_ticker": event_ticker,
            "snapshot_time": meta["snapshot_time"],
            "markets": event_markets,
        }, indent=2, default=str))

    print(f"Wrote {len(by_event)} events / {len(markets)} markets to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
