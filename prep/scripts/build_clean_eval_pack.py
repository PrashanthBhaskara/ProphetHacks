"""Produce a refined eval_pack with ONLY trustworthy data.

Filters eval_pack.jsonl down to markets that meet all of:
  - >= 2 snapshots (excludes single-snapshot backfill records where the
    price reflects post-settlement state, not pre-trade live bid/ask)
  - All snapshot prices in [0, 1]
  - Non-null close_time

Output: data/eval_pack_live_clean.jsonl

This is the trustworthy subset that's safe to share/publish without the
backfill contamination that distorts aggregate P&L numbers.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PREP_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PREP_ROOT / "data" / "eval_pack.jsonl"
OUT_PATH = PREP_ROOT / "data" / "eval_pack_live_clean.jsonl"


def _valid_snapshot(s: dict) -> bool:
    for k in ("yes_ask", "no_ask", "last_price"):
        v = s.get(k)
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            return False
        if not (0.0 <= v <= 1.0):
            return False
    return True


def main() -> int:
    if not IN_PATH.exists():
        print(f"Missing {IN_PATH}. Run consolidate.py first.")
        return 1

    kept = 0
    dropped_single = 0
    dropped_bad_price = 0
    dropped_missing = 0
    cats: Counter = Counter()

    with IN_PATH.open() as fin, OUT_PATH.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            snaps = row.get("snapshots") or []
            if len(snaps) < 2:
                dropped_single += 1
                continue
            if not all(_valid_snapshot(s) for s in snaps):
                dropped_bad_price += 1
                continue
            event = row.get("event") or {}
            if not event.get("close_time") or not event.get("market_ticker"):
                dropped_missing += 1
                continue

            fout.write(json.dumps(row) + "\n")
            kept += 1
            cats[event.get("category", "Other")] += 1

    print(f"Kept:    {kept:,}")
    print(f"Dropped (single-snapshot/backfill): {dropped_single:,}")
    print(f"Dropped (bad price):                {dropped_bad_price:,}")
    print(f"Dropped (missing fields):           {dropped_missing:,}")
    print()
    print("Categories kept:")
    for cat, n in cats.most_common():
        print(f"  {cat}: {n:,}")
    print()
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
