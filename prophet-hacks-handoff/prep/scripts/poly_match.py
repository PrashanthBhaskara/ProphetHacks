"""Match Kalshi -> Polymarket via Gamma public-search.

Pipeline (deterministic, no LLM):
  1. Build a topical query from the Kalshi market (keywords minus stopwords,
     minus short_label so we match the event, not just the outcome).
  2. Hit Gamma /public-search.
  3. Filter to markets where active=true, closed=false, endDate in the future.
  4. Among those, keep ones whose question contains the Kalshi short_label
     (case-insensitive substring). Tiebreak by volume24hr (most liquid wins).
  5. Write to data/kalshi_polymarket/map.csv or rejected.csv.

Usage:
  python scripts/poly_match.py SNAPSHOT.json [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PREP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREP_ROOT / "src"))

from prep.polymarket import (  # noqa: E402
    _load_map,
    _load_negative,
    meta_from_snapshot_row,
    resolve_mapping,
    write_match,
    write_reject,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="seconds between Gamma calls (default 0.3)")
    args = parser.parse_args()

    snap = json.loads(args.snapshot.read_text())
    kalshi = [m for m in snap["markets"] if m.get("source") == "kalshi"]
    if args.limit:
        kalshi = kalshi[: args.limit]

    cached = _load_map()
    rejected = _load_negative()

    n_match = n_reject = n_skip = 0
    for i, m in enumerate(kalshi, 1):
        meta = meta_from_snapshot_row(m)
        if meta["ticker"] in cached or meta["ticker"] in rejected:
            n_skip += 1
            continue

        chosen, outcome, query, n_cands = resolve_mapping(meta)
        if chosen is None:
            write_reject(meta, query, n_cands)
            n_reject += 1
            print(
                f"  [{i:3}/{len(kalshi)}] {meta['ticker']:32} no-match  "
                f"({n_cands} cand)  q={query!r}"
            )
        else:
            write_match(meta, chosen, outcome)
            n_match += 1
            v24 = float(chosen.get("volume24hr") or 0)
            print(
                f"  [{i:3}/{len(kalshi)}] {meta['ticker']:32} MATCH "
                f"[{meta['short_label']}] -> "
                f"{str(chosen.get('question'))[:70]} (vol24h={v24:.0f})"
            )
        time.sleep(args.sleep)

    print(f"\ntotal: {n_match} matched, {n_reject} rejected, {n_skip} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
