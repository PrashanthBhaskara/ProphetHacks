"""Match prophet retrieve events.json -> Polymarket (map.csv / rejected.csv).

Expands each multi-outcome event to binary Kalshi markets via list_markets,
then runs the same Gamma matcher as poly_match.py.

Usage:
  prophet forecast retrieve --dataset hackathon-day -o events.json
  python scripts/poly_match_events.py ../../events.json
  python scripts/poly_match_events.py events.json --filter HOUSINGSTART --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PREP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREP_ROOT / "src"))

from prep.kalshi_expand import (  # noqa: E402
    build_event_search_query,
    expand_retrieve_event,
    load_retrieve_events,
)
from prep.polymarket import (  # noqa: E402
    DATA_DIR,
    MAP_CSV,
    _load_map,
    _load_negative,
    _months_mentioned,
    _poly_semantic_mismatch,
    active_search,
    resolve_mapping,
    write_match,
    write_reject,
)

EVENT_INDEX_PATH = DATA_DIR / "event_index.json"


def _load_index() -> dict:
    if not EVENT_INDEX_PATH.exists():
        return {}
    return json.loads(EVENT_INDEX_PATH.read_text())


def _save_index(index: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EVENT_INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True))


def _validate_map_rows(rows: list[dict]) -> list[str]:
    """Return kalshi_tickers in map.csv that fail semantic validation."""
    bad: list[str] = []
    for row in rows:
        ticker = row.get("kalshi_ticker", "")
        kq = row.get("kalshi_question", "")
        pq = row.get("poly_question", "")
        poly_m = {"endDateIso": row.get("poly_end_date", "")}
        if "end of 2026" in pq.lower() and "fomc" in kq.lower():
            bad.append(ticker)
            continue
        km, pm = _months_mentioned(kq), _months_mentioned(pq)
        if km and pm and not km.intersection(pm):
            bad.append(ticker)
            continue
        # extract rough short_label from poly question for threshold check
        sl = ""
        for part in row.get("kalshi_ticker", "").split("-"):
            if part.startswith("T") and len(part) > 1:
                sl = part[1:].replace("T", ".", 1) if part[1:1].isdigit() else ""
        if _poly_semantic_mismatch(sl or "x", kq, pq, poly_m):
            if sl:
                bad.append(ticker)
    return bad


def _index_put(
    index: dict,
    event_ticker: str,
    outcome: str,
    kalshi_ticker: str,
    poly_cid: str,
    poly_outcome: str,
) -> None:
    index.setdefault(event_ticker, {})[outcome] = {
        "kalshi_ticker": kalshi_ticker,
        "poly_condition_id": poly_cid,
        "poly_outcome": poly_outcome,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kalshi retrieve events -> polymarket map.csv",
    )
    parser.add_argument("events", type=Path, help="events.json from prophet forecast retrieve")
    parser.add_argument("--filter", type=str, default=None,
                        help="only events whose title contains this (case-insensitive)")
    parser.add_argument("--filter-ticker", type=str, default=None,
                        help="only events whose event_ticker contains this (case-insensitive)")
    parser.add_argument("--limit", type=int, default=None,
                        help="max outcome rows to poly-match (after expand)")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="seconds between per-outcome work (default 0.3)")
    parser.add_argument("--no-index", action="store_true",
                        help="do not write data/kalshi_polymarket/event_index.json")
    parser.add_argument("--retry-rejected", action="store_true",
                        help="retry tickers already in rejected.csv")
    args = parser.parse_args()

    events = load_retrieve_events(args.events)
    if args.filter:
        needle = args.filter.lower()
        events = [e for e in events if needle in (e.get("title") or "").lower()]
        print(f"filter {args.filter!r}: {len(events)} event(s)")
    if args.filter_ticker:
        needle = args.filter_ticker.lower()
        events = [
            e for e in events
            if needle in (e.get("event_ticker") or "").lower()
        ]
        print(f"filter-ticker {args.filter_ticker!r}: {len(events)} event(s)")

    cached = _load_map()
    rejected = _load_negative()
    index = _load_index() if not args.no_index else {}

    all_metas: list[dict] = []
    for ev in events:
        metas, unaligned = expand_retrieve_event(ev)
        et = ev.get("event_ticker") or ""
        if unaligned:
            preview = ", ".join(unaligned[:3])
            suffix = "..." if len(unaligned) > 3 else ""
            print(f"  {et}: {len(unaligned)} unaligned outcome(s) ({preview}{suffix})")
        if not metas:
            print(f"  {et}: no Kalshi binary markets matched")
            continue
        all_metas.extend(metas)

    if args.limit:
        all_metas = all_metas[: args.limit]

    print(f"{len(events)} event(s) -> {len(all_metas)} binary row(s) to match")

    n_match = n_reject = n_skip = 0
    event_cands: dict[str, list[dict]] = {}

    for i, meta in enumerate(all_metas, 1):
        ticker = meta["ticker"]
        if ticker in cached:
            n_skip += 1
            continue
        if ticker in rejected and not args.retry_rejected:
            n_skip += 1
            continue

        et = meta.get("event_ticker") or ""
        if et not in event_cands:
            q = meta.get("search_query") or build_event_search_query(
                {"title": meta.get("question"), "event_ticker": et},
            )
            event_cands[et] = active_search(q)
            time.sleep(args.sleep)

        chosen, outcome, query, n_cands = resolve_mapping(
            meta, candidates=event_cands[et],
        )
        if chosen is None:
            write_reject(meta, query, n_cands)
            rejected.add(ticker)
            n_reject += 1
            print(
                f"  [{i:3}/{len(all_metas)}] {ticker:36} [{meta['short_label'][:20]}] "
                f"no-match ({n_cands} cand) q={query!r}"
            )
        else:
            write_match(meta, chosen, outcome)
            cached[ticker] = (chosen.get("conditionId", ""), outcome)
            n_match += 1
            if not args.no_index:
                _index_put(
                    index, et, meta["outcome_label"],
                    ticker, chosen.get("conditionId", ""), outcome,
                )
            v24 = float(chosen.get("volume24hr") or 0)
            print(
                f"  [{i:3}/{len(all_metas)}] {ticker:36} [{meta['short_label'][:20]}] "
                f"MATCH -> {str(chosen.get('question'))[:55]} (vol24h={v24:.0f})"
            )
        time.sleep(args.sleep)

    if not args.no_index:
        _save_index(index)

    # Post-run validation on full map.csv
    import csv
    all_rows = list(csv.DictReader(MAP_CSV.open())) if MAP_CSV.exists() else []
    bad = _validate_map_rows(all_rows)
    if bad:
        print(f"\nWARNING: {len(bad)} map row(s) failed semantic validation:")
        for t in bad[:10]:
            print(f"  - {t}")
        if len(bad) > 10:
            print(f"  ... and {len(bad) - 10} more")
    else:
        print("\nValidation: all map.csv rows passed semantic checks")

    print(f"\ntotal: {n_match} matched, {n_reject} rejected, {n_skip} skipped")
    if not args.no_index:
        print(f"event index: {EVENT_INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
