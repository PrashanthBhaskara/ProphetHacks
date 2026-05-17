"""Run the sibling-aware (multi-candidate) Grok predictor over a sample.

Groups markets by event_ticker, makes one LLM call per event (covering all
sibling markets), and writes per-ticker predictions out as jsonl.

This is the architectural fix for the multi-candidate failure mode we
diagnosed on Politics-500 (Brier 0.32 — the model said 80% YES for every
candidate of a 5-way election). The expected behavior with event-aware
prompting is for the model to normalize so probabilities across siblings
sum near 1, dropping multi-candidate Brier substantially.

Usage:
    # On the 2026 contamination-free sample
    python prep/scripts/predict_event_aware.py \\
        --sample prep_handoff/Kalshitopvolmarkets/samples/llm_calls_x10_seed42.jsonl \\
        --weekly-dir prep_handoff/Kalshitopvolmarkets/markets \\
        -o prep/data/predictions/grok_2026_event_aware.jsonl

    # On subset_1200 Politics (contamination caveat applies)
    python prep/scripts/predict_event_aware.py \\
        --source subset_1200 --category Politics --limit 500 \\
        -o prep/data/predictions/grok_subset1200_politics_event_aware.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.baselines.openrouter_event_aware import predict_event  # noqa: E402


def _load_2026_sample(path: Path, weekly_dir: Path | None) -> list[dict]:
    """Load a 2026 Kalshitopvolmarkets sample and join with per-event sibling lookup."""
    samples = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    samples = [s for s in samples if s.get("outcome_yes") in (0, 1)]

    # Build event_ticker → list of sibling markets (with prices) from the weekly market files
    if weekly_dir is None:
        return _samples_to_records(samples, sibling_map={})

    sibling_map: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(weekly_dir.glob("*_selected_markets.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            et = row.get("event_ticker")
            tk = row.get("ticker")
            if et and tk:
                sibling_map[et].append(row)
    return _samples_to_records(samples, sibling_map=dict(sibling_map))


def _samples_to_records(samples: list[dict], sibling_map: dict) -> list[dict]:
    out: list[dict] = []
    for s in samples:
        et = s["event"].get("event_ticker", "")
        out.append({
            "ticker": s["ticker"],
            "event_ticker": et,
            "event_title": s["event"]["title"],
            "event_category": s["event"]["category"],
            "close_time": s["event"].get("close_time"),
            "market_info": s["market_packet"].get("kalshi", {}),
            "market_meta": s["event"],
            "outcome": int(s["outcome_yes"]),
            "siblings_meta": sibling_map.get(et, []),
            "market_mid": s["quote"]["market_mid"],
        })
    return out


def _load_subset_records(source: str, category: str | None, limit: int | None) -> list[dict]:
    from prep.data import load_subset_1200, load_subset_100
    samples = load_subset_1200() if source == "subset_1200" else load_subset_100()
    if category:
        samples = [s for s in samples if s.event.get("category") == category]
    if limit:
        samples = samples[:limit]
    # Build per-event ticker list for sibling expansion
    by_event: dict[str, list] = defaultdict(list)
    for s in samples:
        by_event[s.event.get("event_ticker", "")].append(s)
    out: list[dict] = []
    for s in samples:
        et = s.event.get("event_ticker", "")
        out.append({
            "ticker": s.event.get("market_ticker", ""),
            "event_ticker": et,
            "event_title": s.event.get("title", ""),
            "event_category": s.event.get("category", ""),
            "close_time": s.event.get("close_time"),
            "market_info": s.market_info,
            "market_meta": s.event,
            "outcome": s.outcome,
            "siblings_meta": [
                {
                    "ticker": sib.event.get("market_ticker", ""),
                    "title": sib.event.get("title", ""),
                    "subtitle": sib.event.get("subtitle"),
                    "yes_sub_title": (sib.market_info or {}).get("yes_sub_title"),
                    "rules_primary": (sib.market_info or {}).get("rules_primary") or sib.event.get("rules"),
                    "_market_info": sib.market_info,
                }
                for sib in by_event[et]
            ],
            "market_mid": None,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sample", type=Path, help="Path to 2026 jsonl sample")
    grp.add_argument("--source", choices=("subset_1200", "hf"), help="Use built-in dataset")
    parser.add_argument("--weekly-dir", type=Path, default=Path("prep_handoff/Kalshitopvolmarkets/markets"))
    parser.add_argument("--category", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    if args.sample:
        records = _load_2026_sample(args.sample, args.weekly_dir)
    else:
        records = _load_subset_records(args.source, args.category, args.limit)

    print(f"Loaded {len(records)} markets", flush=True)

    # Group by event_ticker. Each group becomes one LLM call.
    by_event: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_event[r["event_ticker"]].append(r)
    print(f"  → {len(by_event)} unique events", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("w", buffering=1)

    def _do_event(item):
        et, group = item
        # Build siblings list for the LLM call: prefer the weekly_dir siblings if available
        rep = group[0]
        sibmeta = rep["siblings_meta"]
        if sibmeta:
            siblings = [
                (sm.get("ticker") or sm.get("market_ticker") or "",
                 sm,
                 sm.get("_market_info") or rep["market_info"])
                for sm in sibmeta
            ]
        else:
            siblings = [(r["ticker"], r["market_meta"], r["market_info"]) for r in group]
        try:
            preds = predict_event(
                rep["event_title"],
                rep["event_category"],
                siblings,
                close_time=str(rep["close_time"] or ""),
            )
        except Exception as e:
            sys.stderr.write(f"[event_aware] {et}: {e}\n")
            preds = {}
        return et, group, preds

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_do_event, item) for item in by_event.items()]
        for fut in as_completed(futs):
            et, group, preds = fut.result()
            for r in group:
                p = preds.get(r["ticker"], r.get("market_mid") or 0.5)
                f.write(json.dumps({
                    "market_ticker": r["ticker"],
                    "event_ticker": et,
                    "category": r["event_category"],
                    "p_yes": max(0.01, min(0.99, float(p))),
                    "outcome": r["outcome"],
                }) + "\n")
            done += 1
            if done % max(1, len(by_event) // 20) == 0 or done == len(by_event):
                print(f"  events: {done}/{len(by_event)}  ({time.time()-t0:.0f}s)", flush=True)
    f.close()

    print(f"\nDone: {done} events → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
