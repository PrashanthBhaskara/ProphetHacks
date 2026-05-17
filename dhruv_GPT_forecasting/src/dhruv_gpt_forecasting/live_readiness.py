"""Timed live prediction smoke checks for the five-minute forecast budget."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .arena_agent import forecast_arena_event


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--deadline-seconds", type=float, default=300.0)
    parser.add_argument("--with-gpt", action="store_true")
    parser.add_argument("--live-data", action="store_true")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    events = _load_events(args.events_json)[: args.limit]
    rows = []
    started = time.monotonic()
    for event in events:
        event_started = time.monotonic()
        forecast = forecast_arena_event(
            event,
            use_gpt=args.with_gpt,
            use_live_data=args.live_data,
            deadline_seconds=args.deadline_seconds,
        )
        elapsed = time.monotonic() - event_started
        rows.append({
            "market_ticker": event.get("market_ticker"),
            "event_ticker": event.get("event_ticker"),
            "elapsed_seconds": elapsed,
            "within_deadline": elapsed <= args.deadline_seconds,
            "source": forecast.source,
            "probabilities": forecast.probabilities,
            "live_evidence_count": forecast.audit.get("live_evidence_count", 0),
            "errors": forecast.audit.get("errors", []),
            "audit": {
                "elapsed_seconds": forecast.audit.get("elapsed_seconds"),
                "response_deadline_seconds": forecast.audit.get("response_deadline_seconds"),
                "within_response_deadline": forecast.audit.get("within_response_deadline"),
            },
        })
    elapsed_total = time.monotonic() - started
    result = {
        "events_json": str(args.events_json),
        "n_events": len(events),
        "deadline_seconds": args.deadline_seconds,
        "with_gpt": args.with_gpt,
        "live_data": args.live_data,
        "elapsed_total_seconds": elapsed_total,
        "max_event_seconds": max((row["elapsed_seconds"] for row in rows), default=0.0),
        "all_within_deadline": all(row["within_deadline"] for row in rows),
        "rows": rows,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _load_events(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        events = data.get("events") or data.get("data") or []
        if isinstance(events, list):
            return [item for item in events if isinstance(item, dict)]
        return [data]
    return []


if __name__ == "__main__":
    raise SystemExit(main())
