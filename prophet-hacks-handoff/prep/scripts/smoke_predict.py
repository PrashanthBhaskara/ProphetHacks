"""Verify our predict() output matches the Prophet Hacks wire contract.

Runs independently of the `prophet` CLI (the locally-cloned ai-prophet is on
the old binary p_yes schema and would KeyError on our multi-outcome output).

Two modes:
  - Default: calls `agent_server.predict` in-process. Fast, no server needed.
  - --url:   POSTs to a running FastAPI server.

For each event, the script asserts:
  1. Response has a `probabilities` list.
  2. Each item has `market` (str) and `probability` (float).
  3. `market` labels equal `event.outcomes` exactly, in any order.
  4. Each probability is in [0, 1].

Usage:
    python scripts/smoke_predict.py                      # one synthetic event
    python scripts/smoke_predict.py --events events.json # JSON list of events
    python scripts/smoke_predict.py --events file.jsonl  # JSONL — one per line
    python scripts/smoke_predict.py --url http://localhost:8000/predict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))


SYNTHETIC_EVENT = {
    "event_ticker": "task-001",
    "market_ticker": "task-001",
    "title": "Who will win: Pittsburgh or Atlanta?",
    "subtitle": None,
    "description": "Predict the winner of the scheduled matchup.",
    "category": "Sports",
    "rules": "Resolves to the official winner after the game is final.",
    "close_time": "2026-03-21T23:59:59Z",
    "outcomes": ["Pittsburgh", "Atlanta"],
    "resolved_outcome": None,
}


def _load_events(path: Path) -> list[dict]:
    text = path.read_text().strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else [parsed]


def _predict_local(event: dict) -> dict:
    from agent_server import ArenaEvent, predict

    return predict(ArenaEvent(**event)).model_dump()


def _predict_http(event: dict, url: str) -> dict:
    import requests

    resp = requests.post(url, json=event, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _validate(event: dict, response: dict) -> list[str]:
    problems: list[str] = []
    probs = response.get("probabilities")
    if not isinstance(probs, list):
        problems.append("`probabilities` is missing or not a list")
        return problems

    seen_markets: list[str] = []
    for i, item in enumerate(probs):
        if not isinstance(item, dict):
            problems.append(f"probabilities[{i}] is not an object")
            continue
        market = item.get("market")
        prob = item.get("probability")
        if not isinstance(market, str):
            problems.append(f"probabilities[{i}].market is not a string ({market!r})")
        else:
            seen_markets.append(market)
        if not isinstance(prob, (int, float)):
            problems.append(f"probabilities[{i}].probability is not a number ({prob!r})")
        elif not (0.0 <= float(prob) <= 1.0):
            problems.append(f"probabilities[{i}].probability {prob} outside [0, 1]")

    expected = set(event.get("outcomes") or [])
    got = set(seen_markets)
    missing = expected - got
    extra = got - expected
    if missing:
        problems.append(f"missing outcomes in response: {sorted(missing)}")
    if extra:
        problems.append(f"unexpected outcomes in response: {sorted(extra)}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=None,
                        help="Path to events JSON or JSONL. Omit for one synthetic event.")
    parser.add_argument("--url", default=None,
                        help="POST to this URL instead of calling predict() in-process.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only test the first N events.")
    args = parser.parse_args()

    events = _load_events(args.events) if args.events else [SYNTHETIC_EVENT]
    if args.limit:
        events = events[: args.limit]
    if not events:
        print("No events to test.", file=sys.stderr)
        return 1

    fail_count = 0
    for event in events:
        ticker = event.get("market_ticker") or event.get("task_id") or event.get("event_ticker") or "?"
        try:
            response = _predict_http(event, args.url) if args.url else _predict_local(event)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {ticker}: predict raised {type(exc).__name__}: {exc}")
            fail_count += 1
            continue

        problems = _validate(event, response)
        if problems:
            fail_count += 1
            print(f"FAIL  {ticker}:")
            for p in problems:
                print(f"      - {p}")
            print(f"      response: {json.dumps(response)[:200]}")
        else:
            top = max(response["probabilities"], key=lambda x: x["probability"])
            print(f"PASS  {ticker}: top={top['market']!r} @ {top['probability']:.3f} "
                  f"({len(response['probabilities'])} outcomes)")

    print()
    print(f"{len(events) - fail_count}/{len(events)} passed")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
