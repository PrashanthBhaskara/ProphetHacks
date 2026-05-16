"""Attach outcomes to previously-snapshotted markets.

Bulk-fetches settled markets from Kalshi via
`list_markets(status='settled', min_close_ts=...)` and writes one JSONL row
per market we've snapshotted that has now resolved. Massively faster than
querying each market individually — one paginated bulk fetch instead of
N per-market round-trips.

State file (`data/resolve_state.json`) tracks the high-watermark close_ts
so subsequent runs only re-scan the new window (with a safety overlap).

Idempotent — already-resolved tickers are skipped.

Usage:
    python scripts/resolve.py                 # default lookback
    python scripts/resolve.py --lookback-days 7
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
STATE_PATH = PREP_ROOT / "data" / "resolve_state.json"
OVERLAP_SECONDS = 6 * 3600


def _all_snapshotted_tickers() -> set[str]:
    tickers: set[str] = set()
    if not SNAPSHOT_ROOT.exists():
        return tickers
    for snap_dir in SNAPSHOT_ROOT.iterdir():
        if not snap_dir.is_dir():
            continue
        for fp in snap_dir.glob("*.json"):
            if fp.name == "_meta.json":
                continue
            try:
                data = json.loads(fp.read_text())
                for m in data.get("markets", []):
                    if m.get("ticker"):
                        tickers.add(m["ticker"])
            except Exception:
                continue
    return tickers


def _already_resolved() -> set[str]:
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


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _outcome_from_result(result: str | None) -> int | None:
    if not result:
        return None
    r = result.lower()
    if r in ("yes", "true", "y"):
        return 1
    if r in ("no", "false", "n"):
        return 0
    return None


def _bulk_iter_settled(min_close_ts: int, max_close_ts: int, pause: float = 0.1):
    """Yield finalized markets page by page."""
    cursor: str | None = None
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
        for m in data.get("markets", []):
            yield m
        cursor = data.get("cursor") or None
        if not cursor:
            return
        time.sleep(pause)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=None,
                        help="how far back to scan settled markets (default: since last run)")
    args = parser.parse_args()

    OUTCOMES_PATH.parent.mkdir(parents=True, exist_ok=True)

    state = _load_state()
    now_ts = int(time.time())

    if args.lookback_days is not None:
        min_close_ts = now_ts - args.lookback_days * 24 * 3600
    elif state.get("last_max_close_ts"):
        min_close_ts = int(state["last_max_close_ts"]) - OVERLAP_SECONDS
    else:
        # First run: scan the past 7 days.
        min_close_ts = now_ts - 7 * 24 * 3600

    max_close_ts = now_ts

    snapshotted = _all_snapshotted_tickers()
    already = _already_resolved()
    print(f"Snapshotted: {len(snapshotted)} | already resolved: {len(already)}")
    print(f"Scanning settled markets from {datetime.fromtimestamp(min_close_ts, tz=timezone.utc).isoformat()} "
          f"to {datetime.fromtimestamp(max_close_ts, tz=timezone.utc).isoformat()}")

    # Save the watermark up front. If the scan crashes mid-page, next run
    # only needs to re-cover the OVERLAP_SECONDS window — no progress lost.
    state["last_max_close_ts"] = max_close_ts
    state["last_run_started_at"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)

    newly_resolved = 0
    pages = 0
    seen = 0
    with OUTCOMES_PATH.open("a") as fh:
        for m in _bulk_iter_settled(min_close_ts, max_close_ts):
            seen += 1
            if seen % 200 == 0:
                pages += 1
                print(f"  scanned {seen} settled markets, matched {newly_resolved} so far")
            ticker = m.get("ticker")
            if not ticker or ticker not in snapshotted or ticker in already:
                continue
            outcome = _outcome_from_result(m.get("result"))
            if outcome is None:
                continue
            fh.write(json.dumps({
                "market_ticker": ticker,
                "event_ticker": m.get("event_ticker"),
                "result": m.get("result"),
                "outcome": outcome,
                "settled_at": m.get("expiration_time") or m.get("close_time"),
            }) + "\n")
            fh.flush()
            newly_resolved += 1
            already.add(ticker)

    state["last_run_completed_at"] = datetime.now(timezone.utc).isoformat()
    state["last_newly_resolved"] = newly_resolved
    state["last_total_scanned"] = seen
    _save_state(state)

    print(f"Done. Scanned {seen} settled markets. Newly resolved (matched our snapshots): {newly_resolved}.")
    print(f"Total outcomes on disk: {len(already)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
