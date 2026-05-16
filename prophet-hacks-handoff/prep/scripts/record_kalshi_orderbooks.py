"""Record live Kalshi L2 order books going forward.

Historical full-depth Kalshi order books are not available from the official
API. The official API can provide the current book, so this recorder builds our
own L2 archive for any markets we care about.

Examples:
  python scripts/record_kalshi_orderbooks.py --tickers KXFEDDECISION-26JUN-H0 --seconds 10
  python scripts/record_kalshi_orderbooks.py --series-ticker KXFEDDECISION --seconds 30 --iterations 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep import kalshi  # noqa: E402

PREP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PREP_ROOT / "data" / "kalshi_training" / "live_l2" / "orderbooks.jsonl"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def discover_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return [t.strip() for t in args.tickers.split(",") if t.strip()]
    markets = kalshi.list_markets(
        status="open",
        series_ticker=args.series_ticker,
        mve_filter="exclude",
        max_items=args.max_markets,
        limit=1000,
    )
    return [m["ticker"] for m in markets if m.get("ticker")]


def write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll current Kalshi full-depth books into a JSONL archive.")
    parser.add_argument("--tickers", default="", help="Comma-separated market tickers.")
    parser.add_argument("--series-ticker", default=None, help="Discover open markets in this series if tickers omitted.")
    parser.add_argument("--max-markets", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=60.0, help="Polling interval.")
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--depth", type=int, default=0, help="0 means all levels.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tickers = discover_tickers(args)
    if not tickers:
        print("No tickers found.")
        return 1

    print(f"Recording {len(tickers)} books to {args.out}")
    n = 0
    while True:
        pulled_at = iso_now()
        for ticker in tickers:
            try:
                payload = kalshi.get_orderbook(ticker, depth=args.depth)
                row = {
                    "_source": "kalshi_official_live_l2_recorder",
                    "_pulled_at": pulled_at,
                    "ticker": ticker,
                    "payload": payload,
                }
            except requests.HTTPError as exc:
                row = {
                    "_source": "kalshi_official_live_l2_recorder_error",
                    "_pulled_at": pulled_at,
                    "ticker": ticker,
                    "payload": {"error": str(exc)},
                }
            write_row(args.out, row)
        n += 1
        if args.iterations and n >= args.iterations:
            break
        time.sleep(args.seconds)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
