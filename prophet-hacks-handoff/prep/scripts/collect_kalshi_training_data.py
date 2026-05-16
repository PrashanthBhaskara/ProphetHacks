"""Collect Kalshi market data for training and backtests.

This script keeps provider payloads raw and source-tagged. That is deliberate:
for model training we can normalize after the pull, but we cannot recover fields
that were dropped too early.

Sources supported:
  - Kalshi official API: market metadata, trades, L1 candles, current L2 books.
  - Oddpool API: historical Kalshi L1 top-of-book, historical trades, and
    historical L2 orderbook snapshots. Requires ODDPOOL_API_KEY.
  - OddsPipe API: normalized Kalshi candles when a key is present. Requires
    ODDSPIPE_API_KEY and relies on its market search endpoint for ticker mapping.
  - pmxt: optional SDK fallback for current books/OHLCV when installed.

Examples:
  # Pull a small smoke sample from official Kalshi only.
  python scripts/collect_kalshi_training_data.py --series-ticker KXFED --max-markets 5

  # Pull all official Kalshi data for a series, plus Oddpool historical L2.
  python scripts/collect_kalshi_training_data.py \
    --series-ticker KXFEDDECISION --max-markets 0 --oddpool --l2

  # Pull specific tickers across every configured source.
  python scripts/collect_kalshi_training_data.py \
    --tickers KXFEDDECISION-26JUN-H0,KXFEDDECISION-26JUN-T25 \
    --oddpool --oddspipe --pmxt --l2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep import kalshi  # noqa: E402

PREP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PREP_ROOT / "data" / "kalshi_training"
ODDPOOL_BASE = "https://api.oddpool.com"
ODDSPIPE_BASE = "https://oddspipe.com"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: str | None, *, default: datetime) -> datetime:
    if not value:
        return default
    raw = value.strip()
    if raw.isdigit():
        n = int(raw)
        if n > 10_000_000_000:
            n //= 1000
        return datetime.fromtimestamp(n, tz=timezone.utc)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def unix_s(dt: datetime) -> int:
    return int(dt.timestamp())


def unix_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(path)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def clean_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def series_from_market(market: dict[str, Any]) -> str | None:
    if market.get("series_ticker"):
        return str(market["series_ticker"])
    ticker = market.get("ticker") or market.get("market_ticker")
    if ticker and "-" in ticker:
        return str(ticker).split("-", 1)[0]
    event_ticker = market.get("event_ticker")
    if event_ticker and "-" in event_ticker:
        return str(event_ticker).split("-", 1)[0]
    return event_ticker


def source_row(source: str, payload: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"_source": source, "_pulled_at": iso(utc_now()), **extra, "payload": payload}


def request_json(
    base_url: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> Any:
    backoff = 1.0
    last_exc: Exception | None = None
    url = base_url.rstrip("/") + path
    for _ in range(8):
        try:
            resp = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            last_exc = exc
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    if last_exc:
        raise last_exc
    resp.raise_for_status()
    return resp.json()


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def window_chunks(start: datetime, end: datetime, *, interval_minutes: int) -> Iterable[tuple[datetime, datetime]]:
    """Split candle requests into provider-friendly windows."""
    if interval_minutes == 1:
        step_days = 7
    elif interval_minutes == 60:
        step_days = 90
    else:
        step_days = 365
    step = step_days * 24 * 60 * 60
    cur = unix_s(start)
    end_s = unix_s(end)
    while cur < end_s:
        nxt = min(cur + step, end_s)
        yield datetime.fromtimestamp(cur, timezone.utc), datetime.fromtimestamp(nxt, timezone.utc)
        cur = nxt


def load_markets(args: argparse.Namespace) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    seen: set[str] = set()

    tickers = [t.strip() for t in (args.tickers or "").split(",") if t.strip()]
    if tickers:
        try:
            historical = kalshi.list_historical_markets(tickers=tickers, exclude_mve=args.exclude_mve)
        except requests.HTTPError as exc:
            print(f"warning: historical ticker discovery failed: {exc}", file=sys.stderr)
            historical = []
        live_by_ticker = {}
        for ticker in tickers:
            live = kalshi.get_market(ticker)
            if live:
                live_by_ticker[ticker] = live
        for m in historical + list(live_by_ticker.values()):
            ticker = m.get("ticker")
            if ticker and ticker not in seen:
                markets.append(m)
                seen.add(ticker)
        for ticker in tickers:
            if ticker not in seen:
                markets.append({"ticker": ticker, "_missing_metadata": True})
                seen.add(ticker)
    else:
        if args.historical_markets:
            remaining = None if not args.max_markets else max(args.max_markets - len(markets), 0)
            try:
                historical_markets = kalshi.list_historical_markets(
                    series_ticker=args.series_ticker,
                    exclude_mve=args.exclude_mve,
                    pause=args.pause,
                    max_items=remaining,
                )
            except requests.HTTPError as exc:
                print(f"warning: historical market discovery failed: {exc}", file=sys.stderr)
                historical_markets = []
            for m in historical_markets:
                ticker = m.get("ticker")
                if ticker and ticker not in seen:
                    markets.append(m)
                    seen.add(ticker)
                if args.max_markets and len(markets) >= args.max_markets:
                    break

        if args.live_markets and (not args.max_markets or len(markets) < args.max_markets):
            for status in args.live_status:
                remaining = None if not args.max_markets else max(args.max_markets - len(markets), 0)
                if remaining == 0:
                    break
                for m in kalshi.list_markets(
                    status=None if status == "all" else status,
                    series_ticker=args.series_ticker,
                    mve_filter="exclude" if args.exclude_mve else None,
                    limit=1000,
                    max_items=remaining,
                ):
                    ticker = m.get("ticker")
                    if ticker and ticker not in seen:
                        markets.append(m)
                        seen.add(ticker)
                    if args.max_markets and len(markets) >= args.max_markets:
                        break
                if args.max_markets and len(markets) >= args.max_markets:
                    break

    if args.min_volume is not None:
        kept = []
        for m in markets:
            vol = m.get("volume_fp") or m.get("volume") or 0
            try:
                if float(vol) >= args.min_volume:
                    kept.append(m)
            except (TypeError, ValueError):
                pass
        markets = kept

    if args.max_markets:
        markets = markets[:args.max_markets]
    return markets


def collect_kalshi_official(args: argparse.Namespace, markets: list[dict[str, Any]], start: datetime, end: datetime) -> None:
    cutoff = kalshi.historical_cutoff()
    write_json(args.out / "kalshi" / "cutoff.json", cutoff)

    if args.trades:
        for m in markets:
            ticker = m.get("ticker")
            if not ticker:
                continue
            rows: list[dict[str, Any]] = []
            for historical in (True, False):
                try:
                    trades = kalshi.list_trades(
                        ticker=ticker,
                        min_ts=unix_s(start),
                        max_ts=unix_s(end),
                        historical=historical,
                        pause=args.pause,
                    )
                except requests.HTTPError as exc:
                    rows.append(source_row(
                        "kalshi_official_trades_error",
                        {"error": str(exc), "historical": historical},
                        ticker=ticker,
                    ))
                    continue
                rows.extend(
                    source_row(
                        "kalshi_official_historical_trades" if historical else "kalshi_official_live_trades",
                        trade,
                        ticker=ticker,
                    )
                    for trade in trades
                )
            append_jsonl(args.out / "kalshi" / "trades" / f"{clean_name(ticker)}.jsonl", rows)
            time.sleep(args.pause)

    if args.candles:
        for m in markets:
            ticker = m.get("ticker")
            if not ticker:
                continue
            series = series_from_market(m)
            for interval in args.kalshi_intervals:
                out_path = args.out / "kalshi" / "candles" / f"{interval}m" / f"{clean_name(ticker)}.jsonl"
                rows: list[dict[str, Any]] = []
                for a, b in window_chunks(start, end, interval_minutes=interval):
                    fetched = False
                    # Old settled markets are in the historical tier; recent/open markets are live.
                    for historical in (True, False):
                        try:
                            data = kalshi.get_market_candlesticks(
                                ticker=ticker,
                                series_ticker=series,
                                start_ts=unix_s(a),
                                end_ts=unix_s(b),
                                period_interval=interval,
                                historical=historical,
                            )
                        except (requests.HTTPError, ValueError):
                            continue
                        rows.extend(
                            source_row(
                                "kalshi_official_historical_candles" if historical else "kalshi_official_live_candles",
                                candle,
                                ticker=ticker,
                                interval_minutes=interval,
                            )
                            for candle in data.get("candlesticks", [])
                        )
                        fetched = True
                        break
                    if not fetched:
                        rows.append(source_row(
                            "kalshi_official_candles_error",
                            {"start": iso(a), "end": iso(b), "interval_minutes": interval},
                            ticker=ticker,
                        ))
                    time.sleep(args.pause)
                append_jsonl(out_path, rows)

    if args.l2:
        for m in markets:
            ticker = m.get("ticker")
            if not ticker:
                continue
            try:
                book = kalshi.get_orderbook(ticker, depth=args.current_depth)
            except requests.HTTPError as exc:
                book = {"error": str(exc)}
            write_json(
                args.out / "kalshi" / "current_orderbooks" / f"{clean_name(ticker)}.json",
                source_row("kalshi_official_current_l2", book, ticker=ticker),
            )
            time.sleep(args.pause)


def oddpool_headers() -> dict[str, str]:
    key = os.getenv("ODDPOOL_API_KEY")
    if not key:
        raise RuntimeError("ODDPOOL_API_KEY is required for --oddpool")
    return {"X-API-Key": key}


def collect_oddpool_endpoint(
    *,
    path: str,
    rows_key: str,
    base_params: dict[str, Any],
    source: str,
    out_path: Path,
    pause: float,
) -> int:
    headers = oddpool_headers()
    pagination_key: str | None = None
    written = 0
    while True:
        params = dict(base_params)
        if pagination_key:
            params["pagination_key"] = pagination_key
        data = request_json(ODDPOOL_BASE, path, params=params, headers=headers)
        rows = data.get(rows_key, [])
        written += append_jsonl(out_path, (source_row(source, row, market_id=base_params.get("market_id")) for row in rows))
        page = data.get("pagination") or {}
        if not page.get("has_more"):
            break
        pagination_key = page.get("pagination_key")
        if not pagination_key:
            break
        time.sleep(pause)
    return written


def collect_oddpool(args: argparse.Namespace, markets: list[dict[str, Any]], start: datetime, end: datetime) -> None:
    start_ms = unix_ms(start)
    end_ms = unix_ms(end)

    if args.oddpool_ohlcv:
        headers = oddpool_headers()
        ids = [m["ticker"] for m in markets if m.get("ticker")]
        for interval in args.oddpool_intervals:
            for batch in chunked(ids, 50):
                data = request_json(
                    ODDPOOL_BASE,
                    "/markets/ohlcv",
                    params={"market_ids": ",".join(batch), "from": iso(start), "to": iso(end), "interval": interval},
                    headers=headers,
                )
                out_path = args.out / "oddpool" / "ohlcv" / interval / f"batch_{clean_name(batch[0])}.jsonl"
                append_jsonl(out_path, (source_row("oddpool_ohlcv", row, interval=interval) for row in data))
                time.sleep(args.pause)

    for m in markets:
        ticker = m.get("ticker")
        if not ticker:
            continue
        common = {"market_id": ticker, "start_time": start_ms, "end_time": end_ms, "limit": 200}

        if args.l1:
            for granularity in args.oddpool_granularity:
                collect_oddpool_endpoint(
                    path="/historical/kalshi/top-of-book",
                    rows_key="snapshots",
                    base_params={**common, "granularity": granularity},
                    source="oddpool_kalshi_top_of_book",
                    out_path=args.out / "oddpool" / "top_of_book" / granularity / f"{clean_name(ticker)}.jsonl",
                    pause=args.pause,
                )

        if args.l2:
            for granularity in args.oddpool_granularity:
                collect_oddpool_endpoint(
                    path="/historical/kalshi/orderbook",
                    rows_key="snapshots",
                    base_params={**common, "granularity": granularity},
                    source="oddpool_kalshi_l2_orderbook",
                    out_path=args.out / "oddpool" / "orderbook" / granularity / f"{clean_name(ticker)}.jsonl",
                    pause=args.pause,
                )

        if args.trades:
            collect_oddpool_endpoint(
                path="/historical/kalshi/trades",
                rows_key="trades",
                base_params=common,
                source="oddpool_kalshi_trades",
                out_path=args.out / "oddpool" / "trades" / f"{clean_name(ticker)}.jsonl",
                pause=args.pause,
            )


def oddspipe_headers() -> dict[str, str]:
    key = os.getenv("ODDSPIPE_API_KEY")
    if not key:
        raise RuntimeError("ODDSPIPE_API_KEY is required for --oddspipe")
    return {"X-API-Key": key}


def resolve_oddspipe_market_id(ticker: str, out_dir: Path) -> str | None:
    data = request_json(
        ODDSPIPE_BASE,
        "/v1/markets/search",
        params={"q": ticker, "platform": "kalshi"},
        headers=oddspipe_headers(),
    )
    write_json(out_dir / "oddspipe" / "search" / f"{clean_name(ticker)}.json", data)
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return None

    for item in items:
        values = {
            str(item.get("ticker", "")),
            str(item.get("market_ticker", "")),
            str(item.get("external_id", "")),
            str(item.get("slug", "")),
        }
        if ticker in values:
            return str(item.get("id") or item.get("market_id") or ticker)
    if items:
        first = items[0]
        return str(first.get("id") or first.get("market_id") or "")
    return None


def collect_oddspipe(args: argparse.Namespace, markets: list[dict[str, Any]]) -> None:
    for m in markets:
        ticker = m.get("ticker")
        if not ticker:
            continue
        market_id = resolve_oddspipe_market_id(ticker, args.out)
        if not market_id:
            continue
        for interval in args.oddspipe_intervals:
            try:
                data = request_json(
                    ODDSPIPE_BASE,
                    f"/v1/markets/{market_id}/candlesticks",
                    params={"interval": interval},
                    headers=oddspipe_headers(),
                )
            except requests.HTTPError as exc:
                data = {"error": str(exc), "market_id": market_id, "ticker": ticker, "interval": interval}
            write_json(
                args.out / "oddspipe" / "candlesticks" / interval / f"{clean_name(ticker)}.json",
                source_row("oddspipe_candlesticks", data if isinstance(data, dict) else {"items": data}, ticker=ticker),
            )
            time.sleep(args.pause)


def dataclass_to_dict(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, list):
        return [dataclass_to_dict(x) for x in value]
    if isinstance(value, dict):
        return {k: dataclass_to_dict(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {k: dataclass_to_dict(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def collect_pmxt(args: argparse.Namespace, markets: list[dict[str, Any]], start: datetime, end: datetime) -> None:
    try:
        import pmxt  # type: ignore
    except ImportError:
        write_json(args.out / "pmxt" / "error.json", {
            "error": "pmxt is not installed. Install with: pip install pmxt && npm install -g pmxtjs",
        })
        return

    exchange = pmxt.Kalshi()
    for m in markets:
        ticker = m.get("ticker")
        if not ticker:
            continue
        try:
            found = exchange.fetch_markets(query=ticker, limit=5)
        except Exception as exc:  # pmxt wraps provider errors.
            write_json(args.out / "pmxt" / "markets" / f"{clean_name(ticker)}.json", {"error": str(exc)})
            continue
        write_json(args.out / "pmxt" / "markets" / f"{clean_name(ticker)}.json", dataclass_to_dict(found))

        exact = None
        for candidate in found:
            if getattr(candidate, "market_id", None) == ticker or ticker in str(getattr(candidate, "url", "")):
                exact = candidate
                break
        if exact is None and found:
            exact = found[0]
        outcomes = list(getattr(exact, "outcomes", []) or []) if exact else []
        if not outcomes:
            continue

        outcome_id = getattr(outcomes[0], "outcome_id", None)
        if not outcome_id:
            continue

        if args.candles:
            for resolution in args.pmxt_resolutions:
                try:
                    candles = exchange.fetch_ohlcv(
                        outcome_id,
                        resolution=resolution,
                        start=start.date().isoformat(),
                        end=end.date().isoformat(),
                        limit=1000,
                    )
                except Exception as exc:
                    candles = {"error": str(exc)}
                write_json(
                    args.out / "pmxt" / "ohlcv" / resolution / f"{clean_name(ticker)}.json",
                    source_row("pmxt_kalshi_ohlcv", {"items": dataclass_to_dict(candles)}, ticker=ticker),
                )

        if args.l2:
            try:
                book = exchange.fetch_order_book(outcome_id)
            except Exception as exc:
                book = {"error": str(exc)}
            write_json(
                args.out / "pmxt" / "current_orderbooks" / f"{clean_name(ticker)}.json",
                source_row("pmxt_kalshi_current_orderbook", {"book": dataclass_to_dict(book)}, ticker=ticker),
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Kalshi training data across official and third-party APIs.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--since", default="2023-01-01T00:00:00Z")
    parser.add_argument("--until", default=None, help="Default: now")
    parser.add_argument("--series-ticker", default=None, help="Optional Kalshi series filter, e.g. KXFEDDECISION")
    parser.add_argument("--tickers", default="", help="Comma-separated market tickers. Overrides discovery.")
    parser.add_argument("--max-markets", type=int, default=25,
                        help="Safety cap. Use 0 for unlimited after filtering.")
    parser.add_argument("--min-volume", type=float, default=None)
    parser.add_argument("--live-markets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--historical-markets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-status", nargs="+", default=["open", "closed", "settled"],
                        choices=["unopened", "open", "closed", "settled", "all"])
    parser.add_argument("--exclude-mve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pause", type=float, default=0.2)

    parser.add_argument("--l1", action=argparse.BooleanOptionalAction, default=True,
                        help="Collect top-of-book/candle-style data when available.")
    parser.add_argument("--l2", action=argparse.BooleanOptionalAction, default=False,
                        help="Collect full-depth order books where available.")
    parser.add_argument("--trades", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--current-depth", type=int, default=0,
                        help="Kalshi current orderbook depth, 0 means all levels.")

    parser.add_argument("--kalshi-intervals", type=int, nargs="+", default=[1, 60, 1440],
                        choices=[1, 60, 1440])
    parser.add_argument("--oddpool", action="store_true",
                        help="Use Oddpool for historical L1/L2/trades. Requires ODDPOOL_API_KEY.")
    parser.add_argument("--oddpool-granularity", nargs="+", default=["1m"], choices=["1m", "5m"])
    parser.add_argument("--oddpool-ohlcv", action="store_true")
    parser.add_argument("--oddpool-intervals", nargs="+", default=["6h", "1d"], choices=["6h", "1d", "1w", "1m"])
    parser.add_argument("--oddspipe", action="store_true",
                        help="Use OddsPipe normalized candles. Requires ODDSPIPE_API_KEY.")
    parser.add_argument("--oddspipe-intervals", nargs="+", default=["1m", "5m", "1h", "1d"])
    parser.add_argument("--pmxt", action="store_true",
                        help="Use optional pmxt SDK fallback. Requires pip/npm pmxt installs.")
    parser.add_argument("--pmxt-resolutions", nargs="+", default=["1m", "1h", "1d"])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out = args.out.resolve()
    start = parse_dt(args.since, default=datetime(2023, 1, 1, tzinfo=timezone.utc))
    end = parse_dt(args.until, default=utc_now())
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Discovering markets into {args.out}")
    markets = load_markets(args)
    print(f"Discovered {len(markets)} markets")

    manifest_rows = [source_row("kalshi_market_manifest", m, ticker=m.get("ticker")) for m in markets]
    append_jsonl(args.out / "manifest" / "markets.jsonl", manifest_rows)
    write_json(args.out / "manifest" / "run.json", {
        "pulled_at": iso(utc_now()),
        "since": iso(start),
        "until": iso(end),
        "market_count": len(markets),
        "sources": {
            "kalshi_official": True,
            "oddpool": args.oddpool,
            "oddspipe": args.oddspipe,
            "pmxt": args.pmxt,
        },
        "notes": [
            "Kalshi official L1 history is trades and candlesticks; official L2 is current orderbook only.",
            "Oddpool historical Kalshi data starts 2026-03-19 per docs.",
            "OddsPipe free tier advertises 30 days history; paid archive may go back further.",
            "pmxt provides current orderbook and OHLCV/trades abstractions, not historical L2 snapshots.",
        ],
    })

    if args.dry_run:
        print("Dry run complete; wrote manifest only.")
        return 0

    collect_kalshi_official(args, markets, start, end)

    if args.oddpool:
        collect_oddpool(args, markets, start, end)

    if args.oddspipe:
        collect_oddspipe(args, markets)

    if args.pmxt:
        collect_pmxt(args, markets, start, end)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
