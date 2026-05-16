#!/usr/bin/env python3
"""Collect weekly top-volume Kalshi markets and OHLCV candles."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import requests


BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
UTC = timezone.utc


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    @property
    def label(self) -> str:
        return self.start.date().isoformat()

    @property
    def start_ts(self) -> int:
        return int(self.start.timestamp())

    @property
    def end_ts(self) -> int:
        return int(self.end.timestamp())


class KalshiClient:
    def __init__(self, base_url: str, sleep_s: float, timeout_s: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.sleep_s = sleep_s
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "prophet-hacks-kalshi-top-volume/0.1",
                "Accept": "application/json",
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url + path
        backoff = 1.0
        last_exc: Exception | None = None
        for _ in range(8):
            if self.sleep_s:
                time.sleep(self.sleep_s)
            try:
                response = self.session.get(url, params=params or {}, timeout=self.timeout_s)
                if response.status_code == 429 or response.status_code >= 500:
                    time.sleep(backoff + random.random() * 0.25)
                    backoff = min(backoff * 2, 60)
                    continue
                response.raise_for_status()
                return response.json()
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                last_exc = exc
                time.sleep(backoff + random.random() * 0.25)
                backoff = min(backoff * 2, 60)
        if last_exc is not None:
            raise last_exc
        response.raise_for_status()
        return response.json()

    def paginate(
        self,
        path: str,
        params: dict[str, Any],
        key: str,
        *,
        max_pages: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        cursor: str | None = None
        pages = 0
        while True:
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            data = self.get(path, page_params)
            for item in data.get(key, []):
                yield item
            pages += 1
            if max_pages is not None and pages >= max_pages:
                break
            cursor = data.get("cursor") or None
            if not cursor:
                break


def parse_args() -> argparse.Namespace:
    default_out = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-05-09", help="Inclusive end date.")
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--candidate-multiplier", type=int, default=8)
    parser.add_argument("--period-interval", type=int, choices=[1, 60, 1440], default=1)
    parser.add_argument("--market-batch-size", type=int, default=40)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--max-trade-pages", type=int)
    parser.add_argument(
        "--ranking-source",
        choices=["market_close_volume", "trade_volume"],
        default="market_close_volume",
        help=(
            "market_close_volume ranks markets closing in the week by final volume_fp. "
            "trade_volume aggregates every trade in the week and is much slower."
        ),
    )
    parser.add_argument("--rank-only", action="store_true")
    parser.add_argument(
        "--candles-from-rankings",
        action="store_true",
        help="Skip ranking and download candles from existing weekly top300 CSV/metadata files.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def at_utc_midnight(value: date) -> datetime:
    return datetime.combine(value, dt_time.min, tzinfo=UTC)


def weekly_windows(start: date, inclusive_end: date) -> list[Window]:
    end_exclusive = at_utc_midnight(inclusive_end + timedelta(days=1))
    cursor = at_utc_midnight(start)
    windows: list[Window] = []
    while cursor < end_exclusive:
        next_end = min(cursor + timedelta(days=7), end_exclusive)
        windows.append(Window(start=cursor, end=next_end))
        cursor = next_end
    return windows


def parse_iso_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def decimal_value(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def decimal_to_str(value: Decimal) -> str:
    return format(value.normalize(), "f")


MARKET_CACHE_FIELDS = [
    "ticker",
    "event_ticker",
    "series_ticker",
    "market_type",
    "title",
    "subtitle",
    "rules_primary",
    "open_time",
    "close_time",
    "settlement_ts",
    "status",
    "result",
    "volume_fp",
    "open_interest_fp",
    "mve_collection_ticker",
    "mve_selected_legs",
]


def slim_market(market: dict[str, Any], source: str) -> dict[str, Any]:
    slim = {key: market.get(key) for key in MARKET_CACHE_FIELDS if key in market}
    slim["_metadata_source"] = source
    return slim


def ensure_dirs(out_dir: Path) -> None:
    for rel in ["rankings", "markets", "logs", "state"]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)
    (out_dir / "ohlcv").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def append_error(out_dir: Path, payload: dict[str, Any]) -> None:
    path = out_dir / "logs" / "errors.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp, path)


def aggregate_trade_volume(
    client: KalshiClient,
    window: Window,
    trades_cutoff: datetime,
    *,
    limit: int,
    max_pages: int | None,
) -> tuple[dict[str, Decimal], dict[str, int]]:
    volumes: dict[str, Decimal] = defaultdict(Decimal)
    counts: dict[str, int] = defaultdict(int)

    spans: list[tuple[str, int, int]] = []
    if window.end <= trades_cutoff:
        spans.append(("/historical/trades", window.start_ts, window.end_ts))
    elif window.start >= trades_cutoff:
        spans.append(("/markets/trades", window.start_ts, window.end_ts))
    else:
        spans.append(("/historical/trades", window.start_ts, int(trades_cutoff.timestamp())))
        spans.append(("/markets/trades", int(trades_cutoff.timestamp()), window.end_ts))

    for path, min_ts, max_ts in spans:
        if min_ts >= max_ts:
            continue
        params = {"limit": limit, "min_ts": min_ts, "max_ts": max_ts}
        for trade in client.paginate(path, params, "trades", max_pages=max_pages):
            ticker = trade.get("ticker")
            if not ticker:
                continue
            volumes[ticker] += decimal_value(trade.get("count_fp"))
            counts[ticker] += 1
    return volumes, counts


def fetch_live_markets_close_window(
    client: KalshiClient,
    week: Window,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    params = {
        "limit": limit,
        "min_close_ts": week.start_ts,
        "max_close_ts": week.end_ts,
        "mve_filter": "exclude",
    }
    markets: list[dict[str, Any]] = []
    for market in client.paginate("/markets", params, "markets"):
        markets.append(slim_market(market, "live"))
    return markets


def load_historical_market_cache(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    markets: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                markets.append(json.loads(line))
    return markets


def build_historical_market_cache(
    client: KalshiClient,
    path: Path,
    *,
    overall_start: datetime,
    overall_end: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    markets: list[dict[str, Any]] = []
    older_pages_seen = 0
    with tmp.open("w", encoding="utf-8") as handle:
        cursor: str | None = None
        pages = 0
        while True:
            params: dict[str, Any] = {"limit": limit, "mve_filter": "exclude"}
            if cursor:
                params["cursor"] = cursor
            data = client.get("/historical/markets", params)
            pages += 1
            page = data.get("markets", [])
            page_close_times = [parse_iso_ts(market.get("close_time")) for market in page]
            page_relevant = 0
            for market in page:
                close_time = parse_iso_ts(market.get("close_time"))
                if close_time is None:
                    continue
                if overall_start <= close_time < overall_end:
                    market = slim_market(market, "historical")
                    markets.append(market)
                    handle.write(json.dumps(market, sort_keys=True) + "\n")
                    page_relevant += 1

            valid_times = [value for value in page_close_times if value is not None]
            if valid_times and max(valid_times) < overall_start:
                older_pages_seen += 1
            else:
                older_pages_seen = 0
            cursor = data.get("cursor") or None
            if pages % 25 == 0:
                print(
                    f"  historical cache pages={pages} kept={len(markets)}",
                    flush=True,
                )
            if not cursor:
                break
            # The historical endpoint is observed to page newest-to-oldest. Keep
            # a small buffer before stopping so one out-of-order page does not
            # cut off relevant markets.
            if older_pages_seen >= 3 and page_relevant == 0:
                break
    os.replace(tmp, path)
    return markets


def historical_market_cache(
    client: KalshiClient,
    out_dir: Path,
    *,
    overall_start: datetime,
    overall_end: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    path = (
        out_dir
        / "markets"
        / f"historical_markets_{overall_start.date().isoformat()}_{(overall_end - timedelta(days=1)).date().isoformat()}.jsonl"
    )
    cached = load_historical_market_cache(path)
    if cached is not None:
        return cached
    return build_historical_market_cache(
        client,
        path,
        overall_start=overall_start,
        overall_end=overall_end,
        limit=limit,
    )


def rank_markets_by_close_volume(
    client: KalshiClient,
    week: Window,
    historical_cache: list[dict[str, Any]],
    *,
    limit: int,
) -> tuple[dict[str, Decimal], dict[str, int], dict[str, dict[str, Any]]]:
    live = fetch_live_markets_close_window(client, week, limit=limit)
    historical = [
        market
        for market in historical_cache
        if (close_time := parse_iso_ts(market.get("close_time"))) is not None
        and week.start <= close_time < week.end
    ]
    markets_by_ticker: dict[str, dict[str, Any]] = {}
    volumes: dict[str, Decimal] = {}
    trade_counts: dict[str, int] = {}
    for market in historical + live:
        ticker = market.get("ticker")
        if not ticker:
            continue
        markets_by_ticker[ticker] = market
        volumes[ticker] = decimal_value(market.get("volume_fp"))
        trade_counts[ticker] = 0
    return volumes, trade_counts, markets_by_ticker


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def fetch_market_metadata(
    client: KalshiClient,
    tickers: list[str],
    *,
    batch_size: int,
    limit: int,
) -> dict[str, dict[str, Any]]:
    markets: dict[str, dict[str, Any]] = {}
    for batch in chunked(tickers, batch_size):
        params = {
            "limit": limit,
            "tickers": ",".join(batch),
            "mve_filter": "exclude",
        }
        for path, source in [("/markets", "live"), ("/historical/markets", "historical")]:
            try:
                data = client.get(path, params)
            except requests.HTTPError:
                continue
            for market in data.get("markets", []):
                ticker = market.get("ticker")
                if ticker:
                    market = dict(market)
                    market["_metadata_source"] = source
                    markets[ticker] = market
    return markets


def is_multivariate_or_combo(market: dict[str, Any]) -> bool:
    if market.get("mve_collection_ticker") or market.get("mve_selected_legs"):
        return True
    event_ticker = str(market.get("event_ticker") or "")
    ticker = str(market.get("ticker") or "")
    if event_ticker.startswith(("KXMVE", "KXMV")) or ticker.startswith(("KXMVE", "KXMV")):
        return True
    custom = market.get("custom_strike")
    if isinstance(custom, dict):
        custom_keys = " ".join(custom.keys()).lower()
        if "multivariate" in custom_keys or "associated markets" in custom_keys:
            return True
    return False


def eligibility_reason(market: dict[str, Any] | None) -> str | None:
    if not market:
        return "metadata_missing"
    if market.get("market_type") != "binary":
        return "not_binary"
    if is_multivariate_or_combo(market):
        return "multivariate_or_combo"
    if not market.get("ticker") or not market.get("event_ticker"):
        return "ticker_missing"
    if not market.get("title"):
        return "title_missing"
    if not market.get("rules_primary"):
        return "rules_missing"
    if not market.get("open_time") or not market.get("close_time"):
        return "time_missing"
    status = str(market.get("status") or "").lower()
    if status in {"unopened", "paused", "canceled", "cancelled"}:
        return f"bad_status:{status}"
    return None


def selected_top_markets(
    client: KalshiClient,
    sorted_tickers: list[str],
    volumes: dict[str, Decimal],
    trade_counts: dict[str, int],
    *,
    top_n: int,
    candidate_multiplier: int,
    batch_size: int,
    limit: int,
    known_markets: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    markets: dict[str, dict[str, Any]] = dict(known_markets or {})
    exclusion_counts: dict[str, int] = defaultdict(int)
    target_candidates = min(len(sorted_tickers), max(top_n * candidate_multiplier, top_n))
    cursor = 0

    while len(selected) < top_n and cursor < len(sorted_tickers):
        batch_end = min(len(sorted_tickers), max(cursor + batch_size, target_candidates))
        candidate_slice = sorted_tickers[cursor:batch_end]
        missing = [ticker for ticker in candidate_slice if ticker not in markets]
        if missing:
            markets.update(fetch_market_metadata(client, missing, batch_size=batch_size, limit=limit))
        for ticker in candidate_slice:
            market = markets.get(ticker)
            reason = eligibility_reason(market)
            if reason:
                exclusion_counts[reason] += 1
                continue
            if any(row["ticker"] == ticker for row in selected):
                continue
            selected.append(
                {
                    "rank": len(selected) + 1,
                    "ticker": ticker,
                    "weekly_volume": decimal_to_str(volumes[ticker]),
                    "trade_count": trade_counts.get(ticker, 0),
                    "event_ticker": market.get("event_ticker"),
                    "series_ticker": infer_series_ticker(market),
                    "title": market.get("title"),
                    "subtitle": market.get("subtitle"),
                    "market_type": market.get("market_type"),
                    "status": market.get("status"),
                    "open_time": market.get("open_time"),
                    "close_time": market.get("close_time"),
                    "settlement_ts": market.get("settlement_ts"),
                    "result": market.get("result"),
                    "volume_fp": market.get("volume_fp"),
                    "open_interest_fp": market.get("open_interest_fp"),
                    "metadata_source": market.get("_metadata_source"),
                }
            )
            if len(selected) >= top_n:
                break
        cursor = batch_end
        target_candidates = min(len(sorted_tickers), target_candidates + top_n)

    if selected:
        selected_tickers = [row["ticker"] for row in selected]
        full_markets = fetch_market_metadata(
            client,
            selected_tickers,
            batch_size=batch_size,
            limit=limit,
        )
        markets.update(full_markets)
        for row in selected:
            market = markets.get(row["ticker"], {})
            row.update(
                {
                    "event_ticker": market.get("event_ticker", row.get("event_ticker")),
                    "series_ticker": infer_series_ticker(market) if market else row.get("series_ticker"),
                    "title": market.get("title", row.get("title")),
                    "subtitle": market.get("subtitle", row.get("subtitle")),
                    "market_type": market.get("market_type", row.get("market_type")),
                    "status": market.get("status", row.get("status")),
                    "open_time": market.get("open_time", row.get("open_time")),
                    "close_time": market.get("close_time", row.get("close_time")),
                    "settlement_ts": market.get("settlement_ts", row.get("settlement_ts")),
                    "result": market.get("result", row.get("result")),
                    "volume_fp": market.get("volume_fp", row.get("volume_fp")),
                    "open_interest_fp": market.get("open_interest_fp", row.get("open_interest_fp")),
                    "metadata_source": market.get("_metadata_source", row.get("metadata_source")),
                }
            )

    return selected, markets, dict(exclusion_counts)


def infer_series_ticker(market: dict[str, Any]) -> str:
    series = market.get("series_ticker")
    if series:
        return str(series)
    event_ticker = str(market.get("event_ticker") or "")
    if "-" in event_ticker:
        return event_ticker.split("-", 1)[0]
    return event_ticker


def market_uses_historical_candles(market: dict[str, Any], cutoff: datetime) -> bool:
    settlement = parse_iso_ts(market.get("settlement_ts"))
    if settlement is not None and settlement < cutoff:
        return True
    status = str(market.get("status") or "").lower()
    close_time = parse_iso_ts(market.get("close_time"))
    return status in {"settled", "finalized"} and close_time is not None and close_time < cutoff


def candle_path(
    out_dir: Path,
    *,
    period_interval: int,
    week_label: str,
    ticker: str,
) -> Path:
    safe_ticker = ticker.replace("/", "_")
    return out_dir / "ohlcv" / f"period_{period_interval}m" / f"week={week_label}" / f"{safe_ticker}.csv.gz"


def candle_value(block: dict[str, Any] | None, name: str) -> Any:
    if not isinstance(block, dict):
        return ""
    return block.get(name) or block.get(f"{name}_dollars") or ""


def flatten_candle(
    candle: dict[str, Any],
    *,
    ticker: str,
    week: Window,
    endpoint_source: str,
) -> dict[str, Any]:
    end_ts = candle.get("end_period_ts")
    end_time = ""
    if isinstance(end_ts, int):
        end_time = datetime.fromtimestamp(end_ts, UTC).isoformat().replace("+00:00", "Z")
    price = candle.get("price") or {}
    yes_bid = candle.get("yes_bid") or {}
    yes_ask = candle.get("yes_ask") or {}
    return {
        "week_start": week.start.isoformat().replace("+00:00", "Z"),
        "week_end": week.end.isoformat().replace("+00:00", "Z"),
        "ticker": ticker,
        "end_period_ts": end_ts or "",
        "end_period_time": end_time,
        "price_open": candle_value(price, "open"),
        "price_high": candle_value(price, "high"),
        "price_low": candle_value(price, "low"),
        "price_close": candle_value(price, "close"),
        "price_mean": candle_value(price, "mean"),
        "price_previous": candle_value(price, "previous"),
        "price_min": candle_value(price, "min"),
        "price_max": candle_value(price, "max"),
        "yes_bid_open": candle_value(yes_bid, "open"),
        "yes_bid_high": candle_value(yes_bid, "high"),
        "yes_bid_low": candle_value(yes_bid, "low"),
        "yes_bid_close": candle_value(yes_bid, "close"),
        "yes_ask_open": candle_value(yes_ask, "open"),
        "yes_ask_high": candle_value(yes_ask, "high"),
        "yes_ask_low": candle_value(yes_ask, "low"),
        "yes_ask_close": candle_value(yes_ask, "close"),
        "volume": candle.get("volume") or candle.get("volume_fp") or "",
        "open_interest": candle.get("open_interest") or candle.get("open_interest_fp") or "",
        "endpoint_source": endpoint_source,
    }


CANDLE_FIELDS = [
    "week_start",
    "week_end",
    "ticker",
    "end_period_ts",
    "end_period_time",
    "price_open",
    "price_high",
    "price_low",
    "price_close",
    "price_mean",
    "price_previous",
    "price_min",
    "price_max",
    "yes_bid_open",
    "yes_bid_high",
    "yes_bid_low",
    "yes_bid_close",
    "yes_ask_open",
    "yes_ask_high",
    "yes_ask_low",
    "yes_ask_close",
    "volume",
    "open_interest",
    "endpoint_source",
]


def fetch_candles_once(
    client: KalshiClient,
    market: dict[str, Any],
    *,
    start_ts: int,
    end_ts: int,
    period_interval: int,
    historical: bool,
) -> tuple[str, list[dict[str, Any]]]:
    ticker = market["ticker"]
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval}
    if historical:
        data = client.get(f"/historical/markets/{ticker}/candlesticks", params)
        return "historical", data.get("candlesticks", [])
    series = infer_series_ticker(market)
    data = client.get(f"/series/{series}/markets/{ticker}/candlesticks", params)
    return "live", data.get("candlesticks", [])


def fetch_candles_range_with_fallback(
    client: KalshiClient,
    market: dict[str, Any],
    *,
    start_ts: int,
    end_ts: int,
    period_interval: int,
    cutoff: datetime,
) -> tuple[str, list[dict[str, Any]]]:
    preferred_historical = market_uses_historical_candles(market, cutoff)
    attempts = [preferred_historical, not preferred_historical]
    last_exc: Exception | None = None
    for historical in attempts:
        try:
            return fetch_candles_once(
                client,
                market,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=period_interval,
                historical=historical,
            )
        except requests.HTTPError as exc:
            last_exc = exc
            status = getattr(exc.response, "status_code", None)
            if status not in {400, 404}:
                raise
    if last_exc:
        raise last_exc
    return "unknown", []


def fetch_candles_with_fallback(
    client: KalshiClient,
    market: dict[str, Any],
    week: Window,
    *,
    period_interval: int,
    cutoff: datetime,
    max_candles_per_request: int = 4800,
) -> list[tuple[str, dict[str, Any]]]:
    step_seconds = max_candles_per_request * period_interval * 60
    cursor = week.start_ts
    rows: list[tuple[str, dict[str, Any]]] = []
    while cursor < week.end_ts:
        chunk_end = min(cursor + step_seconds, week.end_ts)
        endpoint_source, candles = fetch_candles_range_with_fallback(
            client,
            market,
            start_ts=cursor,
            end_ts=chunk_end,
            period_interval=period_interval,
            cutoff=cutoff,
        )
        rows.extend((endpoint_source, candle) for candle in candles)
        cursor = chunk_end
    return rows


def write_candles(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CANDLE_FIELDS})
    os.replace(tmp, path)


def download_week_candles(
    client: KalshiClient,
    out_dir: Path,
    week: Window,
    selected: list[dict[str, Any]],
    markets: dict[str, dict[str, Any]],
    *,
    period_interval: int,
    cutoff: datetime,
    force: bool,
) -> dict[str, Any]:
    stats = {"downloaded": 0, "skipped": 0, "errors": 0, "rows": 0}
    total = len(selected)
    for idx, row in enumerate(selected, start=1):
        ticker = row["ticker"]
        path = candle_path(
            out_dir,
            period_interval=period_interval,
            week_label=week.label,
            ticker=ticker,
        )
        if path.exists() and path.stat().st_size > 40 and not force:
            stats["skipped"] += 1
            continue
        market = markets.get(ticker)
        if not market:
            stats["errors"] += 1
            append_error(
                out_dir,
                {"week": week.label, "ticker": ticker, "stage": "candles", "error": "metadata_missing"},
            )
            continue
        try:
            candle_items = fetch_candles_with_fallback(
                client,
                market,
                week,
                period_interval=period_interval,
                cutoff=cutoff,
            )
            flat_rows = [
                flatten_candle(candle, ticker=ticker, week=week, endpoint_source=endpoint_source)
                for endpoint_source, candle in candle_items
            ]
            write_candles(path, flat_rows)
            stats["downloaded"] += 1
            stats["rows"] += len(flat_rows)
        except Exception as exc:  # noqa: BLE001 - collect-and-continue job.
            stats["errors"] += 1
            append_error(
                out_dir,
                {
                    "week": week.label,
                    "ticker": ticker,
                    "stage": "candles",
                    "error": repr(exc),
                },
            )
        if idx % 25 == 0 or idx == total:
            print(
                f"  {week.label} candles {idx}/{total}: "
                f"downloaded={stats['downloaded']} skipped={stats['skipped']} "
                f"errors={stats['errors']} rows={stats['rows']}",
                flush=True,
            )
    return stats


def write_week_outputs(
    out_dir: Path,
    week: Window,
    sorted_tickers: list[str],
    volumes: dict[str, Decimal],
    trade_counts: dict[str, int],
    selected: list[dict[str, Any]],
    markets: dict[str, dict[str, Any]],
) -> None:
    volume_rows = [
        {
            "rank": idx + 1,
            "ticker": ticker,
            "weekly_volume": decimal_to_str(volumes[ticker]),
            "trade_count": trade_counts.get(ticker, 0),
        }
        for idx, ticker in enumerate(sorted_tickers)
    ]
    write_csv(
        out_dir / "rankings" / f"{week.label}_volume_all.csv",
        volume_rows,
        ["rank", "ticker", "weekly_volume", "trade_count"],
    )
    selected_rows = []
    for row in selected:
        out = dict(row)
        out["week_start"] = week.start.isoformat().replace("+00:00", "Z")
        out["week_end"] = week.end.isoformat().replace("+00:00", "Z")
        selected_rows.append(out)
    selected_fields = [
        "week_start",
        "week_end",
        "rank",
        "ticker",
        "weekly_volume",
        "trade_count",
        "event_ticker",
        "series_ticker",
        "title",
        "subtitle",
        "market_type",
        "status",
        "open_time",
        "close_time",
        "settlement_ts",
        "result",
        "volume_fp",
        "open_interest_fp",
        "metadata_source",
    ]
    write_csv(out_dir / "rankings" / f"{week.label}_top300.csv", selected_rows, selected_fields)
    meta_path = out_dir / "markets" / f"{week.label}_selected_markets.jsonl"
    with meta_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            market = markets.get(row["ticker"])
            if market:
                handle.write(json.dumps(market, sort_keys=True) + "\n")


def update_combined_top_csv(out_dir: Path, weeks: list[Window]) -> None:
    combined: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None
    for week in weeks:
        path = out_dir / "rankings" / f"{week.label}_top300.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or fieldnames
            combined.extend(reader)
    if fieldnames:
        write_csv(out_dir / "weekly_top_markets.csv", combined, fieldnames)


def read_week_selection(out_dir: Path, week: Window) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    ranking_path = out_dir / "rankings" / f"{week.label}_top300.csv"
    metadata_path = out_dir / "markets" / f"{week.label}_selected_markets.jsonl"
    if not ranking_path.exists():
        raise FileNotFoundError(f"missing ranking file: {ranking_path}")
    selected: list[dict[str, Any]] = []
    with ranking_path.open(newline="", encoding="utf-8") as handle:
        selected.extend(csv.DictReader(handle))

    markets: dict[str, dict[str, Any]] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                market = json.loads(line)
                ticker = market.get("ticker")
                if ticker:
                    markets[ticker] = market
    return selected, markets


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    ensure_dirs(out_dir)

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    weeks = weekly_windows(start, end)
    client = KalshiClient(args.base_url, sleep_s=args.sleep)

    cutoff = client.get("/historical/cutoff")
    market_cutoff = parse_iso_ts(cutoff.get("market_settled_ts")) or datetime(1970, 1, 1, tzinfo=UTC)
    trades_cutoff = parse_iso_ts(cutoff.get("trades_created_ts")) or market_cutoff

    manifest = {
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "start_date": args.start_date,
        "end_date_inclusive": args.end_date,
        "week_windowing": "contiguous 7-day windows anchored at start_date; final window may be partial",
        "top_n": args.top_n,
        "period_interval_minutes": args.period_interval,
        "ranking_source": args.ranking_source,
        "base_url": args.base_url,
        "historical_cutoff": cutoff,
        "ranking_source_notes": {
            "market_close_volume": (
                "Ranks markets whose close_time falls in the week by final volume_fp. "
                "This is tractable for large historical pulls but is not causal for pre-week selection."
            ),
            "trade_volume": (
                "Aggregates count_fp from every trade in the week. This is causal, "
                "but very slow on active days due to public trades pagination."
            ),
        },
        "filters": [
            "market_type == binary",
            "mve_collection_ticker and mve_selected_legs absent",
            "event/ticker prefixes KXMVE and KXMV excluded",
            "title, rules_primary, open_time, close_time present",
            "unopened/paused/canceled markets excluded",
        ],
        "source_docs": [
            "https://docs.kalshi.com/api-reference/market/get-trades",
            "https://docs.kalshi.com/api-reference/historical/get-historical-trades",
            "https://docs.kalshi.com/api-reference/market/get-market-candlesticks",
            "https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks",
            "https://docs.kalshi.com/getting_started/historical_data",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    if args.candles_from_rankings:
        run_stats: list[dict[str, Any]] = []
        for idx, week in enumerate(weeks, start=1):
            print(f"[{idx}/{len(weeks)}] {week.label}: downloading candles from saved rankings", flush=True)
            selected, markets = read_week_selection(out_dir, week)
            candle_stats = download_week_candles(
                client,
                out_dir,
                week,
                selected,
                markets,
                period_interval=args.period_interval,
                cutoff=market_cutoff,
                force=args.force,
            )
            print(f"[{week.label}] candles {candle_stats}", flush=True)
            run_stats.append(
                {
                    "week_start": week.label,
                    "week_end": week.end.date().isoformat(),
                    "selected": len(selected),
                    "candles": candle_stats,
                }
            )
            write_json(out_dir / "state" / "candle_run_state.json", {"weeks": run_stats})
        print(f"Done. Candle outputs are in {out_dir / 'ohlcv'}", flush=True)
        return 0

    historical_cache: list[dict[str, Any]] = []
    if args.ranking_source == "market_close_volume":
        print("Building/loading historical market metadata cache...", flush=True)
        historical_cache = historical_market_cache(
            client,
            out_dir,
            overall_start=weeks[0].start,
            overall_end=weeks[-1].end,
            limit=args.limit,
        )
        print(f"Historical cache contains {len(historical_cache)} markets in range", flush=True)

    run_stats: list[dict[str, Any]] = []
    for idx, week in enumerate(weeks, start=1):
        print(f"[{idx}/{len(weeks)}] {week.label}: ranking markets", flush=True)
        known_markets: dict[str, dict[str, Any]] | None = None
        if args.ranking_source == "trade_volume":
            volumes, trade_counts = aggregate_trade_volume(
                client,
                week,
                trades_cutoff,
                limit=args.limit,
                max_pages=args.max_trade_pages,
            )
        else:
            volumes, trade_counts, known_markets = rank_markets_by_close_volume(
                client,
                week,
                historical_cache,
                limit=args.limit,
            )
        sorted_tickers = sorted(volumes, key=lambda ticker: (-volumes[ticker], ticker))
        print(f"[{week.label}] ranked {len(sorted_tickers)} traded tickers", flush=True)

        selected, markets, exclusion_counts = selected_top_markets(
            client,
            sorted_tickers,
            volumes,
            trade_counts,
            top_n=args.top_n,
            candidate_multiplier=args.candidate_multiplier,
            batch_size=args.market_batch_size,
            limit=args.limit,
            known_markets=known_markets,
        )
        write_week_outputs(out_dir, week, sorted_tickers, volumes, trade_counts, selected, markets)
        print(
            f"[{week.label}] selected {len(selected)} markets; exclusions={exclusion_counts}",
            flush=True,
        )

        candle_stats = {"downloaded": 0, "skipped": 0, "errors": 0, "rows": 0}
        if not args.rank_only:
            candle_stats = download_week_candles(
                client,
                out_dir,
                week,
                selected,
                markets,
                period_interval=args.period_interval,
                cutoff=market_cutoff,
                force=args.force,
            )
            print(f"[{week.label}] candles {candle_stats}", flush=True)

        run_stats.append(
            {
                "week_start": week.label,
                "week_end": week.end.date().isoformat(),
                "traded_tickers": len(sorted_tickers),
                "selected": len(selected),
                "exclusions": exclusion_counts,
                "candles": candle_stats,
            }
        )
        write_json(out_dir / "state" / "run_state.json", {"weeks": run_stats})

    update_combined_top_csv(out_dir, weeks)
    print(f"Done. Outputs are in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
