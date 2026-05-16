#!/usr/bin/env python3
"""Sample random market-time points for sparse LLM trading backtests.

The core hyperparameter is --calls-per-week. With the Jan 1-May 9 dataset this
has 19 weekly windows, so --calls-per-week 1 produces 19 planned LLM calls.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UTC = timezone.utc


@dataclass(frozen=True)
class Quote:
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float

    @property
    def market_mid(self) -> float:
        return max(0.01, min(0.99, (self.yes_bid + self.yes_ask) / 2.0))

    @property
    def spread(self) -> float:
        return max(0.0, self.yes_ask - self.yes_bid)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--calls-per-week", "-x", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--period-interval", type=int, default=1)
    parser.add_argument("--min-time-to-close-minutes", type=float, default=30.0)
    parser.add_argument("--with-replacement", action="store_true")
    parser.add_argument("--out-jsonl", type=Path)
    parser.add_argument("--out-csv", type=Path)
    return parser.parse_args()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_week_metadata(data_dir: Path, week: str) -> dict[str, dict[str, Any]]:
    path = data_dir / "markets" / f"{week}_selected_markets.jsonl"
    if not path.exists():
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ticker = row.get("ticker")
            if ticker:
                metadata[ticker] = row
    return metadata


def load_weekly_markets(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    path = data_dir / "weekly_top_markets.csv"
    by_week: dict[str, list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            week = row["week_start"][:10]
            by_week.setdefault(week, []).append(row)
    for week, rows in by_week.items():
        metadata = load_week_metadata(data_dir, week)
        for row in rows:
            ticker = row.get("ticker")
            if ticker in metadata:
                # Preserve ranking fields from CSV, but enrich with full Kalshi
                # metadata such as rules_primary for LLM prompts.
                row.update({k: v for k, v in metadata[ticker].items() if v not in (None, "")})
    return dict(sorted(by_week.items()))


def candle_path(data_dir: Path, period_interval: int, week: str, ticker: str) -> Path:
    return data_dir / "ohlcv" / f"period_{period_interval}m" / f"week={week}" / f"{ticker}.csv.gz"


def quote_from_candle(row: dict[str, Any]) -> Quote | None:
    yes_bid = parse_float(row.get("yes_bid_close"))
    yes_ask = parse_float(row.get("yes_ask_close"))
    if yes_bid is None or yes_ask is None:
        return None
    if not (0.0 <= yes_bid <= 1.0 and 0.0 <= yes_ask <= 1.0):
        return None
    if yes_ask <= 0.0 or yes_bid >= 1.0 or yes_bid > yes_ask:
        return None
    # Kalshi binary NO prices are implied from YES top-of-book.
    no_bid = max(0.0, min(1.0, 1.0 - yes_ask))
    no_ask = max(0.0, min(1.0, 1.0 - yes_bid))
    return Quote(yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask)


def eligible_candles(
    path: Path,
    market: dict[str, Any],
    *,
    min_time_to_close_minutes: float,
) -> list[tuple[dict[str, Any], Quote, float]]:
    close_time = parse_iso(market.get("close_time"))
    if close_time is None or not path.exists():
        return []

    out: list[tuple[dict[str, Any], Quote, float]] = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_time = parse_iso(row.get("end_period_time"))
            if sample_time is None:
                continue
            minutes_to_close = (close_time - sample_time).total_seconds() / 60.0
            if minutes_to_close < min_time_to_close_minutes:
                continue
            quote = quote_from_candle(row)
            if quote is None:
                continue
            out.append((row, quote, minutes_to_close))
    return out


def result_to_outcome(result: str | None) -> int | None:
    value = (result or "").strip().lower()
    if value == "yes":
        return 1
    if value == "no":
        return 0
    return None


def category_from_series(series: str | None) -> str:
    s = (series or "").upper()
    if any(s.startswith(prefix) for prefix in ("KXNBA", "KXNCAAMB", "KXNCAAWB", "KXMLB", "KXNHL", "KXNFL", "KXATP", "KXWTA", "KXUFC", "KXEPL", "KXUCL", "KXIPL")):
        return "Sports"
    if any(s.startswith(prefix) for prefix in ("KXBTC", "KXETH", "KXSOL", "KXDOGE", "KXXRP")):
        return "Crypto"
    if any(s.startswith(prefix) for prefix in ("KXFED", "KXCPI", "KXGDP", "KXJOBS")):
        return "Economics"
    if any(s.startswith(prefix) for prefix in ("KXHIGH", "KXLOW", "KXTEMP", "KXRAIN", "KXSNOW")):
        return "Weather"
    if any(s.startswith(prefix) for prefix in ("KXPRES", "KXSENATE", "KXHOUSE", "KXELEC")):
        return "Politics"
    return "Other"


def build_sample(
    *,
    sample_id: str,
    week: str,
    market: dict[str, Any],
    candle: dict[str, Any],
    quote: Quote,
    minutes_to_close: float,
) -> dict[str, Any]:
    outcome_yes = result_to_outcome(market.get("result"))
    event = {
        "event_ticker": market.get("event_ticker") or "",
        "market_ticker": market.get("ticker") or "",
        "title": market.get("title") or "",
        "subtitle": market.get("subtitle") or None,
        "description": None,
        "category": category_from_series(market.get("series_ticker")),
        "rules": market.get("rules_primary") or None,
        "close_time": market.get("close_time") or "",
        "outcomes": ["YES", "NO"],
    }
    packet = {
        "as_of": candle.get("end_period_time"),
        "event_ticker": event["event_ticker"],
        "market_ticker": event["market_ticker"],
        "title": event["title"],
        "subtitle": event["subtitle"],
        "rules": event["rules"],
        "category": event["category"],
        "close_time": event["close_time"],
        "outcomes": ["YES", "NO"],
        "kalshi": {
            "yes_bid": quote.yes_bid,
            "yes_ask": quote.yes_ask,
            "no_bid": quote.no_bid,
            "no_ask": quote.no_ask,
            "last_price": parse_float(candle.get("price_close")),
            "volume": parse_float(candle.get("volume")),
            "open_interest": parse_float(candle.get("open_interest")),
            "snapshot_time": candle.get("end_period_time"),
        },
        "retrieval": {},
    }
    return {
        "sample_id": sample_id,
        "week": week,
        "sample_time": candle.get("end_period_time"),
        "minutes_to_close": round(minutes_to_close, 3),
        "ticker": market.get("ticker"),
        "rank": int(market.get("rank") or 0),
        "weekly_volume": parse_float(market.get("weekly_volume")),
        "series_ticker": market.get("series_ticker"),
        "market_result": market.get("result"),
        "outcome_yes": outcome_yes,
        "quote": {
            "yes_bid": quote.yes_bid,
            "yes_ask": quote.yes_ask,
            "no_bid": quote.no_bid,
            "no_ask": quote.no_ask,
            "market_mid": quote.market_mid,
            "spread": quote.spread,
        },
        "event": event,
        "market_packet": packet,
        "candle": candle,
    }


def choose_market_sample(
    *,
    rng: random.Random,
    data_dir: Path,
    week: str,
    markets: list[dict[str, Any]],
    period_interval: int,
    min_time_to_close_minutes: float,
    used_tickers: set[str],
    with_replacement: bool,
    sample_id: str,
) -> dict[str, Any]:
    candidates = list(markets)
    rng.shuffle(candidates)
    for market in candidates:
        ticker = market.get("ticker") or ""
        if not with_replacement and ticker in used_tickers:
            continue
        path = candle_path(data_dir, period_interval, week, ticker)
        candles = eligible_candles(
            path,
            market,
            min_time_to_close_minutes=min_time_to_close_minutes,
        )
        if not candles:
            continue
        candle, quote, minutes_to_close = rng.choice(candles)
        used_tickers.add(ticker)
        return build_sample(
            sample_id=sample_id,
            week=week,
            market=market,
            candle=candle,
            quote=quote,
            minutes_to_close=minutes_to_close,
        )
    raise RuntimeError(f"No eligible market-time sample found for week {week}")


CSV_FIELDS = [
    "sample_id",
    "week",
    "sample_time",
    "minutes_to_close",
    "ticker",
    "rank",
    "weekly_volume",
    "series_ticker",
    "market_result",
    "outcome_yes",
    "yes_bid",
    "yes_ask",
    "no_bid",
    "no_ask",
    "market_mid",
    "spread",
    "title",
]


def write_outputs(samples: list[dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_tmp = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with jsonl_tmp.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
    os.replace(jsonl_tmp, jsonl_path)

    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for sample in samples:
            quote = sample["quote"]
            row = {
                "sample_id": sample["sample_id"],
                "week": sample["week"],
                "sample_time": sample["sample_time"],
                "minutes_to_close": sample["minutes_to_close"],
                "ticker": sample["ticker"],
                "rank": sample["rank"],
                "weekly_volume": sample["weekly_volume"],
                "series_ticker": sample["series_ticker"],
                "market_result": sample["market_result"],
                "outcome_yes": sample["outcome_yes"],
                "yes_bid": quote["yes_bid"],
                "yes_ask": quote["yes_ask"],
                "no_bid": quote["no_bid"],
                "no_ask": quote["no_ask"],
                "market_mid": quote["market_mid"],
                "spread": quote["spread"],
                "title": sample["event"]["title"],
            }
            writer.writerow(row)
    os.replace(csv_tmp, csv_path)


def main() -> int:
    args = parse_args()
    if args.calls_per_week < 1:
        raise ValueError("--calls-per-week must be >= 1")

    data_dir = args.data_dir.resolve()
    out_jsonl = args.out_jsonl or data_dir / "samples" / f"llm_calls_x{args.calls_per_week}_seed{args.seed}.jsonl"
    out_csv = args.out_csv or data_dir / "samples" / f"llm_calls_x{args.calls_per_week}_seed{args.seed}.csv"

    rng = random.Random(args.seed)
    by_week = load_weekly_markets(data_dir)
    samples: list[dict[str, Any]] = []
    for week, markets in by_week.items():
        if args.calls_per_week > len(markets) and not args.with_replacement:
            raise ValueError(
                f"week {week} has only {len(markets)} markets; use --with-replacement "
                "or lower --calls-per-week"
            )
        used_tickers: set[str] = set()
        for idx in range(1, args.calls_per_week + 1):
            sample = choose_market_sample(
                rng=rng,
                data_dir=data_dir,
                week=week,
                markets=markets,
                period_interval=args.period_interval,
                min_time_to_close_minutes=args.min_time_to_close_minutes,
                used_tickers=used_tickers,
                with_replacement=args.with_replacement,
                sample_id=f"{week}-{idx:03d}",
            )
            samples.append(sample)

    write_outputs(samples, out_jsonl, out_csv)
    print(
        json.dumps(
            {
                "weeks": len(by_week),
                "calls_per_week": args.calls_per_week,
                "total_calls": len(samples),
                "seed": args.seed,
                "out_jsonl": str(out_jsonl),
                "out_csv": str(out_csv),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
