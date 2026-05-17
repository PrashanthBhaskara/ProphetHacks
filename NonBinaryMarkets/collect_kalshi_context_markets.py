#!/usr/bin/env python3
"""Collect weekly Kalshi context groups and component-market candles.

The target trading dataset is binary. This collector builds event-level context
groups around related component markets so an LLM can see sibling outcomes,
spreads, totals, ladders, and combo-style markets when forecasting a target.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
KALSHI_TOPVOL = ROOT / "Kalshitopvolmarkets"
sys.path.insert(0, str(KALSHI_TOPVOL))

import collect_kalshi_top_volume as base  # noqa: E402


UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    default_out = Path(__file__).resolve().parent
    default_cache = (
        KALSHI_TOPVOL
        / "markets"
        / "historical_markets_2026-01-01_2026-05-09.jsonl"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-05-09", help="Inclusive end date.")
    parser.add_argument(
        "--resume-from-week",
        help="Only process weekly windows whose label is this date or later, preserving combined outputs for earlier weeks.",
    )
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--base-url", default=base.BASE_URL)
    parser.add_argument("--top-groups-per-week", type=int, default=250)
    parser.add_argument(
        "--max-markets-per-group",
        type=int,
        default=0,
        help="Maximum component markets per selected group. Use 0 for complete component sets.",
    )
    parser.add_argument("--min-markets-per-group", type=int, default=2)
    parser.add_argument("--period-interval", type=int, choices=[1, 60, 1440], default=1)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--historical-cache", type=Path, default=default_cache)
    parser.add_argument(
        "--topvol-selected-dir",
        type=Path,
        default=KALSHI_TOPVOL / "markets",
        help="Directory containing {week}_selected_markets.jsonl from the binary top-volume pull.",
    )
    parser.add_argument(
        "--include-live-supplement",
        action="store_true",
        help="Also query /markets by weekly close window. Slower; unnecessary for the completed Jan-May historical pull.",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Build a fresh context cache from /historical/markets instead of reusing the binary pull cache.",
    )
    parser.add_argument("--rank-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> None:
    for rel in ["rankings", "markets", "ohlcv", "logs", "state", "indexes"]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp, path)


def decimal_to_str(value: Decimal) -> str:
    return format(value.normalize(), "f")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def context_slim_market(market: dict[str, Any], source: str) -> dict[str, Any]:
    fields = set(base.MARKET_CACHE_FIELDS) | {
        "custom_strike",
        "yes_sub_title",
        "no_sub_title",
        "rules_secondary",
        "volume_24h_fp",
        "yes_bid_dollars",
        "yes_ask_dollars",
        "no_bid_dollars",
        "no_ask_dollars",
    }
    slim = {key: market.get(key) for key in fields if key in market}
    slim["_metadata_source"] = market.get("_metadata_source") or source
    return slim


def build_context_historical_cache(
    client: base.KalshiClient,
    out_dir: Path,
    *,
    overall_start: datetime,
    overall_end: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    path = (
        out_dir
        / "markets"
        / f"historical_context_markets_{overall_start.date().isoformat()}_{(overall_end - timedelta(days=1)).date().isoformat()}.jsonl"
    )
    cached = load_jsonl(path)
    if cached:
        return cached

    tmp = path.with_suffix(path.suffix + ".tmp")
    markets: list[dict[str, Any]] = []
    older_pages_seen = 0
    with tmp.open("w", encoding="utf-8") as handle:
        cursor: str | None = None
        pages = 0
        while True:
            params: dict[str, Any] = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
            data = client.get("/historical/markets", params)
            pages += 1
            page = data.get("markets", [])
            page_relevant = 0
            page_close_times = [base.parse_iso_ts(market.get("close_time")) for market in page]
            for market in page:
                close_time = base.parse_iso_ts(market.get("close_time"))
                if close_time is None:
                    continue
                if overall_start <= close_time < overall_end:
                    slim = context_slim_market(market, "historical")
                    markets.append(slim)
                    handle.write(json.dumps(slim, sort_keys=True) + "\n")
                    page_relevant += 1
            valid_times = [value for value in page_close_times if value is not None]
            if valid_times and max(valid_times) < overall_start:
                older_pages_seen += 1
            else:
                older_pages_seen = 0
            cursor = data.get("cursor") or None
            if pages % 25 == 0:
                print(f"  context historical cache pages={pages} kept={len(markets)}", flush=True)
            if not cursor:
                break
            if older_pages_seen >= 3 and page_relevant == 0:
                break
    os.replace(tmp, path)
    return markets


def fetch_live_context_markets(
    client: base.KalshiClient,
    week: base.Window,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    markets_by_ticker: dict[str, dict[str, Any]] = {}
    base_params = {
        "limit": limit,
        "min_close_ts": week.start_ts,
        "max_close_ts": week.end_ts,
    }
    for mve_filter in (None, "only"):
        params = dict(base_params)
        if mve_filter:
            params["mve_filter"] = mve_filter
        try:
            for market in client.paginate("/markets", params, "markets"):
                ticker = market.get("ticker")
                if ticker:
                    markets_by_ticker[ticker] = context_slim_market(market, "live")
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in {400, 404}:
                raise
    return list(markets_by_ticker.values())


def fetch_context_market_metadata(
    client: base.KalshiClient,
    tickers: list[str],
    *,
    batch_size: int = 40,
    limit: int = 1000,
) -> dict[str, dict[str, Any]]:
    markets: dict[str, dict[str, Any]] = {}
    for batch in base.chunked(tickers, batch_size):
        params = {"limit": limit, "tickers": ",".join(batch)}
        for path, source in [("/markets", "live"), ("/historical/markets", "historical")]:
            try:
                data = client.get(path, params)
            except requests.HTTPError:
                continue
            for market in data.get("markets", []):
                ticker = market.get("ticker")
                if ticker:
                    markets[ticker] = context_slim_market(market, source)
    return markets


def group_key(market: dict[str, Any]) -> tuple[str, str, str]:
    mve_collection = market.get("mve_collection_ticker")
    if mve_collection:
        return f"mve:{mve_collection}", "mve_collection", str(mve_collection)
    event_ticker = market.get("event_ticker") or market.get("ticker") or "UNKNOWN"
    return f"event:{event_ticker}", "event", str(event_ticker)


def is_usable_component(market: dict[str, Any]) -> bool:
    if not market.get("ticker") or not market.get("event_ticker"):
        return False
    if not market.get("title"):
        return False
    if not market.get("open_time") or not market.get("close_time"):
        return False
    status = str(market.get("status") or "").lower()
    return status not in {"unopened", "paused", "canceled", "cancelled"}


def is_context_group(markets: list[dict[str, Any]], min_markets: int) -> bool:
    if len(markets) >= min_markets:
        return True
    for market in markets:
        if market.get("market_type") != "binary":
            return True
        if market.get("mve_collection_ticker") or market.get("mve_selected_legs"):
            return True
    return False


def market_volume(market: dict[str, Any]) -> Decimal:
    return base.decimal_value(market.get("volume_fp"))


def market_trade_count(_: dict[str, Any]) -> int:
    return 0


def classify_group(group_kind: str, group_ticker: str, components: list[dict[str, Any]]) -> tuple[str, str]:
    series = sorted({base.infer_series_ticker(market).upper() for market in components})
    series_text = ",".join(series)
    title_text = " ".join(str(market.get("title") or "").lower() for market in components)
    ticker = group_ticker.upper()
    component_count = len(components)

    if group_kind == "mve_collection":
        return "mve_combo", "multivariate combo/collection"
    if "SPREAD" in series_text or "wins by over" in title_text or "points?" in title_text:
        return "spread_ladder", series_text
    if "TOTAL" in series_text or "total" in series_text or "combined score" in title_text:
        return "total_ladder", series_text
    if any(token in series_text or token in ticker for token in ("PGATOUR", "MARMAD", "OSCAR", "NASCAR", "NEXT", "WORLD", "MVP", "CONTEST")):
        return "multi_outcome_winner", series_text
    if component_count == 2 and ("GAME" in series_text or "MATCH" in series_text or "FIGHT" in series_text):
        return "two_sided_winner", series_text
    if component_count > 2:
        return "multi_component_event", series_text
    return "related_binary_pair", series_text


def selected_context_groups(
    week: base.Window,
    markets: list[dict[str, Any]],
    *,
    top_groups: int,
    max_markets_per_group: int,
    min_markets_per_group: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, Any]] = {}
    for market in markets:
        close_time = base.parse_iso_ts(market.get("close_time"))
        if close_time is None or not (week.start <= close_time < week.end):
            continue
        if not is_usable_component(market):
            continue
        key, kind, ticker = group_key(market)
        row = groups.setdefault(
            key,
            {
                "group_key": key,
                "group_kind": kind,
                "group_ticker": ticker,
                "markets": [],
                "group_volume": Decimal("0"),
                "event_tickers": set(),
                "series_tickers": set(),
            },
        )
        row["markets"].append(market)
        row["group_volume"] += market_volume(market)
        row["event_tickers"].add(market.get("event_ticker") or "")
        row["series_tickers"].add(base.infer_series_ticker(market))

    candidate_groups = [
        group for group in groups.values()
        if is_context_group(group["markets"], min_markets_per_group)
    ]
    candidate_groups.sort(key=lambda g: (-g["group_volume"], g["group_key"]))
    selected = candidate_groups[:top_groups]

    group_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    markets_by_ticker: dict[str, dict[str, Any]] = {}
    for rank, group in enumerate(selected, start=1):
        components = sorted(
            group["markets"],
            key=lambda market: (-market_volume(market), market.get("ticker") or ""),
        )
        selected_components = (
            components
            if max_markets_per_group <= 0
            else components[:max_markets_per_group]
        )
        group_type_label, group_type_detail = classify_group(
            group["group_kind"],
            group["group_ticker"],
            selected_components,
        )
        close_times = [
            base.parse_iso_ts(market.get("close_time"))
            for market in selected_components
            if base.parse_iso_ts(market.get("close_time")) is not None
        ]
        group_row = {
            "week_start": week.start.isoformat().replace("+00:00", "Z"),
            "week_end": week.end.isoformat().replace("+00:00", "Z"),
            "rank": rank,
            "group_key": group["group_key"],
            "group_kind": group["group_kind"],
            "group_type_label": group_type_label,
            "group_type_detail": group_type_detail,
            "group_ticker": group["group_ticker"],
            "group_volume": decimal_to_str(group["group_volume"]),
            "component_count": len(components),
            "selected_component_count": len(selected_components),
            "component_set_complete": str(len(selected_components) == len(components)).lower(),
            "event_tickers": ",".join(sorted(value for value in group["event_tickers"] if value)),
            "series_tickers": ",".join(sorted(value for value in group["series_tickers"] if value)),
            "representative_title": selected_components[0].get("title") if selected_components else "",
            "min_close_time": min(close_times).isoformat().replace("+00:00", "Z") if close_times else "",
            "max_close_time": max(close_times).isoformat().replace("+00:00", "Z") if close_times else "",
            "component_tickers": ",".join(market.get("ticker") or "" for market in selected_components),
        }
        group_rows.append(group_row)
        for component_rank, market in enumerate(selected_components, start=1):
            ticker = market["ticker"]
            markets_by_ticker[ticker] = market
            component_rows.append(
                {
                    "week_start": group_row["week_start"],
                    "week_end": group_row["week_end"],
                    "group_rank": rank,
                    "component_rank": component_rank,
                    "group_key": group["group_key"],
                    "group_kind": group["group_kind"],
                    "group_type_label": group_type_label,
                    "group_type_detail": group_type_detail,
                    "group_ticker": group["group_ticker"],
                    "ticker": ticker,
                    "weekly_volume": decimal_to_str(market_volume(market)),
                    "trade_count": market_trade_count(market),
                    "event_ticker": market.get("event_ticker"),
                    "series_ticker": base.infer_series_ticker(market),
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
    return group_rows, component_rows, markets_by_ticker


def bucket_markets_by_week(
    markets: list[dict[str, Any]],
    weeks: list[base.Window],
) -> dict[str, dict[str, dict[str, Any]]]:
    if not weeks:
        return {}
    start = weeks[0].start
    end = weeks[-1].end
    buckets: dict[str, dict[str, dict[str, Any]]] = {week.label: {} for week in weeks}
    labels = [week.label for week in weeks]
    for market in markets:
        ticker = market.get("ticker")
        if not ticker:
            continue
        close_time = base.parse_iso_ts(market.get("close_time"))
        if close_time is None or close_time < start or close_time >= end:
            continue
        idx = int((close_time - start).total_seconds() // (7 * 24 * 60 * 60))
        if 0 <= idx < len(labels):
            buckets[labels[idx]][ticker] = market
    return buckets


def load_topvol_selected_by_week(
    source_dir: Path,
    weeks: list[base.Window],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_week: dict[str, dict[str, dict[str, Any]]] = {week.label: {} for week in weeks}
    for week in weeks:
        path = source_dir / f"{week.label}_selected_markets.jsonl"
        if not path.exists():
            continue
        for market in load_jsonl(path):
            ticker = market.get("ticker")
            if ticker:
                by_week[week.label][ticker] = context_slim_market(market, "topvol_selected")
    return by_week


GROUP_FIELDS = [
    "week_start",
    "week_end",
    "rank",
    "group_key",
    "group_kind",
    "group_type_label",
    "group_type_detail",
    "group_ticker",
    "group_volume",
    "component_count",
    "selected_component_count",
    "component_set_complete",
    "event_tickers",
    "series_tickers",
    "representative_title",
    "min_close_time",
    "max_close_time",
    "component_tickers",
]

COMPONENT_FIELDS = [
    "week_start",
    "week_end",
    "group_rank",
    "component_rank",
    "group_key",
    "group_kind",
    "group_type_label",
    "group_type_detail",
    "group_ticker",
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


def write_week_outputs(
    out_dir: Path,
    week: base.Window,
    group_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    markets_by_ticker: dict[str, dict[str, Any]],
) -> None:
    write_csv(out_dir / "rankings" / f"{week.label}_top_groups.csv", group_rows, GROUP_FIELDS)
    write_csv(out_dir / "rankings" / f"{week.label}_component_markets.csv", component_rows, COMPONENT_FIELDS)

    with (out_dir / "markets" / f"{week.label}_context_groups.jsonl").open("w", encoding="utf-8") as handle:
        for row in group_rows:
            payload = dict(row)
            payload["component_tickers"] = [
                ticker for ticker in str(row.get("component_tickers") or "").split(",") if ticker
            ]
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    with (out_dir / "markets" / f"{week.label}_component_markets.jsonl").open("w", encoding="utf-8") as handle:
        for row in component_rows:
            market = markets_by_ticker.get(row["ticker"])
            if market:
                payload = dict(market)
                payload["_context_group_key"] = row["group_key"]
                payload["_context_group_rank"] = row["group_rank"]
                payload["_context_component_rank"] = row["component_rank"]
                handle.write(json.dumps(payload, sort_keys=True) + "\n")


def remove_unselected_candles(
    out_dir: Path,
    week: base.Window,
    *,
    period_interval: int,
    selected_tickers: set[str],
) -> int:
    week_dir = out_dir / "ohlcv" / f"period_{period_interval}m" / f"week={week.label}"
    if not week_dir.exists():
        return 0
    selected_names = {ticker.replace("/", "_") for ticker in selected_tickers}
    removed = 0
    for path in week_dir.glob("*.csv.gz"):
        ticker = path.name.removesuffix(".csv.gz")
        if ticker not in selected_names:
            path.unlink()
            removed += 1
    return removed


def download_one_candle(
    *,
    base_url: str,
    sleep_s: float,
    out_dir: Path,
    week: base.Window,
    row: dict[str, Any],
    market: dict[str, Any] | None,
    period_interval: int,
    cutoff: datetime,
    force: bool,
) -> dict[str, Any]:
    ticker = row["ticker"]
    path = base.candle_path(
        out_dir,
        period_interval=period_interval,
        week_label=week.label,
        ticker=ticker,
    )
    if path.exists() and path.stat().st_size > 40 and not force:
        return {"ticker": ticker, "status": "skipped", "rows": 0}
    if not market:
        return {
            "ticker": ticker,
            "status": "error",
            "rows": 0,
            "error": "metadata_missing",
        }
    try:
        worker_client = base.KalshiClient(base_url, sleep_s=sleep_s)
        candle_items = base.fetch_candles_with_fallback(
            worker_client,
            market,
            week,
            period_interval=period_interval,
            cutoff=cutoff,
        )
        flat_rows = [
            base.flatten_candle(candle, ticker=ticker, week=week, endpoint_source=endpoint_source)
            for endpoint_source, candle in candle_items
        ]
        base.write_candles(path, flat_rows)
        return {"ticker": ticker, "status": "downloaded", "rows": len(flat_rows)}
    except Exception as exc:  # noqa: BLE001 - collect-and-continue job.
        return {"ticker": ticker, "status": "error", "rows": 0, "error": repr(exc)}


def download_week_candles_parallel(
    *,
    base_url: str,
    sleep_s: float,
    out_dir: Path,
    week: base.Window,
    selected: list[dict[str, Any]],
    markets: dict[str, dict[str, Any]],
    period_interval: int,
    cutoff: datetime,
    force: bool,
    workers: int,
) -> dict[str, Any]:
    stats = {"downloaded": 0, "skipped": 0, "errors": 0, "rows": 0}
    total = len(selected)
    if total == 0:
        return stats

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(
                download_one_candle,
                base_url=base_url,
                sleep_s=sleep_s,
                out_dir=out_dir,
                week=week,
                row=row,
                market=markets.get(row["ticker"]),
                period_interval=period_interval,
                cutoff=cutoff,
                force=force,
            )
            for row in selected
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status = result["status"]
            if status == "downloaded":
                stats["downloaded"] += 1
                stats["rows"] += int(result.get("rows") or 0)
            elif status == "skipped":
                stats["skipped"] += 1
            else:
                stats["errors"] += 1
                base.append_error(
                    out_dir,
                    {
                        "week": week.label,
                        "ticker": result.get("ticker"),
                        "stage": "candles",
                        "error": result.get("error"),
                    },
                )
            if idx % 50 == 0 or idx == total:
                print(
                    f"  {week.label} candles {idx}/{total}: "
                    f"downloaded={stats['downloaded']} skipped={stats['skipped']} "
                    f"errors={stats['errors']} rows={stats['rows']}",
                    flush=True,
                )
    return stats


def update_combined_top_groups(out_dir: Path, weeks: list[base.Window]) -> None:
    combined: list[dict[str, Any]] = []
    for week in weeks:
        path = out_dir / "rankings" / f"{week.label}_top_groups.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                combined.extend(csv.DictReader(handle))
    if combined:
        write_csv(out_dir / "weekly_top_groups.csv", combined, GROUP_FIELDS)


def load_run_stats(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        row["week_start"]: row
        for row in data.get("weeks", [])
        if row.get("week_start")
    }


def write_target_context_links(out_dir: Path, target_csv: Path) -> None:
    if not target_csv.exists():
        return
    groups_by_week_event: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    combined_path = out_dir / "weekly_top_groups.csv"
    if not combined_path.exists():
        return
    with combined_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            week = row["week_start"][:10]
            for event_ticker in str(row.get("event_tickers") or "").split(","):
                if event_ticker:
                    groups_by_week_event[(week, event_ticker)].append(row)

    link_path = out_dir / "indexes" / "target_to_context_links.jsonl"
    tmp = link_path.with_suffix(link_path.suffix + ".tmp")
    count = 0
    with target_csv.open(newline="", encoding="utf-8") as targets, tmp.open("w", encoding="utf-8") as out:
        for target in csv.DictReader(targets):
            week = target["week_start"][:10]
            event_ticker = target.get("event_ticker") or ""
            for group in groups_by_week_event.get((week, event_ticker), []):
                payload = {
                    "week": week,
                    "relation": "same_event",
                    "target_ticker": target.get("ticker"),
                    "target_rank": target.get("rank"),
                    "event_ticker": event_ticker,
                    "context_group_key": group.get("group_key"),
                    "context_group_rank": group.get("rank"),
                    "context_group_type_label": group.get("group_type_label"),
                    "context_component_tickers": group.get("component_tickers"),
                }
                out.write(json.dumps(payload, sort_keys=True) + "\n")
                count += 1
    os.replace(tmp, link_path)
    print(f"Wrote {count} target/context links to {link_path}", flush=True)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    ensure_dirs(out_dir)

    start = base.parse_date(args.start_date)
    end = base.parse_date(args.end_date)
    weeks = base.weekly_windows(start, end)
    process_weeks = weeks
    if args.resume_from_week:
        process_weeks = [week for week in weeks if week.label >= args.resume_from_week]
        if not process_weeks:
            raise ValueError(f"resume week {args.resume_from_week} is outside the selected date range")
    client = base.KalshiClient(args.base_url, sleep_s=args.sleep)

    cutoff = client.get("/historical/cutoff")
    market_cutoff = base.parse_iso_ts(cutoff.get("market_settled_ts")) or datetime(1970, 1, 1, tzinfo=UTC)

    if args.force_rebuild_cache or not args.historical_cache.exists():
        print("Building/loading context historical cache...", flush=True)
        historical_cache = build_context_historical_cache(
            client,
            out_dir,
            overall_start=weeks[0].start,
            overall_end=weeks[-1].end,
            limit=args.limit,
        )
        historical_cache_source = "fresh_context_cache"
    else:
        print(f"Loading existing historical cache: {args.historical_cache}", flush=True)
        historical_cache = load_jsonl(args.historical_cache)
        historical_cache_source = str(args.historical_cache)
    print(f"Historical cache contains {len(historical_cache)} markets", flush=True)
    historical_by_week = bucket_markets_by_week(historical_cache, weeks)
    del historical_cache

    manifest = {
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "start_date": args.start_date,
        "end_date_inclusive": args.end_date,
        "resume_from_week": args.resume_from_week,
        "processed_week_count": len(process_weeks),
        "top_groups_per_week": args.top_groups_per_week,
        "max_markets_per_group": args.max_markets_per_group,
        "component_sets": "complete" if args.max_markets_per_group <= 0 else "truncated",
        "min_markets_per_group": args.min_markets_per_group,
        "period_interval_minutes": args.period_interval,
        "download_workers": args.download_workers,
        "base_url": args.base_url,
        "historical_cutoff": cutoff,
        "historical_cache_source": historical_cache_source,
        "topvol_selected_dir": str(args.topvol_selected_dir),
        "include_live_supplement": args.include_live_supplement,
        "ranking": "Event/MVE groups ranked by summed component volume_fp for markets closing in each week.",
        "notes": [
            "Kalshi markets are primarily binary; this dataset groups related component markets as context.",
            "Default run reuses the prior historical cache to stay under the requested runtime budget.",
            "OOS prompt builders must filter context candles to timestamps at or before the target as_of.",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    topvol_by_week = load_topvol_selected_by_week(args.topvol_selected_dir, weeks)
    for week_label, markets in topvol_by_week.items():
        historical_by_week.setdefault(week_label, {}).update(markets)

    state_path = out_dir / "state" / "run_state.json"
    run_stats_by_week = load_run_stats(state_path)
    for idx, week in enumerate(process_weeks, start=1):
        print(f"[{idx}/{len(process_weeks)}] {week.label}: selecting context groups", flush=True)
        live_markets = (
            fetch_live_context_markets(client, week, limit=args.limit)
            if args.include_live_supplement
            else []
        )
        markets_by_ticker = {
            market["ticker"]: market
            for market in list(historical_by_week.get(week.label, {}).values()) + live_markets
            if market.get("ticker")
        }
        group_rows, component_rows, selected_markets = selected_context_groups(
            week,
            list(markets_by_ticker.values()),
            top_groups=args.top_groups_per_week,
            max_markets_per_group=args.max_markets_per_group,
            min_markets_per_group=args.min_markets_per_group,
        )
        missing_full = [ticker for ticker in selected_markets if not selected_markets[ticker].get("rules_primary")]
        if missing_full:
            selected_markets.update(fetch_context_market_metadata(client, missing_full, limit=args.limit))
        write_week_outputs(out_dir, week, group_rows, component_rows, selected_markets)

        candle_stats = {"downloaded": 0, "skipped": 0, "errors": 0, "rows": 0}
        if not args.rank_only:
            removed_stale = remove_unselected_candles(
                out_dir,
                week,
                period_interval=args.period_interval,
                selected_tickers={row["ticker"] for row in component_rows if row.get("ticker")},
            )
            if removed_stale:
                print(f"[{week.label}] removed {removed_stale} stale candle files", flush=True)
            candle_stats = download_week_candles_parallel(
                base_url=args.base_url,
                sleep_s=args.sleep,
                out_dir=out_dir,
                week=week,
                selected=component_rows,
                markets=selected_markets,
                period_interval=args.period_interval,
                cutoff=market_cutoff,
                force=args.force,
                workers=args.download_workers,
            )
            print(f"[{week.label}] candles {candle_stats}", flush=True)

        run_stats_by_week[week.label] = {
            "week_start": week.label,
            "week_end": week.end.date().isoformat(),
            "groups": len(group_rows),
            "component_markets": len(component_rows),
            "live_supplement_markets": len(live_markets),
            "candles": candle_stats,
        }
        ordered_stats = [
            run_stats_by_week[window.label]
            for window in weeks
            if window.label in run_stats_by_week
        ]
        write_json(state_path, {"weeks": ordered_stats})

    update_combined_top_groups(out_dir, weeks)
    write_target_context_links(out_dir, KALSHI_TOPVOL / "weekly_top_markets.csv")
    print(f"Done. Outputs are in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
