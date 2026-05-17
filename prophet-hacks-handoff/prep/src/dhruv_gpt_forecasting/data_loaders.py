"""Backtest data loaders, including Kalshitopvolmarkets."""

from __future__ import annotations

import csv
import gzip
import json
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .features import normalize_category, parse_dt, price_to_prob


GIT_LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"
REPO_ROOT = Path(__file__).resolve().parents[4]
HANDOFF_ROOT = REPO_ROOT / "prophet-hacks-handoff"
PREP_DATA = HANDOFF_ROOT / "prep" / "data"
TOPVOL_ROOT = REPO_ROOT / "Kalshitopvolmarkets"
NONBINARY_ROOT = REPO_ROOT / "NonBinaryMarkets"
PROPHET_SUBSET_1200 = PREP_DATA / "external" / "subset_1200.csv"


@dataclass
class BacktestSample:
    event: dict[str, Any]
    market_info: dict[str, Any]
    snapshots: list[dict[str, Any]]
    outcome: int


def load_eval_pack(path: Path | None = None, limit: int | None = None) -> list[BacktestSample]:
    fp = path or PREP_DATA / "eval_pack_live_clean.jsonl"
    samples: list[BacktestSample] = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        snapshots = row.get("snapshots") or []
        market_info = dict(snapshots[-1]) if snapshots else {}
        market_info["snapshots"] = snapshots
        samples.append(BacktestSample(
            event=dict(row["event"]),
            market_info=market_info,
            snapshots=snapshots,
            outcome=int(row["outcome"]),
        ))
        if limit and len(samples) >= limit:
            break
    return samples


def load_topvol_samples(
    root: Path | None = None,
    *,
    limit: int | None = None,
    candle_stride_minutes: int = 15,
    min_snapshots: int = 2,
) -> list[BacktestSample]:
    root = root or TOPVOL_ROOT
    samples: list[BacktestSample] = []
    market_files = sorted((root / "markets").glob("*_selected_markets.jsonl"))
    for market_file in market_files:
        week = market_file.name.split("_", 1)[0]
        for market in _iter_jsonl(market_file):
            result = str(market.get("result") or "").lower()
            if result not in {"yes", "no"}:
                continue
            ticker = market.get("ticker")
            if not ticker:
                continue
            snapshots = load_topvol_candles(root, week, ticker, market.get("close_time"), candle_stride_minutes)
            if len(snapshots) < min_snapshots:
                continue
            event = {
                "event_ticker": market.get("event_ticker") or "",
                "market_ticker": ticker,
                "title": market.get("title") or "",
                "subtitle": market.get("subtitle") or market.get("yes_sub_title"),
                "description": None,
                "category": normalize_category(None, market.get("event_ticker")),
                "rules": market.get("rules_primary"),
                "close_time": market.get("close_time"),
                "outcomes": ["YES", "NO"],
            }
            market_info = dict(market)
            market_info.update(snapshots[-1])
            market_info["snapshots"] = snapshots
            samples.append(BacktestSample(
                event=event,
                market_info=market_info,
                snapshots=snapshots,
                outcome=1 if result == "yes" else 0,
            ))
            if limit and len(samples) >= limit:
                return samples
    return samples


def load_nonbinary_component_samples(
    root: Path | None = None,
    *,
    limit: int | None = None,
    candle_stride_minutes: int = 15,
    min_snapshots: int = 2,
) -> list[BacktestSample]:
    """Load resolved component markets from NonBinaryMarkets as binary samples."""
    root = root or NONBINARY_ROOT
    samples: list[BacktestSample] = []
    market_files = sorted((root / "markets").glob("*_component_markets.jsonl"))
    for market_file in market_files:
        week = market_file.name.split("_", 1)[0]
        for market in _iter_jsonl(market_file):
            result = str(market.get("result") or "").lower()
            if result not in {"yes", "no"}:
                continue
            ticker = market.get("ticker")
            if not ticker:
                continue
            snapshots = load_topvol_candles(root, week, ticker, market.get("close_time"), candle_stride_minutes)
            if len(snapshots) < min_snapshots:
                continue
            label = market.get("yes_sub_title") or market.get("subtitle")
            event = {
                "event_ticker": market.get("event_ticker") or "",
                "market_ticker": ticker,
                "title": market.get("title") or "",
                "subtitle": label,
                "description": None,
                "category": normalize_category(market.get("category"), market.get("event_ticker")),
                "rules": market.get("rules_primary"),
                "close_time": market.get("close_time"),
                "outcomes": ["YES", "NO"],
            }
            market_info = dict(market)
            market_info.update(snapshots[-1])
            market_info["snapshots"] = snapshots
            samples.append(BacktestSample(
                event=event,
                market_info=market_info,
                snapshots=snapshots,
                outcome=1 if result == "yes" else 0,
            ))
            if limit and len(samples) >= limit:
                return samples
    return samples


def load_unified_binary_samples(
    *,
    limit: int | None = None,
    candle_stride_minutes: int = 15,
    min_snapshots: int = 2,
) -> list[BacktestSample]:
    """Load all binary-style samples we can score from top-volume and component data."""
    per_source_limit = limit if limit is not None else None
    samples = [
        *load_topvol_samples(
            limit=per_source_limit,
            candle_stride_minutes=candle_stride_minutes,
            min_snapshots=min_snapshots,
        ),
        *load_nonbinary_component_samples(
            limit=per_source_limit,
            candle_stride_minutes=candle_stride_minutes,
            min_snapshots=min_snapshots,
        ),
    ]
    samples.sort(key=lambda sample: (
        _sample_close_sort_key(sample),
        sample.event.get("market_ticker") or sample.market_info.get("ticker") or "",
    ))
    return samples[:limit] if limit is not None else samples


def load_prophet_subset_events(path: Path | None = None, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load the teammate/official subset_1200 rows as Arena-style event packets.

    The CSV is multi-outcome at the row level and carries curated source
    snippets. We preserve those snippets as timestamped evidence using the row's
    snapshot_time as the collection time.
    """
    fp = path or PROPHET_SUBSET_1200
    events: list[dict[str, Any]] = []
    with fp.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            outcomes = _parse_jsonish(row.get("markets")) or []
            if isinstance(outcomes, dict):
                outcomes = list(outcomes)
            outcomes = [str(item) for item in outcomes if str(item)]
            if not outcomes:
                market_data = _parse_jsonish(row.get("market_data")) or {}
                outcomes = [str(item) for item in market_data] if isinstance(market_data, dict) else []
            event = {
                "event_ticker": row.get("event_ticker") or row.get("submission_id") or "",
                "market_ticker": row.get("submission_id") or row.get("event_ticker") or "",
                "title": row.get("augmented_title") or row.get("title") or "",
                "subtitle": row.get("title"),
                "description": row.get("augmented_title") or row.get("title"),
                "category": normalize_category(row.get("category"), row.get("event_ticker")),
                "rules": row.get("rules"),
                "close_time": row.get("close_time"),
                "outcomes": outcomes or ["YES", "NO"],
                "as_of": row.get("snapshot_time"),
                "snapshot_time": row.get("snapshot_time"),
                "resolved_outcome": _resolved_subset_outcome(row.get("market_outcome")),
                "features": {
                    "source_dataset": "prophet_subset_1200",
                    "market_data": _parse_jsonish(row.get("market_data")),
                    "market_outcome": _parse_jsonish(row.get("market_outcome")),
                    "curated_sources": _normalize_subset_sources(
                        _parse_jsonish(row.get("sources")),
                        row.get("snapshot_time"),
                        row.get("event_ticker") or row.get("submission_id") or "",
                        row.get("submission_id") or row.get("event_ticker") or "",
                    ),
                },
            }
            events.append(event)
            if limit and len(events) >= limit:
                return events
    return events


def load_topvol_candles(
    root: Path,
    week: str,
    ticker: str,
    close_time: str | None,
    stride_minutes: int,
) -> list[dict[str, Any]]:
    fp = root / "ohlcv" / "period_1m" / f"week={week}" / f"{ticker}.csv.gz"
    if not fp.exists() or is_git_lfs_pointer_file(fp):
        return []
    close_dt = parse_dt(close_time)
    out: list[dict[str, Any]] = []
    last_bucket: int | None = None
    with gzip.open(fp, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_dt = parse_dt(row.get("end_period_time"))
            if close_dt and row_dt and row_dt > close_dt:
                continue
            ts_raw = row.get("end_period_ts")
            bucket = int(int(ts_raw) // (stride_minutes * 60)) if ts_raw else None
            if bucket is not None and bucket == last_bucket:
                if out:
                    out[-1] = snapshot_from_candle(row)
                continue
            out.append(snapshot_from_candle(row))
            last_bucket = bucket
    return out


def snapshot_from_candle(row: dict[str, Any]) -> dict[str, Any]:
    yes_bid = price_to_prob(row.get("yes_bid_close"))
    yes_ask = price_to_prob(row.get("yes_ask_close"))
    return {
        "t": row.get("end_period_time"),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_ask": None if yes_bid is None else 1.0 - yes_bid,
        "last_price": price_to_prob(row.get("price_close")),
        "volume": _float_or_zero(row.get("volume")),
        "open_interest": _float_or_zero(row.get("open_interest")),
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    yield from iter_jsonl_rows(path)


def iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if is_git_lfs_pointer_text(text):
        return
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc.msg}") from exc


def is_git_lfs_pointer_text(text: str) -> bool:
    return text.startswith(GIT_LFS_POINTER_PREFIX)


def is_git_lfs_pointer_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(len(GIT_LFS_POINTER_PREFIX)).decode("utf-8", errors="ignore") == GIT_LFS_POINTER_PREFIX
    except OSError:
        return False


def _float_or_zero(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _sample_close_sort_key(sample: BacktestSample) -> float:
    close_dt = parse_dt(sample.event.get("close_time") or sample.market_info.get("close_time"))
    return close_dt.timestamp() if close_dt is not None else 0.0


def _parse_jsonish(value: Any) -> Any:
    if value in (None, "", "nan"):
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return literal_eval(value)
        except Exception:
            return None


def _resolved_subset_outcome(value: Any) -> str | None:
    outcome = _parse_jsonish(value)
    if not isinstance(outcome, dict):
        return None
    winners = [str(label) for label, resolved in outcome.items() if str(resolved) in {"1", "True", "true"} or resolved is True]
    return winners[0] if len(winners) == 1 else None


def _normalize_subset_sources(
    sources: Any,
    snapshot_time: str | None,
    event_ticker: str,
    market_ticker: str,
) -> list[dict[str, Any]]:
    if not isinstance(sources, list):
        return []
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(sources):
        if not isinstance(item, dict):
            continue
        rows.append({
            "source": "prophet_subset_source",
            "published_at": item.get("published_at") or item.get("date") or snapshot_time,
            "collected_at": snapshot_time,
            "title": item.get("title"),
            "text": item.get("summary") or item.get("text"),
            "summary": item.get("summary"),
            "url": item.get("url"),
            "ranking": item.get("ranking"),
            "source_id": str(item.get("source_id") or idx),
            "event_ticker": event_ticker,
            "market_ticker": market_ticker,
            "timestamp_basis": "prophet_subset_snapshot_time" if not item.get("published_at") else "source_published_at",
        })
    return rows
