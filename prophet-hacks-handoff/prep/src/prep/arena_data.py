"""Event-level loaders for Prophet Arena style backtests."""

from __future__ import annotations

import json
from ast import literal_eval
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PREP_ROOT = Path(__file__).resolve().parents[2]
SUBSET_1200_CSV = PREP_ROOT / "data" / "external" / "subset_1200.csv"
REPO_ROOT = PREP_ROOT.parents[1]
KALSHI_TOPVOL_CSV = REPO_ROOT / "Kalshitopvolmarkets" / "weekly_top_markets.csv"
KALSHI_TOPVOL_OHLCV = REPO_ROOT / "Kalshitopvolmarkets" / "ohlcv" / "period_1m"
NONBINARY_LINKS = REPO_ROOT / "NonBinaryMarkets" / "indexes" / "target_to_context_links.jsonl"
KALSHI_TOPVOL_MARKETS = REPO_ROOT / "Kalshitopvolmarkets" / "markets"


@dataclass
class ArenaBacktestEvent:
    event: dict[str, Any]
    actuals: dict[str, int]
    market_data: dict[str, Any]
    sources: list[dict[str, Any]]
    submission_id: str
    snapshot_time: str | None

    @property
    def outcomes(self) -> list[str]:
        return list(self.event.get("outcomes") or [])

    @property
    def is_binary(self) -> bool:
        return len(self.outcomes) == 2

    @property
    def is_exclusive(self) -> bool:
        return sum(1 for value in self.actuals.values() if bool(value)) == 1


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = literal_eval(text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = literal_eval(text)
        except Exception:
            return []
    return parsed if isinstance(parsed, list) else []


def _market_mid(market: dict[str, Any]) -> float | None:
    try:
        yes_ask = market.get("yes_ask")
        no_ask = market.get("no_ask")
        if yes_ask is not None and no_ask is not None:
            ya = float(yes_ask) / 100.0 if float(yes_ask) > 1 else float(yes_ask)
            na = float(no_ask) / 100.0 if float(no_ask) > 1 else float(no_ask)
            return max(0.0, min(1.0, (ya + (1.0 - na)) / 2.0))
    except (TypeError, ValueError):
        return None
    return None


def load_subset_1200_events(
    csv_path: Path = SUBSET_1200_CSV,
    *,
    include_binary: bool = True,
    include_nonbinary: bool = True,
    max_outcomes: int | None = None,
) -> list[ArenaBacktestEvent]:
    """Load subset_1200 as one event per row.

    This preserves multi-outcome/nonbinary rows instead of flattening them into
    separate binary markets.
    """
    df = pd.read_csv(csv_path).sort_values("snapshot_time").reset_index(drop=True)
    events: list[ArenaBacktestEvent] = []
    for _, row in df.iterrows():
        market_data = _parse_mapping(row.get("market_data"))
        actuals_raw = _parse_mapping(row.get("market_outcome"))
        markets = [str(m) for m in _parse_list(row.get("markets"))]
        if not markets:
            markets = [str(m) for m in actuals_raw.keys()]
        if not markets:
            markets = [str(m) for m in market_data.keys()]
        if not markets:
            continue
        if max_outcomes is not None and len(markets) > max_outcomes:
            continue
        is_binary = len(markets) == 2
        if is_binary and not include_binary:
            continue
        if not is_binary and not include_nonbinary:
            continue

        actuals = {outcome: int(bool(actuals_raw.get(outcome, 0))) for outcome in markets}
        sources = _parse_list(row.get("sources"))
        mids = {
            outcome: mid
            for outcome in markets
            if (mid := _market_mid(market_data.get(outcome) or {})) is not None
        }
        event_ticker = str(row.get("event_ticker") or row.get("submission_id") or "")
        event = {
            "event_ticker": event_ticker,
            "market_ticker": event_ticker,
            "task_id": event_ticker,
            "title": row.get("augmented_title") or row.get("title") or "",
            "subtitle": None,
            "description": row.get("title") or None,
            "category": row.get("category") or "Other",
            "rules": row.get("rules") or None,
            "close_time": row.get("close_time") or None,
            "predict_by": row.get("snapshot_time") or None,
            "snapshot_time": row.get("snapshot_time") or None,
            "outcomes": markets,
            "resolved_outcome": actuals,
            "retrieval": {
                "sources": sources,
                "market_data": market_data,
                "market_implied_probabilities": mids,
                "actuals_are_multilabel": sum(actuals.values()) != 1,
            },
        }
        events.append(ArenaBacktestEvent(
            event=event,
            actuals=actuals,
            market_data=market_data,
            sources=sources,
            submission_id=str(row.get("submission_id") or ""),
            snapshot_time=row.get("snapshot_time") or None,
        ))
    return events


def _parse_dt(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_horizon(value: str) -> timedelta:
    """Parse horizons like 7d, 1d, 6h, 30m."""
    text = value.strip().lower()
    if not text:
        raise ValueError("horizon cannot be empty")
    unit = text[-1]
    try:
        amount = float(text[:-1])
    except ValueError as exc:
        raise ValueError(f"Invalid horizon: {value!r}") from exc
    if unit == "d":
        return timedelta(days=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "m":
        return timedelta(minutes=amount)
    raise ValueError("horizon must end with d, h, or m")


def _load_context_links() -> dict[str, list[dict[str, Any]]]:
    links: dict[str, list[dict[str, Any]]] = {}
    if not NONBINARY_LINKS.exists():
        return links
    for line in NONBINARY_LINKS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        ticker = row.get("target_ticker")
        if ticker:
            links.setdefault(str(ticker), []).append(row)
    return links


def _latest_candle_before(ticker: str, week: str, as_of: datetime) -> dict[str, Any] | None:
    path = KALSHI_TOPVOL_OHLCV / f"week={week}" / f"{ticker}.csv.gz"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, compression="gzip")
    except Exception:
        return None
    if df.empty or "end_period_time" not in df:
        return None
    times = pd.to_datetime(
        df["end_period_time"],
        utc=True,
        errors="coerce",
        format="%Y-%m-%dT%H:%M:%SZ",
    )
    eligible = df[times <= pd.Timestamp(as_of)]
    if eligible.empty:
        return None
    row = eligible.iloc[-1].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def _candle_mid(candle: dict[str, Any]) -> float | None:
    for key in ("price_close", "price_previous", "price_mean"):
        value = candle.get(key)
        if value is not None:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                pass
    try:
        bid = candle.get("yes_bid_close")
        ask = candle.get("yes_ask_close")
        if bid is not None and ask is not None:
            return max(0.0, min(1.0, (float(bid) + float(ask)) / 2.0))
    except (TypeError, ValueError):
        return None
    return None


def _selected_market_meta(
    week: str,
    ticker: str,
    cache: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if week not in cache:
        path = KALSHI_TOPVOL_MARKETS / f"{week}_selected_markets.jsonl"
        rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("ticker"):
                    rows[str(row["ticker"])] = row
        cache[week] = rows
    return cache[week].get(ticker, {})


def _binary_actual(result: Any) -> dict[str, int]:
    yes = str(result or "").strip().lower() == "yes"
    return {"YES": int(yes), "NO": int(not yes)}


def load_kalshi_topvol_horizon_events(
    *,
    horizon: str,
    csv_path: Path = KALSHI_TOPVOL_CSV,
    max_rank: int | None = 300,
) -> list[ArenaBacktestEvent]:
    """Load local top-volume Kalshi binary targets at a fixed pre-close horizon.

    The event packet includes the most recent 1m candle at or before
    close_time - horizon, plus linked nonbinary/context-group metadata when
    available.
    """
    if not csv_path.exists():
        return []
    horizon_delta = parse_horizon(horizon)
    df = pd.read_csv(csv_path)
    links = _load_context_links()
    selected_cache: dict[str, dict[str, dict[str, Any]]] = {}
    events: list[ArenaBacktestEvent] = []
    for _, row in df.iterrows():
        try:
            rank = int(row.get("rank"))
        except (TypeError, ValueError):
            rank = 0
        if max_rank is not None and rank > max_rank:
            continue
        ticker = str(row.get("ticker") or "")
        week = str(row.get("week_start") or "")[:10]
        close_dt = _parse_dt(row.get("close_time"))
        open_dt = _parse_dt(row.get("open_time"))
        if not ticker or not week or close_dt is None:
            continue
        as_of = close_dt - horizon_delta
        if open_dt is not None and as_of < open_dt:
            continue
        candle = _latest_candle_before(ticker, week, as_of)
        if not candle:
            continue
        meta = _selected_market_meta(week, ticker, selected_cache)
        mid = _candle_mid(candle)
        context_links = links.get(ticker, [])[:3]
        market_data = {
            "ticker": ticker,
            "yes_mid": mid,
            "latest_pre_horizon_candle": candle,
            "weekly_volume": row.get("weekly_volume"),
            "rank": rank,
            "selected_market_metadata": meta,
        }
        actuals = _binary_actual(row.get("result"))
        event = {
            "event_ticker": row.get("event_ticker") or ticker,
            "market_ticker": ticker,
            "task_id": ticker,
            "title": meta.get("title") or row.get("title") or "",
            "subtitle": meta.get("subtitle") or meta.get("yes_sub_title") or row.get("subtitle") or None,
            "description": row.get("title") or None,
            "category": None,
            "rules": meta.get("rules_primary") or row.get("rules_primary") or None,
            "close_time": close_dt.isoformat().replace("+00:00", "Z"),
            "predict_by": as_of.isoformat().replace("+00:00", "Z"),
            "snapshot_time": as_of.isoformat().replace("+00:00", "Z"),
            "outcomes": ["YES", "NO"],
            "resolved_outcome": actuals,
            "retrieval": {
                "source": "Kalshitopvolmarkets",
                "sampling_horizon": horizon,
                "source_week": week,
                "market_data": market_data,
                "market_implied_probabilities": {"YES": mid, "NO": None if mid is None else 1.0 - mid},
                "context_links": context_links,
                "source_cutoff": as_of.isoformat().replace("+00:00", "Z"),
            },
        }
        events.append(ArenaBacktestEvent(
            event=event,
            actuals=actuals,
            market_data=market_data,
            sources=[],
            submission_id=f"{ticker}:{horizon}",
            snapshot_time=event["snapshot_time"],
        ))
    return events
