"""Related-market context for binary Kalshi forecasts.

The context folders are useful side-channel evidence, but some rows are
post-settlement metadata. This module only emits static market labels and
pre-as_of candle summaries into prompts.
"""

from __future__ import annotations

import csv
import gzip
from functools import lru_cache
from pathlib import Path
from typing import Any

from .data_loaders import PREP_DATA, REPO_ROOT, TOPVOL_ROOT, is_git_lfs_pointer_file, iter_jsonl_rows, snapshot_from_candle
from .features import parse_dt, quote_from_market_info
from .market_linker import infer_linked_market_distribution
from .schemas import FeaturePacket


NONBINARY_ROOT = REPO_ROOT / "NonBinaryMarkets"
POLYMARKET_ROOT = PREP_DATA / "kalshi_polymarket"

STATIC_MARKET_KEYS = (
    "ticker",
    "event_ticker",
    "title",
    "subtitle",
    "yes_sub_title",
    "no_sub_title",
    "market_type",
    "open_time",
    "close_time",
    "rules_primary",
)


def build_related_context_evidence(
    packet: FeaturePacket,
    *,
    max_groups: int = 2,
    max_components: int = 8,
    max_poly_matches: int = 3,
) -> list[dict[str, Any]]:
    """Build compact OOS-safe related-market evidence for a target market."""
    target_ticker = packet.market_ticker
    if not target_ticker and not packet.event_ticker:
        return []

    evidence: list[dict[str, Any]] = []

    nonbinary = _nonbinary_evidence(packet, max_groups=max_groups, max_components=max_components)
    evidence.extend(nonbinary)

    if not nonbinary:
        topvol = _topvol_same_event_evidence(packet, max_components=max_components)
        if topvol:
            evidence.append(topvol)

    polymarket = _polymarket_evidence(packet, max_matches=max_poly_matches)
    if polymarket:
        evidence.append(polymarket)

    linked = infer_linked_market_distribution(packet, evidence)
    if linked:
        evidence.insert(0, linked.to_evidence())

    return evidence


def _nonbinary_evidence(
    packet: FeaturePacket,
    *,
    max_groups: int,
    max_components: int,
) -> list[dict[str, Any]]:
    root = NONBINARY_ROOT
    links = _links_by_target(str(root)).get(packet.market_ticker, [])
    if not links and packet.event_ticker:
        links = _links_by_event(str(root)).get(packet.event_ticker, [])
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for link in links:
        week = str(link.get("week") or "")
        group_key = str(link.get("context_group_key") or "")
        if not week or not group_key or (week, group_key) in seen:
            continue
        seen.add((week, group_key))
        components = _components_for_group(str(root), week, group_key)
        if not components:
            continue
        component_summaries = [
            _component_summary(
                component,
                root=root,
                period_dir="period_1m",
                week=week,
                as_of=packet.as_of,
                stride_minutes=15,
            )
            for component in components[:max_components]
        ]
        event = {
            "source": "kalshi_nonbinary_context",
            "relation": str(link.get("relation") or "same_event"),
            "group_key": group_key,
            "event_ticker": str(link.get("event_ticker") or packet.event_ticker),
            "target_ticker": packet.market_ticker,
            "components": component_summaries,
            "derived": _component_derived(packet.market_ticker, component_summaries),
        }
        out.append(event)
        if len(out) >= max_groups:
            break
    return out


def _topvol_same_event_evidence(
    packet: FeaturePacket,
    *,
    max_components: int,
) -> dict[str, Any] | None:
    if not packet.event_ticker:
        return None
    rows = _topvol_by_event(str(TOPVOL_ROOT)).get(packet.event_ticker, [])
    if not rows:
        return None
    component_summaries = [
        _component_summary(
            row,
            root=TOPVOL_ROOT,
            period_dir="period_1m",
            week=week,
            as_of=packet.as_of,
            stride_minutes=15,
        )
        for week, row in rows[:max_components]
    ]
    return {
        "source": "kalshi_topvol_same_event",
        "relation": "same_event",
        "event_ticker": packet.event_ticker,
        "target_ticker": packet.market_ticker,
        "components": component_summaries,
        "derived": _component_derived(packet.market_ticker, component_summaries),
    }


def _polymarket_evidence(packet: FeaturePacket, *, max_matches: int) -> dict[str, Any] | None:
    if not packet.market_ticker:
        return None
    matches = _poly_map_by_kalshi(str(POLYMARKET_ROOT)).get(packet.market_ticker, [])
    if matches:
        return {
            "source": "kalshi_polymarket_map",
            "relation": "cross_venue_question_match",
            "kalshi_ticker": packet.market_ticker,
            "matches": [
                {
                    "poly_condition_id": row.get("poly_condition_id"),
                    "poly_outcome": row.get("poly_outcome"),
                    "kalshi_question": row.get("kalshi_question"),
                    "poly_question": row.get("poly_question"),
                    "poly_end_date": row.get("poly_end_date"),
                    "poly_vol_24h": _float_or_none(row.get("poly_vol_24h")),
                }
                for row in matches[:max_matches]
            ],
        }
    rejected = _poly_rejections_by_kalshi(str(POLYMARKET_ROOT)).get(packet.market_ticker)
    if rejected:
        return {
            "source": "kalshi_polymarket_map_gap",
            "relation": "no_reliable_cross_venue_match",
            "kalshi_ticker": packet.market_ticker,
            "query": rejected.get("query"),
            "candidate_count": _int_or_none(rejected.get("n_candidates")),
        }
    return None


def _component_summary(
    component: dict[str, Any],
    *,
    root: Path,
    period_dir: str,
    week: str,
    as_of: str | None,
    stride_minutes: int,
) -> dict[str, Any]:
    ticker = str(component.get("ticker") or "")
    summary = {
        key: component.get(key)
        for key in STATIC_MARKET_KEYS
        if component.get(key) not in (None, "")
    }
    snapshots = _load_candles_until(
        root / "ohlcv" / period_dir / f"week={week}" / f"{ticker}.csv.gz",
        as_of=as_of,
        stride_minutes=stride_minutes,
    )
    if snapshots:
        quote = quote_from_market_info(snapshots[-1])
        summary["pre_as_of_quote"] = {
            "market_mid": quote.market_mid,
            "spread": quote.spread,
            "yes_bid": quote.yes_bid,
            "yes_ask": quote.yes_ask,
            "no_ask": quote.no_ask,
            "last_price": quote.last_price,
            "volume": quote.volume,
            "open_interest": quote.open_interest,
            "snapshot_time": quote.snapshot_time,
        }
        summary["n_pre_as_of_snapshots"] = len(snapshots)
        if len(snapshots) >= 2:
            first = quote_from_market_info(snapshots[0]).market_mid
            summary["market_mid_momentum"] = quote.market_mid - first
        if len(snapshots) >= 5:
            recent = quote_from_market_info(snapshots[-5]).market_mid
            summary["recent_market_mid_momentum"] = quote.market_mid - recent
            mids = [quote_from_market_info(snap).market_mid for snap in snapshots[-20:]]
            if len(mids) >= 2:
                diffs = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
                summary["recent_market_mid_volatility"] = (
                    sum(diff * diff for diff in diffs) / len(diffs)
                ) ** 0.5
    else:
        summary["n_pre_as_of_snapshots"] = 0
    return summary


def _component_derived(target_ticker: str, components: list[dict[str, Any]]) -> dict[str, Any]:
    priced = [
        (component, component.get("pre_as_of_quote", {}).get("market_mid"))
        for component in components
        if component.get("pre_as_of_quote", {}).get("market_mid") is not None
    ]
    target_mid = None
    for component, mid in priced:
        if component.get("ticker") == target_ticker:
            target_mid = mid
            break
    top_component = None
    if priced:
        component, mid = max(priced, key=lambda item: item[1])
        top_component = {
            "ticker": component.get("ticker"),
            "yes_sub_title": component.get("yes_sub_title"),
            "market_mid": mid,
        }
    sum_mid = sum(float(mid) for _, mid in priced) if priced else None
    normalized: list[dict[str, Any]] = []
    target_rank = None
    target_normalized = None
    favorite_gap = None
    if sum_mid and sum_mid > 0.0:
        normalized = sorted(
            [
                {
                    "ticker": component.get("ticker"),
                    "yes_sub_title": component.get("yes_sub_title"),
                    "normalized_probability": float(mid) / sum_mid,
                    "market_mid": mid,
                }
                for component, mid in priced
            ],
            key=lambda row: row["normalized_probability"],
            reverse=True,
        )
        for idx, row in enumerate(normalized, start=1):
            if row.get("ticker") == target_ticker:
                target_rank = idx
                target_normalized = row["normalized_probability"]
                favorite_gap = normalized[0]["normalized_probability"] - row["normalized_probability"]
                break
    entropy = None
    if normalized:
        import math

        values = [row["normalized_probability"] for row in normalized if row["normalized_probability"] > 0.0]
        if len(values) > 1:
            entropy = -sum(value * math.log(value) for value in values) / math.log(len(values))
    return {
        "component_count": len(components),
        "priced_component_count": len(priced),
        "sum_yes_market_mid": sum_mid,
        "target_yes_market_mid": target_mid,
        "target_normalized_probability": target_normalized,
        "target_rank_by_normalized_probability": target_rank,
        "target_gap_to_favorite_probability": favorite_gap,
        "normalized_probability_entropy": entropy,
        "normalized_distribution_top": normalized[:5],
        "top_component": top_component,
    }


def _load_candles_until(
    path: Path,
    *,
    as_of: str | None,
    stride_minutes: int,
) -> list[dict[str, Any]]:
    as_of_dt = parse_dt(as_of)
    if as_of_dt is None or not path.exists() or is_git_lfs_pointer_file(path):
        return []
    out: list[dict[str, Any]] = []
    last_bucket: int | None = None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_dt = parse_dt(row.get("end_period_time"))
            if row_dt and row_dt > as_of_dt:
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


@lru_cache(maxsize=4)
def _links_by_target(root_str: str) -> dict[str, list[dict[str, Any]]]:
    links: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl_rows(Path(root_str) / "indexes" / "target_to_context_links.jsonl"):
        ticker = str(row.get("target_ticker") or "")
        if ticker:
            links.setdefault(ticker, []).append(row)
    return links


@lru_cache(maxsize=4)
def _links_by_event(root_str: str) -> dict[str, list[dict[str, Any]]]:
    links: dict[str, list[dict[str, Any]]] = {}
    for rows in _links_by_target(root_str).values():
        for row in rows:
            event_ticker = str(row.get("event_ticker") or "")
            if event_ticker:
                links.setdefault(event_ticker, []).append(row)
    return links


@lru_cache(maxsize=128)
def _components_for_group(root_str: str, week: str, group_key: str) -> list[dict[str, Any]]:
    fp = Path(root_str) / "markets" / f"{week}_component_markets.jsonl"
    rows = [
        row
        for row in iter_jsonl_rows(fp)
        if str(row.get("_context_group_key") or "") == group_key
    ]
    return sorted(rows, key=lambda row: _int_or_none(row.get("_context_component_rank")) or 999)


@lru_cache(maxsize=4)
def _topvol_by_event(root_str: str) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    root = Path(root_str)
    out: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for fp in sorted((root / "markets").glob("*_selected_markets.jsonl")):
        week = fp.name.split("_", 1)[0]
        for row in iter_jsonl_rows(fp):
            event_ticker = str(row.get("event_ticker") or "")
            if event_ticker:
                out.setdefault(event_ticker, []).append((week, row))
    return out


@lru_cache(maxsize=4)
def _poly_map_by_kalshi(root_str: str) -> dict[str, list[dict[str, Any]]]:
    return _csv_rows_by_key(Path(root_str) / "map.csv", "kalshi_ticker")


@lru_cache(maxsize=4)
def _poly_rejections_by_kalshi(root_str: str) -> dict[str, dict[str, Any]]:
    rows = _csv_rows_by_key(Path(root_str) / "rejected.csv", "kalshi_ticker")
    return {key: values[0] for key, values in rows.items() if values}


def _csv_rows_by_key(path: Path, key: str) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = str(row.get(key) or "")
            if value:
                out.setdefault(value, []).append(row)
    return out

def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
