"""Expand prophet retrieve events.json into per-binary Kalshi matcher rows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .kalshi import list_markets
from .polymarket import FAMILY_QUERY, _keywords, _norm

_MACRO_PREFIXES = (
    "KXECON", "KXHOUSING", "KXFED", "KXCB", "KXESGDP", "KXUSTYLD",
    "KXAAAGAS", "KXDEGDP", "KX30Y",
)


def load_retrieve_events(path: Path | str) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("events", "tasks", "data"):
            if isinstance(raw.get(key), list):
                return raw[key]
    raise ValueError(f"Unrecognized events file shape: {path}")


def normalize_label_for_align(label: str) -> str:
    t = _norm(label or "")
    return re.sub(r"\s+", " ", t).strip()


def label_core(label: str) -> str:
    """Strip leading bucket words for looser Kalshi/Poly alignment."""
    t = normalize_label_for_align(label)
    for prefix in (
        "above ", "exactly ", "cut ", "hike ", "maintain current rate ",
        "maintains rate ", "cut more than ", "hike more than ",
    ):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    return t


def is_macro_family(family: str) -> bool:
    fam = (family or "").upper()
    return any(fam.startswith(p) for p in _MACRO_PREFIXES)


def align_market_to_outcome(market: dict, outcome: str) -> bool:
    oc = normalize_label_for_align(outcome)
    core = label_core(outcome)
    if not oc:
        return False
    for field in ("yes_sub_title", "subtitle", "title"):
        raw = market.get(field)
        if not raw:
            continue
        mk = normalize_label_for_align(str(raw))
        if oc == mk or oc in mk or mk in oc:
            return True
        if core and len(core) >= 2 and (core in mk or core == label_core(str(raw))):
            return True
    return False


def build_event_search_query(event: dict) -> str:
    family = (event.get("event_ticker") or event.get("market_ticker") or "").split("-")[0]
    if family in FAMILY_QUERY:
        return FAMILY_QUERY[family]
    title = event.get("title") or ""
    return _keywords(title) or title


def expand_retrieve_event(event: dict) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (matcher metas, unaligned outcome labels)."""
    event_ticker = event.get("event_ticker") or event.get("market_ticker") or ""
    if not event_ticker:
        return [], []

    markets = list_markets(event_ticker=event_ticker, status="open")
    desc = " ".join(
        x for x in (event.get("rules"), event.get("description")) if x
    )
    family = event_ticker.split("-")[0]
    title = event.get("title") or ""

    outcomes = event.get("outcomes") or []
    if not outcomes and event.get("subtitle"):
        outcomes = [event["subtitle"]]

    metas: list[dict[str, Any]] = []
    unaligned: list[str] = []
    used: set[str] = set()

    for outcome in outcomes:
        label = str(outcome).strip()
        if not label:
            continue
        hit = next((m for m in markets if align_market_to_outcome(m, label)), None)
        if hit is None:
            unaligned.append(label)
            continue
        ticker = hit.get("ticker") or ""
        if not ticker or ticker in used:
            continue
        used.add(ticker)
        metas.append({
            "ticker": ticker,
            "question": title,
            "short_label": label,
            "description": desc,
            "family": family,
            "event_ticker": event_ticker,
            "outcome_label": label,
            "search_query": build_event_search_query(event),
        })

    return metas, unaligned
