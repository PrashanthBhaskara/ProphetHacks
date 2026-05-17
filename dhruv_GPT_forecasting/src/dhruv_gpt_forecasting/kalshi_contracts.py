"""Parsing helpers for Kalshi compound contract text."""

from __future__ import annotations

import re
from typing import Any


LEG_RE = re.compile(r"^\s*(yes|no)\s+(.+?)\s*$", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,3}")
GENERIC_PROP_RE = re.compile(
    r"^(?:over|under)\s+-?\d+(?:\.\d+)?\s+"
    r"(?:points?|runs?|goals?|rebounds?|assists?|threes?|strikeouts?|hits?)\b",
    re.IGNORECASE,
)


def parse_kalshi_multileg_contract(event: dict[str, Any] | str) -> dict[str, Any]:
    """Return structured legs for KXMVE-style comma-separated contracts.

    Kalshi's multi-leg sports/cross-category contracts often encode a single
    YES/NO market as text like ``yes San Antonio,yes Over 202.5 points``. The
    YES outcome means every listed leg resolves as specified; NO means at least
    one leg fails. Passing this structure to GPT is much clearer than making it
    infer the semantics from a long comma-separated title.
    """
    if isinstance(event, str):
        text = event
        outcomes: list[str] = []
    else:
        text = _event_text(event)
        outcomes = [str(item) for item in (event.get("outcomes") or [])]
    legs = _parse_legs(text)
    if len(legs) < 2:
        return {"is_multileg": False, "component_count": len(legs), "legs": []}
    yes_no = [outcome.upper() for outcome in outcomes] in (["YES", "NO"], ["NO", "YES"])
    search_terms = _search_terms(legs)
    return {
        "is_multileg": True,
        "component_count": len(legs),
        "legs": legs,
        "search_terms": search_terms,
        "joint_yes_semantics": (
            "YES means every component leg resolves exactly as listed; "
            "NO means at least one component leg fails."
        ) if yes_no or not outcomes else None,
        "contract_format": "kalshi_comma_separated_yes_no_legs",
    }


def _event_text(event: dict[str, Any]) -> str:
    values = [
        event.get("title"),
        event.get("subtitle"),
        event.get("yes_sub_title"),
        event.get("no_sub_title"),
    ]
    return " ".join(str(value) for value in values if value)


def _parse_legs(text: str) -> list[dict[str, Any]]:
    raw_parts = [part.strip() for part in str(text or "").split(",") if part.strip()]
    legs: list[dict[str, Any]] = []
    for idx, part in enumerate(raw_parts, start=1):
        match = LEG_RE.match(part)
        if not match:
            continue
        side = match.group(1).lower()
        condition = _clean_condition(match.group(2))
        if not condition:
            continue
        legs.append({
            "index": idx,
            "side": side.upper(),
            "condition": condition,
            "search_term": _leg_search_term(condition),
            "raw": part,
        })
    if raw_parts and len(legs) / len(raw_parts) < 0.70:
        return []
    return legs


def _clean_condition(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip(".?")).strip()


def _leg_search_term(condition: str) -> str | None:
    text = _clean_condition(condition)
    if not text:
        return None
    if GENERIC_PROP_RE.match(text):
        return text
    before_colon = text.split(":", 1)[0].strip()
    if before_colon and before_colon != text:
        return before_colon
    win_match = re.match(r"(.+?)\s+wins?\b", text, re.IGNORECASE)
    if win_match:
        return win_match.group(1).strip()
    phrase = WORD_RE.match(text)
    if phrase:
        return phrase.group(0).strip()
    return text[:80]


def _search_terms(legs: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for leg in legs:
        term = str(leg.get("search_term") or "").strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        terms.append(term)
        seen.add(key)
    return terms[:12]
