"""JSON extraction and lane response validation."""

from __future__ import annotations

import json
from typing import Any

from .constraints import enforce_constraints
from .schemas import FeaturePacket, LaneForecast


VALID_RECS = {"BUY_YES", "BUY_NO", "BUY_YES_SMALL", "BUY_NO_SMALL", "NO_TRADE"}


def extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def lane_from_payload(payload: dict[str, Any], packet: FeaturePacket) -> LaneForecast:
    raw_probs = payload.get("probabilities") or {}
    if not isinstance(raw_probs, dict):
        raw_probs = {}
    probs = enforce_constraints(
        {str(k): float(v) for k, v in raw_probs.items() if _is_number(v)},
        packet.outcomes,
        packet.event_structure,
    )
    rec = str(payload.get("trade_recommendation") or "NO_TRADE")
    if rec not in VALID_RECS:
        rec = "NO_TRADE"
    return LaneForecast(
        probabilities=probs,
        confidence=_bounded(payload.get("confidence"), 0.5),
        uncertainty=_bounded(payload.get("uncertainty"), 0.5),
        defer_to_market=bool(payload.get("defer_to_market", True)),
        market_delta_bps=int(float(payload.get("market_delta_bps", 0) or 0)),
        reason_codes=[str(x) for x in payload.get("reason_codes") or []][:12],
        key_evidence=list(payload.get("key_evidence") or [])[:8],
        counterarguments=list(payload.get("counterarguments") or [])[:8],
        information_gaps=[str(x) for x in payload.get("information_gaps") or []][:8],
        trade_recommendation=rec,  # type: ignore[arg-type]
        raw=payload,
    )


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _bounded(value: Any, default: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = default
    return max(0.0, min(1.0, x))

