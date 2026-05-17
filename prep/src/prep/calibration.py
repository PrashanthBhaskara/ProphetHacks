"""Calibration helpers for shrinking model forecasts toward market prices."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .schemas import MarketPacket, clamp_prob


def logit(p: float) -> float:
    import math

    p = clamp_prob(p)
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    import math

    return clamp_prob(1.0 / (1.0 + math.exp(-x)))


def time_to_close_hours(packet: MarketPacket) -> float | None:
    if not packet.close_time or not packet.as_of:
        return None
    try:
        close = datetime.fromisoformat(packet.close_time.replace("Z", "+00:00"))
        as_of = datetime.fromisoformat(packet.as_of.replace("Z", "+00:00"))
        if close.tzinfo is None:
            close = close.replace(tzinfo=timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        return max(0.0, (close - as_of).total_seconds() / 3600.0)
    except ValueError:
        return None


@dataclass
class CalibrationConfig:
    """How much to trust model edge over the market anchor.

    `base_weight=0.35` means keep 35% of the supervisor's disagreement with
    Kalshi. Category and horizon multipliers let us trust models differently
    where backtests show edge.
    """

    base_weight: float = 0.35
    category_weights: dict[str, float] = field(default_factory=lambda: {
        "Crypto": 0.08,
        "Sports": 0.20,
        "Weather": 0.45,
        "Politics": 0.40,
        "Economics": 0.25,
        "Entertainment": 0.25,
        "Other": 0.20,
    })
    horizon_weights: dict[str, float] = field(default_factory=lambda: {
        "under_3h": 0.30,
        "under_24h": 0.70,
        "over_24h": 1.00,
    })

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CalibrationConfig":
        if not data:
            return cls()
        cfg = cls(base_weight=float(data.get("base_weight", 0.35)))
        cfg.category_weights.update(data.get("category_weights") or {})
        cfg.horizon_weights.update(data.get("horizon_weights") or {})
        return cfg

    def shrink_weight(self, packet: MarketPacket) -> float:
        cat_w = float(self.category_weights.get(packet.category, self.category_weights.get("Other", 0.20)))
        hours = time_to_close_hours(packet)
        if hours is None:
            horizon_key = "over_24h"
        elif hours < 3:
            horizon_key = "under_3h"
        elif hours < 24:
            horizon_key = "under_24h"
        else:
            horizon_key = "over_24h"
        horizon_w = float(self.horizon_weights[horizon_key])
        return max(0.0, min(1.0, self.base_weight * cat_w * horizon_w))


def calibrate_to_market(raw_p: float, packet: MarketPacket, config: CalibrationConfig) -> tuple[float, float]:
    market_p = packet.kalshi.market_mid
    weight = config.shrink_weight(packet)
    return clamp_prob(market_p + weight * (raw_p - market_p)), weight
