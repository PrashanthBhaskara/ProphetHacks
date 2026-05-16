"""Deterministic trade decision and sizing rules.

⚠️  DEPRECATION NOTE (per STRATEGY_FINDINGS.md):

The Kelly-fraction sizing logic in this module is **NOT** what we should use
for the live trading-track run. Per backtests on the official `subset_1200`
benchmark, `kelly_lite` is consistently the worst-performing strategy
(`noisy + kelly_lite` = −$640 aggregate, `inverse_market + kelly_lite` =
full $10k ruin). It over-sizes bets when probabilities are noisy.

For the live agent, use `prep.trading.strategies.build_recommended_strategy()`
which returns `RebalancingStrategy(max_spread=1.02)` wrapped to skip Crypto
markets. That's the universal winner across forecasters in the benchmark.

This module is kept because the forecasting harness (`backtest_ensemble.py`)
reads `RiskConfig.from_dict(cfg.get("risk"))` to size diagnostic bets during
ensemble dry-runs. Don't remove until that path is migrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from prep.schemas import MarketPacket, SupervisorForecast


Side = Literal["YES", "NO", "NONE"]


@dataclass
class RiskConfig:
    min_edge: float = 0.06
    spread_buffer: float = 0.01
    fee_buffer: float = 0.01
    uncertainty_buffer: float = 0.02
    max_stake: float = 1.0
    kelly_fraction: float = 0.10
    min_confidence: float = 0.35

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RiskConfig":
        if not data:
            return cls()
        return cls(
            min_edge=float(data.get("min_edge", 0.06)),
            spread_buffer=float(data.get("spread_buffer", 0.01)),
            fee_buffer=float(data.get("fee_buffer", 0.01)),
            uncertainty_buffer=float(data.get("uncertainty_buffer", 0.02)),
            max_stake=float(data.get("max_stake", 1.0)),
            kelly_fraction=float(data.get("kelly_fraction", 0.10)),
            min_confidence=float(data.get("min_confidence", 0.35)),
        )


@dataclass
class TradeDecision:
    side: Side
    price: float | None
    stake: float
    edge: float
    threshold: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "price": self.price,
            "stake": self.stake,
            "edge": self.edge,
            "threshold": self.threshold,
            "reason": self.reason,
        }


def _kelly_fraction(prob: float, price: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    # Binary contract paying $1. b = net odds per $1 staked.
    b = (1.0 - price) / price
    q = 1.0 - prob
    return max(0.0, (b * prob - q) / b)


def decide_trade(packet: MarketPacket, supervisor: SupervisorForecast, risk: RiskConfig) -> TradeDecision:
    if supervisor.confidence < risk.min_confidence:
        return TradeDecision("NONE", None, 0.0, 0.0, risk.min_edge, "supervisor confidence below threshold")

    yes_ask = packet.kalshi.yes_ask
    no_ask = packet.kalshi.no_ask
    if yes_ask is None or no_ask is None or yes_ask <= 0 or no_ask <= 0:
        return TradeDecision("NONE", None, 0.0, 0.0, risk.min_edge, "missing executable ask price")

    spread = packet.kalshi.spread or 0.0
    threshold = risk.min_edge + risk.fee_buffer + risk.spread_buffer * spread + risk.uncertainty_buffer * (1.0 - supervisor.confidence)
    p_yes = supervisor.calibrated_p_yes
    yes_edge = p_yes - yes_ask
    no_edge = (1.0 - p_yes) - no_ask

    if yes_edge < threshold and no_edge < threshold:
        return TradeDecision("NONE", None, 0.0, max(yes_edge, no_edge), threshold, "no edge after buffers")

    if yes_edge >= no_edge:
        kelly = _kelly_fraction(p_yes, yes_ask) * risk.kelly_fraction
        stake = min(risk.max_stake, risk.max_stake * kelly)
        return TradeDecision("YES", yes_ask, stake, yes_edge, threshold, "YES edge cleared buffers")

    kelly = _kelly_fraction(1.0 - p_yes, no_ask) * risk.kelly_fraction
    stake = min(risk.max_stake, risk.max_stake * kelly)
    return TradeDecision("NO", no_ask, stake, no_edge, threshold, "NO edge cleared buffers")
