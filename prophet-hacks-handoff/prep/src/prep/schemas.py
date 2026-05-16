"""Shared schemas for the ensemble forecasting and trading stack.

These dataclasses intentionally stay dependency-free. They are strict enough
to keep every model adapter interoperable, but simple enough to serialize into
JSONL for backtests and live audit logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TradeRecommendation = Literal["BUY_YES", "BUY_NO", "NO_TRADE", "BUY_YES_SMALL", "BUY_NO_SMALL"]


def clamp_prob(value: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, float(value)))


@dataclass
class KalshiQuote:
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    last_price: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    snapshot_time: str | None = None

    @property
    def market_mid(self) -> float:
        if self.yes_ask is not None and self.no_ask is not None:
            return clamp_prob((self.yes_ask + (1.0 - self.no_ask)) / 2.0)
        if self.last_price is not None:
            return clamp_prob(self.last_price)
        return 0.5

    @property
    def spread(self) -> float | None:
        if self.yes_ask is None or self.no_ask is None:
            return None
        return max(0.0, self.yes_ask + self.no_ask - 1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketPacket:
    as_of: str | None
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None
    rules: str | None
    category: str
    close_time: str | None
    kalshi: KalshiQuote
    retrieval: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kalshi"] = self.kalshi.to_dict()
        return data


@dataclass
class ForecastValues:
    p_yes: float
    confidence: float = 0.5
    uncertainty: float = 0.5
    fair_yes_price: float | None = None
    max_yes_buy_price: float | None = None
    max_no_buy_price: float | None = None
    trade_recommendation: TradeRecommendation = "NO_TRADE"

    def __post_init__(self) -> None:
        self.p_yes = clamp_prob(self.p_yes)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.uncertainty = max(0.0, min(1.0, float(self.uncertainty)))
        if self.fair_yes_price is None:
            self.fair_yes_price = self.p_yes
        if self.max_yes_buy_price is None:
            self.max_yes_buy_price = clamp_prob(self.p_yes - self.uncertainty * 0.25)
        if self.max_no_buy_price is None:
            self.max_no_buy_price = clamp_prob((1.0 - self.p_yes) - self.uncertainty * 0.25)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReasoningTrack:
    summary: str
    base_rate: str = ""
    market_analysis: str = ""
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    counterarguments: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    information_gaps: list[str] = field(default_factory=list)
    what_would_change_my_mind: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastDiagnostics:
    evidence_quality: Literal["low", "medium", "high"] = "medium"
    rules_clarity: Literal["low", "medium", "high"] = "medium"
    liquidity_quality: Literal["low", "medium", "high"] = "medium"
    market_disagreement_reason: str = ""
    should_defer_to_market: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelForecast:
    model_id: str
    provider: str
    as_of: str | None
    market_ticker: str
    forecast: ForecastValues
    reasoning_track: ReasoningTrack
    diagnostics: ForecastDiagnostics = field(default_factory=ForecastDiagnostics)
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def p_yes(self) -> float:
        return self.forecast.p_yes

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "as_of": self.as_of,
            "market_ticker": self.market_ticker,
            "forecast": self.forecast.to_dict(),
            "reasoning_track": self.reasoning_track.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "raw_response": self.raw_response,
        }


@dataclass
class SupervisorForecast:
    market_ticker: str
    raw_p_yes: float
    calibrated_p_yes: float
    confidence: float
    model_assessment: list[dict[str, Any]]
    disagreement_summary: str
    final_trade_thesis: str
    risk_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.raw_p_yes = clamp_prob(self.raw_p_yes)
        self.calibrated_p_yes = clamp_prob(self.calibrated_p_yes)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
