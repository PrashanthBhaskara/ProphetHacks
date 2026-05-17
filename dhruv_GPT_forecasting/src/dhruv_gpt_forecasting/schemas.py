"""Canonical interfaces for Dhruv's GPT forecasting lane."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TradeRecommendation = Literal["BUY_YES", "BUY_NO", "BUY_YES_SMALL", "BUY_NO_SMALL", "NO_TRADE"]
TradeSide = Literal["YES", "NO", "NONE"]
EventStructure = Literal[
    "binary",
    "mutually_exclusive",
    "threshold_ladder",
    "range_bucket",
    "independent_binary",
]


def clamp_prob(value: float | int | str | None, lo: float = 0.01, hi: float = 0.99) -> float:
    if value is None:
        value = 0.5
    return max(lo, min(hi, float(value)))


@dataclass
class MarketQuote:
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    last_price: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    liquidity: float | None = None
    snapshot_time: str | None = None

    @property
    def market_mid(self) -> float:
        if self.yes_ask is not None and self.no_ask is not None:
            return clamp_prob((self.yes_ask + (1.0 - self.no_ask)) / 2.0)
        if self.yes_bid is not None and self.yes_ask is not None:
            return clamp_prob((self.yes_bid + self.yes_ask) / 2.0)
        if self.last_price is not None:
            return clamp_prob(self.last_price)
        return 0.5

    @property
    def spread(self) -> float | None:
        if self.yes_ask is not None and self.no_ask is not None:
            return max(0.0, self.yes_ask + self.no_ask - 1.0)
        if self.yes_bid is not None and self.yes_ask is not None:
            return max(0.0, self.yes_ask - self.yes_bid)
        return None

    @property
    def executable_yes(self) -> float | None:
        return self.yes_ask

    @property
    def executable_no(self) -> float | None:
        if self.no_ask is not None:
            return self.no_ask
        if self.yes_bid is not None:
            return 1.0 - self.yes_bid
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeaturePacket:
    as_of: str | None
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None
    rules: str | None
    category: str
    close_time: str | None
    outcomes: list[str]
    quote: MarketQuote
    price_trajectory: list[dict[str, Any]] = field(default_factory=list)
    horizon_hours: float | None = None
    event_structure: EventStructure = "binary"
    evidence_digest: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def is_binary_yes_no(self) -> bool:
        return self.outcomes == ["YES", "NO"]

    def compact_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "event_ticker": self.event_ticker,
            "market_ticker": self.market_ticker,
            "title": self.title,
            "subtitle": self.subtitle,
            "rules": self.rules,
            "category": self.category,
            "close_time": self.close_time,
            "outcomes": self.outcomes,
            "event_structure": self.event_structure,
            "quote": self.quote.to_dict(),
            "horizon_hours": self.horizon_hours,
            "features": self.features,
            "evidence_digest": self.evidence_digest[:8],
            "price_trajectory_tail": self.price_trajectory[-12:],
        }


@dataclass
class StatForecast:
    probabilities: dict[str, float]
    market_prior: float
    calibrated_probability: float
    uncertainty: float
    confidence: float
    reason_codes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LaneForecast:
    probabilities: dict[str, float]
    confidence: float = 0.5
    uncertainty: float = 0.5
    defer_to_market: bool = True
    market_delta_bps: int = 0
    reason_codes: list[str] = field(default_factory=list)
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    counterarguments: list[dict[str, Any]] = field(default_factory=list)
    information_gaps: list[str] = field(default_factory=list)
    trade_recommendation: TradeRecommendation = "NO_TRADE"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TradeDecision:
    side: TradeSide
    price: float | None
    edge: float
    threshold: float
    stake: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SupervisorDecision:
    probabilities: dict[str, float]
    confidence: float
    uncertainty: float
    source: str
    trade_recommendation: TradeRecommendation
    trade_decision: TradeDecision
    audit_summary: dict[str, Any] = field(default_factory=dict)

    def to_prediction_response(self) -> dict[str, Any]:
        return {
            "probabilities": [
                {"market": outcome, "probability": float(prob)}
                for outcome, prob in self.probabilities.items()
            ]
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["prediction_response"] = self.to_prediction_response()
        return data


@dataclass
class ApiCallLog:
    provider: str
    model: str
    prompt_hash: str
    api_key_env: str | None
    api_key_fingerprint: str | None
    latency_sec: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    cache_key: str | None
    fallback_path: str | None
    search_grounding_enabled: bool = False
    search_grounding_engine: str | None = None
    response_annotation_count: int = 0
    provider_response_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
