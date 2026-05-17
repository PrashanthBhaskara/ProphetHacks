"""Shared schemas for the ensemble forecasting and trading stack.

These dataclasses intentionally stay dependency-free. They are strict enough
to keep every model adapter interoperable, but simple enough to serialize into
JSONL for backtests and live audit logs.

Schema follows the Prophet Arena dev docs (prophetarena.co/developer):
  - Each Event carries `outcomes: list[str]` (binary cases use ["YES", "NO"]).
  - Each Prediction returns `probabilities: dict[outcome_label, probability]`.

For binary back-compat (Kalshi YES/NO), `ForecastValues.p_yes` stays as a
derived property so trading code that reads it keeps working.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TradeRecommendation = Literal["BUY_YES", "BUY_NO", "NO_TRADE", "BUY_YES_SMALL", "BUY_NO_SMALL"]

BINARY_OUTCOMES = ("YES", "NO")


def clamp_prob(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
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
    yes_bid_size: float | None = None
    yes_ask_size: float | None = None
    # Top-N order book levels: list of (price, size) sorted best-first
    yes_bid_levels: list[tuple[float, float]] = field(default_factory=list)
    no_bid_levels: list[tuple[float, float]] = field(default_factory=list)

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

    def multilevel_microprice(self, n: int = 3) -> float:
        """Volume-weighted microprice from the top-n bid and ask levels.

        yes_bid_levels = [(price, size), ...] sorted best-bid first (descending price)
        no_bid_levels  = [(price, size), ...] sorted best-ask first (ascending yes-ask price)
        Falls back to the simple top-of-book microprice if level data is absent.
        """
        yes_levels = self.yes_bid_levels[:n]
        no_levels = self.no_bid_levels[:n]
        if not yes_levels or not no_levels:
            return self.market_mid

        bid_sz = sum(s for _, s in yes_levels)
        ask_sz = sum(s for _, s in no_levels)
        if bid_sz + ask_sz < 1.0:
            return self.market_mid

        vw_bid = sum(p * s for p, s in yes_levels) / bid_sz
        # no_bid levels sorted descending by NO price; best NO bid → lowest YES ask = 1 - NO price
        vw_ask = sum((1.0 - p) * s for p, s in no_levels) / ask_sz
        # Microprice: weight mid toward whichever side has more size
        mid = (vw_bid + vw_ask) / 2.0
        imbalance = (ask_sz - bid_sz) / (bid_sz + ask_sz)
        spread = max(0.0, vw_ask - vw_bid)
        return clamp_prob(mid + imbalance * spread * 0.5)

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
    # New for multi-outcome support. Defaults to binary YES/NO so existing
    # Kalshi-derived packets work unchanged.
    outcomes: list[str] = field(default_factory=lambda: list(BINARY_OUTCOMES))
    retrieval: dict[str, Any] = field(default_factory=dict)

    @property
    def is_binary(self) -> bool:
        return tuple(self.outcomes) == BINARY_OUTCOMES

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kalshi"] = self.kalshi.to_dict()
        return data


def normalize_distribution(probs: dict[str, float]) -> dict[str, float]:
    """Clamp each prob into [0.001, 0.999] then renormalize so they sum to 1.0.

    The Prophet Arena scorer normalizes before scoring anyway, but doing it
    locally keeps reasoning interpretable and tests deterministic.
    """
    if not probs:
        return {}
    clamped = {k: clamp_prob(v) for k, v in probs.items()}
    s = sum(clamped.values())
    if s <= 0:
        return clamped
    return {k: v / s for k, v in clamped.items()}


@dataclass
class ForecastValues:
    """Per-event forecast values.

    `probabilities` is the canonical multi-outcome distribution. For binary
    events it carries {"YES": p, "NO": 1-p}. `p_yes` is a derived shim for
    back-compat with trading code that reads it directly.
    """

    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    uncertainty: float = 0.5
    fair_yes_price: float | None = None
    max_yes_buy_price: float | None = None
    max_no_buy_price: float | None = None
    trade_recommendation: TradeRecommendation = "NO_TRADE"

    def __post_init__(self) -> None:
        if self.probabilities:
            self.probabilities = normalize_distribution(self.probabilities)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.uncertainty = max(0.0, min(1.0, float(self.uncertainty)))
        py = self.p_yes
        if self.fair_yes_price is None:
            self.fair_yes_price = py
        if self.max_yes_buy_price is None:
            self.max_yes_buy_price = clamp_prob(py - self.uncertainty * 0.25)
        if self.max_no_buy_price is None:
            self.max_no_buy_price = clamp_prob((1.0 - py) - self.uncertainty * 0.25)

    @property
    def p_yes(self) -> float:
        """Back-compat scalar for trading code. Returns YES probability for
        binary events, the first outcome's probability for multi-outcome."""
        if "YES" in self.probabilities:
            return self.probabilities["YES"]
        if self.probabilities:
            return next(iter(self.probabilities.values()))
        return 0.5

    @classmethod
    def from_p_yes(cls, p_yes: float, **kwargs) -> "ForecastValues":
        """Build a binary ForecastValues from a single p_yes (back-compat
        for code that hasn't migrated to passing a full distribution)."""
        p = clamp_prob(p_yes)
        return cls(probabilities={"YES": p, "NO": 1.0 - p}, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["p_yes"] = self.p_yes  # keep wire-format consumers happy
        return d


@dataclass
class ReasoningTrack:
    summary: str
    base_rate: str = ""
    market_analysis: str = ""
    context_market_analysis: str = ""
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    source_audit: list[dict[str, Any]] = field(default_factory=list)
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

    @property
    def probabilities(self) -> dict[str, float]:
        return self.forecast.probabilities

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
    raw_probabilities: dict[str, float]
    calibrated_probabilities: dict[str, float]
    confidence: float
    model_assessment: list[dict[str, Any]]
    disagreement_summary: str
    final_trade_thesis: str
    risk_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.raw_probabilities = normalize_distribution(self.raw_probabilities)
        self.calibrated_probabilities = normalize_distribution(self.calibrated_probabilities)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    # Back-compat scalars for trading code that reads `.raw_p_yes` / `.calibrated_p_yes`.
    @property
    def raw_p_yes(self) -> float:
        return self.raw_probabilities.get("YES", next(iter(self.raw_probabilities.values()), 0.5))

    @property
    def calibrated_p_yes(self) -> float:
        return self.calibrated_probabilities.get("YES", next(iter(self.calibrated_probabilities.values()), 0.5))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["raw_p_yes"] = self.raw_p_yes
        d["calibrated_p_yes"] = self.calibrated_p_yes
        return d
