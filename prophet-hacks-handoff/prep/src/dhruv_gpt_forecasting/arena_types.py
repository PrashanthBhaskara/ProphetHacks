"""Prophet Arena forecasting-only data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .schemas import EventStructure


@dataclass
class ArenaForecastPacket:
    as_of: str
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None
    description: str | None
    context: str | None
    category: str
    rules: str | None
    close_time: str | None
    outcomes: list[str]
    event_structure: EventStructure
    horizon_hours: float | None
    extracted_entities: dict[str, Any] = field(default_factory=dict)
    historical_analogs: list[dict[str, Any]] = field(default_factory=list)
    live_evidence: list[dict[str, Any]] = field(default_factory=list)
    deterministic_priors: dict[str, float] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)

    def compact_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "event_ticker": self.event_ticker,
            "market_ticker": self.market_ticker,
            "title": self.title,
            "subtitle": self.subtitle,
            "description": self.description,
            "context": self.context,
            "category": self.category,
            "rules": self.rules,
            "close_time": self.close_time,
            "outcomes": self.outcomes,
            "event_structure": self.event_structure,
            "horizon_hours": self.horizon_hours,
            "extracted_entities": self.extracted_entities,
            "historical_analogs": self.historical_analogs[:8],
            "live_evidence": self.live_evidence[:8],
            "deterministic_priors": self.deterministic_priors,
            "features": self.features,
        }


@dataclass
class ArenaPrior:
    probabilities: dict[str, float]
    confidence: float
    uncertainty: float
    source: str
    reason_codes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArenaForecast:
    probabilities: dict[str, float]
    confidence: float
    uncertainty: float
    source: str
    reason_codes: list[str] = field(default_factory=list)
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    counterarguments: list[dict[str, Any]] = field(default_factory=list)
    information_gaps: list[str] = field(default_factory=list)
    calibration_note: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)

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

