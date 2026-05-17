"""FastAPI agent server for the Prophet Arena forecasting track.

Matches the wire contract from https://prophetarena.co/developer :
  - POST /predict accepts an Event JSON body
  - Returns {"probabilities": [{"market": <outcome_label>, "probability": <float>}, ...]}

Behavior:
  - Loads `config/ensemble.example.json` (or path from PROPHET_CONFIG env var)
  - Runs every enabled forecaster lane in parallel via ThreadPoolExecutor
  - Aggregates with the existing ensemble + calibration stack
  - Returns the calibrated distribution, mapping back onto the event's `outcomes`

Run locally:
    uvicorn scripts.agent_server:app --host 0.0.0.0 --port 8000

Then point the CLI at it:
    prophet forecast predict --events events.json --agent-url http://localhost:8000/predict
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.calibration import CalibrationConfig  # noqa: E402
from prep.ensemble import JudgeConfig, aggregate_forecasts, forecast_members_parallel  # noqa: E402
from prep.forecasters import ForecasterConfig  # noqa: E402
from prep.packets import packet_from_arena_event  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "ensemble.example.json"
CONFIG_PATH = Path(os.environ.get("PROPHET_CONFIG", DEFAULT_CONFIG_PATH))


# --- Wire schema (Prophet Arena dev docs) ---------------------------------

class ArenaEvent(BaseModel):
    """Inbound Event body. Extra fields ignored — Arena adds metadata over time."""

    event_ticker: str | None = None
    market_ticker: str | None = None
    task_id: str | None = None  # newer dataset format may send this
    title: str
    subtitle: str | None = None
    description: str | None = None
    context: str | None = None  # newer dataset rename
    category: str | None = None
    rules: str | None = None
    close_time: str | None = None
    predict_by: str | None = None  # newer dataset rename
    outcomes: list[str]
    resolved_outcome: dict | None = None  # always None for live predict, present in historical

    model_config = {"extra": "allow"}


class OutcomeProbability(BaseModel):
    market: str
    probability: float


class PredictionResponse(BaseModel):
    probabilities: list[OutcomeProbability]


# --- Config loading -------------------------------------------------------

def _load_models() -> tuple[list[ForecasterConfig], CalibrationConfig, float, JudgeConfig]:
    cfg = json.loads(CONFIG_PATH.read_text())
    models = [
        ForecasterConfig.from_dict(m)
        for m in cfg.get("models", [])
        if m.get("enabled", True)
    ]
    calibration = CalibrationConfig.from_dict(cfg.get("calibration"))
    ensemble_cfg = cfg.get("ensemble", {})
    market_anchor_weight = float(ensemble_cfg.get("market_anchor_weight", 1.5))
    judge = JudgeConfig.from_dict(ensemble_cfg.get("judge"))
    return models, calibration, market_anchor_weight, judge


_models: list[ForecasterConfig] = []
_calibration: CalibrationConfig = CalibrationConfig()
_market_anchor_weight: float = 1.5
_judge: JudgeConfig = JudgeConfig()


# --- FastAPI app ----------------------------------------------------------

app = FastAPI(title="ProphetHacks Forecast Agent")


@app.on_event("startup")
def _startup() -> None:
    global _models, _calibration, _market_anchor_weight, _judge
    _models, _calibration, _market_anchor_weight, _judge = _load_models()
    logger.info("Loaded %d enabled lanes: %s", len(_models), [m.name for m in _models])


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "enabled_lanes": [m.name for m in _models],
        "judge_enabled": _judge.enabled,
        "judge_model": _judge.model if _judge.enabled else None,
        "config": str(CONFIG_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(event: ArenaEvent) -> PredictionResponse:
    packet = packet_from_arena_event(event.model_dump())

    if not _models:
        # Degenerate case: no lanes enabled. Return uniform over outcomes so the
        # eval server still gets a valid response.
        share = 1.0 / max(1, len(event.outcomes))
        return PredictionResponse(
            probabilities=[OutcomeProbability(market=o, probability=share) for o in event.outcomes]
        )

    run = forecast_members_parallel(_models, packet, continue_on_error=True)
    forecasts = run.members
    errors = run.errors
    for error in errors:
        logger.warning("lane failed: %s", error)

    if not forecasts:
        # All lanes failed (provider outage, bad key, etc). Don't 502 the eval —
        # degrade to the anchor distribution that the empty-members path in
        # aggregate_forecasts already produces (market_mid for binary Kalshi,
        # uniform otherwise). Strictly better than returning no answer.
        logger.warning("all lanes failed for %s: %s", packet.market_ticker, "; ".join(errors))
        supervisor = aggregate_forecasts(
            packet,
            [],
            calibration=_calibration,
            market_anchor_weight=_market_anchor_weight,
            judge=_judge,
        )
    else:
        supervisor = aggregate_forecasts(
            packet,
            forecasts,
            calibration=_calibration,
            market_anchor_weight=_market_anchor_weight,
            judge=_judge,
        )

    # Map calibrated distribution onto the event's outcomes exactly (preserve order).
    dist = supervisor.calibrated_probabilities
    return PredictionResponse(
        probabilities=[
            OutcomeProbability(market=o, probability=float(dist.get(o, 0.0)))
            for o in event.outcomes
        ]
    )


# --- Local predict() entrypoint for `prophet forecast predict --local` ---

def predict(event: dict) -> dict:
    """For `prophet forecast predict --local scripts.agent_server`.

    Mirrors the wire contract of `POST /predict`. Returns a dict shaped
    `{"probabilities": [{"market": ..., "probability": ...}, ...]}`.
    """
    global _models, _calibration, _market_anchor_weight, _judge
    if not _models:
        _models, _calibration, _market_anchor_weight, _judge = _load_models()
    arena = ArenaEvent(**event)
    response = predict_endpoint(arena)
    return response.model_dump()
