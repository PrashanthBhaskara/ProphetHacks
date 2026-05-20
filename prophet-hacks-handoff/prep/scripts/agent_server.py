"""FastAPI agent server for the Prophet Arena forecasting track.

Matches the wire contract from https://prophetarena.co/developer :
  - POST /predict accepts an Event JSON body
  - Returns {"probabilities": [{"market": <outcome_label>, "probability": <float>}, ...]}

Behavior:
  - Loads `config/FINAL.json` (or path from PROPHET_CONFIG env var)
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
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.calibration import CalibrationConfig  # noqa: E402
from prep.ensemble import (  # noqa: E402
    JudgeConfig,
    aggregate_forecasts,
    final_mutually_exclusive_probabilities,
    forecast_members_parallel,
)
from prep.forecasters import ForecasterConfig  # noqa: E402
from prep.kalshi import get_market, list_markets  # noqa: E402
from prep.packets import packet_from_arena_event  # noqa: E402
from prep.schemas import KalshiQuote, MarketPacket, is_yes_no_outcomes, normalize_distribution  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

PREP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PREP_ROOT.parent.parent


def _load_local_env() -> None:
    """Load local .env values without overriding exported environment vars."""
    if os.environ.get("PROPHET_SKIP_DOTENV"):
        return
    for env_path in (Path.cwd() / ".env", PREP_ROOT / ".env", REPO_ROOT / ".env"):
        if not env_path.exists():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env()

DEFAULT_CONFIG_PATH = PREP_ROOT / "config" / "FINAL.json"
CONFIG_PATH = Path(os.environ.get("PROPHET_CONFIG", DEFAULT_CONFIG_PATH))

# Wall-clock budget for the entire /predict call (lane fan-out + aggregation +
# judge). 9m45s keeps the whole call under the 10-minute Prophet Arena ceiling
# after 7.5-minute lane budgets and a 2-minute judge budget. On timeout we
# return the current Kalshi market-implied distribution when available.
DEFAULT_ENSEMBLE_TIMEOUT_SECONDS = 585.0


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


def _f(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _market_mid(m: dict) -> float | None:
    bid = _f(m.get("yes_bid_dollars"))
    ask = _f(m.get("yes_ask_dollars"))
    last = _f(m.get("last_price_dollars"))
    if bid is not None and ask is not None and (bid > 0 or ask > 0):
        return (bid + ask) / 2.0
    if last is not None and last > 0:
        return last
    return None


def _flag_value(data: dict, *keys: str) -> bool | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            folded = value.strip().lower()
            if folded in {"true", "yes", "1", "mutually_exclusive", "exclusive"}:
                return True
            if folded in {"false", "no", "0", "non_exclusive", "non-exclusive", "multilabel", "multi_label", "component"}:
                return False
    return None


def _is_mutually_exclusive_event(event: ArenaEvent, packet: MarketPacket) -> bool:
    """Best-effort exclusivity check for the final output-only threshold policy."""
    event_data = event.model_dump()
    retrieval = getattr(packet, "retrieval", {}) or {}

    explicit = _flag_value(
        event_data,
        "is_mutually_exclusive",
        "mutually_exclusive",
        "exclusive",
    )
    if explicit is not None:
        return explicit
    explicit = _flag_value(
        retrieval,
        "is_mutually_exclusive",
        "mutually_exclusive",
        "exclusive",
    )
    if explicit is not None:
        return explicit

    structure_values = [
        event_data.get("event_structure"),
        event_data.get("outcome_structure"),
        event_data.get("outcome_type"),
        event_data.get("resolution_type"),
        retrieval.get("event_structure"),
        retrieval.get("outcome_structure"),
        retrieval.get("outcome_type"),
        retrieval.get("resolution_type"),
    ]
    non_exclusive = {"component", "components", "multilabel", "multi_label", "non_exclusive", "non-exclusive", "independent", "independent_binary"}
    exclusive = {"binary", "mutually_exclusive", "exclusive", "range_bucket", "threshold_ladder", "single_winner", "winner"}
    for value in structure_values:
        if not isinstance(value, str):
            continue
        folded = value.strip().lower()
        if folded in non_exclusive:
            return False
        if folded in exclusive:
            return True

    text = " ".join(
        str(part or "") for part in (
            event.title,
            event.subtitle,
            event.description,
            event.context,
            event.rules,
        )
    ).lower()
    if any(phrase in text for phrase in ("multiple outcomes can", "multiple labels can", "select all", "each outcome independently")):
        return False

    return len(event.outcomes) > 1


def _final_response_distribution(
    dist: dict[str, float],
    outcomes: list[str],
    *,
    mutually_exclusive: bool,
) -> dict[str, float]:
    if mutually_exclusive:
        return final_mutually_exclusive_probabilities(dist, outcomes)
    return {outcome: float(dist.get(outcome, 0.0)) for outcome in outcomes}


def _match_market(outcome: str, markets: list[dict]) -> dict | None:
    """Match an outcome label to its Kalshi market by searching title/subtitle."""
    outcome_lower = outcome.lower()
    for m in markets:
        title = (m.get("title") or "").lower()
        sub = (m.get("yes_sub_title") or m.get("subtitle") or "").lower()
        if outcome_lower in title or outcome_lower in sub:
            return m
    return None


def _enrich_packet(packet: MarketPacket) -> MarketPacket:
    """Fetch live Kalshi data and inject it into the packet before forecasting.

    Binary events: fetch the single market by market_ticker.
    Multi-outcome events: fetch all markets under the event_ticker and build
    a market_implied_probabilities dict in packet.retrieval.
    """
    market_ticker = packet.market_ticker
    event_ticker = packet.event_ticker

    try:
        # --- Binary: try fetching by market_ticker first ---
        if packet.is_binary and market_ticker and market_ticker.startswith("KX"):
            market = get_market(market_ticker)
            if market:
                packet.kalshi = KalshiQuote(
                    yes_bid=_f(market.get("yes_bid_dollars")),
                    yes_ask=_f(market.get("yes_ask_dollars")),
                    no_bid=_f(market.get("no_bid_dollars")),
                    no_ask=_f(market.get("no_ask_dollars")),
                    last_price=_f(market.get("last_price_dollars")),
                    volume=_f(market.get("volume_fp")),
                    open_interest=_f(market.get("open_interest_fp")),
                    yes_bid_size=_f(market.get("yes_bid_size_fp")),
                    yes_ask_size=_f(market.get("yes_ask_size_fp")),
                )
                if market.get("rules_primary") and not packet.retrieval.get("description"):
                    packet.retrieval["description"] = market["rules_primary"]
                if market.get("yes_sub_title") and not packet.subtitle:
                    packet.subtitle = market["yes_sub_title"]
                logger.info(
                    "Enriched binary %s: mid=%.3f vol=%s oi=%s",
                    market_ticker, packet.kalshi.market_mid,
                    market.get("volume_fp"), market.get("open_interest_fp"),
                )
                return packet

        # --- Multi-outcome: fetch all markets under the event ---
        ticker = event_ticker or market_ticker
        if not ticker:
            return packet
        if not ticker.startswith("KX"):
            return packet

        markets = list_markets(event_ticker=ticker, status=None, limit=100)
        if not markets:
            logger.warning("Kalshi returned no data for ticker %s", ticker)
            return packet

        # Build market_implied_probabilities keyed by outcome label
        implied: dict[str, float] = {}
        market_data: list[dict] = []
        for outcome in packet.outcomes:
            m = _match_market(outcome, markets)
            if m is None:
                continue
            mid = _market_mid(m)
            # Kalshi floor: bid=0, ask=0.01 → mid=0.005. Anything at or below
            # the minimum tick is a confirmed-eliminated outcome; store as 0.0
            # so models and the logit pool don't treat it as a live signal.
            if mid is not None:
                implied[outcome] = 0.0 if mid <= 0.01 else round(mid, 4)
            market_data.append({
                "outcome": outcome,
                "ticker": m.get("ticker"),
                "yes_bid": _f(m.get("yes_bid_dollars")),
                "yes_ask": _f(m.get("yes_ask_dollars")),
                "mid": 0.0 if (mid is not None and mid <= 0.01) else mid,
                "volume": _f(m.get("volume_fp")),
                "open_interest": _f(m.get("open_interest_fp")),
            })

        if implied:
            packet.retrieval["market_implied_probabilities"] = implied
        if market_data:
            packet.retrieval["market_data"] = market_data

        logger.info(
            "Enriched multi-outcome %s: %d/%d outcomes matched, implied=%s",
            ticker, len(implied), len(packet.outcomes),
            {k: f"{v:.3f}" for k, v in implied.items()},
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Kalshi enrichment failed for %s: %s", market_ticker or event_ticker, exc)

    return packet


def _market_price_response(packet, outcomes: list[str]) -> PredictionResponse:
    """Build a market-price response mapped onto `outcomes`.

    Used as the timeout/error fallback. Binary YES/NO events with a Kalshi
    quote return market_mid. Multi-outcome events use live
    market_implied_probabilities when Kalshi enrichment found them. Uniform is
    only used when no current market information is available.
    """
    kalshi = getattr(packet, "kalshi", None)
    if is_yes_no_outcomes(outcomes) and kalshi is not None:
        try:
            mid = float(kalshi.market_mid)
        except (TypeError, ValueError, AttributeError):
            mid = 0.5
        dist = {outcomes[0]: mid, outcomes[1]: 1.0 - mid}
    else:
        market_probs = getattr(packet, "retrieval", {}).get("market_implied_probabilities")
        if isinstance(market_probs, dict):
            share = 1.0 / max(1, len(outcomes))
            raw = {}
            for outcome in outcomes:
                try:
                    raw[outcome] = float(market_probs.get(outcome, share))
                except (TypeError, ValueError):
                    raw[outcome] = share
            dist = normalize_distribution(raw)
        else:
            share = 1.0 / max(1, len(outcomes))
            dist = {o: share for o in outcomes}
    return PredictionResponse(
        probabilities=[
            OutcomeProbability(market=o, probability=float(dist.get(o, 0.0)))
            for o in outcomes
        ]
    )


def _compute_ensemble(event: ArenaEvent, deadline: float) -> PredictionResponse:
    """Inner ensemble computation. Caller enforces the wall-clock deadline.

    Lanes that haven't completed by the deadline are dropped from the
    aggregate; aggregation still runs against whatever finished in time.
    Returns the calibrated PredictionResponse mapped onto event.outcomes.
    """
    packet = packet_from_arena_event(event.model_dump())
    packet = _enrich_packet(packet)

    if not _models:
        logger.warning("no lanes enabled for %s; returning market fallback", packet.market_ticker)
        return _market_price_response(packet, event.outcomes)

    # Inner parallel exec is delegated to forecast_members_parallel. Per-lane
    # 7.5-minute budgets live in forecasters/base.py's forecast_from_config; the
    # outer 9m45s budget is enforced by predict_endpoint via fut.result(timeout=).
    run = forecast_members_parallel(_models, packet, continue_on_error=True)
    forecasts = run.members
    errors = run.errors
    for error in errors:
        logger.warning("lane failed: %s", error)

    if not forecasts:
        # All lanes failed or timed out. Use the live Kalshi fallback directly.
        logger.warning("all lanes failed for %s: %s", packet.market_ticker, "; ".join(errors))
        return _market_price_response(packet, event.outcomes)

    for member in forecasts:
        fc = member.forecast
        rt = fc.reasoning_track
        diag = fc.diagnostics
        deferred = getattr(diag, "should_defer_to_market", False) if diag else False
        logger.info(
            "lane result  model=%s  probs=%s  conf=%.2f  evidence=%s  deferred=%s",
            fc.model_id,
            {k: round(v, 3) for k, v in fc.probabilities.items()},
            fc.forecast.confidence,
            getattr(diag, "evidence_quality", "?"),
            deferred,
        )
        if rt and rt.summary:
            logger.info("  summary: %s", rt.summary[:200])
        for ev in (rt.key_evidence if rt else [])[:2]:
            if isinstance(ev, dict):
                logger.info("  evidence: [%s] %s", ev.get("source", "?")[:40], str(ev.get("claim", ""))[:120])


    supervisor = aggregate_forecasts(
        packet,
        forecasts,
        calibration=_calibration,
        market_anchor_weight=_market_anchor_weight,
        judge=_judge,
    )

    judge_entry = next((a for a in supervisor.model_assessment if a.get("role") == "judge"), None)
    if judge_entry:
        logger.info(
            "  judge: %s — %s",
            judge_entry.get("decision", "?"),
            str(judge_entry.get("summary", ""))[:200],
        )

    # Map final distribution onto the event's outcomes exactly (preserve order).
    # The 98%/2% threshold policy is output-only and only applies to mutually
    # exclusive events; lane forecasts and non-exclusive/component outputs keep
    # their native probabilities.
    dist = _final_response_distribution(
        supervisor.calibrated_probabilities,
        event.outcomes,
        mutually_exclusive=_is_mutually_exclusive_event(event, packet),
    )
    logger.info(
        "ensemble result  market=%s  calibrated=%s  final=%s  disagreement=%s",
        packet.market_ticker,
        {k: round(v, 3) for k, v in supervisor.calibrated_probabilities.items()},
        {k: round(v, 3) for k, v in dist.items()},
        supervisor.disagreement_summary,
    )
    return PredictionResponse(
        probabilities=[
            OutcomeProbability(market=o, probability=float(dist.get(o, 0.0)))
            for o in event.outcomes
        ]
    )


def _coerce_single_event(event: ArenaEvent | dict | list[ArenaEvent] | list[dict]) -> ArenaEvent:
    """Accept a single event object or a one-item events list.

    The Prophet CLI reads an events JSON array and normally posts one event at a
    time. This also handles callers that pass the one-item array directly.
    """
    if isinstance(event, list):
        if len(event) != 1:
            raise ValueError("/predict expects one event per request")
        event = event[0]
    if isinstance(event, ArenaEvent):
        return event
    if isinstance(event, dict):
        return ArenaEvent(**event)
    raise TypeError(f"unsupported event payload type: {type(event).__name__}")


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(event: ArenaEvent | list[ArenaEvent]) -> PredictionResponse:
    """Wall-clock-bounded /predict.

    Runs the full ensemble (lane fan-out + calibration + judge) inside a
    deadline. If the whole flow hasn't returned within ENSEMBLE_TIMEOUT_SECONDS
    (default 585s = 9m45s), we abandon it and return market price.
    """
    try:
        event = _coerce_single_event(event)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    budget = float(os.environ.get("ENSEMBLE_TIMEOUT_SECONDS", DEFAULT_ENSEMBLE_TIMEOUT_SECONDS))
    deadline = time.monotonic() + budget

    supervisor_pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut = supervisor_pool.submit(_compute_ensemble, event, deadline)
        try:
            return fut.result(timeout=budget)
        except FuturesTimeoutError:
            packet = _fallback_packet(event)
            logger.warning(
                "ensemble exceeded %.0fs budget for %s; returning market price",
                budget, packet.market_ticker,
            )
            return _market_price_response(packet, event.outcomes)
        except Exception as exc:  # noqa: BLE001
            packet = _fallback_packet(event)
            logger.exception(
                "ensemble raised %s for %s; returning market price",
                type(exc).__name__, packet.market_ticker,
            )
            return _market_price_response(packet, event.outcomes)
    finally:
        supervisor_pool.shutdown(wait=False, cancel_futures=True)


def _fallback_packet(event: ArenaEvent) -> MarketPacket:
    packet = packet_from_arena_event(event.model_dump())
    try:
        return _enrich_packet(packet)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fallback Kalshi enrichment failed for %s: %s", packet.market_ticker, exc)
        return packet


# --- Local predict() entrypoint for `prophet forecast predict --local` ---

def predict(event: dict | list[dict]) -> dict:
    """For `prophet forecast predict --local scripts.agent_server`.

    Mirrors the wire contract of `POST /predict`. Returns a dict shaped
    `{"probabilities": [{"market": ..., "probability": ...}, ...]}`.
    """
    global _models, _calibration, _market_anchor_weight, _judge
    if not _models:
        _models, _calibration, _market_anchor_weight, _judge = _load_models()
    arena = _coerce_single_event(event)
    response = predict_endpoint(arena)
    return response.model_dump()
