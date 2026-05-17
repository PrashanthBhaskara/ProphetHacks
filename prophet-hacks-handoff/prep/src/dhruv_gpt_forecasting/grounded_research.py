"""Gemini native-search source reading for live forecast evidence."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .arena_types import ArenaForecastPacket
from .config import ForecastConfig, resolve_api_key
from .features import parse_dt
from .openrouter import call_openrouter_json
from .prompts import PROMPT_ROOT, prompt_hash


def gather_grounded_research_evidence(
    packet: ArenaForecastPacket,
    cfg: ForecastConfig,
    *,
    enabled: bool | None = None,
    deadline_at: float | None = None,
    existing_evidence: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return one auditable live source-reading digest from Gemini search grounding."""
    if enabled is None:
        enabled = _env_bool("ARENA_ENABLE_GROUNDED_RESEARCH", cfg.arena.grounded_research_enabled_default)
    if not enabled or _env_bool("ARENA_DISABLE_GROUNDED_RESEARCH", False):
        return []
    if _env_bool("ARENA_OFFLINE", False) or _env_bool("ARENA_DISABLE_GPT", False):
        return []
    if not cfg.model.native_search_grounding_enabled:
        return []
    is_live = _is_live_as_of(packet, cfg)
    backtest_internet = _backtest_internet_enabled(cfg)
    if cfg.arena.grounded_research_live_only and not is_live and not backtest_internet:
        return []
    if not _has_grounded_research_budget(cfg, deadline_at):
        return [{
            "source": "gemini_native_search_grounded_research",
            "timestamp": _now(),
            "claim": "Gemini native-search research was skipped because the evidence budget was too small.",
            "error": "deadline_budget_before_grounded_research",
        }]
    if resolve_api_key(cfg.model)[0] is None:
        return [{
            "source": "gemini_native_search_grounded_research",
            "timestamp": _now(),
            "claim": "Gemini native-search research was skipped because the Gemini key is missing.",
            "error": f"missing_api_key:{cfg.model.api_key_env}",
        }]

    questions = targeted_research_questions(packet)
    messages = grounded_research_messages(packet, questions, existing_evidence or [])
    research_model = _grounded_research_model(cfg)
    p_hash = prompt_hash(messages, research_model.model)
    cache_path = _cache_path(cfg, p_hash)
    cached = _read_cache(cache_path)
    if cached:
        item = _evidence_item(
            packet,
            questions,
            cached["payload"],
            cached.get("call_log"),
            cache_hit=True,
            is_live=is_live,
        )
        return [item]

    try:
        payload, call_log = call_openrouter_json(
            model=research_model,
            messages=messages,
            budget=cfg.budget,
            cache_key="grounded_research",
            timeout_seconds=_timeout_seconds(cfg, deadline_at),
            search_grounding=True,
        )
    except Exception as exc:  # noqa: BLE001 - source reading must never block a forecast.
        return [{
            "source": "gemini_native_search_grounded_research",
            "timestamp": _now(),
            "claim": "Gemini native-search research failed; final model should rely on other evidence.",
            "error": f"{type(exc).__name__}:{exc}",
            "targeted_questions": questions,
        }]

    call_log_dict = call_log.to_dict()
    _write_cache(cache_path, {"payload": payload, "call_log": call_log_dict})
    return [_evidence_item(packet, questions, payload, call_log_dict, cache_hit=False, is_live=is_live)]


def grounded_research_messages(
    packet: ArenaForecastPacket,
    questions: list[str],
    existing_evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = {
        "contract": {
            "as_of": packet.as_of,
            "event_ticker": packet.event_ticker,
            "market_ticker": packet.market_ticker,
            "title": packet.title,
            "subtitle": packet.subtitle,
            "description": packet.description,
            "context": packet.context,
            "category": packet.category,
            "rules": packet.rules,
            "close_time": packet.close_time,
            "outcomes": packet.outcomes,
            "event_structure": packet.event_structure,
            "horizon_hours": packet.horizon_hours,
            "extracted_entities": packet.extracted_entities,
        },
        "source_date_policy": {
            "forecast_as_of": packet.as_of,
            "must_verify_source_published_at": True,
            "include_only_sources_published_at_or_before_as_of": True,
            "exclude_sources_with_missing_or_ambiguous_dates": True,
            "all_claims_must_map_to_source_notes": True,
        },
        "targeted_questions": questions,
        "existing_evidence_preview": _compact_existing_evidence(existing_evidence),
        "instruction": (
            "Use native search grounding to read current sources for this exact contract. "
            "Before using any source, verify its own publish/update timestamp is at or before contract.as_of. "
            "Drop undated, ambiguous-date, and post-as_of sources. "
            "Return a compact JSON digest only; do not forecast probabilities. "
            "Keep each string under 180 characters and do not use Markdown."
        ),
    }
    return [
        {"role": "system", "content": (PROMPT_ROOT / "grounded_research_v1_system.txt").read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"), sort_keys=True)},
    ]


def targeted_research_questions(packet: ArenaForecastPacket) -> list[str]:
    base = [
        "What current facts or official announcements could materially change the probability of this exact contract?",
        "What contract-resolution details or ambiguities should the final forecaster account for?",
        "Are there breaking-news items, source conflicts, or stale narratives that change the forecast as of now?",
    ]
    category = packet.category
    if category == "Sports":
        base.extend([
            "What are the latest injuries, lineups, rest/travel factors, matchup notes, and odds movement?",
            "Which team/player-specific news is most likely to matter for the listed outcomes?",
        ])
    elif category in {"Economics", "Financials"}:
        base.extend([
            "What recent official economic releases, central-bank commentary, yields, commodities, or equity moves drive this contract?",
            "What is the timing of the next relevant data release versus the contract close and resolution rules?",
        ])
    elif category in {"Crypto", "Commodities"}:
        base.extend([
            "What recent price action, flows, macro risk sentiment, regulation, exchange, or ETF news drives this asset contract?",
            "Are there market-structure or weekend/liquidity effects that matter before close?",
        ])
    elif category in {"Politics", "Elections"}:
        base.extend([
            "What are the latest polls, official actions, legal developments, endorsements, or campaign events?",
            "Which high-quality sources conflict with social or partisan sentiment?",
        ])
    elif category in {"Entertainment", "Culture"} or "reality" in _event_text(packet):
        base.extend([
            "What official releases, credible spoilers, voting/audience signals, ratings, box office, or awards news matter?",
            "How reliable and fresh are entertainment or social sentiment signals for this contract?",
        ])
    elif category in {"Weather", "Climate and Weather"}:
        base.extend([
            "What are the latest official forecasts, watches/warnings, observations, and model updates?",
            "How does the forecast timing line up with the contract's resolution window?",
        ])
    return base[:7]


def _evidence_item(
    packet: ArenaForecastPacket,
    questions: list[str],
    payload: dict[str, Any],
    call_log: dict[str, Any] | None,
    *,
    cache_hit: bool,
    is_live: bool,
) -> dict[str, Any]:
    sanitized, date_audit = _sanitize_payload_for_pit(packet, payload)
    if date_audit.get("accepted_source_count") == 0:
        return {
            "source": "gemini_native_search_grounded_research",
            "timestamp": _now(),
            "claim": "Gemini native-search research returned no source with a verified pre-as_of publish timestamp.",
            "pit_mode": "native_search_no_verified_published_at_sources",
            "target_event_ticker": packet.event_ticker,
            "target_market_ticker": packet.market_ticker,
            "targeted_questions": questions,
            "source_date_audit": date_audit,
            "information_gaps": [
                "No internet source could be used because every source was undated, ambiguous, or after forecast as_of."
            ],
            "error": "no_pit_verified_sources",
            "api_log": call_log,
            "cache_hit": cache_hit,
        }
    quality = payload.get("evidence_quality") if isinstance(payload.get("evidence_quality"), dict) else {}
    pit_mode = "live_native_search_grounding" if is_live else "backtest_native_search_published_at_verified"
    return {
        "source": "gemini_native_search_grounded_research",
        "timestamp": _now(),
        "claim": "Gemini 3 Flash used native search grounding to read sources for this specific contract.",
        "pit_mode": pit_mode,
        "target_event_ticker": packet.event_ticker,
        "target_market_ticker": packet.market_ticker,
        "targeted_questions": _strings(sanitized.get("targeted_questions")) or questions,
        "summary": str(sanitized.get("summary") or "")[:2000],
        "macroeconomic_drivers": _dict_rows(sanitized.get("macroeconomic_drivers"))[:6],
        "breaking_news": _dict_rows(sanitized.get("breaking_news"))[:6],
        "qualitative_sentiment": _dict_rows(sanitized.get("qualitative_sentiment"))[:6],
        "contract_specific_factors": _dict_rows(sanitized.get("contract_specific_factors"))[:8],
        "source_notes": _dict_rows(sanitized.get("source_notes"))[:10],
        "excluded_sources": _dict_rows(sanitized.get("excluded_sources"))[:12],
        "source_date_audit": date_audit,
        "pit_verified_source_dates": True,
        "information_gaps": _strings(sanitized.get("information_gaps"))[:8],
        "retrieval_confidence": {
            "overall": _bounded(quality.get("overall"), 0.55),
            "source_quality": _bounded(quality.get("source_quality"), 0.55),
            "timestamp_freshness": _bounded(quality.get("freshness"), 0.55),
            "event_match": _bounded(quality.get("event_match"), 0.55),
            "pit_confidence": 0.95,
            "contradiction_or_error": _bounded(quality.get("conflict_level"), 0.0),
        },
        "api_log": call_log,
        "cache_hit": cache_hit,
    }


def _compact_existing_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        out.append({
            key: item.get(key)
            for key in ("source", "claim", "query", "summary", "yes_probability", "source_counts")
            if item.get(key) is not None
        })
    return out


def _sanitize_payload_for_pit(
    packet: ArenaForecastPacket,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    as_of_dt = parse_dt(packet.as_of)
    source_notes = _dict_rows(payload.get("source_notes"))
    excluded = _dict_rows(payload.get("excluded_sources"))
    if as_of_dt is None:
        return {
            "targeted_questions": payload.get("targeted_questions"),
            "information_gaps": _strings(payload.get("information_gaps")) + ["Unable to parse forecast as_of."],
            "excluded_sources": excluded,
        }, {
            "forecast_as_of": packet.as_of,
            "accepted_source_count": 0,
            "discarded_source_count": len(source_notes),
            "discarded_sources": [
                _discarded_source_note(note, reason="unparseable_forecast_as_of")
                for note in source_notes[:20]
            ],
        }

    accepted_notes: list[dict[str, Any]] = []
    discarded_notes: list[dict[str, Any]] = []
    for note in source_notes:
        published_raw = note.get("published_at") or note.get("updated_at")
        published = parse_dt(str(published_raw or ""))
        if published is None:
            discarded_notes.append(_discarded_source_note(note, reason="missing_or_ambiguous_date"))
            continue
        if published > as_of_dt:
            discarded_notes.append(_discarded_source_note(note, reason="after_as_of"))
            continue
        clean = dict(note)
        clean["published_at"] = published.isoformat()
        accepted_notes.append(clean)

    accepted_keys = _source_note_keys(accepted_notes)
    sanitized = {
        "targeted_questions": payload.get("targeted_questions"),
        "summary": payload.get("summary") if accepted_notes else "",
        "macroeconomic_drivers": _filter_claim_rows(payload.get("macroeconomic_drivers"), accepted_keys),
        "breaking_news": _filter_claim_rows(payload.get("breaking_news"), accepted_keys),
        "qualitative_sentiment": _filter_claim_rows(payload.get("qualitative_sentiment"), accepted_keys),
        "contract_specific_factors": _filter_claim_rows(payload.get("contract_specific_factors"), accepted_keys),
        "source_notes": accepted_notes,
        "excluded_sources": [*excluded, *discarded_notes],
        "information_gaps": payload.get("information_gaps"),
    }
    return sanitized, {
        "forecast_as_of": as_of_dt.isoformat(),
        "accepted_source_count": len(accepted_notes),
        "discarded_source_count": len(discarded_notes),
        "accepted_sources": [
            {
                key: note.get(key)
                for key in ("source", "url", "title", "published_at")
                if note.get(key) is not None
            }
            for note in accepted_notes[:10]
        ],
        "discarded_sources": discarded_notes[:20],
        "policy": "require_source_specific_published_at_at_or_before_forecast_as_of",
    }


def _filter_claim_rows(value: Any, accepted_keys: set[str]) -> list[dict[str, Any]]:
    rows = _dict_rows(value)
    if not accepted_keys:
        return []
    return [row for row in rows if _row_has_accepted_source(row, accepted_keys)]


def _row_has_accepted_source(row: dict[str, Any], accepted_keys: set[str]) -> bool:
    for key in ("source", "url"):
        raw = row.get(key)
        if raw is None:
            continue
        for identity in _source_identities(str(raw)):
            if identity in accepted_keys:
                return True
    return False


def _source_note_keys(notes: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for note in notes:
        for field in ("source", "url"):
            raw = note.get(field)
            if raw is not None:
                keys.update(_source_identities(str(raw)))
    return keys


def _source_identities(value: str) -> set[str]:
    raw = value.strip().lower()
    if not raw:
        return set()
    identities = {raw.removeprefix("www.")}
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    if parsed.netloc:
        identities.add(parsed.netloc.lower().removeprefix("www."))
    return identities


def _discarded_source_note(note: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "source": note.get("source"),
            "url": note.get("url"),
            "title": note.get("title"),
            "published_at": note.get("published_at") or note.get("updated_at"),
            "reason": reason,
        }.items()
        if value is not None
    }


def _cache_path(cfg: ForecastConfig, p_hash: str) -> Path:
    model_part = cfg.model.model.replace("/", "_")
    return Path(cfg.budget.log_dir) / "llm_cache" / f"grounded_research_{model_part}_{p_hash}.json"


def _grounded_research_model(cfg: ForecastConfig):
    max_tokens = int(os.environ.get(
        "ARENA_GROUNDED_RESEARCH_MAX_TOKENS",
        str(max(int(cfg.model.max_tokens), 2200)),
    ))
    temperature = float(os.environ.get(
        "ARENA_GROUNDED_RESEARCH_TEMPERATURE",
        str(min(float(cfg.model.temperature), 0.05)),
    ))
    return replace(cfg.model, max_tokens=max_tokens, temperature=temperature)


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not _env_bool("ARENA_ENABLE_FORECAST_CACHE", True) or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    if not _env_bool("ARENA_ENABLE_FORECAST_CACHE", True):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        return


def _timeout_seconds(cfg: ForecastConfig, deadline_at: float | None) -> float:
    configured = float(os.environ.get(
        "ARENA_GROUNDED_RESEARCH_TIMEOUT_SECONDS",
        cfg.arena.grounded_research_timeout_seconds,
    ))
    if deadline_at is None:
        return configured
    remaining = max(0.1, deadline_at - time.monotonic())
    return max(0.1, min(configured, remaining))


def _has_grounded_research_budget(cfg: ForecastConfig, deadline_at: float | None) -> bool:
    if deadline_at is None:
        return True
    required = float(os.environ.get(
        "ARENA_GROUNDED_RESEARCH_MIN_SECONDS",
        cfg.arena.grounded_research_min_seconds,
    ))
    return deadline_at - time.monotonic() >= required


def _is_live_as_of(packet: ArenaForecastPacket, cfg: ForecastConfig) -> bool:
    as_of_dt = parse_dt(packet.as_of)
    if as_of_dt is None:
        return False
    now = datetime.now(timezone.utc)
    max_age = max(0, cfg.arena.pit_external_max_live_age_minutes) * 60
    return abs((now - as_of_dt).total_seconds()) <= max_age


def _backtest_internet_enabled(cfg: ForecastConfig) -> bool:
    return _env_bool(
        "ARENA_ENABLE_BACKTEST_INTERNET",
        _env_bool("ARENA_ALLOW_BACKTEST_INTERNET", cfg.arena.grounded_research_backtest_enabled),
    )


def _event_text(packet: ArenaForecastPacket) -> str:
    return " ".join(str(value or "") for value in (
        packet.title,
        packet.subtitle,
        packet.description,
        packet.context,
        packet.rules,
        " ".join(packet.outcomes),
    )).lower()


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _bounded(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_digest_cache_key(packet: ArenaForecastPacket, questions: list[str]) -> str:
    """Stable helper for debugging grounded source-reading cache keys."""
    payload = {
        "as_of": packet.as_of,
        "market_ticker": packet.market_ticker,
        "title": packet.title,
        "outcomes": packet.outcomes,
        "questions": questions,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
