"""LSEG live news query planning.

LSEG headline search has its own query syntax. For live forecasts we let GPT
write the query, then validate and cache it. If query planning cannot run
inside the evidence budget, the deterministic fallback still gives LSEG a
reasonable category-aware query.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .arena_types import ArenaForecastPacket
from .config import ForecastConfig, ModelConfig, resolve_api_key
from .openrouter import call_openrouter_json
from .prompts import PROMPT_ROOT, prompt_hash


TOKEN_RE = re.compile(r"[A-Za-z0-9.$=:/_-]+")
RIC_RE = re.compile(r"\b[A-Z]{1,5}\.(?:O|N|L|K|TO|AX|PA|DE|MI|HK|SI)\b")


def plan_lseg_news_query(
    packet: ArenaForecastPacket,
    cfg: ForecastConfig,
    *,
    deadline_at: float | None = None,
    allow_llm: bool = True,
) -> dict[str, Any]:
    """Return an LSEG query plan with a safe deterministic fallback."""
    fallback = deterministic_lseg_query(packet)
    if not allow_llm or _env_bool("ARENA_OFFLINE", False) or _env_bool("ARENA_DISABLE_GPT", False):
        return fallback
    if not _env_bool("ARENA_LSEG_LLM_QUERY_ENABLED", True):
        return fallback
    if deadline_at is not None and deadline_at - time.monotonic() < _min_query_seconds():
        fallback["errors"] = ["deadline_budget_before_lseg_query_gpt"]
        return fallback
    model = cfg.model
    if resolve_api_key(model)[0] is None:
        fallback["errors"] = [f"missing_api_key:{model.api_key_env}"]
        return fallback

    messages = lseg_query_messages(packet, fallback["query"])
    p_hash = prompt_hash(messages, model.model)
    cache_path = _cache_path(cfg, model, p_hash)
    cached = _read_cache(cache_path)
    if cached:
        cached["cache_hit"] = True
        return _validated_plan(cached, fallback)
    try:
        payload, call_log = call_openrouter_json(
            model=model,
            messages=messages,
            budget=cfg.budget,
            cache_key="lseg_news_query",
            timeout_seconds=_query_timeout_seconds(deadline_at),
        )
    except Exception as exc:  # noqa: BLE001 - evidence query planning is optional.
        fallback["errors"] = [f"lseg_query_gpt:{type(exc).__name__}:{exc}"]
        return fallback
    plan = _validated_plan(payload, fallback)
    plan["api_log"] = call_log.to_dict()
    _write_cache(cache_path, plan)
    return plan


def deterministic_lseg_query(packet: ArenaForecastPacket) -> dict[str, Any]:
    category = packet.category
    text = " ".join(
        str(value or "")
        for value in (packet.title, packet.subtitle, packet.description, packet.context, " ".join(packet.outcomes))
    )
    rics = _extract_rics(text)
    phrases = _salient_phrases(packet)
    tokens = _salient_tokens(text)
    entity_expr = _or_expr(phrases[:4])
    signal_expr = _or_expr([token for token in tokens if token.lower() not in {p.lower() for p in phrases}][:8])
    if entity_expr and signal_expr:
        keyword_expr = f"({entity_expr}) AND ({signal_expr})"
    else:
        keyword_expr = entity_expr or signal_expr or '"news"'

    filters = ["Language:LEN"]
    category_strategy = "keyword_fallback"
    if category in {"Economics", "Financials"}:
        filters.extend(["Source:RTRS", "Topic:SIGNWS"])
        category_strategy = "macro_professional_news"
    elif category in {"Crypto", "Commodities"}:
        filters.extend(["Source:RTRS", "Topic:SIGNWS"])
        category_strategy = "market_moving_asset_news"
    elif category in {"Politics", "Elections"}:
        filters.extend(["Source:RTRS", "Topic:SIGNWS"])
        category_strategy = "political_significant_news"
    elif category in {"Sports"}:
        category_strategy = "sports_entity_availability_news"
    elif category in {"Entertainment", "Culture"} or "reality" in text.lower():
        category_strategy = "entertainment_people_show_news"
    elif category in {"Weather", "Climate and Weather"}:
        filters.append("Topic:SIGNWS")
        category_strategy = "weather_climate_significant_news"
    elif rics:
        filters.append("Source:RTRS")
        category_strategy = "ric_professional_news"

    if rics and category not in {"Sports", "Entertainment", "Culture"}:
        base = _or_expr([f"R:{ric}" for ric in rics[:4]])
    else:
        base = keyword_expr
    query = _clean_query(" AND ".join([f"({base})", *filters]))
    return {
        "query": query,
        "alternate_queries": [_clean_query(f"({keyword_expr}) AND Language:LEN")],
        "category_strategy": category_strategy,
        "entities": phrases[:8],
        "confidence": 0.45,
        "risks": ["deterministic fallback may miss LSEG topic/RIC syntax"],
        "source": "deterministic_lseg_query",
    }


def lseg_query_messages(packet: ArenaForecastPacket, fallback_query: str) -> list[dict[str, str]]:
    system = (PROMPT_ROOT / "lseg_news_query_v1_system.txt").read_text(encoding="utf-8")
    payload = {
        "event": {
            "as_of": packet.as_of,
            "title": packet.title,
            "subtitle": packet.subtitle,
            "description": packet.description,
            "context": packet.context,
            "category": packet.category,
            "rules": packet.rules,
            "close_time": packet.close_time,
            "outcomes": packet.outcomes,
            "extracted_entities": packet.extracted_entities,
        },
        "fallback_query": fallback_query,
        "documentation_summary": {
            "api": "lseg.data news.get_headlines(query, count, start, end)",
            "runtime_sets_dates": True,
            "safe_filters": ["Language:LEN", "Source:RTRS", "Topic:SIGNWS", "Topic:NEWS1", "Topic:TOPALL"],
            "ric_example": "R:MSFT.O",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"), sort_keys=True)},
    ]


def _validated_plan(payload: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    query = _clean_query(str(payload.get("query") or ""))
    if not query:
        plan = dict(fallback)
        plan.setdefault("errors", []).append("empty_lseg_query")
        return plan
    alternates = [
        _clean_query(str(item))
        for item in (payload.get("alternate_queries") or [])
        if _clean_query(str(item))
    ][:3]
    fallback_query = str(fallback.get("query") or "")
    if fallback_query and fallback_query != query and fallback_query not in alternates:
        alternates.append(fallback_query)
    return {
        "query": query,
        "alternate_queries": alternates or fallback.get("alternate_queries", []),
        "category_strategy": str(payload.get("category_strategy") or fallback.get("category_strategy") or "")[:240],
        "entities": [str(item)[:80] for item in (payload.get("entities") or fallback.get("entities") or [])][:10],
        "confidence": _bounded(payload.get("confidence"), fallback.get("confidence", 0.45)),
        "risks": [str(item)[:160] for item in (payload.get("risks") or fallback.get("risks") or [])][:6],
        "source": "gpt_lseg_query",
    }


def _clean_query(query: str) -> str:
    query = re.sub(r"\s+", " ", query).strip()
    query = query.replace("\n", " ").replace("\r", " ")
    query = "".join(ch for ch in query if ch.isprintable())
    return query[:240]


def _salient_phrases(packet: ArenaForecastPacket) -> list[str]:
    candidates: list[str] = []
    for value in [*packet.outcomes, packet.subtitle or ""]:
        text = str(value or "").strip()
        if not text or text.upper() in {"YES", "NO"}:
            continue
        if len(text) <= 64:
            candidates.append(text)
    full_text = " ".join(str(value or "") for value in (packet.title, packet.subtitle, packet.description, packet.context))
    candidates.extend(_capitalized_entities(full_text))
    tokens = packet.extracted_entities.get("tokens") if isinstance(packet.extracted_entities, dict) else []
    for token in tokens or []:
        if isinstance(token, str) and len(token) >= 3 and token.upper() == token:
            candidates.append(token)
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = item.lower()
        if normalized in seen:
            continue
        out.append(item)
        seen.add(normalized)
    return out


def _salient_tokens(text: str) -> list[str]:
    stop = {
        "will", "what", "when", "where", "which", "who", "win", "wins", "yes", "no",
        "the", "this", "that", "with", "from", "over", "under", "above", "below",
        "market", "predict", "resolve", "resolves", "official", "after", "before",
        "major", "current", "professional", "news", "reports", "report", "there", "are",
        "assess", "whether", "imminent", "tonight", "today", "tomorrow", "use",
    }
    tokens = []
    for raw in TOKEN_RE.findall(text):
        tok = raw.strip(".,?!;:")
        if len(tok) > 2 and tok.lower() not in stop:
            tokens.append(tok)
    return list(dict.fromkeys(tokens))


def _capitalized_entities(text: str) -> list[str]:
    entities = re.findall(r"\b(?:[A-Z][A-Za-z0-9&.$'-]+)(?:\s+[A-Z][A-Za-z0-9&.$'-]+){0,3}\b", text)
    stop = {"Will", "Who", "What", "When", "Where", "Use", "The", "A", "An", "Yes", "No"}
    out = []
    for entity in entities:
        parts = entity.split()
        if parts and parts[0] in stop:
            entity = " ".join(parts[1:])
        entity = entity.strip()
        if len(entity) >= 2 and entity not in stop:
            out.append(entity)
    return out


def _extract_rics(text: str) -> list[str]:
    return list(dict.fromkeys(RIC_RE.findall(text.upper())))


def _or_expr(items: list[str]) -> str:
    cleaned = []
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        if value.startswith("R:") or value.startswith("Topic:") or value.startswith("Source:") or value.startswith("Language:"):
            cleaned.append(value)
        elif " " in value or len(value) > 20:
            cleaned.append(f'"{value[:80]}"')
        else:
            cleaned.append(value[:80])
    if not cleaned:
        return ""
    return " OR ".join(cleaned[:8])


def _cache_path(cfg: ForecastConfig, model: ModelConfig, p_hash: str) -> Path:
    cache_dir = Path(cfg.budget.log_dir) / "llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"lseg_query_v2_{model.model.replace('/', '_')}_{p_hash}.json"


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_cache(path: Path, plan: dict[str, Any]) -> None:
    path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")


def _query_timeout_seconds(deadline_at: float | None) -> float:
    configured = float(os.environ.get("ARENA_LSEG_QUERY_TIMEOUT_SECONDS", "12"))
    if deadline_at is None:
        return max(0.1, configured)
    return max(0.1, min(configured, deadline_at - time.monotonic()))


def _min_query_seconds() -> float:
    return float(os.environ.get("ARENA_LSEG_QUERY_MIN_SECONDS", "4"))


def _bounded(value: Any, default: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = default
    return max(0.0, min(1.0, x))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
