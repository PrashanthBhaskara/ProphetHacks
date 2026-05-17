"""LLM clients with local JSON validation and cost logging."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from .config import BudgetConfig, ModelConfig, resolve_api_key
from .key_utils import key_fingerprint
from .prompts import prompt_hash
from .schemas import ApiCallLog


OPENROUTER_CHAT_COMPLETIONS = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_AUTH_URL = "https://openrouter.ai/api/v1/auth/key"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class OpenRouterError(RuntimeError):
    pass


def call_openrouter_json(
    *,
    model: ModelConfig,
    messages: list[dict[str, str]],
    budget: BudgetConfig,
    cache_key: str | None = None,
    timeout_seconds: float | None = None,
    search_grounding: bool = False,
) -> tuple[dict[str, Any], ApiCallLog]:
    """Call OpenRouter and return a parsed JSON object plus call metadata."""
    if model.provider == "gemini":
        return call_gemini_json(
            model=model,
            messages=messages,
            budget=budget,
            cache_key=cache_key,
            timeout_seconds=timeout_seconds,
            search_grounding=search_grounding,
        )
    if model.provider != "openrouter":
        raise OpenRouterError(f"Unsupported model provider: {model.provider}")
    key, key_env = resolve_api_key(model)
    if not key:
        raise OpenRouterError(f"No API key set for {model.api_key_env} or fallbacks")
    p_hash = prompt_hash(messages, model.model)
    tools = _openrouter_tools(model, search_grounding=search_grounding)
    payload: dict[str, Any] = {
        "model": model.model,
        "messages": messages,
        "temperature": model.temperature,
        "max_tokens": model.max_tokens,
        "response_format": {"type": "json_object"},
    }
    if tools:
        payload["tools"] = tools
    if cache_key:
        payload["prompt_cache_key"] = cache_key
    t0 = time.time()
    if timeout_seconds is None:
        timeout_seconds = float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "90"))
    response = requests.post(
        OPENROUTER_CHAT_COMPLETIONS,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/prophet-hacks",
            "X-Title": "Dhruv GPT Arena Forecasting Agent",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    latency = time.time() - t0
    response.raise_for_status()
    raw = response.json()
    message = raw["choices"][0]["message"]
    text = message.get("content") or ""
    if not text.strip():
        raise OpenRouterError("OpenRouter returned empty message content")
    parsed = extract_json_object(text)
    annotations = message.get("annotations") or []
    usage = raw.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    call_log = ApiCallLog(
        provider="openrouter",
        model=model.model,
        prompt_hash=p_hash,
        api_key_env=key_env,
        api_key_fingerprint=key_fingerprint(key),
        latency_sec=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost(model.model, input_tokens, output_tokens, budget),
        cache_key=cache_key,
        fallback_path=None if key_env == model.api_key_env else key_env,
        search_grounding_enabled=bool(tools),
        search_grounding_engine=model.search_grounding_engine if tools else None,
        response_annotation_count=len(annotations) if isinstance(annotations, list) else 0,
        provider_response_id=raw.get("id"),
    )
    return parsed, call_log


def call_gemini_json(
    *,
    model: ModelConfig,
    messages: list[dict[str, str]],
    budget: BudgetConfig,
    cache_key: str | None = None,
    timeout_seconds: float | None = None,
    search_grounding: bool = False,
) -> tuple[dict[str, Any], ApiCallLog]:
    """Call the direct Gemini API and return parsed JSON plus call metadata."""
    key, key_env = resolve_api_key(model)
    if not key:
        raise OpenRouterError(f"No Gemini API key set for {model.api_key_env} or fallbacks")
    p_hash = prompt_hash(messages, model.model)
    tools = _gemini_tools(model, search_grounding=search_grounding)
    payload = _gemini_payload(model, messages, tools=tools)
    t0 = time.time()
    if timeout_seconds is None:
        timeout_seconds = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "90")))
    response = requests.post(
        f"{GEMINI_MODELS_URL}/{model.model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json=payload,
        timeout=timeout_seconds,
    )
    latency = time.time() - t0
    response.raise_for_status()
    raw = response.json()
    candidate = (raw.get("candidates") or [{}])[0]
    text = _gemini_text(candidate)
    if not text.strip():
        raise OpenRouterError("Gemini returned empty message content")
    json_repaired = False
    repair_usage: dict[str, Any] = {}
    repair_response_id = None
    try:
        parsed = extract_json_object(text)
    except Exception as exc:
        if not tools:
            raise
        parsed, repair_usage, repair_response_id = _repair_gemini_json(
            key=key,
            model=model,
            malformed_text=text,
            timeout_seconds=timeout_seconds,
            parse_error=exc,
        )
        json_repaired = True
    usage = raw.get("usageMetadata") or {}
    input_tokens = _sum_optional_ints(usage.get("promptTokenCount"), repair_usage.get("promptTokenCount"))
    output_tokens = _sum_optional_ints(usage.get("candidatesTokenCount"), repair_usage.get("candidatesTokenCount"))
    grounding = candidate.get("groundingMetadata") if isinstance(candidate, dict) else None
    annotations = _gemini_grounding_count(grounding)
    call_log = ApiCallLog(
        provider="gemini",
        model=model.model,
        prompt_hash=p_hash,
        api_key_env=key_env,
        api_key_fingerprint=key_fingerprint(key),
        latency_sec=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost(model.model, input_tokens, output_tokens, budget),
        cache_key=cache_key,
        fallback_path="gemini_json_repair_after_parse_error" if json_repaired else (None if key_env == model.api_key_env else key_env),
        search_grounding_enabled=bool(tools),
        search_grounding_engine="google_search" if tools else None,
        response_annotation_count=annotations,
        provider_response_id=",".join(
            str(value)
            for value in (raw.get("responseId"), repair_response_id)
            if value
        ) or None,
    )
    return parsed, call_log


def _openrouter_tools(model: ModelConfig, *, search_grounding: bool) -> list[dict[str, Any]]:
    if not search_grounding or not model.native_search_grounding_enabled:
        return []
    parameters: dict[str, Any] = {
        "engine": os.environ.get("OPENROUTER_SEARCH_GROUNDING_ENGINE", model.search_grounding_engine),
        "max_results": int(os.environ.get(
            "OPENROUTER_SEARCH_GROUNDING_MAX_RESULTS",
            model.search_grounding_max_results,
        )),
        "max_total_results": int(os.environ.get(
            "OPENROUTER_SEARCH_GROUNDING_MAX_TOTAL_RESULTS",
            model.search_grounding_max_total_results,
        )),
    }
    context_size = os.environ.get(
        "OPENROUTER_SEARCH_GROUNDING_CONTEXT_SIZE",
        model.search_grounding_context_size or "",
    ).strip()
    if context_size:
        parameters["search_context_size"] = context_size
    return [{"type": "openrouter:web_search", "parameters": parameters}]


def _gemini_tools(model: ModelConfig, *, search_grounding: bool) -> list[dict[str, Any]]:
    if not search_grounding or not model.native_search_grounding_enabled:
        return []
    return [{"google_search": {}}]


def _repair_gemini_json(
    *,
    key: str,
    model: ModelConfig,
    malformed_text: str,
    timeout_seconds: float,
    parse_error: Exception,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    repair_payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": (
                    "Repair the following malformed JSON-like response into one valid JSON object. "
                    "Do not add new factual claims. Preserve fields and values that are clearly present. "
                    "If a list or field is incomplete, return the salvageable prefix or an empty value. "
                    "Return JSON only.\n\n"
                    f"Parse error: {type(parse_error).__name__}: {parse_error}\n\n"
                    f"Malformed response:\n{malformed_text[:12000]}"
                )
            }],
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max(1024, min(4096, int(os.environ.get("GEMINI_JSON_REPAIR_MAX_TOKENS", "2048")))),
            "responseMimeType": "application/json",
        },
    }
    response = requests.post(
        f"{GEMINI_MODELS_URL}/{model.model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json=repair_payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    raw = response.json()
    candidate = (raw.get("candidates") or [{}])[0]
    text = _gemini_text(candidate)
    if not text.strip():
        return _minimal_unusable_grounded_payload("empty_repair_response"), raw.get("usageMetadata") or {}, raw.get("responseId")
    try:
        return extract_json_object(text), raw.get("usageMetadata") or {}, raw.get("responseId")
    except Exception as exc:  # noqa: BLE001 - caller still needs a valid evidence object.
        return (
            _minimal_unusable_grounded_payload(f"repair_parse_failed:{type(exc).__name__}:{exc}"),
            raw.get("usageMetadata") or {},
            raw.get("responseId"),
        )


def _minimal_unusable_grounded_payload(reason: str) -> dict[str, Any]:
    return {
        "targeted_questions": [],
        "macroeconomic_drivers": [],
        "breaking_news": [],
        "qualitative_sentiment": [],
        "contract_specific_factors": [],
        "source_notes": [],
        "excluded_sources": [{"source": "gemini_native_search", "reason": "malformed_json"}],
        "evidence_quality": {
            "overall": 0.0,
            "freshness": 0.0,
            "source_quality": 0.0,
            "event_match": 0.0,
            "conflict_level": 0.0,
        },
        "information_gaps": [f"Grounded search response could not be converted to usable PIT evidence: {reason}"[:300]],
        "summary": "",
    }


def _gemini_payload(
    model: ModelConfig,
    messages: list[dict[str, str]],
    *,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role") or "user"
        content = str(message.get("content") or "")
        if role == "system":
            system_parts.append({"text": content})
        else:
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": content}],
            })
    if not contents:
        contents.append({"role": "user", "parts": [{"text": "Return valid JSON."}]})
    generation_config: dict[str, Any] = {
        "temperature": model.temperature,
        "maxOutputTokens": model.max_tokens,
    }
    # Gemini currently does not support JSON response MIME together with
    # google_search, so grounded calls rely on prompt + local JSON extraction.
    if not tools:
        generation_config["responseMimeType"] = "application/json"
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    if tools:
        payload["tools"] = tools
    return payload


def _sum_optional_ints(*values: Any) -> int | None:
    total = 0
    seen = False
    for value in values:
        if value is None:
            continue
        try:
            total += int(value)
            seen = True
        except (TypeError, ValueError):
            continue
    return total if seen else None


def _gemini_text(candidate: dict[str, Any]) -> str:
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return ""
    return "\n".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))


def _gemini_grounding_count(grounding: Any) -> int:
    if not isinstance(grounding, dict):
        return 0
    chunks = grounding.get("groundingChunks")
    if isinstance(chunks, list):
        return len(chunks)
    supports = grounding.get("groundingSupports")
    if isinstance(supports, list):
        return len(supports)
    queries = grounding.get("webSearchQueries")
    return len(queries) if isinstance(queries, list) else 0


def extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def estimate_cost(
    model_id: str,
    input_tokens: int | None,
    output_tokens: int | None,
    budget: BudgetConfig,
) -> float | None:
    prices = budget.estimated_prices_per_1m_tokens.get(model_id)
    if not prices or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens / 1_000_000.0) * float(prices.get("input", 0.0)) + (
        output_tokens / 1_000_000.0
    ) * float(prices.get("output", 0.0))
