"""Direct Gemini API forecaster."""

from __future__ import annotations

import json
import os
import time

import requests

from .base import (
    ForecasterConfig,
    build_user_prompt,
    extract_json_object,
    forecast_from_response,
    resolve_api_key,
    stable_prompt_hash,
    system_prompt_for_config,
)
from prep.schemas import MarketPacket


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GOOGLE_SEARCH_GROUNDING_INSTRUCTION = """

Google Search grounding is required for this live forecast. Use it to check
current, timestamped source-of-truth facts before setting probabilities. Prefer
official sources, primary releases, market pages, league/team sources, filings,
and reputable reporting. Include source names, timestamps, and why each source
was used or excluded in source_audit. If Search returns weak or conflicting
evidence, say so and shrink toward the Kalshi market-implied prior.
"""


class GeminiParseError(RuntimeError):
    """Raised when Gemini returns a response that cannot be parsed as JSON."""


def _api_key(config: ForecasterConfig) -> str:
    key = resolve_api_key(config, "GEMINI_API_KEY")
    if not key:
        raise RuntimeError(f"No API key found for {config.name} (checked {config.api_key_env} and fallbacks)")
    return key


def _post_generate(url: str, config: ForecasterConfig, payload: dict) -> dict:
    retry_statuses = {429, 500, 502, 503, 504}
    for attempt in range(4):
        resp = requests.post(
            url,
            params={"key": _api_key(config)},
            json=payload,
            timeout=120,
        )
        if resp.status_code in retry_statuses and attempt < 3:
            retry_after = resp.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after is not None else 2 ** attempt
            except ValueError:
                delay = 2 ** attempt
            time.sleep(min(delay, 20))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable retry loop exit")


def _response_text(raw: dict) -> str:
    parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def _finish_reason(raw: dict) -> str | None:
    return raw.get("candidates", [{}])[0].get("finishReason")


def _grounding_enabled(config: ForecasterConfig) -> bool:
    return bool(config.enable_google_search or config.require_google_search_grounding)


def _grounded_system_prompt(config: ForecasterConfig) -> str:
    prompt = system_prompt_for_config(config)
    if _grounding_enabled(config):
        return prompt + GOOGLE_SEARCH_GROUNDING_INSTRUCTION
    return prompt


def _grounding_summary(raw: dict, *, enabled: bool, required: bool) -> dict:
    candidate = (raw.get("candidates") or [{}])[0]
    grounding = candidate.get("groundingMetadata") if isinstance(candidate, dict) else None
    if not isinstance(grounding, dict):
        return {
            "enabled": enabled,
            "required": required,
            "present": False,
            "web_search_queries": [],
            "sources": [],
        }
    chunks = grounding.get("groundingChunks") or []
    sources = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        web = chunk.get("web") or {}
        if isinstance(web, dict):
            sources.append({
                "title": web.get("title"),
                "uri": web.get("uri"),
                "domain": web.get("domain"),
            })
    return {
        "enabled": enabled,
        "required": required,
        "present": True,
        "web_search_queries": grounding.get("webSearchQueries") or [],
        "sources": sources[:12],
        "support_count": len(grounding.get("groundingSupports") or []),
    }


def _balanced_object_at(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _salvage_probabilities(text: str, packet: MarketPacket) -> dict | None:
    """Recover a scoreable forecast when JSON was truncated after probabilities."""
    key_index = text.find('"probabilities"')
    if key_index < 0:
        return None
    brace_index = text.find("{", key_index)
    if brace_index < 0:
        return None
    object_text = _balanced_object_at(text, brace_index)
    if object_text is None:
        return None
    try:
        probabilities = json.loads(object_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(probabilities, dict):
        return None
    missing = [outcome for outcome in packet.outcomes if outcome not in probabilities]
    if missing:
        return None
    return {
        "forecast": {
            "probabilities": probabilities,
            "confidence": 0.5,
            "uncertainty": 0.5,
        },
        "reasoning_track": {
            "summary": "Recovered from a truncated Gemini response after a complete probabilities object.",
            "base_rate": "",
            "market_analysis": "",
            "context_market_analysis": "",
            "key_evidence": [],
            "source_audit": [],
            "counterarguments": [],
            "assumptions": ["Reasoning was omitted because the original response exceeded the output budget."],
            "information_gaps": ["Full reasoning track unavailable for this recovered response."],
            "what_would_change_my_mind": [],
        },
        "diagnostics": {
            "evidence_quality": "medium",
            "rules_clarity": "medium",
            "liquidity_quality": "medium",
            "market_disagreement_reason": "Recovered probabilities from truncated JSON.",
            "should_defer_to_market": True,
        },
        "parse_recovery": {
            "method": "probabilities_object_salvage",
        },
    }


def _repair_json_response(
    *,
    url: str,
    config: ForecasterConfig,
    packet: MarketPacket,
    malformed_text: str,
    parse_error: Exception,
) -> tuple[dict, dict]:
    repair_prompt = {
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You repair malformed model outputs into strict JSON. "
                        "Return only one valid JSON object. Do not add markdown or prose. "
                        "Preserve the forecast values and reasoning from the malformed output."
                    ),
                },
            ],
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "The previous forecasting response was not valid JSON.\n"
                            f"JSON parser error: {type(parse_error).__name__}: {parse_error}\n"
                            f"Required outcome labels: {json.dumps(packet.outcomes)}\n\n"
                            "Malformed response:\n"
                            f"{malformed_text}"
                        ),
                    },
                ],
            },
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": config.max_tokens,
            "responseMimeType": "application/json",
        },
    }
    repair_raw = _post_generate(url, config, repair_prompt)
    repair_text = _response_text(repair_raw)
    try:
        return extract_json_object(repair_text), repair_raw
    except Exception as exc:  # noqa: BLE001
        excerpt = repair_text[:800].replace("\n", "\\n")
        raise GeminiParseError(
            "Gemini returned malformed JSON and the repair pass also failed. "
            f"first_error={type(parse_error).__name__}: {parse_error}; "
            f"repair_error={type(exc).__name__}: {exc}; "
            f"finish_reason={_finish_reason(repair_raw)}; repair_excerpt={excerpt!r}"
        ) from exc


def forecast(config: ForecasterConfig, packet: MarketPacket):
    url = GEMINI_ENDPOINT.format(model=config.model)
    grounding_enabled = _grounding_enabled(config)
    system_prompt = _grounded_system_prompt(config)
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": build_user_prompt(packet)}]}],
        "generationConfig": {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
            "responseMimeType": "application/json",
        },
    }
    if grounding_enabled:
        payload["tools"] = [{"google_search": {}}]
    raw = _post_generate(url, config, payload)
    text = _response_text(raw)
    repair_raw = None
    recovered_from_truncation = False
    try:
        parsed = extract_json_object(text)
    except Exception as exc:  # noqa: BLE001
        parsed = _salvage_probabilities(text, packet)
        if parsed is not None:
            recovered_from_truncation = True
        elif _finish_reason(raw) == "MAX_TOKENS":
            excerpt = text[:800].replace("\n", "\\n")
            raise GeminiParseError(
                "Gemini response was truncated before a complete probabilities object was available. "
                f"Increase max_tokens or reduce max_outcomes. excerpt={excerpt!r}"
            ) from exc
        else:
            parsed, repair_raw = _repair_json_response(
                url=url,
                config=config,
                packet=packet,
                malformed_text=text,
                parse_error=exc,
            )
    parsed["prompt_hash"] = stable_prompt_hash(packet, config)
    return forecast_from_response(
        provider="gemini",
        model_id=config.model,
        packet=packet,
        response=parsed,
        raw_response={
            "api_response": raw,
            "repair_api_response": repair_raw,
            "parsed_response": parsed,
            "recovered_from_truncation": recovered_from_truncation,
            "grounding": _grounding_summary(
                raw,
                enabled=grounding_enabled,
                required=config.require_google_search_grounding,
            ),
        },
    )
