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


def _is_quota_or_credit_response(resp: requests.Response) -> bool:
    if resp.status_code not in {402, 403, 429}:
        return False
    text = resp.text.lower()
    quota_terms = (
        "quota",
        "credit",
        "billing",
        "resource_exhausted",
        "rate limit",
        "rate_limit",
        "insufficient",
    )
    return any(term in text for term in quota_terms)


def _is_quota_or_credit_error(exc: requests.HTTPError) -> bool:
    response = exc.response
    return response is not None and _is_quota_or_credit_response(response)


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
        if _is_quota_or_credit_response(resp):
            resp.raise_for_status()
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


def _grounded_search_brief_prompt(packet: MarketPacket) -> str:
    return (
        "Use Google Search grounding for this live prediction-market forecast. "
        "Find current, source-of-truth facts that can move the exact listed outcome probabilities. "
        "Focus on official/primary sources, market pages, reputable reporting, league/team sources, filings, "
        "or live price data as relevant. Return a concise source brief with source names, timestamps when visible, "
        "and the probability relevance. Do not return final probabilities.\n\n"
        f"MARKET_PACKET:\n{json.dumps(packet.to_dict(), indent=2, sort_keys=True)}"
    )


def _grounded_user_prompt(
    packet: MarketPacket,
    *,
    retry: bool = False,
    grounding_brief: str | None = None,
) -> str:
    instruction = (
        "MANDATORY: Use the Google Search grounding tool before producing the forecast JSON. "
        "The response must include grounding metadata from the tool call. Search targeted, current facts "
        "that can move the exact listed probabilities, then return only the required JSON object.\n\n"
    )
    if retry:
        instruction = (
            "The previous Gemini response did not include Google Search grounding metadata. "
            "You must use the Google Search grounding tool now before returning JSON. "
            "Search targeted, current facts that can move the exact listed probabilities. "
            "Return only the required JSON object after using the tool.\n\n"
        )
    if grounding_brief:
        instruction += (
            "GEMINI_PRO_GROUNDED_SEARCH_BRIEF:\n"
            f"{grounding_brief[:6000]}\n\n"
        )
    return instruction + build_user_prompt(packet)


def _raw_has_grounding(raw: dict) -> bool:
    candidate = (raw.get("candidates") or [{}])[0]
    grounding = candidate.get("groundingMetadata") if isinstance(candidate, dict) else None
    return isinstance(grounding, dict)


def _market_mirror_response(config: ForecasterConfig, packet: MarketPacket, reason: str) -> dict:
    outcomes = packet.outcomes or ["YES", "NO"]
    if packet.is_binary:
        mid = 0.5
        try:
            mid = float(packet.kalshi.market_mid)
        except (TypeError, ValueError, AttributeError):
            mid = 0.5
        probabilities = {outcomes[0]: mid, outcomes[1]: 1.0 - mid}
    else:
        market_probs = packet.retrieval.get("market_implied_probabilities")
        if isinstance(market_probs, dict):
            share = 1.0 / max(1, len(outcomes))
            probabilities = {}
            case_map = {str(key).casefold(): key for key in market_probs}
            for outcome in outcomes:
                key = outcome if outcome in market_probs else case_map.get(outcome.casefold())
                try:
                    probabilities[outcome] = float(market_probs.get(key, share)) if key is not None else share
                except (TypeError, ValueError):
                    probabilities[outcome] = share
        else:
            share = 1.0 / max(1, len(outcomes))
            probabilities = {outcome: share for outcome in outcomes}

    return {
        "forecast": {
            "probabilities": probabilities,
            "confidence": 0.30,
            "uncertainty": 0.80,
        },
        "reasoning_track": {
            "summary": f"{config.name} could not run because {reason}; mirroring the current Kalshi market instead.",
            "base_rate": "",
            "market_analysis": "Gemini lane fallback: deferring to Kalshi market-implied probabilities.",
            "context_market_analysis": "",
            "key_evidence": [],
            "source_audit": [
                {
                    "source": "gemini_api",
                    "source_timestamp": packet.as_of,
                    "cutoff_check": "No Gemini forecast was produced; this lane only mirrors market data already in the packet.",
                    "used": False,
                    "reason": reason,
                }
            ],
            "counterarguments": [],
            "assumptions": ["Fallback lane output is not an independent forecast."],
            "information_gaps": ["Gemini Pro did not provide grounded reasoning for this run."],
            "what_would_change_my_mind": [],
        },
        "diagnostics": {
            "evidence_quality": "low",
            "rules_clarity": "medium",
            "liquidity_quality": "medium",
            "market_disagreement_reason": f"{config.name} fallback: {reason}.",
            "should_defer_to_market": True,
        },
        "gemini_market_fallback": {
            "reason": reason,
            "should_defer_to_market": True,
        },
    }


def _grounding_summary(
    raw: dict,
    *,
    enabled: bool,
    required: bool,
    context_raw: dict | None = None,
) -> dict:
    candidate = (raw.get("candidates") or [{}])[0]
    grounding = candidate.get("groundingMetadata") if isinstance(candidate, dict) else None
    source = "final_forecast"
    if not isinstance(grounding, dict) and context_raw is not None:
        context_candidate = (context_raw.get("candidates") or [{}])[0]
        grounding = context_candidate.get("groundingMetadata") if isinstance(context_candidate, dict) else None
        source = "pre_forecast_grounded_search_brief"
    if not isinstance(grounding, dict):
        return {
            "enabled": enabled,
            "required": required,
            "present": False,
            "source": None,
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
        "source": source,
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
    grounding_brief_raw = None
    grounding_brief = None
    if grounding_enabled:
        brief_payload = {
            "contents": [{"role": "user", "parts": [{"text": _grounded_search_brief_prompt(packet)}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": min(1024, max(512, config.max_tokens)),
            },
            "tools": [{"google_search": {}}],
        }
        try:
            grounding_brief_raw = _post_generate(url, config, brief_payload)
            grounding_brief = _response_text(grounding_brief_raw)
        except requests.HTTPError as exc:
            if not _is_quota_or_credit_error(exc):
                raise
            reason = f"gemini_quota_or_credit_exhausted:{exc.response.status_code if exc.response is not None else 'unknown'}"
            parsed = _market_mirror_response(config, packet, reason)
            parsed["prompt_hash"] = stable_prompt_hash(packet, config)
            return forecast_from_response(
                provider="gemini",
                model_id=config.model,
                packet=packet,
                response=parsed,
                raw_response={
                    "parsed_response": parsed,
                    "gemini_market_fallback": parsed["gemini_market_fallback"],
                    "grounding": {
                        "enabled": grounding_enabled,
                        "required": config.require_google_search_grounding,
                        "present": False,
                        "source": None,
                        "web_search_queries": [],
                        "sources": [],
                    },
                },
            )
        if config.require_google_search_grounding and not _raw_has_grounding(grounding_brief_raw):
            reason = "required_google_search_grounding_missing"
            parsed = _market_mirror_response(config, packet, reason)
            parsed["prompt_hash"] = stable_prompt_hash(packet, config)
            return forecast_from_response(
                provider="gemini",
                model_id=config.model,
                packet=packet,
                response=parsed,
                raw_response={
                    "api_response": grounding_brief_raw,
                    "parsed_response": parsed,
                    "gemini_market_fallback": parsed["gemini_market_fallback"],
                    "grounding": _grounding_summary(
                        {},
                        enabled=grounding_enabled,
                        required=config.require_google_search_grounding,
                        context_raw=grounding_brief_raw,
                    ),
                },
            )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            _grounded_user_prompt(packet, grounding_brief=grounding_brief)
                            if grounding_enabled else build_user_prompt(packet)
                        ),
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
        },
    }
    if grounding_enabled:
        payload["tools"] = [{"google_search": {}}]
    else:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    initial_raw = None
    grounding_retry_attempted = False
    try:
        raw = _post_generate(url, config, payload)
    except requests.HTTPError as exc:
        if not _is_quota_or_credit_error(exc):
            raise
        reason = f"gemini_quota_or_credit_exhausted:{exc.response.status_code if exc.response is not None else 'unknown'}"
        parsed = _market_mirror_response(config, packet, reason)
        parsed["prompt_hash"] = stable_prompt_hash(packet, config)
        return forecast_from_response(
            provider="gemini",
            model_id=config.model,
            packet=packet,
            response=parsed,
            raw_response={
                "parsed_response": parsed,
                "gemini_market_fallback": parsed["gemini_market_fallback"],
                "grounding": {
                    "enabled": grounding_enabled,
                    "required": config.require_google_search_grounding,
                    "present": False,
                    "web_search_queries": [],
                    "sources": [],
                },
            },
        )
    if grounding_enabled and config.require_google_search_grounding and not _raw_has_grounding(raw):
        if _raw_has_grounding(grounding_brief_raw or {}):
            grounding_retry_attempted = False
        else:
            grounding_retry_attempted = True
    if (
        grounding_enabled
        and config.require_google_search_grounding
        and not _raw_has_grounding(raw)
        and not _raw_has_grounding(grounding_brief_raw or {})
    ):
        initial_raw = raw
        retry_payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": _grounded_user_prompt(
                                packet,
                                retry=True,
                                grounding_brief=grounding_brief,
                            ),
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens,
            },
            "tools": [{"google_search": {}}],
        }
        try:
            raw = _post_generate(url, config, retry_payload)
        except requests.HTTPError as exc:
            if not _is_quota_or_credit_error(exc):
                raise
            reason = f"gemini_quota_or_credit_exhausted:{exc.response.status_code if exc.response is not None else 'unknown'}"
            parsed = _market_mirror_response(config, packet, reason)
            parsed["prompt_hash"] = stable_prompt_hash(packet, config)
            return forecast_from_response(
                provider="gemini",
                model_id=config.model,
                packet=packet,
                response=parsed,
                raw_response={
                    "api_response": initial_raw,
                    "parsed_response": parsed,
                    "gemini_market_fallback": parsed["gemini_market_fallback"],
                    "grounding_retry_attempted": grounding_retry_attempted,
                    "grounding": {
                        "enabled": grounding_enabled,
                        "required": config.require_google_search_grounding,
                        "present": False,
                        "web_search_queries": [],
                        "sources": [],
                    },
                },
            )
        if not _raw_has_grounding(raw):
            reason = "required_google_search_grounding_missing"
            parsed = _market_mirror_response(config, packet, reason)
            parsed["prompt_hash"] = stable_prompt_hash(packet, config)
            return forecast_from_response(
                provider="gemini",
                model_id=config.model,
                packet=packet,
                response=parsed,
                raw_response={
                    "api_response": raw,
                    "initial_api_response": initial_raw,
                    "grounding_brief_api_response": grounding_brief_raw,
                    "parsed_response": parsed,
                    "gemini_market_fallback": parsed["gemini_market_fallback"],
                    "grounding_retry_attempted": grounding_retry_attempted,
                    "grounding": {
                        "enabled": grounding_enabled,
                        "required": config.require_google_search_grounding,
                        "present": False,
                        "web_search_queries": [],
                        "sources": [],
                    },
                },
            )
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
            "initial_api_response": initial_raw,
            "grounding_brief_api_response": grounding_brief_raw,
            "repair_api_response": repair_raw,
            "parsed_response": parsed,
            "recovered_from_truncation": recovered_from_truncation,
            "grounding_retry_attempted": grounding_retry_attempted,
            "grounding": _grounding_summary(
                raw,
                enabled=grounding_enabled,
                required=config.require_google_search_grounding,
                context_raw=grounding_brief_raw,
            ),
        },
    )
