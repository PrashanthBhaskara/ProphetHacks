"""Direct Gemini API forecaster."""

from __future__ import annotations

import json
import os

import requests

from .base import (
    ForecasterConfig,
    build_user_prompt,
    extract_json_object,
    forecast_from_response,
    stable_prompt_hash,
    system_prompt_for_config,
)
from prep.schemas import MarketPacket


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiParseError(RuntimeError):
    """Raised when Gemini returns a response that cannot be parsed as JSON."""


def _api_key(config: ForecasterConfig) -> str:
    env_name = config.api_key_env or "GEMINI_API_KEY"
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(f"{env_name} is not set")
    return key


def _post_generate(url: str, config: ForecasterConfig, payload: dict) -> dict:
    resp = requests.post(
        url,
        params={"key": _api_key(config)},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _response_text(raw: dict) -> str:
    parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def _finish_reason(raw: dict) -> str | None:
    return raw.get("candidates", [{}])[0].get("finishReason")


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
    system_prompt = system_prompt_for_config(config)
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": build_user_prompt(packet)}]}],
        "generationConfig": {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
            "responseMimeType": "application/json",
        },
    }
    if config.enable_google_search:
        payload["tools"] = [{"google_search": {}}]
    raw = _post_generate(url, config, payload)
    text = _response_text(raw)
    repair_raw = None
    try:
        parsed = extract_json_object(text)
    except Exception as exc:  # noqa: BLE001
        if _finish_reason(raw) == "MAX_TOKENS":
            excerpt = text[:800].replace("\n", "\\n")
            raise GeminiParseError(
                "Gemini response was truncated before valid JSON was complete. "
                f"Increase max_tokens. excerpt={excerpt!r}"
            ) from exc
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
        },
    )
