"""Direct Gemini API forecaster."""

from __future__ import annotations

import os

import requests

from .base import (
    SYSTEM_PROMPT,
    ForecasterConfig,
    build_user_prompt,
    extract_json_object,
    forecast_from_response,
    stable_prompt_hash,
)
from prep.schemas import MarketPacket


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _api_key(config: ForecasterConfig) -> str:
    env_name = config.api_key_env or "GEMINI_API_KEY"
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(f"{env_name} is not set")
    return key


def forecast(config: ForecasterConfig, packet: MarketPacket):
    url = GEMINI_ENDPOINT.format(model=config.model)
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": build_user_prompt(packet)}]}],
        "generationConfig": {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(
        url,
        params={"key": _api_key(config)},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()
    parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts)
    parsed = extract_json_object(text)
    parsed["prompt_hash"] = stable_prompt_hash(packet, config)
    return forecast_from_response(
        provider="gemini",
        model_id=config.model,
        packet=packet,
        response=parsed,
        raw_response=raw,
    )
