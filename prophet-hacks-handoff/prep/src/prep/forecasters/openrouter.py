"""OpenRouter model adapter for teammates using routed provider access."""

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


OPENROUTER_CHAT_COMPLETIONS = "https://openrouter.ai/api/v1/chat/completions"


def _api_key(config: ForecasterConfig) -> str:
    env_name = config.api_key_env or "OPENROUTER_API_KEY"
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(f"{env_name} is not set")
    return key


def forecast(config: ForecasterConfig, packet: MarketPacket):
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(packet)},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"},
    }
    if config.reasoning_effort:
        payload["extra_body"] = {"reasoning": {"effort": config.reasoning_effort}}

    resp = requests.post(
        OPENROUTER_CHAT_COMPLETIONS,
        headers={
            "Authorization": f"Bearer {_api_key(config)}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/prophet-hacks",
            "X-Title": "ProphetHacks Ensemble",
        },
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    raw = resp.json()
    text = raw["choices"][0]["message"].get("content", "")
    parsed = extract_json_object(text)
    parsed["prompt_hash"] = stable_prompt_hash(packet, config)
    return forecast_from_response(
        provider="openrouter",
        model_id=config.model,
        packet=packet,
        response=parsed,
        raw_response=raw,
    )
