"""Direct xAI Grok adapter.

xAI's API is OpenAI-compatible at https://api.x.ai/v1, so the body shape
mirrors `openrouter.py`. Use this lane to bypass the OpenRouter markup when
running Grok 4.x at volume.
"""

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


XAI_CHAT_COMPLETIONS = "https://api.x.ai/v1/chat/completions"


def _api_key(config: ForecasterConfig) -> str:
    env_name = config.api_key_env or "XAI_API_KEY"
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
    # Grok 4.x reasoning models reason automatically; `reasoning_effort` is a
    # top-level field on xAI's API where supported. Pass it through if set.
    if config.reasoning_effort:
        payload["reasoning_effort"] = config.reasoning_effort

    resp = requests.post(
        XAI_CHAT_COMPLETIONS,
        headers={
            "Authorization": f"Bearer {_api_key(config)}",
            "Content-Type": "application/json",
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
        provider="xai",
        model_id=config.model,
        packet=packet,
        response=parsed,
        raw_response=raw,
    )
