"""Zero-shot Grok baseline — sibling of claude_zero_shot.

No web search, no market price. Calls xAI's Grok with the event and asks
for a probability. Uses the OpenAI-compatible endpoint at api.x.ai/v1.

Requires XAI_API_KEY in the environment.
Optional: GROK_MODEL (default: grok-4.3).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

from openai import OpenAI

SYSTEM_PROMPT = """\
You are an expert forecaster specialized in calibrated probability estimation.

Your task is to estimate the probability that the given event resolves YES.

CALIBRATION GUIDELINES:
- Consider base rates for similar events.
- Weight evidence by reliability and recency.
- Account for uncertainty — don't be overconfident.
- Extremes (p < 0.10 or p > 0.90) require very strong evidence.

Respond with ONLY valid JSON: {"p_yes": <float 0.01-0.99>, "rationale": "<2-3 sentences>"}
Do not include any other text outside the JSON object."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://api.x.ai/v1")


def _build_user_prompt(event: dict) -> str:
    parts = [f"Event: {event['title']}"]
    if event.get("subtitle"):
        parts.append(f"Subtitle: {event['subtitle']}")
    if event.get("description"):
        parts.append(f"Description: {event['description']}")
    if event.get("rules"):
        parts.append(f"Rules: {event['rules']}")
    parts.append(f"Category: {event['category']}")
    parts.append(f"Close time: {event['close_time']}")
    parts.append("\nBased on your knowledge, what is the probability this resolves YES?")
    return "\n".join(parts)


def predict(event: dict) -> dict:
    model = os.environ.get("GROK_MODEL", "grok-4.3")
    resp = _client().chat.completions.create(
        model=model,
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(event)},
        ],
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)
    return {
        "p_yes": float(data["p_yes"]),
        "rationale": data.get("rationale", ""),
    }
