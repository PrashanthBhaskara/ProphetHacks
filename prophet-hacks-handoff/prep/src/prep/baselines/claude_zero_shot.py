"""Zero-shot Claude baseline — equivalent to ai-prophet's example_agent.

No web search, no market price. Just calls Claude with the event and
asks for a probability. This is the baseline the example agent in
ai-prophet/packages/cli/ai_prophet/forecast/example_agent.py uses.

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import anthropic

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
def _client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


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
    model = os.environ.get("FORECAST_MODEL", "claude-sonnet-4-20250514")
    resp = _client().messages.create(
        model=model,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(event)}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)
    return {
        "p_yes": float(data["p_yes"]),
        "rationale": data.get("rationale", ""),
    }
