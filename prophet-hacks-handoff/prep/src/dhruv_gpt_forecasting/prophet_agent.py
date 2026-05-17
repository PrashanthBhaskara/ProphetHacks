"""Prophet CLI-compatible adapter.

The live production endpoint returns the richer ensemble envelope. Use this
module only for local Prophet CLI scoring, which requires a bare
{"probabilities": [...]} response.
"""

from __future__ import annotations

from typing import Any

from .arena_agent import predict_prophet


def predict(event: dict[str, Any]) -> dict[str, Any]:
    return predict_prophet(event)
