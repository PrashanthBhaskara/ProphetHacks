"""Optional Platt-shrinkage calibration on top of market mid-price.

Loads `calibration.json` from this directory if present. Schema:

    {"a": 1.0, "b": 0.0}        # global Platt: p' = sigmoid(a*logit(p) + b)

If the file is absent or malformed, predictions pass through unchanged
(raw market mid-price). Memory says every calibration tested is within
±0.02 Brier of market on Sports-heavy 2026, so passthrough is a fine
floor — Platt only kicks in if we ship a fitted file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_CALIBRATION_PATH = Path(__file__).parent / "calibration.json"

_PARAMS: dict | None = None


def _load() -> dict | None:
    global _PARAMS
    if _PARAMS is not None:
        return _PARAMS
    if not _CALIBRATION_PATH.exists():
        _PARAMS = {}
        return _PARAMS
    try:
        _PARAMS = json.loads(_CALIBRATION_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        _PARAMS = {}
    return _PARAMS


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


def apply(p: float) -> float:
    """Apply global Platt shrinkage if calibration.json is present, else passthrough."""
    params = _load() or {}
    a = params.get("a")
    b = params.get("b")
    if a is None or b is None:
        return p
    return _sigmoid(a * _logit(p) + b)


def info() -> dict:
    """Return current calibration state for the /healthz response."""
    params = _load() or {}
    if "a" in params and "b" in params:
        return {"mode": "platt", "a": params["a"], "b": params["b"]}
    return {"mode": "passthrough"}
