"""FastAPI wrapper for the Prophet Arena forecasting endpoint."""

from __future__ import annotations

from typing import Any

from .arena_agent import forecast_arena_payload_for_ensemble
from .preflight import run_preflight

try:
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover - import-time deployment hint.
    raise RuntimeError("Install fastapi and uvicorn to serve dhruv_gpt_forecasting.server:app") from exc


app = FastAPI(title="Dhruv GPT Forecasting Arena Agent")


@app.post("/predict")
async def predict(payload: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
    return forecast_arena_payload_for_ensemble(payload, mode="real_live")


@app.get("/health")
async def health() -> dict[str, Any]:
    return run_preflight(offline=True)


@app.get("/ready")
async def ready() -> dict[str, Any]:
    report = run_preflight(offline=True)
    return {
        "ok": bool(report.get("checks", {}).get("cache_writable"))
        and bool(report.get("checks", {}).get("offline_prediction_valid"))
        and bool(report.get("llm", {}).get("key", {}).get("key_present")),
        "preflight": report,
    }
