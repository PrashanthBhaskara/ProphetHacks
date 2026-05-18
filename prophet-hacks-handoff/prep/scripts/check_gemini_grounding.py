#!/usr/bin/env python3
"""Offline contract checks for Gemini grounding and market-fallback behavior."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

PREP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREP_ROOT / "src"))

from dhruv_gpt_forecasting.config import load_config  # noqa: E402
from prep.forecasters.base import ForecasterConfig  # noqa: E402
from prep.forecasters import gemini  # noqa: E402
from prep.schemas import KalshiQuote, MarketPacket  # noqa: E402


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        exc = requests.HTTPError(f"{self.status_code} fake error")
        exc.response = self
        raise exc


def _packet() -> MarketPacket:
    return MarketPacket(
        as_of="2026-05-18T00:00:00Z",
        event_ticker="KXBTCD-26MAY1917",
        market_ticker="KXBTCD-26MAY1917-T76999.99",
        title="Bitcoin above $76,999.99 on May 19, 2026?",
        subtitle=None,
        rules="If Bitcoin's price is above $76,999.99 at close on May 19, 2026, then the market resolves to Yes.",
        category="Crypto",
        close_time="2026-05-19T21:00:00Z",
        kalshi=KalshiQuote(last_price=0.73),
        outcomes=["Yes", "No"],
        retrieval={"market_implied_probabilities": {"Yes": 0.73, "No": 0.27}},
    )


def _gemini_config() -> ForecasterConfig:
    return ForecasterConfig(
        name="gemini_pro",
        provider="gemini",
        model="gemini-3-pro-preview",
        api_key_env="GEMINI_API_KEY1",
        temperature=0.1,
        max_tokens=1024,
        enable_google_search=True,
        require_google_search_grounding=True,
        system_prompt="Return grounded forecast JSON.",
    )


def test_direct_gemini_requests_google_search_tool() -> None:
    os.environ["GEMINI_API_KEY1"] = "test-key"
    captured_payloads: list[dict[str, Any]] = []

    forecast_payload = {
        "forecast": {
            "probabilities": {"Yes": 0.7, "No": 0.3},
            "confidence": 0.6,
            "uncertainty": 0.4,
        },
        "reasoning_track": {
            "summary": "Grounded test forecast.",
            "source_audit": [],
        },
        "diagnostics": {
            "evidence_quality": "medium",
            "rules_clarity": "high",
            "liquidity_quality": "medium",
            "should_defer_to_market": False,
        },
    }
    raw = {
        "candidates": [{
            "content": {"parts": [{"text": json.dumps(forecast_payload)}]},
            "groundingMetadata": {
                "webSearchQueries": ["Bitcoin price May 19 2026 Kalshi"],
                "groundingChunks": [{
                    "web": {
                        "title": "Kalshi",
                        "uri": "https://kalshi.com/",
                        "domain": "kalshi.com",
                    },
                }],
                "groundingSupports": [{"segment": {"text": "test"}}],
            },
        }],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10},
    }

    original_post = gemini.requests.post
    try:
        def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
            captured_payloads.append(kwargs["json"])
            return FakeResponse(200, raw)

        gemini.requests.post = fake_post
        result = gemini.forecast(_gemini_config(), _packet())
    finally:
        gemini.requests.post = original_post

    assert captured_payloads, "Gemini API was not called"
    assert captured_payloads[0]["tools"] == [{"google_search": {}}]
    assert "responseMimeType" not in captured_payloads[0]["generationConfig"]
    grounding = result.raw_response["grounding"]
    assert grounding["enabled"] is True
    assert grounding["required"] is True
    assert grounding["present"] is True
    assert grounding["web_search_queries"], "Grounding metadata did not include web search queries"


def test_direct_gemini_quota_falls_back_to_market_with_defer_flag() -> None:
    os.environ["GEMINI_API_KEY1"] = "test-key"
    original_post = gemini.requests.post
    try:
        def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
            return FakeResponse(
                429,
                text='{"error":{"status":"RESOURCE_EXHAUSTED","message":"quota or credits exhausted"}}',
            )

        gemini.requests.post = fake_post
        result = gemini.forecast(_gemini_config(), _packet())
    finally:
        gemini.requests.post = original_post

    assert result.diagnostics.should_defer_to_market is True
    assert result.raw_response["gemini_market_fallback"]["should_defer_to_market"] is True
    assert result.probabilities["Yes"] > 0.70
    assert result.probabilities["No"] < 0.30
    assert "mirroring the current Kalshi market" in result.reasoning_track.summary


def test_direct_gemini_accepts_preflight_grounding_when_final_metadata_missing() -> None:
    os.environ["GEMINI_API_KEY1"] = "test-key"
    calls = 0
    payload = {
        "forecast": {
            "probabilities": {"Yes": 0.7, "No": 0.3},
            "confidence": 0.6,
            "uncertainty": 0.4,
        },
        "reasoning_track": {"summary": "Retry grounded forecast."},
        "diagnostics": {"should_defer_to_market": False},
    }

    original_post = gemini.requests.post
    try:
        def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
            nonlocal calls
            calls += 1
            candidate: dict[str, Any] = {
                "content": {"parts": [{"text": json.dumps(payload)}]},
            }
            if calls == 1:
                candidate["groundingMetadata"] = {
                    "searchEntryPoint": {"renderedContent": "test"},
                    "webSearchQueries": ["Bitcoin price"],
                }
            return FakeResponse(200, {"candidates": [candidate]})

        gemini.requests.post = fake_post
        result = gemini.forecast(_gemini_config(), _packet())
    finally:
        gemini.requests.post = original_post

    assert calls == 2
    assert result.raw_response["grounding_retry_attempted"] is False
    assert result.raw_response["grounding"]["present"] is True
    assert result.raw_response["grounding"]["source"] == "pre_forecast_grounded_search_brief"


def test_dhruv_lane_uses_flash_with_native_google_search() -> None:
    cfg = load_config(PREP_ROOT / "config" / "dhruv_gemini.default.json")
    assert cfg.model.provider == "gemini"
    assert cfg.model.model == "gemini-3-flash-preview"
    assert cfg.model.api_key_env == "GEMINI_API_KEY2"
    assert cfg.model.native_search_grounding_enabled is True
    assert cfg.model.native_search_grounding_live_only is False
    assert cfg.model.search_grounding_engine == "google_search"


def main() -> int:
    tests = [
        test_direct_gemini_requests_google_search_tool,
        test_direct_gemini_quota_falls_back_to_market_with_defer_flag,
        test_direct_gemini_accepts_preflight_grounding_when_final_metadata_missing,
        test_dhruv_lane_uses_flash_with_native_google_search,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
