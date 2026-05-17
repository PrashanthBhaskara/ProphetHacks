import math

import pytest

from dhruv_gpt_forecasting.arena_agent import predict, predict_prophet


def _prob_map(response):
    if "forecast" in response:
        response = response["forecast"]["prediction_response"]
    return {item["market"]: item["probability"] for item in response["probabilities"]}


def test_local_module_predict_matches_current_prophet_docs(monkeypatch):
    monkeypatch.setenv("ARENA_OFFLINE", "1")
    monkeypatch.setenv("ARENA_DISABLE_LIVE_DATA", "1")
    event = {
        "event_ticker": "task-001",
        "market_ticker": "task-001",
        "title": "Who will win: Pittsburgh or Atlanta?",
        "subtitle": None,
        "description": "Predict the winner of the scheduled matchup.",
        "category": "Sports",
        "rules": "Resolves to the official winner after the game is final.",
        "close_time": "2026-12-31T23:59:59Z",
        "outcomes": ["Pittsburgh", "Atlanta"],
        "resolved_outcome": None,
    }

    response = predict(event)
    probs = _prob_map(response)

    assert set(response) == {"run_metadata", "market_comparison", "forecast"}
    assert set(response["forecast"]) == {
        "source",
        "probabilities",
        "confidence",
        "uncertainty",
        "reason_codes",
        "key_evidence",
        "counterarguments",
        "information_gaps",
        "calibration_note",
        "prediction_response",
        "audit",
    }
    assert set(response["forecast"]["audit"]) == {
        "mode",
        "model",
        "native_search_grounding_enabled",
        "search_grounding_engine",
        "final_probability_authority",
        "prior_shrink_weight",
        "fallback_reason",
        "errors",
        "elapsed_seconds",
        "deadline_seconds",
        "within_deadline",
        "live_evidence_count",
        "live_evidence_sources",
        "api_logs",
    }
    assert response["forecast"]["audit"]["errors"] == []
    assert list(probs) == ["Pittsburgh", "Atlanta"]
    assert all(0.0 <= probability <= 1.0 for probability in probs.values())
    assert math.isclose(sum(probs.values()), 1.0)


def test_local_module_predict_handles_nonbinary_outcomes_for_current_docs(monkeypatch):
    monkeypatch.setenv("ARENA_OFFLINE", "1")
    monkeypatch.setenv("ARENA_DISABLE_LIVE_DATA", "1")
    event = {
        "event_ticker": "task-002",
        "market_ticker": "task-002",
        "title": "Which margin bucket will resolve?",
        "description": "Exactly one listed bucket resolves.",
        "category": "Sports",
        "rules": "Resolves to the official final margin bucket.",
        "close_time": "2026-12-31T23:59:59Z",
        "outcomes": ["A by 0-3", "A by 4-7", "B by 0-3", "B by 4+"],
        "resolved_outcome": None,
    }

    response = predict(event)
    probs = _prob_map(response)

    assert list(probs) == ["A by 0-3", "A by 4-7", "B by 0-3", "B by 4+"]
    assert all(0.0 <= probability <= 1.0 for probability in probs.values())
    assert math.isclose(sum(probs.values()), 1.0)


def test_prophet_scoring_adapter_keeps_bare_docs_contract(monkeypatch):
    monkeypatch.setenv("ARENA_OFFLINE", "1")
    monkeypatch.setenv("ARENA_DISABLE_LIVE_DATA", "1")
    event = {
        "event_ticker": "task-prophet",
        "market_ticker": "task-prophet",
        "title": "Who will win: Pittsburgh or Atlanta?",
        "subtitle": None,
        "description": "Predict the winner of the scheduled matchup.",
        "category": "Sports",
        "rules": "Resolves to the official winner after the game is final.",
        "close_time": "2026-12-31T23:59:59Z",
        "outcomes": ["Pittsburgh", "Atlanta"],
        "resolved_outcome": None,
    }

    response = predict_prophet(event)
    probs = _prob_map(response)

    assert set(response) == {"probabilities"}
    assert list(probs) == ["Pittsburgh", "Atlanta"]
    assert math.isclose(sum(probs.values()), 1.0)


def test_http_predict_contract_matches_current_prophet_docs(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    from dhruv_gpt_forecasting.server import app

    monkeypatch.setenv("ARENA_OFFLINE", "1")
    monkeypatch.setenv("ARENA_DISABLE_LIVE_DATA", "1")
    client = TestClient(app)
    event = {
        "event_ticker": "task-003",
        "market_ticker": "task-003",
        "title": "Who will win: Cleveland or Atlanta?",
        "description": "Predict the winner.",
        "category": "Sports",
        "rules": "Resolves to the official winner.",
        "close_time": "2026-12-31T23:59:59Z",
        "outcomes": ["Cleveland", "Atlanta"],
    }

    response = client.post("/predict", json=[event])
    body = response.json()
    probs = _prob_map(body)

    assert response.status_code == 200
    assert set(body) == {"run_metadata", "market_comparison", "forecast"}
    assert list(probs) == ["Cleveland", "Atlanta"]
    assert math.isclose(sum(probs.values()), 1.0)
