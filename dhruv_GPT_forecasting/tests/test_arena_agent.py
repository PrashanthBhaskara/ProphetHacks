import math
import json

from dhruv_gpt_forecasting import predict as package_predict
from dhruv_gpt_forecasting.arena_agent import forecast_arena_event, predict, predict_prophet
from dhruv_gpt_forecasting.config import load_config


def test_arena_predict_preserves_named_outcomes_and_normalizes(monkeypatch):
    monkeypatch.setenv("ARENA_OFFLINE", "1")
    event = {
        "event_ticker": "task-001",
        "market_ticker": "task-001",
        "title": "Who will win: Cleveland or Atlanta?",
        "description": "Predict the winner.",
        "category": "Sports",
        "rules": "Resolves to the official winner after the game is final.",
        "close_time": "2026-03-21T23:59:59Z",
        "outcomes": ["Cleveland", "Atlanta"],
    }
    response = predict(event)
    assert set(response) == {"run_metadata", "market_comparison", "forecast"}
    response = response["forecast"]["prediction_response"]
    probs = {item["market"]: item["probability"] for item in response["probabilities"]}
    assert list(probs) == ["Cleveland", "Atlanta"]
    assert math.isclose(sum(probs.values()), 1.0)
    assert probs["Cleveland"] != probs["Atlanta"]


def test_prophet_adapter_preserves_bare_prediction_contract(monkeypatch):
    monkeypatch.setenv("ARENA_OFFLINE", "1")
    event = {
        "event_ticker": "task-001",
        "market_ticker": "task-001",
        "title": "Who will win: Cleveland or Atlanta?",
        "description": "Predict the winner.",
        "category": "Sports",
        "rules": "Resolves to the official winner after the game is final.",
        "close_time": "2026-03-21T23:59:59Z",
        "outcomes": ["Cleveland", "Atlanta"],
    }
    response = predict_prophet(event)
    probs = {item["market"]: item["probability"] for item in response["probabilities"]}
    assert list(probs) == ["Cleveland", "Atlanta"]
    assert math.isclose(sum(probs.values()), 1.0)
    assert probs["Cleveland"] != probs["Atlanta"]


def test_package_predict_uses_arena_agent(monkeypatch):
    monkeypatch.setenv("ARENA_OFFLINE", "1")
    event = {
        "event_ticker": "task-002",
        "market_ticker": "task-002",
        "title": "Will it rain tomorrow?",
        "category": "Weather",
        "rules": "Resolves Yes if measurable rain occurs.",
        "close_time": "2026-03-21T23:59:59Z",
        "outcomes": ["YES", "NO"],
    }
    response = package_predict(event)
    response = response["forecast"]["prediction_response"]
    probs = {item["market"]: item["probability"] for item in response["probabilities"]}
    assert set(probs) == {"YES", "NO"}
    assert math.isclose(sum(probs.values()), 1.0)


def test_arena_forecast_uses_deterministic_fallback_in_offline_mode(monkeypatch):
    monkeypatch.setenv("ARENA_OFFLINE", "1")
    event = {
        "event_ticker": "task-003",
        "market_ticker": "task-003",
        "title": "Who will win: Houston or Minnesota?",
        "category": "Sports",
        "rules": "Resolves to official winner.",
        "close_time": "2026-03-21T23:59:59Z",
        "outcomes": ["Houston", "Minnesota"],
    }
    forecast = forecast_arena_event(event, use_live_data=False)
    assert forecast.source == "deterministic_arena_prior"
    assert math.isclose(sum(forecast.probabilities.values()), 1.0)
    assert set(forecast.probabilities) == {"Houston", "Minnesota"}


def test_arena_gpt_is_final_probability_authority(monkeypatch):
    monkeypatch.delenv("ARENA_OFFLINE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    monkeypatch.setenv("ARENA_ENABLE_FORECAST_CACHE", "0")
    calls = []

    def fake_cached_json_call(cfg, *, messages, cache_namespace, **kwargs):
        calls.append(kwargs)
        return {
            "probabilities": {"YES": 0.73, "NO": 0.27},
            "confidence": 0.25,
            "uncertainty": 0.90,
            "reason_codes": ["test_payload"],
            "key_evidence": [],
            "counterarguments": [],
            "information_gaps": [],
            "calibration_note": "GPT final probability test.",
        }, {"cache_hit": False, "prompt_hash": "abc", "model": cfg.model.model}

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_agent._cached_json_call", fake_cached_json_call)
    event = {
        "event_ticker": "task-004",
        "market_ticker": "task-004",
        "title": "Will the test event happen?",
        "category": "Politics",
        "rules": "Resolves Yes if it happens.",
        "close_time": "2026-03-21T23:59:59Z",
        "outcomes": ["YES", "NO"],
    }

    cfg = load_config()
    cfg.arena.second_pass_enabled = False
    forecast = forecast_arena_event(event, config=cfg, use_gpt=True, use_live_data=False)

    assert forecast.source == "gpt_primary"
    assert math.isclose(sum(forecast.probabilities.values()), 1.0)
    assert forecast.audit["prior_shrink_weight"] >= 0.35
    assert forecast.audit["final_probability_authority"] == "gpt_with_calibration_shrink"
    assert calls[0]["search_grounding"] is True
    assert forecast.audit["native_search_grounding_enabled"] is True


def test_arena_forecast_disables_native_search_grounding_for_historical_asof(monkeypatch):
    monkeypatch.delenv("ARENA_OFFLINE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    monkeypatch.setenv("ARENA_ENABLE_FORECAST_CACHE", "0")
    calls = []

    def fake_cached_json_call(cfg, *, messages, cache_namespace, **kwargs):
        calls.append(kwargs)
        return {
            "probabilities": {"YES": 0.60, "NO": 0.40},
            "confidence": 0.55,
            "uncertainty": 0.45,
            "reason_codes": ["historical_payload"],
            "key_evidence": [],
            "counterarguments": [],
            "information_gaps": [],
            "calibration_note": "Historical test.",
        }, {"cache_hit": False, "prompt_hash": "abc", "model": cfg.model.model}

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_agent._cached_json_call", fake_cached_json_call)
    cfg = load_config()
    cfg.arena.second_pass_enabled = False
    event = {
        "event_ticker": "task-004-historical",
        "market_ticker": "task-004-historical",
        "title": "Will the historical test event happen?",
        "category": "Politics",
        "rules": "Resolves Yes if it happens.",
        "as_of": "2025-10-01T00:00:00Z",
        "close_time": "2025-10-02T00:00:00Z",
        "outcomes": ["YES", "NO"],
    }

    forecast = forecast_arena_event(event, config=cfg, use_gpt=True, use_live_data=False)

    assert calls[0]["search_grounding"] is False
    assert forecast.audit["native_search_grounding_enabled"] is False


def test_arena_forecast_attaches_grounded_research_evidence(monkeypatch):
    monkeypatch.delenv("ARENA_OFFLINE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    monkeypatch.setenv("ARENA_ENABLE_FORECAST_CACHE", "0")
    monkeypatch.setenv("ARENA_ENABLE_PRE_GROUNDED_RESEARCH", "1")

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_agent.gather_live_evidence", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "dhruv_gpt_forecasting.arena_agent.gather_grounded_research_evidence",
        lambda *args, **kwargs: [{
            "source": "gemini_native_search_grounded_research",
            "claim": "Grounded research digest.",
            "summary": "Sources summarized.",
            "retrieval_confidence": {"overall": 0.7},
        }],
    )

    def fake_cached_json_call(cfg, *, messages, cache_namespace, **kwargs):
        return {
            "probabilities": {"YES": 0.62, "NO": 0.38},
            "confidence": 0.60,
            "uncertainty": 0.40,
            "reason_codes": ["uses_grounded_research"],
            "key_evidence": [],
            "counterarguments": [],
            "information_gaps": [],
            "calibration_note": "Used grounded research.",
        }, {"cache_hit": False, "prompt_hash": "abc", "model": cfg.model.model}

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_agent._cached_json_call", fake_cached_json_call)
    event = {
        "event_ticker": "task-grounded",
        "market_ticker": "KXTEST-26DEC31",
        "title": "Will the grounded live test happen?",
        "category": "Politics",
        "rules": "Resolves Yes if it happens.",
        "close_time": "2026-12-31T23:59:59Z",
        "outcomes": ["YES", "NO"],
    }
    cfg = load_config()
    cfg.arena.second_pass_enabled = False

    forecast = forecast_arena_event(event, config=cfg, use_gpt=True, use_live_data=True)

    assert forecast.audit["live_evidence_count"] == 1
    assert forecast.audit["live_evidence_preview"][0]["source"] == "gemini_native_search_grounded_research"


def test_live_arena_forecast_skips_pre_grounded_research_by_default(monkeypatch):
    monkeypatch.delenv("ARENA_OFFLINE", raising=False)
    monkeypatch.delenv("ARENA_ENABLE_PRE_GROUNDED_RESEARCH", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    monkeypatch.setenv("ARENA_ENABLE_FORECAST_CACHE", "0")

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_agent.gather_live_evidence", lambda *args, **kwargs: [])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("pre-grounded research should be skipped for live final-search forecasts")

    captured = {}

    def fake_cached_json_call(cfg, *, messages, cache_namespace, **kwargs):
        captured["messages"] = messages
        return {
            "probabilities": {"YES": 0.59, "NO": 0.41},
            "confidence": 0.60,
            "uncertainty": 0.40,
            "reason_codes": ["final_search_only"],
            "key_evidence": [],
            "counterarguments": [],
            "information_gaps": [],
            "calibration_note": "Used final native search only.",
        }, {"cache_hit": False, "prompt_hash": "abc", "model": cfg.model.model}

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_agent.gather_grounded_research_evidence", fail_if_called)
    monkeypatch.setattr("dhruv_gpt_forecasting.arena_agent._cached_json_call", fake_cached_json_call)
    event = {
        "event_ticker": "task-live-final-search",
        "market_ticker": "KXTEST-26DEC31",
        "title": "Will the live final-search test happen?",
        "category": "Politics",
        "rules": "Resolves Yes if it happens.",
        "close_time": "2026-12-31T23:59:59Z",
        "outcomes": ["YES", "NO"],
    }
    cfg = load_config()
    cfg.arena.second_pass_enabled = False

    forecast = forecast_arena_event(event, config=cfg, use_gpt=True, use_live_data=True)

    assert forecast.source == "gpt_primary"
    assert forecast.audit["native_search_grounding_enabled"] is True
    payload = json.loads(captured["messages"][1]["content"])
    assert payload["final_gemini_live_brief"]["mode"] == "live_one_pass_grounded_forecast"
    assert payload["final_gemini_live_brief"]["targeted_search_questions"]
    assert payload["event_packet"]["features"]["live_runtime_context"]["deadline_seconds"] == 480


def test_arena_forecast_accepts_supplied_pit_evidence(monkeypatch):
    monkeypatch.delenv("ARENA_OFFLINE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
    monkeypatch.setenv("ARENA_ENABLE_FORECAST_CACHE", "0")

    captured = {}

    def fake_cached_json_call(cfg, *, messages, cache_namespace, **kwargs):
        captured["messages"] = messages
        return {
            "probabilities": {"YES": 0.66, "NO": 0.34},
            "confidence": 0.60,
            "uncertainty": 0.40,
            "reason_codes": ["uses_supplied_pit_evidence"],
            "key_evidence": [],
            "counterarguments": [],
            "information_gaps": [],
            "calibration_note": "Used supplied PIT evidence.",
        }, {"cache_hit": False, "prompt_hash": "abc", "model": cfg.model.model}

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_agent._cached_json_call", fake_cached_json_call)
    event = {
        "event_ticker": "task-supplied-evidence",
        "market_ticker": "task-supplied-evidence",
        "title": "Will the supplied evidence event happen?",
        "category": "Politics",
        "rules": "Resolves Yes if it happens.",
        "close_time": "2026-03-21T23:59:59Z",
        "outcomes": ["YES", "NO"],
    }
    evidence = [{
        "source": "kalshi_random_pit_market_snapshot",
        "published_at": "2026-03-21T20:00:00Z",
        "probabilities": {"YES": 0.64, "NO": 0.36},
        "retrieval_confidence": {"overall": 0.95},
    }]
    cfg = load_config()
    cfg.arena.second_pass_enabled = False

    forecast = forecast_arena_event(
        event,
        config=cfg,
        use_gpt=True,
        use_live_data=False,
        external_evidence=evidence,
    )

    assert forecast.audit["live_evidence_count"] == 1
    assert forecast.audit["live_evidence_preview"][0]["source"] == "kalshi_random_pit_market_snapshot"
    payload = captured["messages"][1]["content"]
    assert "kalshi_random_pit_market_snapshot" in payload


def test_arena_forecast_records_response_deadline(monkeypatch):
    monkeypatch.setenv("ARENA_OFFLINE", "1")
    event = {
        "event_ticker": "task-005",
        "market_ticker": "task-005",
        "title": "Will the timed test event happen?",
        "category": "Politics",
        "rules": "Resolves Yes if it happens.",
        "close_time": "2026-03-21T23:59:59Z",
        "outcomes": ["YES", "NO"],
    }

    forecast = forecast_arena_event(event, use_live_data=False, deadline_seconds=300)

    assert forecast.audit["response_deadline_seconds"] == 300
    assert forecast.audit["deadline_seconds"] == 300
    assert forecast.audit["within_response_deadline"] is True
    assert forecast.audit["within_deadline"] is True
    assert forecast.audit["elapsed_seconds"] >= 0.0


def test_arena_forecast_skips_gpt_when_deadline_budget_is_insufficient(monkeypatch):
    monkeypatch.delenv("ARENA_OFFLINE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("GPT should not be called when deadline budget is insufficient")

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_agent._cached_json_call", fail_if_called)
    event = {
        "event_ticker": "task-006",
        "market_ticker": "task-006",
        "title": "Will the rushed event happen?",
        "category": "Politics",
        "rules": "Resolves Yes if it happens.",
        "close_time": "2026-03-21T23:59:59Z",
        "outcomes": ["YES", "NO"],
    }

    forecast = forecast_arena_event(event, use_gpt=True, use_live_data=False, deadline_seconds=1)

    assert forecast.source == "deterministic_arena_prior"
    assert forecast.audit["fallback_reason"] == "deadline_budget_before_primary_gpt"
    assert forecast.audit["within_deadline"] is True


def test_deadline_fallback_uses_inline_market_quote(monkeypatch):
    monkeypatch.delenv("ARENA_OFFLINE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("GPT should not be called when deadline budget is insufficient")

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_agent._cached_json_call", fail_if_called)
    event = {
        "event_ticker": "task-market-fallback",
        "market_ticker": "KXTEST-26DEC31",
        "title": "Will the market fallback event happen?",
        "category": "Politics",
        "rules": "Resolves Yes if it happens.",
        "close_time": "2026-03-21T23:59:59Z",
        "outcomes": ["YES", "NO"],
        "yes_bid": 63,
        "yes_ask": 67,
    }

    forecast = forecast_arena_event(event, use_gpt=True, use_live_data=False, deadline_seconds=1)

    assert forecast.audit["fallback_reason"] == "deadline_budget_before_primary_gpt"
    assert forecast.probabilities["YES"] > 0.55
    assert forecast.audit["deterministic_prior"]["diagnostics"]["live_distribution"]["YES"] == 0.65
