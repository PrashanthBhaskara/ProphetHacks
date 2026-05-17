from datetime import datetime, timedelta, timezone

from dhruv_gpt_forecasting.arena_priors import build_arena_packet
from dhruv_gpt_forecasting.config import load_config
from dhruv_gpt_forecasting.grounded_research import (
    gather_grounded_research_evidence,
    grounded_research_messages,
    targeted_research_questions,
)


class FakeCallLog:
    def to_dict(self):
        return {
            "model": "gemini-3-flash-preview",
            "prompt_hash": "abc",
            "search_grounding_enabled": True,
            "search_grounding_engine": "native",
            "response_annotation_count": 2,
        }


def test_targeted_research_questions_are_category_specific():
    packet = build_arena_packet({
        "event_ticker": "task-sports",
        "market_ticker": "task-sports",
        "title": "Who will win: San Antonio or Oklahoma City?",
        "category": "Sports",
        "rules": "Resolves to the official winner.",
        "close_time": "2026-12-31T23:59:59Z",
        "outcomes": ["San Antonio", "Oklahoma City"],
    }, include_historical_analogs=False)

    questions = targeted_research_questions(packet)

    assert any("injuries" in question for question in questions)
    assert any("contract-resolution" in question for question in questions)


def test_grounded_research_calls_gemini_with_native_search(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-secret-value")
    monkeypatch.setenv("ARENA_ENABLE_FORECAST_CACHE", "0")
    captured = {}

    def fake_call_openrouter_json(**kwargs):
        captured.update(kwargs)
        return {
            "targeted_questions": ["What matters?"],
            "macroeconomic_drivers": [{"claim": "Rates moved.", "source": "example.com", "impact": "risk"}],
            "breaking_news": [{"claim": "New headline.", "source": "news.example", "impact": "supports YES"}],
            "qualitative_sentiment": [{"claim": "Sentiment is mixed.", "source": "search", "impact": "uncertain"}],
            "contract_specific_factors": [{"claim": "Rules matter.", "source": "kalshi", "impact": "calibration"}],
            "source_notes": [{
                "source": "example.com",
                "url": "https://example.com/story",
                "title": "Story",
                "published_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            }],
            "evidence_quality": {
                "overall": 0.7,
                "freshness": 0.8,
                "source_quality": 0.75,
                "event_match": 0.65,
                "conflict_level": 0.1,
            },
            "information_gaps": ["One missing detail."],
            "summary": "Grounded summary.",
        }, FakeCallLog()

    monkeypatch.setattr("dhruv_gpt_forecasting.grounded_research.call_openrouter_json", fake_call_openrouter_json)
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path)
    packet = build_arena_packet({
        "event_ticker": "task-live",
        "market_ticker": "KXTEST-26DEC31",
        "title": "Will inflation be above 3%?",
        "category": "Economics",
        "rules": "Resolves Yes if the official CPI print is above 3%.",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "close_time": "2026-12-31T23:59:59Z",
        "outcomes": ["YES", "NO"],
    }, include_historical_analogs=False)

    evidence = gather_grounded_research_evidence(packet, cfg, deadline_at=None, existing_evidence=[])

    assert captured["model"].model == "gemini-3-flash-preview"
    assert captured["search_grounding"] is True
    assert evidence[0]["source"] == "gemini_native_search_grounded_research"
    assert evidence[0]["summary"] == "Grounded summary."
    assert evidence[0]["pit_verified_source_dates"] is True
    assert evidence[0]["retrieval_confidence"]["overall"] == 0.7
    assert evidence[0]["api_log"]["search_grounding_enabled"] is True


def test_grounded_research_skips_historical_asof(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-secret-value")

    def fail_if_called(**kwargs):
        raise AssertionError("historical grounded research should not call Gemini")

    monkeypatch.setattr("dhruv_gpt_forecasting.grounded_research.call_openrouter_json", fail_if_called)
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path)
    packet = build_arena_packet({
        "event_ticker": "task-history",
        "market_ticker": "KXTEST-25OCT01",
        "title": "Will the historical event happen?",
        "category": "Politics",
        "rules": "Resolves Yes if it happens.",
        "as_of": "2025-10-01T00:00:00Z",
        "close_time": "2025-10-02T00:00:00Z",
        "outcomes": ["YES", "NO"],
    }, include_historical_analogs=False)

    assert gather_grounded_research_evidence(packet, cfg, deadline_at=None, existing_evidence=[]) == []


def test_grounded_research_backtest_internet_filters_sources_by_publish_date(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-secret-value")
    monkeypatch.setenv("ARENA_ENABLE_FORECAST_CACHE", "0")
    monkeypatch.setenv("ARENA_ENABLE_BACKTEST_INTERNET", "1")

    def fake_call_openrouter_json(**kwargs):
        return {
            "targeted_questions": ["What was known then?"],
            "macroeconomic_drivers": [
                {"claim": "Valid pre-cutoff source.", "source": "valid.example", "impact": "usable"},
                {"claim": "Future source.", "source": "future.example", "impact": "must drop"},
                {"claim": "Undated source.", "source": "unknown.example", "impact": "must drop"},
            ],
            "breaking_news": [{"claim": "Valid linked URL.", "source": "https://valid.example/story", "impact": "usable"}],
            "qualitative_sentiment": [],
            "contract_specific_factors": [],
            "source_notes": [
                {
                    "source": "valid.example",
                    "url": "https://valid.example/story",
                    "title": "Valid",
                    "published_at": "2025-09-30T23:00:00Z",
                },
                {
                    "source": "future.example",
                    "url": "https://future.example/story",
                    "title": "Future",
                    "published_at": "2025-10-01T01:00:00Z",
                },
                {"source": "unknown.example", "url": "https://unknown.example/story", "title": "Unknown"},
            ],
            "evidence_quality": {
                "overall": 0.8,
                "freshness": 0.7,
                "source_quality": 0.8,
                "event_match": 0.8,
                "conflict_level": 0.0,
            },
            "information_gaps": [],
            "summary": "Only pre-cutoff sources should survive.",
        }, FakeCallLog()

    monkeypatch.setattr("dhruv_gpt_forecasting.grounded_research.call_openrouter_json", fake_call_openrouter_json)
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path)
    packet = build_arena_packet({
        "event_ticker": "task-history",
        "market_ticker": "KXTEST-25OCT01",
        "title": "Will the historical event happen?",
        "category": "Economics",
        "rules": "Resolves Yes if it happens.",
        "as_of": "2025-10-01T00:00:00Z",
        "close_time": "2025-10-02T00:00:00Z",
        "outcomes": ["YES", "NO"],
    }, include_historical_analogs=False)

    evidence = gather_grounded_research_evidence(packet, cfg, deadline_at=None, existing_evidence=[])

    assert evidence[0]["pit_mode"] == "backtest_native_search_published_at_verified"
    assert evidence[0]["source_date_audit"]["accepted_source_count"] == 1
    assert evidence[0]["source_date_audit"]["discarded_source_count"] == 2
    assert [row["source"] for row in evidence[0]["source_notes"]] == ["valid.example"]
    assert [row["source"] for row in evidence[0]["macroeconomic_drivers"]] == ["valid.example"]
    assert evidence[0]["breaking_news"][0]["source"] == "https://valid.example/story"


def test_grounded_research_backtest_internet_rejects_undated_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-secret-value")
    monkeypatch.setenv("ARENA_ENABLE_FORECAST_CACHE", "0")
    monkeypatch.setenv("ARENA_ENABLE_BACKTEST_INTERNET", "1")

    def fake_call_openrouter_json(**kwargs):
        return {
            "targeted_questions": ["What was known then?"],
            "macroeconomic_drivers": [{"claim": "No date.", "source": "unknown.example", "impact": "drop"}],
            "breaking_news": [],
            "qualitative_sentiment": [],
            "contract_specific_factors": [],
            "source_notes": [{"source": "unknown.example", "url": "https://unknown.example/story"}],
            "evidence_quality": {"overall": 0.5},
            "information_gaps": [],
            "summary": "Should not be used.",
        }, FakeCallLog()

    monkeypatch.setattr("dhruv_gpt_forecasting.grounded_research.call_openrouter_json", fake_call_openrouter_json)
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path)
    packet = build_arena_packet({
        "event_ticker": "task-history-undated",
        "market_ticker": "KXTEST-25OCT01",
        "title": "Will the historical event happen?",
        "category": "Politics",
        "rules": "Resolves Yes if it happens.",
        "as_of": "2025-10-01T00:00:00Z",
        "close_time": "2025-10-02T00:00:00Z",
        "outcomes": ["YES", "NO"],
    }, include_historical_analogs=False)

    evidence = gather_grounded_research_evidence(packet, cfg, deadline_at=None, existing_evidence=[])

    assert evidence[0]["error"] == "no_pit_verified_sources"
    assert evidence[0]["source_date_audit"]["accepted_source_count"] == 0


def test_grounded_research_prompt_contains_publish_date_policy():
    packet = build_arena_packet({
        "event_ticker": "task-policy",
        "market_ticker": "task-policy",
        "title": "Will the policy test happen?",
        "category": "Politics",
        "rules": "Resolves Yes if it happens.",
        "as_of": "2025-10-01T00:00:00Z",
        "close_time": "2025-10-02T00:00:00Z",
        "outcomes": ["YES", "NO"],
    }, include_historical_analogs=False)

    messages = grounded_research_messages(packet, targeted_research_questions(packet), [])
    joined = "\n".join(message["content"] for message in messages)

    assert "published_at" in joined
    assert "ambiguous-date" in joined or "ambiguous_date" in joined
    assert "at or before" in joined
