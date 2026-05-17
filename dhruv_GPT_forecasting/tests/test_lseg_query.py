from datetime import datetime, timedelta, timezone

from dhruv_gpt_forecasting.arena_live_data import gather_live_evidence
from dhruv_gpt_forecasting.arena_priors import build_arena_packet
from dhruv_gpt_forecasting.config import load_config
from dhruv_gpt_forecasting.lseg_query import deterministic_lseg_query, plan_lseg_news_query


def test_deterministic_lseg_query_uses_macro_filters():
    packet = build_arena_packet({
        "event_ticker": "task-fed",
        "market_ticker": "task-fed",
        "title": "Will the Federal Reserve cut rates at the next FOMC meeting?",
        "category": "Economics",
        "outcomes": ["YES", "NO"],
        "as_of": "2026-05-17T12:00:00Z",
    })

    plan = deterministic_lseg_query(packet)

    assert "Language:LEN" in plan["query"]
    assert "Source:RTRS" in plan["query"]
    assert "Topic:SIGNWS" in plan["query"]
    assert plan["category_strategy"] == "macro_professional_news"


def test_deterministic_lseg_query_keeps_sports_broad():
    packet = build_arena_packet({
        "event_ticker": "task-sports",
        "market_ticker": "task-sports",
        "title": "Who will win: Pittsburgh Steelers or Atlanta Falcons?",
        "category": "Sports",
        "outcomes": ["Pittsburgh Steelers", "Atlanta Falcons"],
        "as_of": "2026-05-17T12:00:00Z",
    })

    plan = deterministic_lseg_query(packet)

    assert "Language:LEN" in plan["query"]
    assert "Topic:SIGNWS" not in plan["query"]
    assert plan["category_strategy"] == "sports_entity_availability_news"


def test_plan_lseg_query_falls_back_when_llm_disabled():
    packet = build_arena_packet({
        "event_ticker": "task-tv",
        "market_ticker": "task-tv",
        "title": "Will Alice be eliminated from Survivor tonight?",
        "category": "Entertainment",
        "outcomes": ["YES", "NO"],
        "as_of": "2026-05-17T12:00:00Z",
    })

    plan = plan_lseg_news_query(packet, load_config(), allow_llm=False)

    assert plan["source"] == "deterministic_lseg_query"
    assert "Language:LEN" in plan["query"]


def test_live_lseg_uses_query_planner(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path / "logs")
    cfg.arena.pit_external_root = str(tmp_path / "external_evidence")
    packet = build_arena_packet({
        "event_ticker": "task-msft",
        "market_ticker": "task-msft",
        "title": "Will Microsoft announce a major AI acquisition?",
        "category": "Financials",
        "close_time": "2026-05-18T12:00:00Z",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "outcomes": ["YES", "NO"],
    })

    captured = {}

    def fake_plan(packet, cfg, *, deadline_at=None, allow_llm=True):
        captured["allow_llm"] = allow_llm
        return {
            "query": "R:MSFT.O AND Language:LEN AND Source:RTRS",
            "alternate_queries": [],
            "category_strategy": "ric_professional_news",
            "entities": ["Microsoft"],
            "confidence": 0.9,
            "risks": [],
            "source": "gpt_lseg_query",
        }

    def fake_fetch(source, query, as_of, cfg, *, deadline_at=None):
        captured["query"] = query
        published_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        return [
            {
                "source": "lseg",
                "published_at": published_at,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "title": "Microsoft deal report",
                "text": "Reuters reports acquisition talks.",
            }
        ], []

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data._live_source_plan", lambda packet: {"lseg"})
    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data.plan_lseg_news_query", fake_plan)
    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data.fetch_vendor_records", fake_fetch)

    evidence = gather_live_evidence(packet, cfg, enabled=True, allow_llm_queries=True)

    assert captured["allow_llm"] is True
    assert captured["query"] == "R:MSFT.O AND Language:LEN AND Source:RTRS"
    lseg = [item for item in evidence if item.get("source") == "lseg"]
    assert lseg[0]["lseg_query_plan"]["source"] == "gpt_lseg_query"


def test_live_lseg_retries_alternate_query(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path / "logs")
    cfg.arena.pit_external_root = str(tmp_path / "external_evidence")
    packet = build_arena_packet({
        "event_ticker": "task-msft",
        "market_ticker": "task-msft",
        "title": "Will Microsoft announce a major AI acquisition?",
        "category": "Financials",
        "close_time": "2026-05-18T12:00:00Z",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "outcomes": ["YES", "NO"],
    })

    attempted = []

    def fake_plan(packet, cfg, *, deadline_at=None, allow_llm=True):
        return {
            "query": "narrow query",
            "alternate_queries": ["R:MSFT.O AND Language:LEN"],
            "category_strategy": "ric_professional_news",
            "entities": ["Microsoft"],
            "confidence": 0.9,
            "risks": [],
            "source": "gpt_lseg_query",
        }

    def fake_fetch(source, query, as_of, cfg, *, deadline_at=None):
        attempted.append(query)
        if query == "narrow query":
            return [], []
        return [
            {
                "source": "lseg",
                "published_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "title": "Microsoft headline",
            }
        ], []

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data._live_source_plan", lambda packet: {"lseg"})
    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data.plan_lseg_news_query", fake_plan)
    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data.fetch_vendor_records", fake_fetch)

    evidence = gather_live_evidence(packet, cfg, enabled=True, allow_llm_queries=True)

    lseg = [item for item in evidence if item.get("source") == "lseg"]
    assert attempted == ["narrow query", "R:MSFT.O AND Language:LEN"]
    assert lseg[0]["query"] == "R:MSFT.O AND Language:LEN"
    assert lseg[0]["attempted_queries"] == attempted
