import json
from datetime import datetime, timedelta, timezone

from dhruv_gpt_forecasting.config import load_config
from dhruv_gpt_forecasting.features import build_feature_packet
from dhruv_gpt_forecasting.pit_evidence import (
    build_evidence_query,
    fetch_external_records_for_packet,
    gather_pit_external_evidence,
)


def test_pit_external_evidence_filters_publish_and_collection_times(tmp_path):
    root = tmp_path / "external_evidence"
    root.mkdir()
    rows = [
        {
            "source": "reddit",
            "market_ticker": "KXTEST-YES",
            "published_at": "2026-03-01T11:30:00Z",
            "collected_at": "2026-03-01T11:45:00Z",
            "title": "Test event discussion before cutoff",
            "text": "YES looks plausible before the cutoff.",
        },
        {
            "source": "x",
            "market_ticker": "KXTEST-YES",
            "published_at": "2026-03-01T12:05:00Z",
            "collected_at": "2026-03-01T12:05:00Z",
            "text": "This is after the forecast cutoff.",
        },
        {
            "source": "reddit",
            "market_ticker": "KXTEST-YES",
            "published_at": "2026-03-01T11:50:00Z",
            "collected_at": "2026-03-01T13:00:00Z",
            "title": "Collected after the simulated forecast cutoff",
        },
    ]
    (root / "reddit.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    cfg = load_config()
    cfg.arena.pit_external_root = str(root)
    packet = build_feature_packet(
        {
            "event_ticker": "KXTEST",
            "market_ticker": "KXTEST-YES",
            "title": "Will the test event happen?",
            "category": "Politics",
            "rules": "Resolves Yes if it happens.",
            "close_time": "2026-03-01T18:00:00Z",
            "outcomes": ["YES", "NO"],
        },
        {"last_price": 0.5},
        as_of="2026-03-01T12:00:00Z",
    )

    evidence = gather_pit_external_evidence(packet, cfg, enabled=True, allow_network=False)

    assert len(evidence) == 1
    assert evidence[0]["source"] == "pit_external_evidence"
    assert evidence[0]["record_count"] == 1
    assert evidence[0]["records"][0]["title"] == "Test event discussion before cutoff"
    assert evidence[0]["records"][0]["sentiment_model"] == "lexicon_v1"
    assert "sentiment" in evidence[0]


def test_pit_external_evidence_does_not_network_for_historical_as_of(monkeypatch, tmp_path):
    def fail_get(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("network should not be used for historical PIT evidence")

    monkeypatch.setattr("dhruv_gpt_forecasting.pit_evidence.requests.get", fail_get)
    monkeypatch.setenv("PIT_EXTERNAL_ALLOW_NETWORK", "1")
    cfg = load_config()
    cfg.arena.pit_external_root = str(tmp_path / "missing")
    packet = build_feature_packet(
        {
            "event_ticker": "KXOLD",
            "market_ticker": "KXOLD-YES",
            "title": "Will the old test event happen?",
            "category": "Politics",
            "close_time": "2026-03-01T18:00:00Z",
            "outcomes": ["YES", "NO"],
        },
        {"last_price": 0.5},
        as_of="2026-03-01T12:00:00Z",
    )

    evidence = gather_pit_external_evidence(packet, cfg, enabled=True)

    assert evidence == []


def test_fetch_external_records_marks_historical_backfill_quality(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.arena.pit_external_root = str(tmp_path / "missing")
    packet = build_feature_packet(
        {
            "event_ticker": "KXOLD",
            "market_ticker": "KXOLD-YES",
            "title": "Will the old test event happen?",
            "category": "Politics",
            "close_time": "2026-03-01T18:00:00Z",
            "outcomes": ["YES", "NO"],
        },
        {"last_price": 0.5},
        as_of="2026-03-01T12:00:00Z",
    )

    def fake_fetch_x(query, as_of_dt, cfg, *, allow_historical_archive=None):
        assert allow_historical_archive is True
        return [
            {
                "source": "x",
                "published_at": "2026-03-01T11:55:00Z",
                "collected_at": "2026-05-16T12:00:00Z",
                "text": "Old test event before cutoff.",
            }
        ], []

    monkeypatch.setattr("dhruv_gpt_forecasting.pit_evidence._fetch_x", fake_fetch_x)

    records, errors = fetch_external_records_for_packet(
        packet,
        cfg,
        sources={"x"},
        allow_historical_backfill=True,
    )

    assert errors == []
    assert len(records) == 1
    assert records[0]["pit_mode"] == "x_full_archive_backfill"
    assert records[0]["published_at_pit_eligible"] is True
    assert records[0]["strict_pit_eligible"] is False
    assert records[0]["target_market_ticker"] == "KXOLD-YES"


def test_gdelt_historical_backfill_is_published_at_eligible(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.arena.pit_external_root = str(tmp_path / "missing")
    packet = build_feature_packet(
        {
            "event_ticker": "KXNEWS",
            "market_ticker": "KXNEWS-YES",
            "title": "Will the old news event happen?",
            "category": "Politics",
            "close_time": "2026-03-01T18:00:00Z",
            "outcomes": ["YES", "NO"],
        },
        {"last_price": 0.5},
        as_of="2026-03-01T12:00:00Z",
    )

    def fake_fetch_gdelt(query, as_of_dt, cfg, *, allow_historical_archive=None):
        assert allow_historical_archive is True
        return [
            {
                "source": "gdelt",
                "published_at": "2026-03-01T11:30:00Z",
                "collected_at": "2026-05-16T12:00:00Z",
                "title": "Old news event article",
            }
        ], []

    monkeypatch.setattr("dhruv_gpt_forecasting.pit_evidence._fetch_gdelt", fake_fetch_gdelt)

    records, errors = fetch_external_records_for_packet(
        packet,
        cfg,
        sources={"gdelt"},
        allow_historical_backfill=True,
    )

    assert errors == []
    assert len(records) == 1
    assert records[0]["pit_mode"] == "gdelt_doc_backfill"
    assert records[0]["published_at_pit_eligible"] is True
    assert records[0]["strict_pit_eligible"] is False


def test_binary_evidence_query_drops_yes_no_labels():
    packet = build_feature_packet(
        {
            "event_ticker": "KXGAME",
            "market_ticker": "KXGAME-TEAM",
            "title": "Golden State at Phoenix Winner?",
            "category": "Sports",
            "outcomes": ["YES", "NO"],
        },
        {"last_price": 0.5},
        as_of="2026-03-01T12:00:00Z",
    )

    query = build_evidence_query(packet)

    assert '"YES"' not in query
    assert '"NO"' not in query
    assert "golden" in query
    assert "phoenix" in query


def test_espn_live_capture_records_are_profiled(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.arena.pit_external_root = str(tmp_path / "missing")
    packet = build_feature_packet(
        {
            "event_ticker": "KXGAME",
            "market_ticker": "KXGAME-YES",
            "title": "Golden State at Phoenix NBA winner?",
            "category": "Sports",
            "outcomes": ["YES", "NO"],
        },
        {"last_price": 0.5},
        as_of=datetime.now(timezone.utc).isoformat(),
    )

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "articles": [
                    {
                        "headline": "Golden State injury update before Phoenix game",
                        "description": "Lineup news before tipoff.",
                        "published": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                        "links": {"web": {"href": "https://espn.example/article"}},
                    }
                ]
            }

    monkeypatch.setattr("dhruv_gpt_forecasting.pit_evidence.requests.get", lambda *args, **kwargs: FakeResponse())

    records, errors = fetch_external_records_for_packet(packet, cfg, sources={"espn"})

    assert errors == []
    assert len(records) >= 1
    assert records[0]["source"] == "espn"
    assert records[0]["source_family"] == "sports_news"
    assert records[0]["sentiment_model"] == "lexicon_v1"
    assert records[0]["published_at_pit_eligible"] is True
