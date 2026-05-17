import json
from datetime import datetime, timedelta, timezone

from dhruv_gpt_forecasting.arena_live_data import _kalshi_market_evidence, gather_live_evidence
from dhruv_gpt_forecasting.arena_priors import build_arena_packet
from dhruv_gpt_forecasting.config import load_config
from dhruv_gpt_forecasting.pit_evidence import fetch_external_records_for_packet, gather_pit_external_evidence
from dhruv_gpt_forecasting.vendor_evidence import (
    fetch_vendor_records,
    normalize_file,
    normalize_vendor_payload,
    vendor_env_status,
)


def test_normalize_vendor_payload_handles_lseg_fields():
    records = normalize_vendor_payload(
        "lseg",
        {
            "stories": [
                {
                    "storyId": "abc",
                    "headline": "Fed decision preview",
                    "body": "Markets expect a rate hold.",
                    "versionCreated": "2026-03-01T11:30:00Z",
                    "sourceCode": "RTRS",
                    "subjects": ["FED"],
                }
            ]
        },
        collected_at="2026-03-01T11:45:00Z",
    )

    assert records == [
        {
            "source": "lseg",
            "published_at": "2026-03-01T11:30:00Z",
            "collected_at": "2026-03-01T11:45:00Z",
            "title": "Fed decision preview",
            "text": "Markets expect a rate hold.",
            "url": None,
            "vendor_id": "abc",
            "vendor_source": "RTRS",
            "vendor_metadata": {"subjects": ["FED"]},
        }
    ]


def test_vendor_fetch_uses_lseg_app_key_header(monkeypatch):
    monkeypatch.setenv("LSEG_NEWS_API_URL", "https://vendor.example/lseg")
    monkeypatch.setenv("LSEG_APP_KEY", "app-key")
    cfg = load_config()

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"headline": "Election poll update", "published_at": "2026-03-01T11:00:00Z"}]}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("dhruv_gpt_forecasting.vendor_evidence.requests.get", fake_get)

    records, errors = fetch_vendor_records("lseg", "election poll", "2026-03-01T12:00:00Z", cfg)

    assert errors == []
    assert captured["url"] == "https://vendor.example/lseg"
    assert captured["headers"]["App-Key"] == "app-key"
    assert captured["params"]["q"] == "election poll"
    assert records[0]["source"] == "lseg"
    assert records[0]["title"] == "Election poll update"


def test_vendor_fetch_uses_wrds_basic_auth(monkeypatch):
    monkeypatch.setenv("WRDS_NEWS_API_URL", "https://vendor.example/wrds")
    monkeypatch.delenv("WRDS_API_KEY", raising=False)
    monkeypatch.setenv("WRDS_USERNAME", "user")
    monkeypatch.setenv("WRDS_PASSWORD", "password")
    cfg = load_config()

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"title": "Macro release", "date": "2026-03-01T10:00:00Z", "summary": "CPI context."}]}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("dhruv_gpt_forecasting.vendor_evidence.requests.get", fake_get)

    records, errors = fetch_vendor_records("wrds", "cpi", "2026-03-01T12:00:00Z", cfg)

    assert errors == []
    assert captured["auth"] == ("user", "password")
    assert records[0]["source"] == "wrds"
    assert records[0]["text"] == "CPI context."


def test_vendor_status_accepts_current_env_names(monkeypatch):
    monkeypatch.setenv("LSEG_NEWS_API_URL", "https://vendor.example/lseg")
    monkeypatch.setenv("LSEG_APP_KEY_EIKON", "key")
    monkeypatch.setenv("WRDS_NEWS_API_URL", "https://vendor.example/wrds")
    monkeypatch.setenv("WRDS_USERNAME", "user")
    monkeypatch.setenv("WRDS_PASSWORD", "password")

    assert vendor_env_status("lseg")["configured"] is True
    assert vendor_env_status("lseg")["auth_mode"] == "header:App-Key"
    assert vendor_env_status("wrds")["configured"] is True
    assert vendor_env_status("wrds")["auth_mode"] == "basic_auth"


def test_vendor_status_accepts_native_backend_schemes(monkeypatch):
    monkeypatch.setenv("LSEG_NEWS_API_URL", "lseg-data-library://news")
    monkeypatch.setenv("LSEG_APP_KEY", "key")
    monkeypatch.setenv("WRDS_NEWS_API_URL", "wrds-postgres://news")
    monkeypatch.setenv("WRDS_USERNAME", "user")
    monkeypatch.setenv("WRDS_PASSWORD", "password")
    monkeypatch.setenv("WRDS_NEWS_SQL", "select 1")
    monkeypatch.setattr("dhruv_gpt_forecasting.vendor_evidence._module_available", lambda name: True)

    lseg = vendor_env_status("lseg")
    wrds = vendor_env_status("wrds")

    assert lseg["configured"] is True
    assert lseg["backend"] == "lseg_data_library"
    assert wrds["configured"] is True
    assert wrds["backend"] == "wrds_postgres"


def test_historical_live_evidence_requires_backtest_internet(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path / "logs")
    cfg.arena.grounded_research_backtest_enabled = False
    monkeypatch.delenv("ARENA_ENABLE_BACKTEST_INTERNET", raising=False)
    packet = build_arena_packet({
        "event_ticker": "task-history",
        "market_ticker": "task-history",
        "title": "Will the historical event happen?",
        "category": "Politics",
        "as_of": "2025-10-01T00:00:00Z",
        "close_time": "2025-10-02T00:00:00Z",
        "outcomes": ["YES", "NO"],
    })

    evidence = gather_live_evidence(packet, cfg, enabled=True)

    assert evidence[0]["error"] == "historical_as_of_live_data_disabled"


def test_live_evidence_pit_external_network_is_opt_in(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path / "logs")
    cfg.arena.pit_external_root = str(tmp_path / "external_evidence")
    packet = build_arena_packet({
        "event_ticker": "KXLIVE",
        "market_ticker": "KXLIVE-YES",
        "title": "Will the live event happen?",
        "category": "Politics",
        "close_time": "2026-03-01T18:00:00Z",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "outcomes": ["YES", "NO"],
    })
    captured = []

    def fake_pit(packet, cfg, *, allow_network=None, **kwargs):
        captured.append(allow_network)
        return []

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data._live_source_plan", lambda packet: {"pit_external"})
    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data.gather_pit_external_evidence", fake_pit)
    monkeypatch.delenv("ARENA_LIVE_PIT_EXTERNAL_NETWORK", raising=False)
    gather_live_evidence(packet, cfg, enabled=True)
    monkeypatch.setenv("ARENA_LIVE_PIT_EXTERNAL_NETWORK", "1")
    gather_live_evidence(packet, cfg, enabled=True)

    assert captured == [False, True]


def test_kalshi_market_evidence_parses_dollar_quote_fields(monkeypatch):
    cfg = load_config()
    packet = build_arena_packet({
        "event_ticker": "KXNBAGAME-26MAY18SASOKC",
        "market_ticker": "KXNBAGAME-26MAY18SASOKC-OKC",
        "title": "Game 1: San Antonio at Oklahoma City Winner?",
        "category": "Sports",
        "close_time": "2026-06-02T00:30:00Z",
        "outcomes": ["YES", "NO"],
    })

    def fake_get(*args, **kwargs):
        return {
            "market": {
                "yes_bid_dollars": "0.6700",
                "yes_ask_dollars": "0.6800",
                "no_bid_dollars": "0.3200",
                "no_ask_dollars": "0.3300",
                "last_price_dollars": "0.6800",
                "volume": 100,
            }
        }

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data._cached_get_json", fake_get)
    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data.kalshi_auth_headers", lambda *args: {})

    evidence = _kalshi_market_evidence(packet, cfg)

    assert evidence[0]["source"] == "kalshi_public_market"
    assert evidence[0]["yes_probability"] == 0.675
    assert evidence[0]["raw"]["yes_bid"] == 0.67
    assert evidence[0]["raw"]["no_ask"] == 0.33


def test_historical_vendor_evidence_filters_publish_dates(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path / "logs")
    cfg.arena.pit_external_root = str(tmp_path / "external_evidence")
    cfg.arena.grounded_research_backtest_enabled = True
    packet = build_arena_packet({
        "event_ticker": "task-history-vendor",
        "market_ticker": "task-history-vendor",
        "title": "Will the historical macro event happen?",
        "category": "Economics",
        "as_of": "2025-10-01T00:00:00Z",
        "close_time": "2025-10-02T00:00:00Z",
        "outcomes": ["YES", "NO"],
    })

    def fake_plan(packet, cfg, *, deadline_at=None, allow_llm=True):
        return {"query": "macro", "alternate_queries": [], "source": "deterministic_lseg_query"}

    def fake_fetch(source, query, as_of, cfg, *, deadline_at=None):
        return [
            {
                "source": "lseg",
                "published_at": "2025-09-30T23:00:00Z",
                "title": "Usable pre-cutoff headline",
            },
            {
                "source": "lseg",
                "published_at": "2025-10-01T01:00:00Z",
                "title": "Post-cutoff headline",
            },
            {"source": "lseg", "title": "Undated headline"},
        ], []

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data._live_source_plan", lambda packet: {"lseg"})
    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data.plan_lseg_news_query", fake_plan)
    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data.fetch_vendor_records", fake_fetch)

    evidence = gather_live_evidence(packet, cfg, enabled=True)
    lseg = [item for item in evidence if item.get("source") == "lseg"]

    assert lseg[0]["records"][0]["title"] == "Usable pre-cutoff headline"
    assert len(lseg[0]["records"]) == 1


def test_vendor_fetch_uses_lseg_data_library_backend(monkeypatch):
    monkeypatch.setenv("LSEG_NEWS_API_URL", "lseg-data-library://news")
    monkeypatch.setenv("LSEG_APP_KEY", "app-key")
    cfg = load_config()
    captured = {}

    class News:
        def get_headlines(self, **kwargs):
            captured["headline_kwargs"] = kwargs
            return [
                {
                    "Headline": "Fed decision preview",
                    "Date": "2026-03-01T11:00:00Z",
                    "StoryId": "story-1",
                    "Source": "RTRS",
                }
            ]

    class FakeLseg:
        news = News()

        def open_session(self, **kwargs):
            captured["session_kwargs"] = kwargs

    monkeypatch.setattr("dhruv_gpt_forecasting.vendor_evidence.importlib.import_module", lambda name: FakeLseg())

    records, errors = fetch_vendor_records("lseg", "fed decision", "2026-03-01T12:00:00Z", cfg)

    assert errors == []
    assert captured["session_kwargs"]["app_key"] == "app-key"
    assert captured["headline_kwargs"]["query"] == "fed decision"
    assert records[0]["source"] == "lseg"
    assert records[0]["title"] == "Fed decision preview"
    assert records[0]["vendor_id"] == "story-1"


def test_vendor_fetch_wrds_postgres_requires_sql(monkeypatch):
    monkeypatch.setenv("WRDS_NEWS_API_URL", "wrds-postgres://news")
    monkeypatch.setenv("WRDS_USERNAME", "user")
    monkeypatch.setenv("WRDS_PASSWORD", "password")
    monkeypatch.delenv("WRDS_NEWS_SQL", raising=False)
    monkeypatch.delenv("WRDS_NEWS_SQL_FILE", raising=False)
    cfg = load_config()

    records, errors = fetch_vendor_records("wrds", "cpi", "2026-03-01T12:00:00Z", cfg)

    assert records == []
    assert errors[0]["error"] == "missing_wrds_news_sql"


def test_vendor_fetch_uses_wrds_postgres_backend(monkeypatch):
    monkeypatch.setenv("WRDS_NEWS_API_URL", "wrds-postgres://news")
    monkeypatch.setenv("WRDS_USERNAME", "user")
    monkeypatch.setenv("WRDS_PASSWORD", "password")
    monkeypatch.setenv("WRDS_NEWS_SQL", "select * from news where published_at <= %(as_of)s limit %(limit)s")
    cfg = load_config()
    captured = {}

    class Connection:
        def __init__(self, **kwargs):
            captured["connection_kwargs"] = kwargs

        def _Connection__make_sa_engine_conn(self, raise_err=False):
            captured["connected"] = True

        def raw_sql(self, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params
            return [{"headline": "Macro release", "published_at": "2026-03-01T10:00:00Z", "summary": "CPI context."}]

        def close(self):
            captured["closed"] = True

    class FakeWrds:
        pass

    FakeWrds.Connection = Connection

    monkeypatch.setattr("dhruv_gpt_forecasting.vendor_evidence.importlib.import_module", lambda name: FakeWrds())

    records, errors = fetch_vendor_records("wrds", "cpi", "2026-03-01T12:00:00Z", cfg)

    assert errors == []
    assert captured["connection_kwargs"]["wrds_username"] == "user"
    assert captured["params"]["query"] == "cpi"
    assert captured["closed"] is True
    assert records[0]["source"] == "wrds"
    assert records[0]["text"] == "CPI context."


def test_vendor_normalize_file_writes_jsonl(tmp_path):
    input_path = tmp_path / "lseg.csv"
    output_path = tmp_path / "lseg.jsonl"
    input_path.write_text(
        "headline,body,versionCreated,storyId\n"
        "Fed preview,Rate context,2026-03-01T11:00:00Z,story-1\n",
        encoding="utf-8",
    )

    n = normalize_file("lseg", input_path, output_path, collected_at="2026-03-01T11:30:00Z")

    assert n == 1
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["source"] == "lseg"
    assert row["title"] == "Fed preview"
    assert row["collected_at"] == "2026-03-01T11:30:00Z"


def test_pit_external_replays_normalized_vendor_archive(tmp_path):
    root = tmp_path / "external_evidence"
    root.mkdir()
    (root / "lseg.jsonl").write_text(
        json.dumps({
            "source": "lseg",
            "market_ticker": "KXFED-YES",
            "published_at": "2026-03-01T11:00:00Z",
            "collected_at": "2026-03-01T11:15:00Z",
            "title": "Fed decision preview",
            "text": "Fed decision context before cutoff.",
        }) + "\n",
        encoding="utf-8",
    )
    cfg = load_config()
    cfg.arena.pit_external_root = str(root)
    packet = build_arena_packet({
        "event_ticker": "KXFED",
        "market_ticker": "KXFED-YES",
        "title": "Will the Fed hold rates?",
        "category": "Economics",
        "close_time": "2026-03-01T18:00:00Z",
        "as_of": "2026-03-01T12:00:00Z",
        "outcomes": ["YES", "NO"],
    })

    evidence = gather_pit_external_evidence(packet, cfg, enabled=True, allow_network=False)

    assert evidence[0]["source"] == "pit_external_evidence"
    assert evidence[0]["records"][0]["source"] == "lseg"
    assert evidence[0]["source_profiles"]["lseg"]["family"] == "licensed_news_and_market_data"


def test_vendor_backfill_records_are_not_strict_pit(monkeypatch):
    cfg = load_config()
    packet = build_arena_packet({
        "event_ticker": "KXFED",
        "market_ticker": "KXFED-YES",
        "title": "Will the Fed hold rates?",
        "category": "Economics",
        "close_time": "2026-03-01T18:00:00Z",
        "as_of": "2026-03-01T12:00:00Z",
        "outcomes": ["YES", "NO"],
    })

    def fake_fetch(source, query, as_of, cfg, *, deadline_at=None):
        return [
            {
                "source": source,
                "published_at": "2026-03-01T11:00:00Z",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "title": "Fed decision preview",
            }
        ], []

    monkeypatch.setattr("dhruv_gpt_forecasting.pit_evidence.fetch_vendor_records", fake_fetch)

    records, errors = fetch_external_records_for_packet(packet, cfg, sources={"lseg"}, allow_historical_backfill=True)

    assert errors == []
    assert records[0]["pit_mode"] == "lseg_published_at_backfill"
    assert records[0]["published_at_pit_eligible"] is True
    assert records[0]["strict_pit_eligible"] is False


def test_live_arena_vendor_evidence_reaches_packet(monkeypatch, tmp_path):
    monkeypatch.setenv("LSEG_NEWS_API_URL", "https://vendor.example/lseg")
    monkeypatch.setenv("LSEG_API_KEY", "key")
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path / "logs")
    cfg.arena.pit_external_root = str(tmp_path / "external_evidence")
    packet = build_arena_packet({
        "event_ticker": "KXFED",
        "market_ticker": "KXFED-YES",
        "title": "Will the Fed hold rates?",
        "category": "Economics",
        "close_time": "2026-03-01T18:00:00Z",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "outcomes": ["YES", "NO"],
    })

    def fake_fetch(source, query, as_of, cfg, *, deadline_at=None):
        return [
            {
                "source": "lseg",
                "published_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "title": "Fed decision preview",
                "text": "Professional news context.",
            }
        ], []

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data.fetch_vendor_records", fake_fetch)
    monkeypatch.setattr("dhruv_gpt_forecasting.arena_live_data._live_source_plan", lambda packet: {"lseg"})

    evidence = gather_live_evidence(packet, cfg, enabled=True)

    lseg = [item for item in evidence if item.get("source") == "lseg"]
    assert lseg
    assert lseg[0]["records"][0]["title"] == "Fed decision preview"
