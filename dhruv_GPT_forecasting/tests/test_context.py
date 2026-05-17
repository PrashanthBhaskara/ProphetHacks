import csv
import gzip
import json

import dhruv_gpt_forecasting.context as context_mod
from dhruv_gpt_forecasting.context import build_related_context_evidence
from dhruv_gpt_forecasting.features import build_feature_packet


def test_nonbinary_context_uses_pre_as_of_candles_without_settlement_fields(tmp_path, monkeypatch):
    nonbinary_root = _write_nonbinary_context_fixture(tmp_path / "nonbinary")
    _patch_context_roots(monkeypatch, nonbinary_root=nonbinary_root, topvol_root=tmp_path / "topvol", poly_root=tmp_path / "poly")
    packet = build_feature_packet(
        {
            "event_ticker": "KXFIXTURE",
            "market_ticker": "KXFIXTURE-DEN",
            "title": "Denver vs Philadelphia Winner?",
            "close_time": "2026-01-06T05:49:40Z",
            "outcomes": ["YES", "NO"],
        },
        {
            "ticker": "KXFIXTURE-DEN",
            "yes_bid": 0.18,
            "yes_ask": 0.35,
            "no_ask": 0.82,
            "snapshot_time": "2026-01-01T00:30:00Z",
        },
    )
    evidence = build_related_context_evidence(packet)
    linked = next(item for item in evidence if item["source"] == "linked_market_model")
    assert linked["probabilities"]["YES"] > 0
    assert linked["component_distribution"]
    assert linked["inferred_structure"] in {
        "mutually_exclusive_component_distribution",
        "soft_component_distribution",
        "single_linked_market_quote",
    }
    context = next(item for item in evidence if item["source"] == "kalshi_nonbinary_context")
    assert context["components"]
    assert context["derived"]["priced_component_count"] > 0
    assert context["derived"]["target_normalized_probability"] is not None
    assert context["derived"]["normalized_distribution_top"]
    forbidden = {"result", "settlement_ts", "status", "yes_bid_dollars", "yes_ask_dollars"}
    for component in context["components"]:
        assert forbidden.isdisjoint(component)
        assert "pre_as_of_quote" in component


def test_polymarket_mapping_context_is_attached_for_exact_ticker(tmp_path, monkeypatch):
    poly_root = _write_polymarket_fixture(tmp_path / "poly")
    _patch_context_roots(monkeypatch, nonbinary_root=tmp_path / "nonbinary", topvol_root=tmp_path / "topvol", poly_root=poly_root)
    packet = build_feature_packet(
        {
            "event_ticker": "KXBOND-30",
            "market_ticker": "KXBOND-30-CAL",
            "title": "Will Callum Turner be the next James Bond?",
            "outcomes": ["YES", "NO"],
        },
        {
            "ticker": "KXBOND-30-CAL",
            "yes_bid": 0.1,
            "yes_ask": 0.2,
            "no_ask": 0.9,
            "snapshot_time": "2026-05-16T00:00:00Z",
        },
    )
    evidence = build_related_context_evidence(packet)
    polymarket = next(item for item in evidence if item["source"] == "kalshi_polymarket_map")
    assert polymarket["matches"][0]["poly_question"]
    assert polymarket["matches"][0]["poly_condition_id"]


def _patch_context_roots(monkeypatch, *, nonbinary_root, topvol_root, poly_root):
    monkeypatch.setattr(context_mod, "NONBINARY_ROOT", nonbinary_root)
    monkeypatch.setattr(context_mod, "TOPVOL_ROOT", topvol_root)
    monkeypatch.setattr(context_mod, "POLYMARKET_ROOT", poly_root)
    context_mod._links_by_target.cache_clear()
    context_mod._links_by_event.cache_clear()
    context_mod._components_for_group.cache_clear()
    context_mod._topvol_by_event.cache_clear()
    context_mod._poly_map_by_kalshi.cache_clear()
    context_mod._poly_rejections_by_kalshi.cache_clear()


def _write_nonbinary_context_fixture(root):
    week = "2026-01-01"
    group_key = "fixture-game"
    target = "KXFIXTURE-DEN"
    _write_jsonl(
        root / "indexes" / "target_to_context_links.jsonl",
        [{
            "target_ticker": target,
            "event_ticker": "KXFIXTURE",
            "week": week,
            "context_group_key": group_key,
            "relation": "same_event",
        }],
    )
    components = [
        {
            "ticker": target,
            "event_ticker": "KXFIXTURE",
            "title": "Denver vs Philadelphia Winner?",
            "yes_sub_title": "Denver",
            "rules_primary": "Resolves to the official winner.",
            "_context_group_key": group_key,
            "_context_component_rank": 1,
            "result": "yes",
            "settlement_ts": "2026-01-02T00:00:00Z",
            "yes_bid_dollars": 0.9,
        },
        {
            "ticker": "KXFIXTURE-PHI",
            "event_ticker": "KXFIXTURE",
            "title": "Denver vs Philadelphia Winner?",
            "yes_sub_title": "Philadelphia",
            "rules_primary": "Resolves to the official winner.",
            "_context_group_key": group_key,
            "_context_component_rank": 2,
            "result": "no",
            "settlement_ts": "2026-01-02T00:00:00Z",
            "yes_bid_dollars": 0.1,
        },
    ]
    _write_jsonl(root / "markets" / f"{week}_component_markets.jsonl", components)
    _write_candles(root, week, target, base_price=52)
    _write_candles(root, week, "KXFIXTURE-PHI", base_price=42)
    return root


def _write_polymarket_fixture(root):
    root.mkdir(parents=True)
    with (root / "map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kalshi_ticker",
                "poly_condition_id",
                "poly_outcome",
                "kalshi_question",
                "poly_question",
                "poly_end_date",
                "poly_vol_24h",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "kalshi_ticker": "KXBOND-30-CAL",
            "poly_condition_id": "condition-1",
            "poly_outcome": "Yes",
            "kalshi_question": "Will Callum Turner be the next James Bond?",
            "poly_question": "Will Callum Turner be announced as James Bond?",
            "poly_end_date": "2026-12-31",
            "poly_vol_24h": "1000",
        })
    return root


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_candles(root, week: str, ticker: str, *, base_price: int):
    path = root / "ohlcv" / "period_1m" / f"week={week}" / f"{ticker}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "end_period_time",
                "end_period_ts",
                "yes_bid_close",
                "yes_ask_close",
                "price_close",
                "volume",
                "open_interest",
            ],
        )
        writer.writeheader()
        for minute in (0, 16, 32):
            writer.writerow({
                "end_period_time": f"2026-01-01T00:{minute:02d}:00Z",
                "end_period_ts": str(1767225600 + minute * 60),
                "yes_bid_close": str(base_price + minute),
                "yes_ask_close": str(base_price + minute + 4),
                "price_close": str(base_price + minute + 2),
                "volume": str(10 + minute),
                "open_interest": str(100 + minute),
            })
