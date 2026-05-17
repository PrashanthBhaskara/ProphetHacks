from dhruv_gpt_forecasting.context import build_related_context_evidence
from dhruv_gpt_forecasting.features import build_feature_packet


def test_nonbinary_context_uses_pre_as_of_candles_without_settlement_fields():
    packet = build_feature_packet(
        {
            "event_ticker": "KXNBAGAME-26JAN05DENPHI",
            "market_ticker": "KXNBAGAME-26JAN05DENPHI-DEN",
            "title": "Denver vs Philadelphia Winner?",
            "close_time": "2026-01-06T05:49:40Z",
            "outcomes": ["YES", "NO"],
        },
        {
            "ticker": "KXNBAGAME-26JAN05DENPHI-DEN",
            "yes_bid": 0.18,
            "yes_ask": 0.35,
            "no_ask": 0.82,
            "snapshot_time": "2026-01-05T00:00:00Z",
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


def test_polymarket_mapping_context_is_attached_for_exact_ticker():
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
