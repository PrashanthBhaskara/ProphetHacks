from dhruv_gpt_forecasting.arena_priors import build_arena_packet
from dhruv_gpt_forecasting.kalshi_contracts import parse_kalshi_multileg_contract
from dhruv_gpt_forecasting.pit_evidence import build_evidence_query


def test_parse_kalshi_multileg_contract_extracts_joint_legs():
    event = {
        "title": (
            "yes San Antonio,yes Over 202.5 points scored,"
            "no Oklahoma City wins by over 8.5 points,yes Victor Wembanyama: 1+"
        ),
        "outcomes": ["YES", "NO"],
    }

    parsed = parse_kalshi_multileg_contract(event)

    assert parsed["is_multileg"] is True
    assert parsed["component_count"] == 4
    assert parsed["legs"][0] == {
        "index": 1,
        "side": "YES",
        "condition": "San Antonio",
        "search_term": "San Antonio",
        "raw": "yes San Antonio",
    }
    assert parsed["legs"][2]["side"] == "NO"
    assert parsed["legs"][2]["search_term"] == "Oklahoma City"
    assert parsed["legs"][3]["search_term"] == "Victor Wembanyama"
    assert "every component leg" in parsed["joint_yes_semantics"]


def test_arena_packet_exposes_kalshi_multileg_semantics():
    event = {
        "event_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-S2026",
        "market_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-S2026-ABC",
        "title": "yes San Antonio,yes Cade Cunningham: 20+,yes Oklahoma City wins by over 1.5 points",
        "category": "Sports",
        "rules": "Resolves Yes if every listed leg occurs.",
        "outcomes": ["YES", "NO"],
    }

    packet = build_arena_packet(event, include_historical_analogs=False)

    contract = packet.extracted_entities["kalshi_multileg_contract"]
    assert contract["is_multileg"] is True
    assert contract["component_count"] == 3
    assert contract["contract_format"] == "kalshi_comma_separated_yes_no_legs"
    assert contract["search_terms"] == ["San Antonio", "Cade Cunningham", "Oklahoma City"]


def test_multileg_evidence_query_uses_meaningful_entities_not_raw_yes_no():
    event = {
        "event_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-S2026",
        "market_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-S2026-ABC",
        "title": "yes San Antonio,yes Cade Cunningham: 20+,yes Oklahoma City wins by over 1.5 points",
        "category": "Sports",
        "outcomes": ["YES", "NO"],
    }
    packet = build_arena_packet(event, include_historical_analogs=False)

    query = build_evidence_query(packet)

    assert '"San Antonio"' in query
    assert '"Cade Cunningham"' in query
    assert '"Oklahoma City"' in query
    assert "yes" not in query.lower()
