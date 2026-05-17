from dhruv_gpt_forecasting.features import build_feature_packet, classify_event_structure
from dhruv_gpt_forecasting.features import normalize_category


def test_feature_packet_uses_quote_mid_and_spread():
    event = {
        "event_ticker": "KXNBAGAME-26MAY07LALOKC",
        "market_ticker": "KXNBAGAME-26MAY07LALOKC-LAL",
        "title": "Game 2 winner?",
        "rules": "If Los Angeles wins, resolves Yes.",
        "close_time": "2026-05-08T04:20:34Z",
    }
    market_info = {
        "yes_bid": 0.12,
        "yes_ask": 0.13,
        "last_price": 0.13,
        "snapshot_time": "2026-05-07T00:00:00Z",
    }
    packet = build_feature_packet(event, market_info)
    assert round(packet.quote.market_mid, 3) == 0.125
    assert round(packet.quote.spread or 0.0, 3) == 0.010
    assert packet.category == "Sports"
    assert packet.horizon_hours is not None


def test_structure_classifier_detects_thresholds():
    assert classify_event_structure(["Above $450", "Above $500"], "RWA market cap") == "threshold_ladder"
    assert classify_event_structure(["YES", "NO"], "Will it rain?") == "binary"


def test_top_volume_sports_prefixes_are_classified():
    assert normalize_category(None, "KXNCAAMBGAME-26JAN01ABCXYZ") == "Sports"
    assert normalize_category(None, "KXPGATOUR-MAST26-SLOW") == "Sports"
