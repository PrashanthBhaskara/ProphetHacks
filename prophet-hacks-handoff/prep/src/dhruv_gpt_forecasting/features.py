"""Feature extraction from Prophet Arena, Kalshi, and candle-shaped data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schemas import EventStructure, FeaturePacket, MarketQuote, clamp_prob


SPORT_PREFIXES = (
    "KXNBA",
    "KXWNBA",
    "KXNCAA",
    "KXNFL",
    "KXMLB",
    "KXNHL",
    "KXATPMATCH",
    "KXATPCHALLENGER",
    "KXWTAMATCH",
    "KXWTA",
    "KXEPL",
    "KXUCL",
    "KXUEL",
    "KXLALIGA",
    "KXSERIEA",
    "KXBUNDESLIGA",
    "KXLIGUE1",
    "KXAFCON",
    "KXCONCACAF",
    "KXEFLCUP",
    "KXIPLGAME",
    "KXT20MATCH",
    "KXUFC",
    "KXBOXING",
    "KXPGATOUR",
    "KXWOMHOCKEY",
    "KXUNITEDCUPMATCH",
    "KXMVESPORTS",
    "KXMVSPORTS",
    "KXSPORTS",
)
WEATHER_PREFIXES = ("KXRAIN", "KXTEMP", "KXSNOW", "KXHURR")
ECON_PREFIXES = ("KXFED", "KXCPI", "KXGDP", "KXJOBS", "KX30Y", "KXOIL", "KXNATGAS")
CRYPTO_PREFIXES = ("KXBTC", "KXETH", "KXSOL", "KXXRP", "KXDOGE")
POLITICS_PREFIXES = ("KXPRES", "KXSENATE", "KXHOUSE", "KXGOV", "KXELEC")
ENTERTAINMENT_PREFIXES = (
    "KXOSCARS",
    "KXOSCAR",
    "KXSUPERBOWLAD",
    "KXFIRSTSUPERBOWLSONG",
    "KXNETFLIX",
    "KXLATENIGHTMENTION",
    "KXTOPMODEL",
    "KXPERFORMSUPERBOWL",
)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def price_to_prob(value: Any) -> float | None:
    if value is None or value == "":
        return None
    raw = float(value)
    if raw > 1.0:
        raw /= 100.0
    return clamp_prob(raw)


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if data.get(key) is not None:
            return data.get(key)
    return None


def quote_from_market_info(market_info: dict[str, Any] | None) -> MarketQuote:
    data = market_info or {}
    return MarketQuote(
        yes_bid=price_to_prob(_first_present(data, ("yes_bid", "yes_bid_dollars", "best_bid"))),
        yes_ask=price_to_prob(_first_present(data, ("yes_ask", "yes_ask_dollars", "best_ask"))),
        no_bid=price_to_prob(_first_present(data, ("no_bid", "no_bid_dollars"))),
        no_ask=price_to_prob(_first_present(data, ("no_ask", "no_ask_dollars"))),
        last_price=price_to_prob(_first_present(data, ("last_price", "last_price_dollars", "price_close"))),
        volume=_float_or_none(_first_present(data, ("volume", "volume_fp", "weekly_volume"))),
        open_interest=_float_or_none(_first_present(data, ("open_interest", "open_interest_fp"))),
        liquidity=_float_or_none(_first_present(data, ("liquidity", "liquidity_dollars"))),
        snapshot_time=_first_present(data, ("snapshot_time", "end_period_time", "as_of")),
    )


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def normalize_category(category: str | None, event_ticker: str | None = None) -> str:
    if category and category != "Other":
        if category == "Climate and Weather":
            return "Climate and Weather"
        return category
    prefix = (event_ticker or "").split("-")[0].upper()
    if prefix.startswith(SPORT_PREFIXES):
        return "Sports"
    if prefix.startswith(WEATHER_PREFIXES):
        return "Climate and Weather"
    if prefix.startswith(ECON_PREFIXES):
        return "Economics"
    if prefix.startswith(CRYPTO_PREFIXES):
        return "Crypto"
    if prefix.startswith(POLITICS_PREFIXES):
        return "Politics"
    if prefix.startswith(ENTERTAINMENT_PREFIXES):
        return "Entertainment"
    return category or "Other"


def classify_event_structure(outcomes: list[str], title: str = "", rules: str | None = None) -> EventStructure:
    labels = " ".join(outcomes + [title, rules or ""]).lower()
    if [str(outcome).casefold() for outcome in outcomes] == ["yes", "no"]:
        return "binary"
    if any(token in labels for token in ("above", "over", "under", "below", "at least", "less than", ">")):
        return "threshold_ladder"
    if any(token in labels for token in ("range", "between")) or any("-" in outcome for outcome in outcomes):
        return "range_bucket"
    if len(outcomes) > 1:
        return "mutually_exclusive"
    return "independent_binary"


def trajectory_from_snapshots(snapshots: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for snap in snapshots or []:
        quote = quote_from_market_info(snap)
        points.append({
            "t": snap.get("t") or snap.get("snapshot_time") or snap.get("end_period_time"),
            "market_mid": quote.market_mid,
            "yes_bid": quote.yes_bid,
            "yes_ask": quote.yes_ask,
            "no_ask": quote.no_ask,
            "last_price": quote.last_price,
            "spread": quote.spread,
            "volume": quote.volume,
            "open_interest": quote.open_interest,
        })
    points = [point for point in points if point.get("market_mid") is not None]
    points.sort(key=lambda point: point.get("t") or "")
    return points


def build_feature_packet(
    event: dict[str, Any],
    market_info: dict[str, Any] | None = None,
    *,
    price_trajectory: list[dict[str, Any]] | None = None,
    external_evidence: list[dict[str, Any]] | None = None,
    as_of: str | None = None,
) -> FeaturePacket:
    quote = quote_from_market_info(market_info)
    trajectory = trajectory_from_snapshots(price_trajectory or (market_info or {}).get("snapshots"))
    if trajectory:
        last = trajectory[-1]
        quote = MarketQuote(
            yes_bid=last.get("yes_bid"),
            yes_ask=last.get("yes_ask"),
            no_ask=last.get("no_ask"),
            last_price=last.get("last_price") or last.get("market_mid"),
            volume=last.get("volume"),
            open_interest=last.get("open_interest"),
            snapshot_time=last.get("t"),
        )
    as_of_value = as_of or quote.snapshot_time or datetime.now(timezone.utc).isoformat()
    close_time = event.get("close_time") or market_info and market_info.get("close_time")
    as_of_dt = parse_dt(as_of_value)
    close_dt = parse_dt(close_time)
    horizon_hours = None
    if as_of_dt and close_dt:
        horizon_hours = max(0.0, (close_dt - as_of_dt).total_seconds() / 3600.0)
    outcomes = list(event.get("outcomes") or ["YES", "NO"])
    category = normalize_category(event.get("category"), event.get("event_ticker"))
    event_structure = classify_event_structure(outcomes, event.get("title") or "", event.get("rules"))
    momentum = 0.0
    if len(trajectory) >= 2:
        momentum = float(trajectory[-1]["market_mid"] - trajectory[0]["market_mid"])
    features = {
        "market_prior": quote.market_mid,
        "spread": quote.spread,
        "n_snapshots": len(trajectory),
        "price_momentum": momentum,
        "abs_price_momentum": abs(momentum),
        "volume": quote.volume,
        "open_interest": quote.open_interest,
        "liquidity": quote.liquidity,
    }
    return FeaturePacket(
        as_of=as_of_value,
        event_ticker=event.get("event_ticker") or market_info and market_info.get("event_ticker") or "",
        market_ticker=event.get("market_ticker") or market_info and market_info.get("ticker") or "",
        title=event.get("title") or market_info and market_info.get("title") or "",
        subtitle=event.get("subtitle") or market_info and market_info.get("subtitle"),
        rules=event.get("rules") or market_info and market_info.get("rules_primary"),
        category=category,
        close_time=close_time,
        outcomes=outcomes,
        quote=quote,
        price_trajectory=trajectory,
        horizon_hours=horizon_hours,
        event_structure=event_structure,
        evidence_digest=external_evidence or [],
        features=features,
    )
