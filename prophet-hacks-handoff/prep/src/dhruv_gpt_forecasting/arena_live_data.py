"""Optional live evidence retrieval for Prophet Arena forecasts."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .arena_types import ArenaForecastPacket
from .config import ForecastConfig
from .context import build_related_context_evidence
from .evidence_sources import annotate_evidence_items
from .features import build_feature_packet, parse_dt
from .kalshi_auth import kalshi_auth_headers
from .lseg_query import plan_lseg_news_query
from .pit_evidence import build_evidence_query, gather_pit_external_evidence
from .vendor_evidence import archive_vendor_records, fetch_vendor_records, records_to_live_evidence


KALSHI_BASE_URL = "https://api.elections.kalshi.com"
KALSHI_MARKET_PATH = "/trade-api/v2/markets/{ticker}"
POLYMARKET_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
POLYGON_PREV_AGG_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
ESPN_NEWS_URL = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/news"
ESPN_LEAGUES = (
    ("basketball", "nba"),
    ("football", "nfl"),
    ("baseball", "mlb"),
    ("hockey", "nhl"),
    ("football", "college-football"),
    ("basketball", "mens-college-basketball"),
    ("soccer", "eng.1"),
)


def gather_live_evidence(
    packet: ArenaForecastPacket,
    cfg: ForecastConfig,
    *,
    enabled: bool | None = None,
    deadline_at: float | None = None,
    allow_llm_queries: bool = True,
) -> list[dict[str, Any]]:
    """Return compact live evidence. All failures become audit evidence."""
    if enabled is None:
        enabled = _env_bool("ARENA_ENABLE_LIVE_DATA", cfg.arena.live_data_enabled_default)
    if not enabled or _env_bool("ARENA_DISABLE_LIVE_DATA", False):
        return []

    evidence: list[dict[str, Any]] = []
    historical_as_of = not _is_live_as_of(packet, cfg)
    backtest_internet = _backtest_internet_enabled(cfg)
    if historical_as_of and not backtest_internet:
        return [{
            "source": "live_fetch_error",
            "timestamp": _now(),
            "claim": "Live internet evidence was skipped for a historical as_of because backtest internet mode is disabled.",
            "error": "historical_as_of_live_data_disabled",
        }]

    source_plan = _live_source_plan(packet)
    if historical_as_of:
        source_plan = _historical_publish_date_safe_source_plan(source_plan)
    evidence.append({
        "source": "live_source_plan",
        "timestamp": _now(),
        "claim": "Category-routed live data source plan for this forecast.",
        "category": packet.category,
        "sources": sorted(source_plan),
        "pit_external_sources": cfg.arena.pit_external_sources if "pit_external" in source_plan else [],
        "historical_backtest_internet": historical_as_of,
        "source_date_policy": (
            "historical mode allows only sources with source-specific publish/update timestamps at or before packet.as_of"
            if historical_as_of
            else "live mode"
        ),
    })
    if "pit_external" in source_plan and _can_continue(deadline_at):
        # Live forecasts should not run the full PIT network backfill path by
        # default; category-routed live sources below handle current data. Keep
        # local/archive PIT context, and require an explicit opt-in for the
        # slower historical/social fetch sweep.
        evidence.extend(gather_pit_external_evidence(
            packet,
            cfg,
            allow_network=_env_bool("ARENA_LIVE_PIT_EXTERNAL_NETWORK", False),
        ))
    if "local_linked_markets" in source_plan and _can_continue(deadline_at):
        evidence.extend(_local_linked_market_evidence(packet))
    if "kalshi" in source_plan and _can_continue(deadline_at):
        evidence.extend(_kalshi_market_evidence(packet, cfg, deadline_at=deadline_at))
    if "polymarket" in source_plan and _can_continue(deadline_at):
        evidence.extend(_polymarket_search_evidence(packet, cfg, deadline_at=deadline_at))
    if "espn" in source_plan and _can_continue(deadline_at):
        evidence.extend(_espn_news_evidence(packet, cfg, deadline_at=deadline_at))
    if "oddspipe" in source_plan and _can_continue(deadline_at):
        evidence.extend(_oddspipe_evidence(packet, cfg, deadline_at=deadline_at))
    if "fred" in source_plan and _can_continue(deadline_at):
        evidence.extend(_fred_evidence(packet, cfg, deadline_at=deadline_at))
    if "polygon" in source_plan and _can_continue(deadline_at):
        evidence.extend(_polygon_evidence(packet, cfg, deadline_at=deadline_at))
    if "wrds" in source_plan and _can_continue(deadline_at):
        evidence.extend(_generic_vendor_evidence(packet, cfg, "wrds", deadline_at=deadline_at))
    if "lseg" in source_plan and _can_continue(deadline_at):
        evidence.extend(_generic_vendor_evidence(
            packet,
            cfg,
            "lseg",
            deadline_at=deadline_at,
            allow_llm_query=allow_llm_queries,
        ))
    if not _can_continue(deadline_at):
        evidence.append({
            "source": "live_fetch_error",
            "timestamp": _now(),
            "claim": "Live evidence gathering stopped because the total evidence budget expired.",
            "error": "total_evidence_timeout",
        })
    return annotate_evidence_items(evidence[: cfg.arena.max_live_evidence], packet.category)


def _live_source_plan(packet: ArenaForecastPacket) -> set[str]:
    base = {"pit_external", "local_linked_markets", "kalshi", "polymarket"}
    category = packet.category
    if category == "Sports":
        return base | {"espn", "oddspipe", "lseg"}
    if category in {"Economics", "Financials"}:
        return base | {"fred", "polygon", "wrds", "lseg"}
    if category in {"Crypto", "Commodities"}:
        return base | {"polygon", "fred", "lseg"}
    if category in {"Politics", "Elections"}:
        return base | {"lseg"}
    if category in {"Climate and Weather", "Weather"}:
        return base | {"lseg"}
    return base | {"lseg"}


def _historical_publish_date_safe_source_plan(source_plan: set[str]) -> set[str]:
    unsafe_current_context = {"kalshi", "polymarket", "polygon"}
    return {source for source in source_plan if source not in unsafe_current_context}


def _kalshi_market_evidence(
    packet: ArenaForecastPacket,
    cfg: ForecastConfig,
    *,
    deadline_at: float | None = None,
) -> list[dict[str, Any]]:
    ticker = packet.market_ticker
    if not ticker.startswith("KX"):
        return []
    path = KALSHI_MARKET_PATH.format(ticker=ticker)
    data = _cached_get_json(
        (os.environ.get("KALSHI_BASE_URL") or KALSHI_BASE_URL).rstrip("/") + path,
        params={},
        headers=kalshi_auth_headers("GET", path),
        cfg=cfg,
        source="kalshi_public_market",
        deadline_at=deadline_at,
    )
    if isinstance(data, dict) and data.get("error"):
        return [{
            "source": "kalshi_public_market",
            "timestamp": _now(),
            "claim": "Kalshi market quote retrieval failed; fallback should use any event-payload quote if present.",
            "error": data.get("error"),
        }]
    market = data.get("market") if isinstance(data, dict) else None
    if not isinstance(market, dict):
        return []
    yes_bid = _market_price(market, "yes_bid")
    yes_ask = _market_price(market, "yes_ask")
    no_bid = _market_price(market, "no_bid")
    no_ask = _market_price(market, "no_ask")
    last_price = _market_price(market, "last_price")
    p_yes = None
    if yes_bid is not None and yes_ask is not None:
        p_yes = (yes_bid + yes_ask) / 2.0
    elif yes_ask is not None and no_ask is not None:
        p_yes = (yes_ask + (1.0 - no_ask)) / 2.0
    elif yes_bid is not None and no_bid is not None:
        p_yes = (yes_bid + (1.0 - no_bid)) / 2.0
    elif last_price is not None:
        p_yes = last_price
    if p_yes is None:
        return []
    return [{
        "source": "kalshi_public_market",
        "timestamp": _now(),
        "claim": "Public Kalshi market quote was available for this ticker.",
        "yes_probability": p_yes,
        "raw": {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "last_price": last_price,
            "volume": market.get("volume"),
            "volume_dollars": market.get("volume_dollars"),
            "open_interest": market.get("open_interest"),
            "open_interest_fp": market.get("open_interest_fp"),
        },
    }]


def _local_linked_market_evidence(packet: ArenaForecastPacket) -> list[dict[str, Any]]:
    """Attach local linked-market model output for KX tickers when available."""
    if not packet.market_ticker.startswith("KX"):
        return []
    feature_packet = build_feature_packet(
        {
            "event_ticker": packet.event_ticker,
            "market_ticker": packet.market_ticker,
            "title": packet.title,
            "subtitle": packet.subtitle,
            "description": packet.description,
            "category": packet.category,
            "rules": packet.rules,
            "close_time": packet.close_time,
            "outcomes": packet.outcomes,
        },
        {
            "ticker": packet.market_ticker,
            "event_ticker": packet.event_ticker,
            "snapshot_time": packet.as_of,
            "last_price": 0.5,
        },
        as_of=packet.as_of,
    )
    return [
        item
        for item in build_related_context_evidence(feature_packet)
        if item.get("source") in {"linked_market_model", "kalshi_nonbinary_context", "kalshi_topvol_same_event"}
    ][:4]


def _polymarket_search_evidence(
    packet: ArenaForecastPacket,
    cfg: ForecastConfig,
    *,
    deadline_at: float | None = None,
) -> list[dict[str, Any]]:
    if not packet.title:
        return []
    data = _cached_get_json(
        POLYMARKET_SEARCH_URL,
        params={"q": packet.title, "limit": 3},
        headers={},
        cfg=cfg,
        source="polymarket_public_search",
        deadline_at=deadline_at,
    )
    markets = (data.get("markets") or data.get("results")) if isinstance(data, dict) else None
    if not isinstance(markets, list):
        return []
    compact = []
    for market in markets[:3]:
        if not isinstance(market, dict):
            continue
        compact.append({
            "question": market.get("question") or market.get("title"),
            "condition_id": market.get("conditionId") or market.get("condition_id"),
            "end_date": market.get("endDate") or market.get("end_date_iso"),
            "volume": market.get("volume") or market.get("volume24hr"),
            "liquidity": market.get("liquidity"),
        })
    if not compact:
        return []
    return [{
        "source": "polymarket_public_search",
        "timestamp": _now(),
        "claim": "Similar Polymarket questions were found by public search.",
        "matches": compact,
    }]


def _fred_evidence(
    packet: ArenaForecastPacket,
    cfg: ForecastConfig,
    *,
    deadline_at: float | None = None,
) -> list[dict[str, Any]]:
    key = os.environ.get("FRED_API_KEY")
    if not key or packet.category not in {"Economics", "Crypto", "Commodities"}:
        return []
    series_id = _infer_fred_series(packet)
    if not series_id:
        return []
    data = _cached_get_json(
        FRED_OBSERVATIONS_URL,
        params={
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 3,
            **_fred_pit_params(packet),
        },
        headers={},
        cfg=cfg,
        source=f"fred_{series_id}",
        deadline_at=deadline_at,
    )
    observations = data.get("observations") if isinstance(data, dict) else None
    if not isinstance(observations, list) or not observations:
        return []
    return [{
        "source": "fred",
        "timestamp": _now(),
        "claim": f"Latest FRED observations for {series_id}.",
        "series_id": series_id,
        "observations": observations[:3],
    }]


def _polygon_evidence(
    packet: ArenaForecastPacket,
    cfg: ForecastConfig,
    *,
    deadline_at: float | None = None,
) -> list[dict[str, Any]]:
    key = os.environ.get("POLYGON_API_KEY")
    ticker = _infer_polygon_ticker(packet)
    if not key or not ticker:
        return []
    data = _cached_get_json(
        POLYGON_PREV_AGG_URL.format(ticker=ticker),
        params={"adjusted": "true", "apiKey": key},
        headers={},
        cfg=cfg,
        source=f"polygon_{ticker}",
        deadline_at=deadline_at,
    )
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        return []
    return [{
        "source": "polygon",
        "timestamp": _now(),
        "claim": f"Previous aggregate price for {ticker}.",
        "ticker": ticker,
        "results": results[:2],
    }]


def _espn_news_evidence(
    packet: ArenaForecastPacket,
    cfg: ForecastConfig,
    *,
    deadline_at: float | None = None,
) -> list[dict[str, Any]]:
    if packet.category != "Sports":
        return []
    query = build_evidence_query(packet)
    query_tokens = _tokens(query)
    records = []
    errors = []
    for sport, league in ESPN_LEAGUES:
        if not _can_continue(deadline_at):
            errors.append({"league": league, "error": "total_evidence_timeout"})
            break
        data = _cached_get_json(
            ESPN_NEWS_URL.format(sport=sport, league=league),
            params={"limit": min(20, max(1, cfg.arena.pit_external_max_records))},
            headers={},
            cfg=cfg,
            source=f"espn_{league.replace('.', '_').replace('-', '_')}",
            deadline_at=deadline_at,
        )
        if data.get("error"):
            errors.append({"league": league, "error": data.get("error")})
            continue
        for item in data.get("articles") or []:
            if not isinstance(item, dict):
                continue
            title = item.get("headline") or item.get("title")
            text = item.get("description") or item.get("story")
            record_text = f"{title or ''} {text or ''}"
            if len(_tokens(record_text) & query_tokens) < 2:
                continue
            link = item.get("links", {}).get("web", {}).get("href") if isinstance(item.get("links"), dict) else None
            records.append({
                "title": title,
                "text_excerpt": str(text or "")[:360] or None,
                "published_at": item.get("published") or item.get("lastModified"),
                "url": link,
                "sport": sport,
                "league": league,
            })
    records, discarded = _filter_records_with_publish_dates(records, packet.as_of)
    if not records and not errors:
        return []
    return [{
        "source": "espn",
        "timestamp": _now(),
        "claim": "Relevant ESPN sports news was retrieved for this forecast.",
        "query": query,
        "records": records[: cfg.arena.max_live_evidence],
        "discarded_record_count": discarded,
        "source_date_policy": "records require published_at at or before packet.as_of",
        "errors": errors[:2],
    }]


def _oddspipe_evidence(
    packet: ArenaForecastPacket,
    cfg: ForecastConfig,
    *,
    deadline_at: float | None = None,
) -> list[dict[str, Any]]:
    if packet.category != "Sports":
        return []
    url = os.environ.get("ODDSPIPE_API_URL")
    key = os.environ.get("ODDSPIPE_API_KEY")
    if not url or not key:
        return []
    data = _cached_get_json(
        url,
        params={
            "q": build_evidence_query(packet),
            "as_of": packet.as_of,
            "limit": min(10, cfg.arena.max_live_evidence),
        },
        headers={"Authorization": f"Bearer {key}"},
        cfg=cfg,
        source="oddspipe",
        deadline_at=deadline_at,
    )
    rows = data.get("events") or data.get("odds") or data.get("results")
    if not isinstance(rows, list):
        return []
    compact = []
    for item in rows[: cfg.arena.max_live_evidence]:
        if not isinstance(item, dict):
            continue
        compact.append({
            "event": item.get("event") or item.get("name") or item.get("title"),
            "market": item.get("market") or item.get("market_type"),
            "home_team": item.get("home_team"),
            "away_team": item.get("away_team"),
            "book": item.get("book") or item.get("sportsbook"),
            "odds": item.get("odds") or item.get("prices") or item.get("moneyline"),
            "timestamp": item.get("timestamp") or item.get("as_of") or item.get("updated_at"),
        })
    if not compact:
        return []
    return [{
        "source": "oddspipe",
        "timestamp": _now(),
        "claim": "Sports odds context was retrieved through ODDSPIPE_API_URL.",
        "records": compact,
    }]


def _generic_vendor_evidence(
    packet: ArenaForecastPacket,
    cfg: ForecastConfig,
    source: str,
    *,
    deadline_at: float | None = None,
    allow_llm_query: bool = True,
) -> list[dict[str, Any]]:
    """Optional normalized HTTP connector for licensed WRDS/LSEG-style feeds.

    The package cannot assume a user's licensed endpoint shape. If
    WRDS_API_URL or LSEG_API_URL is supplied, we call it with a small common
    query contract and normalize common result keys.
    """
    query_plan = None
    if source == "lseg":
        query_plan = plan_lseg_news_query(packet, cfg, deadline_at=deadline_at, allow_llm=allow_llm_query)
        query = str(query_plan.get("query") or build_evidence_query(packet))
    else:
        query = build_evidence_query(packet)
    records, errors = fetch_vendor_records(source, query, packet.as_of, cfg, deadline_at=deadline_at)
    attempted_queries = [query]
    if source == "lseg" and query_plan and not records and _can_continue(deadline_at):
        for alternate_query in query_plan.get("alternate_queries") or []:
            alternate = str(alternate_query or "").strip()
            if not alternate or alternate in attempted_queries:
                continue
            alt_records, alt_errors = fetch_vendor_records(source, alternate, packet.as_of, cfg, deadline_at=deadline_at)
            attempted_queries.append(alternate)
            errors.extend(alt_errors)
            if alt_records:
                query = alternate
                records = alt_records
                break
    records, discarded = _filter_records_with_publish_dates(records, packet.as_of)
    archive_vendor_records(records, cfg)
    evidence = records_to_live_evidence(source, records, query=query, max_records=cfg.arena.max_live_evidence)
    if query_plan:
        if evidence:
            evidence[0]["lseg_query_plan"] = {
                key: value for key, value in query_plan.items()
                if key not in {"api_log"}
            }
            evidence[0]["attempted_queries"] = attempted_queries
            if "api_log" in query_plan:
                evidence[0]["lseg_query_api_log"] = query_plan["api_log"]
        else:
            evidence.append({
                "source": "lseg_query_plan",
                "timestamp": _now(),
                "claim": "GPT planned the LSEG query, but LSEG returned no normalized records.",
                "query": query,
                "attempted_queries": attempted_queries,
                "lseg_query_plan": {
                    key: value for key, value in query_plan.items()
                    if key not in {"api_log"}
                },
                "lseg_query_api_log": query_plan.get("api_log"),
            })
    if errors:
        evidence.append({
            "source": source,
            "timestamp": _now(),
            "claim": f"Licensed {source.upper()} evidence retrieval failed.",
            "errors": errors[:3],
        })
    if discarded and not evidence:
        evidence.append({
            "source": source,
            "timestamp": _now(),
            "claim": f"Licensed {source.upper()} records were discarded because they were not PIT publish-date eligible.",
            "discarded_record_count": discarded,
            "error": "no_pit_publish_date_eligible_records",
        })
    return evidence


def _cached_get_json(
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str],
    cfg: ForecastConfig,
    source: str,
    deadline_at: float | None = None,
) -> dict[str, Any]:
    cache_dir = Path(cfg.budget.log_dir) / "live_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        json.dumps({"url": url, "params": _redact_params(params)}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{source}_{cache_key}.json"
    ttl = cfg.arena.live_cache_ttl_seconds
    if cache_path.exists() and time.time() - cache_path.stat().st_mtime <= ttl:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    try:
        response = requests.get(url, params=params, headers=headers, timeout=_source_timeout(cfg, deadline_at))
        response.raise_for_status()
        data = response.json()
        cache_path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
        return data
    except Exception as exc:  # noqa: BLE001 - evidence retrieval must never block forecasts.
        return {
            "source": source,
            "error": f"{type(exc).__name__}:{exc}",
            "timestamp": _now(),
        }


def _redact_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: ("__REDACTED__" if "key" in key.lower() else value) for key, value in params.items()}


def _can_continue(deadline_at: float | None) -> bool:
    return deadline_at is None or time.monotonic() < deadline_at


def _source_timeout(cfg: ForecastConfig, deadline_at: float | None) -> float:
    configured = float(os.environ.get(
        "ARENA_EVIDENCE_SOURCE_TIMEOUT_SECONDS",
        cfg.arena.evidence_source_timeout_seconds,
    ))
    if deadline_at is None:
        return max(0.1, configured)
    remaining = deadline_at - time.monotonic()
    return max(0.1, min(configured, remaining))


def _is_live_as_of(packet: ArenaForecastPacket, cfg: ForecastConfig) -> bool:
    as_of_dt = parse_dt(packet.as_of)
    if as_of_dt is None:
        return False
    now = datetime.now(timezone.utc)
    max_age = max(0, cfg.arena.pit_external_max_live_age_minutes) * 60
    return abs((now - as_of_dt).total_seconds()) <= max_age


def _backtest_internet_enabled(cfg: ForecastConfig) -> bool:
    value = os.environ.get("ARENA_ENABLE_BACKTEST_INTERNET") or os.environ.get("ARENA_ALLOW_BACKTEST_INTERNET")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return cfg.arena.grounded_research_backtest_enabled


def _filter_records_with_publish_dates(
    records: list[dict[str, Any]],
    as_of: str,
) -> tuple[list[dict[str, Any]], int]:
    as_of_dt = parse_dt(as_of)
    if as_of_dt is None:
        return [], len(records)
    eligible: list[dict[str, Any]] = []
    discarded = 0
    for record in records:
        published = parse_dt(str(
            record.get("published_at")
            or record.get("published")
            or record.get("created_at")
            or record.get("timestamp")
            or ""
        ))
        if published is None or published > as_of_dt:
            discarded += 1
            continue
        clean = dict(record)
        clean["published_at"] = published.isoformat()
        eligible.append(clean)
    return eligible, discarded


def _fred_pit_params(packet: ArenaForecastPacket) -> dict[str, str]:
    as_of_dt = parse_dt(packet.as_of)
    if as_of_dt is None:
        return {}
    return {"observation_end": as_of_dt.date().isoformat()}


def _infer_fred_series(packet: ArenaForecastPacket) -> str | None:
    text = f"{packet.title} {packet.description or ''} {packet.rules or ''}".lower()
    if "core cpi" in text:
        return "CPILFESL"
    if "cpi" in text or "inflation" in text:
        return "CPIAUCSL"
    if "unemployment" in text:
        return "UNRATE"
    if "fed" in text or "interest rate" in text:
        return "FEDFUNDS"
    if "gdp" in text:
        return "GDP"
    if "oil" in text:
        return "DCOILWTICO"
    return None


def _infer_polygon_ticker(packet: ArenaForecastPacket) -> str | None:
    text = f"{packet.title} {packet.description or ''} {packet.rules or ''}".lower()
    if "bitcoin" in text or "btc" in text:
        return "X:BTCUSD"
    if "ethereum" in text or "eth" in text:
        return "X:ETHUSD"
    if "solana" in text or " sol " in f" {text} ":
        return "X:SOLUSD"
    return None


def _tokens(text: str) -> set[str]:
    import re

    stop = {"the", "and", "for", "with", "from", "this", "that", "will", "yes", "no"}
    return {tok for tok in re.findall(r"[a-z0-9]+", str(text).lower()) if len(tok) > 2 and tok not in stop}


def _price_to_prob(value: Any) -> float | None:
    if value in (None, ""):
        return None
    raw = float(value)
    if raw > 1.0:
        raw /= 100.0
    return max(0.0, min(1.0, raw))


def _market_price(market: dict[str, Any], key: str) -> float | None:
    value = _price_to_prob(market.get(key))
    if value is not None:
        return value
    return _price_to_prob(market.get(f"{key}_dollars"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
