"""Point-in-time external evidence retrieval and filtering.

External social/search data is only useful for OOS tests if it is auditable.
This module accepts locally archived JSONL records and optional live fetches,
then filters every record against the packet's forecast timestamp before it can
reach GPT.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from .config import PACKAGE_ROOT, ForecastConfig
from .evidence_sources import annotate_evidence_items, compact_source_profile
from .features import parse_dt
from .sentiment import aggregate_sentiment, annotate_record_sentiment
from .vendor_evidence import fetch_vendor_records


REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
X_RECENT_SEARCH_URL = "https://api.x.com/2/tweets/search/recent"
X_FULL_ARCHIVE_SEARCH_URL = "https://api.x.com/2/tweets/search/all"
ESPN_NEWS_URL = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/news"
TOKEN_RE = re.compile(r"[a-z0-9]+")
ESPN_LEAGUES = (
    ("basketball", "nba"),
    ("football", "nfl"),
    ("baseball", "mlb"),
    ("hockey", "nhl"),
    ("football", "college-football"),
    ("basketball", "mens-college-basketball"),
    ("soccer", "eng.1"),
)


def gather_pit_external_evidence(
    packet: Any,
    cfg: ForecastConfig,
    *,
    enabled: bool | None = None,
    allow_network: bool | None = None,
    strict_collected_at: bool | None = None,
) -> list[dict[str, Any]]:
    """Return compact PIT-safe evidence for a forecast packet.

    For historical OOS timestamps, network fetches are disabled by default. The
    backtest path should use locally archived rows with collection timestamps.
    For live forecasts, Reddit/GDELT/ESPN fetches can run when enabled. X code
    remains available only for explicit legacy experiments and is not enabled
    in the default source list.
    """
    if enabled is None:
        enabled = _env_bool("FORECAST_ENABLE_PIT_EXTERNAL", cfg.arena.pit_external_enabled_default)
    if not enabled or _env_bool("FORECAST_DISABLE_PIT_EXTERNAL", False):
        return []

    as_of_dt = _packet_as_of(packet)
    if as_of_dt is None:
        return [{
            "source": "pit_external_evidence_error",
            "timestamp": _now(),
            "claim": "External evidence disabled because forecast as_of timestamp could not be parsed.",
        }]

    strict = (
        cfg.arena.pit_external_strict_collected_at
        if strict_collected_at is None
        else strict_collected_at
    )
    query = build_evidence_query(packet)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    sources = set(cfg.arena.pit_external_sources)

    if "local_jsonl" in sources:
        records.extend(_load_local_records(packet, cfg, query))

    if allow_network is None:
        allow_network = _env_bool("PIT_EXTERNAL_ALLOW_NETWORK", _is_live_as_of(as_of_dt, cfg))

    if allow_network and "reddit" in sources and _is_live_as_of(as_of_dt, cfg):
        fetched, fetch_errors = _fetch_reddit(query, cfg)
        _archive_live_records(fetched, cfg)
        records.extend(fetched)
        errors.extend(fetch_errors)

    if allow_network and "espn" in sources and _is_sports_packet(packet):
        fetched, fetch_errors = _fetch_espn(query, cfg)
        _archive_live_records(fetched, cfg)
        records.extend(fetched)
        errors.extend(fetch_errors)

    if allow_network and "gdelt" in sources:
        fetched, fetch_errors = _fetch_gdelt(query, as_of_dt, cfg)
        _archive_live_records(fetched, cfg)
        records.extend(fetched)
        errors.extend(fetch_errors)

    for vendor_source in ("wrds", "lseg"):
        if allow_network and vendor_source in sources and _is_live_as_of(as_of_dt, cfg):
            fetched, fetch_errors = fetch_vendor_records(vendor_source, query, as_of_dt, cfg)
            _archive_live_records(fetched, cfg)
            records.extend(fetched)
            errors.extend(fetch_errors)

    if allow_network and "x" in sources:
        fetched, fetch_errors = _fetch_x(query, as_of_dt, cfg)
        _archive_live_records(fetched, cfg)
        records.extend(fetched)
        errors.extend(fetch_errors)

    pit_records = [
        record for record in records
        if _is_relevant_record(record, packet, query)
        and _is_pit_record(record, as_of_dt, strict=strict, cfg=cfg)
    ]
    ranked = _rank_records(pit_records, packet, query)[: cfg.arena.pit_external_max_records]

    evidence: list[dict[str, Any]] = []
    if ranked:
        evidence.append(_summarize_records(ranked, packet, query, as_of_dt, strict))
    if errors:
        evidence.append({
            "source": "pit_external_fetch_error",
            "timestamp": _now(),
            "pit_cutoff": as_of_dt.isoformat(),
            "claim": "One or more enabled external evidence fetches failed.",
            "errors": errors[:4],
        })
    return annotate_evidence_items(evidence, getattr(packet, "category", None))


def fetch_external_records_for_packet(
    packet: Any,
    cfg: ForecastConfig,
    *,
    sources: set[str] | None = None,
    allow_historical_backfill: bool = False,
    allow_reddit_historical_backfill: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch and annotate raw external records for archival.

    Historical backfills are useful for prompt research but are not equivalent
    to a live collection log. Rows carry `pit_mode` and `strict_pit_eligible`
    so evaluators can decide whether to use them.
    """
    as_of_dt = _packet_as_of(packet)
    if as_of_dt is None:
        return [], [{"source": "pit_external_archiver", "error": "unparseable packet.as_of"}]

    query = build_evidence_query(packet)
    selected_sources = sources or {"reddit", "gdelt", "espn"}
    live = _is_live_as_of(as_of_dt, cfg)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if "reddit" in selected_sources:
        if live:
            fetched, fetch_errors = _fetch_reddit(query, cfg)
            records.extend(_annotate_records(fetched, packet, cfg, query, mode="live_capture"))
            errors.extend(fetch_errors)
        elif allow_reddit_historical_backfill:
            fetched, fetch_errors = _fetch_reddit(query, cfg, time_filter="all")
            records.extend(_annotate_records(
                fetched,
                packet,
                cfg,
                query,
                mode="reddit_published_at_only_backfill",
            ))
            errors.extend(fetch_errors)
        else:
            errors.append({
                "source": "reddit",
                "error": "skipped_historical_reddit_without_prior_archive",
                "detail": "Reddit public search cannot prove old search-index state for strict PIT backtests.",
            })

    if "espn" in selected_sources:
        if live and _is_sports_packet(packet):
            fetched, fetch_errors = _fetch_espn(query, cfg)
            records.extend(_annotate_records(fetched, packet, cfg, query, mode="live_capture"))
            errors.extend(fetch_errors)
        elif not _is_sports_packet(packet):
            errors.append({
                "source": "espn",
                "error": "skipped_non_sports_packet",
            })
        else:
            errors.append({
                "source": "espn",
                "error": "skipped_historical_espn_without_prior_archive",
                "detail": "ESPN live news capture is PIT-clean only when archived before forecast as_of.",
            })

    if "gdelt" in selected_sources:
        fetched, fetch_errors = _fetch_gdelt(
            query,
            as_of_dt,
            cfg,
            allow_historical_archive=allow_historical_backfill,
        )
        mode = "live_capture" if live else "gdelt_doc_backfill"
        records.extend(_annotate_records(fetched, packet, cfg, query, mode=mode))
        errors.extend(fetch_errors)

    if "x" in selected_sources:
        fetched, fetch_errors = _fetch_x(
            query,
            as_of_dt,
            cfg,
            allow_historical_archive=allow_historical_backfill,
        )
        mode = "live_capture" if live else "x_full_archive_backfill"
        records.extend(_annotate_records(fetched, packet, cfg, query, mode=mode))
        errors.extend(fetch_errors)

    for vendor_source in ("wrds", "lseg"):
        if vendor_source not in selected_sources:
            continue
        if live or allow_historical_backfill:
            fetched, fetch_errors = fetch_vendor_records(vendor_source, query, as_of_dt, cfg)
            mode = "live_capture" if live else f"{vendor_source}_published_at_backfill"
            records.extend(_annotate_records(fetched, packet, cfg, query, mode=mode))
            errors.extend(fetch_errors)
        else:
            errors.append({
                "source": vendor_source,
                "error": f"skipped_historical_{vendor_source}_without_prior_archive",
                "detail": (
                    f"{vendor_source.upper()} live connectors can be used for exploratory published_at backfills "
                    "only with --allow-historical-backfill. Strict PIT backtests should replay prior local archives."
                ),
            })

    return records, errors


def annotate_external_records(
    records: list[dict[str, Any]],
    packet: Any,
    cfg: ForecastConfig,
    query: str,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    """Public wrapper for adding PIT metadata/source profiles/sentiment."""
    return _annotate_records(records, packet, cfg, query, mode=mode)


def build_evidence_query(packet: Any) -> str:
    """Create a compact source-search query from event text and outcomes."""
    title = str(getattr(packet, "title", "") or "")
    subtitle = str(getattr(packet, "subtitle", "") or "")
    outcomes = [str(outcome) for outcome in (getattr(packet, "outcomes", None) or [])]
    multileg = _packet_multileg_contract(packet)
    if multileg.get("is_multileg"):
        terms = [str(term) for term in multileg.get("search_terms") or [] if str(term).strip()]
        if terms:
            quoted = [f'"{term}"' for term in terms if len(term) <= 64]
            broad_terms = [
                tok
                for tok in TOKEN_RE.findall(" ".join(terms).lower())
                if len(tok) > 2 and tok not in {"yes", "no", "over", "under", "points", "scored"}
            ]
            query = " OR ".join(quoted[:6])
            tail = " ".join(list(dict.fromkeys(broad_terms))[:10])
            return f"({query}) {tail}".strip()[:480]
    text = " ".join([title, subtitle, *outcomes])
    stop = {
        "will", "what", "when", "where", "which", "who", "win", "wins", "yes", "no", "the",
        "this", "that", "with", "from", "over", "under", "above", "below", "market", "predict",
    }
    tokens = [tok for tok in TOKEN_RE.findall(text.lower()) if len(tok) > 2 and tok not in stop]
    deduped = list(dict.fromkeys(tokens))
    is_yes_no = [outcome.upper() for outcome in outcomes] == ["YES", "NO"]
    if outcomes and len(outcomes) <= 6 and not is_yes_no:
        quoted = [f'"{outcome}"' for outcome in outcomes if len(outcome) <= 64]
        query = " OR ".join(quoted[:4])
        tail = " ".join(deduped[:8])
        return f"({query}) {tail}".strip()[:480] if query else tail[:480]
    return " ".join(deduped[:14])[:480]


def _packet_multileg_contract(packet: Any) -> dict[str, Any]:
    entities = getattr(packet, "extracted_entities", None)
    if isinstance(entities, dict):
        contract = entities.get("kalshi_multileg_contract")
        if isinstance(contract, dict):
            return contract
    features = getattr(packet, "features", None)
    if isinstance(features, dict):
        contract = features.get("kalshi_multileg_contract")
        if isinstance(contract, dict):
            return contract
    return {}


def _load_local_records(packet: Any, cfg: ForecastConfig, query: str) -> list[dict[str, Any]]:
    root = _resolve_evidence_root(cfg)
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row.setdefault("source", _infer_source_from_path(path))
                row.setdefault("record_path", str(path))
                records.append(row)
    return records


def _resolve_evidence_root(cfg: ForecastConfig) -> Path:
    root = Path(cfg.arena.pit_external_root)
    if root.is_absolute():
        return root
    candidates = [
        Path.cwd() / root,
        PACKAGE_ROOT / root,
        PACKAGE_ROOT.parent / root,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PACKAGE_ROOT.parent / root


def _archive_live_records(records: list[dict[str, Any]], cfg: ForecastConfig) -> None:
    if not records or not cfg.arena.pit_external_archive_live_fetches:
        return
    root = _resolve_evidence_root(cfg)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for record in records:
        source = re.sub(r"[^a-z0-9_-]+", "_", str(record.get("source") or "external").lower())
        archive_dir = root / "live_fetches" / source
        archive_dir.mkdir(parents=True, exist_ok=True)
        path = archive_dir / f"{date}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _fetch_reddit(
    query: str,
    cfg: ForecastConfig,
    *,
    time_filter: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    headers = {
        "User-Agent": os.environ.get("REDDIT_USER_AGENT", "ProphetHacksGPTForecasting/0.1"),
    }
    params = {
        "q": query,
        "sort": "new",
        "t": time_filter or ("day" if cfg.arena.pit_external_live_lookback_hours <= 24 else "week"),
        "limit": min(25, max(1, cfg.arena.pit_external_max_records)),
        "raw_json": 1,
    }
    try:
        response = requests.get(REDDIT_SEARCH_URL, params=params, headers=headers, timeout=8)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - evidence fetches must not block forecasts.
        return [], [{"source": "reddit", "error": f"{type(exc).__name__}:{exc}"}]
    children = ((data.get("data") or {}).get("children") or []) if isinstance(data, dict) else []
    records = []
    for child in children:
        item = child.get("data") if isinstance(child, dict) else None
        if not isinstance(item, dict):
            continue
        created = item.get("created_utc")
        published_at = datetime.fromtimestamp(float(created), tz=timezone.utc).isoformat() if created else None
        records.append({
            "source": "reddit",
            "published_at": published_at,
            "collected_at": _now(),
            "title": item.get("title"),
            "text": item.get("selftext"),
            "url": f"https://www.reddit.com{item.get('permalink')}" if item.get("permalink") else item.get("url"),
            "subreddit": item.get("subreddit"),
            "score": item.get("score"),
            "num_comments": item.get("num_comments"),
        })
    return records, []


def _fetch_gdelt(
    query: str,
    as_of_dt: datetime,
    cfg: ForecastConfig,
    *,
    allow_historical_archive: bool | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    live = _is_live_as_of(as_of_dt, cfg)
    historical = (
        _env_bool("PIT_EXTERNAL_ALLOW_HISTORICAL_ARCHIVE_PULLS", False)
        if allow_historical_archive is None
        else allow_historical_archive
    )
    if not live and not historical:
        return [], []
    start_dt = as_of_dt - timedelta(hours=cfg.arena.pit_external_live_lookback_hours)
    params = {
        "query": query[:512],
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(75, max(1, cfg.arena.pit_external_max_records)),
        "sort": "datedesc",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": as_of_dt.strftime("%Y%m%d%H%M%S"),
    }
    try:
        for attempt in range(3):
            response = requests.get(GDELT_DOC_URL, params=params, timeout=10)
            if response.status_code != 429:
                break
            time.sleep(2.0 + attempt * 3.0)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        return [], [{"source": "gdelt", "error": f"{type(exc).__name__}:{exc}"}]

    items = data.get("articles") or []
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue
        published = _gdelt_seen_date(item.get("seendate"))
        records.append({
            "source": "gdelt",
            "published_at": published,
            "collected_at": _now(),
            "title": item.get("title"),
            "text": item.get("title"),
            "url": item.get("url"),
            "domain": item.get("domain"),
            "language": item.get("language"),
            "source_country": item.get("sourcecountry"),
            "seendate": item.get("seendate"),
            "socialimage": item.get("socialimage"),
        })
    return records, []


def _fetch_espn(
    query: str,
    cfg: ForecastConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for sport, league in ESPN_LEAGUES:
        url = ESPN_NEWS_URL.format(sport=sport, league=league)
        params = {"limit": min(20, max(1, cfg.arena.pit_external_max_records))}
        try:
            response = requests.get(url, params=params, timeout=8)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            errors.append({"source": "espn", "league": league, "error": f"{type(exc).__name__}:{exc}"})
            continue
        for item in data.get("articles") or []:
            if not isinstance(item, dict):
                continue
            title = item.get("headline") or item.get("title")
            description = item.get("description") or item.get("story")
            published_at = item.get("published") or item.get("lastModified")
            url = item.get("links", {}).get("web", {}).get("href") if isinstance(item.get("links"), dict) else None
            record = {
                "source": "espn",
                "published_at": published_at,
                "collected_at": _now(),
                "title": title,
                "text": description,
                "url": url,
                "sport": sport,
                "league": league,
            }
            if _is_relevant_record(record, _QueryOnlyPacket(query), query):
                records.append(record)
    ranked = _rank_records(records, _QueryOnlyPacket(query), query)
    return ranked[: cfg.arena.pit_external_max_records], errors[:4]


def _fetch_x(
    query: str,
    as_of_dt: datetime,
    cfg: ForecastConfig,
    *,
    allow_historical_archive: bool | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    token = os.environ.get("X_BEARER_TOKEN") or os.environ.get("TWITTER_BEARER_TOKEN")
    if not token:
        return [], []

    historical_archive = (
        _env_bool("PIT_EXTERNAL_ALLOW_HISTORICAL_ARCHIVE_PULLS", False)
        if allow_historical_archive is None
        else allow_historical_archive
    )
    live = _is_live_as_of(as_of_dt, cfg)
    if not live and not historical_archive:
        return [], []

    endpoint = X_RECENT_SEARCH_URL if live else X_FULL_ARCHIVE_SEARCH_URL
    start_dt = as_of_dt - timedelta(hours=cfg.arena.pit_external_live_lookback_hours)
    params = {
        "query": f"({query}) lang:en -is:retweet"[:512],
        "max_results": min(100, max(10, cfg.arena.pit_external_max_records)),
        "tweet.fields": "created_at,public_metrics,lang",
        "sort_order": "recency",
    }
    if not live:
        params["start_time"] = start_dt.isoformat().replace("+00:00", "Z")
        params["end_time"] = as_of_dt.isoformat().replace("+00:00", "Z")
    try:
        response = requests.get(
            endpoint,
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        return [], [{"source": "x", "error": f"{type(exc).__name__}:{exc}"}]

    records = []
    for item in data.get("data") or []:
        if not isinstance(item, dict):
            continue
        metrics = item.get("public_metrics") or {}
        records.append({
            "source": "x",
            "published_at": item.get("created_at"),
            "collected_at": _now(),
            "title": None,
            "text": item.get("text"),
            "url": f"https://x.com/i/web/status/{item.get('id')}" if item.get("id") else None,
            "retweet_count": metrics.get("retweet_count"),
            "reply_count": metrics.get("reply_count"),
            "like_count": metrics.get("like_count"),
            "quote_count": metrics.get("quote_count"),
        })
    return records, []


def _annotate_records(
    records: list[dict[str, Any]],
    packet: Any,
    cfg: ForecastConfig,
    query: str,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    as_of_dt = _packet_as_of(packet)
    out = []
    for record in records:
        enriched = annotate_record_sentiment(record)
        enriched.update({
            "target_event_ticker": getattr(packet, "event_ticker", None),
            "target_market_ticker": getattr(packet, "market_ticker", None),
            "forecast_as_of": getattr(packet, "as_of", None),
            "forecast_close_time": getattr(packet, "close_time", None),
            "forecast_title": getattr(packet, "title", None),
            "forecast_query": query,
            "pit_mode": mode,
        })
        profile = compact_source_profile(enriched.get("source"))
        enriched.setdefault("source_family", profile["family"])
        enriched.setdefault("source_reliability", profile["reliability"])
        enriched.setdefault("source_pit_mode", profile["pit_mode"])
        enriched.setdefault("source_caution", profile["caution"])
        if "event_ticker" not in enriched and getattr(packet, "event_ticker", None):
            enriched["event_ticker"] = getattr(packet, "event_ticker")
        if "market_ticker" not in enriched and getattr(packet, "market_ticker", None):
            enriched["market_ticker"] = getattr(packet, "market_ticker")
        enriched["strict_pit_eligible"] = (
            _is_pit_record(enriched, as_of_dt, strict=True, cfg=cfg) if as_of_dt else False
        )
        enriched["published_at_pit_eligible"] = (
            _is_pit_record(enriched, as_of_dt, strict=False, cfg=cfg) if as_of_dt else False
        )
        out.append(enriched)
    return out


def _summarize_records(
    records: list[dict[str, Any]],
    packet: Any,
    query: str,
    as_of_dt: datetime,
    strict: bool,
) -> dict[str, Any]:
    records = [annotate_record_sentiment(record) for record in records]
    source_counts = Counter(str(record.get("source") or "unknown") for record in records)
    outcome_mentions = {
        str(outcome): sum(_mentions_outcome(record, str(outcome)) for record in records)
        for outcome in getattr(packet, "outcomes", []) or []
    }
    sentiment = aggregate_sentiment(records)
    return {
        "source": "pit_external_evidence",
        "timestamp": _now(),
        "pit_cutoff": as_of_dt.isoformat(),
        "query": query,
        "strict_collected_at": strict,
        "claim": "Timestamp-filtered external evidence was available before the forecast cutoff.",
        "record_count": len(records),
        "source_counts": dict(source_counts),
        "source_profiles": {
            source: compact_source_profile(source)
            for source in sorted(source_counts)
        },
        "outcome_mentions": outcome_mentions,
        "sentiment": sentiment,
        "records": [_compact_record(record) for record in records],
    }


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    text = str(record.get("text") or "")
    title = str(record.get("title") or "")
    compact = {
        "source": record.get("source"),
        "published_at": record.get("published_at") or record.get("created_at"),
        "collected_at": record.get("collected_at"),
        "title": title[:220] or None,
        "text_excerpt": text[:500] or None,
        "url": record.get("url"),
    }
    for key in ("score", "num_comments", "like_count", "retweet_count", "reply_count", "quote_count", "subreddit"):
        if record.get(key) is not None:
            compact[key] = record.get(key)
    for key in ("sentiment_score", "sentiment_label", "sentiment_model"):
        if record.get(key) is not None:
            compact[key] = record.get(key)
    for key in ("domain", "sport", "league", "source_family", "source_reliability"):
        if record.get(key) is not None:
            compact[key] = record.get(key)
    compact["source_profile"] = compact_source_profile(record.get("source"))
    return compact


def _rank_records(records: list[dict[str, Any]], packet: Any, query: str) -> list[dict[str, Any]]:
    q_tokens = _tokens(query)
    outcome_tokens = set()
    for outcome in getattr(packet, "outcomes", []) or []:
        outcome_tokens.update(_tokens(str(outcome)))

    def score(record: dict[str, Any]) -> float:
        text_tokens = _tokens(_record_text(record))
        overlap = len(text_tokens & q_tokens)
        outcome_overlap = len(text_tokens & outcome_tokens)
        engagement = 0.0
        for key in ("score", "num_comments", "like_count", "retweet_count", "reply_count", "quote_count"):
            value = record.get(key)
            if _is_number(value):
                engagement += math.log1p(max(0.0, float(value)))
        ticker_bonus = 0.0
        if record.get("market_ticker") == getattr(packet, "market_ticker", None):
            ticker_bonus += 20.0
        if record.get("event_ticker") == getattr(packet, "event_ticker", None):
            ticker_bonus += 10.0
        return ticker_bonus + overlap + 2.0 * outcome_overlap + 0.05 * engagement

    return sorted(records, key=score, reverse=True)


def _is_relevant_record(record: dict[str, Any], packet: Any, query: str) -> bool:
    if record.get("market_ticker") and record.get("market_ticker") == getattr(packet, "market_ticker", None):
        return True
    if record.get("event_ticker") and record.get("event_ticker") == getattr(packet, "event_ticker", None):
        return True
    text_tokens = _tokens(_record_text(record))
    if not text_tokens:
        return False
    q_tokens = _tokens(query)
    return len(text_tokens & q_tokens) >= 2


class _QueryOnlyPacket:
    def __init__(self, query: str) -> None:
        self.market_ticker = None
        self.event_ticker = None
        self.outcomes = []
        self.category = "Sports"
        self.title = query


def _is_pit_record(
    record: dict[str, Any],
    as_of_dt: datetime,
    *,
    strict: bool,
    cfg: ForecastConfig,
) -> bool:
    if (
        record.get("pit_mode") == "reddit_published_at_only_backfill"
        and not _env_bool("PIT_EXTERNAL_ALLOW_REDDIT_BACKFILL_REPLAY", False)
    ):
        return False
    published = parse_dt(record.get("published_at") or record.get("created_at") or record.get("timestamp"))
    if published is None or published > as_of_dt:
        return False
    collected = parse_dt(record.get("collected_at") or record.get("retrieved_at") or record.get("ingested_at"))
    if strict:
        if collected is None:
            return False
        tolerance = timedelta(seconds=cfg.arena.pit_external_clock_tolerance_seconds)
        if collected > as_of_dt + tolerance:
            return False
    return True


def _mentions_outcome(record: dict[str, Any], outcome: str) -> int:
    if not outcome:
        return 0
    haystack = _record_text(record).lower()
    return 1 if outcome.lower() in haystack else 0


def _record_text(record: dict[str, Any]) -> str:
    return " ".join(str(record.get(key) or "") for key in ("title", "text", "body", "summary", "claim"))


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "this", "that", "will", "yes", "no"}
    return {tok for tok in TOKEN_RE.findall(text.lower()) if len(tok) > 2 and tok not in stop}


def _packet_as_of(packet: Any) -> datetime | None:
    value = getattr(packet, "as_of", None)
    return parse_dt(str(value)) if value else None


def _is_live_as_of(as_of_dt: datetime, cfg: ForecastConfig) -> bool:
    now = datetime.now(timezone.utc)
    max_age = timedelta(minutes=cfg.arena.pit_external_max_live_age_minutes)
    return abs(now - as_of_dt) <= max_age


def _infer_source_from_path(path: Path) -> str:
    lower = path.name.lower()
    if "reddit" in lower:
        return "reddit"
    if "gdelt" in lower:
        return "gdelt"
    if "espn" in lower:
        return "espn"
    if "wrds" in lower:
        return "wrds"
    if "lseg" in lower:
        return "lseg"
    if "twitter" in lower or lower.startswith("x_") or "_x_" in lower:
        return "x"
    return "external_jsonl"


def _is_sports_packet(packet: Any) -> bool:
    category = str(getattr(packet, "category", "") or "").lower()
    text = " " + " ".join(str(getattr(packet, key, "") or "") for key in ("title", "subtitle", "rules")).lower() + " "
    return (
        "sport" in category
        or any(token in text for token in (" nba ", " nfl ", " mlb ", " nhl ", " ncaa ", " soccer ", " match ", " game "))
    )


def _gdelt_seen_date(value: Any) -> str | None:
    text = str(value or "")
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    parsed = parse_dt(text)
    return parsed.isoformat() if parsed else None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
