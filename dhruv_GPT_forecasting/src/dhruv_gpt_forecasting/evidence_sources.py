"""Evidence source taxonomy for PIT forecasting prompts.

The LLM should not treat every record as equally reliable. This module keeps
the source metadata local and deterministic so archived records, live fetches,
and prompt payloads use the same source categories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class EvidenceSourceProfile:
    source: str
    family: str
    reliability: str
    pit_mode: str
    best_for: str
    caution: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


SOURCE_PROFILES: dict[str, EvidenceSourceProfile] = {
    "kalshi_public_market": EvidenceSourceProfile(
        source="kalshi_public_market",
        family="prediction_market_quote",
        reliability="high_for_market_belief_medium_for_truth",
        pit_mode="live_or_archived_quote_timestamp",
        best_for="Current crowd-implied probability, liquidity, and price movement.",
        caution="Market prices can be stale, low-liquidity, or structurally biased.",
    ),
    "polymarket_public_search": EvidenceSourceProfile(
        source="polymarket_public_search",
        family="prediction_market_context",
        reliability="medium",
        pit_mode="live_search_timestamp",
        best_for="Finding related market questions and cross-market context.",
        caution="Search matches are not guaranteed to resolve under identical rules.",
    ),
    "kalshi_nonbinary_context": EvidenceSourceProfile(
        source="kalshi_nonbinary_context",
        family="related_market_structure",
        reliability="medium_high",
        pit_mode="archived_market_context",
        best_for="Inferring structured event probabilities from sibling or component markets.",
        caution="Component markets can be inconsistent and may use different liquidity regimes.",
    ),
    "kalshi_topvol_same_event": EvidenceSourceProfile(
        source="kalshi_topvol_same_event",
        family="related_market_structure",
        reliability="medium",
        pit_mode="archived_market_context",
        best_for="Sibling-market pressure and same-event probability shape.",
        caution="Use as supporting context, not direct resolution evidence.",
    ),
    "kalshi_polymarket_map": EvidenceSourceProfile(
        source="kalshi_polymarket_map",
        family="related_market_structure",
        reliability="medium",
        pit_mode="archived_mapping",
        best_for="Mapping comparable Kalshi and Polymarket questions.",
        caution="Mapping quality varies and missing mappings are not negative evidence.",
    ),
    "linked_market_model": EvidenceSourceProfile(
        source="linked_market_model",
        family="secondary_probability_model",
        reliability="medium_high_when_group_is_coherent",
        pit_mode="derived_from_pre_as_of_linked_market_quotes",
        best_for="Inferring event-level or sibling-market distributions from linked prediction markets.",
        caution="Normalize only coherent same-event groups; discount if component sum, liquidity, or mapping quality is poor.",
    ),
    "reddit": EvidenceSourceProfile(
        source="reddit",
        family="social_discussion",
        reliability="low_medium",
        pit_mode="strict_only_when_collected_before_as_of",
        best_for="Crowd narratives, injuries noticed by fans, local reporting pointers.",
        caution="Noisy, biased, and not strict historical PIT unless captured live before cutoff.",
    ),
    "gdelt": EvidenceSourceProfile(
        source="gdelt",
        family="general_news_search",
        reliability="medium",
        pit_mode="timestamp_bounded_by_article_seen_date",
        best_for="Broad timestamp-bounded news discovery across domains.",
        caution="Article matching is lexical; summaries can be thin and rate limits are common.",
    ),
    "espn": EvidenceSourceProfile(
        source="espn",
        family="sports_news",
        reliability="high_for_sports_availability_and_match_context",
        pit_mode="live_capture_or_archived_record",
        best_for="Sports injuries, lineup news, schedules, match previews, and game status.",
        caution="Not a betting odds source; use with odds/market priors when available.",
    ),
    "fred": EvidenceSourceProfile(
        source="fred",
        family="official_economic_series",
        reliability="high",
        pit_mode="release_timestamp_or_live_capture",
        best_for="Macro time series, rates, inflation, labor, GDP, and commodities series.",
        caution="Some observations are revised; make sure vintage/release timing is PIT-safe.",
    ),
    "bea": EvidenceSourceProfile(
        source="bea",
        family="official_economic_series",
        reliability="high",
        pit_mode="release_timestamp_or_live_capture",
        best_for="GDP, income, consumption, and national accounts context.",
        caution="Use release vintage when backtesting historical forecasts.",
    ),
    "eia": EvidenceSourceProfile(
        source="eia",
        family="official_energy_series",
        reliability="high",
        pit_mode="release_timestamp_or_live_capture",
        best_for="Oil, gas, electricity, and energy inventory forecasts.",
        caution="Release calendars and revisions matter for PIT tests.",
    ),
    "polygon": EvidenceSourceProfile(
        source="polygon",
        family="market_data",
        reliability="high_for_traded_assets",
        pit_mode="market_timestamp_or_live_capture",
        best_for="Crypto, equities, indices, commodities proxies, and recent price moves.",
        caution="Asset price movement is a signal, not a direct event-resolution label.",
    ),
    "oddspipe": EvidenceSourceProfile(
        source="oddspipe",
        family="sports_odds",
        reliability="high_for_sports_market_belief",
        pit_mode="quote_timestamp_or_live_capture",
        best_for="Sports moneylines, spreads, totals, and bookmaker-implied priors.",
        caution="Book odds include vig and can disagree with event rules.",
    ),
    "wrds": EvidenceSourceProfile(
        source="wrds",
        family="licensed_financial_and_economic_data",
        reliability="high",
        pit_mode="vendor_timestamp_or_archived_record",
        best_for="Structured financial, macro, accounting, and market microstructure datasets.",
        caution="Use normalized local exports or a licensed connector; enforce vintage timestamps.",
    ),
    "lseg": EvidenceSourceProfile(
        source="lseg",
        family="licensed_news_and_market_data",
        reliability="high",
        pit_mode="vendor_timestamp_or_archived_record",
        best_for="Professional news, market data, macro releases, and corporate-event context.",
        caution="Requires licensed access; archive source timestamps for backtests.",
    ),
    "pit_news_digest": EvidenceSourceProfile(
        source="pit_news_digest",
        family="local_extract_digest",
        reliability="depends_on_underlying_sources",
        pit_mode="derived_from_pit_filtered_records",
        best_for="Token-efficient summary of timestamp-filtered news records.",
        caution="Digest is extractive and should not override the underlying source timestamps.",
    ),
    "prophet_subset_source": EvidenceSourceProfile(
        source="prophet_subset_source",
        family="curated_benchmark_source",
        reliability="medium_high",
        pit_mode="benchmark_snapshot_time",
        best_for="Curated source snippets bundled with the Prophet subset_1200 row.",
        caution="If no article publication timestamp is present, collected_at is the row snapshot time, not the original publication time.",
    ),
    "live_source_plan": EvidenceSourceProfile(
        source="live_source_plan",
        family="retrieval_metadata",
        reliability="not_evidence",
        pit_mode="current_request_metadata",
        best_for="Auditing which live data sources were attempted for this forecast category.",
        caution="This is not outcome evidence; use only to identify possible missing data.",
    ),
}


ALIASES = {
    "kalshi": "kalshi_public_market",
    "kalshi_market": "kalshi_public_market",
    "polymarket": "polymarket_public_search",
    "espn_news": "espn",
    "fred_series": "fred",
    "bea_series": "bea",
    "eia_series": "eia",
    "polygon_price": "polygon",
    "wrds_news": "wrds",
    "wrds_vendor": "wrds",
    "lseg_news": "lseg",
    "lseg_vendor": "lseg",
    "prophet_sources": "prophet_subset_source",
    "prophet_subset": "prophet_subset_source",
}


RECOMMENDED_BY_CATEGORY = {
    "Sports": ["linked_market_model", "kalshi_public_market", "polymarket_public_search", "oddspipe", "espn", "reddit", "gdelt"],
    "Economics": ["linked_market_model", "kalshi_public_market", "polymarket_public_search", "fred", "bea", "eia", "wrds", "lseg", "gdelt"],
    "Politics": ["linked_market_model", "kalshi_public_market", "polymarket_public_search", "gdelt", "lseg", "reddit"],
    "Elections": ["linked_market_model", "kalshi_public_market", "polymarket_public_search", "gdelt", "lseg", "reddit"],
    "Weather": ["linked_market_model", "kalshi_public_market", "polymarket_public_search", "gdelt"],
    "Climate and Weather": ["linked_market_model", "kalshi_public_market", "polymarket_public_search", "gdelt", "eia"],
    "Crypto": ["linked_market_model", "kalshi_public_market", "polymarket_public_search", "polygon", "lseg", "gdelt", "reddit"],
    "Commodities": ["linked_market_model", "kalshi_public_market", "polymarket_public_search", "polygon", "eia", "fred", "lseg", "gdelt"],
}


def canonical_source_name(source: Any) -> str:
    raw = str(source or "unknown").strip().lower()
    if raw.startswith("fred_"):
        return "fred"
    if raw.startswith("polygon_"):
        return "polygon"
    if raw.startswith("espn_"):
        return "espn"
    if raw.startswith("wrds_"):
        return "wrds"
    if raw.startswith("lseg_"):
        return "lseg"
    return ALIASES.get(raw, raw)


def source_profile(source: Any) -> EvidenceSourceProfile:
    canonical = canonical_source_name(source)
    return SOURCE_PROFILES.get(
        canonical,
        EvidenceSourceProfile(
            source=canonical,
            family="uncategorized_external_evidence",
            reliability="unknown",
            pit_mode="record_timestamp_required",
            best_for="Use only if the record is specific, timestamped, and relevant.",
            caution="Unknown source quality; discount unless corroborated.",
        ),
    )


def compact_source_profile(source: Any) -> dict[str, str]:
    return source_profile(source).to_dict()


def annotate_evidence_items(items: list[dict[str, Any]], category: str | None = None) -> list[dict[str, Any]]:
    annotated = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        profile = compact_source_profile(row.get("source"))
        row.setdefault("source_family", profile["family"])
        row.setdefault("source_reliability", profile["reliability"])
        row.setdefault("source_pit_mode", profile["pit_mode"])
        row.setdefault("source_caution", profile["caution"])
        row.setdefault("retrieval_confidence", retrieval_confidence(row, category))
        annotated.append(row)
    return annotated


def retrieval_confidence(item: dict[str, Any], category: str | None = None) -> dict[str, Any]:
    """Score evidence quality for prompt conditioning and calibration shrink."""
    profile = source_profile(item.get("source"))
    source_quality = _source_quality(profile.reliability)
    timestamp_quality = _timestamp_quality(item)
    pit_confidence = _pit_confidence(profile.pit_mode, item)
    match_confidence = _match_confidence(item, category)
    contradiction_flag = bool(item.get("contradiction") or item.get("error"))
    overall = 0.35 * source_quality + 0.25 * timestamp_quality + 0.25 * pit_confidence + 0.15 * match_confidence
    if contradiction_flag:
        overall *= 0.60
    return {
        "overall": round(max(0.0, min(1.0, overall)), 3),
        "source_quality": round(source_quality, 3),
        "timestamp_freshness": round(timestamp_quality, 3),
        "event_match": round(match_confidence, 3),
        "pit_confidence": round(pit_confidence, 3),
        "contradiction_or_error": contradiction_flag,
    }


def _source_quality(reliability: str) -> float:
    if "high" in reliability and "medium" not in reliability:
        return 0.92
    if "medium_high" in reliability or "high_for" in reliability:
        return 0.82
    if "medium" in reliability:
        return 0.62
    if "low" in reliability:
        return 0.38
    if reliability == "not_evidence":
        return 0.10
    return 0.45


def _timestamp_quality(item: dict[str, Any]) -> float:
    raw = item.get("published_at") or item.get("timestamp") or item.get("as_of") or item.get("collected_at")
    if not raw:
        return 0.35
    parsed = _parse_dt(str(raw))
    if parsed is None:
        return 0.45
    age_hours = max(0.0, (datetime.now(UTC) - parsed).total_seconds() / 3600.0)
    if age_hours <= 6:
        return 0.95
    if age_hours <= 24:
        return 0.80
    if age_hours <= 24 * 7:
        return 0.62
    return 0.42


def _pit_confidence(pit_mode: str, item: dict[str, Any]) -> float:
    if item.get("error"):
        return 0.15
    if "strict" in pit_mode and not (item.get("collected_at") or item.get("timestamp")):
        return 0.45
    if "timestamp" in pit_mode or "archived" in pit_mode or "release" in pit_mode:
        return 0.78
    if pit_mode == "current_request_metadata":
        return 0.20
    return 0.58


def _match_confidence(item: dict[str, Any], category: str | None) -> float:
    if item.get("match_confidence") is not None:
        try:
            return max(0.0, min(1.0, float(item["match_confidence"])))
        except (TypeError, ValueError):
            pass
    if item.get("yes_probability") is not None or item.get("probabilities") is not None:
        return 0.82
    if item.get("records") or item.get("matches") or item.get("observations") or item.get("results"):
        return 0.68
    if item.get("category") == category:
        return 0.60
    return 0.50


def _parse_dt(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def evidence_source_policy(category: str | None, observed_sources: list[str] | None = None) -> dict[str, Any]:
    normalized_category = str(category or "Other")
    recommended = RECOMMENDED_BY_CATEGORY.get(normalized_category, [
        "kalshi_public_market",
        "polymarket_public_search",
        "gdelt",
        "reddit",
    ])
    observed = sorted({canonical_source_name(source) for source in (observed_sources or []) if source})
    active = sorted(set(recommended) | set(observed))
    return {
        "category": normalized_category,
        "gpt_model_role": {
            "primary_model": "gemini-3-flash-preview",
            "search_grounding": "native search grounding on live/current Arena forecasts",
            "role": "final_probability_model",
            "deterministic_models_role": "calibrated priors, diagnostics, and warning flags",
            "finalization": "GPT chooses final probabilities; code only validates labels, clamps, and normalizes.",
        },
        "recommended_sources": recommended,
        "observed_sources": observed,
        "profiles": {source: compact_source_profile(source) for source in active},
    }
