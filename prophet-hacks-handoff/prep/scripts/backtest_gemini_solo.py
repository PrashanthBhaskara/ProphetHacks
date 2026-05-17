"""Backtest Gemini models one at a time on event-level Prophet Arena data.

Requires GEMINI_API_KEY. The script caches per-event predictions in JSONL so
reruns do not spend API calls for already completed model/event pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.arena_data import (  # noqa: E402
    ArenaBacktestEvent,
    load_kalshi_topvol_horizon_events,
    load_subset_1200_events,
    parse_horizon,
)
from prep.arena_score import (  # noqa: E402
    event_brier_classical,
    should_normalize_actuals,
    summarize_event_scores,
)
from prep.forecasters import ForecasterConfig, forecast_from_config  # noqa: E402
from prep.packets import packet_from_arena_event  # noqa: E402


PREP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PREP_ROOT / "runs" / "gemini_solo"

MODEL_PRESETS = {
    "flash": {
        "name": "gemini_3_flash",
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "api_key_env": "GEMINI_API_KEY",
        "enabled": True,
        "weight": 1.0,
        "temperature": 0.0,
        "max_tokens": 1800,
        "system_prompt_path": "prompts/gemini_context_forecasting_system.txt",
    },
    "pro": {
        "name": "gemini_3_1_pro",
        "provider": "gemini",
        "model": "gemini-3.1-pro-preview",
        "api_key_env": "GEMINI_API_KEY",
        "enabled": True,
        "weight": 1.0,
        "temperature": 0.0,
        "max_tokens": 2200,
        "system_prompt_path": "prompts/gemini_context_forecasting_system.txt",
    },
}

MODEL_ALIASES = {
    "flash": ["gemini-3.1-flash-preview"],
    "pro": ["gemini-3-pro-preview", "gemini-3-pro"],
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _event_key(row: dict[str, Any]) -> str:
    submission_id = row.get("submission_id")
    if submission_id:
        return str(submission_id)
    return f"{row.get('event_ticker')}|{row.get('snapshot_time')}"


def _load_cache(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (str(row.get("model_name")), _event_key(row))
        rows[key] = row
    return rows


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _raw_probabilities(row_or_forecast: Any) -> dict[str, float]:
    if isinstance(row_or_forecast, dict):
        raw = row_or_forecast.get("raw_probabilities") or row_or_forecast.get("probabilities") or {}
    else:
        parsed = (row_or_forecast.raw_response or {}).get("parsed_response") or {}
        raw = ((parsed.get("forecast") or {}).get("probabilities") or row_or_forecast.probabilities or {})
    out = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _coerce_probability(value: Any) -> float | None:
    if value is None:
        return None
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(prob) or math.isinf(prob):
        return None
    if prob > 1.0:
        prob = prob / 100.0
    return max(0.0, min(1.0, prob))


def _market_mid_from_quote(market: dict[str, Any]) -> float | None:
    yes_ask = _coerce_probability(market.get("yes_ask"))
    no_ask = _coerce_probability(market.get("no_ask"))
    if yes_ask is not None and no_ask is not None:
        return max(0.0, min(1.0, (yes_ask + (1.0 - no_ask)) / 2.0))
    last_price = _coerce_probability(market.get("last_price"))
    if last_price is not None:
        return last_price
    return None


def _market_probabilities(event: ArenaBacktestEvent) -> dict[str, float] | None:
    retrieval = event.event.get("retrieval") or {}
    implied = retrieval.get("market_implied_probabilities") or {}
    probabilities: dict[str, float] = {}
    for outcome in event.outcomes:
        prob = _coerce_probability(implied.get(outcome))
        if prob is not None:
            probabilities[outcome] = prob

    if len(probabilities) == len(event.outcomes):
        return probabilities

    market_data = retrieval.get("market_data") or event.market_data or {}
    if event.outcomes == ["YES", "NO"]:
        yes_mid = _coerce_probability(market_data.get("yes_mid"))
        if yes_mid is not None:
            return {"YES": yes_mid, "NO": 1.0 - yes_mid}

    probabilities = {}
    for outcome in event.outcomes:
        market = market_data.get(outcome)
        if isinstance(market, dict):
            mid = _market_mid_from_quote(market)
            if mid is not None:
                probabilities[outcome] = mid
    if len(probabilities) == len(event.outcomes):
        return probabilities
    return None


def _market_baseline(event: ArenaBacktestEvent) -> dict[str, Any] | None:
    probabilities = _market_probabilities(event)
    if probabilities is None:
        return None
    return {
        "probabilities": probabilities,
        "score": _score_row(event, probabilities),
    }


def _blend_probabilities(
    model_probabilities: dict[str, float],
    market_probabilities: dict[str, float],
    outcomes: list[str],
    model_weight: float,
) -> dict[str, float] | None:
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model_weight must be between 0 and 1")
    blended: dict[str, float] = {}
    for outcome in outcomes:
        model_prob = _coerce_probability(model_probabilities.get(outcome))
        market_prob = _coerce_probability(market_probabilities.get(outcome))
        if model_prob is None or market_prob is None:
            return None
        blended[outcome] = max(0.0, min(1.0, market_prob + model_weight * (model_prob - market_prob)))
    return blended


def _calibrated_blend(
    event: ArenaBacktestEvent,
    model_probabilities: dict[str, float],
    baseline: dict[str, Any] | None,
    *,
    model_weight: float,
) -> dict[str, Any] | None:
    if baseline is None:
        return None
    probabilities = _blend_probabilities(
        model_probabilities,
        baseline["probabilities"],
        event.outcomes,
        model_weight,
    )
    if probabilities is None:
        return None
    return {
        "probabilities": probabilities,
        "model_weight": model_weight,
        "market_weight": 1.0 - model_weight,
        "score": _score_row(event, probabilities),
    }


def _event_lookup_key(event: ArenaBacktestEvent) -> str:
    if event.submission_id:
        return event.submission_id
    return f"{event.event.get('event_ticker')}|{event.snapshot_time}"


def _event_lookup_for_rows(rows: list[dict[str, Any]]) -> dict[str, ArenaBacktestEvent]:
    lookup: dict[str, ArenaBacktestEvent] = {}
    sources = {(row.get("event_traits") or {}).get("source") or "subset_1200" for row in rows}
    if "subset_1200" in sources:
        for event in load_subset_1200_events():
            lookup[_event_lookup_key(event)] = event

    topvol_horizons = set()
    for row in rows:
        traits = row.get("event_traits") or {}
        if traits.get("source") != "kalshi_topvol":
            continue
        horizon = traits.get("sampling_horizon")
        if not horizon and isinstance(row.get("submission_id"), str) and ":" in row["submission_id"]:
            horizon = row["submission_id"].rsplit(":", 1)[-1]
        if horizon:
            topvol_horizons.add(str(horizon))
    for horizon in topvol_horizons:
        for event in load_kalshi_topvol_horizon_events(horizon=horizon, max_rank=300):
            lookup[_event_lookup_key(event)] = event
    return lookup


def _add_market_baseline(
    row: dict[str, Any],
    event: ArenaBacktestEvent,
    *,
    blend_model_weight: float = 0.0,
) -> dict[str, Any]:
    baseline = _market_baseline(event)
    if baseline is None:
        return row
    updated = dict(row)
    updated["market_baseline_probabilities"] = baseline["probabilities"]
    updated["market_baseline_score"] = baseline["score"]
    if updated.get("score"):
        model_brier = updated["score"]["classical_brier"]
        market_brier = baseline["score"]["classical_brier"]
        updated["model_minus_market_brier"] = model_brier - market_brier
        updated["model_beats_market"] = model_brier < market_brier
        if blend_model_weight > 0:
            raw_probs = _raw_probabilities(updated)
            blend = _calibrated_blend(
                event,
                raw_probs,
                baseline,
                model_weight=blend_model_weight,
            )
            if blend is not None:
                updated["calibrated_blend_probabilities"] = blend["probabilities"]
                updated["calibrated_blend_score"] = blend["score"]
                updated["calibration"] = {
                    "type": "market_linear_blend",
                    "model_weight": blend["model_weight"],
                    "market_weight": blend["market_weight"],
                }
                updated["calibrated_minus_market_brier"] = (
                    blend["score"]["classical_brier"] - market_brier
                )
                updated["calibrated_minus_raw_brier"] = (
                    blend["score"]["classical_brier"] - model_brier
                )
    return updated


def _add_market_baselines_to_rows(
    rows: list[dict[str, Any]],
    *,
    blend_model_weight: float = 0.0,
) -> list[dict[str, Any]]:
    lookup = _event_lookup_for_rows(rows)
    enriched = []
    for row in rows:
        event = lookup.get(_event_key(row))
        enriched.append(
            _add_market_baseline(row, event, blend_model_weight=blend_model_weight)
            if event else row
        )
    return enriched


def _parse_event_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_horizon_seconds(event: ArenaBacktestEvent) -> float | None:
    close_time = _parse_event_dt(event.event.get("close_time"))
    snapshot_time = _parse_event_dt(
        event.snapshot_time
        or event.event.get("snapshot_time")
        or event.event.get("predict_by")
    )
    if close_time is None or snapshot_time is None:
        return None
    return (close_time - snapshot_time).total_seconds()


def _horizon_bucket(event: ArenaBacktestEvent) -> str:
    seconds = _event_horizon_seconds(event)
    if seconds is None:
        return "unknown"
    if seconds < 0:
        return "post_close"
    hour = 60 * 60
    day = 24 * hour
    if seconds < hour:
        return "<1h"
    if seconds < 6 * hour:
        return "1h-6h"
    if seconds < day:
        return "6h-1d"
    if seconds < 3 * day:
        return "1d-3d"
    if seconds < 7 * day:
        return "3d-7d"
    if seconds < 14 * day:
        return "7d-14d"
    return "14d+"


def _outcome_count_bucket(event: ArenaBacktestEvent) -> str:
    count = len(event.outcomes)
    if count == 2:
        return "binary"
    if count <= 5:
        return "3-5"
    if count <= 20:
        return "6-20"
    return "21+"


def _bucketed_sample(
    events: list[ArenaBacktestEvent],
    sample_size: int,
    rng: random.Random,
    key_fn,
) -> list[ArenaBacktestEvent]:
    buckets: dict[str, list[ArenaBacktestEvent]] = {}
    for event in events:
        buckets.setdefault(str(key_fn(event)), []).append(event)
    selected: list[ArenaBacktestEvent] = []
    keys = sorted(buckets)
    while len(selected) < sample_size and keys:
        next_keys = []
        for key in keys:
            bucket = buckets[key]
            if not bucket:
                continue
            selected.append(bucket.pop(rng.randrange(len(bucket))))
            if bucket:
                next_keys.append(key)
            if len(selected) >= sample_size:
                break
        keys = next_keys
    return selected


def _apply_target_horizon_filter(
    events: list[ArenaBacktestEvent],
    target_horizon: str | None,
    tolerance: str,
) -> list[ArenaBacktestEvent]:
    if not target_horizon:
        return events
    target_seconds = parse_horizon(target_horizon).total_seconds()
    tolerance_seconds = parse_horizon(tolerance).total_seconds()
    out = []
    for event in events:
        seconds = _event_horizon_seconds(event)
        if seconds is None:
            continue
        if abs(seconds - target_seconds) <= tolerance_seconds:
            out.append(event)
    return out


def _event_traits(event: ArenaBacktestEvent, args: argparse.Namespace) -> dict[str, Any]:
    retrieval = event.event.get("retrieval") or {}
    return {
        "source": args.source,
        "source_week": retrieval.get("source_week"),
        "source_cutoff": retrieval.get("source_cutoff") or event.snapshot_time,
        "sampling_horizon": retrieval.get("sampling_horizon"),
        "horizon_seconds": _event_horizon_seconds(event),
        "horizon_bucket": _horizon_bucket(event),
        "outcome_count": len(event.outcomes),
        "sample_mode": "binary_nonbinary" if args.stratified else args.sample_mode,
        "google_search_enabled": bool(args.enable_google_search),
        "market_blend_weight": float(getattr(args, "market_blend_weight", 0.0)),
    }


def _grounding_metadata(forecast: Any) -> dict[str, Any]:
    raw = forecast.raw_response or {}
    api_response = raw.get("api_response") or {}
    candidates = api_response.get("candidates") or []
    if not candidates:
        return {}
    return candidates[0].get("groundingMetadata") or {}


def _select_events(args: argparse.Namespace) -> list[ArenaBacktestEvent]:
    if args.source == "subset_1200":
        events = load_subset_1200_events(
            include_binary=not args.nonbinary_only,
            include_nonbinary=not args.binary_only,
            max_outcomes=args.max_outcomes,
        )
    else:
        events = load_kalshi_topvol_horizon_events(
            horizon=args.horizon,
            max_rank=args.max_rank,
        )
    if args.category:
        events = [event for event in events if event.event.get("category") == args.category]
    if args.binary_only:
        events = [event for event in events if event.is_binary]
    if args.nonbinary_only:
        events = [event for event in events if not event.is_binary]
    events = _apply_target_horizon_filter(events, args.target_horizon, args.horizon_tolerance)
    if args.sample:
        rng = random.Random(args.seed)
        sample_mode = "binary_nonbinary" if args.stratified else args.sample_mode
        if sample_mode == "binary_nonbinary" and args.sample < len(events):
            binary = [event for event in events if event.is_binary]
            nonbinary = [event for event in events if not event.is_binary]
            half = args.sample // 2
            selected = []
            selected.extend(rng.sample(binary, min(len(binary), half)))
            selected.extend(rng.sample(nonbinary, min(len(nonbinary), args.sample - len(selected))))
            if len(selected) < args.sample:
                remaining = [event for event in events if event not in selected]
                selected.extend(rng.sample(remaining, min(len(remaining), args.sample - len(selected))))
            events = selected
        elif sample_mode == "week" and args.sample < len(events):
            events = _bucketed_sample(
                events,
                args.sample,
                rng,
                lambda event: (event.event.get("retrieval") or {}).get("source_week") or str(event.snapshot_time or "")[:10],
            )
        elif sample_mode == "category" and args.sample < len(events):
            events = _bucketed_sample(events, args.sample, rng, lambda event: event.event.get("category") or "unknown")
        elif sample_mode == "horizon" and args.sample < len(events):
            events = _bucketed_sample(events, args.sample, rng, _horizon_bucket)
        elif sample_mode == "outcome_count" and args.sample < len(events):
            events = _bucketed_sample(events, args.sample, rng, _outcome_count_bucket)
        else:
            events = rng.sample(events, min(len(events), args.sample))
    elif args.limit:
        events = events[:args.limit]
    return events


def _score_row(event: ArenaBacktestEvent, probabilities: dict[str, float]) -> dict[str, Any]:
    normalize = should_normalize_actuals(event.actuals)
    score = event_brier_classical(
        probabilities,
        event.actuals,
        event.outcomes,
        normalize=normalize,
    )
    return {
        "classical_brier": score,
        "arena_brier_score": 1.0 - score,
        "normalize_prediction": normalize,
        "is_binary": event.is_binary,
        "is_exclusive": event.is_exclusive,
        "outcome_count": len(event.outcomes),
    }


def _is_model_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 404:
        return True
    text = str(exc).lower()
    return "not found" in text or "is not found" in text


def _forecast_with_fallback(
    config: ForecasterConfig,
    packet,
    aliases: list[str],
    *,
    allow_fallback: bool,
):
    models = [config.model, *aliases] if allow_fallback else [config.model]
    last_exc: Exception | None = None
    for model_id in models:
        attempt = replace(config, model=model_id)
        try:
            return forecast_from_config(attempt, packet), model_id
        except Exception as exc:  # noqa: BLE001
            if _is_model_not_found(exc) and model_id != models[-1]:
                last_exc = exc
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"No model aliases configured for {config.name}")


def _score_summary(
    rows: list[dict[str, Any]],
    bootstrap_resamples: int,
    seed: int,
    *,
    score_key: str = "score",
) -> dict[str, Any]:
    scores = [row[score_key]["classical_brier"] for row in rows]
    outcome_counts = [row[score_key]["outcome_count"] for row in rows]
    exclusive_flags = [row[score_key]["is_exclusive"] for row in rows]
    return summarize_event_scores(
        scores,
        outcome_counts=outcome_counts,
        exclusive_flags=exclusive_flags,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    ).to_dict()


def _model_report(
    rows: list[dict[str, Any]],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    report = _score_summary(rows, bootstrap_resamples, seed)
    market_rows = [row for row in rows if row.get("market_baseline_score")]
    if market_rows:
        report["market_baseline_on_same_events"] = _score_summary(
            market_rows,
            bootstrap_resamples,
            seed,
            score_key="market_baseline_score",
        )
        diffs = [
            row["score"]["classical_brier"] - row["market_baseline_score"]["classical_brier"]
            for row in market_rows
        ]
        if diffs:
            report["model_vs_market"] = {
                "n": len(diffs),
                "mean_model_minus_market_brier": sum(diffs) / len(diffs),
                "model_beats_market_n": sum(1 for value in diffs if value < 0),
                "market_beats_model_n": sum(1 for value in diffs if value > 0),
                "ties_n": sum(1 for value in diffs if value == 0),
                "better_baseline": "model" if sum(diffs) / len(diffs) < 0 else "market",
            }
    calibrated_rows = [row for row in rows if row.get("calibrated_blend_score")]
    if calibrated_rows:
        report["calibrated_blend"] = _score_summary(
            calibrated_rows,
            bootstrap_resamples,
            seed,
            score_key="calibrated_blend_score",
        )
        calibration = calibrated_rows[0].get("calibration") or {}
        report["calibrated_blend"]["calibration"] = calibration
        market_diffs = [
            row["calibrated_blend_score"]["classical_brier"] - row["market_baseline_score"]["classical_brier"]
            for row in calibrated_rows
            if row.get("market_baseline_score")
        ]
        raw_diffs = [
            row["calibrated_blend_score"]["classical_brier"] - row["score"]["classical_brier"]
            for row in calibrated_rows
        ]
        if market_diffs:
            report["calibrated_blend_vs_market"] = {
                "n": len(market_diffs),
                "mean_calibrated_minus_market_brier": sum(market_diffs) / len(market_diffs),
                "calibrated_beats_market_n": sum(1 for value in market_diffs if value < 0),
                "market_beats_calibrated_n": sum(1 for value in market_diffs if value > 0),
                "ties_n": sum(1 for value in market_diffs if value == 0),
                "better_baseline": "calibrated" if sum(market_diffs) / len(market_diffs) < 0 else "market",
            }
        if raw_diffs:
            report["calibrated_blend_vs_raw_model"] = {
                "n": len(raw_diffs),
                "mean_calibrated_minus_raw_brier": sum(raw_diffs) / len(raw_diffs),
                "calibrated_beats_raw_n": sum(1 for value in raw_diffs if value < 0),
                "raw_beats_calibrated_n": sum(1 for value in raw_diffs if value > 0),
                "ties_n": sum(1 for value in raw_diffs if value == 0),
                "better_baseline": "calibrated" if sum(raw_diffs) / len(raw_diffs) < 0 else "raw_model",
            }
    return report


def _segment_value(row: dict[str, Any], group_by: str) -> str:
    if group_by == "category":
        return str(row.get("category") or "unknown")
    if group_by == "is_binary":
        score = row.get("score") or {}
        return "binary" if score.get("is_binary") else "nonbinary"
    traits = row.get("event_traits") or {}
    value = traits.get(group_by)
    if value is None:
        return "unknown"
    return str(value)


def _print_report(
    rows: list[dict[str, Any]],
    bootstrap_resamples: int,
    seed: int,
    *,
    group_by: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    by_model = sorted({row["model_name"] for row in rows})
    for model_name in by_model:
        model_rows = [row for row in rows if row["model_name"] == model_name and row.get("score")]
        if model_rows:
            report[model_name] = _model_report(model_rows, bootstrap_resamples, seed)

    if len(by_model) == 2:
        left, right = by_model
        left_scores = {
            _event_key(row): row["score"]["classical_brier"]
            for row in rows
            if row["model_name"] == left and row.get("score")
        }
        right_scores = {
            _event_key(row): row["score"]["classical_brier"]
            for row in rows
            if row["model_name"] == right and row.get("score")
        }
        common = sorted(set(left_scores) & set(right_scores))
        diffs = [left_scores[key] - right_scores[key] for key in common]
        if diffs:
            mean_diff = sum(diffs) / len(diffs)
            report["paired_difference"] = {
                "left_model": left,
                "right_model": right,
                "n": len(diffs),
                "classical_brier_left_minus_right": mean_diff,
                "better_model": left if mean_diff < 0 else right,
            }

    if group_by:
        segments: dict[str, dict[str, Any]] = {}
        segment_keys = sorted({_segment_value(row, group_by) for row in rows if row.get("score")})
        for segment in segment_keys:
            segment_rows = [
                row
                for row in rows
                if row.get("score") and _segment_value(row, group_by) == segment
            ]
            segments[segment] = {}
            for model_name in by_model:
                model_rows = [row for row in segment_rows if row["model_name"] == model_name]
                if model_rows:
                    segments[segment][model_name] = _model_report(
                        model_rows,
                        bootstrap_resamples,
                        seed,
                    )
        report["segments"] = {"group_by": group_by, "values": segments}

    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_PRESETS), default=["flash", "pro"])
    parser.add_argument("--report-existing", type=Path, default=None, help="score an existing Gemini JSONL run against the same-event market baseline without model calls")
    parser.add_argument("--source", choices=("subset_1200", "kalshi_topvol"), default="subset_1200")
    parser.add_argument("--horizon", default="7d", help="for --source kalshi_topvol: prediction offset before close, e.g. 7d, 1d, 6h, 1h")
    parser.add_argument("--target-horizon", default=None, help="filter already-snapshotted events by time to close, e.g. 7d or 1d")
    parser.add_argument("--horizon-tolerance", default="12h", help="allowed +/- window for --target-horizon")
    parser.add_argument("--max-rank", type=int, default=300, help="for --source kalshi_topvol: use top N markets per week; 0 disables")
    parser.add_argument("--sample", type=int, default=0, help="random sample size across selected events")
    parser.add_argument("--limit", type=int, default=0, help="first N events after filtering; ignored when --sample is set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-mode",
        choices=("random", "binary_nonbinary", "week", "category", "horizon", "outcome_count"),
        default="random",
    )
    parser.add_argument("--stratified", action="store_true", help="alias for --sample-mode binary_nonbinary")
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument("--nonbinary-only", action="store_true")
    parser.add_argument("--category", default=None)
    parser.add_argument("--max-outcomes", type=int, default=0, help="skip very large outcome sets to control token cost; 0 disables")
    parser.add_argument("--max-tokens", type=int, default=0, help="override model max output tokens; useful for search-grounded runs")
    parser.add_argument(
        "--market-blend-weight",
        type=float,
        default=0.0,
        help="also score a calibrated linear blend: final = market + weight * (model - market); 0 disables",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable-google-search", action="store_true", help="enable Gemini Google Search grounding; prompt enforces source dates before as_of")
    parser.add_argument(
        "--group-report-by",
        choices=("none", "horizon_bucket", "category", "outcome_count", "source_week", "is_binary"),
        default="none",
        help="also print segmented score summaries for the selected characteristic",
    )
    parser.add_argument("--no-model-fallback", action="store_true", help="do not try alternate Gemini preview aliases")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()

    if not 0.0 <= args.market_blend_weight <= 1.0:
        raise SystemExit("--market-blend-weight must be between 0 and 1")

    if args.report_existing:
        rows = list(_load_cache(args.report_existing).values())
        rows = _add_market_baselines_to_rows(
            rows,
            blend_model_weight=args.market_blend_weight,
        )
        scored = [row for row in rows if row.get("score")]
        group_by = None if args.group_report_by == "none" else args.group_report_by
        print(f"Read audit log: {args.report_existing}")
        _print_report(scored, args.bootstrap_resamples, args.seed, group_by=group_by)
        return 0

    if args.binary_only and args.nonbinary_only:
        raise SystemExit("--binary-only and --nonbinary-only are mutually exclusive")
    if args.max_outcomes == 0:
        args.max_outcomes = None
    if args.max_rank == 0:
        args.max_rank = None
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set. Export it, then rerun this script.")

    events = _select_events(args)
    if not events:
        raise SystemExit("No events matched the selected filters.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or DEFAULT_OUT_DIR / f"gemini_solo_{stamp}.jsonl"
    if args.overwrite and out_path.exists():
        out_path.unlink()
    cache = {} if args.overwrite else _load_cache(out_path)

    configs = []
    for name in args.models:
        config = ForecasterConfig.from_dict(MODEL_PRESETS[name])
        if args.enable_google_search:
            config = replace(config, enable_google_search=True)
        if args.max_tokens:
            config = replace(config, max_tokens=args.max_tokens)
        configs.append((name, config))
    total = len(events) * len(configs)
    done = 0
    for preset_name, config in configs:
        for event in events:
            done += 1
            row_event_key = event.submission_id or f"{event.event['event_ticker']}|{event.snapshot_time}"
            key = (config.name, row_event_key)
            if key in cache:
                cached = cache[key]
                if cached.get("score") and (
                    not cached.get("market_baseline_score")
                    or (args.market_blend_weight > 0 and not cached.get("calibrated_blend_score"))
                ):
                    cached = _add_market_baseline(
                        cached,
                        event,
                        blend_model_weight=args.market_blend_weight,
                    )
                    _append_row(out_path, cached)
                    cache[key] = cached
                print(f"[{done}/{total}] cached {config.name} {event.event['event_ticker']}", flush=True)
                continue

            packet = packet_from_arena_event(event.event)
            try:
                forecast, actual_model = _forecast_with_fallback(
                    config,
                    packet,
                    MODEL_ALIASES.get(preset_name, []),
                    allow_fallback=not args.no_model_fallback,
                )
                raw_probs = _raw_probabilities(forecast)
                score = _score_row(event, raw_probs)
                baseline = _market_baseline(event)
                row = {
                    "model_name": config.name,
                    "model": actual_model,
                    "configured_model": config.model,
                    "event_ticker": event.event["event_ticker"],
                    "submission_id": event.submission_id,
                    "snapshot_time": event.snapshot_time,
                    "category": event.event.get("category"),
                    "outcomes": event.outcomes,
                    "actuals": event.actuals,
                    "event_traits": _event_traits(event, args),
                    "raw_probabilities": raw_probs,
                    "normalized_model_probabilities": forecast.probabilities,
                    "score": score,
                    "reasoning_track": forecast.reasoning_track.to_dict(),
                    "diagnostics": forecast.diagnostics.to_dict(),
                    "grounding_metadata": _grounding_metadata(forecast),
                }
                if baseline is not None:
                    row["market_baseline_probabilities"] = baseline["probabilities"]
                    row["market_baseline_score"] = baseline["score"]
                    row["model_minus_market_brier"] = (
                        score["classical_brier"] - baseline["score"]["classical_brier"]
                    )
                    row["model_beats_market"] = row["model_minus_market_brier"] < 0
                    if args.market_blend_weight > 0:
                        blend = _calibrated_blend(
                            event,
                            raw_probs,
                            baseline,
                            model_weight=args.market_blend_weight,
                        )
                        if blend is not None:
                            row["calibrated_blend_probabilities"] = blend["probabilities"]
                            row["calibrated_blend_score"] = blend["score"]
                            row["calibration"] = {
                                "type": "market_linear_blend",
                                "model_weight": blend["model_weight"],
                                "market_weight": blend["market_weight"],
                            }
                            row["calibrated_minus_market_brier"] = (
                                blend["score"]["classical_brier"] - baseline["score"]["classical_brier"]
                            )
                            row["calibrated_minus_raw_brier"] = (
                                blend["score"]["classical_brier"] - score["classical_brier"]
                            )
            except Exception as exc:  # noqa: BLE001
                if not args.continue_on_error:
                    raise
                row = {
                    "model_name": config.name,
                    "model": config.model,
                    "configured_model": config.model,
                    "event_ticker": event.event["event_ticker"],
                    "submission_id": event.submission_id,
                    "snapshot_time": event.snapshot_time,
                    "category": event.event.get("category"),
                    "outcomes": event.outcomes,
                    "actuals": event.actuals,
                    "event_traits": _event_traits(event, args),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _append_row(out_path, row)
            cache[key] = row
            print(f"[{done}/{total}] wrote {config.name} {event.event['event_ticker']}", flush=True)
            time.sleep(0.2)

    scored = [row for row in _load_cache(out_path).values() if row.get("score")]
    print(f"\nWrote audit log: {out_path}")
    group_by = None if args.group_report_by == "none" else args.group_report_by
    _print_report(scored, args.bootstrap_resamples, args.seed, group_by=group_by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
