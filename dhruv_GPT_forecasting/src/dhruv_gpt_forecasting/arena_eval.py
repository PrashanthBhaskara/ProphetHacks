"""Local Prophet Arena prediction evaluator."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .constraints import normalize_distribution


def evaluate_predictions(
    predictions: Any,
    actuals: dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event_by_ticker = {
        str(event.get("market_ticker") or event.get("task_id") or event.get("event_ticker")): event
        for event in (events or [])
    }
    rows = []
    for ticker, pred in _iter_prediction_items(predictions):
        actual = actuals.get(ticker)
        if actual is None:
            continue
        event = event_by_ticker.get(ticker, {})
        outcomes = list(event.get("outcomes") or _prediction_outcomes(pred, actual))
        probs = _prediction_probabilities(pred, outcomes)
        score = event_brier(probs, actual, outcomes)
        actual_label = actual_market_label(actual)
        rows.append({
            "market_ticker": ticker,
            "actual": actual_label,
            "brier": score,
            "category": event.get("category") or "Unknown",
            "n_outcomes": len(outcomes),
            "event_structure": _structure_label(outcomes),
        })
    by_category = _segment(rows, "category")
    by_outcome_count = _segment(rows, "n_outcomes")
    by_structure = _segment(rows, "event_structure")
    return {
        "n": len(rows),
        "brier": sum(row["brier"] for row in rows) / len(rows) if rows else float("nan"),
        "category_metrics": by_category,
        "outcome_count_metrics": by_outcome_count,
        "structure_metrics": by_structure,
        "matched_tickers": [row["market_ticker"] for row in rows],
        "missing_actuals": sorted(set(_prediction_tickers(predictions)) - set(actuals)),
    }


def event_brier(probs: dict[str, float], actual: Any, outcomes: list[str]) -> float:
    actual_label = actual_market_label(actual)
    clean = normalize_distribution(probs, outcomes, lo=0.0, hi=1.0)
    if actual_label not in outcomes:
        raise ValueError(f"Actual outcome {actual_label!r} missing from outcomes")
    return sum((clean.get(outcome, 0.0) - (1.0 if outcome == actual_label else 0.0)) ** 2 for outcome in outcomes)


def load_actuals(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "predictions" in raw or "probabilities" in raw:
            raise ValueError("actuals must not be a prediction/submission object")
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, list):
        out = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("market_ticker") or item.get("task_id") or item.get("event_ticker") or "")
            actual = item.get("resolved_outcome") or item.get("actual_outcome")
            if ticker and actual is not None:
                out[ticker] = actual
        return out
    raise ValueError("actuals must be a JSON object or list")


def load_actuals_from_tasks(path: Path) -> dict[str, Any]:
    out = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        ticker = str(item.get("market_ticker") or item.get("task_id") or item.get("event_ticker") or "")
        actual = item.get("resolved_outcome") or item.get("actual_outcome")
        if ticker and actual is not None:
            out[ticker] = actual
    return out


def load_events(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped and not stripped.startswith(("[", "{")):
        return [_normalize_event_row(json.loads(line)) for line in text.splitlines() if line.strip()]
    raw = json.loads(text)
    if isinstance(raw, list):
        return [_normalize_event_row(item) for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        events = raw.get("events") or raw.get("data")
        if isinstance(events, list):
            return [_normalize_event_row(item) for item in events if isinstance(item, dict)]
        return [_normalize_event_row(raw)]
    return None


def _normalize_event_row(item: dict[str, Any]) -> dict[str, Any]:
    if "market_ticker" in item and "category" in item:
        return item
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    event = dict(item)
    task_id = str(item.get("task_id") or item.get("market_ticker") or item.get("event_ticker") or "")
    event.setdefault("market_ticker", task_id)
    event.setdefault("event_ticker", source.get("event_ticker") or task_id)
    event.setdefault("category", metadata.get("category") or source.get("category") or "Unknown")
    event.setdefault("close_time", item.get("predict_by") or source.get("close_time"))
    event.setdefault("description", item.get("context"))
    event.setdefault("rules", source.get("rules") or item.get("context"))
    return event


def actual_market_label(value: Any) -> str:
    """Match ai-prophet's distribution scorer for resolved_outcome payloads."""
    if isinstance(value, dict):
        if "value" in value:
            return actual_market_label(value["value"])
        for key in ("market", "outcome", "label", "name"):
            if value.get(key) is not None:
                return str(value[key])
    if isinstance(value, list):
        if not value:
            raise ValueError("Actual outcome list is empty")
        return actual_market_label(value[0])
    return str(value)


def _iter_prediction_items(predictions: Any):
    if isinstance(predictions, dict) and "predictions" in predictions:
        predictions = predictions["predictions"]
    if isinstance(predictions, dict) and "probabilities" in predictions:
        ticker = str(predictions.get("market_ticker") or predictions.get("task_id") or predictions.get("event_ticker") or "")
        yield ticker, predictions
    elif isinstance(predictions, dict):
        for ticker, pred in predictions.items():
            yield str(ticker), pred
    elif isinstance(predictions, list):
        for item in predictions:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("market_ticker") or item.get("task_id") or item.get("event_ticker") or "")
            yield ticker, item


def _prediction_tickers(predictions: Any) -> list[str]:
    return [ticker for ticker, _ in _iter_prediction_items(predictions) if ticker]


def _prediction_outcomes(prediction: Any, actual: str) -> list[str]:
    if isinstance(prediction, dict):
        probs = prediction.get("probabilities")
        if isinstance(probs, list):
            labels = [str(item.get("market")) for item in probs if isinstance(item, dict) and item.get("market") is not None]
            return labels or [actual]
        if isinstance(probs, dict):
            labels = [str(key) for key in probs]
            return labels or [actual]
    return [actual]


def _prediction_probabilities(prediction: Any, outcomes: list[str]) -> dict[str, float]:
    if isinstance(prediction, dict):
        probs = prediction.get("probabilities")
        if isinstance(probs, list):
            return {
                str(item.get("market")): float(item.get("probability"))
                for item in probs
                if isinstance(item, dict) and item.get("market") is not None and _is_number(item.get("probability"))
            }
        if isinstance(probs, dict):
            return {str(key): float(value) for key, value in probs.items() if _is_number(value)}
    return {outcome: 1.0 / max(1, len(outcomes)) for outcome in outcomes}


def _segment(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(float(row["brier"]))
    return [
        {"segment": str(segment), "n": len(scores), "brier": sum(scores) / len(scores)}
        for segment, scores in sorted(grouped.items(), key=lambda item: (-len(item[1]), str(item[0])))
    ]


def _structure_label(outcomes: list[str]) -> str:
    if [outcome.upper() for outcome in outcomes] == ["YES", "NO"]:
        return "binary_yes_no"
    if len(outcomes) == 2:
        return "binary_named"
    return "multi_outcome"


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--actuals", type=Path)
    parser.add_argument("--tasks-jsonl", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.actuals and not args.tasks_jsonl:
        raise SystemExit("provide --actuals or --tasks-jsonl")
    predictions = json.loads(args.submission.read_text(encoding="utf-8"))
    actuals = load_actuals(args.actuals) if args.actuals else load_actuals_from_tasks(args.tasks_jsonl)
    events = load_events(args.events)
    result = evaluate_predictions(predictions, actuals, events=events)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
