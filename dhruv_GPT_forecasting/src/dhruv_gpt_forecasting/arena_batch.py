"""Batch helpers for local Prophet Arena sample-dataset iteration."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .arena_agent import forecast_arena_event
from .arena_eval import actual_market_label, evaluate_predictions
from .config import load_config, load_local_env, resolve_api_key
from .key_utils import key_fingerprint
from .prophet_api import write_events


def predict_events(
    events: list[dict[str, Any]],
    *,
    use_gpt: bool = False,
    use_live_data: bool = False,
    backtest_internet: bool = False,
    deadline_seconds: float | None = 300.0,
) -> dict[str, Any]:
    """Return a Prophet-style submission for a retrieved events array."""
    cfg = load_config()
    if backtest_internet:
        cfg.arena.grounded_research_backtest_enabled = True
        use_live_data = True
    rows = []
    for event in events:
        market_ticker = str(event.get("market_ticker") or event.get("task_id") or event.get("event_ticker") or "")
        if not market_ticker:
            continue
        forecast = forecast_arena_event(
            event,
            config=cfg,
            use_gpt=use_gpt,
            use_live_data=use_live_data,
            deadline_seconds=deadline_seconds,
        )
        rows.append({
            "market_ticker": market_ticker,
            "probabilities": [
                {"market": market, "probability": probability}
                for market, probability in forecast.probabilities.items()
            ],
            "rationale": forecast.audit.get("calibration_note") or forecast.source,
        })
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "predictions": rows,
    }


def actuals_from_events(events: list[dict[str, Any]]) -> dict[str, str]:
    """Extract actuals from retrieved resolved events using market_ticker join keys."""
    actuals: dict[str, str] = {}
    for event in events:
        ticker = str(event.get("market_ticker") or event.get("task_id") or event.get("event_ticker") or "")
        actual = event.get("resolved_outcome") or event.get("actual_outcome")
        if ticker and actual is not None:
            actuals[ticker] = actual_market_label(actual)
    return actuals


def benchmark_events(
    events: list[dict[str, Any]],
    *,
    output_dir: Path,
    dataset: str | None = None,
    release: str | None = None,
    seed: int = 17,
    as_of: str | None = None,
    evidence_mode: str = "strict_pit",
    evidence_manifest_ids: list[str] | None = None,
    limit: int | None = None,
    with_gpt: bool = False,
    live_data: bool = False,
    backtest_internet: bool = False,
    deadline_seconds: float | None = 300.0,
) -> dict[str, Any]:
    """Run a resolved raw-task benchmark and write a reproducible run folder."""
    load_local_env()
    cfg = load_config()
    if backtest_internet:
        cfg.arena.grounded_research_backtest_enabled = True
        live_data = True
    resolved = [event for event in events if event.get("resolved_outcome") is not None or event.get("actual_outcome") is not None]
    rng = random.Random(seed)
    rng.shuffle(resolved)
    selected = resolved[:limit] if limit else resolved
    if as_of:
        for event in selected:
            event["as_of"] = as_of
    actuals = actuals_from_events(selected)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_by_variant: dict[str, Any] = {
        "uniform": _uniform_predictions(selected),
        "deterministic": predict_events(
            selected,
            use_gpt=False,
            use_live_data=False,
            deadline_seconds=deadline_seconds,
        ),
    }
    if with_gpt:
        predictions_by_variant["gpt"] = _predict_with_config(
            selected,
            cfg=cfg,
            use_gpt=True,
            use_live_data=live_data,
            backtest_internet=backtest_internet,
            deadline_seconds=deadline_seconds,
        )
        shrink_cfg = copy.deepcopy(cfg)
        shrink_cfg.arena.prior_shrink_weight = max(0.25, shrink_cfg.arena.prior_shrink_weight)
        predictions_by_variant["gpt_shrink"] = _predict_with_config(
            selected,
            cfg=shrink_cfg,
            use_gpt=True,
            use_live_data=live_data,
            backtest_internet=backtest_internet,
            deadline_seconds=deadline_seconds,
        )

    metrics = {
        name: evaluate_predictions(predictions, actuals, events=selected)
        for name, predictions in predictions_by_variant.items()
    }
    key, key_env = resolve_api_key(cfg.cheap_model)
    run_config = {
        "dataset": dataset,
        "release": release,
        "source_path": None,
        "random_seed": seed,
        "forecast_as_of": as_of,
        "evidence_mode": evidence_mode,
        "evidence_manifest_ids": evidence_manifest_ids or [],
        "model_id": cfg.cheap_model.model,
        "prompt_hash": "per_event_prompt_hash_recorded_in_llm_cache",
        "api_key_env": key_env,
        "api_key_fingerprint": key_fingerprint(key),
        "with_gpt": with_gpt,
        "live_data": live_data,
        "backtest_internet": backtest_internet,
        "backtest_internet_policy": (
            "native-search and internet records require source-specific published_at at or before forecast_as_of"
            if backtest_internet
            else "disabled"
        ),
        "deadline_seconds": deadline_seconds,
        "n_resolved_events": len(resolved),
        "n_selected_events": len(selected),
        "variants": sorted(predictions_by_variant),
    }
    _write_json(output_dir / "run_config.json", run_config)
    _write_json(output_dir / "actuals.json", actuals)
    _write_json(output_dir / "metrics.json", metrics)
    for name, predictions in predictions_by_variant.items():
        _write_json(output_dir / f"predictions_{name}.json", predictions)
    _write_json(output_dir / "predictions.json", predictions_by_variant.get("gpt_shrink") or predictions_by_variant["deterministic"])
    (output_dir / "report.md").write_text(_benchmark_report(run_config, metrics), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "run_config": run_config,
        "metrics": metrics,
    }


def _load_events(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped and not stripped.startswith(("[", "{")):
        return [_normalize_task_row(json.loads(line)) for line in text.splitlines() if line.strip()]
    raw = json.loads(text)
    if isinstance(raw, list):
        return [_normalize_task_row(item) for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        events = raw.get("events") or raw.get("data")
        if isinstance(events, list):
            return [_normalize_task_row(item) for item in events if isinstance(item, dict)]
        return [_normalize_task_row(raw)]
    return []


def _uniform_predictions(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for event in events:
        ticker = str(event.get("market_ticker") or event.get("task_id") or event.get("event_ticker") or "")
        outcomes = [str(outcome) for outcome in event.get("outcomes") or ["YES", "NO"]]
        p = 1.0 / len(outcomes) if outcomes else 1.0
        rows.append({
            "market_ticker": ticker,
            "probabilities": [{"market": outcome, "probability": p} for outcome in outcomes],
            "rationale": "uniform_baseline",
        })
    return {"timestamp": datetime.now(UTC).isoformat(), "predictions": rows}


def _predict_with_config(
    events: list[dict[str, Any]],
    *,
    cfg,
    use_gpt: bool,
    use_live_data: bool,
    deadline_seconds: float | None,
    backtest_internet: bool = False,
) -> dict[str, Any]:
    if backtest_internet:
        cfg.arena.grounded_research_backtest_enabled = True
        use_live_data = True
    rows = []
    for event in events:
        ticker = str(event.get("market_ticker") or event.get("task_id") or event.get("event_ticker") or "")
        forecast = forecast_arena_event(
            event,
            config=cfg,
            use_gpt=use_gpt,
            use_live_data=use_live_data,
            deadline_seconds=deadline_seconds,
        )
        rows.append({
            "market_ticker": ticker,
            "probabilities": [
                {"market": market, "probability": probability}
                for market, probability in forecast.probabilities.items()
            ],
            "rationale": forecast.audit.get("calibration_note") or forecast.source,
            "audit": {
                "source": forecast.source,
                "model": forecast.audit.get("model"),
                "api_logs": forecast.audit.get("api_logs", []),
                "prior_shrink_weight": forecast.audit.get("prior_shrink_weight"),
                "fallback_reason": forecast.audit.get("fallback_reason"),
                "live_evidence_sources": forecast.audit.get("live_evidence_sources"),
                "live_evidence_errors": forecast.audit.get("live_evidence_errors"),
            },
        })
    return {"timestamp": datetime.now(UTC).isoformat(), "predictions": rows}


def _default_run_dir(prefix: str = "arena_benchmark") -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("dhruv_GPT_forecasting/logs/runs") / f"{stamp}_{prefix}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _benchmark_report(run_config: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        "# Arena Resolved Benchmark",
        "",
        f"- Dataset: {run_config.get('dataset') or 'local'}",
        f"- Release: {run_config.get('release') or 'unspecified'}",
        f"- Evidence mode: {run_config.get('evidence_mode')}",
        f"- Model: {run_config.get('model_id')}",
        f"- Key fingerprint: {run_config.get('api_key_fingerprint') or 'missing'}",
        f"- Selected events: {run_config.get('n_selected_events')}",
        "",
        "## Brier by Variant",
        "",
        "| Variant | N | Brier |",
        "| --- | ---: | ---: |",
    ]
    for name, result in sorted(metrics.items()):
        brier = result.get("brier")
        brier_text = f"{brier:.6f}" if isinstance(brier, float) else str(brier)
        lines.append(f"| {name} | {result.get('n', 0)} | {brier_text} |")
    lines.extend(["", "## Slices", ""])
    for name, result in sorted(metrics.items()):
        lines.append(f"### {name}")
        lines.append("")
        for segment_name in ("category_metrics", "outcome_count_metrics", "structure_metrics"):
            lines.append(f"- {segment_name}: {json.dumps(result.get(segment_name, []), sort_keys=True)}")
        lines.append("")
    return "\n".join(lines)


def _normalize_task_row(item: dict[str, Any]) -> dict[str, Any]:
    """Accept retrieved Event JSON and raw ai-prophet-datasets task rows."""
    if "market_ticker" in item and "close_time" in item:
        return item
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    event = dict(item)
    task_id = str(item.get("task_id") or item.get("market_ticker") or item.get("event_ticker") or "")
    event.setdefault("event_ticker", source.get("event_ticker") or item.get("source") or task_id)
    event.setdefault("market_ticker", task_id)
    event.setdefault("description", item.get("context"))
    event.setdefault("rules", source.get("rules") or item.get("context"))
    event.setdefault("category", metadata.get("category") or source.get("category") or "Unknown")
    event.setdefault("close_time", item.get("predict_by") or source.get("close_time"))
    return event


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    predict = sub.add_parser("predict")
    predict.add_argument("--events", type=Path, required=True)
    predict.add_argument("-o", "--output", type=Path, required=True)
    predict.add_argument("--limit", type=int)
    predict.add_argument("--with-gpt", action="store_true")
    predict.add_argument("--live-data", action="store_true")
    predict.add_argument(
        "--backtest-internet",
        action="store_true",
        help="Allow historical backtests to use internet source-reading only after PIT publish-date verification.",
    )
    predict.add_argument("--deadline-seconds", type=float, default=300.0)

    actuals = sub.add_parser("actuals")
    actuals.add_argument("--events", type=Path, required=True)
    actuals.add_argument("-o", "--output", type=Path, required=True)

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--events", type=Path, required=True)
    benchmark.add_argument("-o", "--output-dir", type=Path)
    benchmark.add_argument("--dataset")
    benchmark.add_argument("--release")
    benchmark.add_argument("--seed", type=int, default=17)
    benchmark.add_argument("--as-of")
    benchmark.add_argument("--evidence-mode", choices=["strict_pit", "relaxed_published_at", "live_smoke"], default="strict_pit")
    benchmark.add_argument("--evidence-manifest-id", action="append", default=[])
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--with-gpt", action="store_true")
    benchmark.add_argument("--live-data", action="store_true")
    benchmark.add_argument(
        "--backtest-internet",
        action="store_true",
        help="Enable internet source-reading in historical runs with strict published_at <= as_of filtering.",
    )
    benchmark.add_argument("--deadline-seconds", type=float, default=300.0)

    runbook = sub.add_parser("runbook")
    runbook.add_argument("--events", type=Path)
    runbook.add_argument("-o", "--output-dir", type=Path)
    runbook.add_argument("--status", choices=["all", "open", "closed"], default="open")
    runbook.add_argument("--dataset")
    runbook.add_argument("--release")
    runbook.add_argument("--seed", type=int, default=17)
    runbook.add_argument("--limit", type=int)
    runbook.add_argument("--with-gpt", action="store_true")
    runbook.add_argument("--live-data", action="store_true")
    runbook.add_argument("--backtest-internet", action="store_true")

    args = parser.parse_args()
    if args.cmd == "runbook" and args.events is None:
        if not os.environ.get("PA_SERVER_API_KEY"):
            load_local_env()
        if not os.environ.get("PA_SERVER_API_KEY"):
            raise RuntimeError("runbook requires --events or PA_SERVER_API_KEY for fetching events")
        out_dir = args.output_dir or _default_run_dir("runbook")
        events_path = out_dir / "events.json"
        events = write_events(events_path, status=args.status)
        _write_json(events_path, events)
    else:
        events = _load_events(args.events)
    if args.cmd == "actuals":
        payload = actuals_from_events(events)
    elif args.cmd in {"benchmark", "runbook"}:
        out_dir = args.output_dir or _default_run_dir(args.cmd)
        payload = benchmark_events(
            events,
            output_dir=out_dir,
            dataset=args.dataset,
            release=args.release,
            seed=args.seed,
            as_of=getattr(args, "as_of", None),
            evidence_mode=getattr(args, "evidence_mode", "live_smoke" if args.cmd == "runbook" else "strict_pit"),
            evidence_manifest_ids=getattr(args, "evidence_manifest_id", []),
            limit=args.limit,
            with_gpt=args.with_gpt,
            live_data=args.live_data,
            backtest_internet=getattr(args, "backtest_internet", False),
            deadline_seconds=getattr(args, "deadline_seconds", 300.0),
        )
    else:
        selected = events[: args.limit] if args.limit else events
        payload = predict_events(
            selected,
            use_gpt=args.with_gpt,
            use_live_data=args.live_data,
            backtest_internet=getattr(args, "backtest_internet", False),
            deadline_seconds=args.deadline_seconds,
        )
    if args.cmd in {"benchmark", "runbook"}:
        print(json.dumps({"output_dir": payload["output_dir"], "n": payload["run_config"]["n_selected_events"]}, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n": len(payload.get("predictions", payload))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
