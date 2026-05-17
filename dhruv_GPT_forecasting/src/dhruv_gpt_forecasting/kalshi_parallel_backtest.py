"""Parallel random-PIT Kalshi backtest runner."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .arena_priors import build_arena_packet
from .backtest import brier, ece
from .config import ForecastConfig, load_config, load_local_env
from .data_loaders import BacktestSample, load_nonbinary_component_samples, load_topvol_samples
from .experiments import random_point_in_time_samples
from .features import build_feature_packet
from .forecaster import forecast_event
from .grounded_research import gather_grounded_research_evidence
from .pit_evidence import gather_pit_external_evidence
from .schemas import FeaturePacket, clamp_prob
from .stat_router import forecast_stat_routed


ForecastMode = str


@dataclass(frozen=True)
class TaggedSample:
    source_dataset: str
    sample: BacktestSample


def run_parallel_kalshi_backtest(
    *,
    total: int = 300,
    topvol_count: int | None = None,
    nonbinary_count: int | None = None,
    seed: int = 20260517,
    candle_stride_minutes: int = 1,
    min_history_snapshots: int = 5,
    min_horizon_minutes: float = 5.0,
    max_horizon_hours: float | None = None,
    decision_budget_minutes: float = 5.0,
    forecast_mode: ForecastMode = "gpt",
    max_workers: int = 12,
    include_context: bool = True,
    force_cheap: bool = True,
    with_supervisor: bool = False,
    pit_external_evidence: bool = False,
    pit_allow_network: bool = False,
    pit_nonstrict_collected_at: bool = False,
    backtest_internet: bool = False,
    per_contract_timeout_seconds: float = 480.0,
    output_dir: Path | None = None,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Run one random request-time forecast per sampled Kalshi market."""
    load_local_env()
    cfg = load_config()
    cfg.supervisor_model.enabled = bool(with_supervisor)
    if backtest_internet:
        cfg.arena.grounded_research_backtest_enabled = True
        os.environ.setdefault("ARENA_ENABLE_BACKTEST_INTERNET", "1")
    samples = _sample_universe(
        total=total,
        topvol_count=topvol_count,
        nonbinary_count=nonbinary_count,
        seed=seed,
        candle_stride_minutes=candle_stride_minutes,
        min_history_snapshots=min_history_snapshots,
        min_horizon_minutes=min_horizon_minutes,
        max_horizon_hours=max_horizon_hours,
        decision_budget_minutes=decision_budget_minutes,
    )
    if not samples:
        raise ValueError("no eligible random point-in-time Kalshi samples")

    started = time.time()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    max_workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _forecast_one,
                idx,
                tagged,
                cfg,
                forecast_mode=forecast_mode,
                include_context=include_context,
                force_cheap=force_cheap,
                pit_external_evidence=pit_external_evidence,
                pit_allow_network=pit_allow_network,
                pit_nonstrict_collected_at=pit_nonstrict_collected_at,
                backtest_internet=backtest_internet,
                per_contract_timeout_seconds=per_contract_timeout_seconds,
            ): idx
            for idx, tagged in enumerate(samples)
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001 - a single market should not kill the run.
                errors.append({
                    "index": futures[future],
                    "error": f"{type(exc).__name__}:{exc}",
                })
            if progress_every and (completed % progress_every == 0 or completed == len(futures)):
                print(json.dumps({
                    "completed": completed,
                    "total": len(futures),
                    "errors": len(errors),
                    "elapsed_seconds": round(time.time() - started, 2),
                }), flush=True)

    rows.sort(key=lambda row: (row.get("as_of") or "", row.get("market_ticker") or ""))
    summary = _summarize_rows(rows, errors)
    run_config = {
        "total_requested": total,
        "topvol_count_requested": topvol_count,
        "nonbinary_count_requested": nonbinary_count,
        "seed": seed,
        "candle_stride_minutes": candle_stride_minutes,
        "min_history_snapshots": min_history_snapshots,
        "min_horizon_minutes": min_horizon_minutes,
        "max_horizon_hours": max_horizon_hours,
        "decision_budget_minutes": decision_budget_minutes,
        "forecast_mode": forecast_mode,
        "max_workers": max_workers,
        "include_context": include_context,
        "force_cheap": force_cheap,
        "with_supervisor": with_supervisor,
        "pit_external_evidence": pit_external_evidence,
        "pit_allow_network": pit_allow_network,
        "pit_nonstrict_collected_at": pit_nonstrict_collected_at,
        "backtest_internet": backtest_internet,
        "per_contract_timeout_seconds": per_contract_timeout_seconds,
        "model": cfg.cheap_model.model if forecast_mode == "gpt" else None,
        "started_at": datetime.fromtimestamp(started, tz=UTC).isoformat(),
        "elapsed_seconds": time.time() - started,
    }
    result = {
        "run_config": run_config,
        "summary": summary,
        "rows": rows,
        "errors": errors,
    }
    out_dir = output_dir or _default_run_dir()
    _write_run(out_dir, result)
    result["output_dir"] = str(out_dir)
    return result


def _sample_universe(
    *,
    total: int,
    topvol_count: int | None,
    nonbinary_count: int | None,
    seed: int,
    candle_stride_minutes: int,
    min_history_snapshots: int,
    min_horizon_minutes: float,
    max_horizon_hours: float | None,
    decision_budget_minutes: float,
) -> list[TaggedSample]:
    if topvol_count is None and nonbinary_count is None:
        topvol_count = total // 2
        nonbinary_count = total - topvol_count
    elif topvol_count is None:
        topvol_count = max(0, total - int(nonbinary_count or 0))
    elif nonbinary_count is None:
        nonbinary_count = max(0, total - int(topvol_count or 0))
    topvol_raw = load_topvol_samples(
        candle_stride_minutes=candle_stride_minutes,
        min_snapshots=min_history_snapshots,
    )
    nonbinary_raw = load_nonbinary_component_samples(
        candle_stride_minutes=candle_stride_minutes,
        min_snapshots=min_history_snapshots,
    )
    topvol = random_point_in_time_samples(
        topvol_raw,
        n_events=topvol_count,
        seed=seed,
        min_horizon_minutes=min_horizon_minutes,
        max_horizon_hours=max_horizon_hours,
        min_history_snapshots=min_history_snapshots,
        decision_budget_minutes=decision_budget_minutes,
    )
    nonbinary = random_point_in_time_samples(
        nonbinary_raw,
        n_events=nonbinary_count,
        seed=seed + 17,
        min_horizon_minutes=min_horizon_minutes,
        max_horizon_hours=max_horizon_hours,
        min_history_snapshots=min_history_snapshots,
        decision_budget_minutes=decision_budget_minutes,
    )
    tagged = [*(TaggedSample("topvol_binary", sample) for sample in topvol)]
    tagged.extend(TaggedSample("nonbinary_component", sample) for sample in nonbinary)
    shortfall = max(0, total - len(tagged))
    if shortfall and len(topvol) < int(topvol_count or 0):
        extra = random_point_in_time_samples(
            nonbinary_raw,
            n_events=shortfall,
            seed=seed + 31,
            min_horizon_minutes=min_horizon_minutes,
            max_horizon_hours=max_horizon_hours,
            min_history_snapshots=min_history_snapshots,
            decision_budget_minutes=decision_budget_minutes,
        )
        tagged.extend(TaggedSample("nonbinary_component", sample) for sample in extra)
    elif shortfall:
        extra = random_point_in_time_samples(
            topvol_raw,
            n_events=shortfall,
            seed=seed + 43,
            min_horizon_minutes=min_horizon_minutes,
            max_horizon_hours=max_horizon_hours,
            min_history_snapshots=min_history_snapshots,
            decision_budget_minutes=decision_budget_minutes,
        )
        tagged.extend(TaggedSample("topvol_binary", sample) for sample in extra)
    import random

    rng = random.Random(seed + 101)
    rng.shuffle(tagged)
    return tagged[:total]


def _forecast_one(
    idx: int,
    tagged: TaggedSample,
    cfg: ForecastConfig,
    *,
    forecast_mode: ForecastMode,
    include_context: bool,
    force_cheap: bool,
    pit_external_evidence: bool,
    pit_allow_network: bool,
    pit_nonstrict_collected_at: bool,
    backtest_internet: bool,
    per_contract_timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline_at = started + max(1.0, per_contract_timeout_seconds)
    sample = tagged.sample
    packet = build_feature_packet(sample.event, sample.market_info, price_trajectory=sample.snapshots)
    market_p = clamp_prob(packet.quote.market_mid)
    stat = forecast_stat_routed(packet, cfg.stat, include_context=include_context)
    stat_p = clamp_prob(stat.probabilities.get("YES", stat.calibrated_probability))
    external_evidence = _external_evidence(
        packet,
        sample,
        cfg,
        pit_external_evidence=pit_external_evidence,
        pit_allow_network=pit_allow_network,
        pit_nonstrict_collected_at=pit_nonstrict_collected_at,
        backtest_internet=backtest_internet,
        use_gpt=forecast_mode == "gpt",
        deadline_at=deadline_at,
    )
    api_logs = _evidence_api_logs(external_evidence)
    error = None
    model_source = forecast_mode
    if forecast_mode == "market":
        model_p = market_p
        audit: dict[str, Any] = {"mode": "market"}
    elif forecast_mode == "stat":
        model_p = stat_p
        audit = {"mode": "stat", "stat": stat.to_dict()}
    elif forecast_mode in {"dryrun", "gpt"}:
        try:
            worker_cfg = copy.deepcopy(cfg)
            decision = forecast_event(
                sample.event,
                sample.market_info,
                price_trajectory=sample.snapshots,
                external_evidence=external_evidence,
                dry_run=(forecast_mode == "dryrun"),
                config=worker_cfg,
                include_context=include_context,
                force_cheap=force_cheap,
                deadline_at=deadline_at,
            )
            model_p = clamp_prob(decision.probabilities.get("YES", stat_p))
            model_source = decision.source
            audit = decision.audit_summary
            api_logs.extend(audit.get("api_logs") or [])
        except Exception as exc:  # noqa: BLE001 - fallback keeps the run complete.
            model_p = stat_p
            audit = {"mode": forecast_mode, "fallback": "stat", "error": f"{type(exc).__name__}:{exc}"}
            error = audit["error"]
            model_source = "stat_fallback_after_error"
    else:
        raise ValueError(f"unknown forecast mode: {forecast_mode}")

    outcome = int(sample.outcome)
    model_brier = (model_p - outcome) ** 2
    market_brier = (market_p - outcome) ** 2
    return {
        "index": idx,
        "source_dataset": tagged.source_dataset,
        "market_ticker": packet.market_ticker,
        "event_ticker": packet.event_ticker,
        "title": packet.title,
        "subtitle": packet.subtitle,
        "category": packet.category,
        "as_of": packet.as_of,
        "close_time": packet.close_time,
        "horizon_hours": packet.horizon_hours,
        "n_snapshots": len(sample.snapshots),
        "outcome": outcome,
        "market_p_yes": market_p,
        "stat_p_yes": stat_p,
        "model_p_yes": model_p,
        "market_brier": market_brier,
        "model_brier": model_brier,
        "brier_improvement_vs_market": market_brier - model_brier,
        "spread": packet.quote.spread,
        "volume": packet.quote.volume,
        "open_interest": packet.quote.open_interest,
        "forecast_mode": forecast_mode,
        "model_source": model_source,
        "api_call_count": len(api_logs),
        "estimated_cost_usd": _api_cost(api_logs),
        "api_logs": api_logs,
        "external_evidence_count": len(external_evidence),
        "external_evidence_sources": dict(Counter(str(item.get("source") or "unknown") for item in external_evidence)),
        "external_evidence_status": _external_evidence_status(external_evidence),
        "external_evidence_errors": _external_evidence_errors(external_evidence),
        "grounded_research_status": _grounded_research_status(external_evidence),
        "error": error,
        "elapsed_seconds": time.monotonic() - started,
        "deadline_seconds": per_contract_timeout_seconds,
        "within_deadline": time.monotonic() - started <= per_contract_timeout_seconds,
        "audit": _compact_audit(audit),
    }


def _external_evidence(
    packet: FeaturePacket,
    sample: BacktestSample,
    cfg: ForecastConfig,
    *,
    pit_external_evidence: bool,
    pit_allow_network: bool,
    pit_nonstrict_collected_at: bool,
    backtest_internet: bool,
    use_gpt: bool,
    deadline_at: float | None,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if pit_external_evidence:
        evidence.extend(gather_pit_external_evidence(
            packet,
            cfg,
            enabled=True,
            allow_network=pit_allow_network,
            strict_collected_at=not pit_nonstrict_collected_at,
        ))
    if backtest_internet and use_gpt:
        event = dict(sample.event)
        event["as_of"] = packet.as_of
        arena_packet = build_arena_packet(event, include_historical_analogs=False)
        evidence.extend(gather_grounded_research_evidence(
            arena_packet,
            cfg,
            enabled=True,
            deadline_at=deadline_at,
            existing_evidence=evidence,
        ))
    return evidence


def _summarize_rows(rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    model_preds = [float(row["model_p_yes"]) for row in rows]
    market_preds = [float(row["market_p_yes"]) for row in rows]
    stat_preds = [float(row["stat_p_yes"]) for row in rows]
    outcomes = [int(row["outcome"]) for row in rows]
    summary = {
        "n": len(rows),
        "errors": len(errors) + sum(1 for row in rows if row.get("error")),
        "model_brier": brier(model_preds, outcomes),
        "market_brier": brier(market_preds, outcomes),
        "stat_brier": brier(stat_preds, outcomes),
        "model_improvement_vs_market": brier(market_preds, outcomes) - brier(model_preds, outcomes),
        "stat_improvement_vs_market": brier(market_preds, outcomes) - brier(stat_preds, outcomes),
        "model_ece": ece(model_preds, outcomes),
        "market_ece": ece(market_preds, outcomes),
        "estimated_cost_usd": sum(float(row.get("estimated_cost_usd") or 0.0) for row in rows),
        "api_call_count": sum(int(row.get("api_call_count") or 0) for row in rows),
        "by_source_dataset": _segment_metrics(rows, "source_dataset"),
        "by_category": _segment_metrics(rows, "category"),
        "by_horizon_bucket": _segment_metrics(rows, "horizon_bucket"),
        "by_spread_bucket": _segment_metrics(rows, "spread_bucket"),
    }
    return summary


def _segment_metrics(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if key == "horizon_bucket":
            label = _horizon_bucket(row.get("horizon_hours"))
        elif key == "spread_bucket":
            label = _spread_bucket(row.get("spread"))
        else:
            label = str(row.get(key) or "unknown")
        groups[label].append(row)
    out = []
    for label, items in sorted(groups.items(), key=lambda item: -len(item[1])):
        model = [float(row["model_p_yes"]) for row in items]
        market = [float(row["market_p_yes"]) for row in items]
        stat = [float(row["stat_p_yes"]) for row in items]
        outcomes = [int(row["outcome"]) for row in items]
        out.append({
            "segment": label,
            "n": len(items),
            "model_brier": brier(model, outcomes),
            "market_brier": brier(market, outcomes),
            "stat_brier": brier(stat, outcomes),
            "model_improvement_vs_market": brier(market, outcomes) - brier(model, outcomes),
            "avg_market_p_yes": sum(market) / len(market),
            "avg_model_p_yes": sum(model) / len(model),
            "yes_rate": sum(outcomes) / len(outcomes),
        })
    return out


def _horizon_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    hours = float(value)
    if hours < 0.5:
        return "<30m"
    if hours < 2:
        return "30m-2h"
    if hours < 12:
        return "2h-12h"
    if hours < 48:
        return "12h-48h"
    return "48h+"


def _spread_bucket(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "unknown"
    spread = float(value)
    if spread <= 0.03:
        return "<=3pp"
    if spread <= 0.08:
        return "3-8pp"
    if spread <= 0.15:
        return "8-15pp"
    if spread <= 0.30:
        return "15-30pp"
    return ">30pp"


def _api_cost(api_logs: list[dict[str, Any]]) -> float:
    return sum(float(log.get("estimated_cost_usd") or 0.0) for log in api_logs if isinstance(log, dict))


def _evidence_api_logs(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        for key in ("api_log", "lseg_query_api_log"):
            value = item.get(key)
            if isinstance(value, dict):
                logs.append(value)
    return logs


def _grounded_research_status(evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in evidence:
        if item.get("source") != "gemini_native_search_grounded_research":
            continue
        audit = item.get("source_date_audit") if isinstance(item.get("source_date_audit"), dict) else {}
        return {
            "error": item.get("error"),
            "pit_mode": item.get("pit_mode"),
            "cache_hit": item.get("cache_hit"),
            "has_api_log": isinstance(item.get("api_log"), dict),
            "accepted_source_count": audit.get("accepted_source_count"),
            "discarded_source_count": audit.get("discarded_source_count"),
        }
    return None


def _external_evidence_errors(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if item.get("error") or item.get("errors"):
            out.append({
                key: value
                for key, value in {
                    "source": item.get("source"),
                    "error": item.get("error"),
                    "errors": item.get("errors"),
                    "pit_mode": item.get("pit_mode"),
                    "claim": item.get("claim"),
                }.items()
                if value is not None
            })
    return out[:12]


def _external_evidence_status(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        audit = item.get("source_date_audit") if isinstance(item.get("source_date_audit"), dict) else {}
        out.append({
            key: value
            for key, value in {
                "source": item.get("source"),
                "error": item.get("error"),
                "pit_mode": item.get("pit_mode"),
                "has_api_log": isinstance(item.get("api_log"), dict),
                "accepted_source_count": audit.get("accepted_source_count"),
                "discarded_source_count": audit.get("discarded_source_count"),
            }.items()
            if value is not None
        })
    return out[:12]


def _compact_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: audit.get(key)
        for key in (
            "dry_run",
            "gates",
            "planned_models",
            "errors",
            "context_evidence_count",
            "context_sources",
            "deadline_remaining_seconds",
        )
        if audit.get(key) is not None
    }


def _default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("dhruv_GPT_forecasting/logs/runs") / f"{stamp}_kalshi_parallel_backtest"


def _write_run(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(result["run_config"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "errors.json").write_text(
        json.dumps(result["errors"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "report.md").write_text(_report(result["run_config"], result["summary"]), encoding="utf-8")


def _report(run_config: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# Kalshi Parallel Random-PIT Backtest",
        "",
        f"- Forecast mode: `{run_config['forecast_mode']}`",
        f"- Seed: `{run_config['seed']}`",
        f"- Workers: `{run_config['max_workers']}`",
        f"- Samples: `{summary['n']}`",
        f"- Estimated cost: `${summary['estimated_cost_usd']:.6f}`",
        "",
        "## Brier",
        "",
        "| Lane | Brier | Improvement vs market |",
        "| --- | ---: | ---: |",
        f"| Model | {summary['model_brier']:.6f} | {summary['model_improvement_vs_market']:.6f} |",
        f"| Market | {summary['market_brier']:.6f} | 0.000000 |",
        f"| Stat | {summary['stat_brier']:.6f} | {summary['stat_improvement_vs_market']:.6f} |",
        "",
        "## Source Dataset",
        "",
        "| Segment | N | Model Brier | Market Brier | Improvement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["by_source_dataset"]:
        lines.append(
            f"| {row['segment']} | {row['n']} | {row['model_brier']:.6f} | "
            f"{row['market_brier']:.6f} | {row['model_improvement_vs_market']:.6f} |"
        )
    lines.extend([
        "",
        "## Category",
        "",
        "| Segment | N | Model Brier | Market Brier | Improvement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["by_category"]:
        lines.append(
            f"| {row['segment']} | {row['n']} | {row['model_brier']:.6f} | "
            f"{row['market_brier']:.6f} | {row['model_improvement_vs_market']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--topvol-count", type=int)
    parser.add_argument("--nonbinary-count", type=int)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--candle-stride-minutes", type=int, default=1)
    parser.add_argument("--min-history-snapshots", type=int, default=5)
    parser.add_argument("--min-horizon-minutes", type=float, default=5.0)
    parser.add_argument("--max-horizon-hours", type=float)
    parser.add_argument("--decision-budget-minutes", type=float, default=5.0)
    parser.add_argument("--forecast-mode", choices=["market", "stat", "dryrun", "gpt"], default="gpt")
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--respect-gates", action="store_true")
    parser.add_argument("--with-supervisor", action="store_true")
    parser.add_argument("--pit-external-evidence", action="store_true")
    parser.add_argument("--pit-allow-network", action="store_true")
    parser.add_argument("--pit-nonstrict-collected-at", action="store_true")
    parser.add_argument("--backtest-internet", action="store_true")
    parser.add_argument(
        "--per-contract-timeout-seconds",
        type=float,
        default=480.0,
        help="Hard per-contract budget. Default 480s leaves two minutes for a downstream ensemble in a 10-minute window.",
    )
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    result = run_parallel_kalshi_backtest(
        total=args.total,
        topvol_count=args.topvol_count,
        nonbinary_count=args.nonbinary_count,
        seed=args.seed,
        candle_stride_minutes=args.candle_stride_minutes,
        min_history_snapshots=args.min_history_snapshots,
        min_horizon_minutes=args.min_horizon_minutes,
        max_horizon_hours=args.max_horizon_hours,
        decision_budget_minutes=args.decision_budget_minutes,
        forecast_mode=args.forecast_mode,
        max_workers=args.max_workers,
        include_context=not args.no_context,
        force_cheap=not args.respect_gates,
        with_supervisor=args.with_supervisor,
        pit_external_evidence=args.pit_external_evidence,
        pit_allow_network=args.pit_allow_network,
        pit_nonstrict_collected_at=args.pit_nonstrict_collected_at,
        backtest_internet=args.backtest_internet,
        per_contract_timeout_seconds=args.per_contract_timeout_seconds,
        output_dir=args.output_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps({
        "output_dir": result["output_dir"],
        "summary": result["summary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
