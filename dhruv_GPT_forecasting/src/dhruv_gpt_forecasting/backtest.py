"""Offline baselines and dry-run backtests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .config import load_config
from .data_loaders import (
    BacktestSample,
    load_eval_pack,
    load_nonbinary_component_samples,
    load_topvol_samples,
    load_unified_binary_samples,
)
from .features import build_feature_packet
from .forecaster import forecast_event
from .risk import decide_trade
from .stat_lane import forecast_stat


def brier(preds: list[float], outcomes: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(preds, outcomes)) / len(preds) if preds else float("nan")


def ece(preds: list[float], outcomes: list[int], n_bins: int = 10) -> float:
    if not preds:
        return float("nan")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(preds, outcomes):
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    total = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        avg_y = sum(y for _, y in bucket) / len(bucket)
        total += len(bucket) / len(preds) * abs(avg_p - avg_y)
    return total


def load_samples(source: str, limit: int | None) -> list[BacktestSample]:
    if source == "live_clean":
        return load_eval_pack(limit=limit)
    if source == "eval_pack":
        path = Path(__file__).resolve().parents[3] / "prophet-hacks-handoff" / "prep" / "data" / "eval_pack.jsonl"
        return load_eval_pack(path=path, limit=limit)
    if source == "topvol":
        return load_topvol_samples(limit=limit)
    if source == "nonbinary":
        return load_nonbinary_component_samples(limit=limit)
    if source == "unified":
        return load_unified_binary_samples(limit=limit)
    raise ValueError(f"unknown source: {source}")


def run_backtest(samples: list[BacktestSample], mode: str, *, include_context: bool = True) -> dict:
    cfg = load_config()
    preds: list[float] = []
    outcomes: list[int] = []
    by_category: dict[str, list[tuple[float, int]]] = defaultdict(list)
    trade_reasons = Counter()
    trade_sides = Counter()
    context_sources = Counter()
    context_counts: list[int] = []
    for sample in samples:
        packet = build_feature_packet(sample.event, sample.market_info, price_trajectory=sample.snapshots)
        if mode == "market":
            p_yes = packet.quote.market_mid
            decision = None
        elif mode == "stat":
            stat = forecast_stat(packet, cfg.stat)
            p_yes = stat.probabilities.get("YES", stat.calibrated_probability)
            decision = decide_trade(packet, stat.probabilities, stat.confidence, stat.uncertainty, cfg.risk)
        elif mode == "dryrun":
            final = forecast_event(
                sample.event,
                sample.market_info,
                price_trajectory=sample.snapshots,
                dry_run=True,
                config=cfg,
                include_context=include_context,
            )
            p_yes = final.probabilities.get("YES", 0.5)
            decision = final.trade_decision
            context_counts.append(int(final.audit_summary.get("context_evidence_count") or 0))
            for source in final.audit_summary.get("context_sources") or []:
                context_sources[source] += 1
        else:
            raise ValueError(f"unknown mode: {mode}")
        preds.append(float(p_yes))
        outcomes.append(sample.outcome)
        by_category[packet.category].append((float(p_yes), sample.outcome))
        if decision is not None:
            trade_reasons[decision.reason] += 1
            trade_sides[decision.side] += 1
    rows = []
    for category, pairs in sorted(by_category.items(), key=lambda item: -len(item[1])):
        ps = [p for p, _ in pairs]
        ys = [y for _, y in pairs]
        rows.append({
            "category": category,
            "n": len(pairs),
            "brier": brier(ps, ys),
            "ece": ece(ps, ys),
        })
    return {
        "n": len(samples),
        "mode": mode,
        "brier": brier(preds, outcomes),
        "ece": ece(preds, outcomes),
        "yes_rate": sum(outcomes) / len(outcomes) if outcomes else None,
        "category_metrics": rows,
        "trade_reasons": dict(trade_reasons),
        "trade_sides": dict(trade_sides),
        "avg_context_evidence": (
            sum(context_counts) / len(context_counts) if context_counts else None
        ),
        "context_sources": dict(context_sources),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["live_clean", "eval_pack", "topvol", "nonbinary", "unified"], default="live_clean")
    parser.add_argument("--mode", choices=["market", "stat", "dryrun"], default="market")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()
    samples = load_samples(args.source, args.limit)
    result = run_backtest(samples, args.mode, include_context=not args.no_context)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
