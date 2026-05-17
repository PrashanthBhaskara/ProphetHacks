"""Resolved-market statistical model experiments.

This module is intentionally offline-only. It scores candidate deterministic
forecast lanes before we spend GPT credits on prompt variants.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from .backtest import brier, ece, load_samples
from .config import StatConfig
from .context import build_related_context_evidence
from .data_loaders import BacktestSample
from .features import build_feature_packet, parse_dt
from .schemas import FeaturePacket, clamp_prob
from .stat_lane import forecast_stat
from .stat_router import model_registry as routed_model_registry


Predictor = Callable[[FeaturePacket], float]


@dataclass(frozen=True)
class ExperimentResult:
    source: str
    model: str
    n: int
    brier: float
    log_loss: float
    ece: float
    pit_l1: float
    pit_histogram: list[int]
    improvement_vs_market_brier: float
    category_metrics: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "n": self.n,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "pit_l1": self.pit_l1,
            "pit_histogram": self.pit_histogram,
            "improvement_vs_market_brier": self.improvement_vs_market_brier,
            "category_metrics": self.category_metrics,
        }


def run_experiments(
    source: str,
    *,
    limit: int | None = None,
    include_context: bool = False,
    horizon_hours: float | None = None,
) -> list[ExperimentResult]:
    samples = load_samples(source, limit)
    if horizon_hours is not None:
        samples = point_in_time_samples(samples, horizon_hours=horizon_hours)
    packets = [_packet_for_sample(sample, include_context=include_context) for sample in samples]
    outcomes = [sample.outcome for sample in samples]
    market_preds = [packet.quote.market_mid for packet in packets]
    market_brier = brier(market_preds, outcomes)
    results: list[ExperimentResult] = []
    for name, predictor in model_registry(include_context=include_context).items():
        preds = [predictor(packet) for packet in packets]
        results.append(_score(_source_label(source, horizon_hours), name, packets, preds, outcomes, market_brier))
    return sorted(results, key=lambda result: (result.brier, result.ece))


def run_holdout_experiments(
    source: str,
    *,
    limit: int | None = None,
    include_context: bool = False,
    train_fraction: float = 0.70,
    horizon_hours: float | None = None,
) -> list[ExperimentResult]:
    samples = load_samples(source, limit)
    if horizon_hours is not None:
        samples = point_in_time_samples(samples, horizon_hours=horizon_hours)
    packets = [_packet_for_sample(sample, include_context=include_context) for sample in samples]
    outcomes = [sample.outcome for sample in samples]
    split = max(1, min(len(samples) - 1, int(len(samples) * train_fraction)))
    train_packets = packets[:split]
    test_packets = packets[split:]
    train_outcomes = outcomes[:split]
    test_outcomes = outcomes[split:]
    test_market = [packet.quote.market_mid for packet in test_packets]
    market_brier = brier(test_market, test_outcomes)
    results: list[ExperimentResult] = []
    for name, predictor in model_registry(include_context=include_context).items():
        train_preds = [predictor(packet) for packet in train_packets]
        test_preds = [predictor(packet) for packet in test_packets]
        label = _source_label(source, horizon_hours)
        results.append(_score(label, f"{name}:raw_holdout", test_packets, test_preds, test_outcomes, market_brier))

        blend_weight = _fit_market_blend_weight(
            train_preds,
            [packet.quote.market_mid for packet in train_packets],
            train_outcomes,
        )
        blended = [
            _blend(market, pred, blend_weight)
            for market, pred in zip(test_market, test_preds)
        ]
        results.append(_score(
            label,
            f"{name}:market_blend_w={blend_weight:.2f}",
            test_packets,
            blended,
            test_outcomes,
            market_brier,
        ))

        platt = _fit_platt(train_preds, train_outcomes)
        platt_preds = [_apply_platt(pred, platt) for pred in test_preds]
        results.append(_score(
            label,
            f"{name}:platt_a={platt[0]:.2f}_b={platt[1]:.2f}",
            test_packets,
            platt_preds,
            test_outcomes,
            market_brier,
        ))
    return sorted(results, key=lambda result: (result.brier, result.ece))


def point_in_time_samples(samples: list[BacktestSample], *, horizon_hours: float) -> list[BacktestSample]:
    """Select the last snapshot at least `horizon_hours` before close."""
    out: list[BacktestSample] = []
    for sample in samples:
        close_dt = parse_dt(sample.event.get("close_time") or sample.market_info.get("close_time"))
        if close_dt is None:
            continue
        cutoff_ts = close_dt.timestamp() - horizon_hours * 3600.0
        selected_idx = None
        ordered = sorted(sample.snapshots, key=lambda snap: _snapshot_time_key(snap))
        for idx, snap in enumerate(ordered):
            snap_dt = parse_dt(_snapshot_time_value(snap))
            if snap_dt is None:
                continue
            if snap_dt.timestamp() <= cutoff_ts:
                selected_idx = idx
            else:
                break
        if selected_idx is None:
            continue
        truncated = ordered[: selected_idx + 1]
        market_info = dict(sample.market_info)
        market_info.update(truncated[-1])
        market_info["snapshots"] = truncated
        out.append(BacktestSample(
            event=sample.event,
            market_info=market_info,
            snapshots=truncated,
            outcome=sample.outcome,
        ))
    return out


def random_point_in_time_samples(
    samples: list[BacktestSample],
    *,
    n_events: int | None = None,
    seed: int = 20260517,
    min_horizon_minutes: float = 5.0,
    max_horizon_hours: float | None = None,
    min_history_snapshots: int = 5,
    decision_budget_minutes: float = 5.0,
) -> list[BacktestSample]:
    """Choose one random request timestamp per resolved market.

    The returned sample is truncated at the request timestamp, so the market
    baseline and any statistical/LLM lane only see information available when
    the benchmark asks for a probability. `decision_deadline_time` is metadata
    for the five-minute response budget; it never permits future market data.
    """
    rng = random.Random(seed)
    rows: list[BacktestSample] = []
    for sample in samples:
        close_dt = parse_dt(sample.event.get("close_time") or sample.market_info.get("close_time"))
        if close_dt is None:
            continue
        ordered = sorted(sample.snapshots, key=lambda snap: _snapshot_time_key(snap))
        eligible: list[int] = []
        for idx, snap in enumerate(ordered):
            if idx + 1 < min_history_snapshots:
                continue
            snap_dt = parse_dt(_snapshot_time_value(snap))
            if snap_dt is None:
                continue
            horizon_minutes = (close_dt - snap_dt).total_seconds() / 60.0
            if horizon_minutes < min_horizon_minutes:
                continue
            if max_horizon_hours is not None and horizon_minutes > max_horizon_hours * 60.0:
                continue
            eligible.append(idx)
        if not eligible:
            continue
        selected_idx = rng.choice(eligible)
        request_dt = parse_dt(_snapshot_time_value(ordered[selected_idx]))
        if request_dt is None:
            continue
        truncated = ordered[: selected_idx + 1]
        market_info = dict(sample.market_info)
        market_info.update(truncated[-1])
        market_info["snapshots"] = truncated
        market_info["forecast_request_time"] = request_dt.isoformat()
        market_info["decision_budget_minutes"] = decision_budget_minutes
        market_info["decision_deadline_time"] = (
            request_dt + timedelta(minutes=decision_budget_minutes)
        ).isoformat()
        market_info["request_time_policy"] = "random_uniform_over_eligible_market_snapshots"
        rows.append(BacktestSample(
            event=sample.event,
            market_info=market_info,
            snapshots=truncated,
            outcome=sample.outcome,
        ))
    rng.shuffle(rows)
    if n_events is not None:
        rows = rows[:n_events]
    return rows


def _source_label(source: str, horizon_hours: float | None) -> str:
    if horizon_hours is None:
        return source
    clean = int(horizon_hours) if float(horizon_hours).is_integer() else horizon_hours
    return f"{source}_pit_{clean}h"


def _snapshot_time_value(snapshot: dict[str, Any]) -> str | None:
    return snapshot.get("t") or snapshot.get("snapshot_time") or snapshot.get("end_period_time")


def _snapshot_time_key(snapshot: dict[str, Any]) -> str:
    return _snapshot_time_value(snapshot) or ""


def model_registry(*, include_context: bool) -> dict[str, Predictor]:
    return routed_model_registry(include_context=include_context)


def _packet_for_sample(sample: BacktestSample, *, include_context: bool) -> FeaturePacket:
    packet = build_feature_packet(sample.event, sample.market_info, price_trajectory=sample.snapshots)
    if include_context:
        context = build_related_context_evidence(packet)
        packet.evidence_digest = context
        packet.features["related_context_count"] = len(context)
    return packet


def _stat_with_caps(packet: FeaturePacket, *, default_dev: float, max_dev: float) -> float:
    cfg = StatConfig(default_market_deviation=default_dev, max_market_deviation=max_dev)
    stat = forecast_stat(packet, cfg)
    return stat.probabilities.get("YES", stat.calibrated_probability)


def _spread_shrunk_stat(packet: FeaturePacket) -> float:
    market = packet.quote.market_mid
    stat = forecast_stat(packet, StatConfig(default_market_deviation=0.03, max_market_deviation=0.06))
    p_stat = stat.probabilities.get("YES", stat.calibrated_probability)
    spread = float(packet.quote.spread or 0.10)
    n = int(packet.features.get("n_snapshots") or 0)
    quality = max(0.0, min(1.0, 1.0 - spread / 0.30))
    depth = max(0.25, min(1.0, n / 20.0))
    shrink = 0.65 * quality * depth
    return clamp_prob(market + shrink * (p_stat - market))


def _momentum(packet: FeaturePacket, *, scale: float, cap: float) -> float:
    points = packet.price_trajectory
    if len(points) < 2:
        return packet.quote.market_mid
    market = packet.quote.market_mid
    move = float(points[-1]["market_mid"] - points[0]["market_mid"])
    return clamp_prob(market + max(-cap, min(cap, scale * move)))


def _recent_momentum(packet: FeaturePacket, *, scale: float, cap: float) -> float:
    points = packet.price_trajectory
    if len(points) < 5:
        return packet.quote.market_mid
    market = packet.quote.market_mid
    move = float(points[-1]["market_mid"] - points[-5]["market_mid"])
    return clamp_prob(market + max(-cap, min(cap, scale * move)))


def _context_normalized(packet: FeaturePacket, *, weight: float) -> float:
    market = packet.quote.market_mid
    context_prob = _context_target_prob(packet)
    if context_prob is None:
        return market
    spread = float(packet.quote.spread or 0.10)
    quality = max(0.0, min(1.0, 1.0 - spread / 0.30))
    return clamp_prob(_blend(market, context_prob, weight * quality))


def _context_target_prob(packet: FeaturePacket) -> float | None:
    for item in packet.evidence_digest:
        derived = item.get("derived") or {}
        target_mid = derived.get("target_yes_market_mid")
        sum_mid = derived.get("sum_yes_market_mid")
        if target_mid is None or sum_mid is None:
            continue
        sum_mid = float(sum_mid)
        if 0.25 <= sum_mid <= 4.0:
            return clamp_prob(float(target_mid) / sum_mid)
    return None


def _blend(left: float, right: float, weight: float) -> float:
    weight = max(0.0, min(1.0, weight))
    return clamp_prob((1.0 - weight) * left + weight * right)


def _score(
    source: str,
    model: str,
    packets: list[FeaturePacket],
    preds: list[float],
    outcomes: list[int],
    market_brier: float,
) -> ExperimentResult:
    by_category: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for packet, pred, outcome in zip(packets, preds, outcomes):
        by_category[packet.category].append((pred, outcome))
    category_metrics = []
    for category, pairs in sorted(by_category.items(), key=lambda item: -len(item[1])):
        ps = [p for p, _ in pairs]
        ys = [y for _, y in pairs]
        category_metrics.append({
            "category": category,
            "n": len(pairs),
            "brier": brier(ps, ys),
            "ece": ece(ps, ys),
            "log_loss": log_loss(ps, ys),
        })
    current_brier = brier(preds, outcomes)
    return ExperimentResult(
        source=source,
        model=model,
        n=len(preds),
        brier=current_brier,
        log_loss=log_loss(preds, outcomes),
        ece=ece(preds, outcomes),
        pit_l1=pit_l1(preds, outcomes),
        pit_histogram=pit_histogram(preds, outcomes),
        improvement_vs_market_brier=market_brier - current_brier,
        category_metrics=category_metrics,
    )


def _fit_market_blend_weight(
    base_preds: list[float],
    market_preds: list[float],
    outcomes: list[int],
) -> float:
    best_weight = 0.0
    best_brier = float("inf")
    for i in range(0, 21):
        weight = i / 20.0
        preds = [_blend(market, base, weight) for market, base in zip(market_preds, base_preds)]
        score = brier(preds, outcomes)
        if score < best_brier:
            best_brier = score
            best_weight = weight
    return best_weight


def _fit_platt(preds: list[float], outcomes: list[int]) -> tuple[float, float]:
    """Fit sigmoid(a + b * logit(p)) with small L2 regularization."""
    a = 0.0
    b = 1.0
    xs = [math.log(clamp_prob(p, lo=1e-6, hi=1 - 1e-6) / (1.0 - clamp_prob(p, lo=1e-6, hi=1 - 1e-6))) for p in preds]
    ys = [float(y) for y in outcomes]
    if not xs:
        return a, b
    reg = 1e-3
    for _ in range(30):
        g0 = reg * a
        g1 = reg * (b - 1.0)
        h00 = reg
        h01 = 0.0
        h11 = reg
        for x, y in zip(xs, ys):
            z = max(-30.0, min(30.0, a + b * x))
            p = 1.0 / (1.0 + math.exp(-z))
            w = p * (1.0 - p)
            g0 += p - y
            g1 += (p - y) * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        step_a = (h11 * g0 - h01 * g1) / det
        step_b = (-h01 * g0 + h00 * g1) / det
        a -= step_a
        b -= step_b
        if abs(step_a) + abs(step_b) < 1e-6:
            break
    return a, b


def _apply_platt(pred: float, params: tuple[float, float]) -> float:
    a, b = params
    p = clamp_prob(pred, lo=1e-6, hi=1 - 1e-6)
    z = max(-30.0, min(30.0, a + b * math.log(p / (1.0 - p))))
    return clamp_prob(1.0 / (1.0 + math.exp(-z)))


def log_loss(preds: list[float], outcomes: list[int]) -> float:
    if not preds:
        return float("nan")
    total = 0.0
    for p, y in zip(preds, outcomes):
        p = clamp_prob(p, lo=1e-6, hi=1 - 1e-6)
        total -= math.log(p if y else 1.0 - p)
    return total / len(preds)


def pit_histogram(preds: list[float], outcomes: list[int], n_bins: int = 10) -> list[int]:
    """Deterministic randomized-PIT midpoint histogram for binary forecasts."""
    hist = [0 for _ in range(n_bins)]
    for p, y in zip(preds, outcomes):
        u = 1.0 - p / 2.0 if y else (1.0 - p) / 2.0
        hist[min(int(u * n_bins), n_bins - 1)] += 1
    return hist


def pit_l1(preds: list[float], outcomes: list[int], n_bins: int = 10) -> float:
    hist = pit_histogram(preds, outcomes, n_bins=n_bins)
    if not hist or not preds:
        return float("nan")
    expected = len(preds) / n_bins
    return sum(abs(count - expected) for count in hist) / (2.0 * len(preds))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["live_clean", "eval_pack", "topvol", "nonbinary", "unified"], default="topvol")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--horizon-hours", type=float)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.holdout:
        results = run_holdout_experiments(
            args.source,
            limit=args.limit,
            include_context=args.context,
            train_fraction=args.train_fraction,
            horizon_hours=args.horizon_hours,
        )
    else:
        results = run_experiments(
            args.source,
            limit=args.limit,
            include_context=args.context,
            horizon_hours=args.horizon_hours,
        )
    payload = [result.to_dict() for result in results]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(payload[: args.top], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
