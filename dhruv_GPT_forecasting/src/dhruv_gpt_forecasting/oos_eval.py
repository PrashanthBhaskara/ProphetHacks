"""Chronological OOS evaluation and trade diagnostics.

This module scores deterministic forecast lanes over resolved point-in-time
samples. It is designed to answer two offline questions before spending GPT
credits:

1. Which market-anchored statistical variants survive chronological OOS?
2. Which contract segments create executable edge after spread/risk buffers?
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .backtest import brier, ece, load_samples
from .config import RiskConfig, load_config
from .context import build_related_context_evidence
from .data_loaders import (
    BacktestSample,
    load_nonbinary_component_samples,
    load_topvol_samples,
    load_unified_binary_samples,
)
from .evidence_replay import EvidenceReplayIndex, coverage_summary
from .experiments import (
    _apply_platt,
    _blend,
    _fit_market_blend_weight,
    _fit_platt,
    log_loss,
    model_registry,
    pit_l1,
    point_in_time_samples,
    random_point_in_time_samples,
)
from .features import build_feature_packet, parse_dt
from .gating import cheap_gate, supervisor_gate
from .risk import decide_trade
from .schemas import FeaturePacket, LaneForecast, StatForecast, clamp_prob
from .stat_lane import forecast_stat


PacketRow = tuple[BacktestSample, FeaturePacket, int]
Predictor = Callable[[FeaturePacket], float]


@dataclass(frozen=True)
class ModelScore:
    model: str
    n: int
    brier: float
    log_loss: float
    ece: float
    pit_l1: float
    improvement_vs_market_brier: float
    trade_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n": self.n,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "pit_l1": self.pit_l1,
            "improvement_vs_market_brier": self.improvement_vs_market_brier,
            "trade_summary": self.trade_summary,
        }


def run_oos_evaluation(
    *,
    source: str,
    horizon_hours: float,
    candle_stride_minutes: int = 1,
    train_fraction: float = 0.70,
    include_context: bool = False,
    limit: int | None = None,
    since_close: str | None = None,
    until_close: str | None = None,
    random_as_of: bool = False,
    random_seed: int = 20260517,
    min_horizon_minutes: float = 5.0,
    max_horizon_hours: float | None = None,
    min_history_snapshots: int = 5,
    decision_budget_minutes: float = 5.0,
    min_segment_n: int = 30,
    top_segments: int = 12,
    evidence_mode: str = "none",
    evidence_manifest_paths: list[Path] | None = None,
    evidence_max_records: int | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    evidence_index = (
        EvidenceReplayIndex.from_manifests(evidence_manifest_paths or [])
        if evidence_mode != "none" and evidence_manifest_paths
        else None
    )
    loader_limit = None if (since_close or until_close) else limit
    samples = _load_oos_samples(
        source,
        limit=loader_limit,
        horizon_hours=horizon_hours,
        candle_stride_minutes=candle_stride_minutes,
        random_as_of=random_as_of,
    )
    samples = _filter_by_close_time(samples, since_close=since_close, until_close=until_close)
    samples.sort(key=lambda sample: (
        _sample_close_key(sample),
        sample.event.get("market_ticker") or sample.market_info.get("ticker") or "",
    ))
    if limit is not None and loader_limit is None:
        samples = samples[:limit]
    rows = _packet_rows(samples, include_context=include_context)
    if len(rows) < 2:
        raise ValueError("not enough point-in-time samples for OOS evaluation")
    split = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
    train_rows = rows[:split]
    test_rows = rows[split:]
    if random_as_of:
        train_rows = _randomize_rows(
            train_rows,
            seed=random_seed,
            min_horizon_minutes=min_horizon_minutes,
            max_horizon_hours=max_horizon_hours,
            min_history_snapshots=min_history_snapshots,
            decision_budget_minutes=decision_budget_minutes,
        )
        test_rows = _randomize_rows(
            test_rows,
            seed=random_seed + 1,
            min_horizon_minutes=min_horizon_minutes,
            max_horizon_hours=max_horizon_hours,
            min_history_snapshots=min_history_snapshots,
            decision_budget_minutes=decision_budget_minutes,
        )
        rows = train_rows + test_rows

    if evidence_index is not None:
        _attach_replayed_evidence(
            train_rows,
            evidence_index,
            cfg,
            mode=evidence_mode,
            max_records=evidence_max_records,
        )
        _attach_replayed_evidence(
            test_rows,
            evidence_index,
            cfg,
            mode=evidence_mode,
            max_records=evidence_max_records,
        )
        rows = train_rows + test_rows

    train_packets = [row[1] for row in train_rows]
    test_packets = [row[1] for row in test_rows]
    train_outcomes = [row[2] for row in train_rows]
    test_outcomes = [row[2] for row in test_rows]
    test_market = [packet.quote.market_mid for packet in test_packets]
    test_market_brier = brier(test_market, test_outcomes)

    scores: list[ModelScore] = []
    prediction_book: dict[str, list[float]] = {}
    train_prediction_book: dict[str, list[float]] = {}
    test_prediction_book: dict[str, list[float]] = {}
    for name, predictor in model_registry(include_context=include_context).items():
        train_preds = [predictor(packet) for packet in train_packets]
        test_preds = [predictor(packet) for packet in test_packets]
        train_prediction_book[name] = [clamp_prob(pred) for pred in train_preds]
        test_prediction_book[name] = [clamp_prob(pred) for pred in test_preds]
        _add_score(
            scores,
            name,
            test_packets,
            test_preds,
            test_outcomes,
            test_market_brier,
            cfg.risk,
            prediction_book,
        )

        blend_weight = _fit_market_blend_weight(
            train_preds,
            [packet.quote.market_mid for packet in train_packets],
            train_outcomes,
        )
        blended = [
            _blend(market, pred, blend_weight)
            for market, pred in zip(test_market, test_preds)
        ]
        blend_name = f"{name}:market_blend_w={blend_weight:.2f}"
        train_blended = [
            _blend(packet.quote.market_mid, pred, blend_weight)
            for packet, pred in zip(train_packets, train_preds)
        ]
        train_prediction_book[blend_name] = [clamp_prob(pred) for pred in train_blended]
        test_prediction_book[blend_name] = [clamp_prob(pred) for pred in blended]
        _add_score(
            scores,
            blend_name,
            test_packets,
            blended,
            test_outcomes,
            test_market_brier,
            cfg.risk,
            prediction_book,
        )

        platt = _fit_platt(train_preds, train_outcomes)
        platt_preds = [_apply_platt(pred, platt) for pred in test_preds]
        platt_name = f"{name}:platt_a={platt[0]:.2f}_b={platt[1]:.2f}"
        train_platt = [_apply_platt(pred, platt) for pred in train_preds]
        train_prediction_book[platt_name] = [clamp_prob(pred) for pred in train_platt]
        test_prediction_book[platt_name] = [clamp_prob(pred) for pred in platt_preds]
        _add_score(
            scores,
            platt_name,
            test_packets,
            platt_preds,
            test_outcomes,
            test_market_brier,
            cfg.risk,
            prediction_book,
        )

    category_router = _fit_category_model_router(
        train_packets,
        train_outcomes,
        train_prediction_book,
        test_packets,
        test_prediction_book,
        min_segment_n=min_segment_n,
    )
    _add_score(
        scores,
        category_router["model"],
        test_packets,
        category_router["predictions"],
        test_outcomes,
        test_market_brier,
        cfg.risk,
        prediction_book,
    )

    scores.sort(key=lambda score: (score.brier, score.ece))
    best = scores[0]
    best_preds = prediction_book[best.model]
    gate_summary = _gate_shadow_summary(test_packets, cfg.stat)
    segments = _segment_report(
        test_packets,
        best_preds,
        test_market,
        test_outcomes,
        cfg.risk,
        min_segment_n=min_segment_n,
        top_segments=top_segments,
    )
    weekly = _period_metrics(test_packets, best_preds, test_market, test_outcomes)
    return {
        "source": source,
        "horizon_hours": horizon_hours,
        "random_as_of": random_as_of,
        "random_seed": random_seed if random_as_of else None,
        "min_horizon_minutes": min_horizon_minutes if random_as_of else None,
        "max_horizon_hours": max_horizon_hours if random_as_of else None,
        "min_history_snapshots": min_history_snapshots if random_as_of else None,
        "decision_budget_minutes": decision_budget_minutes if random_as_of else None,
        "candle_stride_minutes": candle_stride_minutes if source in {"topvol", "nonbinary", "unified"} else None,
        "include_context": include_context,
        "evidence_replay": _evidence_replay_result(
            evidence_index,
            evidence_mode=evidence_mode,
            evidence_manifest_paths=evidence_manifest_paths or [],
            train_rows=train_rows,
            test_rows=test_rows,
        ),
        "since_close": since_close,
        "until_close": until_close,
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "split_as_of": test_packets[0].as_of,
        "market_test_brier": test_market_brier,
        "best_model": best.to_dict(),
        "top_models": [score.to_dict() for score in scores[:10]],
        "stat_model_routing": category_router["summary"],
        "gate_shadow": gate_summary,
        "segment_report": segments,
        "weekly_metrics": weekly,
    }


def _load_oos_samples(
    source: str,
    *,
    limit: int | None,
    horizon_hours: float,
    candle_stride_minutes: int,
    random_as_of: bool,
) -> list[BacktestSample]:
    if source == "topvol":
        samples = load_topvol_samples(
            limit=limit,
            candle_stride_minutes=candle_stride_minutes,
            min_snapshots=2,
        )
    elif source == "nonbinary":
        samples = load_nonbinary_component_samples(
            limit=limit,
            candle_stride_minutes=candle_stride_minutes,
            min_snapshots=2,
        )
    elif source == "unified":
        samples = load_unified_binary_samples(
            limit=limit,
            candle_stride_minutes=candle_stride_minutes,
            min_snapshots=2,
        )
    else:
        samples = load_samples(source, limit)
    if random_as_of:
        return samples
    return point_in_time_samples(samples, horizon_hours=horizon_hours)


def _randomize_rows(
    rows: list[PacketRow],
    *,
    seed: int,
    min_horizon_minutes: float,
    max_horizon_hours: float | None,
    min_history_snapshots: int,
    decision_budget_minutes: float,
) -> list[PacketRow]:
    randomized = random_point_in_time_samples(
        [row[0] for row in rows],
        seed=seed,
        min_horizon_minutes=min_horizon_minutes,
        max_horizon_hours=max_horizon_hours,
        min_history_snapshots=min_history_snapshots,
        decision_budget_minutes=decision_budget_minutes,
    )
    out = [
        (sample, build_feature_packet(sample.event, sample.market_info, price_trajectory=sample.snapshots), sample.outcome)
        for sample in randomized
    ]
    out.sort(key=lambda row: (_dt_sort_key(row[1].as_of), row[1].market_ticker))
    return out


def _packet_rows(samples: list[BacktestSample], *, include_context: bool) -> list[PacketRow]:
    rows: list[PacketRow] = []
    for sample in samples:
        packet = build_feature_packet(sample.event, sample.market_info, price_trajectory=sample.snapshots)
        if include_context:
            context = build_related_context_evidence(packet)
            if context:
                packet.evidence_digest = context + packet.evidence_digest
                packet.features["related_context_count"] = len(context)
                packet.features["related_context_sources"] = sorted(
                    {str(item.get("source")) for item in context if item.get("source")}
                )
        rows.append((sample, packet, sample.outcome))
    rows.sort(key=lambda row: (_dt_sort_key(row[1].as_of), row[1].market_ticker))
    return rows


def _attach_replayed_evidence(
    rows: list[PacketRow],
    index: EvidenceReplayIndex,
    cfg: Any,
    *,
    mode: str,
    max_records: int | None,
) -> None:
    for _sample, packet, _outcome in rows:
        evidence = index.evidence_for_packet(packet, cfg, mode=mode, max_records=max_records)
        if evidence:
            packet.evidence_digest = evidence + packet.evidence_digest
            summary = evidence[0]
            packet.features["archive_replay_mode"] = mode
            packet.features["archive_replay_record_count"] = int(summary.get("record_count") or 0)
            packet.features["archive_replay_source_counts"] = dict(summary.get("source_counts") or {})
            packet.features["archive_replay_sentiment"] = summary.get("sentiment")
        else:
            packet.features["archive_replay_mode"] = mode
            packet.features["archive_replay_record_count"] = 0
            packet.features["archive_replay_source_counts"] = {}


def _evidence_replay_result(
    index: EvidenceReplayIndex | None,
    *,
    evidence_mode: str,
    evidence_manifest_paths: list[Path],
    train_rows: list[PacketRow],
    test_rows: list[PacketRow],
) -> dict[str, Any]:
    if index is None:
        return {
            "mode": "none",
            "manifest_paths": [str(path) for path in evidence_manifest_paths],
            "loaded_records": 0,
            "train": coverage_summary([], mode="none"),
            "test": coverage_summary([], mode="none"),
        }
    return {
        **index.stats.to_dict(),
        "mode": evidence_mode,
        "manifest_paths": [str(path) for path in evidence_manifest_paths],
        "train": coverage_summary(train_rows, mode=evidence_mode),
        "test": coverage_summary(test_rows, mode=evidence_mode),
    }


def _filter_by_close_time(
    samples: list[BacktestSample],
    *,
    since_close: str | None,
    until_close: str | None,
) -> list[BacktestSample]:
    if not since_close and not until_close:
        return samples
    since_dt = parse_dt(since_close) if since_close else None
    until_dt = parse_dt(until_close) if until_close else None
    out: list[BacktestSample] = []
    for sample in samples:
        close_dt = parse_dt(sample.event.get("close_time") or sample.market_info.get("close_time"))
        if close_dt is None:
            continue
        if since_dt is not None and close_dt < since_dt:
            continue
        if until_dt is not None and close_dt > until_dt:
            continue
        out.append(sample)
    return out


def _dt_sort_key(value: str | None) -> float:
    parsed = parse_dt(value)
    if parsed is None:
        return 0.0
    return parsed.timestamp()


def _sample_close_key(sample: BacktestSample) -> float:
    parsed = parse_dt(sample.event.get("close_time") or sample.market_info.get("close_time"))
    return parsed.timestamp() if parsed is not None else 0.0


def _add_score(
    scores: list[ModelScore],
    name: str,
    packets: list[FeaturePacket],
    preds: list[float],
    outcomes: list[int],
    market_brier: float,
    risk: RiskConfig,
    prediction_book: dict[str, list[float]],
) -> None:
    clean = [clamp_prob(pred) for pred in preds]
    prediction_book[name] = clean
    summary = _trade_summary(packets, clean, outcomes, risk)
    scores.append(ModelScore(
        model=name,
        n=len(clean),
        brier=brier(clean, outcomes),
        log_loss=log_loss(clean, outcomes),
        ece=ece(clean, outcomes),
        pit_l1=pit_l1(clean, outcomes),
        improvement_vs_market_brier=market_brier - brier(clean, outcomes),
        trade_summary=summary,
    ))


def _trade_summary(
    packets: list[FeaturePacket],
    preds: list[float],
    outcomes: list[int],
    risk: RiskConfig,
) -> dict[str, Any]:
    pnl = 0.0
    stake_total = 0.0
    trade_count = 0
    sides = Counter()
    reasons = Counter()
    max_equity = risk.starting_equity
    equity = risk.starting_equity
    peak = equity
    max_drawdown = 0.0
    unit_raw_edge_pnl = 0.0
    unit_raw_edge_stake = 0.0
    unit_raw_edge_count = 0
    for packet, pred, outcome in zip(packets, preds, outcomes):
        probs = {"YES": pred, "NO": 1.0 - pred}
        decision = decide_trade(packet, probs, confidence=0.60, uncertainty=0.40, cfg=risk)
        reasons[decision.reason] += 1
        sides[decision.side] += 1
        if decision.side != "NONE" and decision.price is not None and decision.stake > 0.0:
            trade_count += 1
            stake_total += decision.stake
            trade_pnl = _contract_pnl(decision.side, decision.price, decision.stake, outcome)
            pnl += trade_pnl
            equity += trade_pnl
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
            max_equity = max(max_equity, equity)
        raw_side, raw_price, raw_edge = _best_raw_edge(packet, pred)
        if raw_side != "NONE" and raw_price is not None and raw_edge >= 0.06:
            unit_raw_edge_count += 1
            unit_raw_edge_stake += 1.0
            unit_raw_edge_pnl += _contract_pnl(raw_side, raw_price, 1.0, outcome)
    return {
        "trades": trade_count,
        "stake": stake_total,
        "pnl": pnl,
        "roi_on_stake": pnl / stake_total if stake_total else 0.0,
        "max_drawdown_dollars": max_drawdown,
        "ending_equity": equity,
        "max_equity": max_equity,
        "sides": dict(sides),
        "reasons": dict(reasons),
        "raw_unit_edge_ge_6pp": {
            "trades": unit_raw_edge_count,
            "pnl": unit_raw_edge_pnl,
            "roi_on_stake": unit_raw_edge_pnl / unit_raw_edge_stake if unit_raw_edge_stake else 0.0,
        },
    }


def _fit_category_model_router(
    train_packets: list[FeaturePacket],
    train_outcomes: list[int],
    train_prediction_book: dict[str, list[float]],
    test_packets: list[FeaturePacket],
    test_prediction_book: dict[str, list[float]],
    *,
    min_segment_n: int,
) -> dict[str, Any]:
    """Choose the lowest-Brier statistical model per category on train only."""
    candidates = sorted(set(train_prediction_book) & set(test_prediction_book))
    if not candidates:
        fallback = [packet.quote.market_mid for packet in test_packets]
        return {
            "model": "category_model_router:no_candidates",
            "predictions": fallback,
            "summary": {"fallback_model": "market_mid", "routes": {}, "candidate_count": 0},
        }
    global_scores = {
        name: brier(train_prediction_book[name], train_outcomes)
        for name in candidates
    }
    fallback_model = min(global_scores, key=global_scores.get)
    routes: dict[str, dict[str, Any]] = {}
    train_categories = sorted({packet.category for packet in train_packets})
    for category in train_categories:
        idxs = [i for i, packet in enumerate(train_packets) if packet.category == category]
        if len(idxs) < min_segment_n:
            routes[category] = {
                "model": fallback_model,
                "train_n": len(idxs),
                "train_brier": global_scores[fallback_model],
                "selection": "global_fallback_insufficient_category_n",
            }
            continue
        category_outcomes = [train_outcomes[i] for i in idxs]
        scores = {
            name: brier([train_prediction_book[name][i] for i in idxs], category_outcomes)
            for name in candidates
        }
        best = min(scores, key=scores.get)
        leaderboard = sorted(scores.items(), key=lambda item: item[1])[:5]
        routes[category] = {
            "model": best,
            "train_n": len(idxs),
            "train_brier": scores[best],
            "selection": "category_best_train_brier",
            "top_train_models": [
                {"model": name, "train_brier": score}
                for name, score in leaderboard
            ],
        }
    predictions = []
    test_route_counts: Counter[str] = Counter()
    for i, packet in enumerate(test_packets):
        route = routes.get(packet.category)
        model = route["model"] if route else fallback_model
        test_route_counts[model] += 1
        predictions.append(test_prediction_book[model][i])
    return {
        "model": f"category_model_router:min_n={min_segment_n}",
        "predictions": predictions,
        "summary": {
            "fallback_model": fallback_model,
            "fallback_train_brier": global_scores[fallback_model],
            "candidate_count": len(candidates),
            "routes": routes,
            "test_route_counts": dict(test_route_counts),
        },
    }


def _contract_pnl(side: str, price: float, stake: float, outcome: int) -> float:
    if price <= 0.0 or price >= 1.0:
        return 0.0
    win = (side == "YES" and outcome == 1) or (side == "NO" and outcome == 0)
    return stake * (1.0 - price) / price if win else -stake


def _best_raw_edge(packet: FeaturePacket, pred: float) -> tuple[str, float | None, float]:
    yes_ask = packet.quote.executable_yes
    no_ask = packet.quote.executable_no
    yes_edge = pred - yes_ask if yes_ask is not None else -math.inf
    no_edge = (1.0 - pred) - no_ask if no_ask is not None else -math.inf
    if yes_edge <= 0.0 and no_edge <= 0.0:
        return "NONE", None, max(yes_edge, no_edge)
    if yes_edge >= no_edge:
        return "YES", yes_ask, yes_edge
    return "NO", no_ask, no_edge


def _gate_shadow_summary(packets: list[FeaturePacket], stat_cfg: Any) -> dict[str, Any]:
    cfg = load_config()
    cheap_reasons = Counter()
    supervisor_reasons = Counter()
    cheap_calls = 0
    supervisor_calls = 0
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for packet in packets:
        stat = forecast_stat(packet, stat_cfg)
        call_cheap, reasons = cheap_gate(packet, stat, cfg.gates)
        if call_cheap:
            cheap_calls += 1
        for reason in reasons:
            cheap_reasons[reason] += 1
            by_category[packet.category][reason] += 1
        proxy_cheap = _proxy_lane(packet, stat)
        call_supervisor, reasons = supervisor_gate(packet, stat, proxy_cheap, cfg.gates)
        if call_supervisor:
            supervisor_calls += 1
        for reason in reasons:
            supervisor_reasons[reason] += 1
    n = len(packets)
    return {
        "cheap_call_rate": cheap_calls / n if n else 0.0,
        "cheap_calls": cheap_calls,
        "supervisor_call_rate_if_proxy_lane_used": supervisor_calls / n if n else 0.0,
        "supervisor_calls_if_proxy_lane_used": supervisor_calls,
        "cheap_reasons": dict(cheap_reasons),
        "supervisor_reasons": dict(supervisor_reasons),
        "cheap_reasons_by_category": {
            category: dict(counter)
            for category, counter in sorted(by_category.items(), key=lambda item: -sum(item[1].values()))
        },
    }


def _proxy_lane(packet: FeaturePacket, stat: StatForecast) -> LaneForecast:
    p_yes = stat.probabilities.get("YES", stat.calibrated_probability)
    return LaneForecast(
        probabilities={"YES": p_yes, "NO": 1.0 - p_yes},
        confidence=stat.confidence,
        uncertainty=stat.uncertainty,
        defer_to_market=abs(p_yes - stat.market_prior) < 0.03,
        market_delta_bps=int(round((p_yes - stat.market_prior) * 10000)),
        reason_codes=["proxy_stat_lane_for_gate_shadow"],
    )


def _segment_report(
    packets: list[FeaturePacket],
    preds: list[float],
    market_preds: list[float],
    outcomes: list[int],
    risk: RiskConfig,
    *,
    min_segment_n: int,
    top_segments: int,
) -> dict[str, list[dict[str, Any]]]:
    dimensions: dict[str, Callable[[FeaturePacket, float], str]] = {
        "category": lambda packet, _pred: packet.category,
        "series_prefix": lambda packet, _pred: _series_prefix(packet),
        "spread_bucket": lambda packet, _pred: _spread_bucket(packet.quote.spread),
        "market_price_bucket": lambda packet, _pred: _price_bucket(packet.quote.market_mid),
        "horizon_bucket": lambda packet, _pred: _horizon_bucket(packet.horizon_hours),
        "momentum_bucket": lambda packet, _pred: _momentum_bucket(packet),
        "volume_bucket": lambda packet, _pred: _volume_bucket(packet.quote.volume),
        "raw_edge_bucket": lambda packet, pred: _edge_bucket(_best_raw_edge(packet, pred)[2]),
    }
    report: dict[str, list[dict[str, Any]]] = {}
    for dim, labeler in dimensions.items():
        groups: dict[str, list[tuple[FeaturePacket, float, float, int]]] = defaultdict(list)
        for packet, pred, market, outcome in zip(packets, preds, market_preds, outcomes):
            groups[labeler(packet, pred)].append((packet, pred, market, outcome))
        rows = []
        for label, items in groups.items():
            if len(items) < min_segment_n:
                continue
            ps = [item[1] for item in items]
            ms = [item[2] for item in items]
            ys = [item[3] for item in items]
            trade = _trade_summary(
                [item[0] for item in items],
                ps,
                ys,
                risk,
            )
            market_score = brier(ms, ys)
            model_score = brier(ps, ys)
            rows.append({
                "segment": label,
                "n": len(items),
                "market_brier": market_score,
                "model_brier": model_score,
                "brier_improvement": market_score - model_score,
                "ece": ece(ps, ys),
                "yes_rate": sum(ys) / len(ys),
                "avg_market_prob": sum(ms) / len(ms),
                "avg_model_prob": sum(ps) / len(ps),
                "trade_summary": trade,
            })
        rows.sort(key=lambda row: (row["brier_improvement"], row["n"]), reverse=True)
        report[dim] = rows[:top_segments]
    return report


def _period_metrics(
    packets: list[FeaturePacket],
    preds: list[float],
    market_preds: list[float],
    outcomes: list[int],
) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for packet, pred, market, outcome in zip(packets, preds, market_preds, outcomes):
        groups[_iso_week(packet.as_of)].append((pred, market, outcome))
    rows = []
    for period, items in sorted(groups.items()):
        ps = [item[0] for item in items]
        ms = [item[1] for item in items]
        ys = [item[2] for item in items]
        rows.append({
            "period": period,
            "n": len(items),
            "market_brier": brier(ms, ys),
            "model_brier": brier(ps, ys),
            "brier_improvement": brier(ms, ys) - brier(ps, ys),
        })
    return rows


def _series_prefix(packet: FeaturePacket) -> str:
    ticker = packet.event_ticker or packet.market_ticker
    return ticker.split("-")[0] if ticker else "UNKNOWN"


def _spread_bucket(spread: float | None) -> str:
    if spread is None:
        return "missing"
    if spread < 0.03:
        return "00-03pp"
    if spread < 0.06:
        return "03-06pp"
    if spread < 0.10:
        return "06-10pp"
    if spread < 0.15:
        return "10-15pp"
    if spread < 0.30:
        return "15-30pp"
    return "30pp_plus"


def _price_bucket(prob: float) -> str:
    if prob < 0.10:
        return "00-10"
    if prob < 0.25:
        return "10-25"
    if prob < 0.40:
        return "25-40"
    if prob < 0.60:
        return "40-60"
    if prob < 0.75:
        return "60-75"
    if prob < 0.90:
        return "75-90"
    return "90-100"


def _horizon_bucket(hours: float | None) -> str:
    if hours is None:
        return "missing"
    if hours < 1:
        return "lt_1h"
    if hours < 6:
        return "1h_6h"
    if hours < 24:
        return "6h_24h"
    if hours < 72:
        return "1d_3d"
    if hours < 168:
        return "3d_7d"
    return "7d_plus"


def _momentum_bucket(packet: FeaturePacket) -> str:
    move = float(packet.features.get("price_momentum") or 0.0)
    if move <= -0.15:
        return "down_15pp_plus"
    if move <= -0.05:
        return "down_5_15pp"
    if move < 0.05:
        return "flat_5pp"
    if move < 0.15:
        return "up_5_15pp"
    return "up_15pp_plus"


def _volume_bucket(volume: float | None) -> str:
    if volume is None:
        return "missing"
    if volume < 1000:
        return "lt_1k"
    if volume < 10000:
        return "1k_10k"
    if volume < 100000:
        return "10k_100k"
    return "100k_plus"


def _edge_bucket(edge: float) -> str:
    if edge < 0.0:
        return "negative"
    if edge < 0.03:
        return "00-03pp"
    if edge < 0.06:
        return "03-06pp"
    if edge < 0.10:
        return "06-10pp"
    return "10pp_plus"


def _iso_week(value: str | None) -> str:
    parsed = parse_dt(value)
    if parsed is None:
        return "unknown"
    iso = parsed.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["topvol", "nonbinary", "unified", "live_clean", "eval_pack"], default="topvol")
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--candle-stride-minutes", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--since-close")
    parser.add_argument("--until-close")
    parser.add_argument("--random-as-of", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260517)
    parser.add_argument("--min-horizon-minutes", type=float, default=5.0)
    parser.add_argument("--max-horizon-hours", type=float)
    parser.add_argument("--min-history-snapshots", type=int, default=5)
    parser.add_argument("--decision-budget-minutes", type=float, default=5.0)
    parser.add_argument("--min-segment-n", type=int, default=30)
    parser.add_argument("--top-segments", type=int, default=12)
    parser.add_argument("--evidence-mode", choices=["none", "strict_pit", "relaxed_published_at"], default="none")
    parser.add_argument("--evidence-manifest", type=Path, action="append", default=[])
    parser.add_argument("--evidence-max-records", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_oos_evaluation(
        source=args.source,
        horizon_hours=args.horizon_hours,
        candle_stride_minutes=args.candle_stride_minutes,
        train_fraction=args.train_fraction,
        include_context=args.context,
        limit=args.limit,
        since_close=args.since_close,
        until_close=args.until_close,
        random_as_of=args.random_as_of,
        random_seed=args.random_seed,
        min_horizon_minutes=args.min_horizon_minutes,
        max_horizon_hours=args.max_horizon_hours,
        min_history_snapshots=args.min_history_snapshots,
        decision_budget_minutes=args.decision_budget_minutes,
        min_segment_n=args.min_segment_n,
        top_segments=args.top_segments,
        evidence_mode=args.evidence_mode,
        evidence_manifest_paths=args.evidence_manifest,
        evidence_max_records=args.evidence_max_records,
    )
    text = json.dumps(result, indent=2, sort_keys=True, default=_json_default)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({
        "source": result["source"],
        "horizon_hours": result["horizon_hours"],
        "random_as_of": result["random_as_of"],
        "candle_stride_minutes": result["candle_stride_minutes"],
        "include_context": result["include_context"],
        "evidence_replay": result["evidence_replay"],
        "since_close": result["since_close"],
        "until_close": result["until_close"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "market_test_brier": result["market_test_brier"],
        "best_model": result["best_model"],
        "top_models": result["top_models"][:5],
        "gate_shadow": result["gate_shadow"],
        "stat_model_routing": result["stat_model_routing"],
        "segment_report": {
            key: value[:5]
            for key, value in result["segment_report"].items()
        },
    }, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
