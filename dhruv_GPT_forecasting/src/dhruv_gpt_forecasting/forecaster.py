"""End-to-end GPT lane forecast orchestration."""

from __future__ import annotations

import time
from typing import Any

from .combine import make_supervisor_decision
from .config import ForecastConfig, load_config, load_local_env
from .context import build_related_context_evidence
from .features import build_feature_packet
from .gating import decide_gates, supervisor_gate
from .openrouter import call_openrouter_lane
from .prompts import cheap_messages, supervisor_messages
from .schemas import SupervisorDecision
from .stat_router import forecast_stat_routed


def forecast_event(
    event: dict[str, Any],
    market_info: dict[str, Any] | None = None,
    *,
    price_trajectory: list[dict[str, Any]] | None = None,
    external_evidence: list[dict[str, Any]] | None = None,
    config: ForecastConfig | None = None,
    dry_run: bool | None = None,
    include_context: bool = True,
    force_cheap: bool = False,
    deadline_at: float | None = None,
) -> SupervisorDecision:
    load_local_env()
    cfg = config or load_config()
    if dry_run is None:
        dry_run = cfg.budget.dry_run_default
    packet = build_feature_packet(
        event,
        market_info,
        price_trajectory=price_trajectory,
        external_evidence=external_evidence,
    )
    context_evidence = build_related_context_evidence(packet) if include_context else []
    if context_evidence:
        packet.evidence_digest = context_evidence + packet.evidence_digest
        packet.features["related_context_count"] = len(context_evidence)
        packet.features["related_context_sources"] = sorted(
            {str(item.get("source")) for item in context_evidence if item.get("source")}
        )
    stat = forecast_stat_routed(packet, cfg.stat, include_context=include_context)
    gates = decide_gates(packet, stat, cfg.gates)
    if force_cheap:
        gates.call_cheap = True
        if "force_cheap_call" not in gates.reason_codes:
            gates.reason_codes.append("force_cheap_call")
    api_logs = []
    errors = []
    cheap = None
    supervisor = None
    if gates.call_cheap and cfg.cheap_model.enabled and not dry_run and _has_time_for_call(deadline_at):
        try:
            cheap, call_log = call_openrouter_lane(
                model=cfg.cheap_model,
                messages=cheap_messages(packet, stat),
                packet=packet,
                budget=cfg.budget,
                cache_key=f"cheap:{packet.category}",
                timeout_seconds=_remaining_timeout(deadline_at),
            )
            api_logs.append(call_log.to_dict())
        except Exception as exc:  # noqa: BLE001 - fallback is intentional.
            errors.append(f"cheap_lane:{type(exc).__name__}:{exc}")
    elif gates.call_cheap and cfg.cheap_model.enabled and not dry_run:
        errors.append("cheap_lane:deadline_budget_before_call")
    if cheap is not None:
        call_supervisor, supervisor_reasons = supervisor_gate(packet, stat, cheap, cfg.gates)
        gates.call_supervisor = call_supervisor
        gates.reason_codes.extend(supervisor_reasons)
    if gates.call_supervisor and cfg.supervisor_model.enabled and not dry_run and _has_time_for_call(deadline_at):
        try:
            supervisor, call_log = call_openrouter_lane(
                model=cfg.supervisor_model,
                messages=supervisor_messages(packet, stat, cheap),
                packet=packet,
                budget=cfg.budget,
                cache_key=f"supervisor:{packet.category}",
                timeout_seconds=_remaining_timeout(deadline_at),
            )
            api_logs.append(call_log.to_dict())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"supervisor_lane:{type(exc).__name__}:{exc}")
    elif gates.call_supervisor and cfg.supervisor_model.enabled and not dry_run:
        errors.append("supervisor_lane:deadline_budget_before_call")
    audit = {
        "market_ticker": packet.market_ticker,
        "category": packet.category,
        "event_structure": packet.event_structure,
        "dry_run": dry_run,
        "gates": gates.to_dict(),
        "stat": stat.to_dict(),
        "api_logs": api_logs,
        "errors": errors,
        "context_evidence_count": len(context_evidence),
        "context_sources": packet.features.get("related_context_sources", []),
        "planned_models": {
            "cheap": cfg.cheap_model.model if gates.call_cheap else None,
            "supervisor": cfg.supervisor_model.model if gates.call_supervisor else None,
        },
        "deadline_remaining_seconds": _remaining_seconds(deadline_at),
    }
    return make_supervisor_decision(
        packet,
        stat,
        cfg.risk,
        cheap=cheap,
        supervisor=supervisor,
        audit=audit,
    )


def predict(event: dict[str, Any]) -> dict[str, Any]:
    """Prophet Arena-style entrypoint returning Brier-optimized probabilities."""
    from .arena_agent import predict as arena_predict

    return arena_predict(event)


def predict_p_yes(event: dict[str, Any], market_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compatibility shim for binary backtest harnesses expecting `p_yes`."""
    decision = forecast_event(event, market_info, dry_run=True)
    p_yes = decision.probabilities.get("YES", 0.5)
    return {"p_yes": p_yes, "rationale": decision.source, "audit": decision.audit_summary}


def _remaining_seconds(deadline_at: float | None) -> float | None:
    if deadline_at is None:
        return None
    return max(0.0, deadline_at - time.monotonic())


def _remaining_timeout(deadline_at: float | None, *, reserve_seconds: float = 3.0) -> float | None:
    remaining = _remaining_seconds(deadline_at)
    if remaining is None:
        return None
    return max(0.1, remaining - reserve_seconds)


def _has_time_for_call(deadline_at: float | None, *, reserve_seconds: float = 3.0) -> bool:
    remaining = _remaining_seconds(deadline_at)
    return remaining is None or remaining > reserve_seconds
