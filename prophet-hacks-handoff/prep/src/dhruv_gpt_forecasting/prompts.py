"""Prompt assembly."""

from __future__ import annotations

import json

from .arena_types import ArenaForecast, ArenaForecastPacket, ArenaPrior
from .config import PACKAGE_ROOT
from .evidence_sources import evidence_source_policy


PROMPT_ROOT = PACKAGE_ROOT / "src" / "prep" / "forecasters" / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_ROOT / name).read_text(encoding="utf-8")


def arena_messages(
    packet: ArenaForecastPacket,
    prior: ArenaPrior,
    *,
    model_id: str = "gemini-3-flash-preview",
) -> list[dict[str, str]]:
    observed_sources = [str(item.get("source") or "") for item in packet.live_evidence if isinstance(item, dict)]
    payload = {
        "event_packet": packet.compact_dict(),
        "deterministic_prior": prior.to_dict(),
        "model_handoff": _arena_model_handoff(packet, prior, model_id=model_id),
        "evidence_source_policy": evidence_source_policy(packet.category, observed_sources),
        "instruction": (
            "Return the required JSON object only. Use exact outcome labels from event_packet.outcomes. "
            "You are the final probability model. Use deterministic_prior as model context, not an output cap. "
            "Use retrieval_confidence on evidence items to distinguish strong PIT evidence from noisy or stale context. "
            "Optimize expected Brier score, not trading edge. Every outcome must receive a probability."
        ),
    }
    return [
        {"role": "system", "content": load_prompt("forecasting_brier_v1_system.txt")},
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"), sort_keys=True)},
    ]


def arena_repair_messages(
    packet: ArenaForecastPacket,
    prior: ArenaPrior,
    bad_payload: str,
    *,
    model_id: str = "gemini-3-flash-preview",
) -> list[dict[str, str]]:
    observed_sources = [str(item.get("source") or "") for item in packet.live_evidence if isinstance(item, dict)]
    payload = {
        "event_packet": packet.compact_dict(),
        "deterministic_prior": prior.to_dict(),
        "model_handoff": _arena_model_handoff(packet, prior, model_id=model_id),
        "evidence_source_policy": evidence_source_policy(packet.category, observed_sources),
        "invalid_response": bad_payload[:4000],
        "instruction": (
            "Repair the invalid response. Return only valid JSON with probabilities for every exact outcome label."
        ),
    }
    return [
        {"role": "system", "content": load_prompt("forecasting_brier_v1_system.txt")},
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"), sort_keys=True)},
    ]


def arena_audit_messages(
    packet: ArenaForecastPacket,
    prior: ArenaPrior,
    first_pass: ArenaForecast,
    *,
    model_id: str = "gemini-3-flash-preview",
) -> list[dict[str, str]]:
    observed_sources = [str(item.get("source") or "") for item in packet.live_evidence if isinstance(item, dict)]
    payload = {
        "event_packet": packet.compact_dict(),
        "deterministic_prior": prior.to_dict(),
        "first_pass_forecast": first_pass.to_dict(),
        "model_handoff": _arena_model_handoff(packet, prior, model_id=model_id),
        "evidence_source_policy": evidence_source_policy(packet.category, observed_sources),
        "instruction": (
            "Audit the first pass for Brier-score calibration. Return a final JSON forecast only. "
            "Do not add commentary outside JSON."
        ),
    }
    return [
        {"role": "system", "content": load_prompt("forecasting_brier_v1_system.txt")},
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"), sort_keys=True)},
    ]


def prompt_hash(messages: list[dict[str, str]], model: str) -> str:
    import hashlib

    payload = {"model": model, "messages": messages}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _arena_model_handoff(
    packet: ArenaForecastPacket,
    prior: ArenaPrior,
    *,
    model_id: str,
) -> dict[str, object]:
    return {
        "primary_gpt_model": model_id,
        "native_search_grounding": {
            "role": "live_current_information_tool_when_enabled",
            "policy": (
                "Use search grounding for current facts, injuries, lineups, odds/news changes, macro releases, "
                "and other fast-moving evidence. Respect as_of_policy: do not rely on current web results when "
                "the forecast packet is a historical backtest cutoff."
            ),
        },
        "gpt_role": "final_probability_model",
        "objective": "minimize expected Brier score over the exact outcome labels",
        "deterministic_models": {
            "role": "inputs_to_consider",
            "probabilities": prior.probabilities,
            "confidence": prior.confidence,
            "uncertainty": prior.uncertainty,
            "reason_codes": prior.reason_codes,
            "diagnostics": prior.diagnostics,
        },
        "linked_market_model": {
            "role": "secondary_probability_model_when_present",
            "source_name": "linked_market_model",
            "usage": (
                "Use linked-market probabilities as an explicit model lane. Trust them most when confidence and "
                "quality are high and inferred_structure is a coherent same-event component distribution."
            ),
        },
        "final_probability_rule": (
            "Choose calibrated final probabilities after weighing event text, rules, historical analogs, "
            "time-series or market-derived priors, live evidence, and information gaps. The runtime will "
            "only validate labels, clamp impossible numeric values, and normalize the distribution."
        ),
        "kalshi_multileg_contract_rule": (
            "If event_packet.extracted_entities.kalshi_multileg_contract.is_multileg is true and outcomes are "
            "YES/NO, interpret YES as the joint event that every listed component leg resolves with its stated "
            "YES or NO side. Interpret NO as the complement: at least one listed component leg fails. Estimate "
            "the joint probability, accounting for leg dependence and avoiding a simple average of leg chances."
        ),
        "as_of_policy": (
            f"Use only facts that would have been knowable at {packet.as_of}. "
            "Ignore any source record whose own timestamp appears after that cutoff."
        ),
    }
