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
        "final_gemini_live_brief": _final_gemini_live_brief(packet, prior, model_id=model_id),
        "evidence_source_policy": evidence_source_policy(packet.category, observed_sources),
        "instruction": (
            "Return the required JSON object only. Use exact outcome labels from event_packet.outcomes. "
            "You are the final probability model. Use deterministic_prior as model context, not an output cap. "
            "Use retrieval_confidence on evidence items to distinguish strong PIT evidence from noisy or stale context. "
            "When live native search is enabled, answer the targeted_search_questions in your evidence triage before "
            "setting probabilities, but expose only compact audit fields. "
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
            "Choose calibrated final probabilities after weighing event text, rules, market-derived priors, "
            "live evidence, and information gaps. The runtime will "
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


def _final_gemini_live_brief(
    packet: ArenaForecastPacket,
    prior: ArenaPrior,
    *,
    model_id: str,
) -> dict[str, object]:
    live_runtime = packet.features.get("live_runtime_context") if isinstance(packet.features, dict) else None
    source_status = packet.features.get("live_source_status") if isinstance(packet.features, dict) else None
    return {
        "model": model_id,
        "mode": "live_one_pass_grounded_forecast" if _runtime_grounding_enabled(live_runtime) else "forecast_without_live_grounding",
        "deadline_context": live_runtime or {},
        "source_status": source_status or {},
        "targeted_search_questions": _targeted_search_questions(packet),
        "search_budget_policy": [
            "Search current web only for facts that can materially change this exact contract.",
            (
                "Prioritize official box scores/injury reports/team pages, league sources, market/odds pages, "
                "official economic releases, reputable news wires, and primary source filings."
            ),
            "Avoid broad background research and outdated recap pages when the deadline is tight.",
            "If sources conflict, state the conflict in counterarguments and keep probabilities less extreme.",
            "If a source has no visible timestamp or is stale relative to the event, treat it as weak context.",
        ],
        "calibration_policy_under_deadline": {
            "deterministic_prior": prior.probabilities,
            "prior_confidence": prior.confidence,
            "prior_uncertainty": prior.uncertainty,
            "weak_or_missing_current_evidence": "shrink toward deterministic_prior",
            "fresh_high_quality_contract_specific_evidence": "allow larger movement away from deterministic_prior",
            "nearly_resolved_event": "probabilities may be decisive, but only with directly relevant current facts",
        },
        "contract_interpretation_hints": _contract_interpretation_hints(packet),
    }


def _runtime_grounding_enabled(live_runtime: object) -> bool:
    return isinstance(live_runtime, dict) and bool(live_runtime.get("native_search_grounding_enabled"))


def _targeted_search_questions(packet: ArenaForecastPacket) -> list[str]:
    questions = [
        "What current, timestamped facts materially affect the probability of this exact contract?",
        "Are there official or high-quality sources that confirm the current status of the event or participants?",
        "Do the contract rules create edge cases that change how the listed outcomes should be interpreted?",
    ]
    category = packet.category
    text = _event_text(packet)
    if category == "Sports":
        questions.extend([
            "What are the latest injuries, starters, lineups, rest/travel factors, weather or venue effects, and odds movement?",
            "For player-stat or multi-leg contracts, what are the latest role/minutes/usage and matchup facts for each leg?",
        ])
    elif category in {"Economics", "Financials"}:
        questions.extend([
            "What official releases, central-bank statements, yields, commodities, or market moves have occurred before as_of?",
            "When is the next relevant data release relative to close_time and the resolution rules?",
        ])
    elif category in {"Crypto", "Commodities"}:
        questions.extend([
            "What recent spot/futures price action, inventory/flow data, macro risk sentiment, regulation, or exchange news matters?",
            "Are there weekend, settlement, liquidity, or benchmark-timing effects that affect the exact resolution window?",
        ])
    elif category in {"Politics", "Elections"}:
        questions.extend([
            "What are the latest polls, official actions, court rulings, endorsements, or campaign events from high-quality sources?",
            "Which source families conflict with social sentiment or partisan commentary?",
        ])
    elif category in {"Entertainment", "Culture"} or "survivor" in text or "reality" in text:
        questions.extend([
            "What official releases, credible spoilers, audience/voting signals, ratings, box office, or awards news matter?",
            "How fresh and reliable are entertainment or spoiler sources for the listed outcome labels?",
        ])
    elif category in {"Weather", "Climate and Weather"}:
        questions.extend([
            "What are the latest official forecasts, observations, warnings, and model updates for the resolution window?",
            "Which source timestamp best matches the contract's measurement period?",
        ])
    return questions[:7]


def _contract_interpretation_hints(packet: ArenaForecastPacket) -> dict[str, object]:
    multileg = packet.extracted_entities.get("kalshi_multileg_contract") if isinstance(packet.extracted_entities, dict) else None
    hints: dict[str, object] = {
        "event_structure": packet.event_structure,
        "outcome_labels_are_binding": True,
        "horizon_hours": packet.horizon_hours,
    }
    if isinstance(multileg, dict) and multileg.get("is_multileg"):
        hints["kalshi_multileg"] = {
            "interpret_yes_as_joint_conjunction": True,
            "interpret_no_as_complement": True,
            "components": multileg.get("components"),
        }
    if len(packet.outcomes) > 2:
        hints["multi_outcome"] = {
            "normalize_across_all_listed_labels": True,
            "avoid_binary_yes_no_framing": True,
        }
    return hints


def _event_text(packet: ArenaForecastPacket) -> str:
    return " ".join(
        str(part or "").lower()
        for part in (packet.title, packet.subtitle, packet.description, packet.context, packet.rules)
    )
