"""Supervisor aggregation for model forecasts.

Per Prophet Arena dev docs, predictions are a probability *distribution* over
the event's `outcomes` list. We logit-pool per outcome across lanes, then
renormalize. Binary events (outcomes=["YES","NO"]) reduce cleanly to the
old logit-pool of a single p_yes.

Market anchor only applies when the packet has a Kalshi quote (binary YES/NO).
For multi-outcome events without market data, we anchor toward a uniform prior
with a small weight (so a single noisy model can't dominate).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .calibration import CalibrationConfig, calibrate_to_market, inv_logit, logit
from .schemas import (
    MarketPacket,
    ModelForecast,
    SupervisorForecast,
    clamp_prob,
    normalize_distribution,
)


OPENROUTER_CHAT_COMPLETIONS = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class EnsembleMember:
    forecast: ModelForecast
    configured_weight: float = 1.0

    @property
    def effective_weight(self) -> float:
        diag = self.forecast.diagnostics
        f = self.forecast.forecast
        quality = {"low": 0.55, "medium": 0.85, "high": 1.0}.get(diag.evidence_quality, 0.85)
        clarity = {"low": 0.65, "medium": 0.85, "high": 1.0}.get(diag.rules_clarity, 0.85)
        liquidity = {"low": 0.85, "medium": 0.95, "high": 1.0}.get(diag.liquidity_quality, 0.95)
        defer = 0.65 if diag.should_defer_to_market else 1.0
        reasoning = 1.0
        if not self.forecast.reasoning_track.summary:
            reasoning *= 0.85
        if not self.forecast.reasoning_track.key_evidence and not diag.should_defer_to_market:
            reasoning *= 0.90
        return max(0.01, self.configured_weight * quality * clarity * liquidity * defer * reasoning * (0.5 + f.confidence / 2.0))


@dataclass
class EnsembleRun:
    members: list[EnsembleMember]
    errors: list[str]


@dataclass
class JudgeConfig:
    enabled: bool = False
    provider: str = "openrouter"
    model: str = "openai/gpt-5.4"
    api_key_env: str = "OPENROUTER_API_KEY_JUDGE"
    system_prompt_path: str = "src/prep/forecasters/prompts/judge_aggregator_system.md"
    temperature: float = 0.0
    max_tokens: int = 1400
    blend_weight: float = 0.35
    min_members: int = 2
    timeout_seconds: int = 120

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "JudgeConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            provider=str(data.get("provider", "openrouter")),
            model=str(data.get("model", "openai/gpt-5.4")),
            api_key_env=str(data.get("api_key_env", "OPENROUTER_API_KEY_JUDGE")),
            system_prompt_path=str(data.get("system_prompt_path", "src/prep/forecasters/prompts/judge_aggregator_system.md")),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 1400)),
            blend_weight=float(data.get("blend_weight", 0.35)),
            min_members=int(data.get("min_members", 2)),
            timeout_seconds=int(data.get("timeout_seconds", 120)),
        )


def _prep_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_prompt_file(path_value: str) -> str:
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, _prep_root() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"judge system_prompt_path not found: {path_value}")


def _judge_system_prompt(judge: JudgeConfig) -> str:
    return _read_prompt_file(judge.system_prompt_path)


def forecast_members_parallel(
    model_configs: list[Any],
    packet: MarketPacket,
    *,
    max_workers: int | None = None,
    continue_on_error: bool = True,
) -> EnsembleRun:
    """Run all configured council lanes concurrently and return ordered members.

    The aggregation code consumes `EnsembleMember` objects, so this helper keeps
    the same public aggregation structure while making the standard council
    execution path parallel. Results are sorted back into config order before
    the judge sees them, which keeps audit logs stable across runs.
    """
    from .forecasters import forecast_from_config

    configs = list(model_configs)
    if not configs:
        return EnsembleRun(members=[], errors=[])

    worker_count = max(1, max_workers or len(configs))
    completed: list[tuple[int, EnsembleMember]] = []
    errors: list[str] = []
    pool = ThreadPoolExecutor(max_workers=worker_count)
    futures = {
        pool.submit(forecast_from_config, model_config, packet): (idx, model_config)
        for idx, model_config in enumerate(configs)
    }
    try:
        for fut in as_completed(futures):
            idx, model_config = futures[fut]
            try:
                forecast = fut.result()
            except Exception as exc:  # noqa: BLE001
                model_name = getattr(model_config, "name", f"model_{idx}")
                msg = f"{model_name}: {type(exc).__name__}: {exc}"
                if not continue_on_error:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(msg) from exc
                errors.append(msg)
                continue
            completed.append((
                idx,
                EnsembleMember(
                    forecast=forecast,
                    configured_weight=float(getattr(model_config, "weight", 1.0)),
                ),
            ))
    finally:
        pool.shutdown(wait=continue_on_error, cancel_futures=not continue_on_error)

    completed.sort(key=lambda item: item[0])
    return EnsembleRun(
        members=[member for _, member in completed],
        errors=errors,
    )


def _anchor_distribution(packet: MarketPacket) -> dict[str, float]:
    """Prior distribution used as the market-anchor in the logit pool.

    Binary Kalshi events: YES = market_mid, NO = 1 - market_mid.
    Multi-outcome: uniform over the listed outcomes.
    """
    outs = packet.outcomes or ["YES", "NO"]
    if tuple(outs) == ("YES", "NO") and packet.kalshi is not None:
        mid = packet.kalshi.market_mid
        return {"YES": mid, "NO": 1.0 - mid}
    n = max(1, len(outs))
    return {o: 1.0 / n for o in outs}


def _lookup_probability(probs: dict[str, float], outcome: str, *, is_binary: bool) -> float | None:
    p = probs.get(outcome)
    if p is not None:
        return p
    folded = outcome.casefold()
    matches = [
        value
        for key, value in probs.items()
        if isinstance(key, str) and key.casefold() == folded
    ]
    if is_binary and matches:
        return matches[0]
    if len(matches) == 1:
        return matches[0]
    return None


def _aligned_distribution(
    probs: dict[str, float],
    outcomes: list[str],
    fallback: dict[str, float],
) -> dict[str, float]:
    """Project any lane distribution onto the canonical outcome labels."""
    is_binary = tuple(outcomes) == ("YES", "NO")
    n = max(1, len(outcomes))
    uniform = 1.0 / n
    aligned = {}
    for outcome in outcomes:
        p = _lookup_probability(probs, outcome, is_binary=is_binary)
        if p is None:
            p = fallback.get(outcome, uniform)
        aligned[outcome] = clamp_prob(p)
    return normalize_distribution(aligned)


def _pool_distributions(
    distributions: list[tuple[dict[str, float], float]],
    outcomes: list[str],
) -> dict[str, float]:
    """Weighted logit-pool, per outcome.

    `distributions` is a list of (probs, weight). For each outcome label we
    average weighted logits, then inv-logit, then renormalize across outcomes.
    Missing outcomes in a lane's distribution fall back to uniform (1/N).

    For binary YES/NO events specifically, a case-insensitive secondary
    lookup is used if the exact-case match misses, so a lane that returned
    {"Yes": 0.7} against canonical ["YES", "NO"] still contributes its
    signal rather than being silently substituted with uniform. Multi-
    outcome events are untouched (avoids collision risk on labels that
    differ only by case).
    """
    if not outcomes:
        return {}
    n = len(outcomes)
    uniform = 1.0 / n
    is_binary = tuple(outcomes) == ("YES", "NO")
    raw: dict[str, float] = {}
    for outcome in outcomes:
        weighted_sum = 0.0
        total_w = 0.0
        for probs, w in distributions:
            p = probs.get(outcome)
            if p is None and is_binary:
                folded = outcome.casefold()
                for k, v in probs.items():
                    if isinstance(k, str) and k.casefold() == folded:
                        p = v
                        break
            if p is None or p <= 0 or p >= 1:
                p = clamp_prob(p if p is not None else uniform)
            weighted_sum += w * logit(p)
            total_w += w
        if total_w <= 0:
            raw[outcome] = uniform
        else:
            raw[outcome] = inv_logit(weighted_sum / total_w)
    return normalize_distribution(raw)


def _json_excerpt(value: Any, *, max_chars: int = 1200) -> Any:
    text = json.dumps(value, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return value
    return text[:max_chars] + "...[truncated]"


def _judge_prompt(
    judge: JudgeConfig,
    packet: MarketPacket,
    members: list[EnsembleMember],
    assessments: list[dict[str, Any]],
    *,
    anchor: dict[str, float],
    raw_dist: dict[str, float],
    calibrated_dist: dict[str, float],
) -> list[dict[str, str]]:
    outcomes = packet.outcomes or ["YES", "NO"]
    council = []
    for member, assessment in zip(members, assessments):
        reasoning = member.forecast.reasoning_track
        diagnostics = member.forecast.diagnostics
        council.append({
            "model_id": member.forecast.model_id,
            "provider": member.forecast.provider,
            "configured_weight": member.configured_weight,
            "effective_weight": assessment["effective_weight"],
            "probabilities": assessment["probabilities"],
            "confidence": member.forecast.forecast.confidence,
            "uncertainty": member.forecast.forecast.uncertainty,
            "diagnostics": diagnostics.to_dict(),
            "reasoning_track": {
                "summary": reasoning.summary,
                "base_rate": reasoning.base_rate,
                "market_analysis": reasoning.market_analysis,
                "context_market_analysis": reasoning.context_market_analysis,
                "key_evidence": reasoning.key_evidence[:5],
                "source_audit": reasoning.source_audit[:8],
                "counterarguments": reasoning.counterarguments[:3],
                "assumptions": reasoning.assumptions[:3],
                "information_gaps": reasoning.information_gaps[:3],
                "what_would_change_my_mind": reasoning.what_would_change_my_mind[:3],
            },
        })
    market_context = {
        "as_of": packet.as_of,
        "event_ticker": packet.event_ticker,
        "market_ticker": packet.market_ticker,
        "title": packet.title,
        "subtitle": packet.subtitle,
        "category": packet.category,
        "close_time": packet.close_time,
        "rules": packet.rules,
        "outcomes": outcomes,
        "kalshi": packet.kalshi.to_dict() if packet.kalshi else None,
        "retrieval": {
            key: _json_excerpt(value)
            for key, value in packet.retrieval.items()
            if key in {"description", "sources", "market_data", "market_implied_probabilities"}
        },
    }
    payload = {
        "market": market_context,
        "deterministic_ensemble": {
            "anchor_probabilities": anchor,
            "raw_logit_pool_probabilities": raw_dist,
            "calibrated_probabilities": calibrated_dist,
        },
        "council_members": council,
        "required_outcomes": outcomes,
    }
    schema = {
        "decision": "select_member | mix_members | defer_to_deterministic",
        "probabilities": {outcome: "float probability for this exact outcome label" for outcome in outcomes},
        "confidence": "float 0-1 for the judge aggregation only",
        "selected_model_ids": ["model ids that drove the result, empty if deterministic"],
        "rationale": "short explanation using only the council outputs and packet contents",
        "risk_notes": ["short notes about weak reasoning, leakage risk, disagreement, or missing evidence"],
    }
    system = _judge_system_prompt(judge)
    user = (
        f"Evaluate the council and return one JSON object matching this schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"AGGREGATION_INPUT:\n{json.dumps(payload, indent=2, sort_keys=True, default=str)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start:end + 1]
    return json.loads(text)


def _call_judge_llm(
    judge: JudgeConfig,
    packet: MarketPacket,
    members: list[EnsembleMember],
    assessments: list[dict[str, Any]],
    *,
    anchor: dict[str, float],
    raw_dist: dict[str, float],
    calibrated_dist: dict[str, float],
) -> dict[str, Any]:
    if judge.provider != "openrouter":
        raise ValueError(f"Unsupported judge provider: {judge.provider}")
    api_key = os.environ.get(judge.api_key_env) or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(f"{judge.api_key_env} or OPENROUTER_API_KEY must be set for judge aggregation")
    payload = {
        "model": judge.model,
        "messages": _judge_prompt(
            judge,
            packet,
            members,
            assessments,
            anchor=anchor,
            raw_dist=raw_dist,
            calibrated_dist=calibrated_dist,
        ),
        "temperature": judge.temperature,
        "max_tokens": judge.max_tokens,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        OPENROUTER_CHAT_COMPLETIONS,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/prophet-hacks",
            "X-Title": "ProphetHacks Judge Ensemble",
        },
        json=payload,
        timeout=judge.timeout_seconds,
    )
    resp.raise_for_status()
    raw = resp.json()
    text = raw["choices"][0]["message"].get("content", "")
    parsed = _extract_json_object(text)
    parsed["_raw_judge_response"] = raw
    return parsed


def _apply_judge(
    packet: MarketPacket,
    members: list[EnsembleMember],
    assessments: list[dict[str, Any]],
    *,
    anchor: dict[str, float],
    raw_dist: dict[str, float],
    calibrated_dist: dict[str, float],
    judge: JudgeConfig | None,
) -> tuple[dict[str, float], float | None, dict[str, Any] | None, list[str]]:
    if judge is None or not judge.enabled or len(members) < judge.min_members:
        return calibrated_dist, None, None, []
    judge_result = _call_judge_llm(
        judge,
        packet,
        members,
        assessments,
        anchor=anchor,
        raw_dist=raw_dist,
        calibrated_dist=calibrated_dist,
    )
    judge_probs = judge_result.get("probabilities")
    if not isinstance(judge_probs, dict):
        raise ValueError("judge response missing probabilities object")
    judge_dist = _aligned_distribution(judge_probs, packet.outcomes or ["YES", "NO"], calibrated_dist)
    blend_weight = max(0.0, min(1.0, judge.blend_weight))
    final_dist = normalize_distribution({
        outcome: calibrated_dist.get(outcome, 0.0) * (1.0 - blend_weight) + judge_dist.get(outcome, 0.0) * blend_weight
        for outcome in (packet.outcomes or ["YES", "NO"])
    })
    try:
        judge_confidence = max(0.0, min(1.0, float(judge_result.get("confidence", 0.5))))
    except (TypeError, ValueError):
        judge_confidence = 0.5
    risk_notes = [str(note) for note in judge_result.get("risk_notes") or [] if str(note)]
    judge_result["blend_weight"] = blend_weight
    judge_result["judge_probabilities_normalized"] = judge_dist
    judge_result["final_probabilities_after_blend"] = final_dist
    return final_dist, judge_confidence, judge_result, risk_notes


def aggregate_forecasts(
    packet: MarketPacket,
    members: list[EnsembleMember],
    calibration: CalibrationConfig | None = None,
    *,
    market_anchor_weight: float = 1.5,
    judge: JudgeConfig | dict[str, Any] | None = None,
) -> SupervisorForecast:
    outcomes = packet.outcomes or ["YES", "NO"]
    anchor = _anchor_distribution(packet)
    judge_config = JudgeConfig.from_dict(judge) if isinstance(judge, dict) else judge

    if not members:
        raw_dist = dict(anchor)
        assessments: list[dict[str, Any]] = []
    else:
        contributions: list[tuple[dict[str, float], float]] = [(anchor, market_anchor_weight)]
        assessments = []
        for member in members:
            w = member.effective_weight
            mp = _aligned_distribution(dict(member.forecast.probabilities), outcomes, anchor)
            contributions.append((mp, w))
            assessments.append({
                "model_id": member.forecast.model_id,
                "provider": member.forecast.provider,
                "probabilities": mp,
                "p_yes": member.forecast.p_yes,
                "configured_weight": member.configured_weight,
                "effective_weight": w,
                "confidence": member.forecast.forecast.confidence,
                "summary": member.forecast.reasoning_track.summary,
                "defer_to_market": member.forecast.diagnostics.should_defer_to_market,
            })
        raw_dist = _pool_distributions(contributions, outcomes)

    calibration = calibration or CalibrationConfig()
    # Calibration shrinks each outcome toward the market anchor by the same
    # per-event weight. Multi-outcome non-Kalshi events get the uniform anchor.
    if tuple(outcomes) == ("YES", "NO") and packet.kalshi is not None and packet.kalshi.market_mid != 0.5:
        # Reuse existing binary calibrate_to_market on YES side, mirror to NO
        cal_yes, shrink_weight = calibrate_to_market(raw_dist.get("YES", 0.5), packet, calibration)
        calibrated_dist = normalize_distribution({"YES": cal_yes, "NO": 1.0 - cal_yes})
    else:
        # Multi-outcome shrinkage: pull each prob toward the market anchor.
        # Use market_implied_probabilities from retrieval when available (populated
        # by _enrich_packet for multi-market Kalshi events); fall back to uniform.
        shrink_weight = calibration.shrink_weight(packet)
        market_implied = packet.retrieval.get("market_implied_probabilities") or {}
        n = max(1, len(outcomes))
        calibrated_dist = normalize_distribution({
            o: market_implied.get(o, 1.0 / n) + shrink_weight * (
                raw_dist.get(o, 1.0 / n) - market_implied.get(o, 1.0 / n)
            )
            for o in outcomes
        })

    judge_result: dict[str, Any] | None = None
    judge_confidence: float | None = None
    judge_risk_notes: list[str] = []
    if members:
        try:
            calibrated_dist, judge_confidence, judge_result, judge_risk_notes = _apply_judge(
                packet,
                members,
                assessments,
                anchor=anchor,
                raw_dist=raw_dist,
                calibrated_dist=calibrated_dist,
                judge=judge_config,
            )
        except Exception as exc:  # noqa: BLE001
            judge_risk_notes.append(f"Judge aggregation failed; used deterministic ensemble. {type(exc).__name__}: {exc}")

    # Disagreement: max range of p across lanes, for the most-likely outcome
    if members:
        top_outcome = max(raw_dist, key=raw_dist.get)
        ps = [assessment["probabilities"].get(top_outcome, 1.0 / len(outcomes)) for assessment in assessments]
        disagreement = (max(ps) - min(ps)) if ps else 0.0
    else:
        disagreement = 0.0
    confidence = clamp_prob(1.0 - disagreement, lo=0.0, hi=1.0)

    if disagreement > 0.20:
        disagreement_summary = f"High model disagreement on {top_outcome}: range {min(ps):.3f}-{max(ps):.3f}."
    elif disagreement > 0.08:
        disagreement_summary = f"Moderate disagreement on {top_outcome}: range {min(ps):.3f}-{max(ps):.3f}."
    else:
        disagreement_summary = "Models are broadly aligned." if members else "No model forecasts; using anchor."

    if judge_result:
        assessments.append({
            "model_id": judge_config.model if judge_config else "judge",
            "provider": judge_config.provider if judge_config else "judge",
            "role": "judge",
            "probabilities": judge_result.get("final_probabilities_after_blend", calibrated_dist),
            "p_yes": judge_result.get("final_probabilities_after_blend", calibrated_dist).get("YES", 0.5),
            "configured_weight": judge_result.get("blend_weight"),
            "effective_weight": judge_result.get("blend_weight"),
            "confidence": judge_confidence,
            "summary": judge_result.get("rationale", ""),
            "decision": judge_result.get("decision", ""),
            "selected_model_ids": judge_result.get("selected_model_ids", []),
            "defer_to_market": judge_result.get("decision") == "defer_to_deterministic",
        })
        disagreement_summary = f"{disagreement_summary} Judge decision: {judge_result.get('decision', 'unknown')}."

    top = max(calibrated_dist, key=calibrated_dist.get) if calibrated_dist else "?"
    thesis = (
        f"Distribution over {len(outcomes)} outcomes; "
        f"calibrated top={top} @ {calibrated_dist.get(top, 0):.3f} "
        f"(raw {raw_dist.get(top, 0):.3f}, shrink weight {shrink_weight:.3f})."
    )
    if judge_result:
        thesis += f" Judge {judge_result.get('decision', 'unknown')} at blend weight {judge_result.get('blend_weight', 0):.2f}."
    risk_notes = []
    if packet.kalshi and packet.kalshi.spread is not None and packet.kalshi.spread > 0.08:
        risk_notes.append(f"Wide spread: {packet.kalshi.spread:.3f}.")
    if disagreement > 0.20:
        risk_notes.append("Large model disagreement; reduce size or no-trade.")
    risk_notes.extend(judge_risk_notes)

    return SupervisorForecast(
        market_ticker=packet.market_ticker,
        raw_probabilities=raw_dist,
        calibrated_probabilities=calibrated_dist,
        confidence=confidence,
        model_assessment=assessments,
        disagreement_summary=disagreement_summary,
        final_trade_thesis=thesis,
        risk_notes=risk_notes,
    )
