"""Small real-LLM tester for resolved point-in-time samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import brier, load_samples
from .config import load_config, load_local_env, resolve_api_key
from .data_loaders import load_nonbinary_component_samples, load_topvol_samples, load_unified_binary_samples
from .experiments import point_in_time_samples, random_point_in_time_samples
from .features import build_feature_packet, parse_dt
from .forecaster import forecast_event
from .pit_evidence import gather_pit_external_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["live_clean", "eval_pack", "topvol", "nonbinary", "unified"], default="topvol")
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--candle-stride-minutes", type=int, default=1)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--chronological-oos", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--gated-only", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--with-supervisor", action="store_true")
    parser.add_argument(
        "--force-cheap",
        dest="force_cheap",
        action="store_true",
        default=True,
        help="Call the configured OpenRouter LLM for every selected sample. This is now the default.",
    )
    parser.add_argument(
        "--respect-gates",
        dest="force_cheap",
        action="store_false",
        help="Use legacy statistical gates instead of forcing a GPT call for each selected sample.",
    )
    parser.add_argument(
        "--pit-external-evidence",
        action="store_true",
        help="Attach point-in-time external evidence from local archives and live-only source adapters.",
    )
    parser.add_argument(
        "--pit-allow-network",
        action="store_true",
        help="Allow live PIT evidence network fetches. Historical as_of values still require explicit archive env flags.",
    )
    parser.add_argument(
        "--pit-nonstrict-collected-at",
        action="store_true",
        help="Allow records collected after as_of if their published_at timestamp is before as_of.",
    )
    parser.add_argument("--since-close")
    parser.add_argument("--until-close")
    parser.add_argument("--random-as-of", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260517)
    parser.add_argument("--min-horizon-minutes", type=float, default=5.0)
    parser.add_argument("--max-horizon-hours", type=float)
    parser.add_argument("--min-history-snapshots", type=int, default=5)
    parser.add_argument("--decision-budget-minutes", type=float, default=5.0)
    parser.add_argument("--model", help="Override the cheap-lane OpenRouter model for this test run.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    load_local_env()
    cfg = load_config()
    if args.model:
        cfg.cheap_model.model = args.model
    if not args.with_supervisor:
        cfg.supervisor_model.enabled = False
    cheap_key, _ = resolve_api_key(cfg.cheap_model)
    if not cheap_key:
        raise SystemExit(
            f"No OpenRouter key found for {cfg.cheap_model.api_key_env} "
            f"or {cfg.cheap_model.api_key_fallback_envs}"
        )

    if args.source == "topvol":
        raw_samples = load_topvol_samples(candle_stride_minutes=args.candle_stride_minutes)
    elif args.source == "nonbinary":
        raw_samples = load_nonbinary_component_samples(candle_stride_minutes=args.candle_stride_minutes)
    elif args.source == "unified":
        raw_samples = load_unified_binary_samples(candle_stride_minutes=args.candle_stride_minutes)
    else:
        raw_samples = load_samples(args.source, None)
    samples = _filter_by_close_time(raw_samples, since_close=args.since_close, until_close=args.until_close)
    samples.sort(key=lambda sample: (
        _sample_close(sample) if args.random_as_of else _sample_as_of(sample),
        sample.event.get("market_ticker") or sample.market_info.get("ticker") or "",
    ))
    if args.chronological_oos:
        split = max(1, min(len(samples) - 1, int(len(samples) * args.train_fraction)))
        samples = samples[split:]
    if args.random_as_of:
        samples = random_point_in_time_samples(
            samples,
            seed=args.random_seed,
            min_horizon_minutes=args.min_horizon_minutes,
            max_horizon_hours=args.max_horizon_hours,
            min_history_snapshots=args.min_history_snapshots,
            decision_budget_minutes=args.decision_budget_minutes,
        )
    else:
        samples = point_in_time_samples(samples, horizon_hours=args.horizon_hours)
        samples.sort(key=lambda sample: (
            _sample_as_of(sample),
            sample.event.get("market_ticker") or sample.market_info.get("ticker") or "",
        ))
    candidates = samples[args.offset: args.offset + args.max_candidates]
    selected = []
    for sample in candidates:
        if not args.gated_only:
            selected.append(sample)
        else:
            external_evidence = _external_evidence_for_sample(sample, cfg, args)
            dry = forecast_event(
                sample.event,
                sample.market_info,
                price_trajectory=sample.snapshots,
                external_evidence=external_evidence,
                dry_run=True,
                config=cfg,
                force_cheap=args.force_cheap,
            )
            if dry.audit_summary.get("gates", {}).get("call_cheap"):
                selected.append(sample)
        if len(selected) >= args.limit:
            break
    rows = []
    preds = []
    outcomes = []
    for sample in selected:
        external_evidence = _external_evidence_for_sample(sample, cfg, args)
        decision = forecast_event(
            sample.event,
            sample.market_info,
            price_trajectory=sample.snapshots,
            external_evidence=external_evidence,
            dry_run=False,
            config=cfg,
            force_cheap=args.force_cheap,
        )
        p_yes = float(decision.probabilities.get("YES", 0.5))
        preds.append(p_yes)
        outcomes.append(sample.outcome)
        rows.append({
            "ticker": sample.event.get("market_ticker") or sample.market_info.get("ticker"),
            "outcome": sample.outcome,
            "p_yes": p_yes,
            "source": decision.source,
            "trade_recommendation": decision.trade_recommendation,
            "external_evidence_count": len(external_evidence),
            "audit": decision.audit_summary,
        })

    result = {
        "source": args.source,
        "horizon_hours": args.horizon_hours,
        "random_as_of": args.random_as_of,
        "random_seed": args.random_seed if args.random_as_of else None,
        "min_horizon_minutes": args.min_horizon_minutes if args.random_as_of else None,
        "max_horizon_hours": args.max_horizon_hours if args.random_as_of else None,
        "min_history_snapshots": args.min_history_snapshots if args.random_as_of else None,
        "decision_budget_minutes": args.decision_budget_minutes if args.random_as_of else None,
        "since_close": args.since_close,
        "until_close": args.until_close,
        "model": cfg.cheap_model.model,
        "force_cheap": args.force_cheap,
        "pit_external_evidence": args.pit_external_evidence,
        "n": len(rows),
        "brier": brier(preds, outcomes),
        "rows": rows,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _sample_as_of(sample) -> float:
    if sample.snapshots:
        value = sample.snapshots[-1].get("t") or sample.snapshots[-1].get("snapshot_time")
    else:
        value = sample.market_info.get("snapshot_time")
    parsed = parse_dt(value)
    return parsed.timestamp() if parsed is not None else 0.0


def _sample_close(sample) -> float:
    parsed = parse_dt(sample.event.get("close_time") or sample.market_info.get("close_time"))
    return parsed.timestamp() if parsed is not None else 0.0


def _filter_by_close_time(samples, *, since_close: str | None, until_close: str | None):
    if not since_close and not until_close:
        return samples
    since_dt = parse_dt(since_close) if since_close else None
    until_dt = parse_dt(until_close) if until_close else None
    out = []
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


def _external_evidence_for_sample(sample, cfg, args) -> list[dict]:
    if not args.pit_external_evidence:
        return []
    packet = build_feature_packet(sample.event, sample.market_info, price_trajectory=sample.snapshots)
    return gather_pit_external_evidence(
        packet,
        cfg,
        enabled=True,
        allow_network=args.pit_allow_network,
        strict_collected_at=not args.pit_nonstrict_collected_at,
    )


if __name__ == "__main__":
    raise SystemExit(main())
