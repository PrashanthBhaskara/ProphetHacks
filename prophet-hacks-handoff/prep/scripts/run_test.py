"""Run test events through the full ensemble and print per-model predictions + token usage.

Usage:
    python scripts/run_test.py --events /path/to/test.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_test")


def _load_env() -> None:
    """Load key=value pairs from .env files, searching up from the script."""
    candidates = [
        SCRIPTS_DIR.parent / ".env",
        SCRIPTS_DIR.parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        logger.info("Loaded env from %s", env_path)
        break


_load_env()

from agent_server import _load_models, _enrich_packet, ArenaEvent  # noqa: E402
from prep.ensemble import forecast_members_parallel, aggregate_forecasts  # noqa: E402
from prep.packets import packet_from_arena_event  # noqa: E402


def _token_usage(raw: dict) -> str:
    """Extract token counts from an OpenRouter/Grok raw API response."""
    usage = raw.get("usage") or {}
    inp = usage.get("prompt_tokens")
    out = usage.get("completion_tokens")
    cost = usage.get("cost")
    parts = []
    if inp is not None:
        parts.append(f"in={inp}")
    if out is not None:
        parts.append(f"out={out}")
    if cost is not None:
        parts.append(f"cost=${float(cost):.4f}")
    return " ".join(parts) if parts else "n/a"


def _load_events(path: Path) -> list[dict]:
    text = path.read_text().strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else [parsed]


def run_event(event: dict, models, calibration, market_anchor_weight, judge) -> None:
    ticker = event.get("market_ticker") or event.get("event_ticker") or "?"
    print(f"\n{'='*60}")
    print(f"EVENT: {ticker}")
    print(f"  title:    {event.get('title')}")
    print(f"  outcomes: {event.get('outcomes')}")
    print(f"  category: {event.get('category')}")
    print()

    packet = packet_from_arena_event(event)
    packet = _enrich_packet(packet)
    run = forecast_members_parallel(models, packet, continue_on_error=True)

    if run.errors:
        for err in run.errors:
            logger.warning("lane error: %s", err)

    print("── Per-model predictions ──────────────────────────────────")
    for member in run.members:
        fc = member.forecast
        probs = fc.probabilities
        raw = fc.raw_response or {}
        tokens = _token_usage(raw)
        prob_str = "  ".join(f"{k}={v:.3f}" for k, v in probs.items())
        rt = fc.reasoning_track
        diag = fc.diagnostics
        print(f"  [{fc.provider}/{fc.model_id}]")
        print(f"    probs:        {prob_str}")
        print(f"    conf:         {fc.forecast.confidence:.2f}  uncertainty: {fc.forecast.uncertainty:.2f}")
        print(f"    tokens:       {tokens}")
        print(f"    summary:      {rt.summary}")
        print(f"    base_rate:    {rt.base_rate}")
        print(f"    market_analysis: {rt.market_analysis}")
        if rt.key_evidence:
            print("    key_evidence:")
            for ev in rt.key_evidence:
                claim = ev.get("claim", "")
                src = ev.get("source", "")
                impact = ev.get("impact", "")
                print(f"      - {claim} [{src}] {impact}")
        if rt.counterarguments:
            print("    counterarguments:")
            for ca in rt.counterarguments:
                claim = ca.get("claim", ca) if isinstance(ca, dict) else ca
                print(f"      - {claim}")
        if rt.assumptions:
            print("    assumptions:")
            for a in rt.assumptions:
                print(f"      - {a}")
        if rt.information_gaps:
            print("    information_gaps:")
            for g in rt.information_gaps:
                print(f"      - {g}")
        print(f"    diagnostics:  quality={diag.evidence_quality}  clarity={diag.rules_clarity}  defer={diag.should_defer_to_market}")
        print()

    if not run.members:
        print("  (all lanes failed — no model predictions)")
        return

    supervisor = aggregate_forecasts(
        packet,
        run.members,
        calibration=calibration,
        market_anchor_weight=market_anchor_weight,
        judge=judge,
    )

    print("── Ensemble result ────────────────────────────────────────")
    for outcome, p in supervisor.calibrated_probabilities.items():
        raw_p = supervisor.raw_probabilities.get(outcome, 0.0)
        print(f"  {outcome}: calibrated={p:.3f}  raw={raw_p:.3f}")
    print(f"  confidence:  {supervisor.confidence:.3f}")
    print(f"  disagreement: {supervisor.disagreement_summary}")
    print(f"  thesis: {supervisor.final_trade_thesis}")
    if supervisor.risk_notes:
        for note in supervisor.risk_notes:
            print(f"  [risk] {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True,
                        help="Path to events JSON or JSONL.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    events = _load_events(args.events)
    if args.limit:
        events = events[: args.limit]
    if not events:
        print("No events found.", file=sys.stderr)
        return 1

    models, calibration, market_anchor_weight, judge = _load_models()
    logger.info("Loaded %d lane(s): %s", len(models), [m.name for m in models])

    for event in events:
        run_event(event, models, calibration, market_anchor_weight, judge)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
