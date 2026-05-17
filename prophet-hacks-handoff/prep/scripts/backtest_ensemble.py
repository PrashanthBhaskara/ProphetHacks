"""Run an ensemble forecasting backtest.

The default config uses mock forecasters, so this script runs without API keys.
Enable Gemini/OpenRouter lanes in config/ensemble.example.json once keys are
available in your environment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.calibration import CalibrationConfig  # noqa: E402
from prep.data import filter_by_category, load_eval_pack, load_subset_100  # noqa: E402
from prep.ensemble import JudgeConfig, aggregate_forecasts, forecast_members_parallel  # noqa: E402
from prep.forecasters import ForecasterConfig  # noqa: E402
from prep.packets import packet_from_sample  # noqa: E402
from prep.score import full_report  # noqa: E402
from prep.store import JsonlStore  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "ensemble.example.json"


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def _json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", choices=("eval_pack", "hf"), default="eval_pack")
    parser.add_argument("--snapshot", choices=("latest", "first"), default="latest")
    parser.add_argument("--category", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None, help="optional JSONL audit output")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    models = [ForecasterConfig.from_dict(m) for m in cfg.get("models", []) if m.get("enabled", True)]
    calibration = CalibrationConfig.from_dict(cfg.get("calibration"))
    ensemble_cfg = cfg.get("ensemble", {})
    market_anchor_weight = float(ensemble_cfg.get("market_anchor_weight", 1.5))
    judge = JudgeConfig.from_dict(ensemble_cfg.get("judge"))

    samples = load_eval_pack(snapshot=args.snapshot) if args.source == "eval_pack" else load_subset_100()
    if args.category:
        samples = filter_by_category(samples, args.category)
    if args.limit:
        samples = samples[:args.limit]

    print(f"Loaded {len(samples)} samples from {args.source}")
    print(f"Enabled models: {', '.join(m.name for m in models)}")

    store = JsonlStore(args.out) if args.out else None
    rows = []
    p_yes = []
    outcomes = []
    market_q = []
    for idx, sample in enumerate(samples, 1):
        packet = packet_from_sample(sample)
        run = forecast_members_parallel(
            models,
            packet,
            continue_on_error=args.continue_on_error,
        )
        for error in run.errors:
            print(f"[warn] {packet.market_ticker} {error}", file=sys.stderr)
        members = run.members
        supervisor = aggregate_forecasts(
            packet,
            members,
            calibration=calibration,
            market_anchor_weight=market_anchor_weight,
            judge=judge,
        )

        p_yes.append(supervisor.calibrated_p_yes)
        outcomes.append(int(sample.outcome))
        market_q.append(packet.kalshi.market_mid)

        row = {
            "packet": packet.to_dict(),
            "outcome": sample.outcome,
            "model_forecasts": [member.forecast.to_dict() for member in members],
            "model_errors": run.errors,
            "supervisor": supervisor.to_dict(),
            "score_inputs": {
                "p_yes": supervisor.calibrated_p_yes,
                "outcome": int(sample.outcome),
                "market_q": packet.kalshi.market_mid,
            },
        }
        rows.append(row)
        if store:
            store.append(row)
        if idx % max(1, len(samples) // 10) == 0 or idx == len(samples):
            print(f"  {idx}/{len(samples)}")

    print(json.dumps(_json_safe(full_report(p_yes, outcomes, market_q=market_q)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
