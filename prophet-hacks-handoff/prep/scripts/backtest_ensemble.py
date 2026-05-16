"""Run an ensemble forecast/trading backtest.

The default config uses mock forecasters, so this script runs without API keys.
Enable Gemini/OpenRouter lanes in config/ensemble.example.json once keys are
available in your environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.calibration import CalibrationConfig  # noqa: E402
from prep.data import filter_by_category, load_eval_pack, load_subset_100  # noqa: E402
from prep.ensemble import EnsembleMember, aggregate_forecasts  # noqa: E402
from prep.forecasters import ForecasterConfig, forecast_from_config  # noqa: E402
from prep.packets import packet_from_sample  # noqa: E402
from prep.store import JsonlStore  # noqa: E402
from prep.trading.metrics import summarize_trades  # noqa: E402
from prep.trading.risk import RiskConfig, decide_trade  # noqa: E402
from prep.trading.simulator import simulate_trade  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "ensemble.example.json"


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text())


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
    risk = RiskConfig.from_dict(cfg.get("risk"))
    market_anchor_weight = float(cfg.get("ensemble", {}).get("market_anchor_weight", 1.5))
    fee_rate = float(cfg.get("execution", {}).get("fee_rate", 0.0))

    samples = load_eval_pack(snapshot=args.snapshot) if args.source == "eval_pack" else load_subset_100()
    if args.category:
        samples = filter_by_category(samples, args.category)
    if args.limit:
        samples = samples[:args.limit]

    print(f"Loaded {len(samples)} samples from {args.source}")
    print(f"Enabled models: {', '.join(m.name for m in models)}")

    store = JsonlStore(args.out) if args.out else None
    rows = []
    for idx, sample in enumerate(samples, 1):
        packet = packet_from_sample(sample)
        model_forecasts = []
        for model in models:
            try:
                forecast = forecast_from_config(model, packet)
                model_forecasts.append((model, forecast))
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"[warn] {packet.market_ticker} {model.name} failed: {exc}", file=sys.stderr)

        members = [
            EnsembleMember(forecast=forecast, configured_weight=model.weight)
            for model, forecast in model_forecasts
        ]
        supervisor = aggregate_forecasts(
            packet,
            members,
            calibration=calibration,
            market_anchor_weight=market_anchor_weight,
        )
        decision = decide_trade(packet, supervisor, risk)
        result = simulate_trade(decision, sample.outcome, fee_rate=fee_rate)

        row = {
            "packet": packet.to_dict(),
            "outcome": sample.outcome,
            "model_forecasts": [forecast.to_dict() for _, forecast in model_forecasts],
            "supervisor": supervisor.to_dict(),
            "decision": decision.to_dict(),
            "result": result.to_dict(),
        }
        rows.append(row)
        if store:
            store.append(row)
        if idx % max(1, len(samples) // 10) == 0 or idx == len(samples):
            print(f"  {idx}/{len(samples)}")

    print(json.dumps(summarize_trades(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
