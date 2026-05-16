"""Backtest trading strategies against the local eval_pack data.

Trading mirror of scripts/run.py — but instead of scoring Brier on a
predict function, simulates P&L from (forecast_fn, strategy) pairs.

Usage:
    python scripts/run_trade.py never_trade
    python scripts/run_trade.py market_anchor              # default+market — should never trade
    python scripts/run_trade.py noisy_market               # default+market+noise — what random "edge" earns
    python scripts/run_trade.py noisy_market --strategy kelly
    python scripts/run_trade.py noisy_market --category Sports
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.data import filter_by_category, load_local_snapshots  # noqa: E402
from prep.trade import (  # noqa: E402
    backtest,
    default_strategy,
    kelly_lite_strategy,
    market_mid_forecast,
    never_trade,
)


def market_anchor_forecast(event, market_info):
    return market_mid_forecast(event, market_info)


def noisy_market_forecast(event, market_info, sigma: float = 0.10, rng=None):
    """Market mid + Gaussian noise. Simulates an LLM that's directionally
    correct ~half the time but has no real edge."""
    rng = rng or random.Random(42 + hash(event["market_ticker"]) % 10_000)
    p = market_mid_forecast(event, market_info)
    return max(0.01, min(0.99, p + rng.gauss(0, sigma)))


FORECASTS = {
    "never_trade": None,                # uses never_trade strategy regardless
    "market_anchor": market_anchor_forecast,
    "noisy_market": noisy_market_forecast,
}

STRATEGIES = {
    "default": default_strategy,
    "kelly": kelly_lite_strategy,
    "never": never_trade,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("forecast", choices=FORECASTS.keys())
    parser.add_argument("--strategy", choices=STRATEGIES.keys(), default="default")
    parser.add_argument("--category", default=None)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    samples = load_local_snapshots()
    if args.category:
        samples = filter_by_category(samples, args.category)
    if args.limit:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} samples"
          + (f" (category={args.category})" if args.category else ""))

    if args.forecast == "never_trade":
        strategy = never_trade
        forecast = market_anchor_forecast  # placeholder
    else:
        strategy = STRATEGIES[args.strategy]
        forecast = FORECASTS[args.forecast]

    r = backtest(
        samples,
        forecast_fn=forecast,
        strategy=strategy,
        starting_cash=args.starting_cash,
    )

    print()
    print(f"Forecast: {args.forecast}   Strategy: {args.strategy}")
    print(f"Starting cash: ${r['starting_cash']:,.2f}")
    print(f"Ending cash:   ${r['ending_cash']:,.2f}")
    print(f"Total P&L:     ${r['total_pnl']:+,.2f}   ({r['return_pct']*100:+.2f}%)")
    print(f"Trades:        {r['n_trades']:,}  "
          f"(skipped: strategy={r['skipped_strategy']:,}, no_price={r['skipped_no_price']:,})")
    if r['n_trades']:
        print(f"Win rate:      {r['win_rate']*100:.1f}%   "
              f"(W {r['wins']:,} / L {r['losses']:,})")
        print(f"Sharpe-per-trade: {r['sharpe_per_trade']:.3f}")
    if r["by_category"]:
        print()
        print("Per-category:")
        print(f"  {'category':<12} {'trades':>8} {'P&L':>12} {'win%':>8}")
        for cat, d in sorted(r["by_category"].items(), key=lambda x: -x[1]["pnl"]):
            wr = d["wins"] / d["n"] * 100 if d["n"] else 0
            print(f"  {cat:<12} {int(d['n']):>8,} ${d['pnl']:>10,.2f} {wr:>7.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
