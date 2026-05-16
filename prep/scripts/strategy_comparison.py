"""Run every available trading strategy against the CLEAN backtest data
(markets we caught open with >=2 snapshots, using the earliest snapshot's
prices). Prints one comparison table per category.

This is the script to look at when deciding which betting strategy to
ship in the live trading agent. The numbers here are more realistic than
the headline summary.md baselines because they exclude the backfill data
that contaminates prices toward outcomes.

Usage:
    python scripts/strategy_comparison.py
    python scripts/strategy_comparison.py --category Sports
    python scripts/strategy_comparison.py --min-snapshots 3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.data import filter_by_category, load_clean_eval_set, load_hf_eval_set, load_subset_1200  # noqa: E402
from prep.trade import (  # noqa: E402
    backtest,
    conservative_clamp_strategy,
    default_min_edge_strategy,
    default_strategy,
    default_tight_band_strategy,
    kelly_lite_strategy,
    market_mid_forecast,
    never_trade,
    rebalancing_strategy,
)


def _market(event, market_info):
    return market_mid_forecast(event, market_info)


def _noisy_market(event, market_info, *, sigma=0.10):
    rng = random.Random(42 + hash(event["market_ticker"]) % 10_000)
    p = market_mid_forecast(event, market_info)
    return max(0.01, min(0.99, p + rng.gauss(0, sigma)))


def _confident_noisy(event, market_info):
    # Larger noise — simulates a model that disagrees more strongly with market
    return _noisy_market(event, market_info, sigma=0.15)


def _inverse_market(event, market_info):
    # Sanity check: betting AGAINST the market price. Should lose money
    # systematically (proves direction matters).
    return 1.0 - market_mid_forecast(event, market_info)


FORECASTERS = {
    "market": _market,
    "noisy (sigma=0.10)": _noisy_market,
    "confident_noisy (sigma=0.15)": _confident_noisy,
    "inverse_market": _inverse_market,
}

STRATEGIES = {
    "default": default_strategy,
    "rebalancing": rebalancing_strategy,
    "tight_band (spread 0.95-1.02)": default_tight_band_strategy,
    "min_edge=0.05": default_min_edge_strategy,
    "clamp_p_to_[0.1,0.9]": conservative_clamp_strategy,
    "kelly_lite (qtr Kelly)": kelly_lite_strategy,
}


def _print_row(label, r):
    pnl = r["total_pnl"]
    ret = pnl / r["starting_cash"] * 100 if r["starting_cash"] else 0
    wr = r["win_rate"] * 100 if r["n_trades"] else 0
    sh = r["sharpe_per_trade"]
    sh_s = f"{sh:+.3f}" if r["n_trades"] >= 2 else "  n/a"
    print(f"  {label:42s} {r['n_trades']:>6,}  ${pnl:>+9,.2f}  {ret:>+6.2f}%  {wr:>5.1f}%  {sh_s}")


def _run_table(samples, label):
    print()
    print(f"=== {label}  (N={len(samples):,}) ===")
    print(f"  {'Forecaster + Strategy':42s} {'Trades':>6}  {'P&L':>10}  {'Return':>7}  {'Win%':>5}  Sharpe")
    print(f"  {'-'*42} {'-'*6}  {'-'*10}  {'-'*7}  {'-'*5}  ------")

    # Always include the control
    r = backtest(samples, forecast_fn=_market, strategy=never_trade)
    _print_row("never_trade (control)", r)

    for fc_name, fc in FORECASTERS.items():
        for st_name, st in STRATEGIES.items():
            r = backtest(samples, forecast_fn=fc, strategy=st)
            _print_row(f"{fc_name}  +  {st_name}", r)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None)
    parser.add_argument("--min-snapshots", type=int, default=2,
                        help="only use markets with >= N snapshots (default 2 = exclude single-snapshot backfill)")
    parser.add_argument("--source", choices=("local", "hf", "official_1200"), default="official_1200",
                        help="official_1200 = PA Subset 1200 (THE authoritative benchmark); "
                             "hf = thomaswmitch trades (May–Jul 2025, granular); "
                             "local = our self-polled snapshots")
    args = parser.parse_args()

    if args.source == "official_1200":
        samples = load_subset_1200()
    elif args.source == "hf":
        samples = load_hf_eval_set(min_snapshots=args.min_snapshots)
    else:
        samples = load_clean_eval_set(min_snapshots=args.min_snapshots)
    if not samples:
        print("No samples — make sure consolidate.py has been run.")
        return 1

    if args.category:
        samples = filter_by_category(samples, args.category)
        _run_table(samples, f"{args.category} only")
    else:
        _run_table(samples, "All categories")
        for cat in ("Sports", "Crypto", "Other"):
            sub = filter_by_category(samples, cat)
            if len(sub) >= 100:
                _run_table(sub, cat)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
