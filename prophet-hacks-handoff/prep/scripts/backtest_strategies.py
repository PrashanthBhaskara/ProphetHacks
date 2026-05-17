"""Replay the official Prophet Arena subset_1200 benchmark with our strategies.

The bar to beat per `STRATEGY_FINDINGS.md` (from the same hackathon team):
  - aggregate: > -$51 (vs `noisy + tight_band` baseline)
  - Sports:    > -$2

If `tight_band_skip_crypto` doesn't clear that bar, DO NOT submit it —
fall back to plain `noisy + tight_band` as a control.

Usage:
    python scripts/backtest_strategies.py
    python scripts/backtest_strategies.py --strategy tight_band
    python scripts/backtest_strategies.py --strategy default --forecaster market
    python scripts/backtest_strategies.py --out data/backtest_v1.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from ast import literal_eval
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_prophet_core.betting.strategy import (  # noqa: E402
    DefaultBettingStrategy,
    RebalancingStrategy,
)

from prep.trading.strategies import (  # noqa: E402
    CategorySkipStrategy,
    TightBandDefaultStrategy,
    TightBandStrategy,
    build_recommended_strategy,
)

DEFAULT_CSV = (
    Path(__file__).resolve().parents[1] / "data" / "external" / "subset_1200.csv"
)

# Bankroll scale: per CONSTRAINTS.md the live agent has $10k cash and
# max_per_market=$1000. `BetSignal.shares` is a fractional 0-1 target weight,
# so 1 unit of shares ≈ $100 deployed per market (matches the order of
# magnitude in STRATEGY_FINDINGS.md, where best baseline is -$51 aggregate).
DOLLARS_PER_SHARE_UNIT = 100.0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_jsonish(cell: str) -> dict | None:
    if not cell or cell == "nan":
        return None
    try:
        return json.loads(cell)
    except json.JSONDecodeError:
        try:
            return literal_eval(cell)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Synthetic forecasters (used in STRATEGY_FINDINGS for the baseline bar)
# ---------------------------------------------------------------------------


def noisy_forecast(market_mid: float, rng: random.Random, sigma: float = 0.05) -> float:
    p = market_mid + rng.gauss(0, sigma)
    return max(0.01, min(0.99, p))


def market_forecast(market_mid: float, **_: object) -> float:
    return max(0.01, min(0.99, market_mid))


def inverse_market_forecast(market_mid: float, **_: object) -> float:
    return max(0.01, min(0.99, 1.0 - market_mid))


FORECASTERS = {
    "noisy": noisy_forecast,
    "market": market_forecast,
    "inverse_market": inverse_market_forecast,
}


# ---------------------------------------------------------------------------
# Strategy factories
# ---------------------------------------------------------------------------


def build_strategy(name: str):
    if name == "tight_band":
        return TightBandStrategy()
    if name == "tight_band_skip_crypto":
        return build_recommended_strategy()
    if name == "tight_band_default":
        return TightBandDefaultStrategy()
    if name == "default":
        return DefaultBettingStrategy()
    if name == "rebalancing":
        return RebalancingStrategy()
    raise ValueError(f"unknown strategy: {name}")


STRATEGY_NAMES = (
    "tight_band",
    "tight_band_skip_crypto",
    "tight_band_default",
    "default",
    "rebalancing",
)


# ---------------------------------------------------------------------------
# Backtest core
# ---------------------------------------------------------------------------


def run_backtest(
    strategy,
    forecaster_name: str,
    csv_path: Path,
    seed: int = 42,
    skip_categories: set[str] | None = None,
) -> list[dict]:
    """Walk every (submission, market) in subset_1200 and apply the strategy.

    `skip_categories` lets the backtest filter by category at the row level,
    bypassing the strategy's own market_id-based lookup. Useful when comparing
    `tight_band` (no skip) vs `tight_band_skip_crypto` apples-to-apples.
    """
    rng = random.Random(seed)
    forecaster_fn = FORECASTERS[forecaster_name]
    skip_categories = skip_categories or set()

    trades: list[dict] = []
    rows_processed = 0
    rows_with_data = 0

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_processed += 1
            md = _parse_jsonish(row.get("market_data", "") or "")
            mo = _parse_jsonish(row.get("market_outcome", "") or "")
            if not md or not mo:
                continue
            rows_with_data += 1
            category = row.get("category") or "Other"
            if category in skip_categories:
                continue
            event_ticker = row.get("event_ticker") or ""

            for outcome_name, quote in md.items():
                if not isinstance(quote, dict):
                    continue
                yes_ask_c = quote.get("yes_ask")
                no_ask_c = quote.get("no_ask")
                if yes_ask_c is None or no_ask_c is None:
                    continue
                # CSV values are in cents (0-100); ai-prophet strategies expect 0-1
                try:
                    yes_ask = float(yes_ask_c) / 100.0
                    no_ask = float(no_ask_c) / 100.0
                except (TypeError, ValueError):
                    continue
                if not (0 < yes_ask < 1 and 0 < no_ask < 1):
                    continue

                market_mid = (yes_ask + (1.0 - no_ask)) / 2.0
                if forecaster_name == "noisy":
                    p_yes = forecaster_fn(market_mid, rng=rng)
                else:
                    p_yes = forecaster_fn(market_mid)

                # Backtest market_id encodes the series ticker so the strategy's
                # CategorySkipStrategy lookup works without changes
                market_id = f"{event_ticker}-{outcome_name}".replace(" ", "_")
                try:
                    signal = strategy.evaluate(market_id, p_yes, yes_ask, no_ask)
                except Exception as exc:
                    # Defensive — never blow up the whole backtest on one bad row
                    print(f"  [warn] {market_id} eval error: {exc}", file=sys.stderr)
                    continue
                if signal is None:
                    continue

                resolved = mo.get(outcome_name)
                if resolved is None:
                    continue
                try:
                    resolved = int(resolved)
                except (TypeError, ValueError):
                    continue

                shares = float(signal.shares)
                cost = float(signal.cost)
                if signal.side == "yes":
                    payoff = shares if resolved == 1 else 0.0
                elif signal.side == "no":
                    payoff = shares if resolved == 0 else 0.0
                else:
                    continue
                pnl = payoff - cost

                trades.append({
                    "event_ticker": event_ticker,
                    "outcome": outcome_name,
                    "category": category,
                    "side": signal.side,
                    "p_yes": round(p_yes, 4),
                    "yes_ask": round(yes_ask, 4),
                    "no_ask": round(no_ask, 4),
                    "shares": round(shares, 6),
                    "cost": round(cost, 6),
                    "resolved": resolved,
                    "pnl": round(pnl, 6),
                })

    print(
        f"  rows scanned: {rows_processed}  with-data: {rows_with_data}  "
        f"trades placed: {len(trades)}",
        file=sys.stderr,
    )
    return trades


def summarize(trades: list[dict]) -> dict:
    by_cat: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "pnl_dollars": 0.0, "cost_dollars": 0.0, "wins": 0}
    )
    total_pnl = 0.0
    total_cost = 0.0
    total_wins = 0

    for t in trades:
        pnl_d = t["pnl"] * DOLLARS_PER_SHARE_UNIT
        cost_d = t["cost"] * DOLLARS_PER_SHARE_UNIT
        cat = by_cat[t["category"]]
        cat["n"] += 1
        cat["pnl_dollars"] += pnl_d
        cat["cost_dollars"] += cost_d
        if t["pnl"] > 0:
            cat["wins"] += 1
            total_wins += 1
        total_pnl += pnl_d
        total_cost += cost_d

    return {
        "n_trades": len(trades),
        "win_rate": (total_wins / len(trades)) if trades else 0.0,
        "total_pnl_dollars": round(total_pnl, 2),
        "total_cost_dollars": round(total_cost, 2),
        "roi": round(total_pnl / total_cost, 4) if total_cost > 0 else 0.0,
        "by_category": {
            k: {
                "n": v["n"],
                "pnl_dollars": round(v["pnl_dollars"], 2),
                "cost_dollars": round(v["cost_dollars"], 2),
                "win_rate": round(v["wins"] / v["n"], 4) if v["n"] else 0.0,
            }
            for k, v in by_cat.items()
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy",
        default="tight_band_skip_crypto",
        choices=STRATEGY_NAMES,
    )
    ap.add_argument("--forecaster", default="noisy", choices=list(FORECASTERS))
    ap.add_argument("--source", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=None, help="Optional JSONL audit output")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Average across N seeds (only meaningful for forecaster=noisy). "
        "STRATEGY_FINDINGS.md uses averaging because single-seed runs vary "
        "from +$30 to -$15 aggregate.",
    )
    args = ap.parse_args()

    strategy = build_strategy(args.strategy)
    # subset_1200 event_tickers aren't in our kalshi_series_categories map
    # (they're hackathon-curated), so the strategy's market_id-based category
    # lookup misses. Filter at the row level instead when the strategy implies it.
    skip_categories = set()
    if "skip_crypto" in args.strategy:
        skip_categories.add("Crypto")
    print(
        f"Backtesting `{strategy.name}` with `{args.forecaster}` forecaster "
        f"on {args.source}"
        + (f" (row-skip: {sorted(skip_categories)})" if skip_categories else ""),
        file=sys.stderr,
    )

    seeds = (
        list(range(args.seed, args.seed + args.n_seeds))
        if args.forecaster == "noisy"
        else [args.seed]
    )
    summaries = []
    all_trades: list[dict] = []
    for seed in seeds:
        trades_i = run_backtest(
            strategy,
            args.forecaster,
            Path(args.source),
            seed,
            skip_categories=skip_categories,
        )
        summaries.append(summarize(trades_i))
        all_trades.extend(trades_i)

    # Average aggregate + per-category across seeds
    summary = summaries[0] if len(summaries) == 1 else {
        "n_trades": sum(s["n_trades"] for s in summaries) // len(summaries),
        "win_rate": sum(s["win_rate"] for s in summaries) / len(summaries),
        "total_pnl_dollars": round(
            sum(s["total_pnl_dollars"] for s in summaries) / len(summaries), 2
        ),
        "total_cost_dollars": round(
            sum(s["total_cost_dollars"] for s in summaries) / len(summaries), 2
        ),
        "roi": round(sum(s["roi"] for s in summaries) / len(summaries), 4),
        "n_seeds_averaged": len(summaries),
        "by_category": {},
    }
    if len(summaries) > 1:
        # Average per-category PnL
        cats: dict[str, list[float]] = defaultdict(list)
        ns: dict[str, list[int]] = defaultdict(list)
        for s in summaries:
            for cat, info in s["by_category"].items():
                cats[cat].append(info["pnl_dollars"])
                ns[cat].append(info["n"])
        summary["by_category"] = {
            cat: {
                "n_avg": round(sum(ns[cat]) / len(ns[cat]), 1),
                "pnl_dollars_avg": round(sum(cats[cat]) / len(cats[cat]), 2),
                "pnl_dollars_min": round(min(cats[cat]), 2),
                "pnl_dollars_max": round(max(cats[cat]), 2),
            }
            for cat in cats
        }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for t in all_trades:
                f.write(json.dumps(t) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))

    agg = summary["total_pnl_dollars"]
    sports_entry = summary["by_category"].get("Sports", {})
    sports_pnl = sports_entry.get("pnl_dollars_avg", sports_entry.get("pnl_dollars", 0.0))
    print(
        "\n=== Bar to beat (STRATEGY_FINDINGS.md): -$51 aggregate / -$2 Sports ==="
    )
    print(
        f"  aggregate: ${agg:.2f}  "
        f"{'PASS' if agg > -51 else 'FAIL'}"
    )
    print(
        f"  Sports:    ${sports_pnl:.2f}  "
        f"{'PASS' if sports_pnl > -2 else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
