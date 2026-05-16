"""Trading backtest harness.

The hackathon's trading track scores agents on actual P&L, not Brier.
A trading agent is conceptually: forecast probability -> betting strategy
-> trade -> realize P&L at market close.

This harness lets us backtest any (forecast_fn, strategy) pair against our
local eval_pack data. Each market has yes_ask / no_ask snapshots over time
and a binary outcome — perfect for simulating buy-and-hold trades.

Mirrors the contract in
  ai-prophet/packages/core/ai_prophet_core/betting/strategy.py
so a strategy written here can plug into the production engine unchanged.

Usage:
    from prep.trade import backtest, NEVER_TRADE, DEFAULT_STRATEGY
    from prep.data import load_local_snapshots

    samples = load_local_snapshots()
    result = backtest(
        samples,
        forecast_fn=lambda event, market_info: market_mid(market_info),
        strategy=DEFAULT_STRATEGY,
    )
    print(result["total_pnl"], result["sharpe"], result["win_rate"])
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence

from .data import Sample

# ---------------------------------------------------------------------------
# Strategy contract — mirrors ai_prophet_core.betting.strategy.BettingStrategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetSignal:
    """Output of a strategy for a single market. Mirrors the production BetSignal."""

    side: str          # "yes" or "no"
    shares: float      # fractional contracts to buy (0–1 scale)
    price: float       # limit price per share (0–1)
    cost: float        # shares * price


Strategy = Callable[[float, float, float], "BetSignal | None"]
"""A strategy takes (p_yes, yes_ask, no_ask) and returns a BetSignal or None."""

ForecastFn = Callable[[dict, dict], float]
"""Forecast function: (event, market_info) -> p_yes. The harness will only
call this once per market (using the latest snapshot's prices) — simulating
the production case where the agent forecasts once before placing a trade."""


# ---------------------------------------------------------------------------
# Built-in strategies — straight ports of the production strategies
# ---------------------------------------------------------------------------


def never_trade(p_yes: float, yes_ask: float, no_ask: float) -> BetSignal | None:
    """Control baseline. Always passes. P&L = 0."""
    return None


def default_strategy(
    p_yes: float,
    yes_ask: float,
    no_ask: float,
    *,
    max_spread: float = 1.05,
    min_spread: float = 0.90,
) -> BetSignal | None:
    """Port of DefaultBettingStrategy from ai-prophet.

    Skip if spread is unhealthy (>max_spread or <min_spread).
    Skip if p_yes is within the bid-ask band — no edge.
    Otherwise buy whichever side we disagree with the market on,
    sized by the magnitude of disagreement.
    """
    spread = yes_ask + no_ask
    if spread > max_spread or spread < min_spread:
        return None

    lower_bound = 1.0 - no_ask
    upper_bound = yes_ask
    if lower_bound <= p_yes <= upper_bound:
        return None

    diff = p_yes - yes_ask
    if diff > 0:
        shares = diff
        side, price = "yes", yes_ask
    elif diff < 0:
        shares = abs(diff)
        side, price = "no", no_ask
    else:
        return None

    cost = shares * price
    return BetSignal(side=side, shares=shares, price=price, cost=cost)


def rebalancing_strategy(
    p_yes: float,
    yes_ask: float,
    no_ask: float,
    *,
    max_spread: float = 1.05,
    min_spread: float = 0.90,
    min_trade: float = 0.005,
) -> BetSignal | None:
    """Port of RebalancingStrategy from ai-prophet.

    In a single-trade backtest (which we are), the rebalancing target
    `p - q` collapses to the same bet sizing as default_strategy. The
    difference only matters when you can rebalance over time. We include
    it so the team can see they're equivalent on this dataset — and pick
    rebalancing for the live agent since it handles partial fills and
    portfolio drift correctly.
    """
    spread = yes_ask + no_ask
    if spread > max_spread or spread < min_spread:
        return None

    lower_bound = 1.0 - no_ask
    upper_bound = yes_ask
    if lower_bound <= p_yes <= upper_bound:
        return None

    target = p_yes - yes_ask
    if abs(target) < min_trade:
        return None

    if target > 0:
        return BetSignal(side="yes", shares=target, price=yes_ask, cost=target * yes_ask)
    return BetSignal(side="no", shares=abs(target), price=no_ask, cost=abs(target) * no_ask)


def default_tight_band_strategy(
    p_yes: float,
    yes_ask: float,
    no_ask: float,
) -> BetSignal | None:
    """Default strategy but with a stricter spread filter [0.95, 1.02].

    Markets with wider spreads have higher transaction costs and noisier
    implied probabilities. Tightening the filter trades volume for precision.
    """
    return default_strategy(p_yes, yes_ask, no_ask, max_spread=1.02, min_spread=0.95)


def default_min_edge_strategy(
    p_yes: float,
    yes_ask: float,
    no_ask: float,
    *,
    min_edge: float = 0.05,
) -> BetSignal | None:
    """Default strategy but only fires when the disagreement is >= min_edge.

    Small disagreements are noise. Requiring a 5+ percentage-point edge
    filters out micro-trades that get eaten by spread cost.
    """
    sig = default_strategy(p_yes, yes_ask, no_ask)
    if sig is None or sig.shares < min_edge:
        return None
    return sig


def conservative_clamp_strategy(
    p_yes: float,
    yes_ask: float,
    no_ask: float,
) -> BetSignal | None:
    """Default strategy after clamping p_yes to [0.10, 0.90].

    Per the paper (§4.2.3, Fig 6), LLMs are systematically conservative
    near 0 and 1 but you sometimes get a wild over-confident prediction.
    Clamping defends against that.
    """
    p = max(0.10, min(0.90, p_yes))
    return default_strategy(p, yes_ask, no_ask)


def kelly_lite_strategy(
    p_yes: float,
    yes_ask: float,
    no_ask: float,
    *,
    fraction: float = 0.25,
    min_edge: float = 0.05,
) -> BetSignal | None:
    """Kelly-fraction sizing, capped to a fraction of full Kelly for safety.

    Bets on whichever side has positive expected value, sized as
    `fraction * edge / price`. With fraction=0.25, this is 'quarter Kelly' —
    a common cushion against probability estimation error.
    """
    if yes_ask <= 0 or no_ask <= 0 or yes_ask >= 1 or no_ask >= 1:
        return None

    yes_edge = p_yes - yes_ask          # EV per $1 of YES
    no_edge = (1 - p_yes) - no_ask      # EV per $1 of NO

    if yes_edge >= no_edge and yes_edge > min_edge:
        shares = fraction * yes_edge / yes_ask
        return BetSignal(side="yes", shares=shares, price=yes_ask, cost=shares * yes_ask)
    if no_edge > min_edge:
        shares = fraction * no_edge / no_ask
        return BetSignal(side="no", shares=shares, price=no_ask, cost=shares * no_ask)
    return None


# ---------------------------------------------------------------------------
# Forecast functions you can pair with any strategy
# ---------------------------------------------------------------------------


def market_mid_forecast(event: dict, market_info: dict) -> float:
    """Trivial forecast: just return the market midpoint as p_yes.

    Paired with any reasonable strategy this will NEVER trade (the forecast
    is always inside the bid-ask band). Useful as a sanity-check baseline
    showing zero P&L for an agent with no edge over the market."""
    ya = market_info.get("yes_ask")
    na = market_info.get("no_ask")
    if ya is not None and na is not None:
        ya = ya / 100 if ya > 1 else ya  # support cents and dollars
        na = na / 100 if na > 1 else na
        return max(0.01, min(0.99, (ya + (1 - na)) / 2))
    return 0.5


# ---------------------------------------------------------------------------
# Backtest harness
# ---------------------------------------------------------------------------


def _normalize(price: float | None) -> float | None:
    """Coerce a Kalshi price to dollar units (0–1)."""
    if price is None:
        return None
    return price / 100.0 if price > 1.0 else float(price)


def _market_prices(market_info: dict) -> tuple[float | None, float | None]:
    ya = _normalize(market_info.get("yes_ask"))
    na = _normalize(market_info.get("no_ask"))
    return ya, na


@dataclass
class TradeRecord:
    market_ticker: str
    category: str
    side: str
    shares: float
    price: float
    cost: float
    outcome: int
    payoff: float
    pnl: float


def backtest(
    samples: Sequence[Sample],
    *,
    forecast_fn: ForecastFn,
    strategy: Strategy = default_strategy,
    starting_cash: float = 10_000.0,
    contracts_per_unit: float = 100.0,
) -> dict[str, Any]:
    """Simulate buy-and-hold trades on resolved markets.

    For each sample with valid prices, ask the forecast for p_yes, ask the
    strategy whether to bet, simulate the buy at the ask price, hold to
    resolution, and compute P&L. No bid-side execution (no short sells)
    matches the production engine's buy-only contract.

    `contracts_per_unit` translates fractional `shares` (0–1) into dollar
    sizing. Default 100 means each unit of `shares` represents $100 — so a
    `shares=0.3` bet at `price=0.55` costs $16.50 and pays $30 if it hits.

    Returns a dict with total P&L, win rate, Sharpe, per-category breakdown,
    and the full per-trade record (for further analysis).
    """
    trades: list[TradeRecord] = []
    cash = starting_cash
    skipped_no_price = 0
    skipped_strategy = 0
    insufficient_cash = 0

    for s in samples:
        ya, na = _market_prices(s.market_info)
        if ya is None or na is None:
            skipped_no_price += 1
            continue

        try:
            p_yes = float(forecast_fn(s.event, s.market_info))
        except Exception:
            skipped_no_price += 1
            continue
        p_yes = max(0.01, min(0.99, p_yes))

        sig = strategy(p_yes, ya, na)
        if sig is None:
            skipped_strategy += 1
            continue

        # Scale the fractional bet into dollar cost.
        dollar_cost = sig.cost * contracts_per_unit
        if dollar_cost > cash:
            # Skip rather than partial-fill — matches a conservative production agent.
            insufficient_cash += 1
            continue

        # Realize P&L at resolution.
        won = (sig.side == "yes" and s.outcome == 1) or (sig.side == "no" and s.outcome == 0)
        payoff = sig.shares * contracts_per_unit * (1.0 if won else 0.0)
        pnl = payoff - dollar_cost
        cash += pnl  # equivalent to cash -= dollar_cost; cash += payoff

        trades.append(TradeRecord(
            market_ticker=s.event["market_ticker"],
            category=s.event.get("category", "Unknown"),
            side=sig.side,
            shares=sig.shares,
            price=sig.price,
            cost=dollar_cost,
            outcome=s.outcome,
            payoff=payoff,
            pnl=pnl,
        ))

    pnls = [t.pnl for t in trades]
    total_pnl = sum(pnls)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    n_trades = len(trades)

    # Sharpe-lite: mean / std of per-trade P&L, annualized for context.
    if n_trades >= 2:
        mean = total_pnl / n_trades
        var = sum((p - mean) ** 2 for p in pnls) / (n_trades - 1)
        std = math.sqrt(var)
        sharpe = mean / std if std > 0 else float("nan")
    else:
        sharpe = float("nan")

    # Per-category breakdown.
    by_cat: dict[str, dict[str, float]] = {}
    for t in trades:
        c = by_cat.setdefault(t.category, {"n": 0, "pnl": 0.0, "wins": 0})
        c["n"] += 1
        c["pnl"] += t.pnl
        c["wins"] += 1 if t.pnl > 0 else 0

    return {
        "starting_cash": starting_cash,
        "ending_cash": cash,
        "total_pnl": total_pnl,
        "return_pct": total_pnl / starting_cash if starting_cash else float("nan"),
        "n_samples": len(samples),
        "n_trades": n_trades,
        "skipped_strategy": skipped_strategy,
        "skipped_no_price": skipped_no_price,
        "insufficient_cash": insufficient_cash,
        "win_rate": wins / n_trades if n_trades else float("nan"),
        "wins": wins,
        "losses": losses,
        "sharpe_per_trade": sharpe,
        "by_category": by_cat,
        "trades": trades,
    }
