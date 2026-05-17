"""Trading-track strategies — built on top of ai-prophet's BettingStrategy API.

Per `STRATEGY_FINDINGS.md` (backtested against the official `subset_1200`
benchmark), the winning combination is `RebalancingStrategy(max_spread=1.02)`
wrapped with a Crypto-skip filter. This module makes those choices canonical.

Use `build_recommended_strategy()` to get the version that should go into the
live trading run; use individual classes when you want to A/B test variants.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from ai_prophet_core.betting.strategy import (
    BetSignal,
    BettingStrategy,
    DefaultBettingStrategy,
    PortfolioSnapshot,
    RebalancingStrategy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables — fixed by STRATEGY_FINDINGS.md
# ---------------------------------------------------------------------------

TIGHT_BAND_MAX_SPREAD = 1.02  # vs ai-prophet's default 1.03

# Kalshi category names we skip outright per the backtest findings
# ("market is calibrated; no edge available, expected loss").
DEFAULT_SKIP_CATEGORIES: tuple[str, ...] = ("Crypto",)

# Path to the authoritative series→category map (committed in PR #5).
_SERIES_CATEGORIES_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "kalshi_series_categories.json"
)

_series_category_cache: dict[str, str] | None = None


def _load_series_categories() -> dict[str, str]:
    """Lazy-load the authoritative ticker→category lookup table.

    Format: `{"KXBTC": "Crypto", "KXNBAGAME": "Sports", ...}` — 10,168 entries.
    """
    global _series_category_cache
    if _series_category_cache is not None:
        return _series_category_cache
    if not _SERIES_CATEGORIES_PATH.exists():
        logger.warning(
            "kalshi_series_categories.json not found at %s; category-based "
            "filtering disabled (all markets will pass through).",
            _SERIES_CATEGORIES_PATH,
        )
        _series_category_cache = {}
        return _series_category_cache
    _series_category_cache = json.loads(_SERIES_CATEGORIES_PATH.read_text())
    return _series_category_cache


def get_market_category(market_id: str) -> str | None:
    """Look up the Kalshi category for a market_id by its series-ticker prefix.

    `market_id` formats handled: `"kalshi:KXBTC-26MAY16"`, `"KXBTC-26MAY16"`,
    `"KXBTC"`. Returns None if not found.
    """
    series_map = _load_series_categories()
    if not series_map:
        return None
    ticker = market_id.split(":")[-1]
    series = ticker.split("-")[0]
    return series_map.get(series)


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------


class TightBandStrategy(RebalancingStrategy):
    """`RebalancingStrategy` with the tight 1.02 spread filter.

    STRATEGY_FINDINGS.md: this is the universal winner across forecasters
    on the official `subset_1200` benchmark. `RebalancingStrategy` is
    preferred over `DefaultBettingStrategy` for live agents because it
    handles partial fills + portfolio drift correctly across ticks.
    """

    name = "tight_band_rebalancing"

    def __init__(self) -> None:
        super().__init__(max_spread=TIGHT_BAND_MAX_SPREAD)


class TightBandDefaultStrategy(DefaultBettingStrategy):
    """`DefaultBettingStrategy` with the tight 1.02 spread filter.

    Single-tick variant (no portfolio drift handling). Use this for
    backtests on snapshot data where there's no notion of position-over-time.
    """

    name = "tight_band_default"

    def __init__(self) -> None:
        super().__init__(max_spread=TIGHT_BAND_MAX_SPREAD)


class CategorySkipStrategy(BettingStrategy):
    """Wrap another strategy; skip markets whose category is in `skip_categories`.

    Uses `kalshi_series_categories.json` for the lookup, so it works for any
    Kalshi market_id passed by the runner (no need to thread category through
    the BettingStrategy contract).

    For the backtest path where category is already in the row, callers can
    bypass this and filter at the row level directly — see
    `scripts/backtest_strategies.py`.
    """

    def __init__(
        self,
        inner: BettingStrategy,
        skip_categories: Iterable[str] = DEFAULT_SKIP_CATEGORIES,
    ) -> None:
        self._inner = inner
        self.skip_categories = set(skip_categories)
        self.name = (
            f"{inner.name}_skip_{'_'.join(sorted(self.skip_categories)).lower()}"
        )

    @property
    def portfolio(self) -> PortfolioSnapshot | None:
        return self._inner.portfolio

    def __setattr__(self, name: str, value: object) -> None:
        # The engine sets `_portfolio` via attribute access on the strategy.
        # Delegate to the inner strategy so its evaluate() sees up-to-date state.
        if name == "_portfolio":
            object.__setattr__(self._inner, "_portfolio", value)
            return
        object.__setattr__(self, name, value)

    def evaluate(
        self,
        market_id: str,
        p_yes: float,
        yes_ask: float,
        no_ask: float,
    ) -> BetSignal | None:
        category = get_market_category(market_id)
        if category in self.skip_categories:
            return None
        return self._inner.evaluate(market_id, p_yes, yes_ask, no_ask)


def build_recommended_strategy() -> BettingStrategy:
    """The strategy you should actually run in the live eval.

    Per STRATEGY_FINDINGS.md:
        RebalancingStrategy(max_spread=1.02) + skip Crypto
    """
    return CategorySkipStrategy(
        TightBandStrategy(),
        skip_categories=DEFAULT_SKIP_CATEGORIES,
    )
