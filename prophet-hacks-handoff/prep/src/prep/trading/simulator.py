"""Conservative taker-fill simulator."""

from __future__ import annotations

from dataclasses import dataclass

from .risk import TradeDecision


@dataclass
class BacktestResult:
    pnl: float
    gross_return: float
    stake: float
    won: bool | None

    def to_dict(self) -> dict:
        return {
            "pnl": self.pnl,
            "gross_return": self.gross_return,
            "stake": self.stake,
            "won": self.won,
        }


def simulate_trade(decision: TradeDecision, outcome: int, *, fee_rate: float = 0.0) -> BacktestResult:
    if decision.side == "NONE" or not decision.price or decision.stake <= 0:
        return BacktestResult(pnl=0.0, gross_return=0.0, stake=0.0, won=None)

    won = (decision.side == "YES" and outcome == 1) or (decision.side == "NO" and outcome == 0)
    shares = decision.stake / decision.price
    payoff = shares if won else 0.0
    fees = decision.stake * fee_rate
    pnl = payoff - decision.stake - fees
    return BacktestResult(pnl=pnl, gross_return=payoff, stake=decision.stake, won=won)
