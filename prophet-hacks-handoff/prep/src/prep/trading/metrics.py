"""Portfolio metrics for trading backtests."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def summarize_trades(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    traded = [r for r in rows if r["decision"]["side"] != "NONE"]
    pnl = [r["result"]["pnl"] for r in traded]
    stakes = [r["result"]["stake"] for r in traded]
    wins = [r["result"]["won"] for r in traded if r["result"]["won"] is not None]
    total_pnl = sum(pnl)
    total_stake = sum(stakes)
    avg = total_pnl / len(traded) if traded else 0.0
    var = sum((x - avg) ** 2 for x in pnl) / len(pnl) if pnl else 0.0

    by_category = defaultdict(lambda: {"n": 0, "trades": 0, "pnl": 0.0, "stake": 0.0})
    for row in rows:
        cat = row["packet"]["category"]
        by_category[cat]["n"] += 1
        if row["decision"]["side"] != "NONE":
            by_category[cat]["trades"] += 1
            by_category[cat]["pnl"] += row["result"]["pnl"]
            by_category[cat]["stake"] += row["result"]["stake"]

    return {
        "n_markets": len(rows),
        "n_trades": len(traded),
        "trade_rate": len(traded) / len(rows) if rows else 0.0,
        "total_pnl": total_pnl,
        "total_stake": total_stake,
        "roi": total_pnl / total_stake if total_stake else 0.0,
        "win_rate": sum(1 for w in wins if w) / len(wins) if wins else 0.0,
        "sharpe_per_trade": avg / math.sqrt(var) if var > 0 else 0.0,
        "by_category": dict(by_category),
    }
