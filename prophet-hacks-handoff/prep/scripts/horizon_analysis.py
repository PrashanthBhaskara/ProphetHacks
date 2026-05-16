"""Horizon-aware trajectory features.

Question: at what point before market close does the trajectory edge
disappear (i.e., the market absorbs the information)?

Procedure: for each horizon H ∈ {168h, 72h, 24h, 6h, 1h}, simulate what
a live agent would see by truncating each market's trade history to
trades that occurred BEFORE (close_time − H). Recompute features. Fit
on train, score on test. Report Brier + a simulated P&L.

This tells us where the LLM should intervene vs defer to the data.
Per paper §3.2.1, "markets beat LLMs in the last 3 hours" — we now
test that on our actual data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRADES_DIR = Path("/Users/victor/Prophet Hacks/prep/data/external")
HORIZONS_HOURS = [168, 72, 24, 6, 1]  # 1 week → 1 hour before close


def fit_logistic_l2(X, y, l2=1.0, max_iter=200):
    n_feat = X.shape[1]
    beta = np.zeros(n_feat)
    L2 = np.full(n_feat, l2)
    L2[0] = 0.0
    for _ in range(max_iter):
        z = np.clip(X @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-z))
        W = np.clip(mu * (1 - mu), 1e-6, None)
        g = -X.T @ (y - mu) + L2 * beta
        H = (X.T * W) @ X + np.diag(L2)
        try:
            d = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = max(1.0, float(np.max(np.abs(d))))
        d /= step
        beta -= d
        if float(np.max(np.abs(d))) < 1e-7:
            break
    return beta


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logit_clip(x, lo=1e-4):
    x = np.clip(x, lo, 1 - lo)
    return np.log(x / (1 - x))


def simulated_pnl(p, q, y, spread=0.04, edge_threshold=0.05):
    """Simulate per-market P&L: bet on whichever side p disagrees with q
    on, by `|p - q|`. Strategy filter: skip if |p-q| < edge_threshold.
    Spread is implied (2c each side). Per-market shares scaled to $100.

    Returns total simulated P&L.
    """
    p = np.clip(p, 0.01, 0.99)
    q = np.clip(q, 0.01, 0.99)
    edge = p - q
    pnls = np.zeros(len(p))
    mask = np.abs(edge) >= edge_threshold
    for i in np.where(mask)[0]:
        # Buy YES if p > q (bet costs q + spread/2)
        if edge[i] > 0:
            shares = abs(edge[i]) * 100
            cost = shares * (q[i] + spread / 2)
            payoff = shares if y[i] == 1 else 0
        else:
            shares = abs(edge[i]) * 100
            cost = shares * ((1 - q[i]) + spread / 2)
            payoff = shares if y[i] == 0 else 0
        pnls[i] = payoff - cost
    return float(pnls.sum()), int(mask.sum())


def main():
    # Load trades + markets
    print("Loading trades...")
    t0 = pd.read_parquet(TRADES_DIR / "kalshi_trades_0.parquet")
    t1 = pd.read_parquet(TRADES_DIR / "kalshi_trades_1.parquet")
    trades = pd.concat([t0, t1], ignore_index=True)
    trades["created_time"] = pd.to_datetime(trades["created_time"], format="ISO8601")

    markets = pd.read_parquet(TRADES_DIR / "kalshi_markets.parquet")[
        ["ticker", "event_ticker", "result", "close_time"]
    ]
    markets["close_time"] = pd.to_datetime(markets["close_time"], format="ISO8601")
    markets = markets[markets["result"].isin(["yes", "no"])].copy()
    markets["y"] = (markets["result"] == "yes").astype(int)

    print(f"  trades: {len(trades):,}   resolved markets: {len(markets):,}")

    # Build event sizes once (constant across horizons)
    event_sizes = markets.groupby("event_ticker")["ticker"].count()
    markets["log_event_size"] = np.log(
        markets["event_ticker"].map(event_sizes).clip(lower=1)
    )

    # Merge close_time into trades; pre-compute "time before close"
    trades = trades.merge(markets[["ticker", "close_time"]],
                          left_on="market_ticker", right_on="ticker",
                          how="inner")
    trades["hours_before_close"] = (
        (trades["close_time"] - trades["created_time"]).dt.total_seconds() / 3600.0
    )
    trades = trades[trades["hours_before_close"] >= 0]
    trades = trades.sort_values(["market_ticker", "created_time"])

    # For sorting markets time-wise for train/test split, get each
    # market's first trade time
    first_trade = trades.groupby("market_ticker")["created_time"].first().rename("first_trade")
    markets = markets.merge(first_trade, left_on="ticker", right_index=True, how="inner")
    markets = markets.sort_values("first_trade").reset_index(drop=True)

    cut = int(len(markets) * 0.7)
    train_tickers = set(markets.iloc[:cut]["ticker"])
    test_tickers = set(markets.iloc[cut:]["ticker"])
    print(f"  train: {len(train_tickers):,}   test: {len(test_tickers):,}\n")

    print(f"{'Horizon':>10}  {'#mkts (train/test)':>20}  "
          f"{'Brier_baseline':>14}  {'Brier_fitted':>13}  "
          f"{'BSS':>7}  {'Sim P&L':>9}  {'#trades':>7}")
    print("-" * 95)

    for h_hours in HORIZONS_HOURS:
        # Slice trades: keep only those at least `h_hours` before close
        sliced = trades[trades["hours_before_close"] >= h_hours]
        if sliced.empty:
            continue
        grouped = sliced.groupby("market_ticker", sort=False)
        feats = grouped.agg(
            q_first=("yes_price", lambda x: x.iloc[0] / 100),
            q_last=("yes_price", lambda x: x.iloc[-1] / 100),
            q_vwap=("yes_price",
                    lambda x: float(np.average(x.to_numpy() / 100,
                                               weights=grouped.get_group(x.name)["count"].to_numpy() + 1e-9))),
            vol=("yes_price", lambda x: float(np.std(x.to_numpy() / 100)) if len(x) > 1 else 0.0),
            n_trades=("yes_price", "size"),
        ).reset_index()
        feats["drift"] = feats["q_last"] - feats["q_first"]
        feats = feats.merge(
            markets[["ticker", "y", "log_event_size"]],
            left_on="market_ticker", right_on="ticker"
        )

        # Subset to train/test
        tr = feats[feats["market_ticker"].isin(train_tickers)].copy()
        te = feats[feats["market_ticker"].isin(test_tickers)].copy()
        if len(tr) < 100 or len(te) < 100:
            print(f"{h_hours:>9}h  too few markets, skipping")
            continue

        def build(df):
            return np.column_stack([
                np.ones(len(df)),
                logit_clip(df["q_vwap"].to_numpy()),
                df["drift"].to_numpy(),
                df["vol"].to_numpy(),
                df["log_event_size"].to_numpy(),
            ])

        X_tr = build(tr)
        y_tr = tr["y"].to_numpy().astype(float)
        beta = fit_logistic_l2(X_tr, y_tr, l2=1.0)
        X_te = build(te)
        z = np.clip(X_te @ beta, -30, 30)
        p_test = 1 / (1 + np.exp(-z))
        y_test = te["y"].to_numpy()
        q_test = te["q_last"].clip(0.01, 0.99).to_numpy()

        b_base = brier(q_test, y_test)
        b_fit = brier(p_test, y_test)
        bss = 1 - b_fit / b_base if b_base > 0 else float("nan")

        pnl, n_trades = simulated_pnl(p_test, q_test, y_test)

        print(f"{h_hours:>9}h  {len(tr):>9,}/{len(te):>9,}  "
              f"{b_base:>14.4f}  {b_fit:>13.4f}  "
              f"{bss:>+7.3f}  ${pnl:>+8.2f}  {n_trades:>7}")
        # Coefficient on drift — the headline signal
        print(f"             coefs: q_vwap={beta[1]:.2f}  drift={beta[2]:.2f}  "
              f"vol={beta[3]:.2f}  log_event={beta[4]:.2f}")


if __name__ == "__main__":
    main()
