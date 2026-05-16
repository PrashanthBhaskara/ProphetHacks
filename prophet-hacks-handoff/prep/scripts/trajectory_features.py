"""Trajectory-features forecaster — uses 5M trades from kalshi_trades_*.parquet.

Builds per-market features from the trade time series:
    q_first         — earliest trade yes_price/100 (the "naive" market signal)
    q_last          — latest pre-close trade yes_price/100
    q_vwap          — volume-weighted average price across all trades
    drift           — q_last − q_first (signed)
    vol             — std of yes_price across trades
    n_trades        — log1p of trade count
    taker_imbalance — (n_yes_takers − n_no_takers) / (n_trades)
    log_event_size  — log of markets-per-event

Time-splits the resolved markets: train on the earliest 70%, test on the
latest 30%. Reports Brier, log_loss, BSS vs raw market price.

Honest OOD test — see if trajectory features outperform a single-snapshot
recalibration on the parquet's own distribution.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prep.baselines.data_fair_price import fit_event_size_platt, fit_platt_market  # noqa: E402
from prep.data import Sample, load_subset_1200  # noqa: E402

TRADES_DIR = Path("/Users/victor/Prophet Hacks/prep/data/external")


def build_market_features() -> pd.DataFrame:
    """Per-market summary of the 5M trades + outcome from kalshi_markets.parquet.

    Returns one row per resolved binary market with all trajectory features.
    """
    t0 = pd.read_parquet(TRADES_DIR / "kalshi_trades_0.parquet")
    t1 = pd.read_parquet(TRADES_DIR / "kalshi_trades_1.parquet")
    trades = pd.concat([t0, t1], ignore_index=True)
    trades["created_time"] = pd.to_datetime(trades["created_time"], format="ISO8601")
    trades = trades.sort_values(["market_ticker", "created_time"])

    # Drop trades that look post-resolution (price stuck at 1 or 99 for the
    # entire last few trades). Keep only the first ~80% of each market's
    # trades, so we don't peek at near-settlement contamination.
    print(f"Loaded {len(trades):,} trades across {trades['market_ticker'].nunique()} markets")

    grouped = trades.groupby("market_ticker")

    # Per-market aggregations
    def agg(g):
        n = len(g)
        cutoff = max(1, int(n * 0.8))  # keep first 80% — pre-resolution slice
        g2 = g.iloc[:cutoff]
        prices = g2["yes_price"].to_numpy() / 100.0
        counts = g2["count"].to_numpy()
        q_first = float(prices[0])
        q_last = float(prices[-1])
        q_vwap = float(np.average(prices, weights=counts + 1e-9))
        vol = float(np.std(prices)) if len(prices) > 1 else 0.0
        drift = q_last - q_first
        n_yes = int((g2["taker_side"] == "yes").sum())
        n_no = int((g2["taker_side"] == "no").sum())
        taker_imb = (n_yes - n_no) / max(1, n_yes + n_no)
        return pd.Series({
            "q_first": q_first,
            "q_last": q_last,
            "q_vwap": q_vwap,
            "drift": drift,
            "vol": vol,
            "n_trades": len(g2),
            "taker_imb": taker_imb,
            "first_time": g["created_time"].iloc[0],
        })

    feats = grouped.apply(agg, include_groups=False).reset_index()
    print(f"Features built for {len(feats)} markets")

    # Attach outcomes + event_ticker from markets parquet
    markets = pd.read_parquet(TRADES_DIR / "kalshi_markets.parquet")[
        ["ticker", "event_ticker", "result"]
    ]
    feats = feats.merge(markets, left_on="market_ticker", right_on="ticker", how="inner")
    feats = feats[feats["result"].isin(["yes", "no"])].copy()
    feats["y"] = (feats["result"] == "yes").astype(int)

    # Event size
    sizes = feats.groupby("event_ticker").size()
    feats["n_event"] = feats["event_ticker"].map(sizes)
    feats["log_event_size"] = np.log(feats["n_event"].clip(lower=1))

    return feats


def fit_logistic_l2(X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 200) -> np.ndarray:
    """IRLS logistic with L2. Returns beta vector."""
    n_feat = X.shape[1]
    beta = np.zeros(n_feat)
    L2 = np.full(n_feat, l2)
    L2[0] = 0.0  # don't regularize bias
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


def log_loss(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit_clip(x, lo=1e-4):
    x = np.clip(x, lo, 1 - lo)
    return np.log(x / (1 - x))


def main():
    feats = build_market_features()

    # Time-split by first_time (per-market open). Train = earliest 70%.
    feats = feats.sort_values("first_time").reset_index(drop=True)
    cut = int(len(feats) * 0.7)
    train = feats.iloc[:cut].copy()
    test = feats.iloc[cut:].copy()
    print(f"\nTrain: {len(train):,}  ({train['first_time'].min()} → {train['first_time'].max()})")
    print(f"Test : {len(test):,}  ({test['first_time'].min()} → {test['first_time'].max()})")
    print(f"Train YES rate: {train['y'].mean():.3f}   Test YES rate: {test['y'].mean():.3f}\n")

    # Reference: raw earliest trade as the prediction
    p_market = test["q_first"].clip(0.01, 0.99).to_numpy()
    y_test = test["y"].to_numpy()
    b_market = brier(p_market, y_test)
    ll_market = log_loss(p_market, y_test)
    print(f"=== Baseline: q_first (earliest-trade) ===")
    print(f"  Brier  {b_market:.4f}   LogLoss {ll_market:.4f}")

    # Reference: q_vwap (volume-weighted avg, using first 80% of trades)
    p_vwap = test["q_vwap"].clip(0.01, 0.99).to_numpy()
    print(f"\n=== Baseline: q_vwap (volume-weighted, pre-resolution slice) ===")
    print(f"  Brier  {brier(p_vwap, y_test):.4f}   LogLoss {log_loss(p_vwap, y_test):.4f}   "
          f"BSS vs q_first {1 - brier(p_vwap, y_test)/b_market:+.3f}")

    # Reference: q_last (last trade in pre-resolution slice)
    p_last = test["q_last"].clip(0.01, 0.99).to_numpy()
    print(f"\n=== Baseline: q_last (latest pre-resolution trade) ===")
    print(f"  Brier  {brier(p_last, y_test):.4f}   LogLoss {log_loss(p_last, y_test):.4f}   "
          f"BSS vs q_first {1 - brier(p_last, y_test)/b_market:+.3f}")

    # --- Fitted models, train-only ---
    feature_specs = [
        ("logit_q_first",
         ["q_first"],
         lambda d: np.column_stack([np.ones(len(d)), logit_clip(d["q_first"].to_numpy())])),

        ("logit_q + log_event_size",
         ["q_first", "log_event_size"],
         lambda d: np.column_stack([np.ones(len(d)), logit_clip(d["q_first"].to_numpy()),
                                    d["log_event_size"].to_numpy()])),

        ("trajectory (q_vwap + drift + vol + log_event)",
         ["q_vwap", "drift", "vol", "log_event_size"],
         lambda d: np.column_stack([
             np.ones(len(d)),
             logit_clip(d["q_vwap"].to_numpy()),
             d["drift"].to_numpy(),
             d["vol"].to_numpy(),
             d["log_event_size"].to_numpy(),
         ])),

        ("full (q_vwap, drift, vol, taker_imb, n_trades, log_event)",
         ["q_vwap", "drift", "vol", "taker_imb", "n_trades", "log_event_size"],
         lambda d: np.column_stack([
             np.ones(len(d)),
             logit_clip(d["q_vwap"].to_numpy()),
             d["drift"].to_numpy(),
             d["vol"].to_numpy(),
             d["taker_imb"].to_numpy(),
             np.log1p(d["n_trades"].to_numpy()),
             d["log_event_size"].to_numpy(),
         ])),
    ]

    print(f"\n=== Fitted models (train→test, time-split on parquet) ===")
    print(f"  {'Model':52s}  {'Brier':>7}  {'LogLoss':>8}  {'BSS vs q_first':>14}")
    for name, _, build in feature_specs:
        X_tr = build(train)
        y_tr = train["y"].to_numpy().astype(float)
        beta = fit_logistic_l2(X_tr, y_tr, l2=1.0)
        X_te = build(test)
        z = np.clip(X_te @ beta, -30, 30)
        p_test = 1 / (1 + np.exp(-z))
        b = brier(p_test, y_test)
        ll = log_loss(p_test, y_test)
        bss = 1 - b / b_market
        print(f"  {name:52s}  {b:>7.4f}  {ll:>8.4f}  {bss:>+14.3f}")
        if "trajectory" in name or "full" in name:
            print(f"     coefs: {[round(float(x), 3) for x in beta]}")

    # Also: how does the subset_1200-fit event_size_platt transfer here?
    print(f"\n=== Frozen models from subset_1200 (cross-time, cross-distribution) ===")
    sub_train = load_subset_1200()
    for name, fitter in [("platt_market (subset_1200)", fit_platt_market),
                         ("event_size_platt (subset_1200)", fit_event_size_platt)]:
        m = fitter(sub_train)
        # Build pseudo test samples
        test_samples = []
        for _, r in test.iterrows():
            q = float(r["q_first"])
            info = {"yes_ask": q * 100 + 2, "no_ask": (1 - q) * 100 + 2}
            event = {"event_ticker": r["event_ticker"], "market_ticker": r["market_ticker"],
                     "category": "Other", "title": "", "close_time": ""}
            test_samples.append(Sample(event=event, market_info=info, outcome=int(r["y"])))
        if hasattr(m, "attach_test_sizes"):
            m.attach_test_sizes(test_samples)
        p_pred = np.array([np.clip(m(s.event, s.market_info)["p_yes"], 0.01, 0.99)
                           for s in test_samples])
        b = brier(p_pred, y_test)
        ll = log_loss(p_pred, y_test)
        bss = 1 - b / b_market
        print(f"  {name:52s}  {b:.4f}  {ll:.4f}  BSS {bss:+.3f}")


if __name__ == "__main__":
    main()
