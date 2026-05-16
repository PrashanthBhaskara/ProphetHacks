"""Forecasting benchmark battery for the Prophet Hacks trading track.

Question this script answers:
    Of all available forecasting metrics (Brier, log-loss, ECE,
    AUC-PR, signed_edge, direction-vs-market, ...), which one most
    reliably predicts realized trading P&L on Prophet Arena's
    Subset-1200 benchmark?

How it works:
    1. Build ~12 baseline forecasters spanning the variance space
       (always_half, market, noisy-market, inverse, extremized,
       shrunk, base-rate, ...).
    2. Score every baseline against every forecasting metric.
    3. Run the same baselines through the trading backtest harness
       with the team's recommended `default_tight_band` strategy.
    4. Compute Spearman rank-correlation between each metric and
       realized P&L. The metric with the highest |Spearman| is the
       one to optimize.
    5. Print: full metric table, P&L table, correlation summary.

Usage:
    python scripts/forecast_benchmarks.py
    python scripts/forecast_benchmarks.py --category Sports
    python scripts/forecast_benchmarks.py --no-bootstrap   # faster
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from prep.data import Sample, filter_by_category, load_subset_1200  # noqa: E402


def _load_live_clean() -> list[Sample]:
    """Load eval_pack_live_clean.jsonl directly (the in-repo loader points
    to the wrong filename — `eval_pack.jsonl`). 13K markets with >=2
    snapshots; uses the earliest snapshot's prices."""
    import json
    samples: list[Sample] = []
    p = Path(__file__).resolve().parents[1] / "data" / "eval_pack_live_clean.jsonl"
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        snaps = row.get("snapshots") or []
        if len(snaps) < 2:
            continue
        first = snaps[0]
        info = {
            "yes_ask":    first["yes_ask"] * 100    if first.get("yes_ask") is not None    else None,
            "no_ask":     first["no_ask"] * 100     if first.get("no_ask") is not None     else None,
            "last_price": first["last_price"] * 100 if first.get("last_price") is not None else None,
        }
        if info["yes_ask"] is None or info["no_ask"] is None:
            continue
        samples.append(Sample(event=row["event"], market_info=info, outcome=int(row["outcome"])))
    return samples
from prep.score import (  # noqa: E402
    bootstrap_ci,
    brier,
    full_report,
    log_loss,
    signed_edge,
)
from prep.trade import (  # noqa: E402
    backtest,
    default_min_edge_strategy,
    default_strategy,
    default_tight_band_strategy,
    market_mid_forecast,
)


# ---------------------------------------------------------------------------
# Baseline forecasters — span the variance space so the P&L correlation has
# enough variance to be meaningful.
# ---------------------------------------------------------------------------


def _clip(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, p))


def _market_mid(sample) -> float:
    return _clip(market_mid_forecast(sample.event, sample.market_info))


def _time_split_subset_1200(train_frac: float = 0.7) -> tuple[list, list]:
    """Time-ordered train/test split of subset_1200.

    Splits at the snapshot_time level on SUBMISSIONS (not flattened
    markets), so all markets from an event stay together — no leakage
    between train and test for related markets.
    """
    import pandas as pd
    from ast import literal_eval
    from pathlib import Path
    from prep.data import Sample

    csv = Path(__file__).resolve().parents[1] / "data" / "external" / "subset_1200.csv"
    df = pd.read_csv(csv)
    df = df.sort_values("snapshot_time").reset_index(drop=True)
    cutoff_idx = int(len(df) * train_frac)
    train_df = df.iloc[:cutoff_idx]
    test_df = df.iloc[cutoff_idx:]

    def _to_samples(d):
        out = []
        for _, row in d.iterrows():
            try:
                outcomes = literal_eval(row["market_outcome"]) or {}
                market_data = literal_eval(row["market_data"]) or {}
            except Exception:
                continue
            for market_name, outcome in outcomes.items():
                md = market_data.get(market_name) or {}
                event = {
                    "event_ticker": row["event_ticker"],
                    "market_ticker": f"{row['event_ticker']}-{market_name.replace(' ', '_')}",
                    "title": row.get("title") or "",
                    "subtitle": market_name,
                    "description": None,
                    "category": row.get("category") or "Other",
                    "rules": row.get("rules") or None,
                    "close_time": row.get("close_time") or "",
                }
                out.append(Sample(event=event, market_info=md, outcome=int(outcome)))
        return out

    return _to_samples(train_df), _to_samples(test_df)


def make_baselines(train_samples=None):
    """Return dict of forecaster_name -> (event, market_info) -> p_yes.

    If `train_samples` is provided, also include the fitted
    data-fair-price baselines (mean_bias, platt, decile, category_platt).
    """

    def always_half(e, mi):
        return 0.5

    def base_rate(e, mi):
        # The 29% YES rate from the live eval pack — a constant non-0.5 baseline.
        return 0.29

    def market(e, mi):
        return market_mid_forecast(e, mi)

    def market_shrunk(e, mi):
        # Halfway between market and 0.5 — what an under-confident agent looks like.
        m = market_mid_forecast(e, mi)
        return 0.5 + 0.5 * (m - 0.5)

    def market_extremized(e, mi):
        # Temperature < 1 — push probabilities toward 0 or 1.
        m = market_mid_forecast(e, mi)
        a = m ** 1.5
        b = (1 - m) ** 1.5
        return _clip(a / (a + b))

    def market_plus_bias(e, mi, bias=0.10):
        return _clip(market_mid_forecast(e, mi) + bias)

    def market_minus_bias(e, mi, bias=0.10):
        return _clip(market_mid_forecast(e, mi) - bias)

    def inverse_market(e, mi):
        return _clip(1.0 - market_mid_forecast(e, mi))

    def noisy_market(e, mi, sigma=0.10):
        rng = random.Random(42 + hash(e["market_ticker"]) % 10_000)
        return _clip(market_mid_forecast(e, mi) + rng.gauss(0, sigma))

    def confident_noisy(e, mi):
        return noisy_market(e, mi, sigma=0.20)

    def jitter(e, mi):
        # Tiny noise — almost-market.
        return noisy_market(e, mi, sigma=0.03)

    def smart_extremize_only_high_disagree(e, mi):
        # Mimic an "edgy" agent: market price unless an arbitrary feature
        # of the ticker triggers a 0.15 shift. Gives the metric battery
        # a forecaster whose disagreements are uncorrelated with truth.
        m = market_mid_forecast(e, mi)
        rng = random.Random(99 + hash(e["market_ticker"]) % 10_000)
        if rng.random() < 0.3:
            return _clip(m + rng.choice([-1, 1]) * 0.15)
        return m

    baselines = {
        "always_half": always_half,
        "base_rate_0.29": base_rate,
        "market": market,
        "market_shrunk_to_0.5": market_shrunk,
        "market_extremized_T=0.67": market_extremized,
        "market_plus_0.10": market_plus_bias,
        "market_minus_0.10": market_minus_bias,
        "inverse_market": inverse_market,
        "noisy_market_sigma_0.03": jitter,
        "noisy_market_sigma_0.10": noisy_market,
        "noisy_market_sigma_0.20": confident_noisy,
        "edgy_uncorrelated": smart_extremize_only_high_disagree,
    }

    if train_samples:
        from prep.baselines.data_fair_price import (
            fit_category_platt,
            fit_decile_isotonic,
            fit_mean_bias,
            fit_multi_feature,
            fit_platt_market,
            fit_platt_max_pnl,
        )

        mean_bias = fit_mean_bias(train_samples)
        platt = fit_platt_market(train_samples)
        decile = fit_decile_isotonic(train_samples)
        cat_platt = fit_category_platt(train_samples)
        multi = fit_multi_feature(train_samples)
        platt_pnl = fit_platt_max_pnl(train_samples)

        # Wrap to match the (event, market_info) -> float signature
        def _wrap(f):
            return lambda e, m: f(e, m)["p_yes"]

        baselines["dfp_mean_bias"] = _wrap(mean_bias)
        baselines["dfp_platt_market"] = _wrap(platt)
        baselines["dfp_decile_isotonic"] = _wrap(decile)
        baselines["dfp_category_platt"] = _wrap(cat_platt)
        baselines["dfp_multi_feature"] = _wrap(multi)
        baselines["dfp_platt_max_pnl"] = _wrap(platt_pnl)

        # Pretty-print what was learned
        print(f"  Fitted dfp_mean_bias: shift = {mean_bias.shift:+.4f}")
        print(f"  Fitted dfp_platt_market: slope={platt.slope:.3f}, intercept={platt.intercept:.3f}")
        print(f"  Fitted dfp_decile_isotonic: {len(decile.edges)} bins, rates "
              f"{[round(r, 3) for r in decile.rates]}")
        print(f"  Fitted dfp_category_platt: per-category fits for "
              f"{list(cat_platt.cat_fits.keys())}")
        if hasattr(multi, "beta"):
            top3 = sorted(zip(multi.feature_names, multi.beta), key=lambda x: -abs(x[1]))[:5]
            print(f"  Fitted dfp_multi_feature: top coefs = {[(n, round(v, 3)) for n, v in top3]}")
        print(f"  Fitted dfp_platt_max_pnl: a={platt_pnl.slope}, b={platt_pnl.intercept}, "
              f"train P&L=${platt_pnl.train_pnl:.0f}")

    return baselines


# ---------------------------------------------------------------------------
# Scoring + P&L pipeline
# ---------------------------------------------------------------------------


def score_baseline(name, fn, samples, do_bootstrap: bool) -> dict:
    p_yes = []
    outcomes = []
    market_q = []
    for s in samples:
        try:
            p = float(fn(s.event, s.market_info))
        except Exception:
            continue
        ya = s.market_info.get("yes_ask")
        na = s.market_info.get("no_ask")
        if ya is None or na is None:
            continue
        # Normalize to 0-1
        ya_d = ya / 100 if ya > 1 else ya
        na_d = na / 100 if na > 1 else na
        if ya_d + na_d <= 0:
            continue
        q = (ya_d + (1 - na_d)) / 2
        p_yes.append(_clip(p))
        outcomes.append(int(s.outcome))
        market_q.append(q)

    report = full_report(p_yes, outcomes, market_q=market_q)

    if do_bootstrap and len(p_yes) >= 50:
        lo, hi = bootstrap_ci(brier, p_yes, outcomes, n_resamples=200)
        report["brier_ci"] = (lo, hi)
        lo, hi = bootstrap_ci(signed_edge, p_yes, outcomes, market_q, n_resamples=200)
        report["signed_edge_ci"] = (lo, hi)

    # P&L from THREE strategies — checks that the metric ↔ P&L correlation
    # we identify isn't a quirk of one specific strategy filter.
    for strat_name, strat_fn in (
        ("tight_band", default_tight_band_strategy),
        ("default",    default_strategy),
        ("min_edge_5", default_min_edge_strategy),
    ):
        r = backtest(samples, forecast_fn=fn, strategy=strat_fn)
        report[f"pnl_{strat_name}"] = r["total_pnl"]
        report[f"trades_{strat_name}"] = r["n_trades"]

    report["pnl"] = report["pnl_tight_band"]
    report["n_trades"] = report["trades_tight_band"]
    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_tier1_table(rows):
    print()
    print("=" * 110)
    print("TIER 1 — proper scoring rules  (Brier ↓, log_loss ↓, spherical ↑)")
    print("=" * 110)
    print(f"  {'Forecaster':28s} {'N':>6}  {'Brier':>7}  {'LogLoss':>8}  {'Spherical':>9}  {'BSS vs½':>8}  {'BSS vsM':>8}")
    for name, r in rows:
        bss_half = r.get("bss_vs_half", float("nan"))
        bss_mkt = r.get("bss_vs_market", float("nan"))
        print(f"  {name:28s} {r['n']:>6}  "
              f"{r['brier']:>7.4f}  {r['log_loss']:>8.4f}  {r['spherical']:>9.4f}  "
              f"{bss_half:>+8.3f}  {bss_mkt:>+8.3f}")


def _print_tier2_table(rows):
    print()
    print("=" * 110)
    print("TIER 2 — calibration + decomposition  (lower reliability/ECE/MCE = better; higher resolution = better)")
    print("=" * 110)
    print(f"  {'Forecaster':28s}  {'ECE':>6}  {'ECEadp':>7}  {'MCE':>6}  {'Reli':>6}  {'Resol':>6}  {'Sharp':>6}  {'PSlope':>7}  {'PInt':>6}")
    for name, r in rows:
        print(f"  {name:28s}  "
              f"{r['ece']:>6.3f}  {r['ece_adaptive']:>7.3f}  {r['mce']:>6.3f}  "
              f"{r['reliability']:>6.3f}  {r['resolution']:>6.3f}  {r['sharpness']:>6.3f}  "
              f"{r.get('slope', float('nan')):>+7.2f}  {r.get('intercept', float('nan')):>+6.2f}")


def _print_tier3_table(rows):
    print()
    print("=" * 110)
    print("TIER 3 — discrimination + TRADING-RELEVANT (signed_edge / direction-vs-market should predict P&L)")
    print("=" * 110)
    print(f"  {'Forecaster':28s}  {'AUC-ROC':>7}  {'AUC-PR':>7}  {'Acc@.5':>7}  {'Dir>q':>6}  {'Dir>q±5':>8}  {'SignedEdge':>11}")
    for name, r in rows:
        print(f"  {name:28s}  "
              f"{r['auc_roc']:>7.4f}  {r['auc_pr']:>7.4f}  {r['accuracy_50']:>7.4f}  "
              f"{r.get('direction_vs_market', float('nan')):>6.3f}  "
              f"{r.get('direction_vs_market_deadband_0.05', float('nan')):>8.3f}  "
              f"{r.get('signed_edge', float('nan')):>+11.6f}")


def _print_pnl_table(rows):
    print()
    print("=" * 110)
    print("REALIZED TRADING P&L across 3 strategies  ($10k starting cash) — metric must predict ALL columns")
    print("=" * 110)
    print(f"  {'Forecaster':28s}  {'tight_band $':>12}  {'default $':>10}  {'min_edge_5 $':>13}")
    for name, r in rows:
        print(f"  {name:28s}  ${r['pnl_tight_band']:>+11,.2f}  ${r['pnl_default']:>+9,.2f}  ${r['pnl_min_edge_5']:>+12,.2f}")


def _spearman(xs, ys):
    """Spearman rank-correlation. Avoid sklearn/scipy dependency."""
    n = len(xs)
    if n < 2:
        return float("nan")
    rank_x = _rank(xs)
    rank_y = _rank(ys)
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n
    num = sum((rx - mean_x) * (ry - mean_y) for rx, ry in zip(rank_x, rank_y))
    den_x = math.sqrt(sum((rx - mean_x) ** 2 for rx in rank_x))
    den_y = math.sqrt(sum((ry - mean_y) ** 2 for ry in rank_y))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _rank(vals):
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + j + 1) / 2  # 1-indexed
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


def _print_pnl_correlation(rows):
    """Spearman correlation between every numeric metric and realized P&L
    UNDER EACH OF THREE STRATEGIES. Forecasters that never trade are
    excluded — they leak no information about predictive power.

    Headline: a "best forecasting metric" should rank-correlate strongly
    AND consistently across all three strategy columns. A metric that
    only works for one filter setting is brittle.
    """
    print()
    print("=" * 110)
    print("META — Spearman rank-correlation of each metric vs realized P&L (across 3 strategies)")
    print("=" * 110)
    print("  Forecasters with 0 trades excluded — they carry no signal.")
    print()
    print(f"  {'Metric':30s}  {'tight_band':>11}  {'default':>9}  {'min_edge':>9}  {'avg(|ρ|)':>9}")
    print(f"  {'-'*30}  {'-'*11}  {'-'*9}  {'-'*9}  {'-'*9}")

    metric_keys = [
        "brier", "log_loss", "spherical",
        "bss_vs_half", "bss_vs_market",
        "ece", "ece_adaptive", "mce",
        "reliability", "resolution", "sharpness",
        "slope", "intercept",
        "auc_roc", "auc_pr", "accuracy_50",
        "direction_vs_market", "direction_vs_market_deadband_0.05",
        "signed_edge",
    ]
    strategy_cols = ("pnl_tight_band", "pnl_default", "pnl_min_edge_5")

    summary = []
    for key in metric_keys:
        rhos: list[float] = []
        for strat_col in strategy_cols:
            # Mask: keep rows where (a) the metric is finite AND
            # (b) the forecaster actually traded under this strategy.
            trade_col = "trades_" + strat_col[len("pnl_"):]
            vals = []
            pnls = []
            for _, r in rows:
                v = r.get(key, float("nan"))
                if math.isnan(v):
                    continue
                if r.get(trade_col, 0) == 0:
                    continue
                vals.append(v)
                pnls.append(r[strat_col])
            if len(vals) < 3:
                rhos.append(float("nan"))
                continue
            rhos.append(_spearman(vals, pnls))

        valid = [r for r in rhos if not math.isnan(r)]
        avg_abs = sum(abs(r) for r in valid) / len(valid) if valid else float("nan")
        summary.append((key, rhos, avg_abs))

    summary.sort(key=lambda x: (-x[2] if not math.isnan(x[2]) else 0))

    for key, rhos, avg_abs in summary:
        def fmt(r):
            return f"{r:>+11.3f}" if not math.isnan(r) else f"{'n/a':>11}"
        flag = "  ⭐" if avg_abs > 0.7 else ("   *" if avg_abs > 0.5 else "    ")
        print(f"  {key:30s}  {fmt(rhos[0])}  {fmt(rhos[1])[3:]}  {fmt(rhos[2])[3:]}  {avg_abs:>+9.3f}{flag}")
    print()
    print("  ⭐ |ρ̄| > 0.7 (strong, stable)   * |ρ̄| > 0.5 (moderate)")
    print("  Read the +/− sign: + means higher metric → higher P&L; − means lower → higher.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="restrict to one category (e.g. Sports)")
    parser.add_argument("--no-bootstrap", action="store_true", help="skip bootstrap CIs (faster)")
    parser.add_argument("--source", choices=("subset_1200", "live"), default="subset_1200",
                        help="subset_1200 = authoritative HF benchmark; live = our 13K eval_pack_live_clean")
    parser.add_argument("--fair-price-split", action="store_true",
                        help="time-split subset_1200 into train(earliest 70%%)/test(latest 30%%); "
                             "fit data-fair-price baselines on train, evaluate everything on test")
    args = parser.parse_args()

    train_samples: list | None = None
    if args.fair_price_split:
        if args.source != "subset_1200":
            print("--fair-price-split only supported on subset_1200")
            return 1
        train_samples, samples = _time_split_subset_1200(0.7)
        src_label = "Subset-1200 TIME-SPLIT (train 70% earliest → test 30% latest)"
        print(f"  Train: {len(train_samples):,} markets   Test: {len(samples):,} markets")
    elif args.source == "subset_1200":
        samples = load_subset_1200()
        src_label = "Prophet-Arena-Subset-1200 (full)"
    else:
        samples = _load_live_clean()
        src_label = "eval_pack_live_clean (13K self-polled)"
    if args.category:
        samples = filter_by_category(samples, args.category)
        if train_samples:
            train_samples = filter_by_category(train_samples, args.category)
    print(f"Loaded {len(samples):,} samples from {src_label}" + (f" (filtered to {args.category})" if args.category else ""))

    base_rate = sum(s.outcome for s in samples) / len(samples)
    print(f"  YES rate: {base_rate:.3f}    (Uncertainty = p(1-p) = {base_rate*(1-base_rate):.4f}, the floor on any Brier)")

    baselines = make_baselines(train_samples=train_samples)
    rows = []
    for name, fn in baselines.items():
        try:
            r = score_baseline(name, fn, samples, do_bootstrap=not args.no_bootstrap)
            rows.append((name, r))
        except Exception as e:
            print(f"  !! {name}: {e}")
            continue

    _print_tier1_table(rows)
    _print_tier2_table(rows)
    _print_tier3_table(rows)
    _print_pnl_table(rows)
    _print_pnl_correlation(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
