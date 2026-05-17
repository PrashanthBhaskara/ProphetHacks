"""Walk-forward validation of per-series Platt on the 2026 sample.

Fit coefficients on weeks 1..k, eval on week k+1. Repeat over the whole
range. Reports per-fold Brier delta vs market to check whether the
per-series calibration genuinely generalizes (vs overfit-to-half).

Usage:
    python prep/scripts/walk_forward_per_series.py \\
        --sample prep_handoff/Kalshitopvolmarkets/samples/llm_calls_x50_seed42.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.score import brier  # noqa: E402


def logit(p, eps=1e-4):
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def sig(z):
    z = max(-30.0, min(30.0, z))
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def fit_platt(ps, ys, l2=0.5, iters=100):
    if not ps or len(ps) < 5:
        return 0.0, 1.0
    a, b = 0.0, 1.0
    x = [logit(p) for p in ps]
    for _ in range(iters):
        mu = [sig(a + b * xi) for xi in x]
        g_a = -sum(y - m for y, m in zip(ys, mu))
        g_b = -sum((y - m) * xi for y, m, xi in zip(ys, mu, x)) + l2 * b
        w = [m * (1 - m) for m in mu]
        haa = sum(w)
        hab = sum(wi * xi for wi, xi in zip(w, x))
        hbb = sum(wi * xi * xi for wi, xi in zip(w, x)) + l2
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        d_a = (hbb * g_a - hab * g_b) / det
        d_b = (haa * g_b - hab * g_a) / det
        step = max(1.0, abs(d_a), abs(d_b))
        a -= d_a / step
        b -= d_b / step
        if max(abs(d_a), abs(d_b)) < 1e-7:
            break
    return a, b


def fit_per_series(samples, min_n=15):
    """Returns (per_series_dict, global_coefs)."""
    by_s: dict[str, list] = defaultdict(list)
    for s in samples:
        series = s.get("series_ticker") or s["ticker"].split("-")[0]
        by_s[series].append(s)
    per_series = {}
    for series, ss in by_s.items():
        if len(ss) < min_n:
            continue
        ps = [sp["quote"]["market_mid"] for sp in ss]
        ys = [sp["outcome_yes"] for sp in ss]
        per_series[series] = fit_platt(ps, ys)
    # Global
    g = fit_platt(
        [s["quote"]["market_mid"] for s in samples],
        [s["outcome_yes"] for s in samples],
    )
    return per_series, g


def predict_per_series(p, series, per_series, global_coefs):
    a, b = per_series.get(series, global_coefs)
    return sig(a + b * logit(p))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--min-fit-weeks", type=int, default=8, help="Min weeks of training data before first eval")
    args = parser.parse_args()

    samples = [json.loads(l) for l in args.sample.read_text().splitlines() if l.strip()]
    samples = [s for s in samples if s.get("outcome_yes") in (0, 1)]
    print(f"Loaded {len(samples)} resolved samples")

    by_week: dict[str, list] = defaultdict(list)
    for s in samples:
        by_week[s["week"]].append(s)
    weeks = sorted(by_week.keys())
    print(f"Weeks: {len(weeks)} ({weeks[0]} → {weeks[-1]})")

    print()
    print(f"Walk-forward (fit on weeks 1..k, eval on week k+1, min {args.min_fit_weeks} fit weeks):")
    print(f"  {'eval_week':<14}{'N_fit':>7}{'N_eval':>8}{'mkt Brier':>11}{'PSP Brier':>12}{'Δ':>10}")
    total_n = 0
    total_mkt_loss = 0.0
    total_psp_loss = 0.0
    for i in range(args.min_fit_weeks, len(weeks)):
        fit_weeks = weeks[:i]
        eval_week = weeks[i]
        fit_samples = [s for w in fit_weeks for s in by_week[w]]
        eval_samples = by_week[eval_week]
        if not eval_samples:
            continue
        per_series, global_coefs = fit_per_series(fit_samples)

        mids = [s["quote"]["market_mid"] for s in eval_samples]
        outs = [s["outcome_yes"] for s in eval_samples]
        psp = [
            predict_per_series(s["quote"]["market_mid"],
                               (s.get("series_ticker") or s["ticker"].split("-")[0]),
                               per_series, global_coefs)
            for s in eval_samples
        ]
        b_mkt = brier(mids, outs)
        b_psp = brier(psp, outs)
        delta = (b_psp - b_mkt) * 100
        # Accumulate weighted (so the final aggregate is overall Brier)
        total_n += len(eval_samples)
        total_mkt_loss += sum((m - o) ** 2 for m, o in zip(mids, outs))
        total_psp_loss += sum((p - o) ** 2 for p, o in zip(psp, outs))
        print(f"  {eval_week:<14}{len(fit_samples):>7}{len(eval_samples):>8}{b_mkt:>11.4f}{b_psp:>12.4f}{delta:>+9.2f}pp")

    if total_n > 0:
        agg_mkt = total_mkt_loss / total_n
        agg_psp = total_psp_loss / total_n
        agg_d = (agg_psp - agg_mkt) * 100
        print(f"\n  {'AGGREGATE':<14}{'':>7}{total_n:>8}{agg_mkt:>11.4f}{agg_psp:>12.4f}{agg_d:>+9.2f}pp")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
