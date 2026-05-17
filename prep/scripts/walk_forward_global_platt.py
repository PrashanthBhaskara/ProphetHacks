"""Walk-forward CV of the global Platt calibration on the 2026 sample.

For each test week:
  - Fit Platt on all weeks before it
  - Apply to test week
  - Compare Brier vs raw market

Verifies whether the small Brier gain seen on a single fit/eval split
is real or just overfit.
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
    p = max(eps, min(1 - eps, p)); return math.log(p / (1 - p))


def sig(z):
    z = max(-30, min(30, z))
    if z >= 0: e = math.exp(-z); return 1 / (1 + e)
    e = math.exp(z); return e / (1 + e)


def fit_platt(ps, ys, l2=0.5, iters=200):
    if len(ps) < 5: return 0.0, 1.0
    a, b = 0.0, 1.0
    x = [logit(p) for p in ps]
    for _ in range(iters):
        mu = [sig(a + b * xi) for xi in x]
        g_a = -sum(y - m for y, m in zip(ys, mu))
        g_b = -sum((y - m) * xi for y, m, xi in zip(ys, mu, x)) + l2 * b
        w = [m * (1 - m) for m in mu]
        haa = sum(w); hab = sum(wi * xi for wi, xi in zip(w, x))
        hbb = sum(wi * xi * xi for wi, xi in zip(w, x)) + l2
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12: break
        d_a = (hbb * g_a - hab * g_b) / det
        d_b = (haa * g_b - hab * g_a) / det
        step = max(1.0, abs(d_a), abs(d_b))
        a -= d_a / step; b -= d_b / step
        if max(abs(d_a), abs(d_b)) < 1e-7: break
    return a, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--min-fit-weeks", type=int, default=4)
    args = parser.parse_args()

    samples = [json.loads(l) for l in args.sample.read_text().splitlines() if l.strip()]
    samples = [s for s in samples if s.get("outcome_yes") in (0, 1)]
    print(f"Loaded {len(samples)} resolved samples")

    by_week = defaultdict(list)
    for s in samples:
        by_week[s["week"]].append(s)
    weeks = sorted(by_week.keys())
    print(f"Weeks: {len(weeks)} ({weeks[0]} → {weeks[-1]})")

    print()
    print(f"Walk-forward (fit on all weeks before eval_week, min {args.min_fit_weeks} fit weeks):")
    print(f"  {'eval_week':<14}{'N_fit':>7}{'N_eval':>8}{'mkt Brier':>11}{'Platt Brier':>14}{'  a       b':>14}{'Δ':>10}")
    total_n = 0
    total_mkt_loss = 0.0
    total_plt_loss = 0.0
    for i in range(args.min_fit_weeks, len(weeks)):
        fit_weeks = weeks[:i]
        eval_week = weeks[i]
        fit_samples = [s for w in fit_weeks for s in by_week[w]]
        eval_samples = by_week[eval_week]
        if not eval_samples: continue
        a, b = fit_platt([s["quote"]["market_mid"] for s in fit_samples],
                         [s["outcome_yes"] for s in fit_samples])
        mids = [s["quote"]["market_mid"] for s in eval_samples]
        outs = [s["outcome_yes"] for s in eval_samples]
        plt_preds = [sig(a + b * logit(p)) for p in mids]
        b_mkt = brier(mids, outs)
        b_plt = brier(plt_preds, outs)
        delta = (b_plt - b_mkt) * 100
        total_n += len(eval_samples)
        total_mkt_loss += sum((m - o) ** 2 for m, o in zip(mids, outs))
        total_plt_loss += sum((p - o) ** 2 for p, o in zip(plt_preds, outs))
        print(f"  {eval_week:<14}{len(fit_samples):>7}{len(eval_samples):>8}{b_mkt:>11.4f}{b_plt:>14.4f}{f'{a:+.3f} {b:+.3f}':>14}{delta:>+9.2f}pp")

    if total_n > 0:
        agg_mkt = total_mkt_loss / total_n
        agg_plt = total_plt_loss / total_n
        agg_d = (agg_plt - agg_mkt) * 100
        print(f"\n  AGGREGATE      {'':>7}{total_n:>8}{agg_mkt:>11.4f}{agg_plt:>14.4f}{'':>14}{agg_d:>+9.2f}pp")
        print(f"\n  Per-fold mean Δ: {((agg_plt - agg_mkt) * 100):.3f}pp")
        # Standard error of the per-fold delta
        per_fold = []
        for i in range(args.min_fit_weeks, len(weeks)):
            ew = weeks[i]
            fit_samples_w = [s for w in weeks[:i] for s in by_week[w]]
            eval_samples_w = by_week[ew]
            if not eval_samples_w: continue
            ag, bg = fit_platt([s["quote"]["market_mid"] for s in fit_samples_w],
                                [s["outcome_yes"] for s in fit_samples_w])
            mids = [s["quote"]["market_mid"] for s in eval_samples_w]
            outs = [s["outcome_yes"] for s in eval_samples_w]
            plt_preds = [sig(ag + bg * logit(p)) for p in mids]
            per_fold.append(brier(plt_preds, outs) - brier(mids, outs))
        mean_d = sum(per_fold) / len(per_fold)
        var = sum((x - mean_d) ** 2 for x in per_fold) / len(per_fold)
        sd = math.sqrt(var)
        print(f"  Per-fold Δ (mean ± sd): {mean_d * 100:+.3f}pp ± {sd * 100:.3f}pp  (n_folds={len(per_fold)})")


if __name__ == "__main__":
    main()
