"""Try cheap interventions to beat market on the 2026 contamination-free set.

Tests:
  - Light shrinkage toward 0.5
  - Shrinkage toward observed base rate
  - 2-fold Platt fit (calibrated on 2026 itself)
  - Time-to-close stratification (do we have edge only far from close?)
  - Spread-based confidence weighting

For each: bootstrap Brier + 95% CI, compared to market baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.score import brier  # noqa: E402


def logit(p, eps=1e-4):
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def bootstrap_ci(values, outcomes, n_boot=1500, seed=42):
    n = len(values)
    rng = random.Random(seed)
    briers = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ps = [values[i] for i in idx]
        os_ = [outcomes[i] for i in idx]
        briers.append(sum((p - o) ** 2 for p, o in zip(ps, os_)) / n)
    briers.sort()
    return (sum(briers) / len(briers),
            briers[int(0.025 * n_boot)],
            briers[int(0.975 * n_boot)])


def fit_platt(p_train, y_train, l2=0.5, max_iter=200):
    """1-D Platt: p ≈ sigmoid(a + b·logit(p_train)). IRLS with L2 on slope only."""
    x = [logit(p) for p in p_train]
    # Newton-Raphson
    a, b = 0.0, 1.0
    for _ in range(max_iter):
        # gradient
        z_list = [a + b * xi for xi in x]
        z_list = [max(-30, min(30, z)) for z in z_list]
        mu = [sigmoid(z) for z in z_list]
        # gradient
        g_a = -sum(yi - mi for yi, mi in zip(y_train, mu))
        g_b = -sum((yi - mi) * xi for yi, mi, xi in zip(y_train, mu, x)) + l2 * b
        # hessian
        w = [mi * (1 - mi) for mi in mu]
        h_aa = sum(w)
        h_ab = sum(wi * xi for wi, xi in zip(w, x))
        h_bb = sum(wi * xi * xi for wi, xi in zip(w, x)) + l2
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        # Newton step
        d_a = (h_bb * g_a - h_ab * g_b) / det
        d_b = (h_aa * g_b - h_ab * g_a) / det
        step = max(1.0, abs(d_a), abs(d_b))
        a -= d_a / step
        b -= d_b / step
        if max(abs(d_a), abs(d_b)) < 1e-7:
            break
    return a, b


def apply_platt(p, a, b):
    return sigmoid(a + b * logit(p))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = [json.loads(l) for l in args.sample.read_text().splitlines() if l.strip()]
    samples = [s for s in samples if s.get("outcome_yes") in (0, 1)]
    print(f"Loaded {len(samples)} resolved samples", flush=True)

    outcomes = [int(s["outcome_yes"]) for s in samples]
    mids = [float(s["quote"]["market_mid"]) for s in samples]
    spreads = [float(s["quote"]["spread"]) for s in samples]
    mtc = [float(s["minutes_to_close"]) for s in samples]
    yes_rate = sum(outcomes) / len(outcomes)
    print(f"  YES rate: {yes_rate:.3f}, avg market_mid: {sum(mids)/len(mids):.3f}")
    print(f"  avg spread: {sum(spreads)/len(spreads):.3f}")
    print(f"  minutes-to-close: median={sorted(mtc)[len(mtc)//2]:.0f}, min={min(mtc):.0f}, max={max(mtc):.0f}")

    rng = random.Random(args.seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    half = len(indices) // 2
    fit_idx = set(indices[:half])
    eval_idx = set(indices[half:])

    eval_o = [outcomes[i] for i in sorted(eval_idx)]
    eval_mids = [mids[i] for i in sorted(eval_idx)]

    methods: dict[str, list[float]] = {
        "market alone": eval_mids,
    }

    # Intervention 1: shrink toward 0.5 by various α
    for alpha in (0.05, 0.10, 0.15, 0.20):
        methods[f"market shrunk to 0.5 (α={alpha})"] = [
            (1 - alpha) * p + alpha * 0.5 for p in eval_mids
        ]

    # Intervention 2: shrink toward base rate
    fit_rate = sum(outcomes[i] for i in fit_idx) / len(fit_idx)
    for alpha in (0.05, 0.10, 0.15):
        methods[f"market shrunk to base rate {fit_rate:.2f} (α={alpha})"] = [
            (1 - alpha) * p + alpha * fit_rate for p in eval_mids
        ]

    # Intervention 3: Platt calibration fit on fit_idx
    fit_p = [mids[i] for i in fit_idx]
    fit_y = [outcomes[i] for i in fit_idx]
    a, b = fit_platt(fit_p, fit_y)
    methods[f"Platt (a={a:+.3f}, b={b:+.3f})"] = [apply_platt(p, a, b) for p in eval_mids]

    # Intervention 4: condition on time-to-close. Only shrink markets that are far from close.
    # Hypothesis: prices closer to settlement are more informative; far ones have more LLM-edge.
    eval_mtc = [mtc[i] for i in sorted(eval_idx)]
    median_mtc = sorted(mtc)[len(mtc) // 2]
    methods[f"shrunk only if mtc > {median_mtc:.0f}min (α=0.10)"] = [
        (0.9 * p + 0.1 * 0.5) if m > median_mtc else p
        for p, m in zip(eval_mids, eval_mtc)
    ]

    # Intervention 5: spread-weighted confidence smoothing
    eval_spreads = [spreads[i] for i in sorted(eval_idx)]
    methods["smooth-by-spread (linear)"] = [
        sigmoid((1.0 - min(0.5, sp)) * logit(p))
        for p, sp in zip(eval_mids, eval_spreads)
    ]

    print()
    print(f"{'method':<45}{'N':>5}{'Brier':>9}{'95% CI':>22}")
    print("-" * 84)
    baseline_mean = None
    for label, preds in methods.items():
        m, lo, hi = bootstrap_ci(preds, eval_o)
        b_act = brier(preds, eval_o)
        if label == "market alone":
            baseline_mean = m
        delta = f"  {(m - baseline_mean) * 100:+.2f}pp" if baseline_mean is not None else ""
        print(f"{label:<45}{len(preds):>5}{b_act:>9.4f}  [{lo:.4f}, {hi:.4f}]{delta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
