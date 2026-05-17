"""Exhaustive audit of every saved Grok prediction file.

For each (predictor, dataset, prompt_variant) combination, report:
  - N
  - Brier with bootstrap 95% CI
  - ECE
  - Pearson correlation with market price
  - Avg |grok - market|
  - When Grok strongly disagrees with market, who wins?
  - Per-category Brier breakdown

Plus head-to-head comparisons on the strict intersection of tickers.

This is the work I should have done at the start before celebrating
Subset-100 numbers. Running it now to know exactly what each prediction
file is worth.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.data import load_subset_100, load_subset_1200  # noqa: E402
from prep.score import brier, ece  # noqa: E402


def bootstrap_ci(preds, outcomes, n_boot=2000, seed=42):
    n = len(preds)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    briers = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ps = [preds[i] for i in idx]
        os = [outcomes[i] for i in idx]
        briers.append(sum((p - o) ** 2 for p, o in zip(ps, os)) / n)
    briers.sort()
    return briers[n_boot // 2], briers[int(0.025 * n_boot)], briers[int(0.975 * n_boot)]


def load_preds(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["market_ticker"]] = r
    return out


def market_mid(market_info):
    if not market_info:
        return None
    yes_ask = market_info.get("yes_ask")
    no_ask = market_info.get("no_ask")
    last_price = market_info.get("last_price")
    if yes_ask is not None and no_ask is not None and (yes_ask + no_ask) > 0:
        return (yes_ask + (100 - no_ask)) / 200
    if last_price is not None:
        return last_price / 100
    return None


def analyze(name, pred_dict, samples, restrict_category=None):
    """For one Grok prediction file, compute everything."""
    rows = []
    for s in samples:
        t = s.event.get("market_ticker", "")
        if t not in pred_dict:
            continue
        if restrict_category and s.event.get("category") != restrict_category:
            continue
        mp = market_mid(s.market_info)
        if mp is None:
            continue
        rows.append({
            "ticker": t,
            "category": s.event.get("category"),
            "grok": pred_dict[t]["p_yes"],
            "market": mp,
            "outcome": s.outcome,
        })
    if not rows:
        print(f"  {name}: no rows in intersection")
        return None

    n = len(rows)
    grok_ps = [r["grok"] for r in rows]
    mkt_ps = [r["market"] for r in rows]
    outs = [r["outcome"] for r in rows]

    g_brier = brier(grok_ps, outs)
    m_brier = brier(mkt_ps, outs)
    g_ece = ece(grok_ps, outs)
    m_ece = ece(mkt_ps, outs)

    g_b_m, g_b_lo, g_b_hi = bootstrap_ci(grok_ps, outs)
    m_b_m, m_b_lo, m_b_hi = bootstrap_ci(mkt_ps, outs)

    # Pearson r grok vs market
    g_mean = sum(grok_ps) / n
    m_mean = sum(mkt_ps) / n
    num = sum((g - g_mean) * (m - m_mean) for g, m in zip(grok_ps, mkt_ps))
    den_g = math.sqrt(sum((g - g_mean) ** 2 for g in grok_ps))
    den_m = math.sqrt(sum((m - m_mean) ** 2 for m in mkt_ps))
    pearson = num / (den_g * den_m) if den_g > 0 and den_m > 0 else float("nan")

    # Disagreement
    avg_diff = sum(abs(g - m) for g, m in zip(grok_ps, mkt_ps)) / n
    pct_within_5 = sum(1 for g, m in zip(grok_ps, mkt_ps) if abs(g - m) < 0.05) / n * 100

    # When grok deviates >0.10, who wins?
    deviates = [(r["grok"], r["market"], r["outcome"]) for r in rows if abs(r["grok"] - r["market"]) > 0.10]
    if deviates:
        grok_better = sum(1 for g, m, o in deviates if abs(g - o) < abs(m - o))
        grok_better_pct = grok_better / len(deviates) * 100
    else:
        grok_better_pct = float("nan")

    # Skill beyond market: residual correlation
    g_resid = [g - m for g, m in zip(grok_ps, mkt_ps)]
    o_resid = [o - m for m, o in zip(mkt_ps, outs)]
    rg_mean = sum(g_resid) / n
    ro_mean = sum(o_resid) / n
    num2 = sum((g - rg_mean) * (o - ro_mean) for g, o in zip(g_resid, o_resid))
    den_g2 = math.sqrt(sum((g - rg_mean) ** 2 for g in g_resid))
    den_o2 = math.sqrt(sum((o - ro_mean) ** 2 for o in o_resid))
    skill_r = num2 / (den_g2 * den_o2) if den_g2 > 0 and den_o2 > 0 else float("nan")

    print(f"  {name}{(' ['+restrict_category+']') if restrict_category else ''}:")
    print(f"    n={n}")
    print(f"    Grok   Brier={g_brier:.4f}  [{g_b_lo:.4f},{g_b_hi:.4f}]  ECE={g_ece:.4f}")
    print(f"    Market Brier={m_brier:.4f}  [{m_b_lo:.4f},{m_b_hi:.4f}]  ECE={m_ece:.4f}")
    print(f"    Δ Brier (Grok - Market) = {(g_brier - m_brier)*100:+.2f}pp")
    print(f"    r(Grok, Market) = {pearson:.4f}   avg|G-M|={avg_diff:.4f}   %within 0.05 of mkt={pct_within_5:.1f}%")
    print(f"    When Grok deviates >0.10 (n={len(deviates)}): Grok-better {grok_better_pct:.1f}%")
    print(f"    Skill-beyond-market (residual corr): {skill_r:+.4f}  (>0 = Grok adds info)")
    return {"n": n, "g_brier": g_brier, "m_brier": m_brier, "delta_pp": (g_brier - m_brier)*100,
            "pearson": pearson, "skill_r": skill_r}


def main():
    files = [
        ("subset_100",  "grok_v1_no_market",      "prep/data/predictions/grok_subset100.jsonl"),
        ("subset_100",  "grok_v3_with_market",    "prep/data/predictions/grok_subset100_v3.jsonl"),
        ("subset_1200", "grok_v1_no_market",      "prep/data/predictions/grok_subset1200_politics.jsonl"),
        ("subset_1200", "grok_v2_multi_cand_only", "prep/data/predictions/grok_subset1200_politics_v2.jsonl"),
        ("subset_1200", "grok_v3_with_market",    "prep/data/predictions/grok_subset1200_politics_v3.jsonl"),
    ]

    s100 = load_subset_100()
    s1200 = load_subset_1200()

    print("=" * 90)
    print("AUDIT OF SAVED GROK PREDICTIONS — known to be contaminated for subset_100/1200")
    print("(events from June-Nov 2025, likely in Grok's training data)")
    print("=" * 90)
    print()

    for dataset, label, path in files:
        if not Path(path).exists():
            print(f"  SKIP {path}: missing")
            continue
        preds = load_preds(path)
        samples = s100 if dataset == "subset_100" else s1200
        print(f"\n[{dataset}] {label} ({len(preds)} preds saved)")
        analyze(label, preds, samples)

    # Head-to-head Politics: v1 vs v2 vs v3 on strict intersection
    print()
    print("=" * 90)
    print("HEAD-TO-HEAD: subset_1200 Politics, all three prompt variants on strict intersection")
    print("=" * 90)

    v1 = load_preds("prep/data/predictions/grok_subset1200_politics.jsonl")
    v2 = load_preds("prep/data/predictions/grok_subset1200_politics_v2.jsonl")
    v3 = load_preds("prep/data/predictions/grok_subset1200_politics_v3.jsonl")
    common = set(v1.keys()) & set(v2.keys()) & set(v3.keys())
    print(f"\nStrict intersection of v1∩v2∩v3: {len(common)} tickers")

    if common:
        rows = []
        for s in s1200:
            t = s.event.get("market_ticker", "")
            if t not in common:
                continue
            mp = market_mid(s.market_info)
            if mp is None:
                continue
            rows.append((t, mp, v1[t]["p_yes"], v2[t]["p_yes"], v3[t]["p_yes"], s.outcome))

        outs = [r[5] for r in rows]
        print(f"  scored on {len(rows)} rows (intersection w/ outcomes)")
        print(f"  {'method':<25}{'Brier':>9}{'ECE':>8}{'95% CI':>22}")
        for name, idx in [("market", 1), ("grok v1 (no market)", 2), ("grok v2 (multi-cand)", 3), ("grok v3 (with market)", 4)]:
            ps = [r[idx] for r in rows]
            b = brier(ps, outs)
            e = ece(ps, outs)
            m, lo, hi = bootstrap_ci(ps, outs)
            print(f"  {name:<25}{b:>9.4f}{e:>8.4f}{f'[{lo:.4f},{hi:.4f}]':>22}")


if __name__ == "__main__":
    main()
