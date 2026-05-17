"""Run a sweep of aggregator configurations on the Grok prediction file.

Use this once `prep/data/predictions/grok_subset100.jsonl` (or the
subset_1200 equivalent) lands, to see which post-processing recipe
gives the best Brier.

Configurations tested:
- Grok solo (no aggregator post-processing, sanity check)
- Grok + uniform market shrinkage (α = 0.0, 0.3, 0.5, 0.7)
- Grok + extreme-only shrinkage (threshold 0.05, 0.10, 0.15)
- Grok + isotonic (50/50 split)
- Grok + stub ensemble (Grok + market + always_half + random — proves aggregator wiring)
- Grok + stubs + isotonic + extreme shrinkage (the recommended stack)

Each row prints aggregate Brier/ECE, plus per-category Brier on top
categories so you can see where each move helps/hurts.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.aggregator import (  # noqa: E402
    AggregatorConfig,
    aggregate_all,
    fit_isotonic,
    load_predictions_jsonl,
)
from prep.data import load_subset_100, load_subset_1200  # noqa: E402
from prep.score import brier, ece  # noqa: E402

TOP_CATS = ["Sports", "Politics", "Entertainment", "Crypto", "Mentions", "Companies", "Economics"]


def _load_samples(source: str):
    return load_subset_100() if source == "hf" else load_subset_1200()


def _market_prices(samples) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in samples:
        mi = s.market_info or {}
        if mi.get("yes_ask") is not None and mi.get("no_ask") is not None:
            out[s.event["market_ticker"]] = (mi["yes_ask"] + 100 - mi["no_ask"]) / 200
        elif mi.get("last_price") is not None:
            out[s.event["market_ticker"]] = mi["last_price"] / 100
    return out


def _score(final, outcomes, restrict=None):
    keys = [t for t in final if t in outcomes and (restrict is None or t in restrict)]
    if not keys:
        return 0, float("nan"), float("nan")
    ps = [final[t] for t in keys]
    os_ = [outcomes[t] for t in keys]
    return len(keys), brier(ps, os_), ece(ps, os_)


def _per_cat(final, outcomes, categories, restrict=None) -> dict[str, float]:
    by_cat: dict[str, list[tuple[float, int]]] = {}
    for t, p in final.items():
        if t not in outcomes:
            continue
        if restrict is not None and t not in restrict:
            continue
        c = categories.get(t, "(unknown)")
        by_cat.setdefault(c, []).append((p, outcomes[t]))
    return {c: brier([p for p, _ in rows], [o for _, o in rows]) for c, rows in by_cat.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grok",
        default="prep/data/predictions/grok_subset100.jsonl",
        help="Path to grok prediction jsonl (from run.py --save-predictions).",
    )
    parser.add_argument("--source", choices=("hf", "subset_1200"), default="hf")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.grok).exists():
        print(f"ERROR: {args.grok} not found. Run the openrouter baseline first.")
        return 1

    grok_preds = load_predictions_jsonl(args.grok)
    print(f"Loaded {len(grok_preds)} Grok predictions")

    samples = _load_samples(args.source)
    outcomes = {s.event["market_ticker"]: s.outcome for s in samples}
    market = _market_prices(samples)
    categories = {s.event["market_ticker"]: s.event.get("category") or "(unknown)" for s in samples}

    # Stub legs for the ensemble configs.
    half_preds = {t: 0.5 for t in grok_preds}
    market_preds = {t: market[t] for t in grok_preds if t in market}

    rows: list[tuple[str, dict[str, float], set[str] | None]] = []

    # 1. Grok solo (passes through aggregator with no other models, no calibration).
    final = aggregate_all({"grok": grok_preds}, market, AggregatorConfig())
    rows.append(("grok solo", final, None))

    # 2. Grok + uniform market shrinkage.
    for alpha in (0.3, 0.5, 0.7):
        final = aggregate_all(
            {"grok": grok_preds}, market,
            AggregatorConfig(market_alpha=alpha),
        )
        rows.append((f"grok + market α={alpha}", final, None))

    # 3. Grok + extreme-only shrinkage.
    for thr in (0.05, 0.10, 0.15):
        final = aggregate_all(
            {"grok": grok_preds}, market,
            AggregatorConfig(extreme_shrink_threshold=thr, extreme_shrink_strength=0.7),
        )
        rows.append((f"grok + extreme≤{thr}", final, None))

    # 4. Grok + isotonic (fit on half, eval on other half).
    rng = random.Random(args.seed)
    tickers = sorted(t for t in grok_preds if t in outcomes)
    rng.shuffle(tickers)
    split = len(tickers) // 2
    fit_t = set(tickers[:split])
    eval_t = set(tickers[split:])

    base_final = aggregate_all({"grok": grok_preds}, market, AggregatorConfig())
    iso = fit_isotonic(
        [base_final[t] for t in fit_t],
        [outcomes[t] for t in fit_t],
    )
    final = aggregate_all({"grok": grok_preds}, market, AggregatorConfig(isotonic=iso))
    rows.append(("grok + isotonic (eval half)", final, eval_t))

    # 5. Grok + 3-stub ensemble (validates wiring with real Grok in the mix).
    final = aggregate_all(
        {"grok": grok_preds, "market": market_preds, "half": half_preds},
        market, AggregatorConfig(),
    )
    rows.append(("grok + market + half (logit-pool)", final, None))

    # 6. "Recommended stack": Grok + market leg, isotonic, extreme shrink.
    base = aggregate_all(
        {"grok": grok_preds, "market": market_preds},
        market, AggregatorConfig(),
    )
    iso2 = fit_isotonic(
        [base[t] for t in fit_t if t in base],
        [outcomes[t] for t in fit_t if t in base],
    )
    final = aggregate_all(
        {"grok": grok_preds, "market": market_preds}, market,
        AggregatorConfig(isotonic=iso2, extreme_shrink_threshold=0.10, extreme_shrink_strength=0.7),
    )
    rows.append(("grok + market + iso + extreme (eval half)", final, eval_t))

    # Print table.
    print()
    header = f"{'config':<45}{'N':>6}{'Brier':>10}{'ECE':>10}"
    for cat in TOP_CATS:
        header += f"{cat[:7]:>9}"
    print(header)
    print("-" * len(header))
    for label, final, restrict in rows:
        n, b, e = _score(final, outcomes, restrict=restrict)
        line = f"{label:<45}{n:>6}{b:>10.4f}{e:>10.4f}"
        cat_briers = _per_cat(final, outcomes, categories, restrict=restrict)
        for cat in TOP_CATS:
            v = cat_briers.get(cat)
            line += f"{'':>9}" if v is None else f"{v:>9.4f}"
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
