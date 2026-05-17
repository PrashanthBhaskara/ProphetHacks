"""Fit per-category model weights from a held-out backtest.

The aggregator supports per-category model weights (see
`AggregatorConfig.model_weights_per_category`). This script reads per-model
prediction jsonls + ground-truth samples and produces a weights JSON.

Strategy (intentionally simple — robust under small samples):

  For each category:
    - Compute each model's Brier on the category's tickers.
    - Drop any model whose Brier > kill_threshold (default 0.22 — close to
      random) — it's net-negative signal; assigning it weight only contaminates.
    - For surviving models, weight = (kill_threshold - brier) / sum. Better
      models get more weight; the kill_threshold serves as a soft floor.
  If no model survives, the category gets an empty dict — aggregate_one
  falls back to market_price.

Why not gradient descent / proper log-loss minimization?  Most of these
categories have N < 100. Smooth optimization on tiny holdouts overfits to
noise. Inverse-loss weighting is the standard small-sample answer.

Usage:
    python prep/scripts/fit_category_weights.py \\
        --predictions grok=prep/data/predictions/grok_subset100.jsonl \\
        --predictions claude=prep/data/predictions/claude_subset100.jsonl \\
        --source hf \\
        -o prep/data/category_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.aggregator import load_predictions_jsonl  # noqa: E402
from prep.data import load_local_snapshots, load_subset_100, load_subset_1200  # noqa: E402
from prep.score import brier  # noqa: E402


def _load_samples(source: str):
    if source == "hf":
        return load_subset_100()
    if source == "subset_1200":
        return load_subset_1200()
    return load_local_snapshots()


def _market_price(sample) -> float | None:
    mi = sample.market_info or {}
    if mi.get("yes_ask") is not None and mi.get("no_ask") is not None:
        return (mi["yes_ask"] + 100 - mi["no_ask"]) / 200
    if mi.get("last_price") is not None:
        return mi["last_price"] / 100
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        action="append",
        default=[],
        help="name=path. Repeat for each model. The literal name 'market' is "
             "reserved — it's auto-derived from sample.market_info if not "
             "supplied.",
    )
    parser.add_argument("--source", choices=("hf", "subset_1200", "local"), default="hf")
    parser.add_argument(
        "--kill-threshold",
        type=float,
        default=0.22,
        help="Models with Brier > this on a category get weight 0 for that category.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=20,
        help="Skip categories with fewer than this many scored samples (too noisy).",
    )
    parser.add_argument(
        "--include-market-leg",
        action="store_true",
        help="Add an automatic 'market' model leg derived from sample.market_info. "
             "Most ensembles should have this — the market price is the strongest "
             "single leg per Prophet Arena paper Fig 5.",
    )
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    if not args.predictions and not args.include_market_leg:
        parser.error("need at least one --predictions or --include-market-leg")

    predictions: dict[str, dict[str, float]] = {}
    for spec in args.predictions:
        name, path = spec.split("=", 1)
        if name == "market":
            parser.error("the model name 'market' is reserved; use --include-market-leg or rename")
        predictions[name] = load_predictions_jsonl(path)
        print(f"  loaded '{name}': {len(predictions[name])} preds", flush=True)

    samples = _load_samples(args.source)
    outcomes = {s.event["market_ticker"]: s.outcome for s in samples}
    categories = {s.event["market_ticker"]: (s.event.get("category") or "(unknown)") for s in samples}

    if args.include_market_leg:
        market_preds: dict[str, float] = {}
        for s in samples:
            mp = _market_price(s)
            if mp is not None:
                market_preds[s.event["market_ticker"]] = mp
        predictions["market"] = market_preds
        print(f"  auto-added 'market': {len(market_preds)} preds", flush=True)

    # Bucket tickers by category.
    by_cat: dict[str, list[str]] = {}
    for ticker, cat in categories.items():
        if ticker in outcomes:
            by_cat.setdefault(cat, []).append(ticker)

    weights_per_category: dict[str, dict[str, float]] = {}
    print()
    print(f"{'category':<25}{'N':>6}  per-model Brier (weight)")
    print("-" * 100)

    for cat, tickers in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        if len(tickers) < args.min_n:
            print(f"{cat:<25}{len(tickers):>6}  skipped (N < {args.min_n})")
            continue
        cat_briers: dict[str, float] = {}
        for name, per_ticker in predictions.items():
            ps = [per_ticker[t] for t in tickers if t in per_ticker]
            os_ = [outcomes[t] for t in tickers if t in per_ticker]
            if not ps:
                continue
            cat_briers[name] = brier(ps, os_)

        # Build weights: positive for models below kill_threshold.
        surviving = {n: b for n, b in cat_briers.items() if b < args.kill_threshold}
        if not surviving:
            weights_per_category[cat] = {}
            cells = "  ".join(f"{n}={b:.3f}(0.00)" for n, b in cat_briers.items())
            print(f"{cat:<25}{len(tickers):>6}  {cells}   ⚠ no model below kill_threshold → market-only")
            continue

        raw = {n: (args.kill_threshold - b) for n, b in surviving.items()}
        total = sum(raw.values())
        weights = {n: round(w / total, 4) for n, w in raw.items()}
        # Set zero-weight for models we dropped.
        for n in cat_briers:
            if n not in weights:
                weights[n] = 0.0
        weights_per_category[cat] = weights

        cells = "  ".join(
            f"{n}={cat_briers[n]:.3f}({weights[n]:.2f})"
            for n in sorted(cat_briers, key=lambda x: -weights[x])
        )
        print(f"{cat:<25}{len(tickers):>6}  {cells}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(weights_per_category, indent=2))
    print()
    print(f"Wrote per-category weights for {len(weights_per_category)} categories → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
