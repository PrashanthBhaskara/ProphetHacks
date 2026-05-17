"""Run the ensemble aggregator over per-model prediction jsonls.

Usage:
    python scripts/run_aggregator.py \
        --predictions grok=prep/data/predictions/grok_subset100.jsonl \
        --predictions claude=prep/data/predictions/claude_subset100.jsonl \
        --predictions gpt5=prep/data/predictions/gpt5_subset100.jsonl \
        --predictions gemini=prep/data/predictions/gemini_subset100.jsonl \
        --source hf \
        --isotonic-split 0.5 \
        --market-alpha 0.0 \
        --extreme-shrink 0.10

Prints Brier at each stage (raw pool → +isotonic → +shrinkage) so you
can see which steps actually help on the data you have.
"""

from __future__ import annotations

import argparse
import json
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
from prep.data import load_local_snapshots, load_subset_100, load_subset_1200  # noqa: E402
from prep.score import brier, ece  # noqa: E402


def _load_samples(source: str):
    if source == "hf":
        return load_subset_100()
    if source == "subset_1200":
        return load_subset_1200()
    return load_local_snapshots()


def _samples_to_outcomes(samples) -> dict[str, int]:
    return {s.event["market_ticker"]: s.outcome for s in samples}


def _samples_to_categories(samples) -> dict[str, str]:
    return {s.event["market_ticker"]: (s.event.get("category") or "(unknown)") for s in samples}


def _print_per_category(
    label: str,
    final: dict[str, float],
    outcomes: dict[str, int],
    categories: dict[str, str],
    restrict: set[str] | None = None,
) -> None:
    by_cat: dict[str, list[tuple[float, int]]] = {}
    for ticker, p in final.items():
        if ticker not in outcomes:
            continue
        if restrict is not None and ticker not in restrict:
            continue
        cat = categories.get(ticker, "(unknown)")
        by_cat.setdefault(cat, []).append((p, outcomes[ticker]))
    if len(by_cat) <= 1:
        return
    print(f"  Per-category — {label}:")
    print(f"    {'category':<25}{'N':>6}{'Brier':>10}{'ECE':>10}")
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        rows = by_cat[cat]
        ps = [p for p, _ in rows]
        os_ = [o for _, o in rows]
        print(f"    {cat:<25}{len(rows):>6}{brier(ps, os_):>10.4f}{ece(ps, os_):>10.4f}")


def _samples_to_market_prices(samples) -> dict[str, float]:
    """Extract market p_yes per ticker using same logic as baselines/market.py."""
    out: dict[str, float] = {}
    for s in samples:
        mi = s.market_info or {}
        yes_ask = mi.get("yes_ask")
        no_ask = mi.get("no_ask")
        last_price = mi.get("last_price")
        if yes_ask is not None and no_ask is not None and yes_ask + no_ask > 0:
            p = (yes_ask + (100 - no_ask)) / 200
        elif last_price is not None:
            p = last_price / 100
        else:
            continue
        out[s.event["market_ticker"]] = p
    return out


def _score(final: dict[str, float], outcomes: dict[str, int]) -> tuple[int, float, float]:
    tickers = [t for t in final if t in outcomes]
    p_list = [final[t] for t in tickers]
    o_list = [outcomes[t] for t in tickers]
    return len(tickers), brier(p_list, o_list), ece(p_list, o_list)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        action="append",
        default=[],
        help="name=path. Repeat for each model.",
    )
    parser.add_argument("--source", choices=("hf", "subset_1200", "local"), default="hf")
    parser.add_argument(
        "--pool",
        choices=("logit", "arithmetic"),
        default="logit",
        help="Pooling method.",
    )
    parser.add_argument(
        "--isotonic-split",
        type=float,
        default=0.0,
        help="Fraction of tickers used to FIT isotonic (rest is evaluated). 0 = no isotonic.",
    )
    parser.add_argument(
        "--market-alpha",
        type=float,
        default=0.0,
        help="Uniform market shrinkage weight (0 = none).",
    )
    parser.add_argument(
        "--extreme-shrink",
        type=float,
        default=0.0,
        help="Threshold for extreme shrinkage (e.g. 0.10). 0 = disabled.",
    )
    parser.add_argument(
        "--extreme-strength",
        type=float,
        default=0.7,
        help="How strongly to pull toward market at extremes (0-1).",
    )
    parser.add_argument(
        "--category-weights",
        default=None,
        help="JSON file mapping category -> {model: weight}. Produced by "
             "fit_category_weights.py. Overrides any --model-weights for "
             "those categories.",
    )
    parser.add_argument(
        "--save-submission",
        default=None,
        help="Write final {market_ticker, p_yes} jsonl to PATH.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.predictions:
        parser.error("at least one --predictions name=path is required")

    # Parse predictions.
    predictions: dict[str, dict[str, float]] = {}
    for spec in args.predictions:
        if "=" not in spec:
            parser.error(f"--predictions must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        predictions[name] = load_predictions_jsonl(path)
        print(f"  loaded {len(predictions[name])} preds for '{name}' from {path}")

    samples = _load_samples(args.source)
    outcomes = _samples_to_outcomes(samples)
    market_prices = _samples_to_market_prices(samples)
    categories = _samples_to_categories(samples)
    print(f"  loaded {len(samples)} samples, {len(outcomes)} outcomes, {len(market_prices)} market prices")

    cat_weights = None
    if args.category_weights:
        cat_weights = json.loads(Path(args.category_weights).read_text())
        print(f"  loaded per-category weights for {len(cat_weights)} categories from {args.category_weights}")

    # Stage 1: raw pool (with per-category weights if provided), no calibration.
    config = AggregatorConfig(pool=args.pool, model_weights_per_category=cat_weights)
    raw_final = aggregate_all(predictions, market_prices, config, categories=categories)
    n, b, e = _score(raw_final, outcomes)
    stage1_label = f"{args.pool} pool"
    if cat_weights:
        stage1_label += " + per-category weights"
    stage1_label += ", no calibration"
    print(f"\nStage 1 ({stage1_label}):  n={n}  Brier={b:.4f}  ECE={e:.4f}")
    _print_per_category("Stage 1", raw_final, outcomes, categories)

    # Stage 2: + isotonic, fit on a split.
    if args.isotonic_split > 0:
        rng = random.Random(args.seed)
        scored_tickers = [t for t in raw_final if t in outcomes]
        rng.shuffle(scored_tickers)
        split_idx = int(len(scored_tickers) * args.isotonic_split)
        fit_tickers = set(scored_tickers[:split_idx])
        eval_tickers = set(scored_tickers[split_idx:])

        fit_preds = [raw_final[t] for t in fit_tickers]
        fit_outs = [outcomes[t] for t in fit_tickers]
        iso = fit_isotonic(fit_preds, fit_outs)
        config_iso = AggregatorConfig(pool=args.pool, isotonic=iso, model_weights_per_category=cat_weights)
        iso_final = aggregate_all(predictions, market_prices, config_iso, categories=categories)
        eval_p = [iso_final[t] for t in eval_tickers]
        eval_o = [outcomes[t] for t in eval_tickers]
        b_iso = brier(eval_p, eval_o)
        e_iso = ece(eval_p, eval_o)
        print(f"Stage 2 (+ isotonic, fit on {len(fit_tickers)} / eval on {len(eval_tickers)}):  Brier={b_iso:.4f}  ECE={e_iso:.4f}")
        _print_per_category("Stage 2 (eval half)", iso_final, outcomes, categories, restrict=eval_tickers)
        # Carry forward.
        config = config_iso
        final = iso_final
    else:
        final = raw_final
        eval_tickers = None  # type: ignore

    # Stage 3: + market shrinkage.
    if args.market_alpha > 0 or args.extreme_shrink > 0:
        config_shrunk = AggregatorConfig(
            pool=config.pool,
            isotonic=config.isotonic,
            model_weights_per_category=cat_weights,
            market_alpha=args.market_alpha,
            extreme_shrink_threshold=args.extreme_shrink,
            extreme_shrink_strength=args.extreme_strength,
        )
        shrunk_final = aggregate_all(predictions, market_prices, config_shrunk, categories=categories)
        if args.isotonic_split > 0:
            eval_p = [shrunk_final[t] for t in eval_tickers]
            eval_o = [outcomes[t] for t in eval_tickers]
        else:
            n2, _, _ = _score(shrunk_final, outcomes)
            eval_p = [shrunk_final[t] for t in shrunk_final if t in outcomes]
            eval_o = [outcomes[t] for t in shrunk_final if t in outcomes]
        b_sh = brier(eval_p, eval_o)
        e_sh = ece(eval_p, eval_o)
        flags = []
        if args.market_alpha > 0:
            flags.append(f"α={args.market_alpha}")
        if args.extreme_shrink > 0:
            flags.append(f"extreme≤{args.extreme_shrink}@strength={args.extreme_strength}")
        print(f"Stage 3 (+ market shrinkage [{', '.join(flags)}]):  Brier={b_sh:.4f}  ECE={e_sh:.4f}")
        _print_per_category(
            "Stage 3",
            shrunk_final,
            outcomes,
            categories,
            restrict=eval_tickers if args.isotonic_split > 0 else None,
        )
        final = shrunk_final

    if args.save_submission:
        out_path = Path(args.save_submission)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for ticker, p in sorted(final.items()):
                f.write(json.dumps({"market_ticker": ticker, "p_yes": round(p, 6)}) + "\n")
        print(f"\nSaved submission ({len(final)} predictions) → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
