"""Run a baseline against the public 100-event subset.

Usage:
    python scripts/run.py always_half
    python scripts/run.py market
    python scripts/run.py claude          # requires ANTHROPIC_API_KEY
    python scripts/run.py claude --workers 8
    python scripts/run.py grok            # requires XAI_API_KEY
    python scripts/run.py openrouter      # requires OPENROUTER_API_KEY (default model x-ai/grok-4.3)
    python scripts/run.py market --category Sports
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.data import (  # noqa: E402
    filter_by_category,
    load_local_snapshots,
    load_subset_100,
    load_subset_1200,
)
from prep.eval import evaluate  # noqa: E402
from prep.score import brier, ece  # noqa: E402


BASELINES = {
    "always_half": "prep.baselines.always_half",
    "market": "prep.baselines.market",
    "claude": "prep.baselines.claude_zero_shot",
    "grok": "prep.baselines.grok_zero_shot",
    "openrouter": "prep.baselines.openrouter_zero_shot",
    "openrouter_event": "prep.baselines.openrouter_event_aware",
    "fair_price": "prep.baselines.fair_price_v0",
    "calibrated_market": "prep.baselines.calibrated_market",
    "per_series_platt": "prep.baselines.per_series_platt",
    "multi_feat_logreg": "prep.baselines.multi_feat_logreg",
    "favorite_longshot": "prep.baselines.favorite_longshot",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", choices=BASELINES.keys())
    parser.add_argument("--source", choices=("hf", "subset_1200", "local"), default="hf",
                        help="hf = 100-event HF subset; subset_1200 = authoritative organizer benchmark; local = our own Kalshi snapshots")
    parser.add_argument("--category", default=None, help="filter to category (comma-separated for multiple)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="cap sample count")
    parser.add_argument(
        "--save-predictions",
        default=None,
        help="Write per-sample {market_ticker, p_yes, outcome} jsonl to PATH (for aggregator input).",
    )
    args = parser.parse_args()

    if args.source == "hf":
        samples = load_subset_100()
    elif args.source == "subset_1200":
        samples = load_subset_1200()
    else:
        samples = load_local_snapshots()
    if args.source == "local" and not samples:
        print("No local snapshots with resolved outcomes yet. "
              "Run scripts/snapshot.py and (after markets close) scripts/resolve.py.")
        return 0
    if args.category:
        cats = [c.strip() for c in args.category.split(",") if c.strip()]
        if len(cats) == 1:
            samples = filter_by_category(samples, cats[0])
        else:
            samples = [s for s in samples if s.event.get("category") in cats]
    if args.limit:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} samples", flush=True)
    predict = importlib.import_module(BASELINES[args.baseline]).predict

    def progress(done: int, n: int) -> None:
        if done % max(1, n // 20) == 0 or done == n:
            print(f"  {done}/{n}", flush=True)

    # Incremental prediction sink: write each result as soon as it lands
    # so a mid-run crash (network blip, API throttle) doesn't lose work.
    save_file = None
    on_result = None
    if args.save_predictions:
        out_path = Path(args.save_predictions)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file = out_path.open("w", buffering=1)  # line-buffered

        def on_result(sample, p):  # noqa: F811
            save_file.write(json.dumps({
                "market_ticker": sample.event["market_ticker"],
                "category": sample.event.get("category"),
                "p_yes": p,
                "outcome": sample.outcome,
            }) + "\n")

    try:
        result = evaluate(
            predict, samples, max_workers=args.workers,
            on_progress=progress, on_result=on_result,
        )
    finally:
        if save_file is not None:
            save_file.close()

    if args.save_predictions:
        print(f"Saved {len(result['predictions'])} predictions → {args.save_predictions}", flush=True)

    print()
    print(f"Baseline: {args.baseline}")
    print(f"N: {result['n']}")
    print(f"Brier: {result['brier']:.4f}   (random=0.25, paper market baseline=0.187)")
    print(f"ECE:   {result['ece']:.4f}    (paper market baseline=0.069)")
    print(f"Time:  {result['elapsed_sec']:.1f}s")

    # Per-category breakdown — most informative slice. Tells you which
    # categories the predictor adds skill vs where market dominates.
    by_cat: dict[str, list[tuple[float, int]]] = {}
    for sample, p in zip(samples, result["predictions"]):
        cat = sample.event.get("category") or "(unknown)"
        by_cat.setdefault(cat, []).append((p, sample.outcome))
    if len(by_cat) > 1:
        print()
        print("Per-category breakdown:")
        print(f"  {'category':<25}{'N':>6}{'Brier':>10}{'ECE':>10}")
        for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
            rows = by_cat[cat]
            ps = [p for p, _ in rows]
            os = [o for _, o in rows]
            print(f"  {cat:<25}{len(rows):>6}{brier(ps, os):>10.4f}{ece(ps, os):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
