"""Run a baseline against the public 100-event subset.

Usage:
    python scripts/run.py always_half
    python scripts/run.py market
    python scripts/run.py claude          # requires ANTHROPIC_API_KEY
    python scripts/run.py claude --workers 8
    python scripts/run.py grok            # requires XAI_API_KEY
    python scripts/run.py market --category Sports
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.data import filter_by_category, load_local_snapshots, load_subset_100  # noqa: E402
from prep.eval import evaluate  # noqa: E402


BASELINES = {
    "always_half": "prep.baselines.always_half",
    "market": "prep.baselines.market",
    "claude": "prep.baselines.claude_zero_shot",
    "grok": "prep.baselines.grok_zero_shot",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", choices=BASELINES.keys())
    parser.add_argument("--source", choices=("hf", "local"), default="hf",
                        help="hf = 100-event HF subset; local = our own Kalshi snapshots")
    parser.add_argument("--category", default=None, help="filter to one category")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="cap sample count")
    args = parser.parse_args()

    samples = load_subset_100() if args.source == "hf" else load_local_snapshots()
    if args.source == "local" and not samples:
        print("No local snapshots with resolved outcomes yet. "
              "Run scripts/snapshot.py and (after markets close) scripts/resolve.py.")
        return 0
    if args.category:
        samples = filter_by_category(samples, args.category)
    if args.limit:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} samples")
    predict = importlib.import_module(BASELINES[args.baseline]).predict

    def progress(done: int, n: int) -> None:
        if done % max(1, n // 20) == 0 or done == n:
            print(f"  {done}/{n}")

    result = evaluate(predict, samples, max_workers=args.workers, on_progress=progress)

    print()
    print(f"Baseline: {args.baseline}")
    print(f"N: {result['n']}")
    print(f"Brier: {result['brier']:.4f}   (random=0.25, paper market baseline=0.187)")
    print(f"ECE:   {result['ece']:.4f}    (paper market baseline=0.069)")
    print(f"Time:  {result['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
