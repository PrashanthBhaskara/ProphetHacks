"""Side-by-side comparison of two or more prediction jsonls on the same ground truth.

Useful for measuring the lift of a prompt change, a model swap, or an
aggregator post-processing step.

For each (name, path) pair, computes:
  - Aggregate Brier and ECE (on the intersection of tickers across all files)
  - Per-category Brier
  - Distribution of p_yes (decile bins)
  - Class-conditional mean p_yes (given outcome=YES vs NO)
  - Top disagreements between the first two files

Usage:
    python prep/scripts/compare_predictions.py \\
        --predictions v1=prep/data/predictions/grok_subset1200_politics.jsonl \\
        --predictions v3=prep/data/predictions/grok_subset1200_politics_v3.jsonl \\
        --predictions market=prep/data/predictions/market_subset1200_politics.jsonl \\
        --source subset_1200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.aggregator import load_predictions_jsonl  # noqa: E402
from prep.data import load_local_snapshots, load_subset_100, load_subset_1200  # noqa: E402
from prep.score import brier, ece  # noqa: E402


def _load_samples(source: str):
    if source == "hf":
        return load_subset_100()
    if source == "subset_1200":
        return load_subset_1200()
    return load_local_snapshots()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        action="append",
        default=[],
        help="name=path. Repeat (at least 2).",
    )
    parser.add_argument("--source", choices=("hf", "subset_1200", "local"), default="hf")
    parser.add_argument("--top-disagreements", type=int, default=10)
    args = parser.parse_args()

    if len(args.predictions) < 2:
        parser.error("need at least two --predictions to compare")

    files: dict[str, dict[str, float]] = {}
    for spec in args.predictions:
        name, path = spec.split("=", 1)
        files[name] = load_predictions_jsonl(path)
        print(f"  loaded '{name}': {len(files[name])} preds", flush=True)

    samples = _load_samples(args.source)
    outcomes = {s.event["market_ticker"]: s.outcome for s in samples}
    categories = {s.event["market_ticker"]: (s.event.get("category") or "(unknown)") for s in samples}

    # Restrict to the intersection — apples-to-apples comparison.
    common = set.intersection(*(set(f.keys()) for f in files.values()))
    common &= set(outcomes.keys())
    common = sorted(common)
    print(f"  intersection (with outcomes): {len(common)} tickers")
    print()

    # Per-file Brier + ECE.
    print(f"{'name':<20}{'N':>6}{'Brier':>10}{'ECE':>10}")
    print("-" * 50)
    for name, preds in files.items():
        ps = [preds[t] for t in common]
        os_ = [outcomes[t] for t in common]
        print(f"  {name:<18}{len(common):>6}{brier(ps, os_):>10.4f}{ece(ps, os_):>10.4f}")

    # Per-category breakdown.
    cats_present = sorted({categories[t] for t in common}, key=lambda c: -sum(1 for t in common if categories[t] == c))
    if len(cats_present) > 1:
        print()
        header = f"{'category':<22}{'N':>6}"
        for name in files:
            header += f"  {name + ' Brier':>14}"
        print(header)
        print("-" * len(header))
        for cat in cats_present:
            cat_tickers = [t for t in common if categories[t] == cat]
            if len(cat_tickers) < 3:
                continue
            line = f"{cat:<22}{len(cat_tickers):>6}"
            os_ = [outcomes[t] for t in cat_tickers]
            for name, preds in files.items():
                ps = [preds[t] for t in cat_tickers]
                line += f"  {brier(ps, os_):>14.4f}"
            print(line)

    # Class-conditional means.
    yes_t = [t for t in common if outcomes[t] == 1]
    no_t = [t for t in common if outcomes[t] == 0]
    if yes_t and no_t:
        print()
        print(f"Class balance: YES={len(yes_t)} ({len(yes_t)/len(common)*100:.0f}%), NO={len(no_t)}")
        print()
        print(f"{'name':<20}{'avg p_yes | YES':>20}{'avg p_yes | NO':>20}{'separation':>14}")
        print("-" * 74)
        for name, preds in files.items():
            mu_y = sum(preds[t] for t in yes_t) / len(yes_t)
            mu_n = sum(preds[t] for t in no_t) / len(no_t)
            print(f"  {name:<18}{mu_y:>20.3f}{mu_n:>20.3f}{mu_y - mu_n:>14.3f}")

    # Distribution comparison.
    print()
    print("p_yes distribution (deciles):")
    header = f"  range    "
    for name in files:
        header += f"{name:>6}"
    print(header)
    bucket_counts: dict[str, Counter] = {name: Counter() for name in files}
    for t in common:
        for name, preds in files.items():
            b = min(int(preds[t] * 10), 9)
            bucket_counts[name][b] += 1
    for b in range(10):
        line = f"  {b/10:.1f}-{(b+1)/10:.1f}"
        for name in files:
            line += f"{bucket_counts[name][b]:>6}"
        print(line)

    # Top disagreements between the first two files.
    if len(files) >= 2 and args.top_disagreements > 0:
        names = list(files.keys())
        a, b = names[0], names[1]
        diffs = sorted(
            ((t, files[a][t], files[b][t], outcomes[t]) for t in common),
            key=lambda x: -abs(x[1] - x[2]),
        )
        print()
        print(f"Top {args.top_disagreements} '{a}' vs '{b}' disagreements:")
        print(f"  {'ticker':<40}{a:>8}{b:>8}{'truth':>7}{'who improved':>14}")
        for t, pa, pb, o in diffs[:args.top_disagreements]:
            improved = b if abs(pb - o) < abs(pa - o) else a
            print(f"  {t[:40]:<40}{pa:>8.3f}{pb:>8.3f}{'YES' if o == 1 else 'NO':>7}{improved:>14}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
