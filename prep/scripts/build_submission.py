"""Build a Prophet Arena Submission JSON from per-model prediction jsonls.

This is the day-of submission glue. It takes:

  - events.json (from `prophet forecast events -o events.json`)
  - One or more per-model prediction jsonls (from `run.py --save-predictions`)
  - Optional market-price fetch from Kalshi for shrinkage

and outputs a `submission.json` in the schema that `prophet forecast
submit` expects (see `ai_prophet_core.forecast.schemas.Submission`).

Usage:

    # Fetch tonight's events
    prophet forecast events -o events.json

    # Run each model independently (this saves ticker → p_yes jsonl)
    python prep/scripts/predict_events.py --events events.json --baseline openrouter \\
        --save prep/data/submission/grok.jsonl

    # Build the aggregated submission
    python prep/scripts/build_submission.py \\
        --events events.json \\
        --predictions grok=prep/data/submission/grok.jsonl \\
        --predictions claude=prep/data/submission/claude.jsonl \\
        --predictions gpt5=prep/data/submission/gpt5.jsonl \\
        --predictions gemini=prep/data/submission/gemini.jsonl \\
        --fetch-market-prices \\
        --market-alpha 0.0 --extreme-shrink 0.10 \\
        -o submission.json

    # Submit
    prophet forecast submit --submission submission.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.aggregator import (  # noqa: E402
    AggregatorConfig,
    aggregate_all,
    load_predictions_jsonl,
)
from prep.kalshi import get_market  # noqa: E402

CLAMP_LOW = 0.01
CLAMP_HIGH = 0.99


def _fetch_market_prices(tickers: list[str], pause: float = 0.1) -> dict[str, float]:
    """Pull current Kalshi price for each ticker. Best-effort, skip on errors."""
    import time

    out: dict[str, float] = {}
    for i, ticker in enumerate(tickers):
        try:
            mkt = get_market(ticker)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[kalshi] {ticker}: {e}\n")
            continue
        if not mkt:
            continue
        yes_ask = mkt.get("yes_ask")
        no_ask = mkt.get("no_ask")
        last_price = mkt.get("last_price")
        if yes_ask is not None and no_ask is not None and (yes_ask + no_ask) > 0:
            out[ticker] = (yes_ask + (100 - no_ask)) / 200
        elif last_price is not None:
            out[ticker] = last_price / 100
        if pause > 0 and i < len(tickers) - 1:
            time.sleep(pause)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        required=True,
        type=str,
        help="events.json file (from `prophet forecast events`).",
    )
    parser.add_argument(
        "--predictions",
        action="append",
        default=[],
        help="name=path. Repeat for each model.",
    )
    parser.add_argument(
        "--pool",
        choices=("logit", "arithmetic"),
        default="logit",
    )
    parser.add_argument(
        "--fetch-market-prices",
        action="store_true",
        help="Query Kalshi for current market price per ticker (enables shrinkage).",
    )
    parser.add_argument(
        "--market-prices",
        default=None,
        help="Pre-fetched market prices jsonl (ticker → p_yes). Skip Kalshi calls.",
    )
    parser.add_argument("--market-alpha", type=float, default=0.0)
    parser.add_argument("--extreme-shrink", type=float, default=0.0)
    parser.add_argument("--extreme-strength", type=float, default=0.7)
    parser.add_argument(
        "--model-weights",
        default=None,
        help="JSON dict of model name → weight (uniform if omitted).",
    )
    parser.add_argument(
        "--category-weights",
        default=None,
        help="JSON file mapping category -> {model: weight} (from "
             "fit_category_weights.py). Per-category weights override "
             "--model-weights for those categories.",
    )
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--skip-closed", action="store_true", default=True)
    args = parser.parse_args()

    if not args.predictions:
        parser.error("at least one --predictions name=path is required")

    events = json.loads(Path(args.events).read_text())
    print(f"Loaded {len(events)} events from {args.events}", flush=True)

    now = datetime.now(UTC)
    if args.skip_closed:
        kept = []
        for e in events:
            close_str = e.get("close_time", "")
            if not close_str:
                kept.append(e)
                continue
            try:
                close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                if close_dt > now:
                    kept.append(e)
            except (ValueError, TypeError):
                kept.append(e)
        if len(kept) < len(events):
            print(f"  skipped {len(events) - len(kept)} already-closed events", flush=True)
        events = kept

    event_tickers = [e["market_ticker"] for e in events if e.get("market_ticker")]

    predictions: dict[str, dict[str, float]] = {}
    for spec in args.predictions:
        name, path = spec.split("=", 1)
        predictions[name] = load_predictions_jsonl(path)
        n_match = sum(1 for t in event_tickers if t in predictions[name])
        print(f"  loaded '{name}': {len(predictions[name])} preds, {n_match} match event slate", flush=True)

    # Market prices.
    market_prices: dict[str, float] = {}
    if args.market_prices:
        market_prices = load_predictions_jsonl(args.market_prices)
        print(f"  loaded {len(market_prices)} market prices from {args.market_prices}", flush=True)
    elif args.fetch_market_prices:
        print(f"  fetching market prices for {len(event_tickers)} tickers from Kalshi...", flush=True)
        market_prices = _fetch_market_prices(event_tickers)
        print(f"  got {len(market_prices)} market prices", flush=True)

    # Aggregator config.
    model_weights = json.loads(args.model_weights) if args.model_weights else None
    category_weights = (
        json.loads(Path(args.category_weights).read_text())
        if args.category_weights else None
    )
    config = AggregatorConfig(
        pool=args.pool,
        model_weights=model_weights,
        model_weights_per_category=category_weights,
        market_alpha=args.market_alpha,
        extreme_shrink_threshold=args.extreme_shrink,
        extreme_shrink_strength=args.extreme_strength,
    )

    # Build ticker → category from the events file so per-category weights apply.
    categories: dict[str, str] = {}
    for e in events:
        t = e.get("market_ticker")
        c = e.get("category")
        if t and c:
            categories[t] = c

    final = aggregate_all(predictions, market_prices, config, categories=categories)
    print(f"  aggregator produced {len(final)} aggregated p_yes", flush=True)

    # Build submission, restricted to events in the slate.
    rationale_summary = f"ensemble({','.join(predictions.keys())}, pool={args.pool})"
    if category_weights:
        rationale_summary += ", per-cat weights"
    if args.market_alpha > 0:
        rationale_summary += f", market_α={args.market_alpha}"
    if args.extreme_shrink > 0:
        rationale_summary += f", extreme≤{args.extreme_shrink}"

    sub_predictions = []
    skipped = 0
    for e in events:
        ticker = e.get("market_ticker")
        if not ticker or ticker not in final:
            skipped += 1
            continue
        p = max(CLAMP_LOW, min(CLAMP_HIGH, final[ticker]))
        sub_predictions.append({
            "market_ticker": ticker,
            "p_yes": round(p, 6),
            "rationale": rationale_summary,
        })

    if not sub_predictions:
        sys.stderr.write("ERROR: no predictions matched the event slate. Check that the per-model jsonls cover the live tickers.\n")
        return 1

    submission = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "predictions": sub_predictions,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(submission, indent=2))
    print(f"\nSubmission ({len(sub_predictions)} predictions, {skipped} skipped) → {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
