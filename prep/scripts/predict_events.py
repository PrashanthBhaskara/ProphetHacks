"""Run a registered baseline against a live events.json from `prophet forecast events`.

Unlike `run.py` (which scores against resolved outcomes for backtest),
this just runs the predictor over a live event slate and saves
predictions for the aggregator. No metrics, no outcomes.

Usage:
    prophet forecast events -o events.json
    python prep/scripts/predict_events.py \\
        --events events.json \\
        --baseline openrouter \\
        --workers 8 \\
        -o prep/data/submission/grok.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASELINES = {
    "always_half": "prep.baselines.always_half",
    "claude": "prep.baselines.claude_zero_shot",
    "grok": "prep.baselines.grok_zero_shot",
    "openrouter": "prep.baselines.openrouter_zero_shot",
    "openrouter_websearch": "prep.baselines.openrouter_websearch",
    "openrouter_websearch_multi": "prep.baselines.openrouter_websearch_multi",
    "openrouter_deferring": "prep.baselines.openrouter_deferring",
    "openrouter_trust_extreme": "prep.baselines.openrouter_trust_extreme",
    "grok_filtered": "prep.baselines.grok_filtered",
    "calibrated_market": "prep.baselines.calibrated_market",
    "favorite_longshot": "prep.baselines.favorite_longshot",
    "multi_feat_logreg": "prep.baselines.multi_feat_logreg",
    # 'market' is omitted — needs market_info which a live event doesn't carry.
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, help="events.json from `prophet forecast events`.")
    parser.add_argument("--baseline", required=True, choices=list(BASELINES.keys()))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "--fetch-market",
        action="store_true",
        help="Fetch current Kalshi market_info per ticker and pass it to predict(). "
             "Paper §4.2.2 shows market-in-prompt drops Brier 0.235 → 0.173. "
             "Recommended for any predictor that accepts a market_info arg "
             "(openrouter_zero_shot, etc.).",
    )
    args = parser.parse_args()

    events = json.loads(Path(args.events).read_text())
    print(f"Loaded {len(events)} events", flush=True)

    predict = importlib.import_module(BASELINES[args.baseline]).predict

    # Pre-fetch market_info for all tickers up front (single pass, sequential).
    # Kalshi is rate-limited; fetching upfront also means we don't double-fetch
    # if multiple worker threads ask for the same ticker.
    market_info_by_ticker: dict[str, dict] = {}
    if args.fetch_market:
        from prep.kalshi import get_market
        n_evt = len(events)
        print(f"Fetching Kalshi market_info for {n_evt} tickers...", flush=True)
        for i, evt in enumerate(events):
            t = evt.get("market_ticker")
            if not t:
                continue
            try:
                mi = get_market(t)
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"[kalshi] {t}: {e}\n")
                continue
            if mi:
                market_info_by_ticker[t] = mi
            if (i + 1) % max(1, n_evt // 20) == 0:
                print(f"  market_info: {i + 1}/{n_evt}", flush=True)
        print(f"  got market_info for {len(market_info_by_ticker)}/{n_evt}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file = out_path.open("w", buffering=1)  # line-buffered → crash-safe

    def _do_one(event: dict) -> tuple[dict, dict | None]:
        try:
            mi = market_info_by_ticker.get(event.get("market_ticker", ""))
            # Try the wider (event, market_info) signature first; fall back to
            # event-only for predictors that don't accept it.
            try:
                result = predict(event, mi)
            except TypeError:
                result = predict(event)
            return event, result
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[predict] {event.get('market_ticker','?')}: {e}\n")
            return event, None

    t0 = time.time()
    done = 0
    n = len(events)
    try:
        if args.workers <= 1:
            for event in events:
                event, result = _do_one(event)
                if result is not None:
                    save_file.write(json.dumps({
                        "market_ticker": event["market_ticker"],
                        "category": event.get("category"),
                        "p_yes": max(0.01, min(0.99, float(result["p_yes"]))),
                        "rationale": result.get("rationale", ""),
                    }) + "\n")
                done += 1
                if done % max(1, n // 20) == 0 or done == n:
                    print(f"  {done}/{n}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_do_one, e) for e in events]
                for fut in as_completed(futures):
                    event, result = fut.result()
                    if result is not None:
                        save_file.write(json.dumps({
                            "market_ticker": event["market_ticker"],
                            "category": event.get("category"),
                            "p_yes": max(0.01, min(0.99, float(result["p_yes"]))),
                            "rationale": result.get("rationale", ""),
                        }) + "\n")
                    done += 1
                    if done % max(1, n // 20) == 0 or done == n:
                        print(f"  {done}/{n}", flush=True)
    finally:
        save_file.close()

    elapsed = time.time() - t0
    print(f"\nDone: {done}/{n} in {elapsed:.1f}s → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
