"""Run zero-shot v3 prompt across multiple frontier LLMs on a 2026 sample.

Same prompt as openrouter_zero_shot.py (bidir + market in prompt + multi-candidate hint).
Outputs per-model jsonl files for downstream ensemble tests.

Usage:
    python prep/scripts/test_multi_models.py \\
        --sample prep_handoff/Kalshitopvolmarkets/samples/llm_calls_x10_seed42.jsonl \\
        --models google/gemini-2.5-flash anthropic/claude-sonnet-4.6 openai/gpt-5 x-ai/grok-4.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prep.baselines.openrouter_zero_shot import predict as zs_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("prep/data/predictions"))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    samples = [json.loads(l) for l in args.sample.read_text().splitlines() if l.strip()]
    samples = [s for s in samples if s.get("outcome_yes") in (0, 1)]
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} samples", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        model_slug = model.replace("/", "__")
        out_path = args.out_dir / f"grok_2026_{model_slug}_x10.jsonl"
        print(f"\n=== {model} → {out_path.name} ===", flush=True)

        os.environ["OPENROUTER_MODEL"] = model
        out_file = out_path.open("w", buffering=1)

        def do_one(i_s):
            i, s = i_s
            e, mp = s["event"], s["market_packet"].get("kalshi", {})
            mi = {}
            if mp.get("yes_ask") is not None: mi["yes_ask"] = round(mp["yes_ask"] * 100)
            if mp.get("no_ask") is not None: mi["no_ask"] = round(mp["no_ask"] * 100)
            if mp.get("last_price") is not None and mp.get("last_price"): mi["last_price"] = round(mp["last_price"] * 100)
            try:
                r = zs_predict(e, mi)
                return i, s, r
            except Exception as ex:
                return i, s, {"p_yes": 0.5, "rationale": f"fail: {ex}"}

        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(do_one, (i, s)) for i, s in enumerate(samples)]
            for f in as_completed(futs):
                i, s, r = f.result()
                out_file.write(json.dumps({
                    "market_ticker": s["ticker"],
                    "event_ticker": s["event"].get("event_ticker", ""),
                    "category": s["event"]["category"],
                    "p_yes": max(0.01, min(0.99, r["p_yes"])),
                    "outcome": s["outcome_yes"],
                }) + "\n")
                done += 1
                if done % max(1, len(samples) // 10) == 0 or done == len(samples):
                    print(f"  {done}/{len(samples)} ({time.time()-t0:.0f}s)", flush=True)
        out_file.close()
        print(f"  Done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
