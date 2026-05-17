"""Evaluation harness.

A `predict_fn` is any callable matching the production contract:

    predict_fn(event: dict) -> dict           # {"p_yes": float, "rationale": str}

Some baselines need additional context (the market snapshot) that the
production agent doesn't see directly. Those expose a wider signature and
we adapt at call time. See baselines/market.py for an example.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

from .data import Sample
from .score import brier, ece

PredictFn = Callable[..., dict]


def _call_predict(predict_fn: PredictFn, sample: Sample) -> float:
    # Prefer the wider signature: baselines that opt in to market_info
    # (paper §4.2.2 shows this drops Brier 0.235 → 0.173). Fall back to
    # the production-style event-only signature for baselines that don't.
    try:
        out = predict_fn(sample.event, sample.market_info)
    except TypeError:
        out = predict_fn(sample.event)
    p = float(out["p_yes"])
    return max(0.01, min(0.99, p))


def evaluate(
    predict_fn: PredictFn,
    samples: Sequence[Sample],
    *,
    max_workers: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
    on_result: Callable[[Sample, float], None] | None = None,
) -> dict:
    """Run predict_fn over samples and return aggregated metrics.

    With max_workers > 1, predictions run in parallel — useful when the
    bottleneck is an external API call. Order of returned per-sample
    predictions matches input order.

    `on_result(sample, p_yes)` fires once per successfully-predicted
    sample, in completion order. Use it for incremental sinks (writing
    predictions to disk as they land) so a crash mid-run doesn't lose
    everything.
    """
    n = len(samples)
    preds: list[float | None] = [None] * n
    outcomes = [s.outcome for s in samples]
    t0 = time.time()

    if max_workers <= 1:
        for i, s in enumerate(samples):
            preds[i] = _call_predict(predict_fn, s)
            if on_result:
                on_result(s, preds[i])
            if on_progress:
                on_progress(i + 1, n)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_call_predict, predict_fn, s): (i, s) for i, s in enumerate(samples)}
            done = 0
            for fut in as_completed(futures):
                i, s = futures[fut]
                preds[i] = fut.result()
                if on_result:
                    on_result(s, preds[i])
                done += 1
                if on_progress:
                    on_progress(done, n)

    p_yes = [p for p in preds if p is not None]
    elapsed = time.time() - t0

    return {
        "n": n,
        "brier": brier(p_yes, outcomes),
        "ece": ece(p_yes, outcomes),
        "elapsed_sec": elapsed,
        "predictions": p_yes,
        "outcomes": outcomes,
    }
