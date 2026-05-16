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
    try:
        # Production-style: takes only the event dict.
        out = predict_fn(sample.event)
    except TypeError:
        # Allow baselines that want market_info too.
        out = predict_fn(sample.event, sample.market_info)
    p = float(out["p_yes"])
    return max(0.01, min(0.99, p))


def evaluate(
    predict_fn: PredictFn,
    samples: Sequence[Sample],
    *,
    max_workers: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Run predict_fn over samples and return aggregated metrics.

    With max_workers > 1, predictions run in parallel — useful when the
    bottleneck is an external API call. Order of returned per-sample
    predictions matches input order.
    """
    n = len(samples)
    preds: list[float | None] = [None] * n
    outcomes = [s.outcome for s in samples]
    t0 = time.time()

    if max_workers <= 1:
        for i, s in enumerate(samples):
            preds[i] = _call_predict(predict_fn, s)
            if on_progress:
                on_progress(i + 1, n)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_call_predict, predict_fn, s): i for i, s in enumerate(samples)}
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                preds[i] = fut.result()
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
