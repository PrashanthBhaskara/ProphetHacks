"""Multi-rollout web-search Grok predictor.

Per MIRAI (arxiv 2407.01231 §3.2 / Fig 4a): self-consistency with K rollouts
turns weaker LLMs into stronger ones. Mistral-7B's F1 on 2nd-level relation
prediction went from 6.6 (single sample) to 33.9 (Max@K=10), MATCHING GPT-4o.

For our binary forecasting task: run web-search Grok K times at temperature
0.7 (each rollout's search may surface different evidence and reasoning),
then average the p_yes values. This is distinct from a reasoning-variant
model — same model, multiple samples.

Costs ~K× the single-rollout cost. K=3 is a reasonable trade-off; K=5 if
budget allows.

Returns the AVERAGE p_yes plus per-rollout values for inspection.
"""

from __future__ import annotations

import os
import sys

from .openrouter_websearch import predict as _ws_predict


def predict(
    event: dict,
    market_info: dict | None = None,
    *,
    snapshot_date: str | None = None,
) -> dict:
    """Run K web-search rollouts and average. K controlled by OPENROUTER_K (default 3)."""
    K = int(os.environ.get("OPENROUTER_K", "3"))
    # Use the existing websearch predictor and rely on its temperature/randomness.
    # Each call independently invokes web search + Grok generation.
    p_yes_list = []
    deviated_count = 0
    cost_total = 0.0
    sources_all = []
    for i in range(K):
        # Vary the seed by setting model temperature env per rollout if desired;
        # we keep default 0.3 (from openrouter_websearch) — sampling diversity
        # comes from the web plugin's search ordering and Grok's stochasticity.
        try:
            r = _ws_predict(event, market_info, snapshot_date=snapshot_date)
        except Exception as e:
            sys.stderr.write(f"[ws_multi] rollout {i+1}/{K} failed: {e}\n")
            continue
        p_yes_list.append(float(r.get("p_yes", 0.5)))
        if r.get("deviated"):
            deviated_count += 1
        cost_total += float(r.get("cost_usd", 0.0) or 0.0)
        srcs = r.get("sources_used", [])
        if isinstance(srcs, list):
            sources_all.extend(srcs[:2])

    if not p_yes_list:
        return {"p_yes": (_market_mid(market_info) or 0.5), "rationale": "all rollouts failed",
                "k": K, "rollouts": [], "cost_usd": 0.0}

    # Arithmetic mean is fine for K small; for K large logit-pool would be smoother.
    p_yes = sum(p_yes_list) / len(p_yes_list)
    p_yes = max(0.01, min(0.99, p_yes))
    # Also report the standard deviation across rollouts as a confidence proxy.
    if len(p_yes_list) > 1:
        mean = sum(p_yes_list) / len(p_yes_list)
        sd = (sum((p - mean) ** 2 for p in p_yes_list) / len(p_yes_list)) ** 0.5
    else:
        sd = 0.0
    return {
        "p_yes": p_yes,
        "rollouts": p_yes_list,
        "rollout_sd": sd,
        "deviated_count": deviated_count,
        "sources_used": sources_all[:6],
        "rationale": f"{len(p_yes_list)} rollouts, mean={p_yes:.3f}, sd={sd:.3f}, {deviated_count} deviated",
        "cost_usd": cost_total,
    }


def _market_mid(market_info):
    if not market_info:
        return None
    yes_ask = market_info.get("yes_ask")
    no_ask = market_info.get("no_ask")
    last_price = market_info.get("last_price")
    if yes_ask is not None and no_ask is not None and (yes_ask + no_ask) > 0:
        return (yes_ask + (100 - no_ask)) / 200
    if last_price is not None:
        return last_price / 100
    return None
