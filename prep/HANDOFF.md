# Grok-leg handoff — Prophet Hacks forecast submission

What this package contains, how to use it, and how to ensemble your model's predictions with ours.

## TL;DR for the day-of

```bash
# 1. Fetch the live event slate from Prophet Arena
prophet forecast events -o events.json

# 2. Run the Grok-leg pipeline (websearch + trust-extreme-filter + data-only)
bash prep/scripts/live_submission.sh events.json /tmp/grok_leg_submission.json

# 3. (Or ensemble with teammate legs — see "Ensembling with your leg" below)

# 4. Submit
prophet forecast submit --submission /tmp/grok_leg_submission.json
```

Expected Brier on Sports-heavy live data: **~0.20** (vs market ~0.21). Validated under temporal walk-forward CV at -0.5pp absolute Brier, P(better)=98% on a 60-market held-out set.

## What's in here

### The recommended Grok leg

Three predictors logit-pooled together:

| Predictor | What it does |
|---|---|
| `openrouter_websearch` | Grok-4.20 with web search plugin. Live news/context. Costs ~$0.03–0.05/market. Paper-validated for live (not backtest). |
| `grok_filtered` | Grok-4.20 + "trust market at extremes" prompt + skip filter for high-volume, extreme-priced, or ATP tennis markets. Validated -0.5pp Brier vs market. |
| `favorite_longshot` | Pure data-only Sports calibration (2-feature Platt with favorite-longshot correction). No LLM, $0. Validated -0.30pp Brier on KTV walk-forward. |

Fallback: if all LLM legs fail, `live_submission.sh` automatically degrades to `favorite_longshot` only.

### Files for teammates to read first

- `prep/scripts/live_submission.sh` — day-of pipeline
- `prep/scripts/data_only_submission.sh` — LLM-free safe fallback
- `prep/scripts/build_submission.py` — ensembling tool (THE primary integration point)
- `prep/scripts/predict_events.py` — runs any registered baseline against an events.json
- `prep/src/prep/baselines/` — all the registered baselines

## Ensembling with YOUR leg

If you're running your own forecaster (Claude / GPT-5 / Gemini / custom), here's how to combine it with the Grok leg.

### Step 1: Produce predictions in this format

JSONL with one row per event, exactly these fields:

```json
{"market_ticker": "KXNBAGAME-26MAY17LALBOS-LAL", "p_yes": 0.62, "rationale": "<optional>"}
```

Constraints (from the official `ai_prophet_core.forecast.schemas.Prediction`):
- `p_yes` must be in `[0.01, 0.99]` — clip if needed
- `market_ticker` must match the live event ticker exactly
- `rationale` optional but useful for debugging

### Step 2: Drop it in `prep/data/predictions/your_leg.jsonl` and ensemble

```bash
# After live_submission.sh has produced its Grok leg files in $TMP:
python prep/scripts/build_submission.py \
    --events events.json \
    --predictions grok_websearch=$TMP/grok_websearch.jsonl \
    --predictions grok_filtered=$TMP/grok_filtered.jsonl \
    --predictions favorite_longshot=$TMP/favorite_longshot.jsonl \
    --predictions YOUR_NAME=path/to/your_leg.jsonl \
    --pool logit \
    --fetch-market-prices \
    --extreme-shrink 0.10 --extreme-strength 0.5 \
    -o final_submission.json
```

`--pool logit` is recommended (geometric mean of probabilities, well-behaved at extremes). `arithmetic` is also available.

`--extreme-shrink 0.10` is belt-and-suspenders: when the market is at ≤0.10 or ≥0.90, shrink the ensemble's prediction toward market with strength 0.5. Reduces tail losses.

### Step 3: Alternative — plug your model in as a registered baseline

If you want your model to run inside this pipeline (so `live_submission.sh` calls it natively):

1. Create `prep/src/prep/baselines/your_baseline.py` with this exact interface:

```python
def predict(event: dict, market_info: dict | None = None) -> dict:
    """Returns {"p_yes": float, "rationale": str}"""
    # your logic here
    return {"p_yes": 0.5, "rationale": "..."}
```

2. Register it in `prep/scripts/predict_events.py`:

```python
BASELINES = {
    ...
    "your_baseline": "prep.baselines.your_baseline",
}
```

3. Test it:

```bash
python prep/scripts/predict_events.py \
    --events events.json --baseline your_baseline --fetch-market \
    --workers 4 -o /tmp/your_preds.jsonl
```

## Backtesting your leg (before adding to ensemble)

Use our validated contamination-free 2026 sample (post-Grok-training-cutoff):

```bash
# Generate a sample from the post-cutoff data
python prep/scripts/sample_nonbinary_2026.py \
    --data-dir prep_handoff/NonBinaryMarkets \
    --samples-per-ticker 1 \
    --out-jsonl /tmp/sample_2026.jsonl

# Or use one of the existing samples
# prep_handoff/Kalshitopvolmarkets/samples/llm_calls_x10_seed42.jsonl  (N=190 Sports-heavy)
# prep_handoff/NonBinaryMarkets/samples/llm_calls_nbm_x1_seed42.jsonl  (N=19068 Sports + Other)

# Score market alone + favorite_longshot baselines
python prep/scripts/eval_2026_sample.py \
    --sample prep_handoff/Kalshitopvolmarkets/samples/llm_calls_x10_seed42.jsonl \
    --grok-preds /tmp/your_preds.jsonl
```

Reports Brier with bootstrap 95% CIs per the BACKTEST.md protocol.

**Critical:** σ ≈ 0.011 at N=200. Any Brier delta < 0.02 is statistically indistinguishable. Do not advertise wins under that threshold.

## Environment setup

Required env vars (put in `prep/.env` — copy `prep/.env.example` as a starting point):

```
# ── REQUIRED ──
OPENROUTER_API_KEY=sk-or-v1-...   # Grok legs (websearch + grok_filtered)
PA_SERVER_API_KEY=prophet_...     # `prophet forecast` CLI

# ── EXACT TESTED SETTINGS (do not change unless re-validating) ──
OPENROUTER_MODEL=x-ai/grok-4.20   # `live_submission.sh` overrides any other default to this
OPENROUTER_TEMPERATURE=0.7         # for grok_filtered / trust_extreme leg
OPENROUTER_BIDIR=1                 # bi-directional prompting (ask P(YES) + 1−P(NO), avg)
OPENROUTER_WEB_RESULTS=5           # web-search leg: number of results to retrieve
GROK_VOLUME_THRESHOLD=4000         # skip Grok above this volume (let market speak)
GROK_EXTREME_THRESHOLD=0.15        # skip Grok when mid ≤ 0.15 or ≥ 0.85
```

### Per-leg model settings (what was actually tested)

These are what produced the validated Brier numbers. **Don't change them without re-running the backtest.**

| Leg | Model | Temperature | max_tokens | bidir | other |
|---|---|---|---|---|---|
| `openrouter_websearch` | `x-ai/grok-4.20` (env-overridden) | **0.3** (hardcoded in [openrouter_websearch.py:119](src/prep/baselines/openrouter_websearch.py:119)) | **800** (hardcoded) | n/a (single-direction) | `web_results=5`, `plugins=[{id:"web"}]` |
| `grok_filtered` → `openrouter_trust_extreme` | `x-ai/grok-4.20` (env-overridden) | **0.7** (env, default 0.7) | **300** (hardcoded) | **1** (bi-directional) | Skips Grok if vol > 4000 OR mid ∉ (0.15, 0.85) OR ticker ∈ KXATPMATCH*/KXATPCHALLENGERMATCH* — falls through to market |
| `favorite_longshot` | n/a (pure data) | n/a | n/a | n/a | 2-feature Sports Platt: `bias=+0.1015, logit_mid=+0.7161, abs_mid=-0.6702`, 0.7·model + 0.3·market blend |

**Why two different temperatures?** Web-search calls cost real money per query and we want deterministic-ish reasoning over the retrieved context (0.3). The trust-extreme leg uses 0.7 because we average the YES-direction and NO-direction calls (bidir=1), and a touch of sampling variance there reduces per-call hedging without hurting calibration.

**LLM sampling is non-deterministic.** Even with all settings pinned, a re-run will give slightly different per-market predictions than my test runs. The *aggregate* Brier is what's validated (paired-bootstrap CI on N=190), not any individual market.

Dependencies:
```bash
cd prep && pip install -r requirements.txt
# plus numpy + scikit-learn + pyarrow (for backtest scripts)
```

Verify keys before any expensive run:
```bash
bash prep/scripts/check_keys.sh
```

## Key findings (so teammates know what's validated)

1. **Zero-shot Grok alone is WORSE than market** on contamination-free 2026 data (+0.5 to +1.5pp Brier). Don't ship Grok alone.

2. **trust_extreme + filter is the Grok-leg winner**. The "trust market at extremes" prompt + skip-Grok-on-(high-vol OR extreme-priced OR ATP) gives -0.5pp on temporal holdout, P=98%.

3. **Subset-100 / Subset-1200 are contaminated** for Grok. Pre-Aug 2025 events are in training data. The Brier 0.05 numbers you might see on those datasets are memorization, not skill.

4. **Web search works on LIVE events** (post-cutoff) but contaminates backtests (returns post-event news). Paper Fig 5 documents -0.4pp on truly novel events.

5. **Cross-family ensembling is the highest-EV path** per Prophet Arena paper §C.5. Adding Claude/GPT-5/Gemini legs to our ensemble is exactly what this handoff is designed for.

6. **Data-only baselines are robust** (favorite_longshot, multi_feat_logreg). Use them as the safety floor — they never lose to market by more than the noise floor.

## Common pitfalls

- **API key with no balance** silently falls back to 0.5 predictions, devastating Brier. Always run `check_keys.sh` first. The hardened `live_submission.sh` now detects this and drops the dead leg.
- **Subset-100 / Subset-1200 numbers are not real** — they're contaminated. Always cross-check on `prep_handoff/Kalshitopvolmarkets/` or `prep_handoff/NonBinaryMarkets/`.
- **Web search on historical backtest = leakage**. Only use for live events.
- **Calibration models don't transfer across distributions.** Subset_1200 coefficients applied to 2026 data give +0.64pp WORSE Brier. Refit on the right distribution.

## Repository layout

```
prep/
├── scripts/
│   ├── live_submission.sh         ← DAY-OF entry point (LLM ensemble)
│   ├── data_only_submission.sh    ← Fallback if LLM dead
│   ├── build_submission.py        ← Ensemble + final JSON (PRIMARY integration point for teammates)
│   ├── predict_events.py          ← Single-baseline runner over events.json
│   ├── check_keys.sh              ← Key liveness check (real completion test)
│   ├── eval_2026_sample.py        ← Backtest harness
│   ├── walk_forward_per_series.py ← Walk-forward CV
│   └── ... (more backtest tools)
└── src/prep/
    ├── baselines/
    │   ├── grok_filtered.py            ← THE Grok-leg winner
    │   ├── openrouter_trust_extreme.py ← Anti-hedging prompt
    │   ├── openrouter_websearch.py     ← Web search variant
    │   ├── openrouter_websearch_multi.py ← K-rollout (untested at scale)
    │   ├── openrouter_event_aware.py   ← Multi-candidate sibling prompting
    │   ├── openrouter_deferring.py     ← Self-deferring (tested, no edge)
    │   ├── openrouter_zero_shot.py     ← Base prompt
    │   ├── favorite_longshot.py        ← 2-feature data-only winner
    │   ├── multi_feat_logreg.py        ← 6-feature data-only alternative
    │   ├── calibrated_market.py
    │   ├── per_series_platt.py
    │   ├── fair_price_v0.py
    │   └── market.py
    ├── aggregator.py              ← logit-pool + isotonic + extreme-shrink
    ├── kalshi.py                  ← Live Kalshi price fetch
    ├── eval.py
    └── data.py
```

## What's NOT included / what teammates should bring

- Your own LLM leg (Claude / GPT-5 / Gemini predictions in the format above)
- Your own OpenRouter / Anthropic / OpenAI API keys if running parallel
- Optional: domain-specific data sources (sports injury feeds, weather APIs, etc.) — paper §4.2.2 says these matter for non-Sports markets

## Questions

DM Victor. Reference this file. The validated commit is on `claude/busy-heyrovsky-a38a5d`.
