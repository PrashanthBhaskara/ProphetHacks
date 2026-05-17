#!/bin/bash
# LIVE-DAY submission pipeline.
# Uses Grok-4.20 + web search + trust_extreme + favorite_longshot ensemble.
#
# Day-of:
#   1. prophet forecast events -o events.json
#   2. bash prep/scripts/live_submission.sh events.json submission.json
#   3. prophet forecast submit --submission submission.json
#
# Strategy rationale (all in feedback_final_submission_plan.md):
#   - Grok zero-shot alone: WORSE than market (+0.53 to +1.35pp Brier) on
#     contamination-free 2026 backtest
#   - Web-search Grok backtest is gameable (Grok finds same-day game results)
#     but the paper's Fig 5 documents 0.169 Brier (sources+market) vs 0.187
#     (market alone) on truly unresolved events — live eval has no
#     contamination since events haven't happened yet
#   - grok_filtered (NEW BEST): trust_extreme Grok with noise-removal filter
#     that skips Grok on (top-10% volume OR extreme prices ≤0.15/≥0.85 OR ATP
#     tennis). On contamination-free backtest N=190: Brier 0.2033 vs market
#     0.2118 (Δ -0.85pp, P(better)=96.7%). 3× better than favorite_longshot
#     and 14× better than raw trust_extreme. ~30% of calls fall through to
#     market price (free), reducing API cost too.
#   - favorite_longshot is our data-only winner (-0.30pp Brier vs market,
#     paired CI [-0.50, -0.11] on Kalshitopvolmarkets N=9500; tied to market
#     on NonBinaryMarkets N=19068 — asymmetric risk)
#   - 3-leg logit-pool ensemble: websearch (live skill) + trust_extreme
#     (LLM calibration) + favorite_longshot (data calibration). Maximum
#     diversity, hedges contamination risk on websearch, cheap insurance.

set -euo pipefail
cd "$(dirname "$0")/.."

EVENTS="${1:-events.json}"
OUT="${2:-submission.json}"
TMP="$(mktemp -d)"

echo "[live_submission] events: $EVENTS  →  $OUT"
echo "[live_submission] tmp: $TMP"

source ../.venv-pmxt/bin/activate

# Verify key alive before spending
if ! bash scripts/check_keys.sh > /dev/null 2>&1; then
    echo "[live_submission] WARNING: keys check failed. Falling back to data-only path."
    python scripts/predict_events.py \
        --events "$EVENTS" --baseline favorite_longshot --fetch-market \
        --workers 1 -o "$TMP/favorite_longshot.jsonl"
    python scripts/build_submission.py \
        --events "$EVENTS" \
        --predictions favorite_longshot="$TMP/favorite_longshot.jsonl" \
        --pool logit --fetch-market-prices \
        --extreme-shrink 0.10 --extreme-strength 0.5 \
        -o "$OUT"
    # Sanity check fallback submission too
    python -c "
import json, sys
sys.path.insert(0, 'ai-prophet/packages/core')
from ai_prophet_core.forecast.schemas import Submission
s = Submission.model_validate(json.loads(open('$OUT').read()))
n = len(s.predictions); n_half = sum(1 for p in s.predictions if abs(p.p_yes - 0.5) < 0.001)
print(f'[fallback sanity] {n} predictions, {n_half} at exactly 0.5')
if n == 0: sys.exit('FATAL: no predictions')
if n_half > n * 0.7:
    print(f'WARN: fallback gave {n_half}/{n} predictions at 0.5 — Kalshi market_info fetch likely failed for these tickers')
" || { echo "[live_submission] FATAL fallback sanity check failed."; exit 2; }
    echo "[live_submission] FALLBACK done. prophet forecast submit --submission $OUT"
    exit 0
fi

# Run all three predictors in parallel
echo "[live_submission] Running Grok-4.20 + web search..."
OPENROUTER_MODEL="${OPENROUTER_MODEL:-x-ai/grok-4.20}" \
python scripts/predict_events.py \
    --events "$EVENTS" --baseline openrouter_websearch --fetch-market \
    --workers 4 -o "$TMP/grok_websearch.jsonl" &
WS_PID=$!

echo "[live_submission] Running Grok-filtered (trust-extreme + noise removal)..."
OPENROUTER_MODEL="${OPENROUTER_MODEL:-x-ai/grok-4.20}" \
python scripts/predict_events.py \
    --events "$EVENTS" --baseline grok_filtered --fetch-market \
    --workers 4 -o "$TMP/grok_filtered.jsonl" &
TE_PID=$!

echo "[live_submission] Running favorite_longshot (data-only)..."
python scripts/predict_events.py \
    --events "$EVENTS" --baseline favorite_longshot --fetch-market \
    --workers 1 -o "$TMP/favorite_longshot.jsonl" &
FL_PID=$!

wait $WS_PID
wait $TE_PID
wait $FL_PID

echo "[live_submission] All three predictors done."

# === ROBUSTNESS CHECK ===
# Detect if any leg silently fell back to 0.5 for many markets (e.g. API rate-limit,
# key spend cap exceeded, all retries exhausted). If >30% of a leg's predictions
# are exactly 0.5, that leg is dead — drop it from the ensemble rather than letting
# it pollute the submission.
PREDS_TO_USE=""
for leg in grok_websearch grok_filtered favorite_longshot; do
    f="$TMP/${leg}.jsonl"
    if [ ! -s "$f" ]; then
        echo "[live_submission] WARN: $leg produced no output, dropping from ensemble"
        continue
    fi
    total=$(wc -l < "$f" | tr -d ' ')
    half=$(grep -c '"p_yes": 0.5' "$f" || true)
    if [ "$total" -gt 0 ] && [ "$half" -gt 0 ]; then
        pct=$(( half * 100 / total ))
        if [ "$pct" -gt 30 ]; then
            echo "[live_submission] WARN: $leg has $half/$total = ${pct}% p_yes=0.5 fallback (key probably dead). Dropping from ensemble."
            continue
        fi
    fi
    PREDS_TO_USE="$PREDS_TO_USE --predictions $leg=$f"
done

if [ -z "$PREDS_TO_USE" ]; then
    echo "[live_submission] FATAL: all three legs failed. Cannot build submission."
    exit 1
fi

echo "[live_submission] Using legs:$PREDS_TO_USE"
echo "[live_submission] Building ensemble submission..."

# Logit-pool of surviving legs + extreme-shrink belt-and-suspenders.
python scripts/build_submission.py \
    --events "$EVENTS" \
    $PREDS_TO_USE \
    --pool logit --fetch-market-prices \
    --extreme-shrink 0.10 --extreme-strength 0.5 \
    -o "$OUT"

# === FINAL SUBMISSION SANITY CHECK ===
# Make sure the submission JSON is valid, has predictions, and isn't all 0.5
if ! python -c "
import json, sys
sys.path.insert(0, 'ai-prophet/packages/core')
from ai_prophet_core.forecast.schemas import Submission
s = Submission.model_validate(json.loads(open('$OUT').read()))
n = len(s.predictions)
n_half = sum(1 for p in s.predictions if abs(p.p_yes - 0.5) < 0.001)
n_extreme = sum(1 for p in s.predictions if p.p_yes < 0.02 or p.p_yes > 0.98)
print(f'[sanity] {n} predictions, {n_half} at exactly 0.5, {n_extreme} at extremes')
if n == 0: sys.exit('FATAL: no predictions')
if n_half > n * 0.5: sys.exit(f'FATAL: {n_half}/{n} predictions are exactly 0.5 — pipeline broken')
"; then
    echo "[live_submission] FATAL sanity check failed. Inspect $OUT before submitting."
    exit 2
fi

echo "[live_submission] done. Submit with:"
echo "  prophet forecast submit --submission $OUT"
echo "  (rough budget: \$0.05-0.15 per market via web-search; check actual cost on OpenRouter dashboard)"
