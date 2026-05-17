#!/bin/bash
# Build a "data-only" submission using NO LLM calls.
#
# Uses `favorite_longshot` baseline (Sports-only 2-feature logreg with 0.7/0.3
# market blend, validated walk-forward at N=9500 with Δ −0.38pp Brier vs raw
# market, paired-bootstrap 95% CI [−0.51, −0.25], P(better)=100%).
# Non-Sports markets pass through raw market price.
#
# Day-of submission flow:
#   1. prophet forecast events -o events.json
#   2. bash prep/scripts/data_only_submission.sh events.json submission.json
#   3. prophet forecast submit --submission submission.json
#
# Expected Brier on Sports-heavy 2026-like distribution: ~0.205 (vs market 0.210)
# Expected Brier on subset_1200-like distribution: ~market (model is neutral
#   on OOD data — see feedback_final_submission_plan.md for the asymmetric
#   risk analysis).

set -euo pipefail
cd "$(dirname "$0")/.."

EVENTS="${1:-events.json}"
OUT="${2:-submission.json}"
TMP="$(mktemp -d)"

echo "[data_only_submission] events: $EVENTS  →  $OUT"
echo "[data_only_submission] tmp: $TMP"

source ../.venv-pmxt/bin/activate

# Run the best-validated data-only predictor over the live event slate.
# --fetch-market is REQUIRED — predictor needs Kalshi yes_ask/no_ask per market.
python scripts/predict_events.py \
    --events "$EVENTS" \
    --baseline favorite_longshot \
    --fetch-market \
    --workers 1 \
    -o "$TMP/favorite_longshot.jsonl"

# Build submission JSON in the schema prophet forecast submit expects.
# Single leg — the predictor already handles Sports calibration + non-Sports pass-through.
# Extreme-shrink at 0.10 strength 0.5 as a belt-and-suspenders for markets
# the predictor returns very confident on (rare for the 0.7/0.3 blend, but cheap insurance).
python scripts/build_submission.py \
    --events "$EVENTS" \
    --predictions favorite_longshot="$TMP/favorite_longshot.jsonl" \
    --pool logit \
    --fetch-market-prices \
    --extreme-shrink 0.10 \
    --extreme-strength 0.5 \
    -o "$OUT"

echo "[data_only_submission] done. Submit with:"
echo "  prophet forecast submit --submission $OUT"
