#!/usr/bin/env bash
# Locked-in live trading config. Run this for the actual eval phase.
#
# Strategy choice rationale (see STRATEGY_FINDINGS.md):
#   - `RebalancingStrategy(max_spread=1.02)` is the universal winner in
#     the official subset_1200 backtests
#   - Anthropic Claude (Sonnet 4.6) is the paper's #1 model by trading
#     returns; Sonnet 4.6 should be at least as good as the paper's
#     Claude Opus 4.1
#   - 96 ticks ≈ 24 hours of market windows — enough for a meaningful
#     ROI measurement
#
# Per CONSTRAINTS.md the eval phase is self-funded (the $50 OR grant
# is build-phase only). Estimated cost: 96 ticks × ~$0.05/tick ≈ $5.
#
# Requires:
#   - ai_prophet pip-installed
#   - PA_SERVER_URL and PA_SERVER_API_KEY in env
#   - ANTHROPIC_API_KEY (direct) or OPENROUTER_API_KEY in env
#
# Usage:
#   ./trade_eval.sh                    # default 96 ticks
#   ./trade_eval.sh --max-ticks 24     # smoke run
#   ./trade_eval.sh --slug fresh_v2    # override slug for a re-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present (gitignored)
if [[ -f "$SCRIPT_DIR/../.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/../.env"
  set +a
fi

# Required env vars
: "${PA_SERVER_URL:?Set PA_SERVER_URL — eval phase only; build phase doesn't need it}"
: "${PA_SERVER_API_KEY:?Set PA_SERVER_API_KEY — eval phase only}"

# Configurable
SLUG="${PROPHETHACKS_SLUG:-prophethacks_trade_v1}"
MAX_TICKS="${PROPHETHACKS_MAX_TICKS:-96}"
MODEL="${PROPHETHACKS_MODEL:-anthropic:claude-sonnet-4}"

exec prophet trade eval run \
  -m "$MODEL" \
  --slug "$SLUG" \
  --max-ticks "$MAX_TICKS" \
  "$@"
