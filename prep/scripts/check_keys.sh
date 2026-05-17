#!/bin/bash
# Quick check on whether either LLM key is alive.
# Returns 0 if at least one works; 1 if both dead.
cd "$(dirname "$0")/.."
set -a; source .env 2>/dev/null; set +a

or_ok=0
xai_ok=0

if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  # Real-call test — auth/key endpoint says "alive" even when balance is 0
  resp=$(curl -s -X POST https://openrouter.ai/api/v1/chat/completions \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"x-ai/grok-4.20","max_tokens":3,"messages":[{"role":"user","content":"ok"}]}' 2>/dev/null)
  if echo "$resp" | grep -q '"error"'; then
    short=$(echo "$resp" | head -c 200)
    echo "OPENROUTER: dead ($short)"
  else
    echo "OPENROUTER: alive (real completion succeeded)"
    or_ok=1
  fi
else
  echo "OPENROUTER: no key set"
fi

if [ -n "${XAI_API_KEY:-}" ]; then
  resp=$(curl -s -X POST https://api.x.ai/v1/chat/completions \
    -H "Authorization: Bearer $XAI_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"grok-4.3","messages":[{"role":"user","content":"ok"}],"max_tokens":5}' 2>/dev/null)
  if echo "$resp" | grep -q '"error"\|"code"'; then
    echo "XAI: dead ($(echo "$resp" | head -c 200))"
  else
    echo "XAI: alive"
    xai_ok=1
  fi
else
  echo "XAI: no key set"
fi

if [ "$or_ok" = "1" ] || [ "$xai_ok" = "1" ]; then
  exit 0
else
  exit 1
fi
