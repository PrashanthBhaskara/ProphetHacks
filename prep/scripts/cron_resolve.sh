#!/bin/bash
# Cron entrypoint: resolve newly-settled markets, then refresh the eval pack.
# Either step failing won't stop the other — visibility > strict ordering.

set -u
cd "$(dirname "$0")/.."

PY=/opt/homebrew/bin/python3

echo "=== resolve $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"$PY" scripts/resolve.py || echo "[warn] resolve.py exited non-zero"

echo "=== consolidate $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"$PY" scripts/consolidate.py || echo "[warn] consolidate.py exited non-zero"

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
