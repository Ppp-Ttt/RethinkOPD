#!/usr/bin/env bash
# Launch cross-sample evals at different JS_THRESHOLD values.
# Each threshold gets its own output dir (run_tag includes jsth<val>), so runs
# don't collide. Edit THRESHOLDS below to pick which evals to run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/cross_sample.py"

# Thresholds to evaluate (manually specify).
THRESHOLDS=(0.5 0.4 0.3 0.2 0.16 0.14 0.12 0.1 0.08 0.06 0.04 0.02)

for th in "${THRESHOLDS[@]}"; do
    echo "========================================================================"
    echo "  Running cross_sample with JS_THRESHOLD=$th"
    echo "========================================================================"
    JS_THRESHOLD="$th" python "$PY" "$@"
done
