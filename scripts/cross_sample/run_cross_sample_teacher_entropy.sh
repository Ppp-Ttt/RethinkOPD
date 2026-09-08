#!/usr/bin/env bash
# Launch cross-sample evals routed only by teacher entropy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/cross_sample_entropy.py"

# Thresholds to evaluate (manually specify).
THRESHOLDS=(2.0 1.8 1.6 1.4 1.2 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)

for th in "${THRESHOLDS[@]}"; do
    echo "========================================================================"
    echo "  Running cross_sample with TEACHER entropy threshold=$th"
    echo "========================================================================"
    ENTROPY_SOURCE="teacher" ENTROPY_THRESHOLD="$th" python "$PY" "$@"
done
