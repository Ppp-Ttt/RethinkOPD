#!/usr/bin/env bash
# Launch cross-sample evals with RANDOM student/teacher routing.
# At every token the teacher is used with probability RANDOM_PROB, drawn
# independently per position and per sequence — an information-free baseline
# for the JS / KL routed variants.
# Each threshold gets its own output dir (run_tag includes randp<val>), so runs
# don't collide. Edit THRESHOLDS below to pick which evals to run.
#
# Everything lives inside main(), invoked on the last line: bash reads the file
# incrementally as it executes, so editing a running top-level script shifts the
# byte offsets under it. Wrapping in a function forces the whole body to be
# parsed up front, making the script safe to edit mid-run.
set -euo pipefail

main() {
    local SCRIPT_DIR PY
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PY="$SCRIPT_DIR/cross_sample_random.py"

    # Teacher-sampling probabilities to evaluate (manually specify).
    local THRESHOLDS=(0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.18 0.20 0.22 0.24)

    local th
    for th in "${THRESHOLDS[@]}"; do
        echo "========================================================================"
        echo "  Running cross_sample with RANDOM_PROB=$th"
        echo "========================================================================"
        RANDOM_PROB="$th" python "$PY" "$@"
    done
}

main "$@"
