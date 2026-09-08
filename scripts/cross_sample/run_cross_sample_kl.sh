#!/usr/bin/env bash
# Launch cross-sample evals routed by KL divergence.
# Two directions are swept:
#   rkl : KL(student||teacher)
#   fkl : KL(teacher||student)
# Each (direction, threshold) pair gets its own output dir (run_tag includes
# <dir>th<val>), so runs don't collide. Edit DIRECTIONS / THRESHOLDS below to
# pick which evals to run.
#
# Everything lives inside main(), invoked on the last line: bash reads the file
# incrementally as it executes, so editing a running top-level script shifts the
# byte offsets under it. Wrapping in a function forces the whole body to be
# parsed up front, making the script safe to edit mid-run.
set -euo pipefail

main() {
    local SCRIPT_DIR PY
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PY="$SCRIPT_DIR/cross_sample_kl.py"

    # KL directions to evaluate (rkl and/or fkl).
    local DIRECTIONS=(rkl)

    # Thresholds to evaluate (manually specify).
    local THRESHOLDS=(3.0 2.6 2.2 1.8 1.4 1.0)

    local dir th
    for dir in "${DIRECTIONS[@]}"; do
        for th in "${THRESHOLDS[@]}"; do
            echo "========================================================================"
            echo "  Running cross_sample with KL_DIRECTION=$dir KL_THRESHOLD=$th"
            echo "========================================================================"
            KL_DIRECTION="$dir" KL_THRESHOLD="$th" python "$PY" "$@"
        done
    done
}

main "$@"
