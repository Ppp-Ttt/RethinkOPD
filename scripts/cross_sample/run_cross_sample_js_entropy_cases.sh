#!/usr/bin/env bash
# Run the JS-gated entropy cases serially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/cross_sample_js_entropy_cases.py"

# Shared routing thresholds for all experiments.
JS_THRESHOLD=0.08
ENTROPY_THRESHOLD=0.5

# When 1, teacher-routed positions use argmax (greedy) instead of sampling at
# TEMPERATURE; student-routed positions are unaffected. 0 (default) keeps both
# routes on temperature sampling, matching prior runs.
TEACHER_GREEDY=${TEACHER_GREEDY:-1}

# When TEACHER_GREEDY=0, restrict teacher sampling to the teacher's own top-k
# logits (renormalised within that top-k) before multinomial. Only affects
# teacher-routed positions; student still samples from the full vocab at
# TEMPERATURE. 0 (default) = no truncation (original full-vocab temperature
# sampling). Ignored when TEACHER_GREEDY=1.
TEACHER_SAMPLE_TOP_K=${TEACHER_SAMPLE_TOP_K:-16}

# Run seed for the whole experiment. "none" (default) reproduces the
# original per-batch seed derivation (and the original output directory);
# an integer is used as the initial value of that derivation so different
# values yield different rollouts and the same value reproduces. When set,
# the value is appended to the output run-tag so different seeds land in
# separate output directories (APPEND mode would otherwise skip them).
SEED=${SEED:-42}

# hh/hl/lh/ll order: teacher entropy, then student entropy.
# sh: student entropy high; teacher entropy is ignored.
ROUTING_CASES=(hh)

for routing_case in "${ROUTING_CASES[@]}"; do
    echo "========================================================================"
    echo "  JS_THRESHOLD=$JS_THRESHOLD | ENTROPY_THRESHOLD=$ENTROPY_THRESHOLD"
    echo "  ROUTING_CASE=$routing_case | TEACHER_GREEDY=$TEACHER_GREEDY | TEACHER_SAMPLE_TOP_K=$TEACHER_SAMPLE_TOP_K | SEED=$SEED"
    echo "========================================================================"
    JS_THRESHOLD="$JS_THRESHOLD" \
        ENTROPY_THRESHOLD="$ENTROPY_THRESHOLD" \
        TEACHER_GREEDY="$TEACHER_GREEDY" \
        TEACHER_SAMPLE_TOP_K="$TEACHER_SAMPLE_TOP_K" \
        ROUTING_CASE="$routing_case" \
        SEED="$SEED" \
        python "$PY" "$@"
done
