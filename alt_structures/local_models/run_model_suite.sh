#!/usr/bin/env bash
# Runs all six schemes (eo, vendler, halliday, srl, discourse) through one
# local model, on a balanced sample (per_cell clauses from each of the 27
# EO consensus cells — small balanced samples are what this repo's own
# unique_exemplars.py found gives a clean signal here; large random ones
# don't).
set -euo pipefail
cd "$(dirname "$0")"

MODEL="$1"
THREADS="${2:-2}"
PER_CELL="${3:-5}"
IN="../../run_2026-03-15_122636/classified.jsonl"

for SCHEME in eo vendler halliday srl discourse; do
  echo "=== $MODEL / $SCHEME ==="
  python3 classify_local.py --model "$MODEL" --scheme "$SCHEME" --in "$IN" \
    --balanced-by consensus --per-cell "$PER_CELL" --threads "$THREADS" \
    --out "../results/${MODEL}_${SCHEME}.jsonl"
done
echo "=== $MODEL done ==="
