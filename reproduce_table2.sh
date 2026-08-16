#!/usr/bin/env bash
# Table 2 bootstrap — the frozen invocation (paired session-level, 10,000
# resamples, seed 20260701, best_hit scoring, exact/version family slice).
# Reproduces the paper CIs (Table 2 rows + exact-track slice) verbatim from the shipped gated
# prediction files Regenerate the two prediction files with src/run_gated_eval.py from the
# shipped models (needs the feature matrices; see the README scope note).
#
# Usage: ./reproduce_table2.sh [control_predictions.json] [specialist_predictions.json] [annotations.jsonl] [out.json]
# With no arguments it runs from the shipped ID-only prediction lists.
set -euo pipefail

CONTROL="${1:-evidence/results/gated_predictions_control_devset.idkeyed.json}"
SPECIALIST="${2:-evidence/results/gated_predictions_specialist_devset.idkeyed.json}"
CORRECTIONS="${3:-evidence/annotations/devset_exact_version.idkeyed.jsonl}"
OUT="${4:-work/bootstrap_table2.json}"

mkdir -p "$(dirname "$OUT")"
python src/bootstrap_table2.py \
  --baseline-prediction "$CONTROL" \
  --candidate-prediction "$SPECIALIST" \
  --corrections "$CORRECTIONS" \
  --split devset \
  --scoring best_hit \
  --n-boot 10000 \
  --seed 20260701 \
  --family exact_track_request \
  --family version_duplicate_equivalence \
  --output "$OUT"
echo "wrote $OUT"
