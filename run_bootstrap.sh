#!/usr/bin/env bash
# Table 1 bootstrap — the frozen invocation (paired session-level, 10,000
# resamples, seed 20260701, best_hit scoring, exact/version family slice).
# Reproduces the three CIs of Table 1 verbatim from the shipped gated
# prediction files (see README for how to regenerate the predictions).
#
# Usage: ./run_bootstrap.sh <control_predictions.json> <specialist_predictions.json> [corrections.jsonl] [out.json]
set -euo pipefail

CONTROL="${1:?control (official-label) gated prediction file}"
SPECIALIST="${2:?specialist (request-supplemented) gated prediction file}"
CORRECTIONS="${3:-evidence/corrections/request_corrections_devset_exact_version_v09.idkeyed.jsonl}"
OUT="${4:-work/bootstrap_table1.json}"

mkdir -p "$(dirname "$OUT")"
python src/bootstrap_request_corrected_dev.py \
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
