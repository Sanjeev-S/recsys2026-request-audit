#!/usr/bin/env bash
# Full-train specialist + matched control (Section 6), local invocation.
#
# This replaces the research repo's cloud launcher with a plain local run of
# the same training command. It requires the production feature matrices and
# group-size files, published as the gated HuggingFace dataset
# sanjeevsuresh/recsys2026-request-audit-matrices (hashes pinned in
# evidence/training/published_matrices.json): set MCRS_EXPLORE_ROOT to a
# directory containing
#   exp/features/features_rerank_train_lt_105k_R54src_F10R54.parquet
#   exp/features/group_sizes_train_lt_105k_R54src.npy
#   exp/features/features_rerank_val_R54src_F10R54.parquet
#   exp/features/group_sizes_val_R54src.npy
#
# Trains both cells (official control, exact_positive_weighted specialist)
# with the frozen protocol settings: preset r54_shallow_topk20, 50 rounds,
# corrected group weight 0.1, exact-request feature appended.
# Hardware note: the recorded run used one A100; CPU works but is slow.
set -euo pipefail

: "${MCRS_EXPLORE_ROOT:?set MCRS_EXPLORE_ROOT to the data root (see header)}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
CACHE="${TRAIN_CACHE_DIR:-$ROOT/work/train_cache}"
mkdir -p "$CACHE"/{dmatrix,feature,label} "$ROOT/work/models"

python src/train_request_supplement.py \
  --feature-set F10_R54SRC \
  --preset r54_shallow_topk20 \
  --train-feature-path "$MCRS_EXPLORE_ROOT/exp/features/features_rerank_train_lt_105k_R54src_F10R54.parquet" \
  --train-group-sizes-path "$MCRS_EXPLORE_ROOT/exp/features/group_sizes_train_lt_105k_R54src.npy" \
  --train-split-name train_lt_105k \
  --train-corrections evidence/annotations/train_lt_exact_version.idkeyed.jsonl \
  --extra-train-feature-path "$MCRS_EXPLORE_ROOT/exp/features/features_rerank_val_R54src_F10R54.parquet" \
  --extra-train-group-sizes-path "$MCRS_EXPLORE_ROOT/exp/features/group_sizes_val_R54src.npy" \
  --extra-train-split-name val \
  --extra-train-corrections evidence/annotations/val_exact_version.idkeyed.jsonl \
  --cells official exact_positive_weighted \
  --corrected-group-weight 0.1 \
  --num-boost-round 50 \
  --skip-val-dmatrix \
  --cache-dmatrix \
  --dmatrix-cache-dir "$CACHE/dmatrix" \
  --add-exact-request-feature \
  --request-feature-cache-dir "$CACHE/feature" \
  --label-cache-dir "$CACHE/label" \
  --output-tag fulltrain_local \
  --model-dir work/models \
  --output work/train_summary.json
echo "models in work/models; summary in work/train_summary.json"
