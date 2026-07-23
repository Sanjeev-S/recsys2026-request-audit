"""Run the frozen exact-request protocol on any available split.

This is the transfer/untouched-split runner for the paper. It does not need
gold labels or correction sidecars. It reports only non-oracle behavior:

  * baseline official-model ranking,
  * transparent full-pool exact-request promotion,
  * optional learned request-slice specialist, gated by visible exact matches.

Inputs are visible dialogue, catalog metadata, the reranker candidate pool, and
model scores. The script does not inspect official labels, corrected metrics,
adjudication labels, or synthetic thought traces.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from apply_request_exact_postrank import exact_request_ids  # noqa: E402
from request_corrected_rerank_train import (  # noqa: E402
    EXACT_REQUEST_FEATURE,
)
from request_correction_labels import build_catalog_index, load_sessions  # noqa: E402
from rerank_train import BATCH, FEATURE_SETS, group_sizes_for, parquet_path  # noqa: E402


def load_booster(path: Path) -> xgb.Booster:
    if not path.exists():
        raise FileNotFoundError(path)
    bst = xgb.Booster()
    bst.load_model(path)
    return bst


def rank_of_any(predicted: list[str], requested: set[str], k: int) -> int | None:
    for rank, tid in enumerate(predicted[:k], start=1):
        if tid in requested:
            return rank
    return None


def pct(num: int, den: int) -> float:
    return num / den if den else 0.0


def transfer_sessions(split: str):
    from datasets import load_dataset

    if split == "blindset_A":
        return load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test")
    if split == "blindset_B":
        return load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-B", split="test")
    return load_sessions(split)


def transfer_request_directives(split: str, catalog) -> dict[tuple[str, int], dict[str, Any]]:
    directives: dict[tuple[str, int], dict[str, Any]] = {}
    for session in transfer_sessions(split):
        sid = session["session_id"]
        for row in session["conversations"]:
            if row.get("role") != "user":
                continue
            turn = int(row["turn_number"])
            found = exact_request_ids(row.get("content") or "", catalog)
            if not found:
                continue
            requested_ids: list[str] = []
            for ids in found.values():
                requested_ids.extend(ids)
            directives[(sid, turn)] = {
                "requested_titles": sorted(found),
                "requested_title_to_ids": {title: sorted(set(ids)) for title, ids in sorted(found.items())},
                "requested_track_ids": sorted(set(requested_ids)),
                "user_text": row.get("content") or "",
            }
    return directives


def request_feature_cache_stem(path: Path, feature_set: str, split: str) -> str:
    stat = path.stat()
    return f"{feature_set}_{split}_{EXACT_REQUEST_FEATURE}_{stat.st_mtime_ns:x}_{stat.st_size:x}"


def build_or_load_transfer_request_feature_cache(
    *,
    path: Path,
    gs: np.ndarray,
    feature_set: str,
    split: str,
    directives: dict[tuple[str, int], dict[str, Any]],
    cache_dir: Path,
    force: bool,
) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = request_feature_cache_stem(path, feature_set, split)
    feature_path = cache_dir / f"{stem}.npy"
    summary_path = cache_dir / f"{stem}.summary.json"
    if not force and feature_path.exists() and summary_path.exists():
        return {"feature": feature_path, "summary": summary_path}

    requested_by_key = {
        key: set(value.get("requested_track_ids") or [])
        for key, value in directives.items()
    }
    n_rows = int(gs.sum())
    feature = np.zeros(n_rows, dtype=np.float32)
    off = 0
    matched_rows = 0
    matched_groups: set[tuple[str, int]] = set()
    requested_groups: set[tuple[str, int]] = set()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH, columns=["session_id", "turn_number", "track_id"]):
        sids = batch.column("session_id").to_pylist()
        turns = batch.column("turn_number").to_numpy(zero_copy_only=False)
        tids = batch.column("track_id").to_pylist()
        for i, tid in enumerate(tids):
            key = (sids[i], int(turns[i]))
            requested = requested_by_key.get(key)
            if not requested:
                continue
            requested_groups.add(key)
            if tid in requested:
                feature[off + i] = 1.0
                matched_rows += 1
                matched_groups.add(key)
        off += batch.num_rows

    assert off == n_rows
    np.save(feature_path, feature)
    summary = {
        "path": str(path),
        "feature_set": feature_set,
        "split": split,
        "feature_name": EXACT_REQUEST_FEATURE,
        "n_rows": n_rows,
        "n_groups": int(len(gs)),
        "directive_groups": int(len(directives)),
        "requested_groups_in_pool": int(len(requested_groups)),
        "matched_rows": int(matched_rows),
        "matched_groups": int(len(matched_groups)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"feature": feature_path, "summary": summary_path}


def score_split(
    *,
    split: str,
    feature_set: str,
    official_model: Path,
    request_model: Path | None,
    output_dir: Path,
    output_tag: str,
    request_feature_cache_dir: Path,
    force_request_feature_cache: bool,
    k: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    feats = FEATURE_SETS[feature_set]
    data_path = parquet_path(split, feature_set)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    gs = group_sizes_for(data_path, feats, None)
    catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
    directives = transfer_request_directives(split, catalog)
    request_cache = build_or_load_transfer_request_feature_cache(
        path=data_path,
        gs=gs,
        feature_set=feature_set,
        split=split,
        directives=directives,
        cache_dir=request_feature_cache_dir,
        force=force_request_feature_cache,
    )
    request_feature = np.load(request_cache["feature"], mmap_mode="r")

    official_bst = load_booster(official_model)
    request_bst = load_booster(request_model) if request_model else None
    request_feature_names = feats + [EXACT_REQUEST_FEATURE]
    pf = pq.ParquetFile(data_path)

    baseline_rows: list[dict[str, Any]] = []
    fullpool_rows: list[dict[str, Any]] = []
    specialist_rows: list[dict[str, Any]] = []
    rows_by_key: dict[tuple[str, int], dict[str, list[str]]] = {}

    cur_key: tuple[str, int] | None = None
    cur_tracks: list[str] = []
    cur_official_scores: list[float] = []
    cur_request_scores: list[float] = []
    cur_match: list[float] = []
    off = 0

    def flush() -> None:
        nonlocal cur_key, cur_tracks, cur_official_scores, cur_request_scores, cur_match
        if cur_key is None:
            return
        official_scores = np.asarray(cur_official_scores, dtype=np.float32)
        match = np.asarray(cur_match, dtype=np.float32)
        official_order = np.argsort(-official_scores, kind="stable")
        baseline = [cur_tracks[int(i)] for i in official_order[:k]]
        if match.max(initial=0.0) > 0.0:
            request_order = np.asarray([i for i in official_order if match[int(i)] > 0.0], dtype=np.int64)
            rest_order = np.asarray([i for i in official_order if match[int(i)] <= 0.0], dtype=np.int64)
            full_order = np.concatenate([request_order, rest_order])
        else:
            full_order = official_order
        fullpool = [cur_tracks[int(i)] for i in full_order[:k]]

        if request_bst is not None and match.max(initial=0.0) > 0.0:
            request_scores = np.asarray(cur_request_scores, dtype=np.float32)
            specialist_order = np.argsort(-request_scores, kind="stable")
        else:
            specialist_order = official_order
        specialist = [cur_tracks[int(i)] for i in specialist_order[:k]]

        row_base = {
            "session_id": cur_key[0],
            "turn_number": int(cur_key[1]),
            "predicted_response": "",
        }
        baseline_rows.append({**row_base, "predicted_track_ids": baseline})
        fullpool_rows.append({**row_base, "predicted_track_ids": fullpool})
        if request_bst is not None:
            specialist_rows.append({**row_base, "predicted_track_ids": specialist})
        rows_by_key[cur_key] = {
            "baseline": baseline,
            "fullpool": fullpool,
            "specialist": specialist,
        }
        cur_key = None
        cur_tracks = []
        cur_official_scores = []
        cur_request_scores = []
        cur_match = []

    cols = feats + ["session_id", "turn_number", "track_id"]
    for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
        X = np.empty((batch.num_rows, len(feats)), dtype=np.float32)
        for j, feat in enumerate(feats):
            X[:, j] = batch.column(feat).to_numpy(zero_copy_only=False).astype(np.float32)
        official_scores = official_bst.predict(xgb.DMatrix(X, feature_names=feats))
        if request_bst is not None:
            X_request = np.empty((batch.num_rows, len(request_feature_names)), dtype=np.float32)
            X_request[:, :len(feats)] = X
            X_request[:, len(feats)] = request_feature[off:off + batch.num_rows]
            request_scores = request_bst.predict(xgb.DMatrix(X_request, feature_names=request_feature_names))
        else:
            request_scores = official_scores
        matches = request_feature[off:off + batch.num_rows]
        sids = batch.column("session_id").to_pylist()
        turns = batch.column("turn_number").to_numpy(zero_copy_only=False)
        tids = batch.column("track_id").to_pylist()
        for sid, turn, tid, official_score, request_score, match in zip(
            sids, turns, tids, official_scores, request_scores, matches
        ):
            key = (sid, int(turn))
            if cur_key is not None and key != cur_key:
                flush()
            if cur_key is None:
                cur_key = key
            cur_tracks.append(tid)
            cur_official_scores.append(float(official_score))
            cur_request_scores.append(float(request_score))
            cur_match.append(float(match))
        off += batch.num_rows
    flush()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"request_exact_transfer_{feature_set}_{output_tag}_{split}"
    baseline_path = output_dir / f"{stem}_baseline.json"
    fullpool_path = output_dir / f"{stem}_fullpool.json"
    specialist_path = output_dir / f"{stem}_specialist.json" if request_bst is not None else None
    baseline_path.write_text(json.dumps(baseline_rows), encoding="utf-8")
    fullpool_path.write_text(json.dumps(fullpool_rows), encoding="utf-8")
    if specialist_path is not None:
        specialist_path.write_text(json.dumps(specialist_rows), encoding="utf-8")

    predicted_keys = set(rows_by_key)
    directive_keys = set(directives)
    evaluable_keys = sorted(predicted_keys & directive_keys)
    counts: dict[str, Any] = {
        "n_directives_total": len(directive_keys),
        "n_prediction_groups": len(predicted_keys),
        "n_evaluable_directives": len(evaluable_keys),
        "n_directives_missing_prediction": len(directive_keys - predicted_keys),
        "n_predictions_without_directive": len(predicted_keys - directive_keys),
    }
    for name in ["baseline", "fullpool", "specialist"]:
        if name == "specialist" and request_bst is None:
            continue
        first = 0
        missing_topk = 0
        changed_vs_baseline = 0
        same_top1_as_fullpool = 0
        ranks: list[int] = []
        for key in evaluable_keys:
            requested = set(directives[key].get("requested_track_ids") or [])
            pred = rows_by_key[key][name]
            rank = rank_of_any(pred, requested, k)
            if rank == 1:
                first += 1
            if rank is None:
                missing_topk += 1
            else:
                ranks.append(rank)
            if rows_by_key[key][name] != rows_by_key[key]["baseline"]:
                changed_vs_baseline += 1
            if rows_by_key[key][name][:1] == rows_by_key[key]["fullpool"][:1]:
                same_top1_as_fullpool += 1
        counts[name] = {
            "request_first": first,
            "request_first_rate": pct(first, len(evaluable_keys)),
            "request_missing_topk": missing_topk,
            "changed_vs_baseline": changed_vs_baseline,
            "same_top1_as_fullpool": same_top1_as_fullpool,
            "mean_request_rank_when_present": float(np.mean(ranks)) if ranks else None,
            "median_request_rank_when_present": float(np.median(ranks)) if ranks else None,
        }

    summary = {
        "protocol": "request-exact-transfer-v0.9",
        "split": split,
        "feature_set": feature_set,
        "data_path": str(data_path),
        "official_model": str(official_model),
        "request_model": str(request_model) if request_model else None,
        "request_feature_summary": str(request_cache["summary"]),
        "k": k,
        "baseline_prediction": str(baseline_path),
        "fullpool_prediction": str(fullpool_path),
        "specialist_prediction": str(specialist_path) if specialist_path else None,
        "counts": counts,
        "non_oracle_inputs": [
            "visible dialogue",
            "catalog metadata",
            "candidate-pool membership",
            "official-model scores",
            "request-specialist scores when provided",
        ],
        "forbidden_inputs": [
            "official label",
            "correction sidecar",
            "corrected metric",
            "adjudication labels",
            "synthetic thought trace",
        ],
    }
    summary_path = output_dir / f"{stem}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="F10_R54SRC")
    ap.add_argument("--official-model", type=Path, required=True)
    ap.add_argument("--request-model", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("docs/evidence/transfer_predictions"))
    ap.add_argument("--output-tag", default="v09")
    ap.add_argument("--request-feature-cache-dir", type=Path, default=Path("/tmp/request_exact_transfer_feature_cache"))
    ap.add_argument("--force-request-feature-cache", action="store_true")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args(argv)
    print(json.dumps(score_split(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
