"""Evaluate an official-anchored request-trained reranker blend.

The goal is to constrain request-aware training to visible request turns:

  score = official_score + blend_weight * (request_score - official_score)

for turns present in the correction sidecar. Non-correction turns keep the
official score exactly. This is an experiment runner; it does not mutate the
production reranker or the frozen official evaluator.
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

from evaluate_devset import evaluate  # noqa: E402
from evaluate_request_corrections import evaluate_corrected, load_corrections  # noqa: E402
from apply_request_exact_postrank import request_directives  # noqa: E402
from apply_request_hard_constraint_postrank import hard_artist_directives  # noqa: E402
from request_correction_labels import build_catalog_index  # noqa: E402
from request_corrected_rerank_train import (  # noqa: E402
    EXACT_REQUEST_FEATURE,
    build_or_load_exact_request_feature_cache,
    cache_float,
)
from train_request_supplement import (  # noqa: E402
    HARD_ARTIST_FEATURE,
    build_or_load_hard_artist_feature_cache,
)
from rerank_train import BATCH, FEATURE_SETS, group_sizes_for, parquet_path  # noqa: E402


def load_booster(path: Path) -> xgb.Booster:
    if not path.exists():
        raise FileNotFoundError(path)
    bst = xgb.Booster()
    bst.load_model(path)
    return bst


def sidecar_keys(path: Path) -> set[tuple[str, int]]:
    return set(load_corrections(path))


def dialogue_keys(split: str, max_sessions: int | None = None) -> set[tuple[str, int]]:
    from datasets import load_dataset

    catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
    return set(request_directives(split, catalog, max_sessions=max_sessions))


def hard_artist_keys(
    split: str,
    max_sessions: int | None = None,
    *,
    strict: bool = True,
    exclude_exact_requests: bool = False,
    simple_artist_only: bool = False,
) -> set[tuple[str, int]]:
    from datasets import load_dataset

    catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
    return set(hard_artist_directives(
        split,
        catalog,
        max_sessions=max_sessions,
        strict=strict,
        exclude_exact_requests=exclude_exact_requests,
        simple_artist_only=simple_artist_only,
    ))


def gate_keys(
    mode: str,
    corrections: Path,
    split: str,
    *,
    hard_artist_strict: bool = True,
    hard_artist_exclude_exact_requests: bool = False,
    hard_artist_simple_only: bool = False,
) -> set[tuple[str, int]]:
    if mode == "sidecar":
        return sidecar_keys(corrections)
    if mode == "dialogue":
        return dialogue_keys(split)
    if mode == "hard_artist":
        return hard_artist_keys(
            split,
            strict=hard_artist_strict,
            exclude_exact_requests=hard_artist_exclude_exact_requests,
            simple_artist_only=hard_artist_simple_only,
        )
    raise ValueError(f"unknown gate mode {mode!r}")


def score_blend_to_predictions(
    *,
    official_model: Path,
    request_model: Path,
    feature_set: str,
    split: str,
    corrections: Path,
    gate_mode: str,
    blend_weights: list[float],
    prediction_dir: Path,
    output_tag: str,
    k: int,
    request_features: list[tuple[str, np.ndarray]] | None = None,
    hard_artist_strict: bool = True,
    hard_artist_exclude_exact_requests: bool = False,
    hard_artist_simple_only: bool = False,
) -> dict[float, Path]:
    request_features = list(request_features or [])
    feats = FEATURE_SETS[feature_set]
    official_feature_names = feats
    request_feature_names = feats + [name for name, _ in request_features]
    data_path = parquet_path(split, feature_set)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    official_bst = load_booster(official_model)
    request_bst = load_booster(request_model)
    gated = gate_keys(
        gate_mode,
        corrections,
        split,
        hard_artist_strict=hard_artist_strict,
        hard_artist_exclude_exact_requests=hard_artist_exclude_exact_requests,
        hard_artist_simple_only=hard_artist_simple_only,
    )
    pf = pq.ParquetFile(data_path)

    rows_by_weight: dict[float, list[dict[str, Any]]] = {w: [] for w in blend_weights}
    changed_groups_by_weight: dict[float, int] = {w: 0 for w in blend_weights}
    cur_key: tuple[str, int] | None = None
    cur_tracks: list[str] = []
    cur_official_scores: list[float] = []
    cur_request_scores: list[float] = []
    off = 0

    def flush() -> None:
        nonlocal cur_key, cur_tracks, cur_official_scores, cur_request_scores
        if cur_key is None:
            return
        official_scores = np.asarray(cur_official_scores, dtype=np.float32)
        request_scores = np.asarray(cur_request_scores, dtype=np.float32)
        is_gated = cur_key in gated
        for weight in blend_weights:
            if is_gated:
                scores = official_scores + float(weight) * (request_scores - official_scores)
                changed_groups_by_weight[weight] += 1
            else:
                scores = official_scores
            order = np.argsort(-scores, kind="stable")[:k]
            rows_by_weight[weight].append({
                "session_id": cur_key[0],
                "turn_number": int(cur_key[1]),
                "predicted_track_ids": [cur_tracks[int(i)] for i in order],
                "predicted_response": "",
            })
        cur_key = None
        cur_tracks = []
        cur_official_scores = []
        cur_request_scores = []

    cols = feats + ["session_id", "turn_number", "track_id"]
    for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
        X = np.empty((batch.num_rows, len(feats)), dtype=np.float32)
        for j, feat in enumerate(feats):
            X[:, j] = batch.column(feat).to_numpy(zero_copy_only=False).astype(np.float32)
        official_scores = official_bst.predict(xgb.DMatrix(X, feature_names=official_feature_names))
        if not request_features:
            request_scores = request_bst.predict(xgb.DMatrix(X, feature_names=request_feature_names))
        else:
            X_request = np.empty((batch.num_rows, len(request_feature_names)), dtype=np.float32)
            X_request[:, :len(feats)] = X
            for j, (_, values) in enumerate(request_features):
                X_request[:, len(feats) + j] = values[off:off + batch.num_rows]
            request_scores = request_bst.predict(xgb.DMatrix(X_request, feature_names=request_feature_names))
        sids = batch.column("session_id").to_pylist()
        turns = batch.column("turn_number").to_numpy(zero_copy_only=False)
        tids = batch.column("track_id").to_pylist()
        for sid, turn, tid, official_score, request_score in zip(sids, turns, tids, official_scores, request_scores):
            key = (sid, int(turn))
            if cur_key is not None and key != cur_key:
                flush()
            if cur_key is None:
                cur_key = key
            cur_tracks.append(tid)
            cur_official_scores.append(float(official_score))
            cur_request_scores.append(float(request_score))
        off += batch.num_rows
    flush()

    prediction_dir.mkdir(parents=True, exist_ok=True)
    out: dict[float, Path] = {}
    for weight, rows in rows_by_weight.items():
        path = prediction_dir / (
            f"request_corrected_{feature_set}_blend_{output_tag}_"
            f"w{cache_float(weight)}_{split}.json"
        )
        path.write_text(json.dumps(rows), encoding="utf-8")
        out[weight] = path

    meta = {
        "feature_set": feature_set,
        "split": split,
        "official_model": str(official_model),
        "request_model": str(request_model),
        "corrections": str(corrections),
        "gate_mode": gate_mode,
        "gate_keys": len(gated),
        "blend_weights": blend_weights,
        "request_model_extra_features": [name for name, _ in request_features],
        "hard_artist_gate": {
            "strict": bool(hard_artist_strict),
            "exclude_exact_requests": bool(hard_artist_exclude_exact_requests),
            "simple_artist_only": bool(hard_artist_simple_only),
        },
        "changed_groups_by_weight": changed_groups_by_weight,
    }
    (prediction_dir / f"request_corrected_{feature_set}_blend_{output_tag}_{split}.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="F10_R54SRC")
    ap.add_argument("--split", choices=["devset", "val", "train_lt"], default="devset")
    ap.add_argument("--official-model", type=Path, required=True)
    ap.add_argument("--request-model", type=Path, required=True)
    ap.add_argument("--corrections", type=Path, required=True)
    ap.add_argument("--gate-mode", choices=["sidecar", "dialogue", "hard_artist"], default="sidecar",
                    help="sidecar uses correction rows; dialogue uses visible exact-request directives; hard_artist uses visible hard-artist directives.")
    ap.add_argument("--blend-weight", type=float, action="append", required=True)
    ap.add_argument("--prediction-dir", type=Path, default=Path("work/dev_predictions"))
    ap.add_argument("--output-tag", default="v0")
    ap.add_argument("--output", type=Path, default=Path("work/rerank_blend_eval.json"))
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--scoring", choices=["best_hit", "multilabel"], default="best_hit")
    ap.add_argument("--request-model-add-exact-request-feature", action="store_true",
                    help="Score the request model with the non-oracle exact-request-match feature.")
    ap.add_argument("--request-model-add-hard-artist-feature", action="store_true",
                    help="Score the request model with the non-oracle hard-artist-constraint-match feature.")
    ap.add_argument("--hard-artist-no-strict", action="store_true",
                    help="Use broad hard-artist directives for the hard-artist feature/gate.")
    ap.add_argument("--hard-artist-exclude-exact-requests", action="store_true",
                    help="Skip hard-artist directives that contain a resolved exact-title request.")
    ap.add_argument("--hard-artist-simple-only", action="store_true",
                    help="Use the abstaining simple-artist-only hard-feature/gate.")
    ap.add_argument("--request-feature-cache-dir", type=Path,
                    help="Directory for row-aligned exact-request feature caches.")
    ap.add_argument("--force-request-feature-cache", action="store_true")
    args = ap.parse_args(argv)

    weights = sorted(set(float(w) for w in args.blend_weight))
    request_features: list[tuple[str, np.ndarray]] = []
    request_feature_summaries: list[str] = []
    if args.request_model_add_exact_request_feature:
        from datasets import load_dataset

        feats = FEATURE_SETS[args.feature_set]
        data_path = parquet_path(args.split, args.feature_set)
        gs = group_sizes_for(data_path, feats, None)
        catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
        request_cache = build_or_load_exact_request_feature_cache(
            path=data_path,
            gs=gs,
            feature_set=args.feature_set,
            split=args.split,
            catalog=catalog,
            cache_dir=args.request_feature_cache_dir or Path("/tmp/request_corrected_request_feature_cache"),
            force=args.force_request_feature_cache,
        )
        request_features.append((EXACT_REQUEST_FEATURE, np.load(request_cache["feature"], mmap_mode="r")))
        request_feature_summaries.append(str(request_cache["summary"]))
    if args.request_model_add_hard_artist_feature:
        from datasets import load_dataset

        feats = FEATURE_SETS[args.feature_set]
        data_path = parquet_path(args.split, args.feature_set)
        gs = group_sizes_for(data_path, feats, None)
        catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
        hard_cache = build_or_load_hard_artist_feature_cache(
            path=data_path,
            gs=gs,
            feature_set=args.feature_set,
            split=args.split,
            catalog=catalog,
            cache_dir=args.request_feature_cache_dir or Path("/tmp/request_corrected_request_feature_cache"),
            force=args.force_request_feature_cache,
            strict=not bool(args.hard_artist_no_strict),
            exclude_exact_requests=args.hard_artist_exclude_exact_requests,
            simple_artist_only=args.hard_artist_simple_only,
        )
        request_features.append((HARD_ARTIST_FEATURE, np.load(hard_cache["feature"], mmap_mode="r")))
        request_feature_summaries.append(str(hard_cache["summary"]))
    pred_paths = score_blend_to_predictions(
        official_model=args.official_model,
        request_model=args.request_model,
        feature_set=args.feature_set,
        split=args.split,
        corrections=args.corrections,
        gate_mode=args.gate_mode,
        blend_weights=weights,
        prediction_dir=args.prediction_dir,
        output_tag=args.output_tag,
        k=args.k,
        request_features=request_features,
        hard_artist_strict=not bool(args.hard_artist_no_strict),
        hard_artist_exclude_exact_requests=args.hard_artist_exclude_exact_requests,
        hard_artist_simple_only=args.hard_artist_simple_only,
    )

    summary: dict[str, Any] = {
        "feature_set": args.feature_set,
        "split": args.split,
        "official_model": str(args.official_model),
        "request_model": str(args.request_model),
        "corrections": str(args.corrections),
        "gate_mode": args.gate_mode,
        "scoring": args.scoring,
        "request_model_add_exact_request_feature": bool(args.request_model_add_exact_request_feature),
        "request_model_add_hard_artist_feature": bool(args.request_model_add_hard_artist_feature),
        "request_feature_summaries": request_feature_summaries,
        "hard_artist_gate": {
            "strict": not bool(args.hard_artist_no_strict),
            "exclude_exact_requests": bool(args.hard_artist_exclude_exact_requests),
            "simple_artist_only": bool(args.hard_artist_simple_only),
        },
        "blend_weights": {},
    }
    for weight, pred_path in pred_paths.items():
        corrected = evaluate_corrected(
            prediction_path=pred_path,
            corrections_path=args.corrections,
            split=args.split,
            scoring=args.scoring,
        )
        official_path = None
        official_ndcg20 = corrected["official_ndcg20"]
        official_ndcg10 = None
        official_ndcg1 = None
        if args.split == "devset":
            official_path = args.output.parent / (
                f"request_corrected_{args.feature_set}_blend_{args.output_tag}_"
                f"w{cache_float(weight)}_official_dev.json"
            )
            official = evaluate(
                f"request_corrected_{args.feature_set}_blend_{args.output_tag}_w{cache_float(weight)}",
                str(pred_path),
                str(official_path),
            )
            official_ndcg20 = official["macro"]["ndcg@20"]
            official_ndcg10 = official["macro"]["ndcg@10"]
            official_ndcg1 = official["macro"]["ndcg@1"]
        summary["blend_weights"][cache_float(weight)] = {
            "blend_weight": float(weight),
            "prediction_path": str(pred_path),
            "official_output": str(official_path) if official_path else None,
            "official_ndcg20": official_ndcg20,
            "official_ndcg10": official_ndcg10,
            "official_ndcg1": official_ndcg1,
            "corrected_ndcg20": corrected["corrected_ndcg20"],
            "corrected_delta_minus_official": corrected["delta_corrected_minus_official"],
            "positive_slice": corrected["positive_slice"],
            "positive_slice_by_family": corrected["positive_slice_by_family"],
            "correction_records_by_family": corrected["correction_records_by_family"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"[request-corrected-blend] weight={weight:g}: "
            f"official={official_ndcg20:.5f} corrected={corrected['corrected_ndcg20']:.5f}",
            flush=True,
        )

    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
