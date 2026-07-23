"""Train rerankers with request-aware correction sidecars.

This is an experiment runner, not the production reranker path. It reuses the
existing feature parquet and XGBoost settings, but changes labels/weights at
stream time:

  * exact/version sidecar positives become extra label=1 rows,
  * hard positive constraints label matching in-pool candidates as positives,
  * masked violations zero the policy-selected gold label, and
  * high-confidence rejection-only masks downweight the query group.

Default cells:

  official                 original labels
  exact_positive           add exact/version positives only
  exact_positive_graded    official positives have relevance 2; exact/version
                           positives have relevance 1
  exact_positive_graded_weighted
                           same labels as exact_positive_graded; downweight
                           corrected training groups
  exact_positive_request_preferred
                           official positives have relevance 1; exact/version
                           positives have relevance 2
  exact_positive_weighted  same labels as exact_positive; downweight corrected
                           training groups
  request_positive         add exact/version + hard-constraint positives
  request_positive_weighted
                           same labels as request_positive; downweight corrected
                           training groups
  request_positive_masked  add positives + mask/downweight violations
  violation_drop           keep official labels but drop clear violation groups
                           from the training loss

Backward-compatible aliases:

  positive        -> request_positive
  positive_masked -> request_positive_masked
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

ROOT = Path(os.environ.get("MCRS_EXPLORE_ROOT", str(Path(__file__).resolve().parent.parent)))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from day5_train_ltr import ndcg20  # noqa: E402
from apply_request_exact_postrank import request_directives  # noqa: E402
from apply_request_hard_constraint_postrank import hard_artist_directives  # noqa: E402
from evaluate_request_corrections import load_corrections  # noqa: E402
from request_correction_labels import build_catalog_index, track_satisfies_constraint  # noqa: E402
from rerank_train import (  # noqa: E402
    BATCH,
    FEAT,
    FEATURE_SETS,
    MODEL_DIR,
    PARAM_PRESETS,
    PARAMS,
    SCORE,
    gids_from_sizes,
    group_sizes_for,
    parse_xgb_params,
    parquet_path,
)


def cleanup_xgb_external_cache(cache_prefix: str | None) -> int:
    """Remove external-memory pages after a cell has finished training."""
    if not cache_prefix:
        return 0
    prefix = Path(cache_prefix)
    removed = 0
    for path in prefix.parent.glob(prefix.name + "*"):
        if not path.exists() or path.is_dir():
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed

EXACT_REQUEST_FEATURE = "exact_request_match"
HARD_ARTIST_FEATURE = "hard_artist_constraint_match"


@dataclass
class TrainSource:
    path: Path
    split: str
    gs: np.ndarray
    corrections_path: Path | None
    corrections: dict[tuple[str, int], dict[str, Any]]
    extra_features: list[tuple[str, np.ndarray]] = field(default_factory=list)
    gids: np.ndarray | None = None


def merge_corrections(rows: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[tuple[str, int], dict[str, Any]]:
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for key, recs in rows.items():
        positives: set[str] = set()
        exact_positives: set[str] = set()
        constraints: list[dict[str, Any]] = []
        mask_gold = False
        weight = 1.0
        families = set()
        for rec in recs:
            family = rec["family"]
            families.add(family)
            added = set(rec.get("additional_track_ids") or [])
            positives.update(added)
            if family in {"exact_track_request", "version_duplicate_equivalence"}:
                exact_positives.update(added)
            if rec.get("positive_constraint"):
                constraints.append(rec["positive_constraint"])
            mask_gold = mask_gold or bool(rec.get("mask_gold"))
            weight = min(weight, float(rec.get("group_weight", 1.0)))
        merged[key] = {
            "additional_track_ids": positives,
            "exact_track_ids": exact_positives,
            "positive_constraints": constraints,
            "mask_gold": mask_gold,
            "group_weight": weight,
            "families": sorted(families),
        }
    return merged


def load_merged_corrections(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    return merge_corrections(load_corrections(path)) if path else {}


def optional_list(values: list[Any] | None, n: int, *, default: Any = None, name: str) -> list[Any]:
    if not values:
        return [default for _ in range(n)]
    if len(values) != n:
        raise ValueError(f"{name}: expected {n} values, got {len(values)}")
    return values


class CorrectedPoolIter(xgb.DataIter):
    def __init__(
        self,
        path: Path,
        feats: list[str],
        gids: np.ndarray,
        corrections: dict[tuple[str, int], dict[str, Any]],
        catalog,
        mode: str,
    ):
        self.path = path
        self.feats = feats
        self.gids = gids
        self.corrections = corrections
        self.catalog = catalog
        self.mode = mode
        self._mk()
        super().__init__()

    def _mk(self):
        self.pf = pq.ParquetFile(self.path)
        self.itr = self.pf.iter_batches(
            batch_size=BATCH,
            columns=self.feats + ["label", "session_id", "turn_number", "track_id"],
        )
        self.off = 0

    def reset(self):
        self._mk()

    def corrected_labels(self, b) -> np.ndarray:
        y = b.column("label").to_numpy(zero_copy_only=False).astype(np.float32)
        if self.mode == "official":
            return y
        if self.mode == "violation_drop":
            return y
        if self.mode in {"exact_positive_graded", "exact_positive_graded_weighted"}:
            y[y > 0.0] = 2.0
        sids = b.column("session_id").to_pylist()
        turns = b.column("turn_number").to_numpy(zero_copy_only=False)
        tids = b.column("track_id").to_pylist()
        for i, tid in enumerate(tids):
            corr = self.corrections.get((sids[i], int(turns[i])))
            if not corr:
                continue
            if self.mode in {"exact_positive_graded", "exact_positive_graded_weighted"}:
                if tid in corr["exact_track_ids"]:
                    y[i] = max(float(y[i]), 1.0)
                continue
            if self.mode == "exact_positive_request_preferred":
                if tid in corr["exact_track_ids"]:
                    y[i] = max(float(y[i]), 2.0)
                continue
            if self.mode in {"positive", "positive_masked"}:
                if tid in corr["additional_track_ids"]:
                    y[i] = 1.0
                elif any(track_satisfies_constraint(tid, c, self.catalog) for c in corr["positive_constraints"]):
                    y[i] = 1.0
            if self.mode == "positive_masked" and corr["mask_gold"] and y[i] == 1.0 and tid not in corr["additional_track_ids"]:
                if not any(track_satisfies_constraint(tid, c, self.catalog) for c in corr["positive_constraints"]):
                    y[i] = 0.0
        return y

    def next(self, input_data):
        try:
            b = next(self.itr)
        except StopIteration:
            return 0
        X = np.empty((b.num_rows, len(self.feats)), dtype=np.float32)
        for j, f in enumerate(self.feats):
            X[:, j] = b.column(f).to_numpy(zero_copy_only=False).astype(np.float32)
        y = self.corrected_labels(b)
        gid = self.gids[self.off:self.off + b.num_rows]
        input_data(data=X, label=y, qid=gid.astype(np.uint32), feature_names=self.feats)
        self.off += b.num_rows
        return 1


def load_yturn_corrected(
    path: Path,
    gs: np.ndarray,
    corrections: dict[tuple[str, int], dict[str, Any]],
    catalog,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    pf = pq.ParquetFile(path)
    ys, ts = [], []
    for b in pf.iter_batches(batch_size=BATCH, columns=["label", "turn_number", "session_id", "track_id"]):
        y = b.column("label").to_numpy(zero_copy_only=False).astype(np.float32)
        t = b.column("turn_number").to_numpy(zero_copy_only=False).astype(np.int32)
        if mode != "official":
            if mode in {"exact_positive_graded", "exact_positive_graded_weighted"}:
                y[y > 0.0] = 2.0
            sids = b.column("session_id").to_pylist()
            tids = b.column("track_id").to_pylist()
            for i, tid in enumerate(tids):
                corr = corrections.get((sids[i], int(t[i])))
                if not corr:
                    continue
                if mode in {"exact_positive_graded", "exact_positive_graded_weighted"}:
                    if tid in corr["exact_track_ids"]:
                        y[i] = max(float(y[i]), 1.0)
                    continue
                if mode == "exact_positive_request_preferred":
                    if tid in corr["exact_track_ids"]:
                        y[i] = max(float(y[i]), 2.0)
                    continue
                if tid in corr["additional_track_ids"]:
                    y[i] = 1.0
                elif any(track_satisfies_constraint(tid, c, catalog) for c in corr["positive_constraints"]):
                    y[i] = 1.0
                if mode == "positive_masked" and corr["mask_gold"] and y[i] == 1.0 and tid not in corr["additional_track_ids"]:
                    if not any(track_satisfies_constraint(tid, c, catalog) for c in corr["positive_constraints"]):
                        y[i] = 0.0
        ys.append(y.astype(np.int8))
        ts.append(t)
    yout = np.concatenate(ys)
    tout = np.concatenate(ts)
    assert len(yout) == int(gs.sum())
    return yout, tout


def group_weights_from_dmatrix(
    dmat: xgb.DMatrix,
    gids: np.ndarray,
    group_weights: np.ndarray,
) -> np.ndarray:
    gptr = dmat.get_uint_info("group_ptr").astype(np.int64)
    parent = gids[gptr[:-1]]
    return group_weights[parent]


def per_group_weights(path: Path, gs: np.ndarray, corrections: dict[tuple[str, int], dict[str, Any]]) -> np.ndarray:
    starts = np.zeros(len(gs), dtype=np.int64)
    starts[1:] = np.cumsum(gs)[:-1]
    weights = np.ones(len(gs), dtype=np.float32)
    pf = pq.ParquetFile(path)
    off = 0
    for b in pf.iter_batches(batch_size=BATCH, columns=["session_id", "turn_number"]):
        lo = np.searchsorted(starts, off, side="left")
        hi = np.searchsorted(starts, off + b.num_rows, side="left")
        if hi > lo:
            loc = starts[lo:hi] - off
            sid_col = b.column("session_id")
            tn_col = b.column("turn_number").to_numpy(zero_copy_only=False)
            for k, i in enumerate(loc):
                key = (sid_col[int(i)].as_py(), int(tn_col[int(i)]))
                corr = corrections.get(key)
                if corr:
                    weights[lo + k] = float(corr["group_weight"])
        off += b.num_rows
    return weights


def per_group_request_weights(
    path: Path,
    gs: np.ndarray,
    corrections: dict[tuple[str, int], dict[str, Any]],
    corrected_group_weight: float,
) -> np.ndarray:
    starts = np.zeros(len(gs), dtype=np.int64)
    starts[1:] = np.cumsum(gs)[:-1]
    weights = np.ones(len(gs), dtype=np.float32)
    pf = pq.ParquetFile(path)
    off = 0
    for b in pf.iter_batches(batch_size=BATCH, columns=["session_id", "turn_number"]):
        lo = np.searchsorted(starts, off, side="left")
        hi = np.searchsorted(starts, off + b.num_rows, side="left")
        if hi > lo:
            loc = starts[lo:hi] - off
            sid_col = b.column("session_id")
            tn_col = b.column("turn_number").to_numpy(zero_copy_only=False)
            for k, i in enumerate(loc):
                key = (sid_col[int(i)].as_py(), int(tn_col[int(i)]))
                corr = corrections.get(key)
                if corr and (corr["additional_track_ids"] or corr["positive_constraints"]):
                    weights[lo + k] = float(corrected_group_weight)
        off += b.num_rows
    return weights


def per_group_exact_weights(
    path: Path,
    gs: np.ndarray,
    corrections: dict[tuple[str, int], dict[str, Any]],
    corrected_group_weight: float,
) -> np.ndarray:
    starts = np.zeros(len(gs), dtype=np.int64)
    starts[1:] = np.cumsum(gs)[:-1]
    weights = np.ones(len(gs), dtype=np.float32)
    pf = pq.ParquetFile(path)
    off = 0
    for b in pf.iter_batches(batch_size=BATCH, columns=["session_id", "turn_number"]):
        lo = np.searchsorted(starts, off, side="left")
        hi = np.searchsorted(starts, off + b.num_rows, side="left")
        if hi > lo:
            loc = starts[lo:hi] - off
            sid_col = b.column("session_id")
            tn_col = b.column("turn_number").to_numpy(zero_copy_only=False)
            for k, i in enumerate(loc):
                key = (sid_col[int(i)].as_py(), int(tn_col[int(i)]))
                corr = corrections.get(key)
                if corr and corr["exact_track_ids"]:
                    weights[lo + k] = float(corrected_group_weight)
        off += b.num_rows
    return weights


def per_group_violation_drop_weights(
    path: Path,
    gs: np.ndarray,
    corrections: dict[tuple[str, int], dict[str, Any]],
) -> np.ndarray:
    starts = np.zeros(len(gs), dtype=np.int64)
    starts[1:] = np.cumsum(gs)[:-1]
    weights = np.ones(len(gs), dtype=np.float32)
    pf = pq.ParquetFile(path)
    off = 0
    for b in pf.iter_batches(batch_size=BATCH, columns=["session_id", "turn_number"]):
        lo = np.searchsorted(starts, off, side="left")
        hi = np.searchsorted(starts, off + b.num_rows, side="left")
        if hi > lo:
            loc = starts[lo:hi] - off
            sid_col = b.column("session_id")
            tn_col = b.column("turn_number").to_numpy(zero_copy_only=False)
            for k, i in enumerate(loc):
                key = (sid_col[int(i)].as_py(), int(tn_col[int(i)]))
                corr = corrections.get(key)
                if corr and corr["mask_gold"]:
                    weights[lo + k] = 0.0
        off += b.num_rows
    return weights


def booster_best_iteration(bst: xgb.Booster, fallback: int) -> int:
    try:
        return int(bst.best_iteration)
    except AttributeError:
        return fallback


MODE_ALIASES = {
    "positive": "request_positive",
    "positive_masked": "request_positive_masked",
}


def canonical_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def cache_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def label_cache_stem(
    path: Path,
    feature_set: str,
    split: str,
    mode: str,
    corrections_path: Path | None,
    corrected_group_weight: float,
) -> str:
    corr = "none" if corrections_path is None else corrections_path.stem
    stat = path.stat()
    weighted_modes = {"exact_positive_weighted", "exact_positive_graded_weighted", "request_positive_weighted"}
    weight = f"_gw{cache_float(corrected_group_weight)}" if mode in weighted_modes else ""
    return f"{feature_set}_{split}_{mode}{weight}_{corr}_{stat.st_mtime_ns:x}_{stat.st_size:x}"


def request_feature_split(split: str) -> str:
    if split.startswith("train_lt"):
        return "train_lt"
    return split


def request_feature_cache_stem(path: Path, feature_set: str, split: str, feature_name: str, variant: str | None = None) -> str:
    stat = path.stat()
    feature_token = feature_name if variant is None else f"{feature_name}_{variant}"
    return f"{feature_set}_{split}_{feature_token}_{stat.st_mtime_ns:x}_{stat.st_size:x}"


def load_group_sizes(path: Path, feats: list[str], group_sizes_path: Path | None) -> np.ndarray:
    if group_sizes_path is None:
        return group_sizes_for(path, feats, None)
    gs = np.load(group_sizes_path).astype(np.int64, copy=False)
    n_rows = pq.ParquetFile(path).metadata.num_rows
    if int(gs.sum()) != int(n_rows):
        raise ValueError(f"{group_sizes_path}: group sum {int(gs.sum())} != parquet rows {n_rows}")
    return gs


def build_or_load_exact_request_feature_cache(
    *,
    path: Path,
    gs: np.ndarray,
    feature_set: str,
    split: str,
    catalog,
    cache_dir: Path,
    force: bool,
) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = request_feature_cache_stem(path, feature_set, split, EXACT_REQUEST_FEATURE)
    feature_path = cache_dir / f"{stem}.npy"
    summary_path = cache_dir / f"{stem}.summary.json"
    if not force and feature_path.exists() and summary_path.exists():
        return {"feature": feature_path, "summary": summary_path}

    directives = request_directives(request_feature_split(split), catalog)
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
    for b in pf.iter_batches(batch_size=BATCH, columns=["session_id", "turn_number", "track_id"]):
        sids = b.column("session_id").to_pylist()
        turns = b.column("turn_number").to_numpy(zero_copy_only=False)
        tids = b.column("track_id").to_pylist()
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
        off += b.num_rows

    assert off == n_rows
    np.save(feature_path, feature)
    summary = {
        "path": str(path),
        "feature_set": feature_set,
        "split": split,
        "directive_split": request_feature_split(split),
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


def hard_artist_variant(*, strict: bool, exclude_exact_requests: bool, simple_artist_only: bool) -> str:
    return "_".join([
        "strict" if strict else "broad",
        "noexact" if exclude_exact_requests else "withexact",
        "simple" if simple_artist_only else "all",
    ])


def build_or_load_hard_artist_feature_cache(
    *,
    path: Path,
    gs: np.ndarray,
    feature_set: str,
    split: str,
    catalog,
    cache_dir: Path,
    force: bool,
    strict: bool = True,
    exclude_exact_requests: bool = False,
    simple_artist_only: bool = False,
) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    variant = hard_artist_variant(
        strict=strict,
        exclude_exact_requests=exclude_exact_requests,
        simple_artist_only=simple_artist_only,
    )
    stem = request_feature_cache_stem(path, feature_set, split, HARD_ARTIST_FEATURE, variant)
    feature_path = cache_dir / f"{stem}.npy"
    summary_path = cache_dir / f"{stem}.summary.json"
    if not force and feature_path.exists() and summary_path.exists():
        return {"feature": feature_path, "summary": summary_path}

    directives = hard_artist_directives(
        request_feature_split(split),
        catalog,
        strict=strict,
        exclude_exact_requests=exclude_exact_requests,
        simple_artist_only=simple_artist_only,
    )
    constraints_by_key = {
        key: list(value.get("constraints") or [])
        for key, value in directives.items()
    }
    n_rows = int(gs.sum())
    feature = np.zeros(n_rows, dtype=np.float32)
    off = 0
    matched_rows = 0
    matched_groups: set[tuple[str, int]] = set()
    directive_groups_in_pool: set[tuple[str, int]] = set()
    satisfy_cache: dict[tuple[str, str], bool] = {}

    def constraint_key(c: dict[str, Any]) -> str:
        return json.dumps(c, sort_keys=True)

    def satisfies(tid: str, c: dict[str, Any]) -> bool:
        key = (tid, constraint_key(c))
        if key not in satisfy_cache:
            satisfy_cache[key] = track_satisfies_constraint(tid, c, catalog)
        return satisfy_cache[key]

    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=BATCH, columns=["session_id", "turn_number", "track_id"]):
        sids = b.column("session_id").to_pylist()
        turns = b.column("turn_number").to_numpy(zero_copy_only=False)
        tids = b.column("track_id").to_pylist()
        for i, tid in enumerate(tids):
            key = (sids[i], int(turns[i]))
            constraints = constraints_by_key.get(key)
            if not constraints:
                continue
            directive_groups_in_pool.add(key)
            if any(satisfies(tid, c) for c in constraints):
                feature[off + i] = 1.0
                matched_rows += 1
                matched_groups.add(key)
        off += b.num_rows

    assert off == n_rows
    np.save(feature_path, feature)
    summary = {
        "path": str(path),
        "feature_set": feature_set,
        "split": split,
        "directive_split": request_feature_split(split),
        "feature_name": HARD_ARTIST_FEATURE,
        "variant": variant,
        "strict": bool(strict),
        "exclude_exact_requests": bool(exclude_exact_requests),
        "simple_artist_only": bool(simple_artist_only),
        "n_rows": n_rows,
        "n_groups": int(len(gs)),
        "directive_groups": int(len(directives)),
        "directive_groups_in_pool": int(len(directive_groups_in_pool)),
        "matched_rows": int(matched_rows),
        "matched_groups": int(len(matched_groups)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"feature": feature_path, "summary": summary_path}


def build_or_load_label_cache(
    *,
    path: Path,
    gs: np.ndarray,
    feature_set: str,
    split: str,
    mode: str,
    corrections_path: Path | None,
    corrections: dict[tuple[str, int], dict[str, Any]],
    catalog,
    cache_dir: Path,
    force: bool,
    corrected_group_weight: float = 0.5,
) -> dict[str, Path]:
    mode = canonical_mode(mode)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = label_cache_stem(path, feature_set, split, mode, corrections_path, corrected_group_weight)
    label_path = cache_dir / f"{stem}.labels.npy"
    turn_path = cache_dir / f"{stem}.turn.npy"
    weight_path = cache_dir / f"{stem}.group_weight.npy"
    summary_path = cache_dir / f"{stem}.summary.json"
    if not force and label_path.exists() and turn_path.exists() and weight_path.exists() and summary_path.exists():
        return {"labels": label_path, "turn": turn_path, "group_weight": weight_path, "summary": summary_path}

    n_rows = int(gs.sum())
    labels = np.empty(n_rows, dtype=np.int8)
    turns = np.empty(n_rows, dtype=np.int32)
    added = 0
    masked = 0
    off = 0
    satisfy_cache: dict[tuple[str, str], bool] = {}

    def constraint_key(c: dict[str, Any]) -> str:
        return json.dumps(c, sort_keys=True)

    def satisfies(tid: str, c: dict[str, Any]) -> bool:
        key = (tid, constraint_key(c))
        if key not in satisfy_cache:
            satisfy_cache[key] = track_satisfies_constraint(tid, c, catalog)
        return satisfy_cache[key]

    pf = pq.ParquetFile(path)
    columns = ["label", "turn_number"] if mode == "official" else ["label", "turn_number", "session_id", "track_id"]
    for b in pf.iter_batches(batch_size=BATCH, columns=columns):
        m = b.num_rows
        base = b.column("label").to_numpy(zero_copy_only=False).astype(np.int8)
        y = base.copy()
        if mode in {"exact_positive_graded", "exact_positive_graded_weighted"}:
            y[base > 0] = 2
        t = b.column("turn_number").to_numpy(zero_copy_only=False).astype(np.int32)
        if mode != "official":
            sids = b.column("session_id").to_pylist()
            tids = b.column("track_id").to_pylist()
            for i, tid in enumerate(tids):
                corr = corrections.get((sids[i], int(t[i])))
                if not corr:
                    continue
                before = int(y[i])
                if mode == "request_positive_weighted":
                    label_mode = "request_positive"
                elif mode == "exact_positive_weighted":
                    label_mode = "exact_positive"
                elif mode == "exact_positive_graded_weighted":
                    label_mode = "exact_positive_graded"
                else:
                    label_mode = mode
                if label_mode == "exact_positive":
                    if tid in corr["exact_track_ids"]:
                        y[i] = 1
                elif label_mode == "exact_positive_graded":
                    if tid in corr["exact_track_ids"]:
                        y[i] = max(int(y[i]), 1)
                elif label_mode == "exact_positive_request_preferred":
                    if tid in corr["exact_track_ids"]:
                        y[i] = max(int(y[i]), 2)
                elif label_mode == "violation_drop":
                    pass
                else:
                    positive = tid in corr["additional_track_ids"] or any(
                        satisfies(tid, c) for c in corr["positive_constraints"]
                    )
                    if positive:
                        y[i] = 1
                    if label_mode == "request_positive_masked" and corr["mask_gold"] and base[i] == 1 and not positive:
                        y[i] = 0
                if before == 0 and y[i] > 0:
                    added += 1
                elif before > 0 and y[i] == 0:
                    masked += 1
        labels[off:off + m] = y
        turns[off:off + m] = t
        off += m

    assert off == n_rows
    if mode == "request_positive_masked":
        group_weight = per_group_weights(path, gs, corrections)
    elif mode == "violation_drop":
        group_weight = per_group_violation_drop_weights(path, gs, corrections)
    elif mode in {"exact_positive_weighted", "exact_positive_graded_weighted"}:
        group_weight = per_group_exact_weights(path, gs, corrections, corrected_group_weight)
    elif mode == "request_positive_weighted":
        group_weight = per_group_request_weights(path, gs, corrections, corrected_group_weight)
    else:
        group_weight = np.ones(len(gs), dtype=np.float32)
    np.save(label_path, labels)
    np.save(turn_path, turns)
    np.save(weight_path, group_weight)
    summary = {
        "path": str(path),
        "feature_set": feature_set,
        "split": split,
        "mode": mode,
        "corrections_path": str(corrections_path) if corrections_path else None,
        "n_rows": n_rows,
        "n_groups": int(len(gs)),
        "base_positive_rows": int((labels > 0).sum() - added + masked),
        "positive_rows": int((labels > 0).sum()),
        "added_positive_rows": int(added),
        "masked_positive_rows": int(masked),
        "max_label": int(labels.max(initial=0)),
        "downweighted_groups": int((group_weight < 1.0).sum()),
        "dropped_groups": int((group_weight == 0.0).sum()),
        "corrected_group_weight": float(corrected_group_weight) if mode in {"exact_positive_weighted", "exact_positive_graded_weighted", "request_positive_weighted"} else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"labels": label_path, "turn": turn_path, "group_weight": weight_path, "summary": summary_path}


class CachedLabelPoolIter(xgb.DataIter):
    def __init__(
        self,
        path: Path,
        feats: list[str],
        gids: np.ndarray,
        labels: np.ndarray,
        request_feature: np.ndarray | None = None,
        extra_features: list[tuple[str, np.ndarray]] | None = None,
        cache_prefix: str | None = None,
    ):
        self.path = path
        self.feats = feats
        self.extra_features = list(extra_features or [])
        if request_feature is not None:
            self.extra_features.insert(0, (EXACT_REQUEST_FEATURE, request_feature))
        self.feature_names = feats + [name for name, _ in self.extra_features]
        self.gids = gids
        self.labels = labels
        self._mk()
        super().__init__(cache_prefix=cache_prefix)

    def _mk(self):
        self.pf = pq.ParquetFile(self.path)
        self.itr = self.pf.iter_batches(batch_size=BATCH, columns=self.feats)
        self.off = 0

    def reset(self):
        self._mk()

    def next(self, input_data):
        try:
            b = next(self.itr)
        except StopIteration:
            return 0
        X = np.empty((b.num_rows, len(self.feature_names)), dtype=np.float32)
        for j, f in enumerate(self.feats):
            X[:, j] = b.column(f).to_numpy(zero_copy_only=False).astype(np.float32)
        for j, (_, values) in enumerate(self.extra_features):
            X[:, len(self.feats) + j] = values[self.off:self.off + b.num_rows]
        y = self.labels[self.off:self.off + b.num_rows].astype(np.float32)
        gid = self.gids[self.off:self.off + b.num_rows]
        input_data(data=X, label=y, qid=gid.astype(np.uint32), feature_names=self.feature_names)
        self.off += b.num_rows
        return 1


class CachedLabelMultiPoolIter(xgb.DataIter):
    def __init__(
        self,
        sources: list[dict[str, Any]],
        feats: list[str],
        cache_prefix: str | None = None,
    ):
        if not sources:
            raise ValueError("at least one training source is required")
        self.sources = sources
        self.feats = feats
        self.extra_names = [name for name, _ in sources[0].get("extra_features", [])]
        for src in sources:
            names = [name for name, _ in src.get("extra_features", [])]
            if names != self.extra_names:
                raise ValueError(f"extra feature mismatch for {src['path']}: {names} != {self.extra_names}")
        self.feature_names = feats + self.extra_names
        self._mk()
        super().__init__(cache_prefix=cache_prefix)

    def _mk(self):
        self.source_i = 0
        self.off = 0
        self.pf = None
        self.itr = None
        self._open_source()

    def _open_source(self):
        if self.source_i >= len(self.sources):
            self.pf = None
            self.itr = None
            return
        path = self.sources[self.source_i]["path"]
        self.pf = pq.ParquetFile(path)
        self.itr = self.pf.iter_batches(batch_size=BATCH, columns=self.feats)
        self.off = 0

    def reset(self):
        self._mk()

    def next(self, input_data):
        while self.source_i < len(self.sources):
            assert self.itr is not None
            try:
                b = next(self.itr)
                break
            except StopIteration:
                self.source_i += 1
                self._open_source()
        else:
            return 0

        src = self.sources[self.source_i]
        X = np.empty((b.num_rows, len(self.feature_names)), dtype=np.float32)
        for j, f in enumerate(self.feats):
            X[:, j] = b.column(f).to_numpy(zero_copy_only=False).astype(np.float32)
        for j, (_, values) in enumerate(src.get("extra_features", [])):
            X[:, len(self.feats) + j] = values[self.off:self.off + b.num_rows]
        y = src["labels"][self.off:self.off + b.num_rows].astype(np.float32)
        gid = src["gids"][self.off:self.off + b.num_rows]
        input_data(data=X, label=y, qid=gid.astype(np.uint32), feature_names=self.feature_names)
        self.off += b.num_rows
        return 1


def train_cells(args) -> dict[str, Any]:
    from datasets import load_dataset

    catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
    val_corr = load_merged_corrections(args.val_corrections)

    feats = FEATURE_SETS[args.feature_set]
    tr_path = args.train_feature_path or parquet_path("train_lt", args.feature_set)
    train_split_name = args.train_split_name
    extra_paths = args.extra_train_feature_path or []
    extra_group_paths = optional_list(
        args.extra_train_group_sizes_path,
        len(extra_paths),
        default=None,
        name="--extra-train-group-sizes-path",
    )
    extra_split_names = args.extra_train_split_name or [f"extra_train_{i}" for i in range(len(extra_paths))]
    if len(extra_split_names) != len(extra_paths):
        raise ValueError(f"--extra-train-split-name: expected {len(extra_paths)} values, got {len(extra_split_names)}")
    extra_correction_paths = optional_list(
        args.extra_train_corrections,
        len(extra_paths),
        default=None,
        name="--extra-train-corrections",
    )
    train_specs = [
        (tr_path, args.train_group_sizes_path, train_split_name, args.train_corrections),
        *zip(extra_paths, extra_group_paths, extra_split_names, extra_correction_paths),
    ]
    train_sources: list[TrainSource] = []
    for path, group_path, split_name, corr_path in train_specs:
        gs = load_group_sizes(path, feats, group_path)
        train_sources.append(TrainSource(
            path=path,
            split=split_name,
            gs=gs,
            corrections_path=corr_path,
            corrections=load_merged_corrections(corr_path),
        ))
    gtr = np.concatenate([src.gs for src in train_sources])
    tr_gids = None
    if not args.prepare_label_cache_only:
        group_offset = 0
        gids_parts = []
        for src in train_sources:
            src.gids = gids_from_sizes(src.gs).astype(np.uint32, copy=False) + np.uint32(group_offset)
            gids_parts.append(src.gids)
            group_offset += len(src.gs)
        tr_gids = np.concatenate(gids_parts)
    va_path = parquet_path("val", args.feature_set)
    gva = None
    va_gids = None
    if not args.skip_val_dmatrix:
        gva = load_group_sizes(va_path, feats, args.val_group_sizes_path)
        va_gids = gids_from_sizes(gva)
    cache_dir = args.label_cache_dir or (Path("/tmp") / "request_corrected_label_cache")
    request_feature_cache_dir = args.request_feature_cache_dir or cache_dir

    params = dict(PARAMS)
    params.update(PARAM_PRESETS[args.preset])
    params.update({"seed": args.seed})
    params.update(parse_xgb_params(args.xgb_param))
    model_dir = args.model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "feature_set": args.feature_set,
        "preset": args.preset,
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
        "skip_val_dmatrix": bool(args.skip_val_dmatrix),
        "cache_dmatrix": bool(args.cache_dmatrix),
        "add_exact_request_feature": bool(args.add_exact_request_feature),
        "add_hard_artist_feature": bool(args.add_hard_artist_feature),
        "hard_artist_feature": {
            "strict": not bool(args.hard_artist_no_strict),
            "exclude_exact_requests": bool(args.hard_artist_exclude_exact_requests),
            "simple_artist_only": bool(args.hard_artist_simple_only),
        },
        "train_feature_path": str(tr_path),
        "train_group_sizes_path": str(args.train_group_sizes_path) if args.train_group_sizes_path else None,
        "train_sources": [
            {
                "path": str(src.path),
                "split": src.split,
                "group_sizes_path": str(group_path) if group_path else None,
                "corrections_path": str(src.corrections_path) if src.corrections_path else None,
                "n_rows": int(src.gs.sum()),
                "n_groups": int(len(src.gs)),
            }
            for src, (_, group_path, _, _) in zip(train_sources, train_specs)
        ],
        "train_n_rows": int(gtr.sum()),
        "train_n_groups": int(len(gtr)),
        "val_feature_path": str(va_path),
        "val_group_sizes_path": str(args.val_group_sizes_path) if args.val_group_sizes_path else None,
        "train_corrections": str(args.train_corrections) if args.train_corrections else None,
        "extra_train_corrections": [str(p) if p else None for p in extra_correction_paths],
        "val_corrections": str(args.val_corrections) if args.val_corrections else None,
        "cells": {},
    }
    tr_request_feature = None
    va_request_feature = None
    tr_request_feature_cache = None
    va_request_feature_cache = None
    tr_extra_features: list[tuple[str, np.ndarray]] = []
    va_extra_features: list[tuple[str, np.ndarray]] = []
    request_feature_summaries: list[dict[str, str | None]] = []
    if args.add_exact_request_feature:
        train_exact_summaries = []
        for src in train_sources:
            tr_request_feature_cache = build_or_load_exact_request_feature_cache(
                path=src.path,
                gs=src.gs,
                feature_set=args.feature_set,
                split=src.split,
                catalog=catalog,
                cache_dir=request_feature_cache_dir,
                force=args.force_request_feature_cache,
            )
            tr_request_feature = np.load(tr_request_feature_cache["feature"], mmap_mode="r")
            src.extra_features.append((EXACT_REQUEST_FEATURE, tr_request_feature))
            train_exact_summaries.append(str(tr_request_feature_cache["summary"]))
        if not args.skip_val_dmatrix:
            assert gva is not None
            va_request_feature_cache = build_or_load_exact_request_feature_cache(
                path=va_path,
                gs=gva,
                feature_set=args.feature_set,
                split="val",
                catalog=catalog,
                cache_dir=request_feature_cache_dir,
                force=args.force_request_feature_cache,
            )
            va_request_feature = np.load(va_request_feature_cache["feature"], mmap_mode="r")
            va_extra_features.append((EXACT_REQUEST_FEATURE, va_request_feature))
        exact_summary = {
            "name": EXACT_REQUEST_FEATURE,
            "train_summary": train_exact_summaries[0],
            "train_summaries": train_exact_summaries,
            "val_summary": str(va_request_feature_cache["summary"]) if va_request_feature_cache else None,
        }
        request_feature_summaries.append(exact_summary)
        summary["request_feature"] = exact_summary
    if args.add_hard_artist_feature:
        hard_strict = not bool(args.hard_artist_no_strict)
        train_hard_summaries = []
        for src in train_sources:
            tr_hard_cache = build_or_load_hard_artist_feature_cache(
                path=src.path,
                gs=src.gs,
                feature_set=args.feature_set,
                split=src.split,
                catalog=catalog,
                cache_dir=request_feature_cache_dir,
                force=args.force_request_feature_cache,
                strict=hard_strict,
                exclude_exact_requests=args.hard_artist_exclude_exact_requests,
                simple_artist_only=args.hard_artist_simple_only,
            )
            src.extra_features.append((HARD_ARTIST_FEATURE, np.load(tr_hard_cache["feature"], mmap_mode="r")))
            train_hard_summaries.append(str(tr_hard_cache["summary"]))
        va_hard_cache = None
        if not args.skip_val_dmatrix:
            assert gva is not None
            va_hard_cache = build_or_load_hard_artist_feature_cache(
                path=va_path,
                gs=gva,
                feature_set=args.feature_set,
                split="val",
                catalog=catalog,
                cache_dir=request_feature_cache_dir,
                force=args.force_request_feature_cache,
                strict=hard_strict,
                exclude_exact_requests=args.hard_artist_exclude_exact_requests,
                simple_artist_only=args.hard_artist_simple_only,
            )
            va_extra_features.append((HARD_ARTIST_FEATURE, np.load(va_hard_cache["feature"], mmap_mode="r")))
        request_feature_summaries.append({
            "name": HARD_ARTIST_FEATURE,
            "train_summary": train_hard_summaries[0],
            "train_summaries": train_hard_summaries,
            "val_summary": str(va_hard_cache["summary"]) if va_hard_cache else None,
        })
    if request_feature_summaries:
        summary["request_features"] = request_feature_summaries

    for raw_mode in args.cells:
        mode = canonical_mode(raw_mode)
        dmatrix_cache_prefix = None
        dtr = None
        dva = None
        bst = None
        wg = None
        scores = None
        y_official = None
        y_corrected = None
        tva = None
        official_cache = None
        print(f"[request-corrected] preparing label cache mode={mode}", flush=True)
        train_caches = []
        train_iter_sources = []
        for src in train_sources:
            tr_cache = build_or_load_label_cache(
                path=src.path,
                gs=src.gs,
                feature_set=args.feature_set,
                split=src.split,
                mode=mode,
                corrections_path=src.corrections_path,
                corrections=src.corrections,
                catalog=catalog,
                cache_dir=cache_dir,
                force=args.force_label_cache,
                corrected_group_weight=args.corrected_group_weight,
            )
            train_caches.append(tr_cache)
            if not args.prepare_label_cache_only:
                assert src.gids is not None
                train_iter_sources.append({
                    "path": src.path,
                    "gids": src.gids,
                    "labels": np.load(tr_cache["labels"], mmap_mode="r"),
                    "extra_features": src.extra_features,
                })
        tr_cache = train_caches[0]
        va_cache = None
        va_labels = None
        if not args.skip_val_dmatrix:
            assert gva is not None
            va_cache = build_or_load_label_cache(
                path=va_path,
                gs=gva,
                feature_set=args.feature_set,
                split="val",
                mode=mode,
                corrections_path=args.val_corrections,
                corrections=val_corr,
                catalog=catalog,
                cache_dir=cache_dir,
                force=args.force_label_cache,
                corrected_group_weight=args.corrected_group_weight,
            )
            va_labels = np.load(va_cache["labels"], mmap_mode="r")
        if args.prepare_label_cache_only:
            summary["cells"][mode] = {
                "model_path": None,
                "best_iteration": None,
                "official_val_ndcg20": None,
                "official_val_recoverable": None,
                "corrected_val_ndcg20": None,
                "corrected_val_recoverable": None,
                "delta_corrected_minus_official": None,
                "official_per_turn": None,
                "corrected_per_turn": None,
                "train_label_cache_summary": str(tr_cache["summary"]),
                "train_label_cache_summaries": [str(cache["summary"]) for cache in train_caches],
                "val_label_cache_summary": str(va_cache["summary"]) if va_cache else None,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            train_caches.clear()
            train_iter_sources.clear()
            va_labels = None
            gc.collect()
            continue
        print(f"[request-corrected] building DMatrix mode={mode}", flush=True)
        assert tr_gids is not None
        dmatrix_kwargs = {}
        if "max_bin" in params:
            dmatrix_kwargs["max_bin"] = int(params["max_bin"])
        if args.cache_dmatrix:
            dmatrix_cache_dir = args.dmatrix_cache_dir or (Path("/tmp") / "request_corrected_dmatrix_cache")
            dmatrix_cache_dir.mkdir(parents=True, exist_ok=True)
            dmatrix_cache_prefix = str(dmatrix_cache_dir / f"train_{args.feature_set}_{mode}_{args.output_tag}")
            dtr = xgb.ExtMemQuantileDMatrix(
                CachedLabelMultiPoolIter(train_iter_sources, feats, cache_prefix=dmatrix_cache_prefix),
                **dmatrix_kwargs,
            )
            print(f"[request-corrected] ExtMemQuantileDMatrix pages -> {dmatrix_cache_prefix}*", flush=True)
        else:
            dtr = xgb.QuantileDMatrix(
                CachedLabelMultiPoolIter(train_iter_sources, feats),
                **dmatrix_kwargs,
            )
        if mode in {"exact_positive_weighted", "exact_positive_graded_weighted",
                    "request_positive_masked", "request_positive_weighted",
                    "violation_drop"}:
            wg = np.concatenate([np.load(cache["group_weight"], mmap_mode="r") for cache in train_caches])
            dtr.set_weight(group_weights_from_dmatrix(dtr, tr_gids, wg))
        evals = []
        early_stopping_rounds = None
        if not args.skip_val_dmatrix:
            assert va_gids is not None and va_labels is not None
            dva = xgb.QuantileDMatrix(
                CachedLabelPoolIter(va_path, feats, va_gids, va_labels, extra_features=va_extra_features),
                ref=dtr,
                **dmatrix_kwargs,
            )
            evals = [(dva, "val")]
            early_stopping_rounds = args.early_stopping_rounds
        bst = xgb.train(
            params,
            dtr,
            num_boost_round=args.num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50,
        )
        best_iteration = booster_best_iteration(bst, args.num_boost_round - 1)
        official_pb = official_rec = corrected_pb = corrected_rec = None
        official_turn = corrected_turn = None
        if not args.skip_val_dmatrix:
            assert gva is not None and va_labels is not None
            scores = bst.predict(dva, iteration_range=(0, best_iteration + 1))
            official_cache = build_or_load_label_cache(
                path=va_path,
                gs=gva,
                feature_set=args.feature_set,
                split="val",
                mode="official",
                corrections_path=None,
                corrections={},
                catalog=catalog,
                cache_dir=cache_dir,
                force=args.force_label_cache,
                corrected_group_weight=args.corrected_group_weight,
            )
            y_official = np.load(official_cache["labels"], mmap_mode="r")
            tva = np.load(official_cache["turn"], mmap_mode="r")
            y_corrected = va_labels
            official_pb, official_rec, official_turn = ndcg20(scores, y_official, gva, tva)
            corrected_pb, corrected_rec, corrected_turn = ndcg20(scores, y_corrected, gva, tva)
        model_path = model_dir / f"request_corrected_{args.feature_set}_{mode}_{args.output_tag}.json"
        bst.save_model(model_path)
        summary["cells"][mode] = {
            "model_path": str(model_path),
            "best_iteration": best_iteration,
            "official_val_ndcg20": official_pb,
            "official_val_recoverable": official_rec,
            "corrected_val_ndcg20": corrected_pb,
            "corrected_val_recoverable": corrected_rec,
            "delta_corrected_minus_official": corrected_pb - official_pb if corrected_pb is not None and official_pb is not None else None,
            "official_per_turn": official_turn,
            "corrected_per_turn": corrected_turn,
            "train_label_cache_summary": str(tr_cache["summary"]),
            "train_label_cache_summaries": [str(cache["summary"]) for cache in train_caches],
            "val_label_cache_summary": str(va_cache["summary"]) if va_cache else None,
            "train_request_feature_summary": str(tr_request_feature_cache["summary"]) if tr_request_feature_cache else None,
            "val_request_feature_summary": str(va_request_feature_cache["summary"]) if va_request_feature_cache else None,
            "request_features": request_feature_summaries,
        }
        if official_pb is None or corrected_pb is None:
            print(f"[request-corrected] {mode}: trained model={model_path}", flush=True)
        else:
            print(
                f"[request-corrected] {mode}: official={official_pb:.5f} corrected={corrected_pb:.5f}",
                flush=True,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        dtr = None
        dva = None
        bst = None
        wg = None
        scores = None
        y_official = None
        y_corrected = None
        tva = None
        official_cache = None
        train_caches.clear()
        train_iter_sources.clear()
        va_labels = None
        gc.collect()
        cleaned = cleanup_xgb_external_cache(dmatrix_cache_prefix)
        if cleaned:
            print(
                f"[request-corrected] cleaned {cleaned} external-memory files for mode={mode}",
                flush=True,
            )
        gc.collect()
    return summary


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="F10_R54SRC")
    ap.add_argument("--preset", choices=sorted(PARAM_PRESETS), default="r54_shallow_topk20")
    ap.add_argument("--train-feature-path", type=Path,
                    help="Override the training feature parquet, e.g. train_lt_105k.")
    ap.add_argument("--train-group-sizes-path", type=Path,
                    help="Optional row-aligned group sizes for --train-feature-path.")
    ap.add_argument("--extra-train-feature-path", type=Path, action="append", default=[],
                    help="Additional training feature parquet; repeatable, e.g. val for full organizer train.")
    ap.add_argument("--extra-train-group-sizes-path", type=Path, action="append", default=[],
                    help="Group sizes for each --extra-train-feature-path; repeatable.")
    ap.add_argument("--extra-train-split-name", action="append", default=[],
                    help="Split name for each extra training feature parquet; repeatable.")
    ap.add_argument("--extra-train-corrections", type=Path, action="append", default=[],
                    help="Correction sidecar for each extra training source; repeatable.")
    ap.add_argument("--val-group-sizes-path", type=Path,
                    help="Optional group sizes for the validation feature parquet.")
    ap.add_argument("--train-split-name", default="train_lt",
                    help="Name used in label-cache summaries when --train-feature-path is supplied.")
    ap.add_argument("--train-corrections", type=Path)
    ap.add_argument("--val-corrections", type=Path)
    ap.add_argument("--cells", nargs="+",
                    choices=["official", "exact_positive", "exact_positive_graded",
                             "exact_positive_request_preferred",
                             "exact_positive_weighted", "exact_positive_graded_weighted",
                             "request_positive",
                             "request_positive_weighted", "request_positive_masked",
                             "violation_drop",
                             "positive", "positive_masked"],
                    default=["official", "exact_positive", "request_positive", "request_positive_masked"])
    ap.add_argument("--num-boost-round", type=int, default=300)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--xgb-param", action="append", default=[],
                    help="Additional XGBoost key=value override; repeatable.")
    ap.add_argument("--output-tag", default="v0")
    ap.add_argument("--model-dir", type=Path,
                    help="Override model output directory; useful for smoke runs in /tmp.")
    ap.add_argument("--label-cache-dir", type=Path,
                    help="Directory for row-aligned corrected label caches.")
    ap.add_argument("--force-label-cache", action="store_true")
    ap.add_argument("--add-exact-request-feature", action="store_true",
                    help="Append a non-oracle exact-request-match feature resolved from visible dialogue.")
    ap.add_argument("--add-hard-artist-feature", action="store_true",
                    help="Append a non-oracle hard-artist-constraint-match feature resolved from visible dialogue.")
    ap.add_argument("--hard-artist-no-strict", action="store_true",
                    help="Use broad hard-artist directives for the hard-artist feature.")
    ap.add_argument("--hard-artist-exclude-exact-requests", action="store_true",
                    help="Skip hard-artist feature directives that contain a resolved exact-title request.")
    ap.add_argument("--hard-artist-simple-only", action="store_true",
                    help="Use the abstaining simple-artist-only hard-feature gate.")
    ap.add_argument("--request-feature-cache-dir", type=Path,
                    help="Directory for row-aligned exact-request feature caches.")
    ap.add_argument("--force-request-feature-cache", action="store_true")
    ap.add_argument("--corrected-group-weight", type=float, default=0.5,
                    help="Training group weight for request_positive_weighted corrected groups.")
    ap.add_argument("--cache-dmatrix", action="store_true",
                    help="Use XGBoost ExtMemQuantileDMatrix for the training matrix.")
    ap.add_argument("--dmatrix-cache-dir", type=Path,
                    help="Directory for external-memory DMatrix pages.")
    ap.add_argument("--prepare-label-cache-only", action="store_true",
                    help="Build corrected row-label caches and exit without training.")
    ap.add_argument("--skip-val-dmatrix", action="store_true",
                    help="Train fixed-round models without holding the validation DMatrix; evaluate saved models separately.")
    ap.add_argument("--output", type=Path, default=ROOT / "exp/scores/request_corrected_rerank_train.json")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train_cells(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
