"""Shared reranker constants and streaming helpers (artifact-trimmed).

This module carries the production feature-set definition, LambdaRank
parameters, and parquet streaming helpers that the training and evaluation
scripts import. It is a trimmed copy of the research repo's rerank_train.py:
the training entrypoint and unrelated feature sets are removed; the constants
below are byte-equal to the production values.

The 49 production feature names (FEATURE_SETS["F10_R54SRC"]) are verifiable
against the shipped model dumps: evidence/models/*.json embed the same list
plus the appended exact_request_match feature.

Data layout: point MCRS_EXPLORE_ROOT at a directory containing exp/features
(feature parquets + group-size .npy files), exp/models, exp/scores. See the
README's data-access notes; feature parquets are not redistributed here.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("MCRS_EXPLORE_ROOT", str(Path(__file__).resolve().parent.parent)))
FEAT = ROOT / "exp/features"
MODEL_DIR = ROOT / "exp/models"
SCORE = ROOT / "exp/scores"

SEED = 42
BATCH = int(os.environ.get("RERANK_BATCH_ROWS", "500000"))

# Production feature columns (matches the shipped model dumps' feature_names,
# minus the exact_request_match column appended at training time).
BASE_FEATURES = [
    "rrf_score", "inv_rank", "pop", "cf_sim", "sem_sim_meta", "sem_sim_lyrics",
    "sem_sim_attrs", "title_overlap", "artist_overlap", "tag_overlap",
    "is_artist_match", "prior_artist_match", "turn_number_feat",
]
LIST_NAMES = ["bm25", "meta_prf", "lyr_prf", "attr_prf", "siglip_prf",
              "meta", "lyr", "attr", "bm25_last_meta", "bm25_full"]
R54_LIST_NAMES = LIST_NAMES + ["r54"]


def ndcg20(scores: np.ndarray, y: np.ndarray, gs: np.ndarray, turn: np.ndarray):
    """Official binary nDCG@20 per group. Returns (pool_bounded_mean, recoverable_mean,
    per_turn_pool_bounded dict). Rows are group-contiguous in gs order."""
    starts = np.zeros(len(gs), dtype=np.int64)
    starts[1:] = np.cumsum(gs)[:-1]
    vals = np.zeros(len(gs), dtype=np.float64)
    has_gold = np.zeros(len(gs), dtype=bool)
    grp_turn = np.zeros(len(gs), dtype=np.int32)
    for gi in range(len(gs)):
        st = starts[gi]; sz = gs[gi]
        sl = slice(st, st + sz)
        yg = y[sl]
        grp_turn[gi] = turn[st]
        if yg.sum() == 0:
            continue  # gold not in pool -> nDCG 0
        has_gold[gi] = True
        order = np.argsort(-scores[sl], kind="stable")
        rank = int(np.where(yg[order] == 1)[0][0]) + 1
        if rank <= 20:
            vals[gi] = 1.0 / np.log2(rank + 1)
    pool_bounded = float(vals.mean())
    recoverable = float(vals[has_gold].mean()) if has_gold.any() else 0.0
    per_turn = {int(t): round(float(vals[grp_turn == t].mean()), 4) for t in np.unique(grp_turn)}
    return pool_bounded, recoverable, per_turn


def per_source_cols(list_names: list[str]) -> list[str]:
    return (
        [f"src_{n}_invrank" for n in list_names]
        + [f"src_{n}_present" for n in list_names]
        + ["n_sources"]
    )


CONTINUITY_COLS = [
    "same_album_last1", "same_album_last3", "same_album_any",
    "album_history_count", "pool_same_album_count", "pool_artist_frac", "pool_artist_count",
]
F10_COLS = ["recency_score", "is_played", "last_tag_jaccard"]
R54_RANK_COLS = ["r54_present", "r54_invrank", "r54_rank_pct"]

FEATURE_SETS = {
    "F10_R54SRC": (
        BASE_FEATURES
        + per_source_cols(R54_LIST_NAMES)
        + CONTINUITY_COLS
        + F10_COLS
        + R54_RANK_COLS
    ),
}
FEATURE_SUFFIX = {
    "F10_R54SRC": "_R54src_F10R54",
}

PARAM_PRESETS = {
    "default": {},
    "r54_shallow_topk20": {
        "max_depth": 4,
        "min_child_weight": 10,
        "lambdarank_pair_method": "topk",
        "lambdarank_num_pair_per_sample": 20,
    },
}
PARAMS = {"objective": "rank:ndcg", "eval_metric": "ndcg@20", "tree_method": "hist",
          "eta": 0.1, "max_depth": 6, "min_child_weight": 5, "subsample": 0.8,
          "colsample_bytree": 0.8, "lambda": 1.0, "seed": 42, "nthread": 8}


def parquet_path(sp, s):
    suf = FEATURE_SUFFIX.get(s, "")
    return FEAT / f"features_rerank_{sp}{suf}.parquet"


def parse_xgb_value(raw: str):
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_xgb_params(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--xgb-param must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = parse_xgb_value(v)
    return out


def group_sizes_for(path, feats, keep):
    """Stream once to compute group_sizes over the (optionally session-filtered) rows,
    in file order. A group is a contiguous (session_id,turn_number) run."""
    pf = pq.ParquetFile(path)
    sizes = []
    cur_key = None
    cur_n = 0
    for b in pf.iter_batches(batch_size=BATCH, columns=["session_id", "turn_number"]):
        sid = b.column("session_id").to_pylist()
        tn = b.column("turn_number").to_pylist()
        for i in range(len(sid)):
            if keep is not None and sid[i] not in keep:
                continue
            k = (sid[i], tn[i])
            if k != cur_key:
                if cur_key is not None:
                    sizes.append(cur_n)
                cur_key = k
                cur_n = 1
            else:
                cur_n += 1
    if cur_key is not None:
        sizes.append(cur_n)
    return np.asarray(sizes, dtype=np.int64)


def gids_from_sizes(gs):
    return np.repeat(np.arange(len(gs), dtype=np.int64), gs)
