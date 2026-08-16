"""Paired bootstrap for request-corrected split readouts.

The unit is session, matching the dialogue structure. This compares two dev
prediction files under both the frozen official label and the opt-in corrected
request-aware label.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(os.environ.get("MCRS_EXPLORE_ROOT", str(Path(__file__).resolve().parent.parent)))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from evaluate_request_slice import (  # noqa: E402
    load_corrections,
    load_predictions,
    ndcg_for_positives,
)
from detect_requests import (  # noqa: E402
    build_catalog_index,
    gold_for_turn,
    load_sessions,
    track_satisfies_constraint,
)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def corrected_turn_score(
    predicted: list[str],
    gold: str,
    rows: list[dict[str, Any]],
    catalog,
    scoring: str,
) -> float:
    positives = {gold}
    for row in rows:
        positives.update(row.get("additional_track_ids") or [])
        if row.get("positive_constraint"):
            constraint = row["positive_constraint"]
            positives.update(tid for tid in predicted[:20] if track_satisfies_constraint(tid, constraint, catalog))
    return ndcg_for_positives(predicted, positives, scoring=scoring)


def session_metric_arrays(
    *,
    prediction_path: Path,
    corrections_path: Path,
    split: str,
    scoring: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from datasets import load_dataset

    preds = load_predictions(prediction_path)
    corrections = load_corrections(corrections_path)
    catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
    sessions = load_sessions(split)
    official_by_session: list[float] = []
    corrected_by_session: list[float] = []
    session_ids: list[str] = []
    for session in sessions:
        sid = session["session_id"]
        official_vals: list[float] = []
        corrected_vals: list[float] = []
        for turn_number in range(1, 9):
            key = (sid, turn_number)
            gold = gold_for_turn(session["conversations"], turn_number)
            if gold is None:
                continue
            predicted = preds[key].get("predicted_track_ids", [])
            rows = corrections.get(key, [])
            official_vals.append(ndcg_for_positives(predicted, {gold}, scoring=scoring))
            corrected_vals.append(corrected_turn_score(predicted, gold, rows, catalog, scoring))
        if official_vals:
            session_ids.append(sid)
            official_by_session.append(mean(official_vals))
            corrected_by_session.append(mean(corrected_vals))
    return (
        np.asarray(official_by_session, dtype=np.float64),
        np.asarray(corrected_by_session, dtype=np.float64),
        session_ids,
    )


def session_family_arrays(
    *,
    prediction_path: Path,
    corrections_path: Path,
    split: str,
    scoring: str,
    family: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from datasets import load_dataset

    preds = load_predictions(prediction_path)
    corrections = load_corrections(corrections_path)
    catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
    sessions = load_sessions(split)
    official_by_session: list[float] = []
    corrected_by_session: list[float] = []
    session_ids: list[str] = []
    for session in sessions:
        sid = session["session_id"]
        official_vals: list[float] = []
        corrected_vals: list[float] = []
        for turn_number in range(1, 9):
            key = (sid, turn_number)
            gold = gold_for_turn(session["conversations"], turn_number)
            if gold is None:
                continue
            predicted = preds[key].get("predicted_track_ids", [])
            for row in corrections.get(key, []):
                if row.get("family") != family:
                    continue
                positives = {gold}
                positives.update(row.get("additional_track_ids") or [])
                if row.get("positive_constraint"):
                    constraint = row["positive_constraint"]
                    positives.update(tid for tid in predicted[:20] if track_satisfies_constraint(tid, constraint, catalog))
                official_vals.append(ndcg_for_positives(predicted, {gold}, scoring=scoring))
                corrected_vals.append(ndcg_for_positives(predicted, positives, scoring=scoring))
        if official_vals:
            session_ids.append(sid)
            official_by_session.append(mean(official_vals))
            corrected_by_session.append(mean(corrected_vals))
    return (
        np.asarray(official_by_session, dtype=np.float64),
        np.asarray(corrected_by_session, dtype=np.float64),
        session_ids,
    )


def array_mean(xs: np.ndarray) -> float | None:
    if len(xs) == 0:
        return None
    return float(xs.mean())


def bootstrap_delta(candidate: np.ndarray, baseline: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    if candidate.shape != baseline.shape:
        raise ValueError(f"shape mismatch: candidate {candidate.shape}, baseline {baseline.shape}")
    diff = candidate - baseline
    if len(diff) == 0:
        return {
            "observed_delta": None,
            "bootstrap_median": None,
            "ci95_low": None,
            "ci95_high": None,
            "p_delta_le_0": None,
            "n_sessions": 0,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    vals = diff[idx].mean(axis=1)
    lo, med, hi = np.quantile(vals, [0.025, 0.5, 0.975])
    return {
        "observed_delta": float(diff.mean()),
        "bootstrap_median": float(med),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_delta_le_0": float(np.mean(vals <= 0.0)),
        "n_sessions": int(len(diff)),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-prediction", type=Path, required=True)
    ap.add_argument("--candidate-prediction", type=Path, required=True)
    ap.add_argument("--baseline-name", default="baseline")
    ap.add_argument("--candidate-name", default="candidate")
    ap.add_argument("--corrections", type=Path, required=True)
    ap.add_argument("--split", choices=["devset", "val", "train_lt"], default="devset")
    ap.add_argument("--scoring", choices=["best_hit", "multilabel"], default="best_hit")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--family", action="append",
                    help="Optional positive-slice family to bootstrap, e.g. exact_track_request.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    base_official, base_corrected, base_sessions = session_metric_arrays(
        prediction_path=args.baseline_prediction,
        corrections_path=args.corrections,
        split=args.split,
        scoring=args.scoring,
    )
    cand_official, cand_corrected, cand_sessions = session_metric_arrays(
        prediction_path=args.candidate_prediction,
        corrections_path=args.corrections,
        split=args.split,
        scoring=args.scoring,
    )
    if base_sessions != cand_sessions:
        raise ValueError("prediction files produced different session order")

    result = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline_prediction": str(args.baseline_prediction),
        "candidate_prediction": str(args.candidate_prediction),
        "corrections": str(args.corrections),
        "split": args.split,
        "scoring": args.scoring,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "official": {
            "baseline_mean": array_mean(base_official),
            "candidate_mean": array_mean(cand_official),
            "delta": bootstrap_delta(cand_official, base_official, args.n_boot, args.seed),
        },
        "corrected": {
            "baseline_mean": array_mean(base_corrected),
            "candidate_mean": array_mean(cand_corrected),
            "delta": bootstrap_delta(cand_corrected, base_corrected, args.n_boot, args.seed + 1),
        },
        "families": {},
    }
    for idx, family in enumerate(args.family or []):
        base_family_official, base_family_corrected, base_family_sessions = session_family_arrays(
            prediction_path=args.baseline_prediction,
            corrections_path=args.corrections,
            split=args.split,
            scoring=args.scoring,
            family=family,
        )
        cand_family_official, cand_family_corrected, cand_family_sessions = session_family_arrays(
            prediction_path=args.candidate_prediction,
            corrections_path=args.corrections,
            split=args.split,
            scoring=args.scoring,
            family=family,
        )
        if base_family_sessions != cand_family_sessions:
            raise ValueError(f"family {family!r} produced different session order")
        result["families"][family] = {
            "n_sessions": len(base_family_sessions),
            "official": {
                "baseline_mean": array_mean(base_family_official),
                "candidate_mean": array_mean(cand_family_official),
                "delta": bootstrap_delta(
                    cand_family_official,
                    base_family_official,
                    args.n_boot,
                    args.seed + 101 + idx * 2,
                ),
            },
            "corrected": {
                "baseline_mean": array_mean(base_family_corrected),
                "candidate_mean": array_mean(cand_family_corrected),
                "delta": bootstrap_delta(
                    cand_family_corrected,
                    base_family_corrected,
                    args.n_boot,
                    args.seed + 102 + idx * 2,
                ),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
