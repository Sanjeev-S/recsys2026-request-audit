"""Evaluate predictions against request-aware correction sidecars.

This is deliberately separate from scripts/evaluate_devset.py, which remains the
frozen official metric. The corrected readout reports:

  * official nDCG@20 against policy-selected labels,
  * request-positive nDCG@20 with additional positives from the sidecar, and
  * constraint/rejection violation rates for rows where the sidecar masks the
    policy-selected label.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("MCRS_EXPLORE_ROOT", str(Path(__file__).resolve().parent.parent)))

from detect_requests import (  # noqa: E402
    build_catalog_index,
    gold_for_turn,
    load_sessions,
    metadata_summary,
    norm_values,
    track_satisfies_constraint,
)


def best_hit_ndcg(predicted: list[str], positives: set[str], k: int = 20) -> float:
    for rank, tid in enumerate(predicted[:k], start=1):
        if tid in positives:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def multilabel_ndcg(predicted: list[str], positives: set[str], k: int = 20) -> float:
    if not positives:
        return 0.0
    dcg = 0.0
    for rank, tid in enumerate(predicted[:k], start=1):
        if tid in positives:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(positives), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def ndcg_for_positives(
    predicted: list[str],
    positives: set[str],
    k: int = 20,
    scoring: str = "best_hit",
) -> float:
    if scoring == "best_hit":
        return best_hit_ndcg(predicted, positives, k)
    if scoring == "multilabel":
        return multilabel_ndcg(predicted, positives, k)
    raise ValueError(f"unknown scoring mode {scoring!r}")


def first_violation_rank(predicted: list[str], constraints: list[dict[str, Any]], catalog, k: int) -> int | None:
    if not constraints:
        return None
    for rank, tid in enumerate(predicted[:k], start=1):
        if any(not track_satisfies_constraint(tid, c, catalog) for c in constraints):
            return rank
    return None


def shares_artist(track_a: str, track_b: str, catalog) -> bool:
    row_a = catalog.by_id.get(track_a)
    row_b = catalog.by_id.get(track_b)
    if not row_a or not row_b:
        return False
    return bool(set(norm_values(row_a.get("artist_name"))) & set(norm_values(row_b.get("artist_name"))))


def first_switchaway_violation_rank(row: dict[str, Any], predicted: list[str], catalog, k: int) -> int | None:
    evidence = row.get("evidence") or {}
    prev = evidence.get("previous_track_id")
    if not prev:
        return None
    repeat_violation = bool(evidence.get("repeat_violation"))
    switch_violation = bool(evidence.get("switch_violation"))
    explicit_artist = evidence.get("explicit_artist_norm")
    explicit_constraint = {"kind": "artist", "artist_norm": explicit_artist} if explicit_artist else None
    for rank, tid in enumerate(predicted[:k], start=1):
        if repeat_violation and tid == prev:
            return rank
        if switch_violation and shares_artist(tid, prev, catalog):
            return rank
        if explicit_constraint and track_satisfies_constraint(tid, explicit_constraint, catalog):
            return rank
    return None


def load_predictions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in rows:
        key = (row["session_id"], int(row["turn_number"]))
        if key in out:
            raise ValueError(f"duplicate prediction key {key}")
        out[key] = row
    return out


def load_corrections(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[(row["session_id"], int(row["turn_number"]))].append(row)
    return dict(out)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def evaluate_corrected(
    *,
    prediction_path: Path,
    corrections_path: Path,
    split: str,
    scoring: str = "best_hit",
    max_sessions: int | None = None,
) -> dict[str, Any]:
    from datasets import load_dataset

    preds = load_predictions(prediction_path)
    corrections = load_corrections(corrections_path)
    catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
    sessions = load_sessions(split)
    if max_sessions is not None:
        sessions = sessions.select(range(min(max_sessions, len(sessions))))

    official_vals: list[float] = []
    corrected_vals: list[float] = []
    positive_slice_official: list[float] = []
    positive_slice_corrected: list[float] = []
    constraint_rows = 0
    constraint_at1 = 0
    constraint_at5 = 0
    constraint_at20 = 0
    switch_rows = 0
    switch_at1 = 0
    switch_at5 = 0
    switch_at20 = 0
    turn_vals: dict[int, list[float]] = defaultdict(list)
    family_counts: dict[str, int] = defaultdict(int)
    family_positive: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"official": [], "corrected": []})
    family_violation: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "at1": 0, "at5": 0, "at20": 0})
    changed_rows: list[dict[str, Any]] = []

    for session in sessions:
        sid = session["session_id"]
        for turn_number in range(1, 9):
            key = (sid, turn_number)
            if key not in preds:
                raise ValueError(f"missing prediction for {key}")
            gold = gold_for_turn(session["conversations"], turn_number)
            if gold is None:
                continue
            predicted = preds[key].get("predicted_track_ids", [])
            official = ndcg_for_positives(predicted, {gold}, scoring=scoring)
            positives = {gold}
            constraints: list[dict[str, Any]] = []
            for row in corrections.get(key, []):
                family = row["family"]
                family_counts[family] += 1
                row_positives = {gold}
                row_positives.update(row.get("additional_track_ids") or [])
                row_constraints = []
                if row.get("positive_constraint"):
                    row_constraints.append(row["positive_constraint"])
                    row_positives.update(
                        tid for tid in predicted[:20]
                        if track_satisfies_constraint(tid, row["positive_constraint"], catalog)
                    )
                positives.update(row.get("additional_track_ids") or [])
                if row.get("positive_constraint"):
                    constraints.append(row["positive_constraint"])
                    positives.update(tid for tid in predicted[:20] if track_satisfies_constraint(tid, row["positive_constraint"], catalog))
                    family_violation[family]["n"] += 1
                    family_violation[family]["at1"] += int(first_violation_rank(predicted, row_constraints, catalog, 1) is not None)
                    family_violation[family]["at5"] += int(first_violation_rank(predicted, row_constraints, catalog, 5) is not None)
                    family_violation[family]["at20"] += int(first_violation_rank(predicted, row_constraints, catalog, 20) is not None)
                if family == "rejection_switchaway_violation":
                    family_violation[family]["n"] += 1
                    family_violation[family]["at1"] += int(first_switchaway_violation_rank(row, predicted, catalog, 1) is not None)
                    family_violation[family]["at5"] += int(first_switchaway_violation_rank(row, predicted, catalog, 5) is not None)
                    family_violation[family]["at20"] += int(first_switchaway_violation_rank(row, predicted, catalog, 20) is not None)
                if row_positives != {gold}:
                    family_positive[family]["official"].append(official)
                    family_positive[family]["corrected"].append(
                        ndcg_for_positives(predicted, row_positives, scoring=scoring)
                    )

            corrected = ndcg_for_positives(predicted, positives, scoring=scoring)
            official_vals.append(official)
            corrected_vals.append(corrected)
            turn_vals[turn_number].append(corrected)
            if corrections.get(key) and positives != {gold}:
                positive_slice_official.append(official)
                positive_slice_corrected.append(corrected)
                if abs(corrected - official) > 1e-12:
                    changed_rows.append({
                        "session_id": sid,
                        "turn_number": turn_number,
                        "official": official,
                        "corrected": corrected,
                        "gold": metadata_summary(catalog.by_id.get(gold)),
                        "positives": [metadata_summary(catalog.by_id.get(tid)) for tid in sorted(positives)[:10]],
                    })
            if constraints:
                constraint_rows += 1
                rank1 = first_violation_rank(predicted, constraints, catalog, 1)
                rank5 = first_violation_rank(predicted, constraints, catalog, 5)
                rank20 = first_violation_rank(predicted, constraints, catalog, 20)
                constraint_at1 += int(rank1 is not None)
                constraint_at5 += int(rank5 is not None)
                constraint_at20 += int(rank20 is not None)
            switch_rows_for_key = [row for row in corrections.get(key, []) if row.get("family") == "rejection_switchaway_violation"]
            if switch_rows_for_key:
                switch_rows += 1
                switch_at1 += int(any(first_switchaway_violation_rank(row, predicted, catalog, 1) is not None for row in switch_rows_for_key))
                switch_at5 += int(any(first_switchaway_violation_rank(row, predicted, catalog, 5) is not None for row in switch_rows_for_key))
                switch_at20 += int(any(first_switchaway_violation_rank(row, predicted, catalog, 20) is not None for row in switch_rows_for_key))

    return {
        "prediction_path": str(prediction_path),
        "corrections_path": str(corrections_path),
        "split": split,
        "scoring": scoring,
        "n_rows": len(official_vals),
        "official_ndcg20": mean(official_vals),
        "corrected_ndcg20": mean(corrected_vals),
        "delta_corrected_minus_official": mean(corrected_vals) - mean(official_vals),
        "positive_slice": {
            "n": len(positive_slice_official),
            "official_ndcg20": mean(positive_slice_official),
            "corrected_ndcg20": mean(positive_slice_corrected),
            "delta": mean(positive_slice_corrected) - mean(positive_slice_official),
        },
        "constraint_violation": {
            "n": constraint_rows,
            "violation_rate_at1": rate(constraint_at1, constraint_rows),
            "violation_rate_at5": rate(constraint_at5, constraint_rows),
            "violation_rate_at20": rate(constraint_at20, constraint_rows),
        },
        "switchaway_violation": {
            "n": switch_rows,
            "violation_rate_at1": rate(switch_at1, switch_rows),
            "violation_rate_at5": rate(switch_at5, switch_rows),
            "violation_rate_at20": rate(switch_at20, switch_rows),
        },
        "corrected_per_turn_ndcg20": {str(k): mean(v) for k, v in sorted(turn_vals.items())},
        "correction_records_by_family": dict(sorted(family_counts.items())),
        "positive_slice_by_family": {
            family: {
                "n": len(vals["official"]),
                "official_ndcg20": mean(vals["official"]),
                "corrected_ndcg20": mean(vals["corrected"]),
                "delta": mean(vals["corrected"]) - mean(vals["official"]),
            }
            for family, vals in sorted(family_positive.items())
        },
        "violation_by_family": {
            family: {
                "n": vals["n"],
                "violation_rate_at1": rate(vals["at1"], vals["n"]),
                "violation_rate_at5": rate(vals["at5"], vals["n"]),
                "violation_rate_at20": rate(vals["at20"], vals["n"]),
            }
            for family, vals in sorted(family_violation.items())
        },
        "changed_rows_preview": changed_rows[:25],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-path", type=Path, required=True)
    ap.add_argument("--corrections", type=Path, required=True)
    ap.add_argument("--split", choices=["devset", "val", "train_lt", "train", "train_full"], default="devset")
    ap.add_argument("--scoring", choices=["best_hit", "multilabel"], default="best_hit",
                    help="best_hit is conservative and preserves one-satisfied-request semantics; "
                         "multilabel is standard graded nDCG over all positives.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-sessions", type=int)
    args = ap.parse_args(argv)
    result = evaluate_corrected(
        prediction_path=args.prediction_path,
        corrections_path=args.corrections,
        split=args.split,
        scoring=args.scoring,
        max_sessions=args.max_sessions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
