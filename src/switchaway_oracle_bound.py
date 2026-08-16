"""Upper-bound audit for switch-away filtering.

This script deliberately uses the switch-away sidecar at intervention time.
That makes it non-deployable, but useful as a diagnostic upper bound:

  if an oracle sidecar filter cannot produce a clean top-k result without
  demoting the policy-selected label, then switch-away is not merely missing a
  better visible regex.

The output should be cited only as an oracle/action-feasibility audit, never as
an allowed inference method.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_request_slice import (
    first_switchaway_violation_rank,
    load_corrections,
    load_predictions,
    shares_artist,
)
from detect_requests import (
    build_catalog_index,
    metadata_summary,
    track_satisfies_constraint,
)


def switchaway_rows(corrections: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    return {
        key: [row for row in rows if row.get("family") == "rejection_switchaway_violation"]
        for key, rows in corrections.items()
        if any(row.get("family") == "rejection_switchaway_violation" for row in rows)
    }


def first_rank(tid: str | None, predicted: list[str]) -> int | None:
    if not tid:
        return None
    try:
        return predicted.index(tid) + 1
    except ValueError:
        return None


def violates_switchaway_sidecar(row: dict[str, Any], tid: str, catalog) -> bool:
    evidence = row.get("evidence") or {}
    prev = evidence.get("previous_track_id")
    if not prev:
        return False
    if evidence.get("repeat_violation") and tid == prev:
        return True
    if evidence.get("switch_violation") and shares_artist(tid, prev, catalog):
        return True
    explicit_artist = evidence.get("explicit_artist_norm")
    if explicit_artist:
        return track_satisfies_constraint(tid, {"kind": "artist", "artist_norm": explicit_artist}, catalog)
    return False


def violates_any_switchaway(rows: list[dict[str, Any]], tid: str, catalog) -> bool:
    return any(violates_switchaway_sidecar(row, tid, catalog) for row in rows)


def first_any_violation_rank(rows: list[dict[str, Any]], predicted: list[str], catalog, k: int) -> int | None:
    ranks = [
        first_switchaway_violation_rank(row, predicted, catalog, k)
        for row in rows
    ]
    ranks = [rank for rank in ranks if rank is not None]
    return min(ranks) if ranks else None


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def rank_hist(ranks: list[int | None]) -> dict[str, int]:
    counts = Counter("none" if rank is None else str(rank) for rank in ranks)
    return dict(sorted(counts.items(), key=lambda kv: (kv[0] == "none", int(kv[0]) if kv[0].isdigit() else 999)))


def apply_oracle_filter(
    *,
    predictions: dict[tuple[str, int], dict[str, Any]],
    switch_rows: dict[tuple[str, int], list[dict[str, Any]]],
    catalog,
    k_filter: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    row_details: list[dict[str, Any]] = []
    changed_keys = 0
    total_demoted = 0
    gold_moved_down = 0
    gold_moved_up = 0
    gold_violating = 0
    feasible_clean_top20 = 0
    before_at1: list[int | None] = []
    before_at5: list[int | None] = []
    before_at20: list[int | None] = []
    after_at1: list[int | None] = []
    after_at5: list[int | None] = []
    after_at20: list[int | None] = []

    for key, row in predictions.items():
        rows = switch_rows.get(key, [])
        predicted = list(row.get("predicted_track_ids") or [])
        new_predicted = predicted
        if rows:
            filtered_window = predicted[:k_filter]
            demote = [tid for tid in filtered_window if violates_any_switchaway(rows, tid, catalog)]
            demote_set = set(demote)
            allowed = [tid for tid in predicted if tid not in demote_set]
            new_predicted = (allowed + demote)[:len(predicted)]
            if new_predicted != predicted:
                changed_keys += 1
                total_demoted += len(demote)

            nonviolating_in_window = len(filtered_window) - len(demote)
            feasible_clean_top20 += int(nonviolating_in_window >= 20)

            b1 = first_any_violation_rank(rows, predicted, catalog, 1)
            b5 = first_any_violation_rank(rows, predicted, catalog, 5)
            b20 = first_any_violation_rank(rows, predicted, catalog, 20)
            a1 = first_any_violation_rank(rows, new_predicted, catalog, 1)
            a5 = first_any_violation_rank(rows, new_predicted, catalog, 5)
            a20 = first_any_violation_rank(rows, new_predicted, catalog, 20)
            before_at1.append(b1)
            before_at5.append(b5)
            before_at20.append(b20)
            after_at1.append(a1)
            after_at5.append(a5)
            after_at20.append(a20)

            gold = rows[0].get("gold_track_id")
            gold_base_rank = first_rank(gold, predicted)
            gold_after_rank = first_rank(gold, new_predicted)
            gold_is_violating = bool(gold and violates_any_switchaway(rows, gold, catalog))
            gold_violating += int(gold_is_violating)
            if gold_base_rank and gold_after_rank:
                gold_moved_down += int(gold_after_rank > gold_base_rank)
                gold_moved_up += int(gold_after_rank < gold_base_rank)

            row_details.append({
                "session_id": key[0],
                "turn_number": key[1],
                "changed": new_predicted != predicted,
                "n_demoted_in_filter": len(demote),
                "n_nonviolating_in_filter": nonviolating_in_window,
                "feasible_clean_top20_from_filter": nonviolating_in_window >= 20,
                "baseline_violation_rank_at1": b1,
                "oracle_violation_rank_at1": a1,
                "baseline_violation_rank_at5": b5,
                "oracle_violation_rank_at5": a5,
                "baseline_violation_rank_at20": b20,
                "oracle_violation_rank_at20": a20,
                "gold_is_violating": gold_is_violating,
                "gold_baseline_rank": gold_base_rank,
                "gold_oracle_rank": gold_after_rank,
                "gold": metadata_summary(catalog.by_id.get(gold)),
                "user_text": (rows[0].get("evidence") or {}).get("user_text"),
            })

        new_row = dict(row)
        new_row["predicted_track_ids"] = new_predicted
        out.append(new_row)

    n_keys = len(switch_rows)
    summary = {
        "oracle_warning": "uses switch-away sidecar labels at intervention time; non-deployable upper bound only",
        "k_filter": k_filter,
        "n_prediction_rows": len(predictions),
        "n_switchaway_keys": n_keys,
        "n_switchaway_sidecar_rows": sum(len(rows) for rows in switch_rows.values()),
        "n_changed_switchaway_keys": changed_keys,
        "n_demoted_ids": total_demoted,
        "n_gold_violating": gold_violating,
        "gold_violating_rate": rate(gold_violating, n_keys),
        "gold_moved_down": gold_moved_down,
        "gold_moved_up": gold_moved_up,
        "n_feasible_clean_top20_from_filter": feasible_clean_top20,
        "feasible_clean_top20_rate": rate(feasible_clean_top20, n_keys),
        "violation_rate_at1_before": rate(sum(rank is not None for rank in before_at1), n_keys),
        "violation_rate_at1_after": rate(sum(rank is not None for rank in after_at1), n_keys),
        "violation_rate_at5_before": rate(sum(rank is not None for rank in before_at5), n_keys),
        "violation_rate_at5_after": rate(sum(rank is not None for rank in after_at5), n_keys),
        "violation_rate_at20_before": rate(sum(rank is not None for rank in before_at20), n_keys),
        "violation_rate_at20_after": rate(sum(rank is not None for rank in after_at20), n_keys),
        "first_violation_rank_at20_before_histogram": rank_hist(before_at20),
        "first_violation_rank_at20_after_histogram": rank_hist(after_at20),
        "nonviolating_in_filter_histogram": dict(sorted(Counter(item["n_nonviolating_in_filter"] for item in row_details).items())),
    }
    return out, summary, row_details


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-path", type=Path, required=True)
    ap.add_argument("--corrections", type=Path, required=True)
    ap.add_argument("--catalog-arrow", type=Path)
    ap.add_argument("--k-filter", type=int, default=100)
    ap.add_argument("--output-prediction", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--output-rows", type=Path)
    args = ap.parse_args(argv)

    if args.catalog_arrow:
        from datasets import Dataset

        catalog_rows = Dataset.from_file(str(args.catalog_arrow))
    else:
        from datasets import load_dataset

        catalog_rows = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")

    catalog = build_catalog_index(catalog_rows)
    predictions = load_predictions(args.prediction_path)
    corrections = load_corrections(args.corrections)
    sidecar_rows = switchaway_rows(corrections)
    oracle_predictions, summary, details = apply_oracle_filter(
        predictions=predictions,
        switch_rows=sidecar_rows,
        catalog=catalog,
        k_filter=args.k_filter,
    )
    summary.update({
        "prediction_path": str(args.prediction_path),
        "corrections": str(args.corrections),
        "output_prediction": str(args.output_prediction),
    })

    args.output_prediction.parent.mkdir(parents=True, exist_ok=True)
    args.output_prediction.write_text(json.dumps(oracle_predictions, indent=2), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_rows:
        write_jsonl(args.output_rows, details)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
