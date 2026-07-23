"""Build an in-pool wrong-positive control sidecar for exact/version training.

The control keeps the same correction groups as an exact/version sidecar, but
replaces each request-satisfying additional positive with a deterministic
non-request candidate from the same reranker pool group. This tests whether the
training result comes from the specific request-positive targets, rather than
from adding arbitrary extra positives on a conflict-selected slice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from evaluate_request_corrections import load_corrections  # noqa: E402
from rerank_train import BATCH  # noqa: E402

EXACT_FAMILIES = {"exact_track_request", "version_duplicate_equivalence"}


def _stable_rank(seed: int, key: tuple[str, int], track_id: str) -> str:
    raw = f"{seed}\t{key[0]}\t{key[1]}\t{track_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _unique_preserve_order(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def exact_requested_by_key(corrections_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    grouped = load_corrections(corrections_path)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        exact_rows = [row for row in rows if row.get("family") in EXACT_FAMILIES]
        requested = _unique_preserve_order([
            tid
            for row in exact_rows
            for tid in (row.get("additional_track_ids") or [])
        ])
        if not requested:
            continue
        template = dict(exact_rows[0])
        out[key] = {
            "template": template,
            "requested_track_ids": requested,
            "gold_track_id": template.get("gold_track_id"),
        }
    return out


def candidates_by_key(feature_path: Path, keys: set[tuple[str, int]]) -> dict[tuple[str, int], list[tuple[str, int]]]:
    candidates: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    pf = pq.ParquetFile(feature_path)
    for batch in pf.iter_batches(batch_size=BATCH, columns=["session_id", "turn_number", "track_id", "label"]):
        sids = batch.column("session_id").to_pylist()
        turns = batch.column("turn_number").to_pylist()
        tids = batch.column("track_id").to_pylist()
        labels = batch.column("label").to_pylist()
        for sid, turn, tid, label in zip(sids, turns, tids, labels):
            key = (sid, int(turn))
            if key in keys:
                candidates[key].append((tid, int(label)))
    return dict(candidates)


def build_shuffled_sidecar(
    *,
    feature_path: Path,
    corrections_path: Path,
    output_path: Path,
    summary_path: Path | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    requested = exact_requested_by_key(corrections_path)
    candidates = candidates_by_key(feature_path, set(requested))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emitted = 0
    listed_requested_positive_ids = 0
    train_effective_requested_positive_ids = 0
    control_positive_ids = 0
    missing_groups: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as out:
        for key in sorted(requested):
            info = requested[key]
            requested_ids = set(info["requested_track_ids"])
            rows = candidates.get(key, [])
            unique_candidates = _unique_preserve_order([tid for tid, _ in rows])
            labels = {tid: label for tid, label in rows}
            effective_requested_ids = [
                tid for tid in sorted(requested_ids) if tid in labels and labels.get(tid, 0) <= 0
            ]
            n_needed = len(effective_requested_ids)
            listed_requested_positive_ids += len(requested_ids)
            if n_needed == 0:
                missing_groups.append({
                    "session_id": key[0],
                    "turn_number": key[1],
                    "needed": n_needed,
                    "eligible": 0,
                })
                continue
            eligible = [
                tid
                for tid in unique_candidates
                if tid not in requested_ids
                and tid != info.get("gold_track_id")
                and labels.get(tid, 0) <= 0
            ]
            if len(eligible) < n_needed:
                eligible = [
                    tid
                    for tid in unique_candidates
                    if tid not in requested_ids and tid != info.get("gold_track_id")
                ]
            if len(eligible) < n_needed:
                missing_groups.append({
                    "session_id": key[0],
                    "turn_number": key[1],
                    "needed": n_needed,
                    "eligible": len(eligible),
                })
                continue

            selected = sorted(eligible, key=lambda tid: _stable_rank(seed, key, tid))[:n_needed]
            row = dict(info["template"])
            evidence = dict(row.get("evidence") or {})
            evidence["control_type"] = "in_pool_wrong_positive"
            evidence["original_additional_track_ids"] = sorted(requested_ids)
            evidence["train_effective_original_additional_track_ids"] = effective_requested_ids
            row["additional_track_ids"] = selected
            row["detector_version"] = f"{row.get('detector_version', 'request-corrections')}-shuffled-control"
            row["evidence"] = evidence
            row["control_type"] = "in_pool_wrong_positive"
            out.write(json.dumps(row, sort_keys=True) + "\n")
            emitted += 1
            train_effective_requested_positive_ids += n_needed
            control_positive_ids += len(selected)

    summary = {
        "feature_path": str(feature_path),
        "corrections_path": str(corrections_path),
        "output_path": str(output_path),
        "seed": seed,
        "requested_groups": len(requested),
        "emitted_groups": emitted,
        "missing_candidate_groups": len(missing_groups),
        "listed_requested_positive_ids": listed_requested_positive_ids,
        "requested_positive_ids": train_effective_requested_positive_ids,
        "control_positive_ids": control_positive_ids,
        "missing_examples": missing_groups[:20],
    }
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-path", type=Path, required=True)
    ap.add_argument("--corrections", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)

    summary = build_shuffled_sidecar(
        feature_path=args.feature_path,
        corrections_path=args.corrections,
        output_path=args.output,
        summary_path=args.summary,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
