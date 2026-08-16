"""Build a cross-dialogue request-positive control sidecar.

This control is stricter than the arbitrary wrong-positive control: each
substitute positive must be a request-positive track in another exact/version
training group, but must not satisfy the current dialogue's exact request by ID.
It tests whether the learned behavior depends on dialogue-aligned request
positives, rather than the marginal distribution of request-positive items.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from shuffled_positive_control import (  # noqa: E402
    _unique_preserve_order,
    candidates_by_key,
    exact_requested_by_key,
)


def _stable_rank(seed: int, key: tuple[str, int], track_id: str, sources: list[tuple[str, int]]) -> str:
    source_text = ",".join(f"{sid}:{turn}" for sid, turn in sorted(sources)[:5])
    raw = f"{seed}\t{key[0]}\t{key[1]}\t{track_id}\t{source_text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def request_positive_sources(
    requested: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, list[tuple[str, int]]]:
    sources: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key, info in requested.items():
        for tid in info["requested_track_ids"]:
            sources[tid].append(key)
    return {tid: sorted(keys) for tid, keys in sources.items()}


def build_cross_dialogue_sidecar(
    *,
    feature_path: Path,
    corrections_path: Path,
    output_path: Path,
    summary_path: Path | None = None,
    seed: int = 23,
) -> dict[str, Any]:
    requested = exact_requested_by_key(corrections_path)
    sources = request_positive_sources(requested)
    candidates = candidates_by_key(feature_path, set(requested))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emitted = 0
    listed_requested_positive_ids = 0
    train_effective_requested_positive_ids = 0
    control_positive_ids = 0
    groups_with_cross_dialogue_candidates = 0
    missing_groups: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as out:
        for key in sorted(requested):
            info = requested[key]
            requested_ids = set(info["requested_track_ids"])
            rows = candidates.get(key, [])
            unique_candidates = _unique_preserve_order([tid for tid, _ in rows])
            labels = {tid: int(label) for tid, label in rows}
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
                    "reason": "no_train_effective_request_positive",
                })
                continue

            eligible: list[tuple[str, list[tuple[str, int]]]] = []
            for tid in unique_candidates:
                if labels.get(tid, 0) > 0:
                    continue
                if tid in requested_ids or tid == info.get("gold_track_id"):
                    continue
                other_sources = [source for source in sources.get(tid, []) if source != key]
                if other_sources:
                    eligible.append((tid, other_sources))
            if eligible:
                groups_with_cross_dialogue_candidates += 1
            if len(eligible) < n_needed:
                missing_groups.append({
                    "session_id": key[0],
                    "turn_number": key[1],
                    "needed": n_needed,
                    "eligible": len(eligible),
                    "reason": "insufficient_cross_dialogue_request_positives",
                })
                continue

            selected_pairs = sorted(
                eligible,
                key=lambda item: _stable_rank(seed, key, item[0], item[1]),
            )[:n_needed]
            selected = [tid for tid, _ in selected_pairs]
            source_map = {
                tid: [
                    {"session_id": source[0], "turn_number": source[1]}
                    for source in sources_for_tid[:5]
                ]
                for tid, sources_for_tid in selected_pairs
            }
            row = dict(info["template"])
            evidence = dict(row.get("evidence") or {})
            evidence["control_type"] = "cross_dialogue_request_positive"
            evidence["original_additional_track_ids"] = sorted(requested_ids)
            evidence["train_effective_original_additional_track_ids"] = effective_requested_ids
            evidence["cross_dialogue_positive_sources"] = source_map
            row["additional_track_ids"] = selected
            row["detector_version"] = f"{row.get('detector_version', 'request-corrections')}-cross-dialogue-control"
            row["evidence"] = evidence
            row["control_type"] = "cross_dialogue_request_positive"
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
        "groups_with_cross_dialogue_candidates": groups_with_cross_dialogue_candidates,
        "listed_requested_positive_ids": listed_requested_positive_ids,
        "requested_positive_ids": train_effective_requested_positive_ids,
        "control_positive_ids": control_positive_ids,
        "unique_global_request_positive_ids": len(sources),
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
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args(argv)

    summary = build_cross_dialogue_sidecar(
        feature_path=args.feature_path,
        corrections_path=args.corrections,
        output_path=args.output,
        summary_path=args.summary,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
