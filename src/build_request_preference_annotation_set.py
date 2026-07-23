"""Build a blinded policy-label vs request-target preference set.

The corrected metric assumes that, when a visible exact request conflicts with
the policy-selected label, the catalog-resolved requested item is the better
request-satisfying target. This exporter creates a blinded pairwise annotation
set to test that assumption.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def option(track: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_name": track.get("track_name"),
        "artist_name": track.get("artist_name"),
        "album_name": track.get("album_name"),
        "release_date": track.get("release_date"),
    }


def build_rows(
    changed_rows: list[dict[str, Any]],
    *,
    seed: int,
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    conflicts = [
        row for row in changed_rows
        if not row.get("gold_is_requested")
        and row.get("gold_track")
        and row.get("promoted_tracks")
    ]
    conflicts.sort(key=lambda r: (r.get("split", ""), r["session_id"], int(r["turn_number"])))
    rng.shuffle(conflicts)
    if max_rows is not None:
        conflicts = conflicts[:max_rows]

    annotation_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    side_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for idx, row in enumerate(conflicts, start=1):
        sample_id = f"request-pref-{idx:04d}"
        request_track = row["promoted_tracks"][0]
        policy_track = row["gold_track"]
        request_side = rng.choice(["A", "B"])
        policy_side = "B" if request_side == "A" else "A"
        side_counts[request_side] += 1
        split_counts[row.get("split", "unknown")] += 1
        candidates = {
            request_side: option(request_track),
            policy_side: option(policy_track),
        }
        annotation_rows.append({
            "sample_id": sample_id,
            "split": row.get("split"),
            "session_id": row["session_id"],
            "turn_number": int(row["turn_number"]),
            "user_text": row.get("user_text"),
            "requested_titles": row.get("requested_titles") or [],
            "candidate_A": candidates["A"],
            "candidate_B": candidates["B"],
            "annotation_schema": {
                "preferred_item": "A/B/tie/unclear",
                "request_satisfying_item": "A/B/both/neither/unclear",
                "explicit_request_visible": "yes/no/unclear",
                "confidence": "high/medium/low",
                "notes": "brief rationale",
            },
        })
        key_rows.append({
            "sample_id": sample_id,
            "split": row.get("split"),
            "session_id": row["session_id"],
            "turn_number": int(row["turn_number"]),
            "request_side": request_side,
            "policy_side": policy_side,
            "request_track_id": request_track.get("track_id"),
            "policy_track_id": policy_track.get("track_id"),
            "request_track": request_track,
            "policy_track": policy_track,
            "requested_titles": row.get("requested_titles") or [],
        })

    summary = {
        "seed": seed,
        "n_input_rows": len(changed_rows),
        "n_conflict_rows_available": len([
            row for row in changed_rows
            if not row.get("gold_is_requested")
            and row.get("gold_track")
            and row.get("promoted_tracks")
        ]),
        "n_annotation_rows": len(annotation_rows),
        "by_split": dict(sorted(split_counts.items())),
        "request_side_counts": dict(sorted(side_counts.items())),
    }
    return annotation_rows, key_rows, summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--changed-rows", type=Path, action="append", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--annotation-output", type=Path, required=True)
    ap.add_argument("--key-output", type=Path, required=True)
    ap.add_argument("--summary-output", type=Path, required=True)
    args = ap.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for path in args.changed_rows:
        rows.extend(load_jsonl(path))

    annotation_rows, key_rows, summary = build_rows(rows, seed=args.seed, max_rows=args.max_rows)
    summary["changed_rows"] = [str(path) for path in args.changed_rows]
    summary["annotation_output"] = str(args.annotation_output)
    summary["key_output"] = str(args.key_output)

    write_jsonl(args.annotation_output, annotation_rows)
    write_jsonl(args.key_output, key_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
