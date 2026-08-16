"""Build the changed-rows packet behind the strict-review and adjudication validity checks.

Rebuilds the row-level packet that the paper's row validations run over: one
record per turn the request-aware postrank intervention changed, carrying the
visible user text, the requested titles, the promoted (request-resolved)
tracks, and the official-label track. The original v0 packets were produced by
an uncommitted script; this builder derives the identical rows from the frozen
coverage artifact plus catalog metadata.

Input rows are the `promoted` category of an compute_coverage
JSON (the turns whose ranking the postranker changed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from detect_requests import build_catalog_index, metadata_summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-json", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary-output", type=Path, required=True)
    args = ap.parse_args(argv)

    coverage = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    catalog = build_catalog_index(
        load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    )

    rows_out: list[dict[str, Any]] = []
    n_gold_differs = 0
    for row in coverage["rows"]:
        if row.get("category") != "promoted":
            continue
        gold_is_requested = row.get("gold_status") == "gold_requested"
        if not gold_is_requested:
            n_gold_differs += 1
        rows_out.append(
            {
                "split": args.split,
                "session_id": row["session_id"],
                "turn_number": int(row["turn_number"]),
                "user_text": row.get("user_text"),
                "requested_titles": row.get("requested_titles") or [],
                "gold_is_requested": gold_is_requested,
                "gold_track": metadata_summary(catalog.by_id.get(row.get("gold_track_id"))),
                "promoted_tracks": [
                    metadata_summary(catalog.by_id.get(tid))
                    for tid in (row.get("promoted_track_ids") or [])
                ],
                "rank_before": row.get("best_rank"),
            }
        )

    rows_out.sort(key=lambda r: (r["session_id"], r["turn_number"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "coverage_json": str(args.coverage_json),
        "split": args.split,
        "n_changed_rows": len(rows_out),
        "n_gold_differs": n_gold_differs,
        "n_gold_requested": len(rows_out) - n_gold_differs,
        "out": str(args.output),
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
