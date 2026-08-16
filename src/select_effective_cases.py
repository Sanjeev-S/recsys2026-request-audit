"""Filter request-correction JSONL sidecars by family.

This keeps family-specific paper readouts reproducible without changing the
frozen detector output.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, Any]], *, source: Path, output: Path, families: set[str]) -> dict[str, Any]:
    by_family: dict[str, int] = defaultdict(int)
    by_action: dict[str, int] = defaultdict(int)
    unique_turns = set()
    versions = set()
    for row in rows:
        by_family[row["family"]] += 1
        by_action[row.get("action", "")] += 1
        unique_turns.add((row["session_id"], int(row["turn_number"])))
        versions.add(row.get("detector_version"))
    return {
        "source": str(source),
        "out": str(output),
        "included_families": sorted(families),
        "detector_versions": sorted(v for v in versions if v),
        "n_records": len(rows),
        "n_unique_turns": len(unique_turns),
        "by_family": dict(sorted(by_family.items())),
        "by_action": dict(sorted(by_action.items())),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--family", action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary-output", type=Path)
    args = ap.parse_args(argv)

    families = set(args.family)
    rows = [row for row in load_rows(args.input) if row.get("family") in families]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = summarize(rows, source=args.input, output=args.output, families=families)
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
