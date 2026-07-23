"""Build robustness tables for exact/version request evidence.

This is a compact readout over existing frozen coverage and sidecar artifacts.
It does not rerun retrieval or training.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
DEFAULT_JSON = EVID / "funnel/request_exact_robustness_table_repro.json"
DEFAULT_MD = EVID / "funnel/request_exact_robustness_table_repro.md"

# Artifact defaults point at the shipped coverage artifacts and ID-keyed
# sidecars; override with --coverage-<split> / --sidecar-<split> to run over
# regenerated files (e.g. from run_funnel.py).
COVERAGE_PATHS = {
    "devset": EVID / "funnel/request_exact_postrank_coverage_v09_devset.json",
    "val": EVID / "funnel/request_exact_postrank_coverage_v09_val.json",
}
SIDECAR_PATHS = {
    "devset": EVID / "corrections/request_corrections_devset_exact_version_v09.idkeyed.jsonl",
    "val": EVID / "corrections/request_corrections_val_exact_version_v09.idkeyed.jsonl",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(count: int, n: int) -> str:
    return f"{count}/{n} ({count / n:.1%})" if n else "0/0 (0.0%)"


def turn_bucket(turn: int) -> str:
    if turn <= 1:
        return "turn 1"
    if turn <= 4:
        return "turns 2-4"
    return "turns 5-8"


def category_count(rows: list[dict[str, Any]], category: str) -> int:
    return sum(1 for row in rows if row.get("category") == category)


def split_report(split: str, coverage: dict[str, Any], sidecar_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = list(coverage["rows"])
    n = len(rows)
    sidecar_family = Counter(row["family"] for row in sidecar_rows)
    sidecar_turns = {(row["session_id"], int(row["turn_number"])) for row in sidecar_rows}

    turn_rows = []
    for bucket in ["turn 1", "turns 2-4", "turns 5-8"]:
        bucket_rows = [row for row in rows if turn_bucket(int(row["turn_number"])) == bucket]
        b_n = len(bucket_rows)
        turn_rows.append({
            "turn_bucket": bucket,
            "n": b_n,
            "gold_differs": sum(1 for row in bucket_rows if row.get("gold_status") == "gold_differs"),
            "already_front": category_count(bucket_rows, "already_front"),
            "promoted": category_count(bucket_rows, "promoted"),
            "not_retrieved_top100": category_count(bucket_rows, "not_retrieved_topk"),
        })

    return {
        "split": split,
        "n_directives": n,
        "sidecar_records": len(sidecar_rows),
        "sidecar_unique_turns": len(sidecar_turns),
        "family_counts": dict(sorted(sidecar_family.items())),
        "gold_differs": coverage["gold_status_counts"].get("gold_differs", 0),
        "gold_requested": coverage["gold_status_counts"].get("gold_requested", 0),
        "requested_present_top100": n - coverage["category_counts"].get("not_retrieved_topk", 0),
        "already_front": coverage["category_counts"].get("already_front", 0),
        "promoted": coverage["category_counts"].get("promoted", 0),
        "not_retrieved_top100": coverage["category_counts"].get("not_retrieved_topk", 0),
        "rank_bucket_counts": coverage.get("rank_bucket_counts", {}),
        "turn_rows": turn_rows,
    }


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_table() -> dict[str, Any]:
    splits = []
    for split in ["devset", "val"]:
        splits.append(split_report(split, load_json(COVERAGE_PATHS[split]), load_jsonl(SIDECAR_PATHS[split])))
    return {
        "artifact": "request-exact-robustness-table-v0.9",
        "sources": {
            split: {
                "coverage": _display(COVERAGE_PATHS[split]),
                "sidecar": _display(SIDECAR_PATHS[split]),
            }
            for split in ["devset", "val"]
        },
        "splits": splits,
        "read": (
            "Exact/version requests are small but not a single cherry-picked row: "
            "they appear across turns, include reachable and missing-top100 cases, "
            "and version/duplicate equivalents are explicitly rare."
        ),
    }


def family_cell(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return "; ".join(f"{name}: {count}" for name, count in counts.items())


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Exact/Version Robustness Table v0.9",
        "",
        "Purpose: address the reviewer concern that exact/version evidence is a tiny or cherry-picked slice.",
        "",
        f"Read: {report['read']}",
        "",
        "## Split Coverage",
        "",
        "| split | visible exact directives | policy/request conflicts | sidecar family counts | requested present top-100 | reachable promotion slice | absent top-100 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in report["splits"]:
        lines.append(
            "| "
            f"{row['split']} | "
            f"{row['n_directives']} | "
            f"{pct(row['gold_differs'], row['n_directives'])} | "
            f"{family_cell(row['family_counts'])} | "
            f"{pct(row['requested_present_top100'], row['n_directives'])} | "
            f"{pct(row['promoted'], row['n_directives'])} | "
            f"{pct(row['not_retrieved_top100'], row['n_directives'])} |"
        )

    lines.extend([
        "",
        "## Turn Position",
        "",
        "| split | turn bucket | directives | policy/request conflicts | already first | promoted | absent top-100 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for split in report["splits"]:
        for row in split["turn_rows"]:
            lines.append(
                "| "
                f"{split['split']} | "
                f"{row['turn_bucket']} | "
                f"{row['n']} | "
                f"{pct(row['gold_differs'], row['n'])} | "
                f"{pct(row['already_front'], row['n'])} | "
                f"{pct(row['promoted'], row['n'])} | "
                f"{pct(row['not_retrieved_top100'], row['n'])} |"
            )

    lines.extend([
        "",
        "Interpretation: the exact/version slice is intentionally narrow, but the evidence is not a single-turn artifact. The main failure mode is retrieval availability: when a request-satisfying item is absent from top-100, neither postranking nor a reranker can promote it from the submitted list.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--md-output", type=Path, default=DEFAULT_MD)
    ap.add_argument("--coverage-devset", type=Path)
    ap.add_argument("--coverage-val", type=Path)
    ap.add_argument("--sidecar-devset", type=Path)
    ap.add_argument("--sidecar-val", type=Path)
    args = ap.parse_args(argv)

    if args.coverage_devset:
        COVERAGE_PATHS["devset"] = args.coverage_devset
    if args.coverage_val:
        COVERAGE_PATHS["val"] = args.coverage_val
    if args.sidecar_devset:
        SIDECAR_PATHS["devset"] = args.sidecar_devset
    if args.sidecar_val:
        SIDECAR_PATHS["val"] = args.sidecar_val

    report = build_table()
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({
        "json": str(args.json_output),
        "markdown": str(args.md_output),
        "splits": {row["split"]: row["n_directives"] for row in report["splits"]},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
