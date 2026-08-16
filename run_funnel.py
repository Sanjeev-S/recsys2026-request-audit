#!/usr/bin/env python3
"""One-command reproduction of the paper's audit funnel (Section 5).

Runs the frozen detector chain over a split and prints the funnel:

    detector -> exact/version filter -> availability coverage -> changed-rows packet

For the development set with the deployed production ranking this reproduces
the paper's Section 5 funnel — 82 visible directives, 41/82 label-request
conflicts, 43 effective correction records — plus two shipped-but-uncited
readouts: the availability window (65/82 requested tracks in the deployed
top-100) and the changed-rows packet (19 rows, 12 with a conflicting label).

Requires: the challenge data reachable via HuggingFace datasets (see README
data-access notes) and a prediction file for the availability window
(--prediction-path; JSON list of {session_id, turn_number, predicted_track_ids}).

Outputs land in --work-dir (default: work/funnel_<split>/). Dialogue-bearing
intermediates stay local; use src/redact_sidecar.py before sharing any sidecar.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SRC / script), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="devset", choices=["devset", "val"])
    ap.add_argument("--prediction-path", required=True,
                    help="ranking used as the availability window (production reranker output)")
    ap.add_argument("--work-dir", type=Path, default=None)
    args = ap.parse_args()

    work = args.work_dir or ROOT / f"work/funnel_{args.split}"
    work.mkdir(parents=True, exist_ok=True)

    corrections_all = work / f"request_corrections_{args.split}_all.jsonl"
    corrections_ev = work / f"request_corrections_{args.split}_exact_version.jsonl"
    coverage_json = work / f"request_exact_postrank_coverage_{args.split}.json"
    coverage_md = work / f"request_exact_postrank_coverage_{args.split}.md"
    changed_rows = work / f"request_changed_rows_{args.split}.jsonl"

    run("request_correction_labels.py", "--split", args.split,
        "--out", str(corrections_all),
        "--summary-out", str(corrections_all.with_suffix(".summary.json")))
    run("filter_request_corrections.py",
        "--input", str(corrections_all),
        "--family", "exact_track_request", "--family", "version_duplicate_equivalence",
        "--output", str(corrections_ev),
        "--summary-output", str(corrections_ev.with_suffix(".summary.json")))
    run("analyze_request_postrank_coverage.py",
        "--prediction-path", args.prediction_path, "--split", args.split,
        "--json-output", str(coverage_json), "--markdown-output", str(coverage_md))
    run("build_request_changed_rows_packet.py",
        "--coverage-json", str(coverage_json), "--split", args.split,
        "--output", str(changed_rows),
        "--summary-output", str(changed_rows.with_suffix(".summary.json")))

    cov = json.loads(coverage_json.read_text())
    ev = [json.loads(l) for l in corrections_ev.open() if l.strip()]
    packet = [json.loads(l) for l in changed_rows.open() if l.strip()]
    funnel = {
        "split": args.split,
        "visible_directives": cov["n_directives"],
        "label_request_conflicts": cov["gold_status_counts"].get("gold_differs", 0),
        "requested_present_top100": cov["n_directives"] - cov["category_counts"].get("not_retrieved_topk", 0),
        "effective_correction_records": len(ev),
        "postrank_changed_rows": len(packet),
        "changed_rows_gold_differs": sum(1 for r in packet if not r.get("gold_is_requested")),
    }
    print(json.dumps(funnel, indent=2))
    if args.split == "devset":
        expected = {"visible_directives": 82, "label_request_conflicts": 41,
                    "effective_correction_records": 43}
        mismatch = {k: (funnel[k], v) for k, v in expected.items() if funnel[k] != v}
        print("PAPER-MATCH: OK" if not mismatch else f"PAPER-MATCH: MISMATCH {mismatch}")


if __name__ == "__main__":
    main()
