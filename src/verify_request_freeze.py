"""Verify or write the request-correction freeze manifest.

The paper-facing v0.9 claim depends on a small exact/version stack and a larger
set of generated evidence artifacts. This helper records SHA-256 hashes for
those files and verifies them later, so "frozen protocol" is machine-checkable
rather than a hand-maintained table in a markdown file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("docs/evidence/request_correction_v09_freeze_manifest.json")

CORE_SCRIPTS = [
    "scripts/request_correction_labels.py",
    "scripts/filter_request_corrections.py",
    "scripts/apply_request_exact_postrank.py",
    "scripts/analyze_request_postrank_coverage.py",
    "scripts/evaluate_request_corrections.py",
    "scripts/bootstrap_request_corrected_dev.py",
    "scripts/build_request_evidence_table.py",
    "scripts/score_request_frame_ablation.py",
    "scripts/verify_request_freeze.py",
]

CORE_ARTIFACTS = [
    "docs/evidence/request_corrections_devset_v09.jsonl",
    "docs/evidence/request_corrections_devset_v09.summary.json",
    "docs/evidence/request_corrections_devset_exact_version_v09.jsonl",
    "docs/evidence/request_corrections_devset_exact_version_v09.summary.json",
    "docs/evidence/request_corrections_val_v09.jsonl",
    "docs/evidence/request_corrections_val_v09.summary.json",
    "docs/evidence/request_corrections_val_exact_version_v09.jsonl",
    "docs/evidence/request_corrections_val_exact_version_v09.summary.json",
    "docs/evidence/request_exact_postrank_locked_table_v09.json",
    "docs/evidence/request_exact_postrank_locked_table_v09.md",
    "docs/evidence/request_exact_postrank_coverage_v09_devset.json",
    "docs/evidence/request_exact_postrank_coverage_v09_devset.md",
    "docs/evidence/request_exact_postrank_coverage_v09_val.json",
    "docs/evidence/request_exact_postrank_coverage_v09_val.md",
    "docs/evidence/request_exact_postrank_strict_v09_rerank_F10_scaled_devset.json",
    "docs/evidence/request_exact_postrank_strict_v09_rerank_F10_scaled_devset.summary.json",
    "docs/evidence/request_exact_postrank_strict_v09_rerank_F10_scaled_exact_version_eval.json",
    "docs/evidence/request_exact_postrank_strict_v09_bootstrap_vs_rerank_F10_scaled_exact_version_v0.json",
    "docs/evidence/request_exact_postrank_strict_v09_rerank_F10_R54SRC_val.json",
    "docs/evidence/request_exact_postrank_strict_v09_rerank_F10_R54SRC_val.summary.json",
    "docs/evidence/request_exact_postrank_strict_v09_rerank_F10_R54SRC_val_exact_version_eval.json",
    "docs/evidence/request_exact_postrank_strict_v09_bootstrap_vs_rerank_F10_R54SRC_val_exact_version_v0.json",
    "docs/evidence/request_exact_postrank_strict_v09_changed_rows_devset_v0.summary.json",
    "docs/evidence/request_exact_postrank_strict_v09_changed_rows_val_v0.summary.json",
    "docs/evidence/request_exact_postrank_strict_v09_changed_rows_labels_v0.summary.json",
    "docs/evidence/request_preference_v09_score.json",
    "docs/evidence/request_preference_v09_score.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_block(paths: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        out[raw] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return out


def build_manifest() -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "protocol": "request-correction-v0.9",
        "frozen_date": "2026-06-21",
        "detector_version": "request-corrections-v0.9",
        "official_noninferiority_margin": 0.001,
        "core_claim_family": ["exact_track_request", "version_duplicate_equivalence"],
        "diagnostic_families": [
            "hard_artist_constraint",
            "hard_album_constraint",
            "rejection_switchaway_violation",
        ],
        "scripts": file_block(CORE_SCRIPTS),
        "artifacts": file_block(CORE_ARTIFACTS),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_block(block: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for raw, expected in sorted(block.items()):
        path = Path(raw)
        if not path.exists():
            mismatches.append({"path": raw, "error": "missing"})
            continue
        actual_hash = sha256(path)
        actual_size = path.stat().st_size
        if actual_hash != expected.get("sha256") or actual_size != expected.get("bytes"):
            mismatches.append({
                "path": raw,
                "expected_sha256": expected.get("sha256"),
                "actual_sha256": actual_hash,
                "expected_bytes": expected.get("bytes"),
                "actual_bytes": actual_size,
            })
    return mismatches


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    script_mismatches = verify_block(manifest.get("scripts", {}))
    artifact_mismatches = verify_block(manifest.get("artifacts", {}))
    ok = not script_mismatches and not artifact_mismatches
    return {
        "ok": ok,
        "protocol": manifest.get("protocol"),
        "n_scripts": len(manifest.get("scripts", {})),
        "n_artifacts": len(manifest.get("artifacts", {})),
        "script_mismatches": script_mismatches,
        "artifact_mismatches": artifact_mismatches,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--write", action="store_true", help="Write the current manifest instead of verifying it.")
    args = ap.parse_args(argv)

    if args.write:
        manifest = build_manifest()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "written": str(args.manifest),
            "n_scripts": len(manifest["scripts"]),
            "n_artifacts": len(manifest["artifacts"]),
        }, indent=2, sort_keys=True))
        return

    result = verify_manifest(load_manifest(args.manifest))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
