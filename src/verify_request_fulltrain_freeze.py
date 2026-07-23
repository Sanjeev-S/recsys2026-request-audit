"""Verify or write the request full-train freeze manifest.

The full-train request experiments fold repo `val` back into organizer train and
use repo `devset` only as a held-out readout. This helper makes the frozen
choices machine-checkable: exact/version uses a single primary blend
(`lambda=1.0`), hard-drop uses a single primary blend (`lambda=0.5`), both train
for 50 rounds with no val DMatrix, and the local cache counts match the
documented full-train denominators.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from verify_request_freeze import file_block, load_manifest, verify_manifest

DEFAULT_MANIFEST = Path("docs/evidence/request_fulltrain_freeze_manifest_v09.json")

CORE_SCRIPTS = [
    "scripts/request_hard_artist_rerank_train.py",
    "scripts/request_exact_fulltrain_launch.py",
    "scripts/request_hard_fulltrain_launch.py",
    "scripts/run_request_fulltrain_eval.py",
    "scripts/verify_request_fulltrain_eval_runner.py",
    "scripts/request_corrected_rerank_blend_eval.py",
    "scripts/apply_request_exact_postrank.py",
    "scripts/apply_request_hard_constraint_postrank.py",
    "scripts/evaluate_request_corrections.py",
    "scripts/request_correction_labels.py",
    "scripts/analyze_challenge_distribution.py",
    "scripts/verify_request_fulltrain_freeze.py",
]

CORE_ARTIFACTS = [
    "docs/evidence/challenge_distribution_v09.json",
    "docs/evidence/challenge_distribution_v09.md",
    "docs/evidence/request_fulltrain_freeze_protocol_v09.md",
    "docs/evidence/request_exact_version_fulltrain_protocol_v09.md",
    "docs/evidence/request_hard_simple_nonneg_fulltrain_violation_drop_protocol_v09.md",
    "docs/evidence/request_exact_version_fulltrain_label_cache_local_v09.json",
    "docs/evidence/request_hard_simple_nonneg_fulltrain_violation_drop_label_cache_local_v09.json",
    "docs/evidence/request_exact_version_fulltrain_train_a_official_label_cache_v09.summary.json",
    "docs/evidence/request_exact_version_fulltrain_train_a_exact_positive_weighted_label_cache_v09.summary.json",
    "docs/evidence/request_exact_version_fulltrain_val_official_label_cache_v09.summary.json",
    "docs/evidence/request_exact_version_fulltrain_val_exact_positive_weighted_label_cache_v09.summary.json",
    "docs/evidence/request_exact_version_fulltrain_train_a_exact_request_feature_cache_v09.summary.json",
    "docs/evidence/request_exact_version_fulltrain_val_exact_request_feature_cache_v09.summary.json",
    "docs/evidence/request_hard_simple_nonneg_full105k_official_label_cache_v09.summary.json",
    "docs/evidence/request_hard_simple_nonneg_full105k_violation_drop_label_cache_v09.summary.json",
    "docs/evidence/request_hard_simple_nonneg_full105k_hard_feature_cache_v09.summary.json",
    "docs/evidence/request_hard_simple_nonneg_fulltrain_val_official_label_cache_v09.summary.json",
    "docs/evidence/request_hard_simple_nonneg_fulltrain_val_violation_drop_label_cache_v09.summary.json",
    "docs/evidence/request_hard_simple_nonneg_fulltrain_val_hard_feature_cache_v09.summary.json",
    "docs/evidence/request_corrections_train_lt_exact_version_v09.jsonl",
    "docs/evidence/request_corrections_train_lt_exact_version_v09.summary.json",
    "docs/evidence/request_corrections_val_exact_version_v09.jsonl",
    "docs/evidence/request_corrections_val_exact_version_v09.summary.json",
    "docs/evidence/request_corrections_devset_exact_version_v09.jsonl",
    "docs/evidence/request_corrections_devset_exact_version_v09.summary.json",
    "docs/evidence/request_corrections_train_lt_v09_simple_nonneg_hard_artist_only.jsonl",
    "docs/evidence/request_corrections_train_lt_v09_simple_nonneg_hard_artist_only.summary.json",
    "docs/evidence/request_corrections_val_v09_simple_nonneg_hard_artist_only.jsonl",
    "docs/evidence/request_corrections_val_v09_simple_nonneg_hard_artist_only.summary.json",
    "docs/evidence/request_corrections_devset_v09_simple_nonneg_hard_artist_only.jsonl",
    "docs/evidence/request_corrections_devset_v09_simple_nonneg_hard_artist_only.summary.json",
]

EXPECTED = {
    "train_n_groups": 121_592,
    "train_n_rows": 203_221_270,
    "num_boost_round": 50,
    "exact": {
        "primary_blend_weight": 1.0,
        "corrected_group_weight": 0.1,
        "added_positive_rows": 1_878,
        "downweighted_groups": 1_873,
        "feature_matched_groups": 2_186,
        "feature_matched_rows": 2_207,
    },
    "hard": {
        "primary_blend_weight": 0.5,
        "dropped_groups": 361,
        "feature_matched_groups": 760,
        "feature_matched_rows": 34_153,
    },
}


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def blend_weights(path: str) -> list[float]:
    text = Path(path).read_text(encoding="utf-8")
    return [float(value) for value in re.findall(r"--blend-weight\s+([0-9.]+)", text)]


def require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def sum_fields(paths: list[str], field: str) -> int:
    return sum(int(load_json(path)[field]) for path in paths)


def verify_invariants(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expected = manifest["expected"]

    exact = load_json("docs/evidence/request_exact_version_fulltrain_label_cache_local_v09.json")
    hard = load_json("docs/evidence/request_hard_simple_nonneg_fulltrain_violation_drop_label_cache_local_v09.json")
    for name, payload in [("exact", exact), ("hard", hard)]:
        require(payload["train_n_groups"] == expected["train_n_groups"], failures, f"{name}: train_n_groups mismatch")
        require(payload["train_n_rows"] == expected["train_n_rows"], failures, f"{name}: train_n_rows mismatch")
        require(payload["num_boost_round"] == expected["num_boost_round"], failures, f"{name}: num_boost_round mismatch")
        require(payload["skip_val_dmatrix"] is True, failures, f"{name}: skip_val_dmatrix must be true")

    require(exact["add_exact_request_feature"] is True, failures, "exact: exact_request_match feature must be enabled")
    require(set(exact["cells"]) == {"official", "exact_positive_weighted"}, failures, "exact: unexpected cells")
    require(hard["add_hard_artist_feature"] is True, failures, "hard: hard_artist feature must be enabled")
    require(set(hard["cells"]) == {"official", "violation_drop"}, failures, "hard: unexpected cells")
    require(
        hard["hard_artist_feature"] == {"strict": True, "exclude_exact_requests": True, "simple_artist_only": True},
        failures,
        "hard: hard feature selector mismatch",
    )

    exact_added = sum_fields([
        "docs/evidence/request_exact_version_fulltrain_train_a_exact_positive_weighted_label_cache_v09.summary.json",
        "docs/evidence/request_exact_version_fulltrain_val_exact_positive_weighted_label_cache_v09.summary.json",
    ], "added_positive_rows")
    exact_downweighted = sum_fields([
        "docs/evidence/request_exact_version_fulltrain_train_a_exact_positive_weighted_label_cache_v09.summary.json",
        "docs/evidence/request_exact_version_fulltrain_val_exact_positive_weighted_label_cache_v09.summary.json",
    ], "downweighted_groups")
    exact_matched_groups = sum_fields([
        "docs/evidence/request_exact_version_fulltrain_train_a_exact_request_feature_cache_v09.summary.json",
        "docs/evidence/request_exact_version_fulltrain_val_exact_request_feature_cache_v09.summary.json",
    ], "matched_groups")
    exact_matched_rows = sum_fields([
        "docs/evidence/request_exact_version_fulltrain_train_a_exact_request_feature_cache_v09.summary.json",
        "docs/evidence/request_exact_version_fulltrain_val_exact_request_feature_cache_v09.summary.json",
    ], "matched_rows")
    require(exact_added == expected["exact"]["added_positive_rows"], failures, "exact: added positives mismatch")
    require(exact_downweighted == expected["exact"]["downweighted_groups"], failures, "exact: downweighted groups mismatch")
    require(exact_matched_groups == expected["exact"]["feature_matched_groups"], failures, "exact: feature matched groups mismatch")
    require(exact_matched_rows == expected["exact"]["feature_matched_rows"], failures, "exact: feature matched rows mismatch")

    hard_dropped = sum_fields([
        "docs/evidence/request_hard_simple_nonneg_full105k_violation_drop_label_cache_v09.summary.json",
        "docs/evidence/request_hard_simple_nonneg_fulltrain_val_violation_drop_label_cache_v09.summary.json",
    ], "dropped_groups")
    hard_matched_groups = sum_fields([
        "docs/evidence/request_hard_simple_nonneg_full105k_hard_feature_cache_v09.summary.json",
        "docs/evidence/request_hard_simple_nonneg_fulltrain_val_hard_feature_cache_v09.summary.json",
    ], "matched_groups")
    hard_matched_rows = sum_fields([
        "docs/evidence/request_hard_simple_nonneg_full105k_hard_feature_cache_v09.summary.json",
        "docs/evidence/request_hard_simple_nonneg_fulltrain_val_hard_feature_cache_v09.summary.json",
    ], "matched_rows")
    require(hard_dropped == expected["hard"]["dropped_groups"], failures, "hard: dropped groups mismatch")
    require(hard_matched_groups == expected["hard"]["feature_matched_groups"], failures, "hard: feature matched groups mismatch")
    require(hard_matched_rows == expected["hard"]["feature_matched_rows"], failures, "hard: feature matched rows mismatch")

    exact_weights = blend_weights("docs/evidence/request_exact_version_fulltrain_protocol_v09.md")
    hard_weights = blend_weights("docs/evidence/request_hard_simple_nonneg_fulltrain_violation_drop_protocol_v09.md")
    require(exact_weights == [expected["exact"]["primary_blend_weight"]] * 2, failures, "exact: protocol blend weights not frozen")
    require(hard_weights == [expected["hard"]["primary_blend_weight"]] * 2, failures, "hard: protocol blend weights not frozen")

    hard_protocol = Path("docs/evidence/request_hard_simple_nonneg_fulltrain_violation_drop_protocol_v09.md").read_text(encoding="utf-8")
    require("request_hard_105k_launch.py" not in hard_protocol, failures, "hard: stale 105k launcher reference")

    for path in ["scripts/request_exact_fulltrain_launch.py", "scripts/request_hard_fulltrain_launch.py"]:
        text = Path(path).read_text(encoding="utf-8")
        require("xgboost==3.2.0" in text and "pyarrow==24.0.0" in text, failures, f"{path}: pod deps are not pinned")
        require(" --blend-weight " not in text, failures, f"{path}: launcher should not evaluate/tune blend weights")

    return failures


def build_manifest() -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "protocol": "request-fulltrain-v0.9-frozen-readout",
        "frozen_date": "2026-06-23",
        "status": "ready_for_approved_pod_training",
        "official_noninferiority_margin": 0.001,
        "training_sources": ["train_lt_105k", "val"],
        "heldout_readout": "devset",
        "expected": EXPECTED,
        "scripts": file_block(CORE_SCRIPTS),
        "artifacts": file_block(CORE_ARTIFACTS),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--write", action="store_true", help="Write the current manifest instead of verifying it.")
    args = ap.parse_args(argv)

    if args.write:
        manifest = build_manifest()
        invariant_failures = verify_invariants(manifest)
        if invariant_failures:
            print(json.dumps({"ok": False, "invariant_failures": invariant_failures}, indent=2, sort_keys=True))
            raise SystemExit(1)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "written": str(args.manifest),
            "n_scripts": len(manifest["scripts"]),
            "n_artifacts": len(manifest["artifacts"]),
        }, indent=2, sort_keys=True))
        return

    manifest = load_manifest(args.manifest)
    result = verify_manifest(manifest)
    invariant_failures = verify_invariants(manifest)
    result["invariant_failures"] = invariant_failures
    result["ok"] = bool(result["ok"] and not invariant_failures)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
