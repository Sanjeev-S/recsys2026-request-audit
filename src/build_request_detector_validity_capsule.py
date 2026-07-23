"""Build a detector-validity capsule for the request-aware paper package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "docs/evidence"
DEFAULT_JSON = EVID / "request_detector_validity_capsule_v09.json"
DEFAULT_MD = EVID / "request_detector_validity_capsule_v09.md"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rate_text(rate: dict[str, Any]) -> str:
    if rate["n"] == 0:
        return "0/0 (n/a)"
    return (
        f"{rate['successes']}/{rate['n']} "
        f"({rate['point']:.3f}, CI [{rate['low']:.3f}, {rate['high']:.3f}])"
    )


def count_text(count: int, n: int) -> str:
    return f"{count}/{n} ({(count / n if n else 0.0):.1%})"


def build_capsule(frame: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    strict = frame["by_bucket"]["strict_detected_exact_request"]
    quoted_control = frame["by_bucket"]["quoted_not_strict_detected"]
    unquoted_control = frame["by_bucket"]["unquoted_request_like"]
    n_conflicts = metadata.get("n_conflict_rows") or metadata["counts"]["n"]
    metadata_counts = metadata["counts"]
    return {
        "date": "2026-06-22",
        "capsule": "request-detector-validity-v0.9",
        "paper_question": (
            "Does the exact/version detector identify an active request frame, "
            "and is the resulting conflict more than generic duplicate/version cleanup?"
        ),
        "frame_ablation": {
            "status": "stratified_annotation_check_not_prevalence_estimate",
            "n_rows": frame["n_rows"],
            "min_confidence": frame["min_confidence"],
            "detector_positive_active": frame["detector_positive_active_rate"],
            "control_active": frame["control_active_rate"],
            "strict_detector_title_overlap_among_active": strict["title_overlap_among_active"],
            "buckets": {
                "strict_detected_exact_request": {
                    "n": strict["n"],
                    "active_exact": strict["active_exact"],
                    "title_overlap_among_active": strict["title_overlap_among_active"],
                },
                "quoted_not_strict_detected": {
                    "n": quoted_control["n"],
                    "active_exact": quoted_control["active_exact"],
                    "title_overlap_among_active": quoted_control["title_overlap_among_active"],
                },
                "unquoted_request_like": {
                    "n": unquoted_control["n"],
                    "active_exact": unquoted_control["active_exact"],
                    "title_overlap_among_active": unquoted_control["title_overlap_among_active"],
                },
            },
        },
        "metadata_baseline": {
            "status": "changed_conflicts_are_not_duplicate_version_cleanup",
            "n_conflict_rows": n_conflicts,
            "same_canonical_title": metadata_counts.get("same_canonical_title", 0),
            "same_normalized_title": metadata_counts.get("same_normalized_title", 0),
            "same_artist": metadata_counts.get("same_artist", 0),
            "same_album": metadata_counts.get("same_album", 0),
            "same_release_year": metadata_counts.get("same_release_year", 0),
        },
        "reviewer_answer": [
            "The detector is not merely finding quotation marks: strict positives are active exact requests far more often than quoted and unquoted controls.",
            "When the independent annotation marks strict positives active, it names the same requested title in 37/37 cases.",
            "Changed exact/version conflicts are 0/94 same-canonical-title cases, so the result is not explained by ordinary duplicate/version cleanup.",
        ],
        "limitations": [
            "The frame ablation is stratified and estimates detector validity, not population prevalence.",
            "The metadata baseline does not prove user preference; it rules out a simpler catalog-dedup explanation for changed conflicts.",
        ],
        "source_artifacts": [
            "docs/evidence/request_frame_ablation_val_v08_score.json",
            "docs/evidence/request_frame_ablation_val_v08_score.md",
            "docs/evidence/request_conflict_metadata_baseline_v09.json",
            "docs/evidence/request_conflict_metadata_baseline_v09.md",
        ],
    }


def markdown(capsule: dict[str, Any]) -> str:
    frame = capsule["frame_ablation"]
    buckets = frame["buckets"]
    metadata = capsule["metadata_baseline"]
    n = metadata["n_conflict_rows"]
    lines = [
        "# Request Detector Validity Capsule v0.9",
        "",
        f"Date: {capsule['date']}",
        "",
        f"Question: {capsule['paper_question']}",
        "",
        "This is evidence for detector validity and simpler-baseline rejection, not a prevalence estimate or a human preference study.",
        "",
        "## Frame Ablation",
        "",
        "| readout | value |",
        "|---|---:|",
        f"| stratified rows | {frame['n_rows']} |",
        f"| minimum confidence | {frame['min_confidence']} |",
        f"| strict detector-positive active exact request | {rate_text(frame['detector_positive_active'])} |",
        f"| control active exact request | {rate_text(frame['control_active'])} |",
        f"| title overlap among active strict positives | {rate_text(frame['strict_detector_title_overlap_among_active'])} |",
        "",
        "## By Bucket",
        "",
        "| bucket | n | active exact request | title overlap among active |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [
        ("strict_detected_exact_request", "strict detected exact request"),
        ("quoted_not_strict_detected", "quoted but not strict-detected"),
        ("unquoted_request_like", "unquoted request-like control"),
    ]:
        row = buckets[key]
        lines.append(
            f"| {label} | {row['n']} | {rate_text(row['active_exact'])} | "
            f"{rate_text(row['title_overlap_among_active'])} |"
        )

    lines.extend([
        "",
        "## Metadata Baseline",
        "",
        "| relation between policy label and request target | count |",
        "|---|---:|",
        f"| same canonical title / likely duplicate-version | {count_text(metadata['same_canonical_title'], n)} |",
        f"| same normalized title | {count_text(metadata['same_normalized_title'], n)} |",
        f"| same artist | {count_text(metadata['same_artist'], n)} |",
        f"| same album | {count_text(metadata['same_album'], n)} |",
        f"| same release year | {count_text(metadata['same_release_year'], n)} |",
        "",
        "## Reviewer Answer",
        "",
    ])
    lines.extend(f"- {item}" for item in capsule["reviewer_answer"])
    lines.extend([
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in capsule["limitations"])
    lines.extend([
        "",
        "## Sources",
        "",
    ])
    lines.extend(f"- `{path}`" for path in capsule["source_artifacts"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frame-score",
        type=Path,
        default=EVID / "request_frame_ablation_val_v08_score.json",
    )
    ap.add_argument(
        "--metadata-baseline",
        type=Path,
        default=EVID / "request_conflict_metadata_baseline_v09.json",
    )
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = ap.parse_args(argv)

    capsule = build_capsule(load(args.frame_score), load(args.metadata_baseline))
    args.json_out.write_text(json.dumps(capsule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.write_text(markdown(capsule), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "md": str(args.md_out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
