"""Score independent request-frame labels against the detector key.

This is a parser-ablation diagnostic, not a prevalence estimator: the input set
is stratified into detector positives and controls. The useful readout is
whether independent labels recover active exact-track requests in detector
positive rows, and how often controls contain active exact-track requests missed
by the detector.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from detect_requests import norm_text
except ModuleNotFoundError:  # pragma: no cover - used when imported from tests
    from scripts.detect_requests import norm_text


CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_labels(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            sid = row["sample_id"]
            if sid in out:
                raise ValueError(f"duplicate label sample_id={sid}")
            out[sid] = row
    return out


def yes(value: Any) -> bool:
    return str(value or "").strip().lower() == "yes"


def label_value(value: Any) -> str:
    value = str(value or "missing").strip().lower()
    return value if value in {"yes", "no", "unclear"} else "other"


def confidence_ok(value: Any, min_confidence: str) -> bool:
    value = str(value or "").strip().lower()
    return CONFIDENCE_RANK.get(value, -1) >= CONFIDENCE_RANK[min_confidence]


def as_titles(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def active_exact(label: dict[str, Any], *, min_confidence: str) -> bool:
    return (
        yes(label.get("is_explicit_track_request"))
        and not yes(label.get("is_reference_or_example"))
        and confidence_ok(label.get("confidence"), min_confidence)
    )


def title_overlap(label: dict[str, Any], key: dict[str, Any]) -> bool:
    labeled = {norm_text(title) for title in as_titles(label.get("requested_titles"))}
    detector = {
        norm_text(item.get("requested_title"))
        for item in key.get("detector_requested") or []
        if item.get("requested_title")
    }
    return bool(labeled & detector)


def detector_title_active(label: dict[str, Any], key: dict[str, Any], *, min_confidence: str) -> bool:
    """Whether the annotator independently named the detector's requested title.

    The annotation field `is_reference_or_example` is row-level: a turn can
    mention one title as a reference and another title as the actual request.
    For detector-positive rows, title overlap is the less lossy check.
    """
    return (
        yes(label.get("is_explicit_track_request"))
        and confidence_ok(label.get("confidence"), min_confidence)
        and title_overlap(label, key)
    )


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, float]:
    if n <= 0:
        return {"point": 0.0, "low": 0.0, "high": 0.0}
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return {"point": p, "low": center - half, "high": center + half}


def rate_block(successes: int, n: int) -> dict[str, Any]:
    out = wilson(successes, n)
    out.update({"successes": successes, "n": n})
    return out


def summarize(keys: list[dict[str, Any]], labels: dict[str, dict[str, Any]], *, min_confidence: str) -> dict[str, Any]:
    key_by_id = {row["sample_id"]: row for row in keys}
    missing = sorted(set(key_by_id) - set(labels))
    extra = sorted(set(labels) - set(key_by_id))
    if missing:
        raise ValueError(f"missing labels for {len(missing)} samples, first={missing[:5]}")
    if extra:
        raise ValueError(f"extra labels for {len(extra)} samples, first={extra[:5]}")

    by_bucket: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for sid, key in key_by_id.items():
        by_bucket[key["bucket"]].append((key, labels[sid]))

    def bucket_block(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
        n = len(pairs)
        if pairs and pairs[0][0]["bucket"] == "strict_detected_exact_request":
            active = sum(
                detector_title_active(label, key, min_confidence=min_confidence)
                for key, label in pairs
            )
        else:
            active = sum(active_exact(label, min_confidence=min_confidence) for _, label in pairs)
        title_match_den = 0
        title_match_num = 0
        explicit_counts = Counter(label_value(label.get("is_explicit_track_request")) for _, label in pairs)
        reference_counts = Counter(label_value(label.get("is_reference_or_example")) for _, label in pairs)
        confidence_counts = Counter(str(label.get("confidence", "missing")).strip().lower() or "missing" for _, label in pairs)
        for key, label in pairs:
            if active_exact(label, min_confidence=min_confidence) or detector_title_active(
                label,
                key,
                min_confidence=min_confidence,
            ):
                title_match_den += 1
                title_match_num += int(title_overlap(label, key))
        return {
            "n": n,
            "active_exact": rate_block(active, n),
            "title_overlap_among_active": rate_block(title_match_num, title_match_den),
            "explicit_counts": dict(sorted(explicit_counts.items())),
            "reference_counts": dict(sorted(reference_counts.items())),
            "confidence_counts": dict(sorted(confidence_counts.items())),
        }

    detector_pairs = by_bucket.get("strict_detected_exact_request", [])
    control_pairs = [
        pair
        for bucket, pairs in by_bucket.items()
        if bucket != "strict_detected_exact_request"
        for pair in pairs
    ]
    detector_active = sum(
        detector_title_active(label, key, min_confidence=min_confidence)
        for key, label in detector_pairs
    )
    control_active = sum(active_exact(label, min_confidence=min_confidence) for _, label in control_pairs)

    return {
        "n_rows": len(keys),
        "min_confidence": min_confidence,
        "by_bucket": {bucket: bucket_block(pairs) for bucket, pairs in sorted(by_bucket.items())},
        "detector_positive_active_rate": rate_block(detector_active, len(detector_pairs)),
        "control_active_rate": rate_block(control_active, len(control_pairs)),
        "label_files": [],
    }


def fmt_rate(block: dict[str, Any]) -> str:
    return f"{block['successes']}/{block['n']} ({block['point']:.3f}, CI [{block['low']:.3f}, {block['high']:.3f}])"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Request-Frame Parser Ablation v0.8",
        "",
        "Date: 2026-06-21",
        "",
        "This is a stratified annotation check, not a prevalence estimate. A detector-positive row is counted as active when the independent label names the same requested title with sufficient confidence. A control row is counted as active when the independent label says `is_explicit_track_request=yes`, `is_reference_or_example` is not `yes`, and confidence meets the configured threshold.",
        "",
        f"Minimum confidence: `{summary['min_confidence']}`",
        "",
        "## Summary",
        "",
        "| readout | count |",
        "|---|---:|",
        f"| detector-positive active exact request | {fmt_rate(summary['detector_positive_active_rate'])} |",
        f"| control active exact request | {fmt_rate(summary['control_active_rate'])} |",
        "",
        "## By Bucket",
        "",
        "| bucket | n | active exact request | title overlap among active |",
        "|---|---:|---:|---:|",
    ]
    for bucket, block in summary["by_bucket"].items():
        lines.append(
            f"| {bucket} | {block['n']} | {fmt_rate(block['active_exact'])} | "
            f"{fmt_rate(block['title_overlap_among_active'])} |"
        )
    lines.extend([
        "",
        "## Read",
        "",
        "- High detector-positive active rate supports that the regex detector is recovering a real request frame.",
        "- Nonzero control active rate identifies detector recall gaps or ambiguous requests for follow-up inspection.",
        "- Title overlap checks whether the independent parser names the same requested item as the detector.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--labels", type=Path, action="append", required=True)
    ap.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    ap.add_argument("--json-output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path, required=True)
    args = ap.parse_args(argv)

    keys = load_jsonl(args.key)
    labels = load_labels(args.labels)
    summary = summarize(keys, labels, min_confidence=args.min_confidence)
    summary["key"] = str(args.key)
    summary["label_files"] = [str(path) for path in args.labels]
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({
        "json_output": str(args.json_output),
        "markdown_output": str(args.markdown_output),
        "n_rows": summary["n_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
