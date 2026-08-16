"""Score independent labels for the request human-check packet."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from score_adjudication import (
    confidence_ok,
    load_jsonl,
    load_labels,
    rate_block,
    yes,
)


ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
DEFAULT_KEY = EVID / "request_human_check_packet_v09_key.jsonl"
DEFAULT_JSON = EVID / "request_human_check_packet_v09_score.json"
DEFAULT_MD = EVID / "request_human_check_packet_v09_score.md"


def norm_packet_choice(value: Any, *, allow_both: bool = False) -> str:
    value = str(value or "").strip().lower()
    if value in {"a", "candidate_a", "candidate a"}:
        return "A"
    if value in {"b", "candidate_b", "candidate b"}:
        return "B"
    if allow_both and value == "both":
        return "both"
    if value in {"tie", "equal"} or (not allow_both and value == "both"):
        return "tie"
    if allow_both and value == "neither":
        return "neither"
    if value in {"unclear", "unknown", ""}:
        return "unclear"
    return "other"


def summarize(keys: list[dict[str, Any]], labels: dict[str, dict[str, Any]], *, min_confidence: str) -> dict[str, Any]:
    key_by_id = {row["sample_id"]: row for row in keys}
    missing = sorted(set(key_by_id) - set(labels))
    extra = sorted(set(labels) - set(key_by_id))
    if missing:
        raise ValueError(f"missing labels for {len(missing)} samples, first={missing[:5]}")
    if extra:
        raise ValueError(f"extra labels for {len(extra)} samples, first={extra[:5]}")

    preferred_counts: Counter[str] = Counter()
    satisfying_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    explicit_counts: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = {}
    scorable_pref = 0
    request_pref = 0
    policy_pref = 0
    scorable_satisfying = 0
    request_side_satisfying = 0
    policy_side_satisfying = 0
    request_exclusive_satisfying = 0
    policy_exclusive_satisfying = 0

    for sid, key in key_by_id.items():
        label = labels[sid]
        split = key.get("split", "unknown")
        by_split.setdefault(split, Counter())
        conf = str(label.get("confidence", "missing")).strip().lower() or "missing"
        explicit = str(label.get("explicit_request_visible", "missing")).strip().lower() or "missing"
        confidence_counts[conf] += 1
        explicit_counts[explicit] += 1

        preferred = norm_packet_choice(label.get("preferred_item_for_request"))
        satisfying = norm_packet_choice(label.get("request_satisfying_item"), allow_both=True)
        preferred_counts[preferred] += 1
        satisfying_counts[satisfying] += 1

        if not confidence_ok(label.get("confidence"), min_confidence):
            by_split[split]["low_confidence"] += 1
            continue
        if not yes(label.get("explicit_request_visible")):
            by_split[split]["no_visible_request"] += 1
            continue

        if preferred in {"A", "B"}:
            scorable_pref += 1
            if preferred == key["request_side"]:
                request_pref += 1
                by_split[split]["request_preferred"] += 1
            elif preferred == key["policy_side"]:
                policy_pref += 1
                by_split[split]["policy_preferred"] += 1
        elif preferred == "tie":
            by_split[split]["preference_tie"] += 1
        else:
            by_split[split]["preference_unclear"] += 1

        if satisfying in {"A", "B", "both"}:
            scorable_satisfying += 1
            if satisfying in {key["request_side"], "both"}:
                request_side_satisfying += 1
                by_split[split]["request_side_satisfying"] += 1
            if satisfying in {key["policy_side"], "both"}:
                policy_side_satisfying += 1
                by_split[split]["policy_side_satisfying"] += 1
            if satisfying == key["request_side"]:
                request_exclusive_satisfying += 1
                by_split[split]["request_exclusive_satisfying"] += 1
            elif satisfying == key["policy_side"]:
                policy_exclusive_satisfying += 1
                by_split[split]["policy_exclusive_satisfying"] += 1
            elif satisfying == "both":
                by_split[split]["both_satisfying"] += 1
        elif satisfying == "neither":
            by_split[split]["neither_satisfying"] += 1
        else:
            by_split[split]["satisfying_unclear"] += 1

    return {
        "artifact": "request-human-check-score-v0.9",
        "n_rows": len(keys),
        "min_confidence": min_confidence,
        "preferred_item_for_request_counts": dict(sorted(preferred_counts.items())),
        "request_satisfying_item_counts": dict(sorted(satisfying_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "explicit_request_visible_counts": dict(sorted(explicit_counts.items())),
        "request_preferred_rate": rate_block(request_pref, scorable_pref),
        "policy_preferred_rate": rate_block(policy_pref, scorable_pref),
        "request_side_satisfying_rate": rate_block(request_side_satisfying, scorable_satisfying),
        "policy_side_satisfying_rate": rate_block(policy_side_satisfying, scorable_satisfying),
        "request_exclusive_satisfying_rate": rate_block(request_exclusive_satisfying, scorable_satisfying),
        "policy_exclusive_satisfying_rate": rate_block(policy_exclusive_satisfying, scorable_satisfying),
        "by_split": {split: dict(sorted(counter.items())) for split, counter in sorted(by_split.items())},
        "read": (
            "This scores independent labels for the blinded human-check packet. "
            "It converts collected annotations into request-side preference and "
            "request-satisfaction rates without using traces, model scores, or "
            "the answer key during labeling."
        ),
    }


def fmt_rate(block: dict[str, Any]) -> str:
    return f"{block['successes']}/{block['n']} ({block['point']:.3f}, CI [{block['low']:.3f}, {block['high']:.3f}])"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Request Human-Check Label Score v0.9",
        "",
        summary["read"],
        "",
        f"Minimum confidence: `{summary['min_confidence']}`",
        "",
        "## Summary",
        "",
        "| readout | count |",
        "|---|---:|",
        f"| request-side item preferred for request | {fmt_rate(summary['request_preferred_rate'])} |",
        f"| policy-side item preferred for request | {fmt_rate(summary['policy_preferred_rate'])} |",
        f"| request-side item marked satisfying | {fmt_rate(summary['request_side_satisfying_rate'])} |",
        f"| policy-side item marked satisfying | {fmt_rate(summary['policy_side_satisfying_rate'])} |",
        f"| request-side item exclusively satisfying | {fmt_rate(summary['request_exclusive_satisfying_rate'])} |",
        f"| policy-side item exclusively satisfying | {fmt_rate(summary['policy_exclusive_satisfying_rate'])} |",
        "",
        "## Raw Counts",
        "",
        f"- preferred_item_for_request: `{summary['preferred_item_for_request_counts']}`",
        f"- request_satisfying_item: `{summary['request_satisfying_item_counts']}`",
        f"- explicit_request_visible: `{summary['explicit_request_visible_counts']}`",
        f"- confidence: `{summary['confidence_counts']}`",
        "",
        "## By Split",
        "",
        "| split | counts |",
        "|---|---|",
    ]
    for split, counts in summary["by_split"].items():
        lines.append(f"| {split} | `{counts}` |")
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "- This is human evidence only after independent labels are collected.",
        "- It measures visible request satisfaction in exact/request conflict rows, not broad music preference.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--labels", type=Path, action="append", required=True)
    ap.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    ap.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--markdown-output", type=Path, default=DEFAULT_MD)
    args = ap.parse_args(argv)

    keys = load_jsonl(args.key)
    labels = load_labels(args.labels)
    summary = summarize(keys, labels, min_confidence=args.min_confidence)
    summary.update({
        "key": str(args.key),
        "label_files": [str(path) for path in args.labels],
    })
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({
        "json_output": str(args.json_output),
        "markdown_output": str(args.markdown_output),
        "n_rows": summary["n_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
