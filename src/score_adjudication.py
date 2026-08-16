"""Score blinded policy-label vs request-target preference labels."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


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


def norm_choice(value: Any) -> str:
    value = str(value or "").strip().lower()
    if value in {"a", "candidate_a", "candidate a"}:
        return "A"
    if value in {"b", "candidate_b", "candidate b"}:
        return "B"
    if value in {"tie", "both", "equal"}:
        return "tie"
    if value in {"neither"}:
        return "neither"
    if value in {"unclear", "unknown", ""}:
        return "unclear"
    return "other"


def yes(value: Any) -> bool:
    return str(value or "").strip().lower() == "yes"


def confidence_ok(value: Any, min_confidence: str) -> bool:
    value = str(value or "").strip().lower()
    return CONFIDENCE_RANK.get(value, -1) >= CONFIDENCE_RANK[min_confidence]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, float]:
    if n <= 0:
        return {"point": 0.0, "low": 0.0, "high": 0.0}
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return {"point": p, "low": max(0.0, center - half), "high": min(1.0, center + half)}


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

    preferred_counts: Counter[str] = Counter()
    satisfying_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    explicit_counts: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = {}
    scorable_pref = 0
    request_pref = 0
    policy_pref = 0
    scorable_satisfying = 0
    request_satisfying = 0

    for sid, key in key_by_id.items():
        label = labels[sid]
        split = key.get("split", "unknown")
        by_split.setdefault(split, Counter())
        conf = str(label.get("confidence", "missing")).strip().lower() or "missing"
        confidence_counts[conf] += 1
        explicit_counts[str(label.get("explicit_request_visible", "missing")).strip().lower() or "missing"] += 1
        pref = norm_choice(label.get("preferred_item"))
        sat = norm_choice(label.get("request_satisfying_item"))
        preferred_counts[pref] += 1
        satisfying_counts[sat] += 1
        if not confidence_ok(label.get("confidence"), min_confidence):
            by_split[split]["low_confidence"] += 1
            continue
        if not yes(label.get("explicit_request_visible")):
            by_split[split]["no_visible_request"] += 1
            continue

        if pref in {"A", "B"}:
            scorable_pref += 1
            if pref == key["request_side"]:
                request_pref += 1
                by_split[split]["request_preferred"] += 1
            elif pref == key["policy_side"]:
                policy_pref += 1
                by_split[split]["policy_preferred"] += 1
        elif pref == "tie":
            by_split[split]["tie"] += 1
        else:
            by_split[split]["unclear"] += 1

        if sat in {"A", "B"}:
            scorable_satisfying += 1
            if sat == key["request_side"]:
                request_satisfying += 1
                by_split[split]["request_satisfying"] += 1
            elif sat == key["policy_side"]:
                by_split[split]["policy_satisfying"] += 1
        elif sat == "both":
            by_split[split]["both_satisfying"] += 1
        elif sat == "neither":
            by_split[split]["neither_satisfying"] += 1
        else:
            by_split[split]["satisfying_unclear"] += 1

    return {
        "n_rows": len(keys),
        "min_confidence": min_confidence,
        "preferred_item_counts": dict(sorted(preferred_counts.items())),
        "request_satisfying_item_counts": dict(sorted(satisfying_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "explicit_request_visible_counts": dict(sorted(explicit_counts.items())),
        "request_preferred_rate": rate_block(request_pref, scorable_pref),
        "policy_preferred_rate": rate_block(policy_pref, scorable_pref),
        "request_satisfying_rate": rate_block(request_satisfying, scorable_satisfying),
        "by_split": {split: dict(sorted(counter.items())) for split, counter in sorted(by_split.items())},
    }


def fmt_rate(block: dict[str, Any]) -> str:
    return f"{block['successes']}/{block['n']} ({block['point']:.3f}, CI [{block['low']:.3f}, {block['high']:.3f}])"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Request Preference Labels",
        "",
        "Date: 2026-06-21",
        "",
        "This scores blinded comparisons between the official policy-selected label and the request-satisfying item promoted by the locked exact-request postranker.",
        "",
        f"Minimum confidence: `{summary['min_confidence']}`",
        "",
        "## Summary",
        "",
        "| readout | count |",
        "|---|---:|",
        f"| request item preferred | {fmt_rate(summary['request_preferred_rate'])} |",
        f"| policy item preferred | {fmt_rate(summary['policy_preferred_rate'])} |",
        f"| request item marked request-satisfying | {fmt_rate(summary['request_satisfying_rate'])} |",
        "",
        "## Raw Counts",
        "",
        f"- preferred_item: `{summary['preferred_item_counts']}`",
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
        "## Read",
        "",
        "- This validates the external meaning of corrected nDCG: whether the request-positive target is preferred for the visible user request.",
        "- It does not validate broad music preference, only request satisfaction in exact-request conflict rows.",
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
