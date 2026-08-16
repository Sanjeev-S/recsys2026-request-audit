"""Build a blinded human-check packet for exact/request conflict rows."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
DEFAULT_ANNOTATION = EVID / "request_human_check_packet_v09_annotation.jsonl"
DEFAULT_LABEL_TEMPLATE = EVID / "request_human_check_packet_v09_label_template.jsonl"
DEFAULT_KEY = EVID / "request_human_check_packet_v09_key.jsonl"
DEFAULT_SUMMARY = EVID / "request_human_check_packet_v09.summary.json"
DEFAULT_PROTOCOL = EVID / "request_human_check_packet_v09.md"

FORBIDDEN_ANNOTATION_FIELDS = {
    "request_side",
    "policy_side",
    "request_track_id",
    "policy_track_id",
    "request_track",
    "policy_track",
    "gold_track",
    "gold_track_id",
    "trace",
    "thought",
    "music_thought",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def sample_annotation_rows(
    annotations: list[dict[str, Any]],
    keys: list[dict[str, Any]],
    *,
    seed: int,
    target_rows: int,
    include_all_dev: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    key_by_id = {row["sample_id"]: row for row in keys}
    if set(key_by_id) != {row["sample_id"] for row in annotations}:
        raise ValueError("annotation/key sample_id mismatch")

    rng = random.Random(seed)
    dev = [row for row in annotations if row.get("split") == "devset"]
    nondev = [row for row in annotations if row.get("split") != "devset"]
    dev.sort(key=lambda row: row["sample_id"])
    nondev.sort(key=lambda row: row["sample_id"])
    rng.shuffle(nondev)

    selected = dev if include_all_dev else []
    remaining = max(0, target_rows - len(selected))
    selected = [*selected, *nondev[:remaining]]
    selected = selected[:target_rows]
    selected.sort(key=lambda row: (row.get("split", ""), row["sample_id"]))
    selected_ids = {row["sample_id"] for row in selected}
    return selected, [key_by_id[row["sample_id"]] for row in selected if row["sample_id"] in selected_ids]


def annotation_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "sample_id": row["sample_id"],
        "split": row.get("split"),
        "session_id": row.get("session_id"),
        "turn_number": row.get("turn_number"),
        "user_text": row.get("user_text"),
        "requested_titles": row.get("requested_titles", []),
        "candidate_A": row.get("candidate_A"),
        "candidate_B": row.get("candidate_B"),
        "annotation_schema": {
            "explicit_request_visible": "yes/no/unclear",
            "request_satisfying_item": "A/B/both/neither/unclear",
            "preferred_item_for_request": "A/B/tie/unclear",
            "confidence": "high/medium/low",
            "notes": "brief rationale, quote the request if useful",
        },
    }
    leaked = sorted(FORBIDDEN_ANNOTATION_FIELDS & set(out))
    if leaked:
        raise ValueError(f"annotation leak: {leaked}")
    return out


def label_template_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "explicit_request_visible": None,
        "request_satisfying_item": None,
        "preferred_item_for_request": None,
        "confidence": None,
        "notes": None,
    }


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def build_packet(
    *,
    annotations: list[dict[str, Any]],
    keys: list[dict[str, Any]],
    seed: int,
    target_rows: int,
    include_all_dev: bool,
) -> dict[str, Any]:
    selected, selected_keys = sample_annotation_rows(
        annotations,
        keys,
        seed=seed,
        target_rows=target_rows,
        include_all_dev=include_all_dev,
    )
    annotation_rows = [annotation_row(row) for row in selected]
    label_rows = [label_template_row(row) for row in selected]
    request_sides = Counter(row["request_side"] for row in selected_keys)
    by_split = Counter(row.get("split", "unknown") for row in selected)
    forbidden_hits = {
        row["sample_id"]: sorted(FORBIDDEN_ANNOTATION_FIELDS & set(row))
        for row in annotation_rows
        if FORBIDDEN_ANNOTATION_FIELDS & set(row)
    }
    return {
        "artifact": "request-human-check-packet-v0.9",
        "seed": seed,
        "target_rows": target_rows,
        "include_all_dev": include_all_dev,
        "n_source_annotations": len(annotations),
        "n_selected": len(annotation_rows),
        "by_split": dict(sorted(by_split.items())),
        "request_side_counts_in_key": dict(sorted(request_sides.items())),
        "forbidden_annotation_fields": sorted(FORBIDDEN_ANNOTATION_FIELDS),
        "forbidden_field_hits": forbidden_hits,
        "annotation_rows": annotation_rows,
        "label_template_rows": label_rows,
        "key_rows": selected_keys,
        "read": (
            "This is a blinded packet for a future human check of exact/request "
            "conflict rows. It is not human evidence until independent labels "
            "are collected and scored. The annotation file contains visible "
            "dialogue and two candidate metadata blocks only; request/policy "
            "side labels are kept in the separate key."
        ),
    }


def markdown(summary: dict[str, Any], *, annotation_path: Path, label_template_path: Path, key_path: Path) -> str:
    lines = [
        "# Request Human-Check Packet v0.9",
        "",
        summary["read"],
        "",
        "## Files",
        "",
        f"- Annotation packet: `{display_path(annotation_path)}`",
        f"- Label template: `{display_path(label_template_path)}`",
        f"- Answer key: `{display_path(key_path)}`",
        "",
        "## Scope",
        "",
        "| readout | value |",
        "|---|---:|",
        f"| source blinded comparisons | {summary['n_source_annotations']} |",
        f"| selected rows | {summary['n_selected']} |",
        f"| seed | {summary['seed']} |",
        f"| forbidden field hits | {len(summary['forbidden_field_hits'])} |",
        "",
        "## Split And Side Balance",
        "",
        f"- By split: `{summary['by_split']}`",
        f"- Request side counts in key: `{summary['request_side_counts_in_key']}`",
        "",
        "## Annotation Instructions",
        "",
        "For each row, read the visible user request and compare `candidate_A` and `candidate_B`.",
        "Do not use synthetic traces, official labels, model scores, or the answer key while labeling.",
        "Fill `explicit_request_visible`, `request_satisfying_item`, `preferred_item_for_request`, `confidence`, and `notes`.",
        "This packet checks request satisfaction in exact-request conflict rows only; it is not a broad music-preference study.",
        "",
        "## Scoring After Labels Exist",
        "",
        "After independent labels are collected in the label-template schema, score them with:",
        "",
        "```bash",
        ".venv/bin/python scripts/score_request_human_check_labels.py \\",
        f"  --key {display_path(key_path)} \\",
        "  --labels docs/evidence/request_human_check_packet_v09_labels_filled.jsonl \\",
        "  --json-output docs/evidence/request_human_check_packet_v09_score.json \\",
        "  --markdown-output docs/evidence/request_human_check_packet_v09_score.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation-source", type=Path, default=EVID / "request_preference_v09_annotation.jsonl")
    ap.add_argument("--key-source", type=Path, default=EVID / "request_preference_v09_key.jsonl")
    ap.add_argument("--seed", type=int, default=20260623)
    ap.add_argument("--target-rows", type=int, default=50)
    ap.add_argument("--no-include-all-dev", action="store_true")
    ap.add_argument("--annotation-out", type=Path, default=DEFAULT_ANNOTATION)
    ap.add_argument("--label-template-out", type=Path, default=DEFAULT_LABEL_TEMPLATE)
    ap.add_argument("--key-out", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--protocol-out", type=Path, default=DEFAULT_PROTOCOL)
    args = ap.parse_args(argv)

    packet = build_packet(
        annotations=load_jsonl(args.annotation_source),
        keys=load_jsonl(args.key_source),
        seed=args.seed,
        target_rows=args.target_rows,
        include_all_dev=not args.no_include_all_dev,
    )
    write_jsonl(args.annotation_out, packet["annotation_rows"])
    write_jsonl(args.label_template_out, packet["label_template_rows"])
    write_jsonl(args.key_out, packet["key_rows"])
    summary = {k: v for k, v in packet.items() if not k.endswith("_rows")}
    summary.update({
        "annotation_source": str(args.annotation_source),
        "key_source": str(args.key_source),
        "annotation_output": str(args.annotation_out),
        "label_template_output": str(args.label_template_out),
        "key_output": str(args.key_out),
    })
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.protocol_out.write_text(
        markdown(summary, annotation_path=args.annotation_out, label_template_path=args.label_template_out, key_path=args.key_out),
        encoding="utf-8",
    )
    print(json.dumps({
        "annotation": str(args.annotation_out),
        "label_template": str(args.label_template_out),
        "key": str(args.key_out),
        "summary": str(args.summary_out),
        "protocol": str(args.protocol_out),
        "n_selected": summary["n_selected"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
