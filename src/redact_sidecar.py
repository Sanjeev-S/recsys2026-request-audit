"""Redact correction sidecars to the ID-keyed release schema.

TalkPlayData 2 is licensed CC-BY-NC-ND-4.0, so released evidence must not
embed dialogue text or track/artist/album metadata. This tool strips a
correction sidecar down to identifiers and audit fields; every verdict stays
re-checkable against the reader's own copy of the benchmark via the IDs.

Kept fields: session_id, turn_number, split, family, action, confidence,
detector_version, group_weight, mask_gold, additional_track_ids,
gold_track_id, plus a derived gold_in_targets flag (true when the official
label is inside the resolved target set, i.e. the official label already
satisfies the request).

Dropped fields: evidence (dialogue span/text), requested_text, gold_metadata,
and any other free-text field.

Also usable as a checker: --check fails if a jsonl contains forbidden fields.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

KEEP_FIELDS = [
    "session_id", "turn_number", "split", "family", "action", "confidence",
    "detector_version", "group_weight", "mask_gold",
    "additional_track_ids", "gold_track_id",
]
FORBIDDEN_FIELDS = {
    "evidence", "requested_text", "gold_metadata", "user_text", "assistant_text",
    "music_thought", "track_name", "artist_name", "album_name", "requested_titles",
    "requested_title_norm", "request_span", "text", "notes",
}


def forbidden_keys(obj: Any, path: str = "") -> list[str]:
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_FIELDS:
                hits.append(f"{path}/{k}")
            hits.extend(forbidden_keys(v, f"{path}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(forbidden_keys(v, f"{path}[{i}]"))
    return hits


def redact(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: row[k] for k in KEEP_FIELDS if k in row}
    targets = set(row.get("additional_track_ids") or [])
    out["gold_in_targets"] = bool(row.get("gold_track_id") in targets)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--check", action="store_true", help="verify a jsonl has no forbidden fields")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]

    if args.check:
        bad = []
        for i, row in enumerate(rows):
            bad.extend(f"row {i}: {p}" for p in forbidden_keys(row))
        if bad:
            print(f"FORBIDDEN FIELDS in {args.input} ({len(bad)} hits):")
            for b in bad[:20]:
                print(" ", b)
            sys.exit(1)
        print(f"OK: {args.input} ({len(rows)} rows, no forbidden fields)")
        return

    if not args.output:
        ap.error("--output required unless --check")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = redact(row)
            leftovers = forbidden_keys(clean)
            if leftovers:
                raise SystemExit(f"redaction left forbidden fields: {leftovers}")
            f.write(json.dumps(clean, sort_keys=True) + "\n")
    print(f"redacted {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
