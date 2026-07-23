"""Analyze metadata relations in policy-label/request-target conflicts.

This addresses a novelty objection: if exact-request conflicts were mostly
ordinary duplicate/version metadata issues, a generic catalog-cleaning audit
could explain them. The request-aware claim is stronger when the policy label
and request target are usually different titles, not duplicate releases.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from request_correction_labels import canon_title, norm_text
except ModuleNotFoundError:  # pragma: no cover - used when imported from tests
    from scripts.request_correction_labels import canon_title, norm_text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def release_year(track: dict[str, Any]) -> str:
    raw = first(track.get("release_date"))
    return raw[:4] if len(raw) >= 4 and raw[:4].isdigit() else ""


def relation(policy: dict[str, Any], request: dict[str, Any]) -> dict[str, bool]:
    policy_title = first(policy.get("track_name"))
    request_title = first(request.get("track_name"))
    policy_artist = norm_text(policy.get("artist_name"))
    request_artist = norm_text(request.get("artist_name"))
    policy_album = norm_text(policy.get("album_name"))
    request_album = norm_text(request.get("album_name"))
    policy_year = release_year(policy)
    request_year = release_year(request)
    return {
        "same_track_id": bool(policy.get("track_id") and policy.get("track_id") == request.get("track_id")),
        "same_normalized_title": bool(norm_text(policy_title) and norm_text(policy_title) == norm_text(request_title)),
        "same_canonical_title": bool(canon_title(policy_title) and canon_title(policy_title) == canon_title(request_title)),
        "same_artist": bool(policy_artist and policy_artist == request_artist),
        "same_album": bool(policy_album and policy_album == request_album),
        "same_release_year": bool(policy_year and policy_year == request_year),
    }


def analyze(changed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    conflicts = [
        row for row in changed_rows
        if not row.get("gold_is_requested")
        and row.get("gold_track")
        and row.get("promoted_tracks")
    ]
    overall: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = {}
    examples: dict[str, list[dict[str, Any]]] = {
        "same_artist_different_title": [],
        "different_artist": [],
        "same_album_different_title": [],
    }
    rows = []
    for row in conflicts:
        split = row.get("split", "unknown")
        by_split.setdefault(split, Counter())
        rel = relation(row["gold_track"], row["promoted_tracks"][0])
        for key, value in rel.items():
            if value:
                overall[key] += 1
                by_split[split][key] += 1
        overall["n"] += 1
        by_split[split]["n"] += 1
        out = {
            "split": split,
            "session_id": row["session_id"],
            "turn_number": int(row["turn_number"]),
            "requested_titles": row.get("requested_titles") or [],
            "policy_track": row["gold_track"],
            "request_track": row["promoted_tracks"][0],
            "relation": rel,
            "user_text": row.get("user_text"),
        }
        rows.append(out)
        if rel["same_artist"] and not rel["same_canonical_title"] and len(examples["same_artist_different_title"]) < 5:
            examples["same_artist_different_title"].append(out)
        if not rel["same_artist"] and len(examples["different_artist"]) < 5:
            examples["different_artist"].append(out)
        if rel["same_album"] and not rel["same_canonical_title"] and len(examples["same_album_different_title"]) < 5:
            examples["same_album_different_title"].append(out)

    n = overall["n"]
    return {
        "n_conflict_rows": n,
        "counts": dict(sorted(overall.items())),
        "rates": {key: (value / n if n else 0.0) for key, value in sorted(overall.items()) if key != "n"},
        "by_split": {
            split: {
                "counts": dict(sorted(counter.items())),
                "rates": {
                    key: (value / counter["n"] if counter["n"] else 0.0)
                    for key, value in sorted(counter.items())
                    if key != "n"
                },
            }
            for split, counter in sorted(by_split.items())
        },
        "examples": examples,
        "rows": rows,
    }


def fmt_count(summary: dict[str, Any], key: str) -> str:
    n = summary["n_conflict_rows"]
    count = summary["counts"].get(key, 0)
    return f"{count}/{n} ({(count / n if n else 0.0):.1%})"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Request Conflict Metadata Baseline",
        "",
        "Date: 2026-06-21",
        "",
        "This checks whether changed exact-request conflicts are ordinary metadata duplicate/version cases. Rows are changed interventions where the policy-selected label differs from the request-satisfying promoted item.",
        "",
        "## Summary",
        "",
        "| relation between policy label and request target | count |",
        "|---|---:|",
    ]
    for key, label in [
        ("same_canonical_title", "same canonical title / likely duplicate-version"),
        ("same_normalized_title", "same normalized title"),
        ("same_artist", "same artist"),
        ("same_album", "same album"),
        ("same_release_year", "same release year"),
    ]:
        lines.append(f"| {label} | {fmt_count(summary, key)} |")

    lines.extend([
        "",
        "## By Split",
        "",
        "| split | n | same canonical title | same artist | same album |",
        "|---|---:|---:|---:|---:|",
    ])
    for split, block in summary["by_split"].items():
        n = block["counts"].get("n", 0)
        def cell(key: str) -> str:
            count = block["counts"].get(key, 0)
            return f"{count}/{n} ({(count / n if n else 0.0):.1%})"
        lines.append(
            f"| {split} | {n} | {cell('same_canonical_title')} | "
            f"{cell('same_artist')} | {cell('same_album')} |"
        )

    lines.extend([
        "",
        "## Read",
        "",
        "- Same canonical title is the generic duplicate/version baseline.",
        "- Low same-title overlap supports that the signal needs dialogue-aware request resolution, not only catalog deduplication.",
        "- Same artist/album overlap can still be high: many generator choices stay near the requested musical context while missing the exact requested item.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--changed-rows", type=Path, action="append", required=True)
    ap.add_argument("--json-output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path, required=True)
    args = ap.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for path in args.changed_rows:
        rows.extend(load_jsonl(path))
    summary = analyze(rows)
    summary["changed_rows"] = [str(path) for path in args.changed_rows]
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({
        "json_output": str(args.json_output),
        "markdown_output": str(args.markdown_output),
        "n_conflict_rows": summary["n_conflict_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
