"""Promote catalog-resolved exact requests without using policy labels.

This is an inference-time counterpart to the request-correction sidecar. It uses
only the visible dialogue, the catalog, and an existing prediction list:

  visible exact request -> catalog-resolved requested track ids -> promote if present

It does not use gold labels or corrected-label sidecars to decide whether to
intervene. That makes it suitable as a non-oracle system behavior experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("MCRS_EXPLORE_ROOT", str(Path(__file__).resolve().parent.parent)))

from detect_requests import (  # noqa: E402
    active_quoted_requests,
    active_request_span,
    build_catalog_index,
    canon_title,
    extract_by_artist,
    extract_exact_album_hint,
    has_unquoted_by_clause,
    load_sessions,
    norm_text,
    norm_values,
    reference_context,
    unquoted_text,
)

VERSION_MODIFIERS = (
    "acoustic",
    "live",
    "remix",
    "remaster",
    "remastered",
    "radio edit",
    "single edit",
    "demo",
)

def filter_version_modifiers(request_text: str, title_raw: str, exact_ids: list[str], catalog) -> list[str]:
    request_norm = norm_text(unquoted_text(request_text))
    title_norm = norm_text(title_raw)
    needed = [
        modifier for modifier in VERSION_MODIFIERS
        if modifier in request_norm and modifier not in title_norm
    ]
    if not needed:
        return exact_ids
    matching = []
    for tid in exact_ids:
        track_norm = norm_text(catalog.by_id[tid].get("track_name"))
        if all(modifier in track_norm for modifier in needed):
            matching.append(tid)
    return matching


def exact_request_ids(user_text: str, catalog) -> dict[str, list[str]]:
    """Return requested title -> catalog ids for exact quoted-track requests.

    Mirrors the high-precision exact detector, but deliberately does not inspect
    the policy-selected label. Ambiguous same-title covers are skipped unless an
    artist or album hint disambiguates them.
    """
    request_text = active_request_span(user_text)
    artist_hint = extract_by_artist(request_text, catalog)
    album_hint = extract_exact_album_hint(request_text, catalog)
    out: dict[str, list[str]] = {}
    for title_raw in active_quoted_requests(request_text):
        if reference_context(request_text, title_raw):
            continue
        title_norm = norm_text(title_raw)
        exact_ids = list(catalog.by_title.get(title_norm, []))
        exact_ids = filter_version_modifiers(request_text, title_raw, exact_ids, catalog)
        if artist_hint:
            exact_ids = [
                tid for tid in exact_ids
                if artist_hint in norm_values(catalog.by_id[tid].get("artist_name"))
            ]
        if album_hint:
            exact_ids = [
                tid for tid in exact_ids
                if album_hint in norm_values(catalog.by_id[tid].get("album_name"))
            ]
        if not artist_hint and not album_hint and has_unquoted_by_clause(request_text):
            exact_ids = []
        elif len({norm_text(catalog.by_id[tid].get("artist_name")) for tid in exact_ids}) > 1:
            exact_ids = []
        if exact_ids:
            out[title_raw] = sorted(set(exact_ids))
    return out


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    for row in rows:
        key = (row["session_id"], int(row["turn_number"]))
        if key in seen:
            raise ValueError(f"duplicate prediction key {key}")
        seen.add(key)
        tids = row.get("predicted_track_ids") or []
        if len(tids) != len(set(tids)):
            raise ValueError(f"duplicate track ids for {key}")
    return rows


def request_directives(split: str, catalog, max_sessions: int | None = None) -> dict[tuple[str, int], dict[str, Any]]:
    sessions = load_sessions(split)
    if max_sessions is not None:
        sessions = sessions.select(range(min(max_sessions, len(sessions))))
    directives: dict[tuple[str, int], dict[str, Any]] = {}
    for session in sessions:
        sid = session["session_id"]
        for row in session["conversations"]:
            if row.get("role") != "user":
                continue
            turn = int(row["turn_number"])
            found = exact_request_ids(row.get("content") or "", catalog)
            if not found:
                continue
            requested_ids: list[str] = []
            for ids in found.values():
                requested_ids.extend(ids)
            directives[(sid, turn)] = {
                "requested_titles": sorted(found),
                "requested_title_to_ids": {title: sorted(set(ids)) for title, ids in sorted(found.items())},
                "requested_track_ids": sorted(set(requested_ids)),
                "user_text": row.get("content") or "",
            }
    return directives


def promote_requested(
    predictions: list[dict[str, Any]],
    directives: dict[tuple[str, int], dict[str, Any]],
    *,
    k_search: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    requested_turns = 0
    present_turns = 0
    changed_turns = 0
    promoted_ids = 0
    rank_before: list[int] = []
    rank_after: list[int] = []
    changed_preview: list[dict[str, Any]] = []

    for row in predictions:
        new_row = dict(row)
        key = (row["session_id"], int(row["turn_number"]))
        predicted = list(row.get("predicted_track_ids") or [])
        directive = directives.get(key)
        if directive:
            requested_turns += 1
            title_to_ids = directive.get("requested_title_to_ids") or {
                "__all__": directive["requested_track_ids"]
            }
            present = []
            for title in sorted(title_to_ids):
                ids = set(title_to_ids[title])
                best = next((tid for tid in predicted[:k_search] if tid in ids), None)
                if best and best not in present:
                    present.append(best)
            if present:
                present_turns += 1
                rank_before.append(1 + min(predicted.index(tid) for tid in present))
                present_set = set(present)
                promoted = present + [tid for tid in predicted if tid not in present_set]
                new_row["predicted_track_ids"] = promoted[:len(predicted)]
                rank_after.append(1)
                if promoted != predicted:
                    changed_turns += 1
                    promoted_ids += len(present)
                    if len(changed_preview) < 25:
                        changed_preview.append({
                            "session_id": key[0],
                            "turn_number": key[1],
                            "requested_titles": directive["requested_titles"],
                            "rank_before": rank_before[-1],
                            "promoted_track_ids": present,
                        })
        out.append(new_row)

    def mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return out, {
        "requested_turns": requested_turns,
        "present_turns": present_turns,
        "changed_turns": changed_turns,
        "promoted_ids": promoted_ids,
        "mean_rank_before_when_present": mean(rank_before),
        "mean_rank_after_when_present": mean(rank_after),
        "k_search": k_search,
        "changed_preview": changed_preview,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-path", type=Path, required=True)
    ap.add_argument("--split", choices=["devset", "val", "train_lt", "train", "train_full"], default="devset")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary-output", type=Path, required=True)
    ap.add_argument("--k-search", type=int, default=100)
    ap.add_argument("--max-sessions", type=int)
    args = ap.parse_args(argv)

    from datasets import load_dataset

    catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
    predictions = load_predictions(args.prediction_path)
    directives = request_directives(args.split, catalog, args.max_sessions)
    postranked, summary = promote_requested(predictions, directives, k_search=args.k_search)
    summary.update({
        "prediction_path": str(args.prediction_path),
        "output": str(args.output),
        "split": args.split,
        "n_predictions": len(predictions),
        "n_directives": len(directives),
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(postranked, indent=2), encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
