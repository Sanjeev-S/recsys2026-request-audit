"""Promote already-retrieved tracks that satisfy strict hard artist requests.

This is a non-oracle probe for hard positive constraints. It uses only visible
dialogue, catalog metadata, and an existing prediction list:

  visible named-artist constraint -> first satisfying predicted track -> rank 1

It does not inspect policy labels or request-correction sidecars to decide
where to intervene.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("MCRS_EXPLORE_ROOT", str(Path(__file__).resolve().parent.parent)))

from apply_request_override import exact_request_ids  # noqa: E402
from detect_requests import (  # noqa: E402
    DECADE_RE,
    QUOTE_RE,
    REQUEST_START_RE,
    SOFT_CUES,
    YEAR_RE,
    active_request_span,
    build_catalog_index,
    extract_by_artists,
    load_sessions,
    norm_text,
    strict_hard_artist_constraint,
    track_satisfies_constraint,
)

HARD_ARTIST_COMPLEX_CUE_RE = re.compile(
    r"\b("
    r"album|era|period|early|mid|late|"
    r"specific song|exact song|specific track|exact track|"
    r"lyrics?|title|called|titled|"
    r"trying to find|trying to identify|looking for that|"
    r"that one|the one|"
    r"think|thinking of|for the mood|in that vein|in the vein|"
    r"example|such as|e\\.g\\.|not by"
    r")\b",
    re.I,
)


def simple_artist_only_request(user_text: str) -> bool:
    """Keep only residual artist requests with no visible title/era subtask."""
    request_text = active_request_span(user_text)
    if QUOTE_RE.search(request_text):
        return False
    if YEAR_RE.search(request_text) or DECADE_RE.search(request_text):
        return False
    if HARD_ARTIST_COMPLEX_CUE_RE.search(request_text):
        return False
    return True


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


def hard_artist_constraints(user_text: str, catalog, *, strict: bool) -> list[dict[str, Any]]:
    request_text = active_request_span(user_text)
    text_norm = norm_text(request_text)
    if not REQUEST_START_RE.search(request_text):
        return []
    if any(cue in text_norm for cue in SOFT_CUES):
        return []
    artists = extract_by_artists(request_text, catalog)
    constraints: list[dict[str, Any]] = []
    if len(artists) == 1:
        constraints.append({"kind": "artist", "artist_norm": artists[0]})
    elif len(artists) > 1:
        constraints.append({"kind": "artist_any", "artist_norms": artists})
    if strict:
        constraints = [
            c for c in constraints
            if strict_hard_artist_constraint(request_text, c)
        ]
    return constraints


def hard_artist_directives(
    split: str,
    catalog,
    max_sessions: int | None = None,
    *,
    strict: bool = True,
    exclude_exact_requests: bool = False,
    simple_artist_only: bool = False,
) -> dict[tuple[str, int], dict[str, Any]]:
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
            text = row.get("content") or ""
            constraints = hard_artist_constraints(text, catalog, strict=strict)
            if not constraints:
                continue
            exact_title_to_ids = {
                title: sorted(set(ids))
                for title, ids in sorted(exact_request_ids(text, catalog).items())
            }
            if exclude_exact_requests and exact_title_to_ids:
                continue
            if simple_artist_only and not simple_artist_only_request(text):
                continue
            directives[(sid, turn)] = {
                "constraints": constraints,
                "exact_title_to_ids": exact_title_to_ids,
                "user_text": text,
            }
    return directives


def exact_candidates_for_directive(directive: dict[str, Any], catalog) -> set[str]:
    constraints = directive["constraints"]
    exact_ids = {
        tid
        for ids in (directive.get("exact_title_to_ids") or {}).values()
        for tid in ids
    }
    return {
        tid for tid in exact_ids
        if any(track_satisfies_constraint(tid, c, catalog) for c in constraints)
    }


def promote_hard_constraints(
    predictions: list[dict[str, Any]],
    directives: dict[tuple[str, int], dict[str, Any]],
    catalog,
    *,
    k_search: int,
    prefer_exact: bool = False,
    skip_fallback_on_exact: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    directive_turns = 0
    present_turns = 0
    changed_turns = 0
    promoted_ids = 0
    rank_before: list[int] = []
    changed_preview: list[dict[str, Any]] = []

    for row in predictions:
        new_row = dict(row)
        key = (row["session_id"], int(row["turn_number"]))
        predicted = list(row.get("predicted_track_ids") or [])
        directive = directives.get(key)
        if directive:
            directive_turns += 1
            constraints = directive["constraints"]
            exact_ids = exact_candidates_for_directive(directive, catalog) if prefer_exact else set()
            best = next((tid for tid in predicted[:k_search] if tid in exact_ids), None)
            selection_reason = "exact_title" if best else "artist_constraint"
            if not best and not (skip_fallback_on_exact and exact_ids):
                best = next(
                    (
                        tid for tid in predicted[:k_search]
                        if any(track_satisfies_constraint(tid, c, catalog) for c in constraints)
                    ),
                    None,
                )
            if best:
                present_turns += 1
                before = predicted.index(best) + 1
                rank_before.append(before)
                promoted = [best] + [tid for tid in predicted if tid != best]
                new_row["predicted_track_ids"] = promoted[:len(predicted)]
                if promoted != predicted:
                    changed_turns += 1
                    promoted_ids += 1
                    if len(changed_preview) < 25:
                        changed_preview.append({
                            "session_id": key[0],
                            "turn_number": key[1],
                            "rank_before": before,
                            "promoted_track_id": best,
                            "constraints": constraints,
                            "selection_reason": selection_reason,
                        })
        out.append(new_row)

    def mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return out, {
        "directive_turns": directive_turns,
        "present_turns": present_turns,
        "changed_turns": changed_turns,
        "promoted_ids": promoted_ids,
        "mean_rank_before_when_present": mean(rank_before),
        "k_search": k_search,
        "prefer_exact": prefer_exact,
        "skip_fallback_on_exact": skip_fallback_on_exact,
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
    ap.add_argument("--no-strict", action="store_true",
                    help="Use broad hard-artist directives instead of the strict predicate.")
    ap.add_argument("--prefer-exact", action="store_true",
                    help="When the same turn resolves an exact title request, promote that title before artist-only matches.")
    ap.add_argument("--skip-fallback-on-exact", action="store_true",
                    help="When an exact title request is present but absent from the search window, do not promote a different same-artist item.")
    ap.add_argument("--exclude-exact-requests", action="store_true",
                    help="Skip hard-artist directives that contain a resolved exact-title request; use the exact-request family for those turns.")
    ap.add_argument("--simple-artist-only", action="store_true",
                    help="Abstain on quoted-title, album, era/year, lyric, exact-song, or example/reference cues.")
    args = ap.parse_args(argv)

    from datasets import load_dataset

    catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
    predictions = load_predictions(args.prediction_path)
    directives = hard_artist_directives(
        args.split,
        catalog,
        args.max_sessions,
        strict=not args.no_strict,
        exclude_exact_requests=args.exclude_exact_requests,
        simple_artist_only=args.simple_artist_only,
    )
    promoted, summary = promote_hard_constraints(
        predictions,
        directives,
        catalog,
        k_search=args.k_search,
        prefer_exact=args.prefer_exact,
        skip_fallback_on_exact=args.skip_fallback_on_exact,
    )
    summary.update({
        "prediction_path": str(args.prediction_path),
        "output": str(args.output),
        "split": args.split,
        "n_predictions": len(predictions),
        "n_directives": len(directives),
        "strict": not args.no_strict,
        "exclude_exact_requests": args.exclude_exact_requests,
        "simple_artist_only": args.simple_artist_only,
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(promoted, indent=2), encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
