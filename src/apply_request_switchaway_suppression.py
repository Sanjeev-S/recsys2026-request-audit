"""Demote visible switch-away violations without using policy labels.

This is an inference-time diagnostic for rejection/switch-away turns. It uses
only the visible dialogue, previous music in the session, catalog metadata, and
an existing prediction list:

  visible rejection/switch-away request -> demote previous track / rejected artist

It does not use gold labels or corrected-label sidecars to decide where to
intervene. Evaluation can still use sidecars to measure violation-rate impact.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("MCRS_EXPLORE_ROOT", str(Path(__file__).resolve().parent.parent)))

from request_correction_labels import (  # noqa: E402
    NEGATIVE_CUES,
    build_catalog_index,
    extract_by_artist,
    load_sessions,
    norm_text,
    norm_values,
    previous_music,
    strict_switchaway_request,
)


SOFT_SWITCH_RE = re.compile(
    r"\b(maybe|perhaps|possibly)\s+(?:from\s+|by\s+)?(?:a\s+)?(?:different|another|new)\s+artist\b"
)
STRONG_SWITCH_RE = re.compile(
    r"\b("
    r"different artists|other artists|other bands|different bands|"
    r"another artist entirely|completely different artist|"
    r"different artist this time|new artist this time|new artists|new artist|new bands|"
    r"new blood|branch out|broaden my horizons|move beyond|"
    r"no more|not more|not by"
    r")\b"
)
NEGATIVE_ARTIST_RE = re.compile(r"\b(not by|avoid|no more|don'?t want|do not want)\b", re.I)
REPEAT_CUES = ("not ", "don't want", "do not want", "skip")
STRICT_REPEAT_RE = re.compile(
    r"\b("
    r"(?:do not|don'?t)\s+(?:play|recommend|suggest|give me)\s+(?:this|that|the same)\s+(?:song|track|one)|"
    r"not\s+(?:this|that|the same)\s+(?:song|track|one)|"
    r"skip\s+(?:this|that|it)|"
    r"no\s+repeats?|"
    r"not\s+again"
    r")\b",
    re.I,
)


def strict_repeat_request(text: str) -> bool:
    return STRICT_REPEAT_RE.search(text) is not None


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


def switchaway_directives(
    split: str,
    catalog,
    max_sessions: int | None = None,
    *,
    strict: bool = False,
    action_scope: str = "all",
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
            text_norm = norm_text(text)
            if not any(cue in text_norm for cue in NEGATIVE_CUES):
                continue
            prev_id = previous_music(session["conversations"], turn)
            prev_row = catalog.by_id.get(prev_id) if prev_id else None
            if not prev_id or not prev_row:
                continue
            explicit_artist = extract_by_artist(text, catalog)
            prev_artists = sorted(set(norm_values(prev_row.get("artist_name"))))
            repeat = any(cue in text_norm for cue in REPEAT_CUES)
            switch_artist = (
                bool(STRONG_SWITCH_RE.search(text_norm))
                and not SOFT_SWITCH_RE.search(text_norm)
                and " or " not in text_norm
            )
            if strict and switch_artist:
                switch_artist = strict_switchaway_request(text)
            if strict:
                repeat = strict_repeat_request(text)
            reject_explicit_artist = bool(explicit_artist and NEGATIVE_ARTIST_RE.search(text))
            if action_scope == "explicit_or_repeat":
                switch_artist = False
            elif action_scope == "explicit_artist":
                switch_artist = False
                repeat = False
            elif action_scope == "repeat":
                switch_artist = False
                reject_explicit_artist = False
            elif action_scope == "switch_artist":
                repeat = False
                reject_explicit_artist = False
            if not (repeat or switch_artist or reject_explicit_artist):
                continue
            directives[(sid, turn)] = {
                "previous_track_id": prev_id,
                "previous_artist_norms": prev_artists,
                "explicit_artist_norm": explicit_artist,
                "repeat": repeat,
                "switch_artist": switch_artist,
                "reject_explicit_artist": reject_explicit_artist,
                "user_text": text,
            }
    return directives


def violates_switchaway(tid: str, directive: dict[str, Any], catalog) -> bool:
    if directive.get("repeat") and tid == directive.get("previous_track_id"):
        return True
    row = catalog.by_id.get(tid)
    if not row:
        return False
    artists = set(norm_values(row.get("artist_name")))
    if directive.get("switch_artist") and artists & set(directive.get("previous_artist_norms") or []):
        return True
    explicit = directive.get("explicit_artist_norm")
    return bool(directive.get("reject_explicit_artist") and explicit and explicit in artists)


def suppress_switchaway(
    predictions: list[dict[str, Any]],
    directives: dict[tuple[str, int], dict[str, Any]],
    catalog,
    *,
    k_search: int,
    mode: str = "demote",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    directive_turns = 0
    present_turns = 0
    changed_turns = 0
    demoted_ids = 0
    promoted_ids = 0
    changed_preview: list[dict[str, Any]] = []

    for row in predictions:
        new_row = dict(row)
        key = (row["session_id"], int(row["turn_number"]))
        predicted = list(row.get("predicted_track_ids") or [])
        directive = directives.get(key)
        if directive:
            directive_turns += 1
            if mode == "demote":
                demote = [
                    tid for tid in predicted[:k_search]
                    if violates_switchaway(tid, directive, catalog)
                ]
                if demote:
                    present_turns += 1
                    demote_set = set(demote)
                    reranked = [tid for tid in predicted if tid not in demote_set] + demote
                    new_row["predicted_track_ids"] = reranked[:len(predicted)]
                    if reranked != predicted:
                        changed_turns += 1
                        demoted_ids += len(demote)
                        if len(changed_preview) < 25:
                            changed_preview.append({
                                "session_id": key[0],
                                "turn_number": key[1],
                                "demoted_track_ids": demote,
                                "previous_track_id": directive["previous_track_id"],
                                "switch_artist": directive["switch_artist"],
                                "repeat": directive["repeat"],
                                "reject_explicit_artist": directive["reject_explicit_artist"],
                            })
            elif mode == "replace_top1":
                top1 = predicted[0] if predicted else None
                if top1 and violates_switchaway(top1, directive, catalog):
                    present_turns += 1
                    replacement = next(
                        (
                            tid for tid in predicted[:k_search]
                            if not violates_switchaway(tid, directive, catalog)
                        ),
                        None,
                    )
                    if replacement and replacement != top1:
                        reranked = [replacement] + [tid for tid in predicted if tid != replacement]
                        new_row["predicted_track_ids"] = reranked[:len(predicted)]
                        changed_turns += 1
                        promoted_ids += 1
                        if len(changed_preview) < 25:
                            changed_preview.append({
                                "session_id": key[0],
                                "turn_number": key[1],
                                "promoted_track_id": replacement,
                                "replaced_top1_track_id": top1,
                                "previous_track_id": directive["previous_track_id"],
                                "switch_artist": directive["switch_artist"],
                                "repeat": directive["repeat"],
                                "reject_explicit_artist": directive["reject_explicit_artist"],
                            })
            else:
                raise ValueError(f"unknown switchaway mode {mode!r}")
        out.append(new_row)

    return out, {
        "directive_turns": directive_turns,
        "present_turns": present_turns,
        "changed_turns": changed_turns,
        "demoted_ids": demoted_ids,
        "promoted_ids": promoted_ids,
        "k_search": k_search,
        "mode": mode,
        "changed_preview": changed_preview,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-path", type=Path, required=True)
    ap.add_argument("--split", choices=["devset", "val", "train_lt", "train", "train_full"], default="devset")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary-output", type=Path, required=True)
    ap.add_argument("--k-search", type=int, default=20)
    ap.add_argument("--max-sessions", type=int)
    ap.add_argument("--strict", action="store_true",
                    help="Use the high-precision switch-away predicate for artist-switch directives.")
    ap.add_argument("--action-scope",
                    choices=["all", "explicit_or_repeat", "explicit_artist", "repeat", "switch_artist"],
                    default="all",
                    help="Restrict the negative action family used for demotion.")
    ap.add_argument("--mode", choices=["demote", "replace_top1"], default="demote",
                    help="demote all visible violations in the search window, or replace only a violating top-1 with the first non-violating candidate.")
    args = ap.parse_args(argv)

    from datasets import load_dataset

    catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
    predictions = load_predictions(args.prediction_path)
    directives = switchaway_directives(
        args.split,
        catalog,
        args.max_sessions,
        strict=args.strict,
        action_scope=args.action_scope,
    )
    suppressed, summary = suppress_switchaway(
        predictions,
        directives,
        catalog,
        k_search=args.k_search,
        mode=args.mode,
    )
    summary.update({
        "prediction_path": str(args.prediction_path),
        "output": str(args.output),
        "split": args.split,
        "n_predictions": len(predictions),
        "n_directives": len(directives),
        "strict": args.strict,
        "action_scope": args.action_scope,
        "mode": args.mode,
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suppressed, indent=2), encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
