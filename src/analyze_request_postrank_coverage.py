"""Analyze exact-request postrank coverage and failure modes.

This is a paper-facing diagnostic for the locked request-aware postranker. It
does not use policy labels to decide interventions; labels are only summarized
afterwards to show how often the policy-selected item already satisfies the
visible exact request.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("MCRS_EXPLORE_ROOT", str(Path(__file__).resolve().parent.parent)))

from apply_request_exact_postrank import load_predictions, request_directives  # noqa: E402
from request_correction_labels import (  # noqa: E402
    build_catalog_index,
    gold_for_turn,
    load_sessions,
    metadata_summary,
)


def prediction_map(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[str]]:
    out: dict[tuple[str, int], list[str]] = {}
    for row in rows:
        out[(row["session_id"], int(row["turn_number"]))] = list(row.get("predicted_track_ids") or [])
    return out


def gold_map(split: str, max_sessions: int | None = None) -> dict[tuple[str, int], str]:
    sessions = load_sessions(split)
    if max_sessions is not None:
        sessions = sessions.select(range(min(max_sessions, len(sessions))))
    out: dict[tuple[str, int], str] = {}
    for session in sessions:
        sid = session["session_id"]
        for turn in range(1, 9):
            gold = gold_for_turn(session["conversations"], turn)
            if gold:
                out[(sid, turn)] = gold
    return out


def rank_of_any(predicted: list[str], ids: set[str]) -> int | None:
    for rank, tid in enumerate(predicted, start=1):
        if tid in ids:
            return rank
    return None


def title_ranks(predicted: list[str], title_to_ids: dict[str, list[str]]) -> dict[str, int | None]:
    return {
        title: rank_of_any(predicted, set(ids))
        for title, ids in sorted(title_to_ids.items())
    }


def intervention_for_row(
    predicted: list[str],
    title_to_ids: dict[str, list[str]],
    *,
    k_search: int,
) -> dict[str, Any]:
    present: list[str] = []
    ranks = title_ranks(predicted, title_to_ids)
    for title in sorted(title_to_ids):
        ids = set(title_to_ids[title])
        best = next((tid for tid in predicted[:k_search] if tid in ids), None)
        if best and best not in present:
            present.append(best)

    if not present:
        return {
            "category": "not_retrieved_topk",
            "changed": False,
            "present": False,
            "best_rank": None,
            "title_ranks": ranks,
            "promoted_track_ids": [],
        }

    present_set = set(present)
    promoted = present + [tid for tid in predicted if tid not in present_set]
    changed = promoted != predicted
    return {
        "category": "promoted" if changed else "already_front",
        "changed": changed,
        "present": True,
        "best_rank": min(predicted.index(tid) + 1 for tid in present),
        "title_ranks": ranks,
        "promoted_track_ids": present,
    }


def rank_bucket(rank: int | None, *, k_search: int) -> str:
    if rank is None or rank > k_search:
        return f">{k_search}/missing"
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 20:
        return "6-20"
    return f"21-{k_search}"


def pct(num: int, den: int) -> float:
    return num / den if den else 0.0


def analyze(
    *,
    prediction_path: Path,
    split: str,
    k_search: int,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    from datasets import load_dataset

    catalog = build_catalog_index(load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"))
    preds = prediction_map(load_predictions(prediction_path))
    directives = request_directives(split, catalog, max_sessions=max_sessions)
    golds = gold_map(split, max_sessions=max_sessions)

    category_counts: Counter[str] = Counter()
    rank_counts: Counter[str] = Counter()
    gold_counts: Counter[str] = Counter()
    title_count_counts: Counter[int] = Counter()
    preview: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for key, directive in sorted(directives.items()):
        predicted = preds.get(key)
        if predicted is None:
            raise ValueError(f"missing prediction for directive key {key}")
        title_to_ids = directive.get("requested_title_to_ids") or {
            "__all__": directive["requested_track_ids"]
        }
        result = intervention_for_row(predicted, title_to_ids, k_search=k_search)
        category = result["category"]
        category_counts[category] += 1
        rank_counts[rank_bucket(result["best_rank"], k_search=k_search)] += 1
        title_count_counts[len(title_to_ids)] += 1

        requested_ids = set(directive["requested_track_ids"])
        gold = golds.get(key)
        if gold in requested_ids:
            gold_status = "gold_requested"
        elif gold:
            gold_status = "gold_differs"
        else:
            gold_status = "gold_missing"
        gold_counts[gold_status] += 1

        row = {
            "session_id": key[0],
            "turn_number": key[1],
            "category": category,
            "best_rank": result["best_rank"],
            "requested_titles": directive["requested_titles"],
            "requested_track_ids": directive["requested_track_ids"],
            "promoted_track_ids": result["promoted_track_ids"],
            "gold_track_id": gold,
            "gold_status": gold_status,
            "title_ranks": result["title_ranks"],
            "user_text": directive["user_text"],
        }
        rows.append(row)

        if len(preview[category]) < 8:
            row_preview = dict(row)
            row_preview["gold_track"] = metadata_summary(catalog.by_id.get(gold))
            row_preview["requested_tracks"] = [
                metadata_summary(catalog.by_id.get(tid))
                for tid in directive["requested_track_ids"][:5]
            ]
            preview[category].append(row_preview)

    n = len(rows)
    return {
        "prediction_path": str(prediction_path),
        "split": split,
        "k_search": k_search,
        "max_sessions": max_sessions,
        "n_directives": n,
        "category_counts": dict(sorted(category_counts.items())),
        "category_rates": {k: pct(v, n) for k, v in sorted(category_counts.items())},
        "rank_bucket_counts": dict(sorted(rank_counts.items())),
        "rank_bucket_rates": {k: pct(v, n) for k, v in sorted(rank_counts.items())},
        "gold_status_counts": dict(sorted(gold_counts.items())),
        "gold_status_rates": {k: pct(v, n) for k, v in sorted(gold_counts.items())},
        "requested_title_count_distribution": {
            str(k): v for k, v in sorted(title_count_counts.items())
        },
        "previews": dict(preview),
        "rows": rows,
    }


def fmt_rate(count: int, n: int) -> str:
    return f"{count}/{n} ({pct(count, n):.1%})"


def markdown(summary: dict[str, Any]) -> str:
    n = summary["n_directives"]
    lines = [
        f"# Request Postrank Coverage: {summary['split']}",
        "",
        "Date: 2026-06-21",
        "",
        "This analyzes the locked exact-request postranker before intervention. It uses visible dialogue, catalog metadata, and the baseline prediction list. Policy labels are summarized only after the request frame is resolved.",
        "",
        f"Prediction file: `{summary['prediction_path']}`",
        f"k-search: `{summary['k_search']}`",
        "",
        "## Funnel",
        "",
        "| readout | count |",
        "|---|---:|",
        f"| exact-request directive turns | {n} |",
    ]
    for key in ("promoted", "already_front", "not_retrieved_topk"):
        count = summary["category_counts"].get(key, 0)
        label = {
            "promoted": "changed by postranker",
            "already_front": "already request-satisfying at front",
            "not_retrieved_topk": f"requested item absent from top-{summary['k_search']}",
        }[key]
        lines.append(f"| {label} | {fmt_rate(count, n)} |")

    lines.extend([
        "",
        "## Rank Of Best Requested Item",
        "",
        "| rank bucket | count |",
        "|---|---:|",
    ])
    for key in ("1", "2-5", "6-20", f"21-{summary['k_search']}", f">{summary['k_search']}/missing"):
        count = summary["rank_bucket_counts"].get(key, 0)
        lines.append(f"| {key} | {fmt_rate(count, n)} |")

    lines.extend([
        "",
        "## Policy Label Relation",
        "",
        "| relation | count |",
        "|---|---:|",
    ])
    for key in ("gold_requested", "gold_differs", "gold_missing"):
        count = summary["gold_status_counts"].get(key, 0)
        lines.append(f"| {key} | {fmt_rate(count, n)} |")

    lines.extend([
        "",
        "## Read",
        "",
        "- `promoted` is the reachable fix slice: the requested item is already in top-k but not at the front.",
        "- `already_front` is not a failure: the existing system already puts a request-satisfying item first.",
        "- `not_retrieved_topk` is the retrieval ceiling for this postranker.",
        "- `gold_differs` is the observable proxy-boundary slice: the policy-selected label differs from the visible request-satisfying item.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-path", type=Path, required=True)
    ap.add_argument("--split", choices=["devset", "val", "train_lt"], required=True)
    ap.add_argument("--k-search", type=int, default=100)
    ap.add_argument("--max-sessions", type=int)
    ap.add_argument("--json-output", type=Path, required=True)
    ap.add_argument("--markdown-output", type=Path, required=True)
    args = ap.parse_args(argv)

    summary = analyze(
        prediction_path=args.prediction_path,
        split=args.split,
        k_search=args.k_search,
        max_sessions=args.max_sessions,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({
        "json_output": str(args.json_output),
        "markdown_output": str(args.markdown_output),
        "n_directives": summary["n_directives"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
