#!/usr/bin/env python3
"""Reproduce paper Table 1: per-turn requested-track ranks, control vs specialist.

Table 1 follows all 82 audited directives (Section 5 funnel) under the matched
control and the specialist, both deployed behind the inference-time dialogue
gate, and buckets the requested track's rank into {1, 2-20, absent}, split by
whether the official label agrees with the request (coverage gold_status).

Two ways to run it:

1. From the shipped evidence (default) — aggregates
   evidence/results/table1_ranks.idkeyed.jsonl and verifies every
   printed Table 1 number (six cells, rank-1 counts 66/46, pairwise 35
   higher / 0 lower, the single specialist-absent row).

2. From gated prediction files (--control-predictions/--specialist-predictions,
   regenerated via src/run_gated_eval.py from the shipped model
   dumps) — recomputes each directive's rank from scratch, verifies the result
   against the shipped per-turn file, and rewrites it with --write-per-turn.

A directive's rank under a model is the position (1-based) of the first
requested track id within the model's top-20 for that turn; None means no
requested track appears in the top-20 ("absent" in the paper). Requested track
ids per directive come from the funnel coverage rows.

The one specialist-absent directive is a detector false fire (paper Section 7):
its quoted phrase names a listening vibe, not a catalog track. It is marked in
the per-turn file; the phrase itself is dialogue text and is not redistributed.

Pool-membership note (Table 1 caption): both models rank the same per-turn
candidate pools, and a gated model can only place a track it scored. Every
conflict-half track the specialist recovers (18 absent -> 1) therefore sat in
the candidate pool for the control as well - the control's misses are ranking
failures, not retrieval failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"

DEFAULT_COVERAGE = EVID / "audit/coverage_devset.json"
DEFAULT_PER_TURN = EVID / "results/table1_ranks.idkeyed.jsonl"
DEFAULT_OUTPUT = ROOT / "work/table1_rank_readout.json"

# Frozen classification from the 2026-07-24 readout walkthrough: the single
# directive whose requested phrase is a listening vibe, not a catalog track.
FALSE_FIRE = {"session_id": "f86d58b6-8005-4713-9900-50c1e7558126", "turn_number": 4}

# The published Table 1 (camera-ready v11), used as the verification target.
PAPER_TABLE = {
    "label_agrees": {"control": {"rank_1": 33, "rank_2_20": 8, "absent": 0},
                     "specialist": {"rank_1": 41, "rank_2_20": 0, "absent": 0}},
    "label_conflicts": {"control": {"rank_1": 13, "rank_2_20": 10, "absent": 18},
                        "specialist": {"rank_1": 25, "rank_2_20": 15, "absent": 1}},
    "rank_first_counts": {"control": 46, "specialist": 66},
    "pairwise": {"specialist_higher": 35, "specialist_lower": 0},
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rank_in_top20(predicted: list[str], requested: set[str]) -> int | None:
    for i, track in enumerate(predicted[:20], 1):
        if track in requested:
            return i
    return None


def prediction_map(path: Path) -> dict[tuple[str, int], list[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(r["session_id"], r["turn_number"]): r["predicted_track_ids"] for r in rows}


def per_turn_from_predictions(coverage_rows: list[dict], control_path: Path,
                              specialist_path: Path, base_path: Path | None) -> list[dict]:
    control = prediction_map(control_path)
    specialist = prediction_map(specialist_path)
    base = prediction_map(base_path) if base_path else None
    out = []
    for cr in coverage_rows:
        key = (cr["session_id"], cr["turn_number"])
        requested = set(cr["requested_track_ids"])
        row = {
            "session_id": cr["session_id"],
            "turn_number": cr["turn_number"],
            "gold_status": cr["gold_status"],
            "gold_track_id": cr["gold_track_id"],
            "requested_track_ids": cr["requested_track_ids"],
            "control_rank": rank_in_top20(control[key], requested),
            "specialist_rank": rank_in_top20(specialist[key], requested),
        }
        if base is not None:
            row["base_rank"] = rank_in_top20(base[key], requested)
        if (cr["session_id"], cr["turn_number"]) == (FALSE_FIRE["session_id"], FALSE_FIRE["turn_number"]):
            row["detector_false_fire"] = True
        out.append(row)
    return out


def bucket(rank: int | None) -> str:
    if rank is None:
        return "absent"
    return "rank_1" if rank == 1 else "rank_2_20"


def aggregate(rows: list[dict]) -> dict:
    halves = {"gold_requested": "label_agrees", "gold_differs": "label_conflicts"}
    table = {h: {m: {"rank_1": 0, "rank_2_20": 0, "absent": 0}
                 for m in ("control", "specialist")} for h in halves.values()}
    rank_first = {"control": 0, "specialist": 0}
    higher = lower = 0
    absent_spec = []
    for r in rows:
        half = halves[r["gold_status"]]
        for model, field in (("control", "control_rank"), ("specialist", "specialist_rank")):
            table[half][model][bucket(r[field])] += 1
            if r[field] == 1:
                rank_first[model] += 1
        c = r["control_rank"] if r["control_rank"] is not None else 10 ** 9
        s = r["specialist_rank"] if r["specialist_rank"] is not None else 10 ** 9
        if s < c:
            higher += 1
        elif s > c:
            lower += 1
        if r["specialist_rank"] is None:
            absent_spec.append({"session_id": r["session_id"], "turn_number": r["turn_number"],
                                "detector_false_fire": bool(r.get("detector_false_fire"))})
    return {
        "n_directives": len(rows),
        "label_agrees": table["label_agrees"],
        "label_conflicts": table["label_conflicts"],
        "rank_first_counts": rank_first,
        "pairwise": {"specialist_higher": higher, "specialist_lower": lower,
                     "unchanged": len(rows) - higher - lower},
        "specialist_absent_rows": absent_spec,
    }


def verify(agg: dict) -> list[str]:
    errors = []
    for half in ("label_agrees", "label_conflicts"):
        for model in ("control", "specialist"):
            if agg[half][model] != PAPER_TABLE[half][model]:
                errors.append(f"{half}/{model}: {agg[half][model]} != {PAPER_TABLE[half][model]}")
    if agg["rank_first_counts"] != PAPER_TABLE["rank_first_counts"]:
        errors.append(f"rank_first_counts: {agg['rank_first_counts']}")
    pw = {k: agg["pairwise"][k] for k in ("specialist_higher", "specialist_lower")}
    if pw != PAPER_TABLE["pairwise"]:
        errors.append(f"pairwise: {pw}")
    absent = agg["specialist_absent_rows"]
    if len(absent) != 1 or not absent[0]["detector_false_fire"]:
        errors.append(f"specialist_absent_rows: {absent}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    ap.add_argument("--per-turn", type=Path, default=DEFAULT_PER_TURN,
                    help="shipped per-turn ranks (read, or written with --write-per-turn)")
    ap.add_argument("--control-predictions", type=Path,
                    help="gated control prediction file; triggers recomputation")
    ap.add_argument("--specialist-predictions", type=Path)
    ap.add_argument("--base-predictions", type=Path,
                    help="optional ungated base ranking, adds base_rank context")
    ap.add_argument("--write-per-turn", action="store_true",
                    help="write the recomputed rows to --per-turn")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the assertion against the published Table 1")
    args = ap.parse_args()

    coverage_rows = json.loads(args.coverage.read_text(encoding="utf-8"))["rows"]

    if args.control_predictions or args.specialist_predictions:
        if not (args.control_predictions and args.specialist_predictions):
            ap.error("--control-predictions and --specialist-predictions go together")
        rows = per_turn_from_predictions(coverage_rows, args.control_predictions,
                                         args.specialist_predictions, args.base_predictions)
        if args.per_turn.exists():
            shipped = load_jsonl(args.per_turn)
            keyed = {(r["session_id"], r["turn_number"]): r for r in shipped}
            diffs = []
            for r in rows:
                s = keyed.get((r["session_id"], r["turn_number"]))
                if s is None:
                    diffs.append(f"{r['session_id']}/{r['turn_number']}: missing from shipped file")
                    continue
                for f in ("gold_status", "control_rank", "specialist_rank"):
                    if r[f] != s.get(f):
                        diffs.append(f"{r['session_id']}/{r['turn_number']}: {f} {r[f]} != shipped {s.get(f)}")
            if diffs:
                print(f"RECOMPUTATION DIFFERS from {args.per_turn} ({len(diffs)}):")
                for d in diffs[:20]:
                    print(" ", d)
                sys.exit(1)
            print(f"recomputed ranks match {args.per_turn} on all {len(rows)} directives")
        if args.write_per_turn:
            args.per_turn.parent.mkdir(parents=True, exist_ok=True)
            with args.per_turn.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            print(f"wrote {args.per_turn}")
    else:
        rows = load_jsonl(args.per_turn)

    agg = aggregate(rows)
    agg["provenance"] = {
        "per_turn_source": str(args.per_turn.relative_to(ROOT)) if args.per_turn.is_relative_to(ROOT) else str(args.per_turn),
        "coverage_source": str(args.coverage.relative_to(ROOT)) if args.coverage.is_relative_to(ROOT) else str(args.coverage),
        "rank_definition": "position of the first requested track id in the model's top-20; absent = none in top-20",
        "gate": "both models deployed behind the inference-time dialogue gate (Section 6)",
    }

    if not args.no_verify:
        errors = verify(agg)
        if errors:
            print("TABLE 1 VERIFICATION FAILED:")
            for e in errors:
                print(" ", e)
            sys.exit(1)
        agg["verified_against_paper_table"] = True
        print("aggregate matches the published Table 1 "
              f"(66/46 rank-1, +{agg['pairwise']['specialist_higher']}/-{agg['pairwise']['specialist_lower']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
