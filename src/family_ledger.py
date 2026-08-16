"""Build a family-search ledger for request-aware correction experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
DEFAULT_JSON = EVID / "ledger/family_search_ledger.json"
DEFAULT_MD = EVID / "ledger/family_search_ledger.md"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def family_row(family_audit: dict[str, Any], family: str) -> dict[str, Any]:
    for row in family_audit.get("families", []):
        if row["family"] == family:
            return row
    raise KeyError(family)


def mined_row(additional: dict[str, Any], family: str) -> dict[str, Any]:
    for row in additional.get("candidates", []):
        if row["family"] == family:
            return row
    raise KeyError(family)


def count_cell(counts: dict[str, int]) -> str:
    return f"train_lt: {counts.get('train_lt', 0)}, val: {counts.get('val', 0)}, devset: {counts.get('devset', 0)}"


def candidate_counts(row: dict[str, Any]) -> dict[str, int]:
    return row.get("counts") or row.get("counts_when_enabled") or {}


def build_ledger(
    *,
    family_audit: dict[str, Any],
    additional: dict[str, Any],
    broad_screening: dict[str, Any],
    trace_summary: dict[str, Any],
    selection_audit: dict[str, Any],
) -> dict[str, Any]:
    exact = family_row(family_audit, "exact/version")
    hard = family_row(family_audit, "hard artist simple/non-negated")
    switch = family_row(family_audit, "rejection/switch-away")
    album = mined_row(additional, "hard album constraint")
    year = mined_row(additional, "hard year/decade constraint")
    semantic = mined_row(additional, "genre, mood, instrumentation, language, popularity")
    screen_categories = broad_screening.get("categories", {})
    request_like = [
        int(row["visible_request_like_by_rubric"])
        for row in screen_categories.values()
    ]
    trusted_targets = [
        int(row["trusted_catalog_resolvable_under_current_contract"])
        for row in screen_categories.values()
    ]
    trace = trace_summary["overall"]

    rows = [
        {
            "order": 1,
            "family": "exact/version",
            "attempt": "visible exact-title and version/duplicate requests",
            "selection_stage": "main development family",
            "pre_or_post_readout": "developed on dev/val; full-train and Blind B readouts frozen separately",
            "decision": exact["decision"],
            "first_unresolved_gate": "frozen_transfer",
            "key_thresholds": [
                exact["gates"]["detector_validity"]["evidence"],
                exact["gates"]["action_precision"]["evidence"],
                exact["gates"]["official_preservation"]["evidence"],
                exact["gates"]["corrected_improvement"]["evidence"],
            ],
            "why_not_cherry_pick": "matched official-label, no-feature, wrong-positive, and cross-dialogue controls are recorded separately",
            "sources": [
                "docs/evidence/request_family_admission_audit_v09.json",
                "docs/evidence/request_training_claim_ledger_v09.json",
                "docs/evidence/request_selection_pressure_audit_v09.json",
            ],
        },
        {
            "order": 2,
            "family": "version/duplicate equivalent",
            "attempt": "same resolver as exact/version, but allowing catalog versions/duplicates",
            "selection_stage": "bundled low-prevalence subfamily",
            "pre_or_post_readout": "bundled with exact/version before final framing; not promoted as separate family",
            "decision": "proven_but_rare",
            "first_unresolved_gate": "coverage",
            "key_thresholds": [
                "devset has 1 version/duplicate record; train has 5 version records",
                "same label contract as exact/version",
            ],
            "why_not_cherry_pick": "reported as rare rather than inflated into a separate contribution",
            "sources": [
                "docs/evidence/request_goal_completion_audit_v09.json",
                "docs/evidence/request_exact_robustness_table_v09.json",
            ],
        },
        {
            "order": 3,
            "family": "hard artist broad",
            "attempt": "imperative named-artist constraints with broad parser",
            "selection_stage": "stress test before simple selector",
            "pre_or_post_readout": "broad failure analysis precedes the stricter simple/non-negated selector",
            "decision": "not_admitted_action_precision_weak",
            "first_unresolved_gate": "action_precision",
            "key_thresholds": [
                "broad val postranking strict precision 37/50",
                "learned broad hard specialist is useful stress evidence but action precision is weaker than exact/version",
            ],
            "why_not_cherry_pick": "broad failure is retained in the selector ladder and limits the hard-family claim",
            "sources": [
                "docs/evidence/request_hard_artist_selector_ladder_v09.json",
                "docs/evidence/request_family_admission_audit_v09.json",
            ],
        },
        {
            "order": 4,
            "family": "hard artist simple/non-negated",
            "attempt": "abstaining hard-artist selector excluding exact-title, quoted-title, album/year/era, lyric/example, and local-negation cases",
            "selection_stage": "supporting training evidence",
            "pre_or_post_readout": "post-broad-failure selector; frozen-transfer candidate only",
            "decision": hard["decision"],
            "first_unresolved_gate": "post_hoc_selector_and_transfer",
            "key_thresholds": [
                hard["gates"]["action_precision"]["evidence"],
                hard["gates"]["corrected_improvement"]["evidence"],
                hard["gates"]["request_behavior"]["evidence"],
            ],
            "why_not_cherry_pick": "reported as supporting, not headline, because selector precision and transfer are weaker than exact/version",
            "sources": [
                "docs/evidence/request_hard_artist_selector_ladder_v09.md",
                "docs/evidence/request_hard_simple_nonneg_training_v09.md",
                "docs/evidence/request_family_admission_audit_v09.json",
            ],
        },
        {
            "order": 5,
            "family": "rejection/switch-away",
            "attempt": "strict requests to move away from a rejected item/source or avoid a repeated previous item",
            "selection_stage": "negative diagnostic",
            "pre_or_post_readout": "tested as conservative masking/suppression; never promoted to headline",
            "decision": switch["decision"],
            "first_unresolved_gate": "useful_action_and_official_preservation",
            "key_thresholds": [
                switch["gates"]["action_precision"]["evidence"],
                switch["gates"]["official_preservation"]["evidence"],
                switch["gates"]["corrected_improvement"]["evidence"],
            ],
            "why_not_cherry_pick": "negative result is retained in the paper to show the contract rejects visible but unsafe actions",
            "sources": [
                "docs/evidence/request_switchaway_strict_v09.md",
                "docs/evidence/request_switchaway_oracle_bound_v09.md",
                "docs/evidence/request_family_admission_audit_v09.json",
            ],
        },
        {
            "order": 6,
            "family": "hard album constraint",
            "attempt": "album-name constraints mined from the same sidecar workflow",
            "selection_stage": "mined but not admitted",
            "pre_or_post_readout": "additional-family mining after initial exact/hard/switch detectors",
            "decision": album["decision"],
            "first_unresolved_gate": "action_safety_and_dev_coverage",
            "key_thresholds": [
                count_cell(candidate_counts(album)),
                album["reason"],
            ],
            "why_not_cherry_pick": "literal salvage subset is disclosed but not promoted because dev coverage is 0",
            "sources": ["docs/evidence/request_additional_family_mining_audit_v09.json"],
        },
        {
            "order": 7,
            "family": "hard year/decade constraint",
            "attempt": "release-year and decade constraints enabled in mining sidecar",
            "selection_stage": "mined but not admitted",
            "pre_or_post_readout": "additional-family mining; not used for training claims",
            "decision": year["decision"],
            "first_unresolved_gate": "resolver_semantics",
            "key_thresholds": [
                count_cell(candidate_counts(year)),
                year["reason"],
            ],
            "why_not_cherry_pick": "broad coverage is reported, but unsafe range/date semantics block admission",
            "sources": ["docs/evidence/request_additional_family_mining_audit_v09.json"],
        },
        {
            "order": 8,
            "family": "genre/mood/instrumentation/language/popularity",
            "attempt": "high-recall broad semantic request-language mining",
            "selection_stage": "mined but not admitted",
            "pre_or_post_readout": "additional-family mining; not converted to labels",
            "decision": semantic["decision"],
            "first_unresolved_gate": "no_trusted_catalog_target_set",
            "key_thresholds": [
                count_cell(candidate_counts(semantic)),
                f"screening request-like {min(request_like)}-{max(request_like)}/25 per category; trusted target sets {sum(trusted_targets)}/125",
                semantic["reason"],
            ],
            "why_not_cherry_pick": "frequent families are retained as rejected evidence because catalog fields cannot certify satisfaction",
            "sources": [
                "docs/evidence/request_additional_family_mining_audit_v09.json",
                "docs/evidence/request_broad_semantic_screening_audit_v09.json",
            ],
        },
        {
            "order": 9,
            "family": "synthetic thought traces",
            "attempt": "inspect generator traces for policy/request boundary language",
            "selection_stage": "audit-only framing evidence",
            "pre_or_post_readout": "offline qualitative audit; forbidden as detector or inference input",
            "decision": "audit_only_not_labels",
            "first_unresolved_gate": "not_visible_user_input",
            "key_thresholds": [
                f"trace mentions policy title {trace['trace_mentions_policy_title']['count']}/{trace['trace_mentions_policy_title']['n']}",
                f"trace request-corroborated {trace['trace_request_corroborated']['count']}/{trace['trace_request_corroborated']['n']}",
            ],
            "why_not_cherry_pick": "traces support vocabulary only; deployable detectors remain visible-dialogue/catalog based",
            "sources": [
                "docs/evidence/request_trace_corroboration_exact_version_v09.summary.json",
                "docs/evidence/request_trace_boundary_examples_v09.json",
                "docs/evidence/request_nonoracle_audit_v09.json",
            ],
        },
    ]

    return {
        "artifact": "request-family-search-ledger-v0.9",
        "date": "2026-06-23",
        "purpose": (
            "Make family search and rejection decisions auditable so exact/version "
            "does not look like a cherry-picked survivor."
        ),
        "read": (
            "The ledger records every request-family class currently used in the "
            "paper package, including negative and mined-but-not-admitted cases. "
            "It is an audit-order ledger reconstructed from dated evidence artifacts, "
            "not a claim of perfect wall-clock experiment chronology."
        ),
        "selection_pressure_verdict": selection_audit.get("verdict"),
        "rows": rows,
        "source_artifacts": [
            "docs/evidence/request_family_admission_audit_v09.json",
            "docs/evidence/request_additional_family_mining_audit_v09.json",
            "docs/evidence/request_broad_semantic_screening_audit_v09.json",
            "docs/evidence/request_trace_corroboration_exact_version_v09.summary.json",
            "docs/evidence/request_selection_pressure_audit_v09.json",
        ],
    }


def markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "# Request Family-Search Ledger v0.9",
        "",
        f"Date: {ledger['date']}",
        "",
        ledger["purpose"],
        "",
        f"Selection-pressure verdict: `{ledger['selection_pressure_verdict']}`.",
        "",
        ledger["read"],
        "",
        "## Family Ledger",
        "",
        "| order | family | selection stage | decision | first unresolved gate | pre/post readout status |",
        "|---:|---|---|---|---|---|",
    ]
    for row in ledger["rows"]:
        lines.append(
            "| "
            f"{row['order']} | {row['family']} | {row['selection_stage']} | "
            f"`{row['decision']}` | `{row['first_unresolved_gate']}` | "
            f"{row['pre_or_post_readout']} |"
        )

    lines.extend([
        "",
        "## Gate Details",
        "",
    ])
    for row in ledger["rows"]:
        lines.extend([
            f"### {row['order']}. {row['family']}",
            "",
            f"- Attempt: {row['attempt']}",
            f"- Why this reduces cherry-pick risk: {row['why_not_cherry_pick']}",
            "- Key thresholds/readouts:",
        ])
        lines.extend(f"  - {item}" for item in row["key_thresholds"])
        lines.extend([
            "- Sources:",
        ])
        lines.extend(f"  - `{source}`" for source in row["sources"])
        lines.append("")

    lines.extend([
        "## Sources",
        "",
    ])
    lines.extend(f"- `{source}`" for source in ledger["source_artifacts"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-audit", type=Path, default=EVID / "ledger/family_admission_audit.json")
    ap.add_argument("--additional", type=Path, default=EVID / "request_additional_family_mining_audit_v09.json")
    ap.add_argument("--broad-screening", type=Path, default=EVID / "request_broad_semantic_screening_audit_v09.json")
    ap.add_argument("--trace-summary", type=Path, default=EVID / "request_trace_corroboration_exact_version_v09.summary.json")
    ap.add_argument("--selection-audit", type=Path, default=EVID / "internal/selection_pressure_audit.json")
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = ap.parse_args(argv)

    ledger = build_ledger(
        family_audit=load(args.family_audit),
        additional=load(args.additional),
        broad_screening=load(args.broad_screening),
        trace_summary=load(args.trace_summary),
        selection_audit=load(args.selection_audit),
    )
    args.json_out.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.write_text(markdown(ledger), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "md": str(args.md_out), "rows": len(ledger["rows"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
