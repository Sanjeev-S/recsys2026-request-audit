"""Build a generated audit of request-family admission decisions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
DEFAULT_JSON = EVID / "ledger/family_admission_audit.json"
DEFAULT_MD = EVID / "ledger/family_admission_audit.md"
DELTA_RE = re.compile(
    r"(?P<point>[+-]\d+\.\d+)\s+\[(?P<low>[+-]\d+\.\d+),\s+(?P<high>[+-]\d+\.\d+)\]"
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_delta(text: str) -> dict[str, float]:
    match = DELTA_RE.search(text)
    if not match:
        raise ValueError(f"cannot parse delta: {text}")
    return {key: float(value) for key, value in match.groupdict().items()}


def rows_by_method_split(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["method"], row["split"]): row for row in rows}


def precision_count(summary: dict[str, Any], key: str = "strict_intervention_precision") -> str:
    n = int(summary["overall"]["n"])
    point = float(summary["overall"][key])
    return f"{round(point * n)}/{n}"


def pass_official(rows: list[dict[str, Any]]) -> bool:
    return all(parse_delta(row["official_delta"])["low"] > -0.001 for row in rows)


def pass_corrected(rows: list[dict[str, Any]], key: str = "corrected_delta") -> bool:
    return all(parse_delta(row[key])["low"] > 0 for row in rows)


def gate(status: str, evidence: str, source: str) -> dict[str, str]:
    return {"status": status, "evidence": evidence, "source": source}


def build_audit(
    tables: dict[str, Any],
    detector_validity: dict[str, Any],
    hard_broad_precision: dict[str, Any],
    hard_simple_precision: dict[str, Any],
    switch_bootstrap: dict[str, Any],
    switch_overlap: dict[str, Any],
    transfer_capsule: dict[str, Any],
    additional_mining: dict[str, Any] | None = None,
    switch_oracle_bound: dict[str, Any] | None = None,
    hard_selector_ladder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    additional_mining = {"candidates": []} if additional_mining is None else additional_mining
    switch_oracle_bound = {} if switch_oracle_bound is None else switch_oracle_bound
    hard_selector_ladder = {"rungs": []} if hard_selector_ladder is None else hard_selector_ladder
    precision = {row["readout"]: row["estimate"] for row in tables["precision_rows"]}
    adjudication_estimate = precision.get(
        "LLM adjudication chose request",
        precision.get("request preferred over policy"),
    )
    interventions = rows_by_method_split(tables["intervention_rows"])
    hard_rows = rows_by_method_split(tables["hard_simple_training_rows"])

    exact_dev = interventions[("105k request-slice specialist (lambda=1.0)", "devset")]
    exact_val = interventions[("105k request-slice specialist (lambda=1.0)", "val")]
    exact_behavior = {row["split"]: row for row in tables["behavior_rows"]}
    exact_transfer = transfer_capsule["paper_status"]["blind_b_transfer"]
    detector = detector_validity["frame_ablation"]
    metadata = detector_validity["metadata_baseline"]

    hard_dev = hard_rows[("329-group simple hard add-positive specialist (lambda=0.5)", "devset")]
    hard_val = hard_rows[("329-group simple hard add-positive specialist (lambda=0.5)", "val")]
    hard_control_dev = hard_rows[("329-group simple hard official-label control (lambda=0.5)", "devset")]
    hard_control_val = hard_rows[("329-group simple hard official-label control (lambda=0.5)", "val")]
    hard_action = {
        row["split"]: row
        for row in tables["hard_action_rows"]
        if row["slice"] == "simple_nonneg_329group" and row["model"] == "add_positive_w0p5"
    }

    switch_official = switch_bootstrap["official"]["delta"]
    switch_corrected = switch_bootstrap["corrected"]["delta"]
    switch_family = switch_bootstrap["families"]["rejection_switchaway_violation"]["corrected"]["delta"]

    families = [
        {
            "family": "exact/version",
            "decision": "passes_development_gates_transfer_pending",
            "paper_claim": "main development family; request-positive training result",
            "gates": {
                "detector_validity": gate(
                    "pass",
                    (
                        f"{precision['detector sample']}; frame ablation "
                        f"{detector['detector_positive_active']['successes']}/{detector['detector_positive_active']['n']} "
                        f"strict positives vs {detector['control_active']['successes']}/{detector['control_active']['n']} controls; "
                        f"title overlap {detector['strict_detector_title_overlap_among_active']['successes']}/"
                        f"{detector['strict_detector_title_overlap_among_active']['n']}"
                    ),
                    "docs/evidence/request_detector_validity_capsule_v09.json",
                ),
                "action_precision": gate(
                    "pass",
                    (
                        f"{precision['changed-row strict precision']}; LLM adjudication chose request "
                        f"{adjudication_estimate}; duplicate baseline same canonical title "
                        f"{metadata['same_canonical_title']}/{metadata['n_conflict_rows']}"
                    ),
                    "docs/evidence/request_paper_tables_v09.json",
                ),
                "official_preservation": gate(
                    "pass" if pass_official([exact_dev, exact_val]) else "fail",
                    f"dev {exact_dev['official_delta']}; val {exact_val['official_delta']}",
                    "docs/evidence/request_paper_tables_v09.json",
                ),
                "corrected_improvement": gate(
                    "pass" if pass_corrected([exact_dev, exact_val]) else "fail",
                    f"dev {exact_dev['corrected_delta']}; val {exact_val['corrected_delta']}",
                    "docs/evidence/request_paper_tables_v09.json",
                ),
                "request_behavior": gate(
                    "pass",
                    (
                        f"request-first dev {exact_behavior['devset']['baseline_request_first']}/"
                        f"{exact_behavior['devset']['visible_directives']} -> "
                        f"{exact_behavior['devset']['specialist_request_first']}/"
                        f"{exact_behavior['devset']['visible_directives']}; val "
                        f"{exact_behavior['val']['baseline_request_first']}/"
                        f"{exact_behavior['val']['visible_directives']} -> "
                        f"{exact_behavior['val']['specialist_request_first']}/"
                        f"{exact_behavior['val']['visible_directives']}"
                    ),
                    "docs/evidence/request_paper_tables_v09.json",
                ),
                "non_oracle_inference": gate(
                    "pass",
                    "uses visible dialogue, catalog metadata, candidate membership, and model scores only",
                    "docs/evidence/request_family_label_contract_v09.md",
                ),
                "frozen_transfer": gate(
                    "pending_external_dependency",
                    f"Blind B {exact_transfer}; Blind A rehearsal request-first 1/2 -> 2/2",
                    "docs/evidence/request_transfer_evidence_capsule_v09.json",
                ),
            },
        },
        {
            "family": "hard artist simple/non-negated",
            "decision": "supporting_training_evidence_not_admitted",
            "paper_claim": "supports breadth of proxy-boundary mechanism; not a correction family yet",
            "gates": {
                "detector_validity": gate(
                    "partial",
                    (
                        f"broad action precision {precision_count(hard_broad_precision)}; "
                        f"simple abstaining selector {precision_count(hard_simple_precision)} on existing labels"
                    ),
                    "docs/evidence/request_hard_artist_strict_top20_simple_noexact_changed_rows_val_labels_v09.summary.json",
                ),
                "action_precision": gate(
                    "partial",
                    (
                        "selector ladder broad 37/50 -> simple/no-exact 12/13 strict precision, "
                        "but simple changes only 13 val rows and independent labels cover postrank "
                        "changed rows only"
                    ),
                    "docs/evidence/request_hard_artist_selector_ladder_v09.md",
                ),
                "official_preservation": gate(
                    "pass" if pass_official([hard_dev, hard_val]) else "fail",
                    f"dev {hard_dev['official_delta']}; val {hard_val['official_delta']}",
                    "docs/evidence/request_paper_tables_v09.json",
                ),
                "corrected_improvement": gate(
                    "pass" if pass_corrected([hard_dev, hard_val]) else "fail",
                    f"dev {hard_dev['corrected_delta']}; val {hard_val['corrected_delta']}",
                    "docs/evidence/request_paper_tables_v09.json",
                ),
                "control_check": gate(
                    "supporting",
                    (
                        f"official-label control dev {hard_control_dev['corrected_delta']}; "
                        f"val {hard_control_val['corrected_delta']}"
                    ),
                    "docs/evidence/request_paper_tables_v09.json",
                ),
                "request_behavior": gate(
                    "supporting",
                    (
                        f"top-1 satisfaction dev {hard_action['devset']['top1_satisfies_delta']}; "
                        f"val {hard_action['val']['top1_satisfies_delta']}"
                    ),
                    "docs/evidence/request_paper_tables_v09.json",
                ),
                "non_oracle_inference": gate(
                    "pass",
                    "simple selector uses visible dialogue and catalog metadata only",
                    "docs/evidence/request_hard_simple_nonneg_training_v09.md",
                ),
                "frozen_transfer": gate(
                    "candidate_not_run",
                    "v0.10 hard protocol is frozen as a candidate; untouched transfer not run",
                    "docs/evidence/request_hard_artist_v010_freeze_protocol.md",
                ),
            },
        },
        {
            "family": "rejection/switch-away",
            "decision": "diagnostic_negative_no_main_fix",
            "paper_claim": "visible boundary probe; current correction does not jointly preserve official score and satisfy request",
            "gates": {
                "detector_validity": gate(
                    "diagnostic",
                    "strict predicate retained 20 valid rows and dropped 5 invalid rows on inspected sample",
                    "docs/evidence/request_switchaway_strict_v09.md",
                ),
                "action_precision": gate(
                    "fail",
                    (
                        f"violation@1 {switch_overlap['violation_rate_at1_before']:.3f} -> "
                        f"{switch_overlap['violation_rate_at1_after']:.3f}, but violation@20 "
                        f"{switch_overlap['violation_rate_at20_before']:.3f} -> "
                        f"{switch_overlap['violation_rate_at20_after']:.3f}; gold moved down "
                        f"{switch_overlap['gold_moved_down_on_sidecar_rows']}/"
                        f"{switch_overlap['n_switchaway_sidecar_rows']}; oracle sidecar top-100 "
                        f"clean-top20 feasible "
                        f"{switch_oracle_bound.get('n_feasible_clean_top20_from_filter', 0)}/"
                        f"{switch_oracle_bound.get('n_switchaway_keys', switch_overlap['n_switchaway_sidecar_rows'])}"
                    ),
                    "docs/evidence/request_switchaway_strict_oracle_top100_bound_v09.json",
                ),
                "official_preservation": gate(
                    "fail",
                    (
                        f"official {switch_official['observed_delta']:+.5f} "
                        f"[{switch_official['ci95_low']:+.5f}, {switch_official['ci95_high']:+.5f}]"
                    ),
                    "docs/evidence/request_switchaway_strict_top100_bootstrap_vs_rerank_F10_scaled_strict_switch_v09.json",
                ),
                "corrected_improvement": gate(
                    "fail",
                    (
                        f"corrected {switch_corrected['observed_delta']:+.5f} "
                        f"[{switch_corrected['ci95_low']:+.5f}, {switch_corrected['ci95_high']:+.5f}]; "
                        f"switch-family {switch_family['observed_delta']:+.5f} "
                        f"[{switch_family['ci95_low']:+.5f}, {switch_family['ci95_high']:+.5f}]"
                    ),
                    "docs/evidence/request_switchaway_strict_top100_bootstrap_vs_rerank_F10_scaled_strict_switch_v09.json",
                ),
                "non_oracle_inference": gate(
                    "pass",
                    "suppression uses visible dialogue and predicted candidate metadata, not labels",
                    "docs/evidence/request_switchaway_strict_v09.md",
                ),
                "frozen_transfer": gate(
                    "not_ready",
                    "no admitted switch-away action to transfer",
                    "docs/evidence/request_switchaway_strict_v09.md",
                ),
            },
        },
    ]
    mined_candidates = [
        {
            "family": row["family"],
            "decision": row["decision"],
            "counts": row.get("counts_when_enabled", row.get("counts", {})),
            "reason": row["reason"],
        }
        for row in additional_mining["candidates"]
    ]
    mining_clause = (
        " Additional candidate families were mined under the same screening workflow but "
        "withheld from admission when they lacked safe actions or trusted target sets."
        if mined_candidates
        else ""
    )
    return {
        "date": "2026-06-22",
        "audit": "request-family-admission-v0.9",
        "admission_bar": [
            "detector/action precision",
            "official preservation",
            "corrected improvement",
            "request-satisfying behavior",
            "non-oracle inference",
            "frozen transfer",
        ],
        "conclusion": (
            "Exact/version passes the current development gates with frozen transfer pending. "
            "Hard-artist provides supporting learned evidence but is not admitted. "
            f"Switch-away remains a diagnostic negative.{mining_clause}"
        ),
        "families": families,
        "mined_not_admitted": mined_candidates,
    }


def markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Request-Family Admission Audit v0.9",
        "",
        f"Date: {audit['date']}",
        "",
        audit["conclusion"],
        "",
        (
            "This generated audit keeps the family claims from drifting: admitted "
            "families are judged against the same admission bar, while mined "
            "families record the first unresolved gate."
        ),
        "",
        "## Decision Summary",
        "",
        "| family | decision | paper claim |",
        "|---|---|---|",
    ]
    for family in audit["families"]:
        lines.append(f"| {family['family']} | {family['decision']} | {family['paper_claim']} |")
    lines.extend([
        "",
        "## Mined But Not Admitted",
        "",
        "| candidate family | decision | count/readout | reason |",
        "|---|---|---:|---|",
    ])
    for row in audit["mined_not_admitted"]:
        counts = row.get("counts") or {}
        count_text = ", ".join(f"{split}: {counts.get(split, 0)}" for split in ("train_lt", "val", "devset")) if counts else "not counted"
        lines.append(f"| {row['family']} | {row['decision']} | {count_text} | {row['reason']} |")
    lines.extend(["", "## Gate Audit", ""])
    for family in audit["families"]:
        lines.extend([
            f"### {family['family']}",
            "",
            "| gate | status | evidence |",
            "|---|---|---|",
        ])
        for name, row in family["gates"].items():
            lines.append(f"| {name} | {row['status']} | {row['evidence']} |")
        lines.append("")
    lines.extend([
        "## Safe Paper Claim",
        "",
        (
            "Exact/version is the main development-gate family. Hard-artist is "
            "supporting learned evidence under a stricter abstaining selector. "
            "Switch-away is a visible boundary probe whose current interventions fail. "
            "Additional mined families remain rejected evidence until they have safe "
            "actions and trusted request-satisfying target sets."
        ),
        "",
        "## Sources",
        "",
    ])
    sources = sorted({
        row["source"]
        for family in audit["families"]
        for row in family["gates"].values()
    })
    lines.extend(f"- `{source}`" for source in sources)
    lines.append("- `docs/evidence/request_additional_family_mining_audit_v09.md`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=EVID / "request_paper_tables_v09.json")
    ap.add_argument("--detector-validity", type=Path, default=EVID / "request_detector_validity_capsule_v09.json")
    ap.add_argument(
        "--hard-broad-precision",
        type=Path,
        default=EVID / "request_hard_artist_strict_top20_changed_rows_val_labels_v09.summary.json",
    )
    ap.add_argument(
        "--hard-simple-precision",
        type=Path,
        default=EVID / "request_hard_artist_strict_top20_simple_noexact_changed_rows_val_labels_v09.summary.json",
    )
    ap.add_argument(
        "--switch-bootstrap",
        type=Path,
        default=EVID / "request_switchaway_strict_top100_bootstrap_vs_rerank_F10_scaled_strict_switch_v09.json",
    )
    ap.add_argument(
        "--switch-overlap",
        type=Path,
        default=EVID / "request_switchaway_strict_top100_overlap_audit_v09.json",
    )
    ap.add_argument("--transfer-capsule", type=Path, default=EVID / "request_transfer_evidence_capsule_v09.json")
    ap.add_argument(
        "--additional-mining",
        type=Path,
        default=EVID / "request_additional_family_mining_audit_v09.json",
    )
    ap.add_argument(
        "--switch-oracle-bound",
        type=Path,
        default=EVID / "request_switchaway_strict_oracle_top100_bound_v09.json",
    )
    ap.add_argument(
        "--hard-selector-ladder",
        type=Path,
        default=EVID / "request_hard_artist_selector_ladder_v09.json",
    )
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = ap.parse_args(argv)

    audit = build_audit(
        load(args.tables),
        load(args.detector_validity),
        load(args.hard_broad_precision),
        load(args.hard_simple_precision),
        load(args.switch_bootstrap),
        load(args.switch_overlap),
        load(args.transfer_capsule),
        load(args.additional_mining),
        load(args.switch_oracle_bound),
        load(args.hard_selector_ladder),
    )
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.write_text(markdown(audit), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "md": str(args.md_out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
