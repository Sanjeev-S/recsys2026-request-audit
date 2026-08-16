"""Build paper-ready request-correction tables from evidence JSON artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def delta_cell(delta: dict[str, Any]) -> str:
    return (
        f"{delta['observed_delta']:+.5f} "
        f"[{delta['ci95_low']:+.5f}, {delta['ci95_high']:+.5f}]"
    )


def ci_cell(value: float, ci: list[float]) -> str:
    return f"{value:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]"


def wilson_cell(rate: dict[str, Any]) -> str:
    return f"{rate['successes']}/{rate['n']} ({rate['point']:.3f} [{rate['low']:.3f}, {rate['high']:.3f}])"


def p_cell(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def count_rate_cell(count: int, n: int) -> str:
    return f"{count}/{n} ({(count / n if n else 0.0):.1%})"


def count_cell(value: int) -> str:
    return f"{value:,}"


def bootstrap_row(name: str, split: str, artifact: dict[str, Any]) -> dict[str, str]:
    exact_family = artifact["families"]["exact_track_request"]
    return {
        "method": name,
        "split": split,
        "official_delta": delta_cell(artifact["official"]["delta"]),
        "corrected_delta": delta_cell(artifact["corrected"]["delta"]),
        "exact_slice_official_delta": delta_cell(exact_family["official"]["delta"]),
        "exact_slice_corrected_delta": delta_cell(exact_family["corrected"]["delta"]),
    }


def bootstrap_row_for_family(name: str, split: str, artifact: dict[str, Any], family: str, family_label: str) -> dict[str, str]:
    family_delta = artifact["families"][family]["corrected"]["delta"]
    return {
        "method": name,
        "split": split,
        "official_delta": delta_cell(artifact["official"]["delta"]),
        "corrected_delta": delta_cell(artifact["corrected"]["delta"]),
        family_label: delta_cell(family_delta),
    }


def overall_bootstrap_row(name: str, split: str, artifact: dict[str, Any]) -> dict[str, str]:
    hard = artifact["families"].get("hard_artist_constraint", {})
    hard_delta = hard.get("corrected", {}).get("delta")
    return {
        "system": name,
        "split": split,
        "official_delta": delta_cell(artifact["official"]["delta"]),
        "corrected_delta": delta_cell(artifact["corrected"]["delta"]),
        "hard_artist_corrected_delta": delta_cell(hard_delta) if hard_delta else "n/a",
    }


def rows_to_markdown(headers: list[str], rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return lines


def build_tables() -> dict[str, Any]:
    locked = load(EVID / "request_exact_postrank_locked_table_v09.json")
    fullpool = {
        "devset": load(EVID / "request_exact_fullpool_bootstrap_official_vs_fullpool_dev_exact_version_v09.json"),
        "val": load(EVID / "request_exact_fullpool_bootstrap_official_vs_fullpool_val_exact_version_v09.json"),
    }
    specialist = {
        "devset": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_plain_blend_dev_exact_version_v09_w1.json"),
        "val": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_plain_blend_val_exact_version_v09_w1.json"),
    }
    specialist_w075 = {
        "devset": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_plain_blend_dev_exact_version_v09_w0p75.json"),
        "val": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_plain_blend_val_exact_version_v09_w0p75.json"),
    }
    official_control = {
        "devset": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_official_control_blend_dev_exact_version_v09_w1.json"),
        "val": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_official_control_blend_val_exact_version_v09_w1.json"),
    }
    nofeature_control = {
        "devset": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_nofeature_blend_dev_exact_version_v09_w1.json"),
        "val": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_nofeature_blend_val_exact_version_v09_w1.json"),
    }
    wrongpos_control = {
        "devset": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_wrongpos_reqfeat_blend_dev_exact_version_v09_w1.json"),
        "val": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_wrongpos_reqfeat_blend_val_exact_version_v09_w1.json"),
    }
    crossdialogue_control = {
        "devset": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_crossdialogue_reqfeat_blend_dev_exact_version_v09_w1.json"),
        "val": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_crossdialogue_reqfeat_blend_val_exact_version_v09_w1.json"),
    }
    hard_positive = {
        "devset": load(EVID / "request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_positive_dev_v09_w0p3.json"),
        "val": load(EVID / "request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_positive_val_v09_w0p3.json"),
    }
    hard_masked = {
        "devset": load(EVID / "request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_masked_dev_v09_w0p1.json"),
        "val": load(EVID / "request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_masked_val_v09_w0p1.json"),
    }
    hard_official_control = {
        "devset": load(EVID / "request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_official_control_dev_v09_w0p3.json"),
        "val": load(EVID / "request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_official_control_val_v09_w0p3.json"),
    }
    hard_simple_positive = {
        "devset": load(EVID / "request_hard_simple_nonneg_bootstrap_official_vs_slice105k_hardfeat_positive_dev_v09_w0p5.json"),
        "val": load(EVID / "request_hard_simple_nonneg_bootstrap_official_vs_slice105k_hardfeat_positive_val_v09_w0p5.json"),
    }
    hard_simple_official_control = {
        "devset": load(EVID / "request_hard_simple_nonneg_bootstrap_official_vs_slice105k_hardfeat_official_control_dev_v09_w0p5.json"),
        "val": load(EVID / "request_hard_simple_nonneg_bootstrap_official_vs_slice105k_hardfeat_official_control_val_v09_w0p5.json"),
    }
    exact_only_full = {
        "devset": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_plain_w1_exact_only_dev_all_v09.json"),
        "val": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_plain_w1_exact_only_val_all_v09.json"),
    }
    exact_plus_hard_selector = {
        "devset": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_plain_w1_plus_hard_simple_dev_all_v09.json"),
        "val": load(EVID / "request_corrected_bootstrap_official_vs_slice105k_plain_w1_plus_hard_simple_val_all_v09.json"),
    }
    exact_plus_learned_hard = {
        "devset": load(EVID / "request_corrected_bootstrap_official_vs_learned_exact_plus_learned_simple_nonneg_hard_dev_all_v09.json"),
        "val": load(EVID / "request_corrected_bootstrap_official_vs_learned_exact_plus_learned_simple_nonneg_hard_val_all_v09.json"),
    }
    hard_action_audit = {
        "devset": load(EVID / "request_hard_artist_learned_action_audit_dev_v09.json"),
        "val": load(EVID / "request_hard_artist_learned_action_audit_val_v09.json"),
    }
    hard_simple_nonneg_action_audit = {
        "devset": load(EVID / "request_hard_artist_simple_nonneg_learned_action_audit_dev_v09.json"),
        "val": load(EVID / "request_hard_artist_simple_nonneg_learned_action_audit_val_v09.json"),
    }
    hard_simple_slice105k_action_audit = {
        "devset": load(EVID / "request_hard_artist_simple_nonneg_slice105k_learned_action_audit_dev_v09.json"),
        "val": load(EVID / "request_hard_artist_simple_nonneg_slice105k_learned_action_audit_val_v09.json"),
    }
    behavior = {
        "devset": load(EVID / "request_slice105k_plain_behavior_dev_exact_version_v09.json"),
        "val": load(EVID / "request_slice105k_plain_behavior_val_exact_version_v09.json"),
    }
    behavior_w075 = {
        "devset": load(EVID / "request_slice105k_plain_w0p75_behavior_dev_exact_version_v09.json"),
        "val": load(EVID / "request_slice105k_plain_w0p75_behavior_val_exact_version_v09.json"),
    }
    control_behavior = {
        "devset": load(EVID / "request_slice105k_official_control_behavior_dev_exact_version_v09.json"),
        "val": load(EVID / "request_slice105k_official_control_behavior_val_exact_version_v09.json"),
    }
    nofeature_behavior = {
        "devset": load(EVID / "request_slice105k_nofeature_behavior_dev_exact_version_v09.json"),
        "val": load(EVID / "request_slice105k_nofeature_behavior_val_exact_version_v09.json"),
    }
    wrongpos_behavior = {
        "devset": load(EVID / "request_slice105k_wrongpos_reqfeat_behavior_dev_exact_version_v09.json"),
        "val": load(EVID / "request_slice105k_wrongpos_reqfeat_behavior_val_exact_version_v09.json"),
    }
    crossdialogue_behavior = {
        "devset": load(EVID / "request_slice105k_crossdialogue_reqfeat_behavior_dev_exact_version_v09.json"),
        "val": load(EVID / "request_slice105k_crossdialogue_reqfeat_behavior_val_exact_version_v09.json"),
    }
    behavior_stats = load(EVID / "request_training_behavior_stats_v09.json")
    slice_summary = load(EVID / "request_exact_slice_105k_feature_parquet_v09.json")
    transfer = load(EVID / "transfer_predictions/request_exact_transfer_F10_R54SRC_exact_version_v09_slice105k_blindset_A.summary.json")
    fulltrain = load(EVID / "request_fulltrain_freeze_manifest_v09.json")
    fulltrain_result = load(EVID / "request_fulltrain_result_card_v09.json")

    audit_rows = []
    for row in locked["splits"]:
        audit_rows.append({
            "split": row["split"],
            "exact_version_records": row["exact_version_records"],
            "visible_request_turns": row["strict_request_turns"],
            "requested_present_top100": row["requested_present_top100"],
            "changed_rows": row["changed_rows"],
            "official_delta": ci_cell(row["official_delta"], row["official_ci95"]),
            "corrected_delta": ci_cell(row["corrected_delta"], row["corrected_ci95"]),
            "affected_corrected_delta": ci_cell(row["affected_corrected_delta"], row["affected_corrected_ci95"]),
        })

    intervention_rows = []
    for split in ("devset", "val"):
        intervention_rows.append(bootstrap_row("full-pool promotion", split, fullpool[split]))
        intervention_rows.append(bootstrap_row("105k request-slice specialist (lambda=1.0)", split, specialist[split]))

    hard_training_rows = []
    for split in ("devset", "val"):
        hard_training_rows.append(bootstrap_row_for_family(
            "105k hard-artist official-label control (lambda=0.3)",
            split,
            hard_official_control[split],
            "hard_artist_constraint",
            "hard_slice_corrected_delta",
        ))
        hard_training_rows.append(bootstrap_row_for_family(
            "105k hard-artist add-positive specialist (lambda=0.3)",
            split,
            hard_positive[split],
            "hard_artist_constraint",
            "hard_slice_corrected_delta",
        ))
        hard_training_rows.append(bootstrap_row_for_family(
            "105k hard-artist masked specialist (lambda=0.1)",
            split,
            hard_masked[split],
            "hard_artist_constraint",
            "hard_slice_corrected_delta",
        ))

    hard_simple_training_rows = []
    for split in ("devset", "val"):
        hard_simple_training_rows.append(bootstrap_row_for_family(
            "329-group simple hard official-label control (lambda=0.5)",
            split,
            hard_simple_official_control[split],
            "hard_artist_constraint",
            "hard_family_corrected_delta",
        ))
        hard_simple_training_rows.append(bootstrap_row_for_family(
            "329-group simple hard add-positive specialist (lambda=0.5)",
            split,
            hard_simple_positive[split],
            "hard_artist_constraint",
            "hard_family_corrected_delta",
        ))

    multifamily_rows = []
    for split in ("devset", "val"):
        multifamily_rows.append(overall_bootstrap_row(
            "trained exact only",
            split,
            exact_only_full[split],
        ))
        multifamily_rows.append(overall_bootstrap_row(
            "trained exact + transparent simple-hard selector",
            split,
            exact_plus_hard_selector[split],
        ))
        multifamily_rows.append(overall_bootstrap_row(
            "trained exact + learned simple/non-negated hard specialist",
            split,
            exact_plus_learned_hard[split],
        ))

    hard_action_rows = []
    for slice_name, audits in [
        ("strict_all", hard_action_audit),
        ("simple_nonneg", hard_simple_nonneg_action_audit),
        ("simple_nonneg_329group", hard_simple_slice105k_action_audit),
    ]:
        for split in ("devset", "val"):
            audit = audits[split]
            for block in audit["models"]:
                n = block["n_directives"]
                sat20 = block["candidate_satisfying_at_k"]["20"]
                hard_action_rows.append({
                    "slice": slice_name,
                    "split": split,
                    "model": block["name"],
                    "top1_changed": count_rate_cell(block["top1_changed"], n),
                    "top1_satisfies": count_rate_cell(block["candidate_top1_satisfies"], n),
                    "top1_satisfies_delta": f"{block['top1_satisfies_delta']:+d} ({block['top1_satisfies_delta_rate']:+.1%})",
                    "changed_toward": block["changed_top1_toward_satisfying"],
                    "changed_away": block["changed_top1_away_from_satisfying"],
                    "satisfying_at_20_delta": f"{sat20['delta']:+d} ({sat20['delta_rate']:+.1%})",
                })

    operating_rows = []
    for split in ("devset", "val"):
        for label, artifacts, behaviors in [
            ("lambda=0.75", specialist_w075, behavior_w075),
            ("lambda=1.0", specialist, behavior),
        ]:
            counts = behaviors[split]["directive_counts"]
            operating_rows.append({
                "split": split,
                "operating_point": label,
                "official_delta": delta_cell(artifacts[split]["official"]["delta"]),
                "corrected_delta": delta_cell(artifacts[split]["corrected"]["delta"]),
                "exact_slice_corrected_delta": delta_cell(
                    artifacts[split]["families"]["exact_track_request"]["corrected"]["delta"]
                ),
                "request_first": f"{counts['adapter_request_first']}/{counts['n_directives']}",
                "baseline_missing_top20_recovered": counts["adapter_request_first_when_baseline_missing"],
            })

    control_rows = []
    for split in ("devset", "val"):
        control_rows.append(bootstrap_row("105k official-label control", split, official_control[split]))
        control_rows.append(bootstrap_row("105k no-feature corrected-label control", split, nofeature_control[split]))
        control_rows.append(bootstrap_row("105k same-count wrong-positive control", split, wrongpos_control[split]))
        control_rows.append(bootstrap_row("105k cross-dialogue request-positive control", split, crossdialogue_control[split]))

    behavior_rows = []
    for split in ("devset", "val"):
        counts = behavior[split]["directive_counts"]
        control_counts = control_behavior[split]["directive_counts"]
        nofeature_counts = nofeature_behavior[split]["directive_counts"]
        wrongpos_counts = wrongpos_behavior[split]["directive_counts"]
        crossdialogue_counts = crossdialogue_behavior[split]["directive_counts"]
        behavior_rows.append({
            "split": split,
            "visible_directives": counts["n_directives"],
            "baseline_request_first": counts["baseline_request_first"],
            "official_control_request_first": control_counts["adapter_request_first"],
            "nofeature_control_request_first": nofeature_counts["adapter_request_first"],
            "wrongpos_control_request_first": wrongpos_counts["adapter_request_first"],
            "crossdialogue_control_request_first": crossdialogue_counts["adapter_request_first"],
            "specialist_request_first": counts["adapter_request_first"],
            "fullpool_request_first": counts["postrank_request_first"],
            "baseline_missing_top20_recovered": counts["adapter_request_first_when_baseline_missing"],
        })

    behavior_stat_rows = []
    for row in behavior_stats["rows"]:
        behavior_stat_rows.append({
            "split": row["split"],
            "system": row["system"],
            "baseline_first": f"{row['baseline_request_first']}/{row['n_directives']}",
            "system_first": f"{row['system_request_first']}/{row['n_directives']}",
            "gains": row["gains_baseline_not_first_to_system_first"],
            "losses": row["losses_baseline_first_to_system_not_first"],
            "net_gain": row["net_request_first_gain"],
            "baseline_missing_recovered": row["baseline_missing_recovered"],
            "exact_p": p_cell(row["mcnemar_exact_p"]),
        })

    precision_rows = [
        {
            "readout": "detector sample",
            "estimate": wilson_cell(locked["detector_precision"]),
        },
        {
            "readout": "changed-row strict precision",
            "estimate": wilson_cell(locked["changed_precision_overall"]),
        },
        {
            "readout": "LLM adjudication chose request",
            "estimate": wilson_cell(locked["preference_validation"]["request_preferred_rate"]),
        },
    ]

    transfer_counts = transfer["counts"]
    transfer_rows = [{
        "split": transfer["split"],
        "prediction_groups": transfer_counts["n_prediction_groups"],
        "visible_directives": transfer_counts["n_directives_total"],
        "target_turn_directives": transfer_counts["n_evaluable_directives"],
        "baseline_request_first": f"{transfer_counts['baseline']['request_first']}/{transfer_counts['n_evaluable_directives']}",
        "fullpool_request_first": f"{transfer_counts['fullpool']['request_first']}/{transfer_counts['n_evaluable_directives']}",
        "specialist_request_first": f"{transfer_counts['specialist']['request_first']}/{transfer_counts['n_evaluable_directives']}",
    }]

    expected = fulltrain["expected"]
    fulltrain_exact_delta = fulltrain_result.get("exact_version", {}).get("delta_request_minus_control", {})
    fulltrain_exact_status = (
        "run; official "
        f"{fulltrain_exact_delta.get('official_ndcg20_text')}, "
        f"request-positive {fulltrain_exact_delta.get('request_positive_ndcg20_text')}, "
        f"slice {fulltrain_exact_delta.get('positive_slice_request_positive_ndcg20_text')}"
        if fulltrain_result.get("answer") == "yes_on_frozen_primary_readout"
        else "frozen, not run"
    )
    fulltrain_rows = [
        {
            "family": "exact/version",
            "training_sources": "train_a + val",
            "groups": count_cell(expected["train_n_groups"]),
            "candidate_rows": count_cell(expected["train_n_rows"]),
            "frozen_primary_blend": expected["exact"]["primary_blend_weight"],
            "label_action_count": (
                f"{count_cell(expected['exact']['added_positive_rows'])} added positives; "
                f"{count_cell(expected['exact']['downweighted_groups'])} downweighted groups"
            ),
            "readout_status": fulltrain_exact_status,
        },
        {
            "family": "hard simple/non-exact",
            "training_sources": "train_a + val",
            "groups": count_cell(expected["train_n_groups"]),
            "candidate_rows": count_cell(expected["train_n_rows"]),
            "frozen_primary_blend": expected["hard"]["primary_blend_weight"],
            "label_action_count": f"{count_cell(expected['hard']['dropped_groups'])} dropped violation groups",
            "readout_status": fulltrain_result.get("hard_violation_drop", {}).get("status", "frozen, not run"),
        },
    ]

    return {
        "sources": {
            "locked": "docs/evidence/request_exact_postrank_locked_table_v09.json",
            "fullpool_dev": "docs/evidence/request_exact_fullpool_bootstrap_official_vs_fullpool_dev_exact_version_v09.json",
            "fullpool_val": "docs/evidence/request_exact_fullpool_bootstrap_official_vs_fullpool_val_exact_version_v09.json",
            "specialist_dev": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_plain_blend_dev_exact_version_v09_w1.json",
            "specialist_val": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_plain_blend_val_exact_version_v09_w1.json",
            "specialist_w0p75_dev": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_plain_blend_dev_exact_version_v09_w0p75.json",
            "specialist_w0p75_val": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_plain_blend_val_exact_version_v09_w0p75.json",
            "official_control_dev": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_official_control_blend_dev_exact_version_v09_w1.json",
            "official_control_val": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_official_control_blend_val_exact_version_v09_w1.json",
            "nofeature_control_dev": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_nofeature_blend_dev_exact_version_v09_w1.json",
            "nofeature_control_val": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_nofeature_blend_val_exact_version_v09_w1.json",
            "wrongpos_control_dev": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_wrongpos_reqfeat_blend_dev_exact_version_v09_w1.json",
            "wrongpos_control_val": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_wrongpos_reqfeat_blend_val_exact_version_v09_w1.json",
            "crossdialogue_control_dev": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_crossdialogue_reqfeat_blend_dev_exact_version_v09_w1.json",
            "crossdialogue_control_val": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_crossdialogue_reqfeat_blend_val_exact_version_v09_w1.json",
            "hard_positive_dev": "docs/evidence/request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_positive_dev_v09_w0p3.json",
            "hard_positive_val": "docs/evidence/request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_positive_val_v09_w0p3.json",
            "hard_masked_dev": "docs/evidence/request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_masked_dev_v09_w0p1.json",
            "hard_masked_val": "docs/evidence/request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_masked_val_v09_w0p1.json",
            "hard_official_control_dev": "docs/evidence/request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_official_control_dev_v09_w0p3.json",
            "hard_official_control_val": "docs/evidence/request_hard_artist_bootstrap_official_vs_slice105k_hardfeat_official_control_val_v09_w0p3.json",
            "hard_simple_positive_dev": "docs/evidence/request_hard_simple_nonneg_bootstrap_official_vs_slice105k_hardfeat_positive_dev_v09_w0p5.json",
            "hard_simple_positive_val": "docs/evidence/request_hard_simple_nonneg_bootstrap_official_vs_slice105k_hardfeat_positive_val_v09_w0p5.json",
            "hard_simple_official_control_dev": "docs/evidence/request_hard_simple_nonneg_bootstrap_official_vs_slice105k_hardfeat_official_control_dev_v09_w0p5.json",
            "hard_simple_official_control_val": "docs/evidence/request_hard_simple_nonneg_bootstrap_official_vs_slice105k_hardfeat_official_control_val_v09_w0p5.json",
            "exact_only_full_dev": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_plain_w1_exact_only_dev_all_v09.json",
            "exact_only_full_val": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_plain_w1_exact_only_val_all_v09.json",
            "exact_plus_hard_selector_dev": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_plain_w1_plus_hard_simple_dev_all_v09.json",
            "exact_plus_hard_selector_val": "docs/evidence/request_corrected_bootstrap_official_vs_slice105k_plain_w1_plus_hard_simple_val_all_v09.json",
            "exact_plus_learned_hard_dev": "docs/evidence/request_corrected_bootstrap_official_vs_learned_exact_plus_learned_simple_nonneg_hard_dev_all_v09.json",
            "exact_plus_learned_hard_val": "docs/evidence/request_corrected_bootstrap_official_vs_learned_exact_plus_learned_simple_nonneg_hard_val_all_v09.json",
            "hard_action_audit_dev": "docs/evidence/request_hard_artist_learned_action_audit_dev_v09.json",
            "hard_action_audit_val": "docs/evidence/request_hard_artist_learned_action_audit_val_v09.json",
            "hard_simple_nonneg_action_audit_dev": "docs/evidence/request_hard_artist_simple_nonneg_learned_action_audit_dev_v09.json",
            "hard_simple_nonneg_action_audit_val": "docs/evidence/request_hard_artist_simple_nonneg_learned_action_audit_val_v09.json",
            "hard_simple_slice105k_action_audit_dev": "docs/evidence/request_hard_artist_simple_nonneg_slice105k_learned_action_audit_dev_v09.json",
            "hard_simple_slice105k_action_audit_val": "docs/evidence/request_hard_artist_simple_nonneg_slice105k_learned_action_audit_val_v09.json",
            "behavior_stats": "docs/evidence/request_training_behavior_stats_v09.json",
            "slice_summary": "docs/evidence/request_exact_slice_105k_feature_parquet_v09.json",
            "transfer_blind_a": "docs/evidence/transfer_predictions/request_exact_transfer_F10_R54SRC_exact_version_v09_slice105k_blindset_A.summary.json",
            "fulltrain_freeze": "docs/evidence/request_fulltrain_freeze_manifest_v09.json",
        },
        "precision_rows": precision_rows,
        "audit_rows": audit_rows,
        "intervention_rows": intervention_rows,
        "hard_training_rows": hard_training_rows,
        "hard_simple_training_rows": hard_simple_training_rows,
        "multifamily_rows": multifamily_rows,
        "hard_action_rows": hard_action_rows,
        "operating_rows": operating_rows,
        "control_rows": control_rows,
        "behavior_rows": behavior_rows,
        "behavior_stat_rows": behavior_stat_rows,
        "slice_training": {
            "source_rows": slice_summary["rows_in"],
            "kept_groups": slice_summary["seen_groups"],
            "kept_rows": slice_summary["rows_out"],
            "missing_groups": slice_summary["missing_groups"],
        },
        "fulltrain_rows": fulltrain_rows,
        "transfer_rows": transfer_rows,
    }


def markdown(tables: dict[str, Any]) -> str:
    lines = [
        "# Request-Correction Paper Tables v0.9",
        "",
        "Date: 2026-06-22",
        "",
        "Generated by `scripts/pilot_tables.py` from locked JSON evidence artifacts. Do not hand-copy these numbers without regenerating this file.",
        "",
        "## Precision",
        "",
    ]
    lines.extend(rows_to_markdown(["readout", "estimate"], tables["precision_rows"]))
    lines.extend([
        "",
        "## Development-Gate Exact/Version Audit",
        "",
    ])
    lines.extend(rows_to_markdown([
        "split",
        "exact_version_records",
        "visible_request_turns",
        "requested_present_top100",
        "changed_rows",
        "official_delta",
        "corrected_delta",
        "affected_corrected_delta",
    ], tables["audit_rows"]))
    lines.extend([
        "",
        "## Main Interventions",
        "",
    ])
    lines.extend(rows_to_markdown([
        "method",
        "split",
        "official_delta",
        "corrected_delta",
        "exact_slice_official_delta",
        "exact_slice_corrected_delta",
    ], tables["intervention_rows"]))
    lines.extend([
        "",
        "## Hard-Artist Learned Stress Result",
        "",
        "Hard-artist rows use the same visible `hard_artist_constraint_match` feature and hard-artist gate. The official-label control isolates whether request-satisfying hard positives add signal beyond the feature/gate.",
        "",
    ])
    lines.extend(rows_to_markdown([
        "method",
        "split",
        "official_delta",
        "corrected_delta",
        "hard_slice_corrected_delta",
    ], tables["hard_training_rows"]))
    lines.extend([
        "",
        "## Simple Non-Negated Hard-Artist Supporting Training Result",
        "",
        "Filtered hard-artist slice using the stricter abstaining selector. This is supporting evidence only; the selector is post-hoc and the action audit is detector-coupled.",
        "",
    ])
    lines.extend(rows_to_markdown([
        "method",
        "split",
        "official_delta",
        "corrected_delta",
        "hard_family_corrected_delta",
    ], tables["hard_simple_training_rows"]))
    lines.extend([
        "",
        "## Multi-Family Learned System Readout",
        "",
        "All rows are compared with the same raw official trained baseline and scored against the full v0.9 sidecar. The transparent selector row is not a pure training result; the learned simple/non-negated hard row composes two trained specialists under disjoint visible gates.",
        "",
    ])
    lines.extend(rows_to_markdown([
        "system",
        "split",
        "official_delta",
        "corrected_delta",
        "hard_artist_corrected_delta",
    ], tables["multifamily_rows"]))
    lines.extend([
        "",
        "## Hard-Artist Learned Action Audit",
        "",
        "Detector-coupled top-1/top-20 constraint-satisfaction audit. This supports mechanism only; it does not replace independent action precision.",
        "",
    ])
    lines.extend(rows_to_markdown([
        "slice",
        "split",
        "model",
        "top1_changed",
        "top1_satisfies",
        "top1_satisfies_delta",
        "changed_toward",
        "changed_away",
        "satisfying_at_20_delta",
    ], tables["hard_action_rows"]))
    lines.extend([
        "",
        "## Learned Specialist Operating Points",
        "",
        "Same trained 105k request-slice specialist and dialogue gate; only the interpolation weight between official and specialist scores changes.",
        "",
    ])
    lines.extend(rows_to_markdown([
        "split",
        "operating_point",
        "official_delta",
        "corrected_delta",
                "exact_slice_corrected_delta",
        "request_first",
        "baseline_missing_top20_recovered",
    ], tables["operating_rows"]))
    lines.extend([
        "",
        "## Training Causal Control",
        "",
        "Same 105k request slice. Controls vary whether the labels are official-only, corrected without the visible request-match feature, same-count wrong in-pool positives, or same-count cross-dialogue request positives.",
        "",
    ])
    lines.extend(rows_to_markdown([
        "method",
        "split",
        "official_delta",
        "corrected_delta",
        "exact_slice_corrected_delta",
    ], tables["control_rows"]))
    lines.extend([
        "",
        "## 105k Specialist Behavior",
        "",
    ])
    lines.extend(rows_to_markdown([
        "split",
        "visible_directives",
        "baseline_request_first",
        "official_control_request_first",
        "nofeature_control_request_first",
        "wrongpos_control_request_first",
        "crossdialogue_control_request_first",
        "specialist_request_first",
        "fullpool_request_first",
        "baseline_missing_top20_recovered",
    ], tables["behavior_rows"]))
    lines.extend([
        "",
        "## Paired Request-First Behavior",
        "",
        "Exact McNemar/binomial sign test over discordant visible exact/version directives.",
        "",
    ])
    lines.extend(rows_to_markdown([
        "split",
        "system",
        "baseline_first",
        "system_first",
        "gains",
        "losses",
        "net_gain",
        "baseline_missing_recovered",
        "exact_p",
    ], tables["behavior_stat_rows"]))
    s = tables["slice_training"]
    lines.extend([
        "",
        "## 105k Training Slice",
        "",
        "| source_rows | kept_groups | kept_rows | missing_groups |",
        "| --- | --- | --- | --- |",
        f"| {s['source_rows']} | {s['kept_groups']} | {s['kept_rows']} | {s['missing_groups']} |",
        "",
        "## Frozen Full-Train Readout",
        "",
        "This is a predeclared organizer-train readout after folding repo `val` back into train. Exact/version is now evaluated at the frozen lambda=1.0 readout; hard-drop remains secondary/pending unless its outputs exist.",
        "",
    ])
    lines.extend(rows_to_markdown([
        "family",
        "training_sources",
        "groups",
        "candidate_rows",
        "frozen_primary_blend",
        "label_action_count",
        "readout_status",
    ], tables["fulltrain_rows"]))
    lines.extend([
        "",
        "## Blind A Transfer Rehearsal",
        "",
    ])
    lines.extend(rows_to_markdown([
        "split",
        "prediction_groups",
        "visible_directives",
        "target_turn_directives",
        "baseline_request_first",
        "fullpool_request_first",
        "specialist_request_first",
    ], tables["transfer_rows"]))
    lines.extend([
        "",
        "Read: Blind A is a no-label portability check, not a performance claim.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-output", type=Path, default=EVID / "pilot/readout_tables.json")
    ap.add_argument("--markdown-output", type=Path, default=EVID / "pilot/readout_tables.md")
    args = ap.parse_args(argv)

    tables = build_tables()
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(tables, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown_output.write_text(markdown(tables) + "\n", encoding="utf-8")
    print(json.dumps({
        "json_output": str(args.json_output),
        "markdown_output": str(args.markdown_output),
        "n_tables": 14,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
