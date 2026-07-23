"""Build a scope/prevalence card for exact/version request evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "docs/evidence"
DEFAULT_JSON = EVID / "request_scope_prevalence_card_v09.json"
DEFAULT_MD = EVID / "request_scope_prevalence_card_v09.md"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.1%})" if total else "0/0 (0.0%)"


def per_1k(count: int, total: int) -> float:
    return 1000.0 * count / total if total else 0.0


def unit_ceiling(count: int, total: int) -> float:
    return count / total if total else 0.0


def paper_row(rows: list[dict[str, Any]], method: str, split: str) -> dict[str, Any]:
    for row in rows:
        if row.get("method") == method and row.get("split") == split:
            return row
    raise KeyError(f"{method}/{split}")


def build_card(
    *,
    locked: dict[str, Any],
    robustness: dict[str, Any],
    paper_tables: dict[str, Any],
) -> dict[str, Any]:
    robust_by_split = {row["split"]: row for row in robustness["splits"]}
    behavior_by_split = {row["split"]: row for row in paper_tables["behavior_rows"]}
    rows = []
    for locked_row in locked["splits"]:
        split = locked_row["split"]
        robust = robust_by_split[split]
        behavior = behavior_by_split[split]
        total_turns = int(locked_row["prediction_rows"])
        visible = int(locked_row["strict_request_turns"])
        changed = int(locked_row["changed_rows"])
        present = int(locked_row["requested_present_top100"])
        missing = visible - present
        already_first = present - changed
        request_first_gain = int(behavior["specialist_request_first"]) - int(behavior["baseline_request_first"])
        fullpool = paper_row(paper_tables["intervention_rows"], "full-pool promotion", split)
        specialist = paper_row(
            paper_tables["intervention_rows"],
            "105k request-slice specialist (lambda=1.0)",
            split,
        )
        rows.append({
            "split": split,
            "total_turns": total_turns,
            "visible_directives": visible,
            "visible_directives_per_1k_turns": per_1k(visible, total_turns),
            "policy_request_conflicts": int(robust["gold_differs"]),
            "policy_request_conflicts_per_1k_turns": per_1k(int(robust["gold_differs"]), total_turns),
            "requested_present_top100": present,
            "requested_absent_top100": missing,
            "already_request_first": already_first,
            "reachable_changed_rows": changed,
            "reachable_changed_rows_per_1k_turns": per_1k(changed, total_turns),
            "request_first_gain": request_first_gain,
            "request_first_gain_per_1k_turns": per_1k(request_first_gain, total_turns),
            "visible_directive_unit_ceiling": unit_ceiling(visible, total_turns),
            "request_first_gain_unit_ceiling": unit_ceiling(request_first_gain, total_turns),
            "fullpool_auxiliary_request_positive_delta": fullpool["corrected_delta"],
            "fullpool_exact_slice_official_delta": fullpool["exact_slice_official_delta"],
            "specialist_auxiliary_request_positive_delta": specialist["corrected_delta"],
            "specialist_exact_slice_official_delta": specialist["exact_slice_official_delta"],
            "baseline_request_first": int(behavior["baseline_request_first"]),
            "specialist_request_first": int(behavior["specialist_request_first"]),
        })
    return {
        "artifact": "request-scope-prevalence-card-v0.9",
        "read": (
            "Exact/version is deliberately narrow. The global auxiliary nDCG "
            "movement should be read together with per-1k prevalence, reachable "
            "coverage, and a unit-bounded prevalence ceiling; official "
            "preservation is whole-split, while exact-slice official deltas can "
            "be negative when request-satisfying items displace policy-selected labels."
        ),
        "unit_ceiling_note": (
            "The unit ceilings are simple prevalence ceilings for a unit-bounded "
            "per-turn score if only that slice could change; they are explanatory "
            "denominator math, not a replacement for paired bootstrap metrics."
        ),
        "rows": rows,
        "sources": [
            "docs/evidence/request_exact_postrank_locked_table_v09.json",
            "docs/evidence/request_exact_robustness_table_v09.json",
            "docs/evidence/request_paper_tables_v09.json",
        ],
    }


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.5f}"


def markdown(card: dict[str, Any]) -> str:
    lines = [
        "# Request Scope And Prevalence Card v0.9",
        "",
        card["read"],
        "",
        card["unit_ceiling_note"],
        "",
        "## Exact/Version Scope",
        "",
        "| split | turns | exact directives | per 1k turns | policy/request conflicts | reachable changes | request-first gain | visible-slice unit ceiling | gain-row unit ceiling |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in card["rows"]:
        lines.append(
            "| "
            f"{row['split']} | "
            f"{row['total_turns']} | "
            f"{row['visible_directives']} | "
            f"{fmt_float(row['visible_directives_per_1k_turns'])} | "
            f"{pct(row['policy_request_conflicts'], row['visible_directives'])} | "
            f"{pct(row['reachable_changed_rows'], row['visible_directives'])} | "
            f"{row['baseline_request_first']} -> {row['specialist_request_first']} "
            f"(+{row['request_first_gain']}) | "
            f"{fmt_delta(row['visible_directive_unit_ceiling'])} | "
            f"{fmt_delta(row['request_first_gain_unit_ceiling'])} |"
        )
    lines.extend([
        "",
        "## Metric Readout Context",
        "",
        "| split | specialist auxiliary request-positive delta | specialist exact-slice official delta | full-pool auxiliary request-positive delta | full-pool exact-slice official delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in card["rows"]:
        lines.append(
            "| "
            f"{row['split']} | "
            f"{row['specialist_auxiliary_request_positive_delta']} | "
            f"{row['specialist_exact_slice_official_delta']} | "
            f"{row['fullpool_auxiliary_request_positive_delta']} | "
            f"{row['fullpool_exact_slice_official_delta']} |"
        )
    lines.extend([
        "",
        "Interpretation: the exact/version result is a conditional behavior result with a narrow denominator. That narrowness explains why whole-split auxiliary request-positive deltas are small even when request-first behavior changes sharply on exact/version turns. Exact-slice official deltas are reported because satisfying the visible request can trade off against the policy-selected single label.",
        "",
        "## Sources",
        "",
    ])
    lines.extend(f"- `{source}`" for source in card["sources"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", type=Path, default=EVID / "request_exact_postrank_locked_table_v09.json")
    ap.add_argument("--robustness", type=Path, default=EVID / "request_exact_robustness_table_v09.json")
    ap.add_argument("--paper-tables", type=Path, default=EVID / "request_paper_tables_v09.json")
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = ap.parse_args(argv)

    card = build_card(
        locked=load(args.locked),
        robustness=load(args.robustness),
        paper_tables=load(args.paper_tables),
    )
    args.json_out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.write_text(markdown(card), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "md": str(args.md_out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
