"""Summarize paired request-first behavior for trained request modules."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_RUNS = [
    ("devset", "corrected_label_specialist", Path("docs/evidence/request_slice105k_plain_behavior_dev_exact_version_v09.json")),
    ("val", "corrected_label_specialist", Path("docs/evidence/request_slice105k_plain_behavior_val_exact_version_v09.json")),
    ("devset", "corrected_label_specialist_w0p75", Path("docs/evidence/request_slice105k_plain_w0p75_behavior_dev_exact_version_v09.json")),
    ("val", "corrected_label_specialist_w0p75", Path("docs/evidence/request_slice105k_plain_w0p75_behavior_val_exact_version_v09.json")),
    ("devset", "no_feature_corrected_control", Path("docs/evidence/request_slice105k_nofeature_behavior_dev_exact_version_v09.json")),
    ("val", "no_feature_corrected_control", Path("docs/evidence/request_slice105k_nofeature_behavior_val_exact_version_v09.json")),
    ("devset", "official_label_control", Path("docs/evidence/request_slice105k_official_control_behavior_dev_exact_version_v09.json")),
    ("val", "official_label_control", Path("docs/evidence/request_slice105k_official_control_behavior_val_exact_version_v09.json")),
    ("devset", "wrong_positive_control", Path("docs/evidence/request_slice105k_wrongpos_reqfeat_behavior_dev_exact_version_v09.json")),
    ("val", "wrong_positive_control", Path("docs/evidence/request_slice105k_wrongpos_reqfeat_behavior_val_exact_version_v09.json")),
    ("devset", "cross_dialogue_request_positive_control", Path("docs/evidence/request_slice105k_crossdialogue_reqfeat_behavior_dev_exact_version_v09.json")),
    ("val", "cross_dialogue_request_positive_control", Path("docs/evidence/request_slice105k_crossdialogue_reqfeat_behavior_val_exact_version_v09.json")),
]


def is_first(rank: int | None) -> bool:
    return rank == 1


def mcnemar_exact_p(gains: int, losses: int) -> float | None:
    """Two-sided exact binomial sign test over discordant paired outcomes."""
    n = gains + losses
    if n == 0:
        return None
    tail = min(gains, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * prob)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    baseline_first = 0
    system_first = 0
    both_first = 0
    gains = 0
    losses = 0
    both_not_first = 0
    baseline_missing_recovered = 0
    baseline_top20_not_first_recovered = 0
    system_missing_when_baseline_first = 0

    for row in rows:
        b_first = is_first(row.get("baseline_rank"))
        s_first = is_first(row.get("adapter_rank"))
        baseline_first += int(b_first)
        system_first += int(s_first)
        both_first += int(b_first and s_first)
        gains += int((not b_first) and s_first)
        losses += int(b_first and not s_first)
        both_not_first += int((not b_first) and (not s_first))
        baseline_missing_recovered += int(row.get("baseline_rank") is None and s_first)
        baseline_top20_not_first_recovered += int(row.get("baseline_rank") not in (None, 1) and s_first)
        system_missing_when_baseline_first += int(b_first and row.get("adapter_rank") is None)

    return {
        "n_directives": n,
        "baseline_request_first": baseline_first,
        "system_request_first": system_first,
        "both_request_first": both_first,
        "gains_baseline_not_first_to_system_first": gains,
        "losses_baseline_first_to_system_not_first": losses,
        "both_not_request_first": both_not_first,
        "net_request_first_gain": gains - losses,
        "baseline_missing_recovered": baseline_missing_recovered,
        "baseline_top20_not_first_recovered": baseline_top20_not_first_recovered,
        "system_missing_when_baseline_first": system_missing_when_baseline_first,
        "mcnemar_exact_p": mcnemar_exact_p(gains, losses),
    }


def summarize_file(split: str, system: str, path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = summarize_rows(data["rows"])
    summary.update({
        "split": split,
        "system": system,
        "path": str(path),
    })
    return summary


def format_p(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# Request Training Behavior Paired Statistics v0.9",
        "",
        "Date: 2026-06-22",
        "",
        "This table treats each visible exact/version directive as a paired",
        "baseline-vs-system request-first outcome. The p-value is a two-sided",
        "exact McNemar/binomial sign test over discordant directives: baseline",
        "not first -> system first versus baseline first -> system not first.",
        "",
        "| split | system | baseline first | system first | gains | losses | net gain | baseline-missing recovered | exact p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {split} | {system} | {baseline}/{n} | {system_first}/{n} | "
            "{gains} | {losses} | {net} | {missing} | {p} |".format(
                split=row["split"],
                system=row["system"],
                baseline=row["baseline_request_first"],
                system_first=row["system_request_first"],
                n=row["n_directives"],
                gains=row["gains_baseline_not_first_to_system_first"],
                losses=row["losses_baseline_first_to_system_not_first"],
                net=row["net_request_first_gain"],
                missing=row["baseline_missing_recovered"],
                p=format_p(row["mcnemar_exact_p"]),
            )
        )
    lines.extend([
        "",
        "Readout:",
        "",
        "- The corrected-label specialist has large one-sided paired gains on both",
        "  dev and val, with no request-first regressions in these behavior audits.",
        "- The 0.75 blend is a conservative operating point for the same specialist:",
        "  it preserves the dev request-first result but gives up a few validation",
        "  request-first wins in exchange for slightly cleaner official nDCG.",
        "- The no-feature corrected-label control still gains many request-first",
        "  outcomes, showing the labels carry signal even without the explicit",
        "  request-match feature.",
        "- The official-label and wrong-positive controls move in the opposite",
        "  direction: they lose baseline request-first outcomes and almost never",
        "  create new ones.",
        "- The cross-dialogue request-positive control also fails, showing that",
        "  positives drawn from other request dialogues are not enough when they",
        "  do not match the current request.",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", type=Path, default=Path("docs/evidence/request_training_behavior_stats_v09.json"))
    ap.add_argument("--output-md", type=Path, default=Path("docs/evidence/request_training_behavior_stats_v09.md"))
    args = ap.parse_args(argv)

    rows = [summarize_file(split, system, path) for split, system, path in DEFAULT_RUNS]
    result = {
        "date": "2026-06-22",
        "statistic": "paired_request_first_mcnemar_exact",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(rows, args.output_md)
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "n_rows": len(rows),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
