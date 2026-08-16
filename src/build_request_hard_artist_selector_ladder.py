"""Build a precision/coverage ladder for hard-artist selectors.

The hard-artist family is useful only if the paper is honest about the tradeoff:
broad coverage has weaker action precision, while the simple/non-negated slice
is cleaner but much smaller and was derived after failure analysis.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
DEFAULT_JSON = EVID / "ledger/hard_artist_selector_ladder.json"
DEFAULT_MD = EVID / "ledger/hard_artist_selector_ladder.md"


RUNGS = [
    {
        "id": "broad_strict",
        "name": "broad strict hard-artist",
        "description": "strict hard-artist parser, top-20 promotion",
        "dev_postrank": "request_hard_artist_strict_postrank_top20_rerank_F10_scaled_devset.summary.json",
        "val_postrank": "request_hard_artist_strict_postrank_top20_rerank_F10_R54SRC_val.summary.json",
        "val_changed": "request_hard_artist_strict_top20_changed_rows_val_v09.summary.json",
        "val_labels": "request_hard_artist_strict_top20_changed_rows_val_labels_v09.summary.json",
    },
    {
        "id": "noexact",
        "name": "exclude exact-title turns",
        "description": "route exact-title turns to exact/version, then apply hard-artist promotion",
        "dev_postrank": "request_hard_artist_strict_postrank_top20_noexact_rerank_F10_scaled_devset.summary.json",
        "val_postrank": "request_hard_artist_strict_postrank_top20_noexact_rerank_F10_R54SRC_val.summary.json",
        "val_changed": "request_hard_artist_strict_top20_noexact_changed_rows_val_v09.summary.json",
        "val_labels": "request_hard_artist_strict_top20_noexact_changed_rows_val_labels_v09.summary.json",
    },
    {
        "id": "simple_noexact",
        "name": "simple/no-exact action selector",
        "description": "exclude exact-title turns and abstain on visible title, album, era, lyric, and reference cues",
        "dev_postrank": "request_hard_artist_strict_postrank_top20_simple_noexact_rerank_F10_scaled_devset.summary.json",
        "val_postrank": "request_hard_artist_strict_postrank_top20_simple_noexact_rerank_F10_R54SRC_val.summary.json",
        "val_changed": "request_hard_artist_strict_top20_simple_noexact_changed_rows_val_v09.summary.json",
        "val_labels": "request_hard_artist_strict_top20_simple_noexact_changed_rows_val_labels_v09.summary.json",
    },
]

SIDECAR_SUMMARIES = {
    "train_lt": "request_corrections_train_lt_v09_simple_nonneg_hard_artist_only.summary.json",
    "devset": "request_corrections_devset_v09_simple_nonneg_hard_artist_only.summary.json",
    "val": "request_corrections_val_v09_simple_nonneg_hard_artist_only.summary.json",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes: int, n: int, z: float = 1.96) -> dict[str, float | None]:
    if n == 0:
        return {"point": None, "low": None, "high": None}
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return {"point": p, "low": max(0.0, center - half), "high": min(1.0, center + half)}


def count_yes(summary: dict[str, Any], field: str) -> int:
    return int(summary["overall"]["fields"][field]["counts"].get("yes", 0))


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def build_ladder() -> dict[str, Any]:
    rungs: list[dict[str, Any]] = []
    for rung in RUNGS:
        dev_post = load(EVID / rung["dev_postrank"])
        val_post = load(EVID / rung["val_postrank"])
        val_changed = load(EVID / rung["val_changed"])
        labels = load(EVID / rung["val_labels"])
        n_labels = int(labels["overall"]["n"])
        strict_success = round(float(labels["overall"]["strict_intervention_precision"]) * n_labels)
        request_success = round(float(labels["overall"]["request_and_satisfaction_precision"]) * n_labels)
        action_ok = count_yes(labels, "action_ok")
        promoted_satisfies = count_yes(labels, "promoted_satisfies")
        valid_request = count_yes(labels, "valid_request")
        gold_drop = int(labels["overall"]["fields"]["gold_should_remain_positive"]["counts"].get("no", 0))
        rungs.append({
            "id": rung["id"],
            "name": rung["name"],
            "description": rung["description"],
            "dev": {
                "directives": int(dev_post["directive_turns"]),
                "present": int(dev_post["present_turns"]),
                "changed": int(dev_post["changed_turns"]),
                "present_rate": rate(int(dev_post["present_turns"]), int(dev_post["directive_turns"])),
                "changed_rate": rate(int(dev_post["changed_turns"]), int(dev_post["directive_turns"])),
            },
            "val": {
                "directives": int(val_post["directive_turns"]),
                "present": int(val_post["present_turns"]),
                "changed": int(val_post["changed_turns"]),
                "present_rate": rate(int(val_post["present_turns"]), int(val_post["directive_turns"])),
                "changed_rate": rate(int(val_post["changed_turns"]), int(val_post["directive_turns"])),
                "changed_rows": int(val_changed["n_rows"]),
                "gold_violates_changed_rows": int(val_changed["gold_violates_constraint"]),
                "gold_satisfies_changed_rows": int(val_changed["gold_satisfies_constraint"]),
            },
            "val_label_audit": {
                "n": n_labels,
                "strict_intervention_successes": strict_success,
                "strict_intervention_precision": wilson(strict_success, n_labels),
                "request_and_satisfaction_successes": request_success,
                "request_and_satisfaction_precision": wilson(request_success, n_labels),
                "valid_request_yes": valid_request,
                "promoted_satisfies_yes": promoted_satisfies,
                "action_ok_yes": action_ok,
                "gold_should_not_remain_positive": gold_drop,
            },
        })

    sidecars = {
        split: load(EVID / path)
        for split, path in SIDECAR_SUMMARIES.items()
    }
    simple_sidecar = {
        split: {
            "input_records": int(summary["input_records"]),
            "selector_keys_before_nonneg_filter": int(summary["selector_keys_before_nonneg_filter"]),
            "selector_keys_after_nonneg_filter": int(summary["selector_keys_after_nonneg_filter"]),
            "kept_records": int(summary["kept_records"]),
            "kept_rate": rate(int(summary["kept_records"]), int(summary["input_records"])),
            "dropped_by_reason": summary["dropped_by_reason"],
        }
        for split, summary in sidecars.items()
    }
    return {
        "date": "2026-06-22",
        "audit": "request-hard-artist-selector-ladder-v0.9",
        "rungs": rungs,
        "simple_nonneg_sidecar": simple_sidecar,
        "interpretation": {
            "paper_safe": (
                "Hard-artist is useful supporting evidence: precision improves as the selector abstains, "
                "but coverage shrinks and the cleanest slice is post-hoc."
            ),
            "independent_label_scope": (
                "Independent labels cover postrank selector changed rows only; learned-specialist action "
                "audits are detector-coupled mechanism checks, not independent changed-row precision."
            ),
            "do_not_claim": "hard-artist is an admitted correction family",
        },
        "sources": [
            *(f"docs/evidence/{rung[key]}" for rung in RUNGS for key in ("dev_postrank", "val_postrank", "val_changed", "val_labels")),
            *(f"docs/evidence/{path}" for path in SIDECAR_SUMMARIES.values()),
        ],
    }


def markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Hard-Artist Selector Ladder v0.9",
        "",
        "Date: 2026-06-22",
        "",
        "Purpose: quantify the precision/coverage tradeoff that keeps hard-artist as supporting evidence rather than an admitted correction family.",
        "",
        "## Selector Ladder",
        "",
        "| selector | dev directives | dev changed | val directives | val changed | val labeled precision | val action ok | gold should drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["rungs"]:
        label = row["val_label_audit"]
        prec = label["strict_intervention_precision"]
        lines.append(
            "| "
            + " | ".join([
                row["name"],
                str(row["dev"]["directives"]),
                str(row["dev"]["changed"]),
                str(row["val"]["directives"]),
                str(row["val"]["changed"]),
                (
                    f"{label['strict_intervention_successes']}/{label['n']} "
                    f"({pct(prec['point'])}, 95% CI {pct(prec['low'])}-{pct(prec['high'])})"
                ),
                f"{label['action_ok_yes']}/{label['n']}",
                f"{label['gold_should_not_remain_positive']}/{label['n']}",
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Simple/Non-Negated Training Sidecar",
        "",
        "| split | broad input records | selector keys before nonneg | selector keys after nonneg | kept conflict records | kept rate | dropped reasons |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for split, row in audit["simple_nonneg_sidecar"].items():
        dropped = ", ".join(f"{k}={v}" for k, v in sorted(row["dropped_by_reason"].items()))
        lines.append(
            "| "
            + " | ".join([
                split,
                str(row["input_records"]),
                str(row["selector_keys_before_nonneg_filter"]),
                str(row["selector_keys_after_nonneg_filter"]),
                str(row["kept_records"]),
                pct(row["kept_rate"]),
                dropped,
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Read",
        "",
        "The broad hard-artist family is learnable, but its changed-action precision is too weak for admission. Excluding exact-title turns alone does not fix precision. The simple/no-exact action selector reaches 12/13 labeled strict precision on val changed rows, but it changes only 13 val rows. The training sidecar applies one more non-negation filter and keeps 329/1,587 train conflict records, 16/31 dev records, and 32/155 val records.",
        "",
        "Independent labels cover postrank selector changed rows only. The learned-specialist action audits are detector-coupled mechanism checks, not independent changed-row precision for the learned model.",
        "",
        "Paper-safe claim: hard-artist is a useful learned stress test and frozen-transfer candidate. It is not an admitted correction family until an untouched split confirms the selector and independent action precision.",
        "",
        "## Sources",
        "",
    ])
    lines.extend(f"- `{source}`" for source in audit["sources"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = ap.parse_args(argv)

    audit = build_ladder()
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.write_text(markdown(audit), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "md": str(args.md_out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
