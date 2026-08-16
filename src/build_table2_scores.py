"""Build a pending-safe card for the frozen full-train readout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
DEFAULT_JSON = EVID / "results/table2_scores.json"
DEFAULT_MD = EVID / "results/table2_scores.md"

EXACT_CONTROL = "docs/evidence/request_exact_version_fulltrain_official_control_blend_dev_v09.json"
EXACT_REQUEST = "docs/evidence/request_exact_version_fulltrain_exact_positive_weighted_blend_dev_v09.json"
EXACT_BOOTSTRAP = "docs/evidence/request_exact_version_fulltrain_bootstrap_official_vs_specialist_dev_v09.json"
HARD_CONTROL = "docs/evidence/request_hard_simple_nonneg_fulltrain_official_control_blend_dev_v09.json"
HARD_REQUEST = "docs/evidence/request_hard_simple_nonneg_fulltrain_violation_drop_blend_dev_v09.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_delta(value: float) -> str:
    return f"{value:+.5f}"


def weight_row(data: dict[str, Any], weight: str) -> dict[str, Any]:
    rows = data.get("blend_weights") or {}
    if weight not in rows:
        raise KeyError(f"blend weight {weight!r} missing; found {sorted(rows)}")
    return rows[weight]


def output_status(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    return {
        "path": rel,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def comparison(root: Path, control_rel: str, request_rel: str, weight: str) -> dict[str, Any]:
    control_path = root / control_rel
    request_path = root / request_rel
    status = {
        "control": output_status(root, control_rel),
        "request": output_status(root, request_rel),
    }
    if not control_path.exists() or not request_path.exists():
        return {
            "status": "pending_outputs",
            "outputs": status,
        }

    control = weight_row(load_json(control_path), weight)
    request = weight_row(load_json(request_path), weight)
    official_delta = request["official_ndcg20"] - control["official_ndcg20"]
    request_positive_delta = request["corrected_ndcg20"] - control["corrected_ndcg20"]
    slice_delta = (
        request["positive_slice"]["corrected_ndcg20"]
        - control["positive_slice"]["corrected_ndcg20"]
    )
    return {
        "status": "present",
        "outputs": status,
        "blend_weight": float(weight),
        "control": {
            "official_ndcg20": control["official_ndcg20"],
            "request_positive_ndcg20": control["corrected_ndcg20"],
            "positive_slice_request_positive_ndcg20": control["positive_slice"]["corrected_ndcg20"],
            "prediction_path": control.get("prediction_path"),
        },
        "request": {
            "official_ndcg20": request["official_ndcg20"],
            "request_positive_ndcg20": request["corrected_ndcg20"],
            "positive_slice_request_positive_ndcg20": request["positive_slice"]["corrected_ndcg20"],
            "prediction_path": request.get("prediction_path"),
        },
        "delta_request_minus_control": {
            "official_ndcg20": official_delta,
            "official_ndcg20_text": fmt_delta(official_delta),
            "request_positive_ndcg20": request_positive_delta,
            "request_positive_ndcg20_text": fmt_delta(request_positive_delta),
            "positive_slice_request_positive_ndcg20": slice_delta,
            "positive_slice_request_positive_ndcg20_text": fmt_delta(slice_delta),
        },
        "passes_official_small_loss_threshold": official_delta > -0.001,
        "request_positive_gain": request_positive_delta > 0,
        "positive_slice_gain": slice_delta > 0,
    }


def ci_text(delta: dict[str, Any]) -> str:
    return f"[{fmt_delta(delta['ci95_low'])}, {fmt_delta(delta['ci95_high'])}]"


def bootstrap_summary(root: Path, rel: str = EXACT_BOOTSTRAP) -> dict[str, Any]:
    path = root / rel
    if not path.exists():
        return {
            "status": "missing",
            "path": rel,
        }
    data = load_json(path)
    exact_track = data["families"]["exact_track_request"]["corrected"]["delta"]
    return {
        "status": "present",
        "path": rel,
        "n_boot": data["n_boot"],
        "seed": data["seed"],
        "unit": "session",
        "official": {
            "observed_delta_text": fmt_delta(data["official"]["delta"]["observed_delta"]),
            "ci95_text": ci_text(data["official"]["delta"]),
            "n_sessions": data["official"]["delta"]["n_sessions"],
        },
        "request_satisfying": {
            "observed_delta_text": fmt_delta(data["corrected"]["delta"]["observed_delta"]),
            "ci95_text": ci_text(data["corrected"]["delta"]),
            "n_sessions": data["corrected"]["delta"]["n_sessions"],
        },
        "exact_track_slice": {
            "observed_delta_text": fmt_delta(exact_track["observed_delta"]),
            "ci95_text": ci_text(exact_track),
            "n_sessions": exact_track["n_sessions"],
        },
        "note": "The version/duplicate component has one session, so slice uncertainty is reported for exact-track requests only.",
    }


def build_card(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    exact = comparison(root, EXACT_CONTROL, EXACT_REQUEST, "1")
    bootstrap = bootstrap_summary(root)
    hard = comparison(root, HARD_CONTROL, HARD_REQUEST, "0p5")
    exact_ready = exact["status"] == "present"
    hard_ready = hard["status"] == "present"
    return {
        "date": "2026-06-23",
        "status": "present" if exact_ready else "pending_outputs",
        "question": (
            "Does the frozen full-train exact/version readout preserve official "
            "dev score while improving request-positive dev score?"
        ),
        "answer": (
            "pending_frozen_readout"
            if not exact_ready
            else (
                "yes_on_frozen_primary_readout"
                if exact["passes_official_small_loss_threshold"] and exact["request_positive_gain"]
                else "mixed_or_negative_on_frozen_primary_readout"
            )
        ),
        "exact_version": exact,
        "exact_version_bootstrap": bootstrap,
        "hard_violation_drop": hard,
        "hard_is_secondary": True,
        "read": (
            "This card is pending-safe: before full-train eval outputs exist it "
            "reports readiness only; after they exist it computes the fixed "
            "primary readout deltas without tuning on repo devset."
        ),
        "sources": [
            "docs/evidence/request_fulltrain_approval_packet_2026_06_23.md",
            "docs/evidence/request_fulltrain_eval_readiness_2026_06_23.md",
            "docs/evidence/request_exact_version_fulltrain_protocol_v09.md",
            "docs/evidence/request_exact_version_fulltrain_bootstrap_official_vs_specialist_dev_v09.json",
            "docs/evidence/request_exact_version_fulltrain_bootstrap_official_vs_specialist_dev_v09.md",
            "docs/evidence/request_hard_simple_nonneg_fulltrain_violation_drop_protocol_v09.md",
        ],
    }


def markdown(card: dict[str, Any]) -> str:
    lines = [
        "# Request Full-Train Result Card v0.9",
        "",
        f"Date: {card['date']}",
        "",
        f"Question: {card['question']}",
        "",
        f"Answer: **{card['answer']}**.",
        "",
        card["read"],
        "",
        "## Exact/Version Primary Readout",
        "",
    ]
    exact = card["exact_version"]
    if exact["status"] != "present":
        lines.extend([
            "Status: `pending_outputs`.",
            "",
            "| output | exists |",
            "|---|---:|",
            f"| official-label exact-feature control | `{exact['outputs']['control']['exists']}` |",
            f"| request-positive exact/version specialist | `{exact['outputs']['request']['exists']}` |",
            "",
            "Interpretation: no full-train performance claim is available until both outputs exist.",
        ])
    else:
        delta = exact["delta_request_minus_control"]
        lines.extend([
            "| metric | request-positive minus official-control | pass criterion |",
            "|---|---:|---|",
            f"| official nDCG@20 | {delta['official_ndcg20_text']} | > -0.001 |",
            f"| auxiliary request-positive nDCG@20 | {delta['request_positive_ndcg20_text']} | > 0 |",
            f"| exact-slice auxiliary request-positive nDCG@20 | {delta['positive_slice_request_positive_ndcg20_text']} | > 0 |",
        ])
    bootstrap = card["exact_version_bootstrap"]
    lines.extend([
        "",
        "## Paired Bootstrap Support",
        "",
    ])
    if bootstrap["status"] != "present":
        lines.append(f"Status: `{bootstrap['status']}`.")
    else:
        lines.extend([
            f"Unit: `{bootstrap['unit']}`. Samples: `{bootstrap['n_boot']}`. Seed: `{bootstrap['seed']}`.",
            "",
            "| readout | observed delta | 95% CI | sessions |",
            "|---|---:|---:|---:|",
            f"| official nDCG@20 | {bootstrap['official']['observed_delta_text']} | {bootstrap['official']['ci95_text']} | {bootstrap['official']['n_sessions']} |",
            f"| request-satisfying nDCG@20 | {bootstrap['request_satisfying']['observed_delta_text']} | {bootstrap['request_satisfying']['ci95_text']} | {bootstrap['request_satisfying']['n_sessions']} |",
            f"| exact-track slice request-satisfying nDCG@20 | {bootstrap['exact_track_slice']['observed_delta_text']} | {bootstrap['exact_track_slice']['ci95_text']} | {bootstrap['exact_track_slice']['n_sessions']} |",
            "",
            bootstrap["note"],
        ])
    lines.extend([
        "",
        "## Hard Violation-Drop Secondary Readout",
        "",
        f"Status: `{card['hard_violation_drop']['status']}`. Hard remains supporting and cannot replace the exact/version headline.",
        "",
        "## Sources",
        "",
    ])
    lines.extend(f"- `{source}`" for source in card["sources"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = ap.parse_args(argv)

    card = build_card(args.root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(card), encoding="utf-8")
    print(json.dumps({
        "json": str(args.output_json),
        "md": str(args.output_md),
        "status": card["status"],
        "answer": card["answer"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
