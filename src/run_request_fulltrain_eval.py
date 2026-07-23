"""Run the frozen post-launch full-train devset evaluations.

This is deliberately a runner, not a search script. It has no blend sweep and no
threshold knobs: exact/version is fixed at lambda=1.0; hard violation-drop is
fixed at lambda=0.5 and is opt-in because it is secondary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "docs/evidence/request_fulltrain_eval_runner_v09.json"
DEFAULT_MD = ROOT / "docs/evidence/request_fulltrain_eval_runner_v09.md"

PRODUCTION_ANCHOR = Path(os.environ.get("MCRS_PRODUCTION_ANCHOR", str(Path(os.environ.get("MCRS_EXPLORE_ROOT", str(Path(__file__).resolve().parent.parent))) / "exp/models/rerank_F10_R54SRC_r54src.json")))

EVAL_SPECS: dict[str, dict[str, Any]] = {
    "exact_version_official_control": {
        "family": "exact/version",
        "role": "official-label exact-feature control",
        "primary_blend": "1",
        "request_model": "docs/evidence/models/request_corrected_F10_R54SRC_official_exact_version_v09_fulltrain_trainval_reqfeat_gw0p1_medium50.json",
        "corrections": "docs/evidence/request_corrections_devset_exact_version_v09.jsonl",
        "gate_mode": "dialogue",
        "feature_flag": "--request-model-add-exact-request-feature",
        "output_tag": "exact_version_v09_fulltrain_official_control_reqfeat_dialogue",
        "output": "docs/evidence/request_exact_version_fulltrain_official_control_blend_dev_v09.json",
    },
    "exact_version_request_positive": {
        "family": "exact/version",
        "role": "request-positive exact/version specialist",
        "primary_blend": "1",
        "request_model": "docs/evidence/models/request_corrected_F10_R54SRC_exact_positive_weighted_exact_version_v09_fulltrain_trainval_reqfeat_gw0p1_medium50.json",
        "corrections": "docs/evidence/request_corrections_devset_exact_version_v09.jsonl",
        "gate_mode": "dialogue",
        "feature_flag": "--request-model-add-exact-request-feature",
        "output_tag": "exact_version_v09_fulltrain_exact_positive_weighted_reqfeat_dialogue",
        "output": "docs/evidence/request_exact_version_fulltrain_exact_positive_weighted_blend_dev_v09.json",
    },
    "hard_violation_drop_official_control": {
        "family": "hard simple/non-negated",
        "role": "official-label hard-feature control",
        "primary_blend": "0.5",
        "request_model": "docs/evidence/models/request_corrected_F10_R54SRC_official_hard_artist_v09_simple_nonneg_fulltrain_trainval_hardfeat_drop_medium50.json",
        "corrections": "docs/evidence/request_corrections_devset_v09_simple_nonneg_hard_artist_only.jsonl",
        "gate_mode": "hard_artist",
        "feature_flag": "--request-model-add-hard-artist-feature",
        "extra_flags": ["--hard-artist-exclude-exact-requests", "--hard-artist-simple-only"],
        "output_tag": "hard_artist_v09_simple_nonneg_fulltrain_official_control_w0p5",
        "output": "docs/evidence/request_hard_simple_nonneg_fulltrain_official_control_blend_dev_v09.json",
    },
    "hard_violation_drop": {
        "family": "hard simple/non-negated",
        "role": "violation-drop hard specialist",
        "primary_blend": "0.5",
        "request_model": "docs/evidence/models/request_corrected_F10_R54SRC_violation_drop_hard_artist_v09_simple_nonneg_fulltrain_trainval_hardfeat_drop_medium50.json",
        "corrections": "docs/evidence/request_corrections_devset_v09_simple_nonneg_hard_artist_only.jsonl",
        "gate_mode": "hard_artist",
        "feature_flag": "--request-model-add-hard-artist-feature",
        "extra_flags": ["--hard-artist-exclude-exact-requests", "--hard-artist-simple-only"],
        "output_tag": "hard_artist_v09_simple_nonneg_fulltrain_violation_drop_w0p5",
        "output": "docs/evidence/request_hard_simple_nonneg_fulltrain_violation_drop_blend_dev_v09.json",
    },
}

EXACT_NAMES = ["exact_version_official_control", "exact_version_request_positive"]
HARD_NAMES = ["hard_violation_drop_official_control", "hard_violation_drop"]


def repo_path(root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def command_for_spec(spec: dict[str, Any], *, official_model: Path = PRODUCTION_ANCHOR) -> list[str]:
    cmd = [
        ".venv/bin/python",
        "scripts/request_corrected_rerank_blend_eval.py",
        "--feature-set",
        "F10_R54SRC",
        "--split",
        "devset",
        "--official-model",
        str(official_model),
        "--request-model",
        spec["request_model"],
        "--corrections",
        spec["corrections"],
        "--gate-mode",
        spec["gate_mode"],
    ]
    cmd.extend(spec.get("extra_flags", []))
    cmd.extend([
        spec["feature_flag"],
        "--blend-weight",
        spec["primary_blend"],
        "--prediction-dir",
        "docs/evidence/dev_predictions",
        "--output-tag",
        spec["output_tag"],
        "--output",
        spec["output"],
    ])
    return cmd


def command_text(cmd: list[str]) -> str:
    return " ".join(cmd)


def selected_names(*, include_hard: bool = False) -> list[str]:
    return EXACT_NAMES + (HARD_NAMES if include_hard else [])


def path_row(root: Path, raw: str | Path) -> dict[str, Any]:
    path = repo_path(root, raw)
    return {
        "path": str(raw),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def build_plan(
    *,
    root: Path = ROOT,
    include_hard: bool = False,
    allow_overwrite: bool = False,
    official_model: Path = PRODUCTION_ANCHOR,
) -> dict[str, Any]:
    root = root.resolve()
    names = selected_names(include_hard=include_hard)
    common = {"production_anchor": path_row(root, official_model)}
    evals = []
    missing_inputs: list[dict[str, Any]] = []
    existing_outputs: list[dict[str, Any]] = []

    for name in names:
        spec = EVAL_SPECS[name]
        request_model = path_row(root, spec["request_model"])
        corrections = path_row(root, spec["corrections"])
        output = path_row(root, spec["output"])
        command = command_for_spec(spec, official_model=official_model)
        row = {
            "name": name,
            "family": spec["family"],
            "role": spec["role"],
            "primary_blend": float(spec["primary_blend"]),
            "request_model": request_model,
            "corrections": corrections,
            "output": output,
            "command": command,
            "command_text": command_text(command),
        }
        evals.append(row)
        if not request_model["exists"]:
            missing_inputs.append({"eval": name, "kind": "request_model_missing", **request_model})
        if not corrections["exists"]:
            missing_inputs.append({"eval": name, "kind": "corrections_missing", **corrections})
        if output["exists"]:
            existing_outputs.append({"eval": name, "kind": "output_exists", **output})

    if not common["production_anchor"]["exists"]:
        missing_inputs.append({
            "eval": "all",
            "kind": "production_anchor_missing",
            **common["production_anchor"],
        })

    blockers = []
    if missing_inputs:
        blockers.append({"kind": "missing_inputs", "items": missing_inputs})
    if existing_outputs and not allow_overwrite:
        blockers.append({"kind": "existing_outputs", "items": existing_outputs})

    if missing_inputs:
        status = "pending_inputs"
    elif existing_outputs and not allow_overwrite:
        status = "blocked_existing_outputs"
    else:
        status = "ready_to_run"

    return {
        "date": "2026-06-23",
        "protocol": "request-fulltrain-v0.9-frozen-devset-eval",
        "status": status,
        "include_hard": include_hard,
        "allow_overwrite": allow_overwrite,
        "common": common,
        "evals": evals,
        "blockers": blockers,
        "read": (
            "This runner is the frozen post-launch devset readout. It uses only "
            "the predeclared primary blend weights: exact/version lambda=1.0 "
            "and, when explicitly included, hard violation-drop lambda=0.5."
        ),
        "after_run_builders": [
            ".venv/bin/python scripts/build_request_fulltrain_result_card.py",
            ".venv/bin/python scripts/build_request_submission_package_manifest.py",
        ],
    }


def markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Full-Train Eval Runner v0.9",
        "",
        f"Date: {plan['date']}",
        "",
        plan["read"],
        "",
        "## Status",
        "",
        f"- Runner status: `{plan['status']}`.",
        f"- Include hard secondary readout: `{plan['include_hard']}`.",
        f"- Allow overwrite: `{plan['allow_overwrite']}`.",
        "",
        "## Fixed Eval Commands",
        "",
        "| eval | family | blend | output exists | command |",
        "|---|---|---:|---:|---|",
    ]
    for row in plan["evals"]:
        lines.append(
            f"| `{row['name']}` | {row['family']} | {row['primary_blend']} | "
            f"`{row['output']['exists']}` | `{row['command_text']}` |"
        )
    lines.extend(["", "## Blockers", ""])
    if not plan["blockers"]:
        lines.append("- None.")
    else:
        for blocker in plan["blockers"]:
            lines.append(f"- `{blocker['kind']}`: {len(blocker['items'])} item(s).")
    lines.extend([
        "",
        "## After Run",
        "",
    ])
    lines.extend(f"- `{cmd}`" for cmd in plan["after_run_builders"])
    lines.append("")
    return "\n".join(lines)


def write_plan(plan: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown(plan), encoding="utf-8")


def run_commands(plan: dict[str, Any], *, rebuild_cards: bool = True) -> None:
    if plan["status"] != "ready_to_run":
        raise RuntimeError(f"full-train eval runner is not ready: {plan['status']}")
    for row in plan["evals"]:
        subprocess.run(row["command"], cwd=ROOT, check=True)
    if rebuild_cards:
        for cmd in plan["after_run_builders"]:
            subprocess.run(cmd.split(), cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--include-hard", action="store_true")
    ap.add_argument("--allow-overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-rebuild-cards", action="store_true")
    ap.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = ap.parse_args(argv)

    plan = build_plan(
        root=args.root,
        include_hard=args.include_hard,
        allow_overwrite=args.allow_overwrite,
    )
    write_plan(plan, output_json=args.output_json, output_md=args.output_md)
    print(json.dumps({
        "status": plan["status"],
        "include_hard": plan["include_hard"],
        "n_evals": len(plan["evals"]),
        "n_blockers": sum(len(blocker["items"]) for blocker in plan["blockers"]),
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, indent=2, sort_keys=True))

    if args.dry_run:
        return
    run_commands(plan, rebuild_cards=not args.no_rebuild_cards)


if __name__ == "__main__":
    main()
