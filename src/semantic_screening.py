"""Build a screening audit for broad semantic request-language matches."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from family_mining import (
    BROAD_SEMANTIC_PATTERNS,
    SPLIT_FILE,
    broad_semantic_matches,
    iter_arrow_rows,
    latest_dataset_cache_root,
    user_turns,
)


EVID = ROOT / "evidence"
DEFAULT_JSON = EVID / "request_broad_semantic_screening_audit_v09.json"
DEFAULT_MD = EVID / "request_broad_semantic_screening_audit_v09.md"

REQUEST_CUE_RE = re.compile(
    r"\b("
    r"want|need|looking for|look for|can you|could you|please|recommend|"
    r"play|find|show me|give me|try|something|more|another|similar|"
    r"mood|vibe|do you have|how about"
    r")\b",
    re.I,
)
COMPOUND_CUE_RE = re.compile(
    r"\"|'|\b(not by|different artist|other artists?|from (?:the )?album|"
    r"by [A-Z][A-Za-z0-9&.' -]{1,40}|lyrics?|era|2000s|90s|80s)\b",
    re.I,
)

CATEGORY_BLOCKERS = {
    "genre_style": (
        "no trusted genre/style taxonomy field in the released catalog; title, "
        "artist, album, release date, and popularity cannot certify genre"
    ),
    "mood_energy_tempo": "no mood, energy, tempo, or audio-feature fields in the released catalog",
    "instrumentation_vocal": "no instrumentation or vocal-tag fields in the released catalog",
    "language_culture": "no language, country, or culture-tag fields in the released catalog",
    "popularity_recency": (
        "popularity and release_date exist, but requests mix classic/recent/era "
        "semantics; year/date semantics are handled by the rejected year/decade audit"
    ),
}


def stable_key(*parts: Any) -> str:
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def visible_request_like(text: str) -> bool:
    return bool(REQUEST_CUE_RE.search(text))


def compound_or_mixed(text: str, categories: list[str]) -> bool:
    return len(categories) > 1 or bool(COMPOUND_CUE_RE.search(text))


def split_for_train_row(session_id: str, split_ids: dict[str, set[str]]) -> str | None:
    if session_id in split_ids["train_lt"]:
        return "train_lt"
    if session_id in split_ids["val"]:
        return "val"
    return None


def iter_split_rows(dataset_root: Path, split_file: Path) -> Any:
    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    split_ids = {
        "train_lt": set(split_data["train_a"]),
        "val": set(split_data["val"]),
    }
    train_path = dataset_root / "talk_play_data-challenge-dataset-train.arrow"
    dev_path = dataset_root / "talk_play_data-challenge-dataset-test.arrow"
    for row in iter_arrow_rows(train_path):
        split = split_for_train_row(row["session_id"], split_ids)
        if split:
            yield split, row
    for row in iter_arrow_rows(dev_path):
        yield "devset", row


def collect_screening_samples(
    *,
    dataset_root: Path | None = None,
    split_file: Path = SPLIT_FILE,
    sample_split: str = "devset",
    per_category: int = 25,
) -> dict[str, Any]:
    dataset_root = dataset_root or latest_dataset_cache_root()
    if dataset_root is None:
        return {
            "available": False,
            "reason": "cached TalkPlayData-Challenge-Dataset Arrow files not found",
            "sample_split": sample_split,
            "per_category": per_category,
            "categories": {},
        }

    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in BROAD_SEMANTIC_PATTERNS}
    for split, row in iter_split_rows(dataset_root, split_file):
        if split != sample_split:
            continue
        for turn_number, text in user_turns(row):
            categories = broad_semantic_matches(text)
            for category in categories:
                buckets[category].append(
                    {
                        "split": split,
                        "session_id": row.get("session_id"),
                        "turn_number": turn_number,
                        "category": category,
                        "matched_categories": categories,
                        "text": text[:500],
                        "sample_key": stable_key(category, row.get("session_id"), turn_number, text),
                    }
                )

    categories_out: dict[str, Any] = {}
    for category, rows in buckets.items():
        rows = sorted(rows, key=lambda row: row["sample_key"])[:per_category]
        screened = []
        for row in rows:
            text = row["text"]
            categories = list(row["matched_categories"])
            screened.append(
                {
                    **row,
                    "visible_request_like_by_rubric": visible_request_like(text),
                    "compound_or_mixed_by_rubric": compound_or_mixed(text, categories),
                    "trusted_catalog_resolvable_under_current_contract": False,
                    "main_blocker": CATEGORY_BLOCKERS[category],
                }
            )
        categories_out[category] = {
            "sampled_matches": len(screened),
            "visible_request_like_by_rubric": sum(
                1 for row in screened if row["visible_request_like_by_rubric"]
            ),
            "compound_or_mixed_by_rubric": sum(
                1 for row in screened if row["compound_or_mixed_by_rubric"]
            ),
            "trusted_catalog_resolvable_under_current_contract": sum(
                1 for row in screened if row["trusted_catalog_resolvable_under_current_contract"]
            ),
            "main_blocker": CATEGORY_BLOCKERS[category],
            "examples": screened[:5],
        }

    return {
        "available": True,
        "audit": "request-broad-semantic-screening-v0.9",
        "date": "2026-06-22",
        "dataset_root": str(dataset_root),
        "sample_split": sample_split,
        "per_category": per_category,
        "read": (
            "Deterministic rubric screen over broad semantic regex matches. "
            "This is not a correction-label audit and not an independent human "
            "precision estimate; it tests whether the mined family has visible "
            "request language and a trusted catalog-resolved target set."
        ),
        "categories": categories_out,
        "paper_implication": (
            "Broad semantic requests are real enough to motivate the admission "
            "workflow, but they remain screening-only evidence because the current "
            "catalog cannot turn them into trusted request-satisfying target sets."
        ),
    }


def markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Broad Semantic Screening Audit v0.9",
        "",
        "Date: 2026-06-22",
        "",
        audit.get("read", ""),
        "",
    ]
    if not audit.get("available"):
        lines.extend(["Audit unavailable.", "", f"Reason: {audit.get('reason')}", ""])
        return "\n".join(lines)

    lines.extend(
        [
            f"Sample split: `{audit['sample_split']}`. Deterministic sample size: "
            f"{audit['per_category']} per category.",
            "",
            "## Screening Table",
            "",
            "| category | sampled | request-like by rubric | compound/mixed by rubric | trusted catalog-resolvable | main blocker |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for category, row in audit["categories"].items():
        lines.append(
            f"| {category} | {row['sampled_matches']} | "
            f"{row['visible_request_like_by_rubric']}/{row['sampled_matches']} | "
            f"{row['compound_or_mixed_by_rubric']}/{row['sampled_matches']} | "
            f"{row['trusted_catalog_resolvable_under_current_contract']}/{row['sampled_matches']} | "
            f"{row['main_blocker']} |"
        )

    lines.extend(
        [
            "",
            "Read: high request-like rates support treating these as visible request-language families. "
            "The zero trusted target-set rate is the admission blocker, not evidence that the requests are unimportant.",
            "",
            "## Representative Examples",
            "",
        ]
    )
    for category, row in audit["categories"].items():
        lines.extend([f"### {category}", ""])
        for example in row["examples"][:3]:
            lines.append(
                f"- `{example['split']}` `{example['session_id']}` turn {example['turn_number']}: "
                f"{example['text']}"
            )
        lines.append("")
    lines.extend(["## Paper Implication", "", audit["paper_implication"], ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    ap.add_argument("--sample-split", default="devset", choices=["train_lt", "val", "devset"])
    ap.add_argument("--per-category", type=int, default=25)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = ap.parse_args(argv)

    audit = collect_screening_samples(
        dataset_root=args.dataset_root,
        split_file=args.split_file,
        sample_split=args.sample_split,
        per_category=args.per_category,
    )
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.write_text(markdown(audit), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "md": str(args.md_out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
