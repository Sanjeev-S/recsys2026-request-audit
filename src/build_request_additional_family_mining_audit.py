"""Build an audit for mined request families that are not admitted yet."""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pyarrow.ipc as ipc


ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "evidence"
DEFAULT_JSON = EVID / "request_additional_family_mining_audit_v09.json"
DEFAULT_MD = EVID / "request_additional_family_mining_audit_v09.md"
BROAD_SCREENING_JSON = EVID / "request_broad_semantic_screening_audit_v09.json"
SPLIT_FILE = ROOT / "data/train_val_split_seed42.json"
DEFAULT_DATASET_CACHE_GLOB = (
    Path(os.environ.get("HF_DATASETS_CACHE", str(Path.home() / ".cache/huggingface/datasets")))
    / "talkpl-ai___talk_play_data-challenge-dataset/default/0.0.0/*"
)

SPLITS = ("train_lt", "val", "devset")
BASE_SUMMARY = {
    "train_lt": EVID / "request_corrections_train_lt_v09.summary.json",
    "val": EVID / "request_corrections_val_v09.summary.json",
    "devset": EVID / "request_corrections_devset_v09.summary.json",
}
YEAR_SUMMARY = {
    "train_lt": EVID / "request_corrections_train_lt_with_year_v09.summary.json",
    "val": EVID / "request_corrections_val_with_year_v09.summary.json",
    "devset": EVID / "request_corrections_devset_with_year_v09.summary.json",
}
BASE_JSONL = {
    "train_lt": EVID / "request_corrections_train_lt_v09.jsonl",
    "val": EVID / "request_corrections_val_v09.jsonl",
    "devset": EVID / "request_corrections_devset_v09.jsonl",
}
YEAR_JSONL = {
    "val": EVID / "request_corrections_val_with_year_v09.jsonl",
    "devset": EVID / "request_corrections_devset_with_year_v09.jsonl",
}

RANGE_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\s*-\s*(\d{2}|19[5-9]\d|20[0-2]\d)\b")
LATE_90S_EARLY_2000S_RE = re.compile(r"\b90s\b[\s\S]{0,80}\b2000s\b|\b2000s\b[\s\S]{0,80}\b90s\b", re.I)
DECADE_WORD_RE = re.compile(r"\b\d0s\b|\b\d{4}s\b", re.I)
RELEASE_RE = re.compile(r"\breleas(?:e|ed|ing)|\bfrom \d{4}\b|\bin \d{4}\b", re.I)
POPULARITY_RE = re.compile(r"\d{2,3}%|popularity", re.I)

BROAD_SEMANTIC_PATTERNS = {
    "genre_style": re.compile(
        r"\b("
        r"genre|style|subgenre|rock|pop|metal|jazz|classical|hip\s*hop|rap|"
        r"country|electronic|edm|punk|indie|folk|blues|soul|r&b|disco|reggae|"
        r"grunge|funk|techno|house|ambient"
        r")\b",
        re.I,
    ),
    "mood_energy_tempo": re.compile(
        r"\b("
        r"mood|vibe|vibes|upbeat|chill|relax(?:ed|ing)?|sad|happy|melanchol(?:y|ic)|"
        r"energetic|mellow|aggressive|intense|calm|tempo|fast|slow|danceable|"
        r"workout|party|sleep|focus|atmospheric"
        r")\b",
        re.I,
    ),
    "instrumentation_vocal": re.compile(
        r"\b("
        r"acoustic|instrumental|guitar|piano|drums?|bass|sax(?:ophone)?|synth|"
        r"violin|strings?|vocals?|singer|duet|choir|beat|percussion"
        r")\b",
        re.I,
    ),
    "language_culture": re.compile(
        r"\b("
        r"spanish|french|korean|japanese|latin|english|german|italian|portuguese|"
        r"language|non-english|k-pop|afrobeats?"
        r")\b",
        re.I,
    ),
    "popularity_recency": re.compile(
        r"\b("
        r"popular|popularity|hit|hits|mainstream|underground|obscure|classic|"
        r"trending|chart|viral|recent|new release|newer|modern|old school|well known"
        r")\b|\d{2,3}%",
        re.I,
    ),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def latest_dataset_cache_root(glob_path: Path = DEFAULT_DATASET_CACHE_GLOB) -> Path | None:
    roots = [
        path for path in glob_path.parent.glob(glob_path.name)
        if path.is_dir()
        and (path / "talk_play_data-challenge-dataset-train.arrow").exists()
        and (path / "talk_play_data-challenge-dataset-test.arrow").exists()
    ]
    if not roots:
        return None
    return max(roots, key=lambda path: path.stat().st_mtime)


def iter_arrow_rows(path: Path) -> Iterable[dict[str, Any]]:
    with ipc.open_stream(path) as reader:
        for batch in reader:
            yield from batch.to_pylist()


def norm_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(v) for v in value if v is not None)
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("&", " and ")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    constraint = row.get("positive_constraint") or {}
    gold = row.get("gold_metadata") or {}
    text = row.get("requested_text") or row.get("evidence", {}).get("user_text") or ""
    return {
        "split": row.get("split"),
        "session_id": row.get("session_id"),
        "turn_number": row.get("turn_number"),
        "constraint": constraint,
        "gold_track": gold.get("track_name"),
        "gold_artist": gold.get("artist_name"),
        "gold_album": gold.get("album_name"),
        "gold_release_date": gold.get("release_date"),
        "text": text[:360],
    }


def user_turns(row: dict[str, Any]) -> Iterable[tuple[int, str]]:
    for turn in row.get("conversations") or []:
        if turn.get("role") == "user":
            yield int(turn.get("turn_number") or 0), turn.get("content") or ""


def broad_semantic_matches(text: str) -> list[str]:
    return [
        name for name, pattern in BROAD_SEMANTIC_PATTERNS.items()
        if pattern.search(text)
    ]


def broad_semantic_request_stats(
    *,
    dataset_root: Path | None = None,
    split_file: Path = SPLIT_FILE,
) -> dict[str, Any]:
    dataset_root = dataset_root or latest_dataset_cache_root()
    if dataset_root is None:
        return {
            "available": False,
            "reason": "cached TalkPlayData-Challenge-Dataset Arrow files not found",
            "by_split": {},
            "examples": {},
        }

    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    split_ids = {
        "train_lt": set(split_data["train_a"]),
        "val": set(split_data["val"]),
    }
    paths = {
        "train": dataset_root / "talk_play_data-challenge-dataset-train.arrow",
        "devset": dataset_root / "talk_play_data-challenge-dataset-test.arrow",
    }
    stats: dict[str, Any] = {
        split: {
            "sessions": 0,
            "user_turns": 0,
            "semantic_sessions": 0,
            "semantic_turns": 0,
            "category_turns": {name: 0 for name in BROAD_SEMANTIC_PATTERNS},
            "category_sessions": {name: 0 for name in BROAD_SEMANTIC_PATTERNS},
        }
        for split in SPLITS
    }
    examples: dict[str, list[dict[str, Any]]] = {name: [] for name in BROAD_SEMANTIC_PATTERNS}

    def consume(split: str, row: dict[str, Any]) -> None:
        stats[split]["sessions"] += 1
        session_categories: set[str] = set()
        session_has_semantic = False
        for turn_number, text in user_turns(row):
            stats[split]["user_turns"] += 1
            categories = broad_semantic_matches(text)
            if not categories:
                continue
            session_has_semantic = True
            stats[split]["semantic_turns"] += 1
            for category in categories:
                stats[split]["category_turns"][category] += 1
                session_categories.add(category)
                if len(examples[category]) < 4:
                    examples[category].append({
                        "split": split,
                        "session_id": row.get("session_id"),
                        "turn_number": turn_number,
                        "category": category,
                        "text": text[:360],
                    })
        if session_has_semantic:
            stats[split]["semantic_sessions"] += 1
        for category in session_categories:
            stats[split]["category_sessions"][category] += 1

    for row in iter_arrow_rows(paths["train"]):
        sid = row["session_id"]
        if sid in split_ids["train_lt"]:
            consume("train_lt", row)
        elif sid in split_ids["val"]:
            consume("val", row)
    for row in iter_arrow_rows(paths["devset"]):
        consume("devset", row)

    return {
        "available": True,
        "dataset_root": str(dataset_root),
        "count_unit": "visible user turns with broad semantic request language",
        "read": (
            "These are high-recall lexical counts over user turns, not correction "
            "records and not request-satisfying target sets."
        ),
        "by_split": stats,
        "examples": examples,
    }


def token_overlap_fraction(needle_norm: str, haystack_norm: str) -> float:
    needle_tokens = {tok for tok in needle_norm.split() if len(tok) > 2}
    haystack_tokens = {tok for tok in haystack_norm.split() if len(tok) > 2}
    if not needle_tokens:
        return 0.0
    return len(needle_tokens & haystack_tokens) / len(needle_tokens)


def family_counts(summary_paths: dict[str, Path], family: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for split in SPLITS:
        summary = load_json(summary_paths[split])
        out[split] = int((summary.get("by_family") or {}).get(family, 0))
    return out


def album_literal_stats(paths: dict[str, Path]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    suspicious_examples: list[dict[str, Any]] = []
    literal_examples: list[dict[str, Any]] = []
    for split, path in paths.items():
        total = 0
        literal = 0
        suspicious = 0
        gold_already_satisfies = 0
        literal_gold_already_satisfies = 0
        literal_gold_violates = 0
        for row in iter_jsonl(path):
            if row.get("family") != "hard_album_constraint":
                continue
            total += 1
            album_norm = norm_text((row.get("positive_constraint") or {}).get("album_norm"))
            gold_album_norm = norm_text((row.get("gold_metadata") or {}).get("album_name"))
            text_norm = norm_text((row.get("evidence") or {}).get("request_span") or row.get("requested_text"))
            is_literal = bool(album_norm and album_norm in text_norm)
            overlap = token_overlap_fraction(album_norm, text_norm)
            is_suspicious = bool(album_norm and not is_literal and overlap < 0.5)
            gold_satisfies_album = bool(album_norm and album_norm == gold_album_norm)
            literal += int(is_literal)
            suspicious += int(is_suspicious)
            gold_already_satisfies += int(gold_satisfies_album)
            literal_gold_already_satisfies += int(is_literal and gold_satisfies_album)
            literal_gold_violates += int(is_literal and not gold_satisfies_album)
            if is_literal and len(literal_examples) < 3:
                literal_examples.append(compact_row(row))
            if is_suspicious and len(suspicious_examples) < 5:
                example = compact_row(row)
                example["album_request_token_overlap"] = round(overlap, 3)
                suspicious_examples.append(example)
        stats[split] = {
            "hard_album_rows": total,
            "album_norm_literal_in_request": literal,
            "album_norm_not_literal_in_request": total - literal,
            "suspicious_nonliteral_resolution": suspicious,
            "gold_album_already_satisfies_constraint": gold_already_satisfies,
            "literal_gold_already_satisfies_constraint": literal_gold_already_satisfies,
            "literal_gold_violates_constraint": literal_gold_violates,
        }
    return {
        "by_split": stats,
        "literal_examples": literal_examples,
        "suspicious_examples": suspicious_examples,
    }


def year_pattern_stats(paths: dict[str, Path]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    examples: dict[str, list[dict[str, Any]]] = {
        "future_decade_parse": [],
        "truncated_range": [],
        "late_90s_early_2000s_truncated": [],
        "release_metadata_language": [],
    }
    for split, path in paths.items():
        counts = {
            "hard_year_rows": 0,
            "decade_word": 0,
            "release_language": 0,
            "popularity_or_percent": 0,
            "explicit_range_text": 0,
            "future_decade_parse": 0,
            "late_90s_early_2000s_text": 0,
        }
        for row in iter_jsonl(path):
            if row.get("family") != "hard_year_range_constraint":
                continue
            counts["hard_year_rows"] += 1
            text = (row.get("evidence") or {}).get("request_span") or row.get("requested_text") or ""
            constraint = row.get("positive_constraint") or {}
            start = int(constraint.get("start", -1))
            end = int(constraint.get("end", -1))
            if DECADE_WORD_RE.search(text):
                counts["decade_word"] += 1
            if RELEASE_RE.search(text):
                counts["release_language"] += 1
                if len(examples["release_metadata_language"]) < 4:
                    examples["release_metadata_language"].append(compact_row(row))
            if POPULARITY_RE.search(text):
                counts["popularity_or_percent"] += 1
            if RANGE_RE.search(text):
                counts["explicit_range_text"] += 1
                if start == end and len(examples["truncated_range"]) < 4:
                    examples["truncated_range"].append(compact_row(row))
            if start >= 2030:
                counts["future_decade_parse"] += 1
                if len(examples["future_decade_parse"]) < 4:
                    examples["future_decade_parse"].append(compact_row(row))
            if LATE_90S_EARLY_2000S_RE.search(text):
                counts["late_90s_early_2000s_text"] += 1
                if start == 1990 and end == 1999 and len(examples["late_90s_early_2000s_truncated"]) < 4:
                    examples["late_90s_early_2000s_truncated"].append(compact_row(row))
        stats[split] = counts
    return {"by_split": stats, "examples": examples}


def build_audit() -> dict[str, Any]:
    album_counts = family_counts(BASE_SUMMARY, "hard_album_constraint")
    year_counts = family_counts(YEAR_SUMMARY, "hard_year_range_constraint")
    album_stats = album_literal_stats(BASE_JSONL)
    year_stats = year_pattern_stats(YEAR_JSONL)
    semantic_stats = broad_semantic_request_stats()
    semantic_screening = load_json(BROAD_SCREENING_JSON) if BROAD_SCREENING_JSON.exists() else None
    return {
        "date": "2026-06-22",
        "audit": "request-additional-family-mining-v0.9",
        "purpose": (
            "Record candidate proxy-boundary families discovered during mining, "
            "and explain why they are not admitted as correction families yet."
        ),
        "source_files": {
            "base_summaries": {split: str(path.relative_to(ROOT)) for split, path in BASE_SUMMARY.items()},
            "year_summaries": {split: str(path.relative_to(ROOT)) for split, path in YEAR_SUMMARY.items()},
            "album_jsonl": {split: str(path.relative_to(ROOT)) for split, path in BASE_JSONL.items()},
            "year_jsonl_readouts": {split: str(path.relative_to(ROOT)) for split, path in YEAR_JSONL.items()},
            "dialogue_arrow_root": semantic_stats.get("dataset_root"),
            "broad_semantic_screening": (
                str(BROAD_SCREENING_JSON.relative_to(ROOT)) if semantic_screening else None
            ),
        },
        "candidates": [
            {
                "family": "hard album constraint",
                "detector_status": "implemented_in_v09_base_sidecar",
                "counts": album_counts,
                "decision": "mined_not_admitted",
                "reason": (
                    "The current parser finds real album conflicts, but fuzzy album-name "
                    "resolution also produces visible false actions. On val, only "
                    f"{album_stats['by_split']['val']['album_norm_literal_in_request']}/"
                    f"{album_stats['by_split']['val']['hard_album_rows']} album constraints "
                    "resolve to an album string literally present in the request, and "
                    f"{album_stats['by_split']['val']['suspicious_nonliteral_resolution']}/"
                    f"{album_stats['by_split']['val']['hard_album_rows']} have low token overlap "
                    "with the visible request. On train, "
                    f"{album_stats['by_split']['train_lt']['gold_album_already_satisfies_constraint']}/"
                    f"{album_stats['by_split']['train_lt']['hard_album_rows']} album sidecar actions "
                    "would mask a gold item whose metadata already satisfies the parsed album."
                ),
                "required_to_admit": [
                    "tighten extraction to quoted or explicit album titles",
                    "resolve albums jointly with artist/title context",
                    "audit changed actions before using mask-gold labels",
                    "show an intervention that preserves official nDCG and improves auxiliary request-positive nDCG",
                ],
                "diagnostics": album_stats,
            },
            {
                "family": "hard year/decade constraint",
                "detector_status": "implemented_but_off_by_default",
                "counts_when_enabled": year_counts,
                "decision": "mined_not_admitted",
                "reason": (
                    "The enabled detector has broad coverage but weak action semantics: "
                    "release_date can mean reissue or catalog date, open-ended era requests "
                    "are treated as hard single intervals, and the current parser truncates "
                    "ranges such as 2014-2015 to the first year."
                ),
                "required_to_admit": [
                    "parse explicit ranges and conjunctions instead of first-year only",
                    "handle bare decades like 30s without mapping them to future decades",
                    "separate era preference from hard release-date constraint",
                    "use metadata that reflects original recording/release where available",
                    "audit changed actions before any training or evaluation correction",
                ],
                "diagnostics": year_stats,
            },
            {
                "family": "genre, mood, instrumentation, language, popularity",
                "detector_status": "not_catalog_resolved_in_current_contract",
                "counts": {
                    split: semantic_stats.get("by_split", {}).get(split, {}).get("semantic_turns", 0)
                    for split in SPLITS
                },
                "count_unit": semantic_stats.get("count_unit"),
                "decision": "not_admitted_no_reusable_target_set",
                "reason": (
                    "These requests are frequent and important, but the current catalog "
                    "does not expose high-precision fields that turn them into reusable "
                    "request-satisfying target sets without model or human adjudication."
                ),
                "required_to_admit": [
                    "define a trusted metadata source or adjudication protocol",
                    "measure detector and action precision on held-out rows",
                    "predeclare the intervention and non-inferiority margin",
                ],
                "diagnostics": semantic_stats,
                "screening": semantic_screening,
            },
        ],
        "paper_implication": (
            "The work should not claim that exact/version was selected because it was the "
            "only possible family. It should claim that a uniform screening-and-admission "
            "workflow examined several visible proxy-boundary families and admitted only "
            "the one that currently has high-precision actions and positive development evidence."
        ),
    }


def markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Additional Family Mining Audit v0.9",
        "",
        f"Date: {audit['date']}",
        "",
        audit["purpose"],
        "",
        "## Decision Summary",
        "",
        "| candidate family | status | count/readout | reason |",
        "|---|---|---:|---|",
    ]
    for candidate in audit["candidates"]:
        if "counts_when_enabled" in candidate:
            counts = candidate["counts_when_enabled"]
        else:
            counts = candidate.get("counts", {})
        count_text = ", ".join(f"{split}: {counts.get(split, 0)}" for split in SPLITS) if counts else "not counted"
        lines.append(
            f"| {candidate['family']} | {candidate['decision']} | {count_text} | {candidate['reason']} |"
        )

    album = audit["candidates"][0]
    album_diag = album["diagnostics"]
    lines.extend([
        "",
        "## Album Diagnostic",
        "",
        "| split | hard album rows | literal album string in request | nonliteral resolution | gold already satisfies album | strict literal violations |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for split in SPLITS:
        row = album_diag["by_split"][split]
        lines.append(
            f"| {split} | {row['hard_album_rows']} | {row['album_norm_literal_in_request']} | "
            f"{row['album_norm_not_literal_in_request']} "
            f"({row['suspicious_nonliteral_resolution']} suspicious) | "
            f"{row['gold_album_already_satisfies_constraint']} | "
            f"{row['literal_gold_violates_constraint']} |"
        )
    lines.extend([
        "",
        (
            "Strict literal violations are the salvage subset: the parsed album "
            "appears literally in the request and the policy-selected gold metadata "
            "does not already satisfy that album. This subset is promising but still "
            "too sparse on dev/val and often overlaps exact-title requests, so it "
            "is not admitted as a separate correction family."
        ),
    ])
    lines.extend([
        "",
        "Representative suspicious album resolutions:",
        "",
    ])
    for row in album_diag["suspicious_examples"]:
        constraint = row["constraint"].get("album_norm")
        lines.append(
            f"- `{row['split']}` `{row['session_id']}` turn {row['turn_number']}: "
            f"resolved album `{constraint}` while gold album is `{row['gold_album']}` "
            f"(token overlap {row['album_request_token_overlap']})."
        )

    year = audit["candidates"][1]
    year_diag = year["diagnostics"]
    lines.extend([
        "",
        "## Year/Decade Diagnostic",
        "",
        "| split | hard year rows | decade words | release-language rows | explicit ranges | future-decade parses |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for split in ("val", "devset"):
        row = year_diag["by_split"][split]
        lines.append(
            f"| {split} | {row['hard_year_rows']} | {row['decade_word']} | "
            f"{row['release_language']} | {row['explicit_range_text']} | {row['future_decade_parse']} |"
        )
    lines.extend([
        "",
        "Representative parser risks:",
        "",
    ])
    for kind, rows in year_diag["examples"].items():
        if not rows:
            continue
        row = rows[0]
        c = row["constraint"]
        lines.append(
            f"- `{kind}`: `{row['split']}` `{row['session_id']}` turn {row['turn_number']} "
            f"parsed `{c.get('start')}-{c.get('end')}` with gold release `{row['gold_release_date']}`."
        )

    semantic = audit["candidates"][2]
    semantic_diag = semantic["diagnostics"]
    lines.extend([
        "",
        "## Broad Semantic Request Diagnostic",
        "",
        semantic_diag.get("read", "Counts unavailable."),
        "",
        "| split | user turns | broad semantic turns | genre/style | mood/energy/tempo | instrumentation/vocal | language/culture | popularity/recency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for split in SPLITS:
        row = semantic_diag.get("by_split", {}).get(split, {})
        category_turns = row.get("category_turns", {})
        lines.append(
            f"| {split} | {row.get('user_turns', 0)} | {row.get('semantic_turns', 0)} | "
            f"{category_turns.get('genre_style', 0)} | "
            f"{category_turns.get('mood_energy_tempo', 0)} | "
            f"{category_turns.get('instrumentation_vocal', 0)} | "
            f"{category_turns.get('language_culture', 0)} | "
            f"{category_turns.get('popularity_recency', 0)} |"
        )
    lines.extend([
        "",
        (
            "Read: broad semantic requests are common enough that ignoring them would "
            "make the mining story look too narrow, but these counts are not labels. "
            "They lack a trusted catalog-resolved target set under the current "
            "contract, so they remain mined-but-not-admitted."
        ),
        "",
        "Representative broad semantic requests:",
        "",
    ])
    for category, rows in semantic_diag.get("examples", {}).items():
        if not rows:
            continue
        row = rows[0]
        lines.append(
            f"- `{category}`: `{row['split']}` `{row['session_id']}` turn "
            f"{row['turn_number']}: {row['text']}"
        )
    screening = semantic.get("screening") or {}
    if screening.get("available"):
        lines.extend([
            "",
            "## Broad Semantic Screening Sample",
            "",
            screening["read"],
            "",
            "| category | sampled | request-like by rubric | compound/mixed by rubric | trusted catalog-resolvable | main blocker |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for category, row in screening["categories"].items():
            lines.append(
                f"| {category} | {row['sampled_matches']} | "
                f"{row['visible_request_like_by_rubric']}/{row['sampled_matches']} | "
                f"{row['compound_or_mixed_by_rubric']}/{row['sampled_matches']} | "
                f"{row['trusted_catalog_resolvable_under_current_contract']}/{row['sampled_matches']} | "
                f"{row['main_blocker']} |"
            )
        lines.extend([
            "",
            (
                "Read: the deterministic devset screen suggests these regex hits are "
                "usually visible request language, often compound with other constraints, "
                "but still have no trusted catalog-resolved target set under the current "
                "contract. This supports rejection as principled, not accidental."
            ),
        ])

    lines.extend([
        "",
        "## Paper Implication",
        "",
        audit["paper_implication"],
        "",
        "## Sources",
        "",
    ])
    for group, files in audit["source_files"].items():
        if isinstance(files, dict):
            for split, path in files.items():
                lines.append(f"- `{group}.{split}`: `{path}`")
        else:
            lines.append(f"- `{group}`: `{files}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = ap.parse_args(argv)

    audit = build_audit()
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.write_text(markdown(audit), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "md": str(args.md_out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
