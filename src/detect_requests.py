"""Build high-precision request-aware label correction sidecars.

This script does not edit the organizer data. It emits JSONL records keyed by
(session_id, turn_number). Each record describes either:

  * extra positives for visible request-satisfying items, or
  * conservative masks/downweights for policy labels that visibly violate a
    hard request or rejection constraint.

The default detectors are intentionally precision-biased. They are dataset-aware
only at the catalog-schema boundary; the reusable protocol is:

  visible constraint -> catalog-resolved satisfying set -> policy-label check.

Usage examples:

  python scripts/detect_requests.py --split devset \
    --out docs/evidence/request_corrections_devset.jsonl

  python scripts/detect_requests.py --split train_lt \
    --out docs/evidence/request_corrections_train_lt.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(os.environ.get("MCRS_EXPLORE_ROOT", str(Path(__file__).resolve().parent.parent)))
SPLIT_FILE = Path(__file__).resolve().parent.parent / "data/train_val_split_seed42.json"
DETECTOR_VERSION = "request-corrections-v0.9"

NEGATIVE_CUES = (
    "not ",
    "no more",
    "don't want",
    "do not want",
    "stop playing",
    "avoid",
    "skip",
    "different artist",
    "another artist",
    "new artist",
    "new artists",
    "change artist",
)

STRICT_SWITCHAWAY_RE = re.compile(
    r"\b("
    r"different(?: \w+){0,4} (?:artists?|bands?)|"
    r"other(?: \w+){0,4} (?:artists?|bands?)|"
    r"another artist entirely|completely different artist|"
    r"different artist this time|new artist this time|"
    r"new blood|branch out|broaden my horizons|move beyond|"
    r"no more|not more|not by|not just more|"
    r"do not want more|don t want more|heard enough|different sources"
    r")\b",
    re.I,
)
SOFT_SWITCHAWAY_RE = re.compile(
    r"\b(maybe|possibly|open to|generally)\b.{0,50}\b(different|another|other)\s+(artists?|bands?)\b|"
    r"\b(different|another|other)\s+(artists?|bands?)\b.{0,50}\b(maybe|possibly|open to|generally)\b",
    re.I,
)
PAST_NEW_ARTIST_RE = re.compile(r"\b(found|discovered|have found|i'?ve found).{0,30}\bnew artists?\b", re.I)

SOFT_CUES = (
    "perhaps",
    "maybe",
    "like",
    "similar to",
    "inspired by",
    "reminds me of",
    "kind of",
    "sort of",
)

REFERENCE_CONTEXT_PATTERNS = (
    r"\b(?:couldn'?t|could not|can'?t|cannot)\s+find\s+['\"]{title}['\"]",
    r"^\s*find\s+['\"]{title}['\"][^.?!]{0,100}\b(?:current selection|catalog|library|selection)\b",
    r"\bsimilar\b[^.?!]{0,120}\bto\s+['\"]{title}['\"]",
    r"\bsample\s+to\s+['\"]{title}['\"]",
    r"\b(?:lyric|lyrics)\b[^.?!]{0,120}\b(?:mention|mentions|about|include|says)\b[^.?!]{0,80}['\"]{title}['\"]",
    r"\bparod(?:y|ies|ied|ying|ic|ies of)\b[^.?!]{0,120}['\"]{title}['\"]",
    r"\blike\s+[^.?\n]{0,180}['\"]{title}['\"]",
    r"\bin\s+the\s+vein\s+of\s+[^.?\n]{0,180}['\"]{title}['\"]",
    r"\b(?:key|tempo|bpm)\b[^.?\n]{0,140}\bof\s+[^.?\n]{0,100}['\"]{title}['\"]",
    r"\b(?:songs?|tracks?|music)\s+from\s+[^.?\n]{0,100}['\"]{title}['\"]",
)

GENERIC_ARTIST_PHRASES = {
    "a",
    "an",
    "the",
    "song",
    "songs",
    "track",
    "tracks",
    "music",
    "artist",
    "artists",
    "band",
    "bands",
    "group",
    "groups",
    "male",
    "female",
    "new",
    "different",
    "another",
    "other",
    "rock",
    "pop",
    "electronic",
    "folk",
    "country",
    "hip hop",
    "rap",
    "metal",
    "jazz",
    "classical",
}
BAD_STRICT_ARTIST_NORMS = {
    "other",
    "space",
}
OPTIONAL_ARTIST_RE = re.compile(
    r"\b("
    r"or similar (?:artists?|bands?)|"
    r"or a similar (?:artist|band)|"
    r"or other [a-z0-9 ]{0,40}(?:artists?|bands?)|"
    r"similar (?:artists?|bands?)|"
    r"similar space|similar aesthetic|"
    r"even if (?:they re|they are|it s|it is)? ?not (?:directly )?by"
    r")\b",
    re.I,
)
ROLE_ARTIST_RE = re.compile(
    r"\b(produced by|producer|featuring|feat\.?|vocals?|vocalist)\b",
    re.I,
)
ACK_ONLY_RE = re.compile(
    r"\b(thanks? for helping me find|finally remembered it|that'?s the one)\b",
    re.I,
)

STRICT_VERSION_WORDS = (
    "remaster",
    "remastered",
    "remix",
    "radio edit",
    "single edit",
    "edit",
    "mono",
    "stereo",
    "version",
    "clean",
    "explicit",
)
VERSION_MODIFIERS = (
    "acoustic",
    "live",
    "remix",
    "remaster",
    "remastered",
    "radio edit",
    "single edit",
    "demo",
)

QUOTE_RE = re.compile(
    r"\"(?P<double>[^\"]{2,120})\"|(?<![A-Za-z0-9])'(?P<single>[^']{2,120})'(?![A-Za-z0-9])"
)
BY_RE = re.compile(r"\bby\s+([A-Za-z0-9][A-Za-z0-9 .,&'’!?:/-]{1,80})", re.I)
FROM_ALBUM_RE = re.compile(r"\bfrom\s+(?:the\s+)?album\s+['\"]?([^'\".,;!?]{2,100})['\"]?", re.I)
FROM_NAMED_ALBUM_RE = re.compile(r"\bfrom\s+(?:the\s+)?['\"]?([^'\".,;!?]{2,100})['\"]?", re.I)
DECADE_RE = re.compile(r"\b(?:from\s+the\s+)?(?:19|20)?(\d0)s\b", re.I)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
ACTIVE_QUOTE_REQUEST_RE = re.compile(
    r"\b(play|hear|listen to|put on|pull up|find|search for|looking for|want|request)\b",
    re.I,
)
ACK_PREFIX_RE = re.compile(r"^\s*(yes|yeah|yep|perfect|great|nice|love it|that'?s|that was)\b", re.I)
REQUEST_START_RE = re.compile(
    r"\b(can you|could you|please|play|recommend|suggest|give me|show me|find|"
    r"finding|looking for|i want|i need|need|do you have|any recommendations|"
    r"any suggestions|what other|what else|what about|what'?s|what is|how about|"
    r"trying to|keen on)\b",
    re.I,
)
QUOTE_CONTEXT_REJECT_RE = re.compile(
    r"\b("
    r"album called|album titled|"
    r"similar to what (?:you'?d|you would) find on|"
    r"what (?:you'?d|you would) find on|"
    r"found on|"
    r"helping me find|starting with|along with|between|"
    r"difference in|"
    r"feels almost|feels like|sound that feels|"
    r"kick drum in"
    r")\b",
    re.I,
)


def norm_text(value: Any) -> str:
    """Lowercase ASCII-ish normalization for catalog joins and phrase checks."""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value if x is not None)
    if value is None:
        return ""
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("’", "'").replace("&", " and ")
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def norm_phrase_contains(haystack: str, needle: str) -> bool:
    return f" {needle} " in f" {haystack} "


def strict_switchaway_request(text: str) -> bool:
    text_norm = norm_text(text)
    if PAST_NEW_ARTIST_RE.search(text_norm):
        return False
    if SOFT_SWITCHAWAY_RE.search(text_norm):
        return False
    return STRICT_SWITCHAWAY_RE.search(text_norm) is not None


def strict_hard_artist_constraint(request_text: str, constraint: dict[str, Any]) -> bool:
    text_norm = norm_text(request_text)
    if constraint.get("kind") not in {"artist", "artist_any"}:
        return True
    artists = set(constraint.get("artist_norms") or [])
    if constraint.get("artist_norm"):
        artists.add(constraint["artist_norm"])
    if not artists or artists & BAD_STRICT_ARTIST_NORMS:
        return False
    if OPTIONAL_ARTIST_RE.search(text_norm):
        return False
    if ROLE_ARTIST_RE.search(text_norm):
        return False
    if ACK_ONLY_RE.search(text_norm):
        return False
    return True


def reference_context(request_text: str, title_raw: str) -> bool:
    title = re.escape(title_raw)
    return any(
        re.search(pattern.replace("{title}", title), request_text, re.I)
        for pattern in REFERENCE_CONTEXT_PATTERNS
    )


def canon_title(value: Any) -> str:
    s = norm_text(value)
    s = re.sub(r"\b(" + "|".join(re.escape(w) for w in STRICT_VERSION_WORDS) + r")\b", " ", s)
    s = re.sub(r"\b\d{4}\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def filter_version_modifiers(request_text: str, title_raw: str, exact_ids: list[str], catalog: CatalogIndex) -> list[str]:
    request_norm = norm_text(unquoted_text(request_text))
    title_norm = norm_text(title_raw)
    needed = [
        modifier for modifier in VERSION_MODIFIERS
        if modifier in request_norm and modifier not in title_norm
    ]
    if not needed:
        return exact_ids
    matching = []
    for tid in exact_ids:
        track_norm = norm_text(catalog.by_id[tid].get("track_name"))
        if all(modifier in track_norm for modifier in needed):
            matching.append(tid)
    return matching


def first(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def norm_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [norm_text(x) for x in value if norm_text(x)]
    n = norm_text(value)
    return [n] if n else []


def canon_values(value: Any) -> list[str]:
    return [canon_title(v) for v in norm_values(value) if canon_title(v)]


def metadata_summary(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "track_id": row.get("track_id"),
        "track_name": first(row.get("track_name")),
        "artist_name": first(row.get("artist_name")),
        "album_name": first(row.get("album_name")),
        "release_date": row.get("release_date"),
        "popularity": row.get("popularity"),
    }


def release_year(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    raw = row.get("release_date")
    if not raw:
        return None
    m = re.match(r"(\d{4})", str(raw))
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class CatalogIndex:
    by_id: dict[str, dict[str, Any]]
    by_title: dict[str, list[str]]
    by_canon_title_artist: dict[tuple[str, str], list[str]]
    by_artist: dict[str, list[str]]
    by_album: dict[str, list[str]]


def build_catalog_index(rows: Iterable[dict[str, Any]]) -> CatalogIndex:
    by_id: dict[str, dict[str, Any]] = {}
    by_title: dict[str, list[str]] = defaultdict(list)
    by_canon_title_artist: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_artist: dict[str, list[str]] = defaultdict(list)
    by_album: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        tid = row["track_id"]
        by_id[tid] = row
        titles = norm_values(row.get("track_name"))
        artists = norm_values(row.get("artist_name"))
        albums = norm_values(row.get("album_name"))
        ctitles = canon_values(row.get("track_name"))
        for title in titles:
            by_title[title].append(tid)
        for ctitle in ctitles:
            for artist in artists:
                if ctitle and artist:
                    by_canon_title_artist[(ctitle, artist)].append(tid)
        for artist in artists:
            by_artist[artist].append(tid)
        for album in albums:
            by_album[album].append(tid)
    return CatalogIndex(
        by_id=by_id,
        by_title=dict(by_title),
        by_canon_title_artist=dict(by_canon_title_artist),
        by_artist=dict(by_artist),
        by_album=dict(by_album),
    )


def gold_for_turn(conversations: list[dict[str, Any]], turn_number: int) -> str | None:
    for row in conversations:
        if int(row.get("turn_number", -1)) == turn_number and row.get("role") == "music":
            return row.get("content")
    return None


def previous_music(conversations: list[dict[str, Any]], turn_number: int) -> str | None:
    prev: str | None = None
    for row in sorted(conversations, key=lambda r: (int(r.get("turn_number", 0)), r.get("role", ""))):
        tn = int(row.get("turn_number", 0))
        if tn >= turn_number:
            break
        if row.get("role") == "music":
            prev = row.get("content")
    return prev


def user_turns(session: dict[str, Any]) -> Iterable[tuple[int, str]]:
    for row in sorted(session["conversations"], key=lambda r: (int(r["turn_number"]), r["role"])):
        if row.get("role") == "user":
            yield int(row["turn_number"]), (row.get("content") or "").strip()


def quoted_titles(text: str) -> list[str]:
    out = []
    for m in QUOTE_RE.finditer(text):
        raw = (m.group("double") or m.group("single") or "").strip()
        n = norm_text(raw)
        if n and len(n) >= 2:
            out.append(raw)
    return out


def active_request_span(text: str) -> str:
    """Return the likely current ask, dropping prior-track acknowledgements."""
    matches = []
    for m in REQUEST_START_RE.finditer(text):
        before = text[max(0, m.start() - 25):m.start()]
        if re.search(r"(helping me|you'?d|you would)\s+$", before, re.I):
            continue
        matches.append(m)
    if not matches:
        return text
    return text[matches[-1].start():]


def active_quoted_requests(text: str) -> list[str]:
    """Quoted titles that are likely the current request, not a prior-track mention."""
    out = []
    for m in QUOTE_RE.finditer(text):
        raw = (m.group("double") or m.group("single") or "").strip()
        prefix = text[max(0, m.start() - 80):m.start()]
        suffix = text[m.end():m.end() + 40]
        near_prefix = norm_text(text[max(0, m.start() - 25):m.start()])
        near_suffix = norm_text(suffix[:20])
        if QUOTE_CONTEXT_REJECT_RE.search(prefix):
            continue
        if near_suffix.startswith("album"):
            continue
        if (
            re.search(r"\balbum\s*$", near_prefix)
            or near_prefix.endswith("from")
            or near_prefix.endswith("from the")
            or re.search(r"\bfrom\s+(?:the\s+|their\s+|an\s+|a\s+)?album\b", prefix, re.I)
        ):
            continue
        if (
            near_prefix.endswith("like")
            or near_prefix.endswith("similar to")
            or near_prefix.endswith("almost")
            or near_prefix.endswith("both")
            or near_prefix.endswith("less")
            or near_prefix.endswith("more")
        ):
            continue
        if ACK_PREFIX_RE.search(text[:m.start()]) and not ACTIVE_QUOTE_REQUEST_RE.search(prefix):
            continue
        if re.match(r"\s+(is|was|sounds|feels|seems|has|had|provides|really|definitely)\b", suffix, re.I):
            continue
        if re.match(r"\s+(always|energy)\b", suffix, re.I):
            continue
        if re.match(r"\s+and\s+this\s+one\b", suffix, re.I):
            continue
        if re.match(r"\s+by\s+[^.,;!?]{1,80}\s+(felt|was|is|sounds|seems|has|had)\b", suffix, re.I):
            continue
        if ACTIVE_QUOTE_REQUEST_RE.search(prefix):
            out.append(raw)
    return out


def unquoted_text(text: str) -> str:
    return QUOTE_RE.sub(" ", text)


def has_unquoted_by_clause(text: str) -> bool:
    return BY_RE.search(unquoted_text(text)) is not None


def extract_by_artist(text: str, catalog: CatalogIndex) -> str | None:
    artists = extract_by_artists(text, catalog)
    return artists[0] if artists else None


def extract_by_artists(text: str, catalog: CatalogIndex) -> list[str]:
    m = BY_RE.search(unquoted_text(text))
    if not m:
        return []
    raw = m.group(1)
    raw = re.split(
        r"\b(?:from|with|that|which|but|please|right|now|specifically|"
        r"particularly|as|if|when|where|while|because|since)\b|[;!?]",
        raw,
        1,
        re.I,
    )[0]
    raw = raw.strip(" ,.")
    if re.search(r"\bother\s+(artists?|composers?|bands?|songs?|tracks?)\b", raw, re.I):
        return []
    parts = re.split(r"\s+or\s+|/", raw, flags=re.I)
    out: list[str] = []
    for part in parts:
        found = resolve_artist_phrase(part, catalog)
        if found and found not in out:
            out.append(found)
    return out


def resolve_artist_phrase(raw: str, catalog: CatalogIndex) -> str | None:
    n = norm_text(raw)
    if n in GENERIC_ARTIST_PHRASES or n in {
        "him", "her", "them", "it", "other", "another", "different", "new",
        "artist", "artists", "band", "bands", "leader", "leaders",
        "performer", "performers", "male", "female",
    }:
        return None
    if n in catalog.by_artist:
        return n
    if len(n) < 5:
        return None
    # Prefer a full catalog artist phrase inside the captured span. The reverse
    # match is only allowed for longer aliases like "Beatles" -> "The Beatles";
    # short fragments such as "Bat" must not resolve to unrelated artists.
    hits = [
        a for a in catalog.by_artist
        if len(a) >= 4 and norm_phrase_contains(n, a)
    ]
    if not hits and len(n) >= 6:
        hits = [
            a for a in catalog.by_artist
            if len(a) >= 4 and norm_phrase_contains(a, n)
        ]
    return max(hits, key=len) if hits else None


def extract_album(text: str, catalog: CatalogIndex) -> str | None:
    m = FROM_ALBUM_RE.search(text)
    if not m:
        return None
    n = norm_text(m.group(1))
    if n in catalog.by_album:
        return n
    hits = [a for a in catalog.by_album if len(a) >= 5 and (a in n or n in a)]
    return max(hits, key=len) if hits else None


def extract_exact_album_hint(text: str, catalog: CatalogIndex) -> str | None:
    album = extract_album(text, catalog)
    if album:
        return album
    m = FROM_NAMED_ALBUM_RE.search(text)
    if not m:
        return None
    n = norm_text(m.group(1))
    if n in {"that album", "this album", "the album", "an album", "a album"}:
        return None
    if n in catalog.by_album:
        return n
    hits = [a for a in catalog.by_album if len(a) >= 5 and (a in n or n in a)]
    return max(hits, key=len) if hits else None


def extract_year_constraint(text: str) -> dict[str, int] | None:
    years = [int(y) for y in YEAR_RE.findall(text)]
    if years:
        y = years[0]
        return {"start": y, "end": y}
    m = DECADE_RE.search(text)
    if not m:
        return None
    suffix = int(m.group(1))
    # "90s" means 1990s in this catalog; explicit "2010s" is captured as 10.
    start = 1900 + suffix if suffix >= 50 else 2000 + suffix
    return {"start": start, "end": start + 9}


def track_satisfies_constraint(tid: str, constraint: dict[str, Any], catalog: CatalogIndex) -> bool:
    row = catalog.by_id.get(tid)
    if not row:
        return False
    kind = constraint.get("kind")
    if kind == "artist":
        return constraint.get("artist_norm") in norm_values(row.get("artist_name"))
    if kind == "artist_any":
        artists = set(norm_values(row.get("artist_name")))
        return bool(artists & set(constraint.get("artist_norms") or []))
    if kind == "album":
        return norm_text(row.get("album_name")) == constraint.get("album_norm")
    if kind == "year_range":
        y = release_year(row)
        return y is not None and int(constraint["start"]) <= y <= int(constraint["end"])
    return False


def exact_and_version_records(
    *,
    split: str,
    session: dict[str, Any],
    turn_number: int,
    user_text: str,
    gold_id: str,
    catalog: CatalogIndex,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    request_text = active_request_span(user_text)
    artist_hint = extract_by_artist(request_text, catalog)
    album_hint = extract_exact_album_hint(request_text, catalog)
    gold_row = catalog.by_id.get(gold_id)
    gold_artists = norm_values(gold_row.get("artist_name")) if gold_row else []
    gold_titles = set(norm_values(gold_row.get("track_name"))) if gold_row else set()
    gold_ctitles = set(canon_values(gold_row.get("track_name"))) if gold_row else set()

    for title_raw in active_quoted_requests(request_text):
        if reference_context(request_text, title_raw):
            continue
        title_norm = norm_text(title_raw)
        ctitle = canon_title(title_raw)
        exact_ids = list(catalog.by_title.get(title_norm, []))
        exact_ids = filter_version_modifiers(request_text, title_raw, exact_ids, catalog)
        if artist_hint:
            exact_ids = [tid for tid in exact_ids if artist_hint in norm_values(catalog.by_id[tid].get("artist_name"))]
        if album_hint:
            exact_ids = [tid for tid in exact_ids if album_hint in norm_values(catalog.by_id[tid].get("album_name"))]
        if not artist_hint and not album_hint and has_unquoted_by_clause(request_text):
            # If the user wrote "by X" but X is not catalog-resolved, do not
            # silently accept an arbitrary same-title cover by another artist.
            exact_ids = []
        elif len({norm_text(catalog.by_id[tid].get("artist_name")) for tid in exact_ids}) > 1:
            exact_ids = []
        exact_ids = sorted(set(exact_ids))
        gold_already_satisfies_title = title_norm in gold_titles or ctitle in gold_ctitles
        if exact_ids and gold_id not in exact_ids and not gold_already_satisfies_title:
            records.append({
                "detector_version": DETECTOR_VERSION,
                "split": split,
                "session_id": session["session_id"],
                "turn_number": turn_number,
                "family": "exact_track_request",
                "action": "add_positive",
                "confidence": "high",
                "requested_text": title_raw,
                "additional_track_ids": exact_ids,
                "mask_gold": False,
                "group_weight": 1.0,
                "gold_track_id": gold_id,
                "gold_metadata": metadata_summary(gold_row),
                "evidence": {
                    "user_text": user_text,
                    "request_span": request_text,
                    "artist_hint_norm": artist_hint,
                    "requested_title_norm": title_norm,
                },
            })

        candidate_artists = [artist_hint] if artist_hint else gold_artists
        version_ids: set[str] = set()
        for artist in candidate_artists:
            version_ids.update(catalog.by_canon_title_artist.get((ctitle, artist), []))
        # Version/duplicate rows are useful even when the official gold is one
        # version and the request names another. Keep the policy label.
        version_ids.discard(gold_id)
        version_ids.difference_update(exact_ids)
        if version_ids and ctitle in gold_ctitles:
            records.append({
                "detector_version": DETECTOR_VERSION,
                "split": split,
                "session_id": session["session_id"],
                "turn_number": turn_number,
                "family": "version_duplicate_equivalence",
                "action": "add_positive",
                "confidence": "medium_high",
                "requested_text": title_raw,
                "additional_track_ids": sorted(version_ids),
                "mask_gold": False,
                "group_weight": 1.0,
                "gold_track_id": gold_id,
                "gold_metadata": metadata_summary(gold_row),
                "evidence": {
                    "user_text": user_text,
                    "request_span": request_text,
                    "canonical_title": ctitle,
                    "artist_hint_norm": artist_hint,
                },
            })
    return records


def hard_constraint_records(
    *,
    split: str,
    session: dict[str, Any],
    turn_number: int,
    user_text: str,
    gold_id: str,
    catalog: CatalogIndex,
    include_year_constraints: bool = False,
    strict_hard_artist: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    request_text = active_request_span(user_text)
    text_norm = norm_text(request_text)
    if not REQUEST_START_RE.search(request_text):
        return records
    if any(cue in text_norm for cue in SOFT_CUES):
        return records

    constraints: list[dict[str, Any]] = []
    artists = extract_by_artists(request_text, catalog)
    if len(artists) == 1:
        constraints.append({"kind": "artist", "artist_norm": artists[0]})
    elif len(artists) > 1:
        constraints.append({"kind": "artist_any", "artist_norms": artists})
    album = extract_album(request_text, catalog)
    if album:
        constraints.append({"kind": "album", "album_norm": album})
    year = extract_year_constraint(request_text) if include_year_constraints else None
    if year:
        constraints.append({"kind": "year_range", **year})

    for c in constraints:
        if strict_hard_artist and not strict_hard_artist_constraint(request_text, c):
            continue
        if track_satisfies_constraint(gold_id, c, catalog):
            continue
        family_kind = "artist" if c["kind"] in {"artist", "artist_any"} else c["kind"]
        family = f"hard_{family_kind}_constraint"
        records.append({
            "detector_version": DETECTOR_VERSION,
            "split": split,
            "session_id": session["session_id"],
            "turn_number": turn_number,
            "family": family,
            "action": "constraint_positive_and_mask_gold",
            "confidence": "medium_high",
            "requested_text": user_text,
            "additional_track_ids": [],
            "positive_constraint": c,
            "mask_gold": True,
            "group_weight": 1.0,
            "gold_track_id": gold_id,
            "gold_metadata": metadata_summary(catalog.by_id.get(gold_id)),
            "evidence": {"user_text": user_text, "request_span": request_text},
        })
    return records


def rejection_records(
    *,
    split: str,
    session: dict[str, Any],
    turn_number: int,
    user_text: str,
    gold_id: str,
    catalog: CatalogIndex,
    strict_switchaway: bool = False,
) -> list[dict[str, Any]]:
    text_norm = norm_text(user_text)
    if not any(cue in text_norm for cue in NEGATIVE_CUES):
        return []

    prev_id = previous_music(session["conversations"], turn_number)
    if not prev_id:
        return []
    gold_row = catalog.by_id.get(gold_id)
    prev_row = catalog.by_id.get(prev_id)
    if not gold_row or not prev_row:
        return []

    gold_artists = set(norm_values(gold_row.get("artist_name")))
    prev_artists = set(norm_values(prev_row.get("artist_name")))
    same_track = gold_id == prev_id
    same_artist = bool(gold_artists & prev_artists)
    explicit_artist = extract_by_artist(user_text, catalog)
    negative_artist_violation = bool(
        explicit_artist
        and explicit_artist in gold_artists
        and re.search(r"\b(not by|avoid|no more|don'?t want|do not want)\b", user_text, re.I)
    )
    soft_switch = re.search(r"\b(maybe|perhaps|possibly)\s+(?:from\s+|by\s+)?(?:a\s+)?(?:different|another|new)\s+artist\b", text_norm)
    strong_switch = re.search(
        r"\b("
        r"different artists|other artists|other bands|different bands|"
        r"another artist entirely|completely different artist|"
        r"different artist this time|new artist this time|new artists|new artist|new bands|"
        r"new blood|branch out|broaden my horizons|move beyond|"
        r"no more|not more|not by"
        r")\b",
        text_norm,
    )
    switch_violation = bool(strong_switch and not soft_switch) and same_artist and " or " not in text_norm
    if strict_switchaway and switch_violation:
        switch_violation = strict_switchaway_request(user_text)
    repeat_violation = same_track and any(cue in text_norm for cue in ("not ", "don't want", "do not want", "skip"))

    if not (negative_artist_violation or switch_violation or repeat_violation):
        return []

    return [{
        "detector_version": DETECTOR_VERSION,
        "split": split,
        "session_id": session["session_id"],
        "turn_number": turn_number,
        "family": "rejection_switchaway_violation",
        "action": "mask_or_downweight_gold",
        "confidence": "medium_high",
        "requested_text": user_text,
        "additional_track_ids": [],
        "mask_gold": True,
        "group_weight": 0.1,
        "gold_track_id": gold_id,
        "gold_metadata": metadata_summary(gold_row),
        "evidence": {
            "user_text": user_text,
            "previous_track_id": prev_id,
            "previous_metadata": metadata_summary(prev_row),
            "explicit_artist_norm": explicit_artist,
            "same_track": same_track,
            "same_artist": same_artist,
            "switch_violation": switch_violation,
            "repeat_violation": repeat_violation,
        },
    }]


def detect_session(
    split: str,
    session: dict[str, Any],
    catalog: CatalogIndex,
    *,
    include_year_constraints: bool = False,
    strict_switchaway: bool = False,
    strict_hard_artist: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for turn_number, text in user_turns(session):
        gold_id = gold_for_turn(session["conversations"], turn_number)
        if not gold_id:
            continue
        records.extend(exact_and_version_records(
            split=split, session=session, turn_number=turn_number,
            user_text=text, gold_id=gold_id, catalog=catalog,
        ))
        records.extend(hard_constraint_records(
            split=split, session=session, turn_number=turn_number,
            user_text=text, gold_id=gold_id, catalog=catalog,
            include_year_constraints=include_year_constraints,
            strict_hard_artist=strict_hard_artist,
        ))
        records.extend(rejection_records(
            split=split, session=session, turn_number=turn_number,
            user_text=text, gold_id=gold_id, catalog=catalog,
            strict_switchaway=strict_switchaway,
        ))
    return records


def load_sessions(split: str):
    from datasets import load_dataset

    if split == "devset":
        return load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")

    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    if split in {"train", "train_full"}:
        return train_ds
    if split not in {"train_lt", "val"}:
        raise ValueError(f"unsupported split {split!r}")
    split_data = json.loads(SPLIT_FILE.read_text())
    allowed_key = "train_a" if split == "train_lt" else "val"
    allowed = set(split_data[allowed_key])
    return train_ds.filter(lambda row: row["session_id"] in allowed)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    return n


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, int] = defaultdict(int)
    by_action: dict[str, int] = defaultdict(int)
    keys = set()
    for row in rows:
        by_family[row["family"]] += 1
        by_action[row["action"]] += 1
        keys.add((row["session_id"], row["turn_number"]))
    return {
        "detector_version": DETECTOR_VERSION,
        "n_records": len(rows),
        "n_unique_turns": len(keys),
        "by_family": dict(sorted(by_family.items())),
        "by_action": dict(sorted(by_action.items())),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["devset", "train_lt", "val", "train", "train_full"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path)
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--include-year-constraints", action="store_true",
                    help="Include year/decade masks. Off by default because release_date often reflects reissues.")
    ap.add_argument("--strict-switchaway", action="store_true",
                    help="Keep only high-precision switch-away wording for rejection masks.")
    ap.add_argument("--strict-hard-artist", action="store_true",
                    help="Keep only high-precision named-artist hard constraints.")
    args = ap.parse_args(argv)

    from datasets import load_dataset

    catalog_rows = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    catalog = build_catalog_index(catalog_rows)
    sessions = load_sessions(args.split)
    if args.max_sessions is not None:
        sessions = sessions.select(range(min(args.max_sessions, len(sessions))))

    rows: list[dict[str, Any]] = []
    for session in sessions:
        rows.extend(detect_session(
            args.split,
            session,
            catalog,
            include_year_constraints=args.include_year_constraints,
            strict_switchaway=args.strict_switchaway,
            strict_hard_artist=args.strict_hard_artist,
        ))

    n = write_jsonl(args.out, rows)
    summary = summarize(rows)
    summary["out"] = str(args.out)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    assert n == len(rows)


if __name__ == "__main__":
    main()
