"""LLM row-level validation runner for the request-audit paper claims.

Two modes, both label rows the exact/version intervention changed:

- ``adjudication``: blinded pairwise adjudication over the annotation set from
  build_request_preference_annotation_set.py. The model sees the visible user
  message and two candidates A/B (side randomized by the set builder, key
  withheld) and judges which satisfies the request. Output labels score with
  score_request_preference_labels.py against the withheld key.

- ``strict-review``: unblinded strict review over the changed-rows packet from
  build_request_changed_rows_packet.py. The model sees the message, the
  detector-extracted titles, and the promoted tracks, and answers
  valid_request / promoted_satisfies / action_ok. A row counts as confirmed
  only if all three are yes.

The exact prompt templates live in this file and are hashed into every output,
so the adjudication protocol is on disk and reproducible (the original v09 run
recorded only aggregates). Backends: the ``claude`` CLI (JSON output mode) or
the ``codex`` CLI (exec mode). Both judge from the visible dialogue text and
catalog metadata only; no policy labels are shown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

ADJUDICATION_PROMPT = """You are an independent annotator for a music-recommendation dialogue study.

Below is the user's message from one dialogue turn, and two candidate tracks, A and B.
Judge only from the visible user message and the candidate metadata shown. Do not use
position to guess anything about where the candidates came from.

User message:
{user_text}

Candidate A: "{a_track}" by {a_artist} (album: {a_album}, released {a_release})
Candidate B: "{b_track}" by {b_artist} (album: {b_album}, released {b_release})

Answer these questions:
1. preferred_item: Which candidate better matches what the user asked for in this message? (A/B/tie/unclear)
2. request_satisfying_item: Which candidate satisfies the user's explicit request, if any? (A/B/both/neither/unclear)
3. explicit_request_visible: Does the message contain an explicit request for a specific song? (yes/no/unclear)
4. confidence: high/medium/low
5. notes: one short sentence of rationale

Respond with a single JSON object with keys exactly:
preferred_item, request_satisfying_item, explicit_request_visible, confidence, notes.
Output only the JSON object, nothing else."""

STRICT_REVIEW_PROMPT = """You are strictly reviewing one changed row of a request-aware ranking intervention
in a music-recommendation dialogue benchmark.

A detector flagged a visible exact-song request in the user's message, resolved it
against the track catalog, and the intervention promoted the resolved track(s).
Review whether this specific action was correct, strictly from the message text and
the metadata shown.

User message:
{user_text}

Detector-extracted requested title(s): {requested_titles}
Promoted track(s):
{promoted_block}

Answer these questions:
1. valid_request: Does the message contain a genuine, active request for a specific song
   (not a passing mention, a reference to something already played, or a broad preference)? (yes/no)
2. promoted_satisfies: Do the promoted track(s) satisfy that request — the same titled work,
   by the referenced artist where the message identifies one? (yes/no)
3. action_ok: Overall, was promoting these track(s) the right request-satisfying action for
   this turn? (yes/no)
4. confidence: high/medium/low
5. notes: one short sentence of rationale

Respond with a single JSON object with keys exactly:
valid_request, promoted_satisfies, action_ok, confidence, notes.
Output only the JSON object, nothing else."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def call_claude(prompt: str, model: str, timeout: int = 240, retries: int = 4) -> str:
    cmd = ["claude", "-p", "--output-format", "json", "--model", model]
    last = ""
    for attempt in range(retries):
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = f"timeout after {timeout}s"
            time.sleep(2 * (attempt + 1))
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                return json.loads(proc.stdout)["result"]
            except (json.JSONDecodeError, KeyError) as e:
                last = f"parse error: {e}; stdout head: {proc.stdout[:200]}"
        else:
            last = f"exit {proc.returncode}; stderr head: {proc.stderr[:200]}"
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"claude call failed after {retries} attempts: {last}")


def call_codex(prompt: str, model: str | None, timeout: int = 360, retries: int = 4) -> str:
    last = ""
    for attempt in range(retries):
        with tempfile.TemporaryDirectory() as scratch:
            out_path = os.path.join(scratch, "last.txt")
            cmd = ["codex", "exec", "-s", "read-only", "--skip-git-repo-check", "-C", scratch, "-o", out_path]
            if model:
                cmd += ["-m", model]
            try:
                proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                last = f"timeout after {timeout}s"
                time.sleep(2 * (attempt + 1))
                continue
            if proc.returncode == 0 and os.path.exists(out_path):
                text = Path(out_path).read_text(encoding="utf-8").strip()
                if text:
                    return text
                last = "empty output file"
            else:
                last = f"exit {proc.returncode}; stderr head: {proc.stderr[:200]}"
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"codex call failed after {retries} attempts: {last}")


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    return json.loads(text[start : end + 1])


def fmt(value: Any) -> str:
    return str(value) if value not in (None, "") else "unknown"


def adjudication_prompt(row: dict[str, Any]) -> str:
    a, b = row["candidate_A"], row["candidate_B"]
    return ADJUDICATION_PROMPT.format(
        user_text=row.get("user_text") or "",
        a_track=fmt(a.get("track_name")), a_artist=fmt(a.get("artist_name")),
        a_album=fmt(a.get("album_name")), a_release=fmt(a.get("release_date")),
        b_track=fmt(b.get("track_name")), b_artist=fmt(b.get("artist_name")),
        b_album=fmt(b.get("album_name")), b_release=fmt(b.get("release_date")),
    )


def strict_review_prompt(row: dict[str, Any]) -> str:
    promoted = row.get("promoted_tracks") or []
    lines = [
        f'- "{fmt(t.get("track_name"))}" by {fmt(t.get("artist_name"))}'
        f' (album: {fmt(t.get("album_name"))}, released {fmt(t.get("release_date"))})'
        for t in promoted
    ]
    return STRICT_REVIEW_PROMPT.format(
        user_text=row.get("user_text") or "",
        requested_titles=", ".join(row.get("requested_titles") or []),
        promoted_block="\n".join(lines) or "- (none)",
    )


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, float]:
    if n <= 0:
        return {"point": 0.0, "low": 0.0, "high": 0.0}
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return {"point": p, "low": max(0.0, center - half), "high": min(1.0, center + half)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["adjudication", "strict-review"], required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--backend", choices=["claude", "codex"], default="claude")
    ap.add_argument("--model", default=None, help="model id/alias passed to the backend CLI")
    ap.add_argument("--labels-output", type=Path, required=True)
    ap.add_argument("--protocol-output", type=Path, required=True)
    ap.add_argument("--summary-output", type=Path)
    args = ap.parse_args(argv)

    rows = load_jsonl(args.input)
    template = ADJUDICATION_PROMPT if args.mode == "adjudication" else STRICT_REVIEW_PROMPT
    make_prompt = adjudication_prompt if args.mode == "adjudication" else strict_review_prompt
    call = call_claude if args.backend == "claude" else call_codex
    model = args.model or ("opus" if args.backend == "claude" else None)

    labels: list[dict[str, Any]] = []
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    with args.labels_output.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            key = row.get("sample_id") or f'{row["session_id"]}:{row["turn_number"]}'
            response = call(make_prompt(row), model)
            label = extract_json(response)
            label["sample_id"] = key
            if "session_id" in row:
                label.setdefault("session_id", row["session_id"])
                label.setdefault("turn_number", row.get("turn_number"))
            labels.append(label)
            f.write(json.dumps(label, sort_keys=True) + "\n")
            f.flush()
            print(f"[{i}/{len(rows)}] {key}: {json.dumps({k: v for k, v in label.items() if k != 'notes'})}")

    protocol = {
        "date": date.today().isoformat(),
        "mode": args.mode,
        "backend": args.backend,
        "model": model,
        "input": str(args.input),
        "n_rows": len(rows),
        "prompt_template": template,
        "prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "blinding": (
            "candidates A/B randomized by build_request_preference_annotation_set.py (key withheld from the model)"
            if args.mode == "adjudication"
            else "unblinded action review; policy label not shown"
        ),
        "labels_output": str(args.labels_output),
    }
    args.protocol_output.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    if args.mode == "strict-review" and args.summary_output:
        n = len(labels)
        def count(field: str) -> int:
            return sum(1 for l in labels if str(l.get(field, "")).strip().lower() == "yes")
        successes = sum(
            1 for l in labels
            if all(str(l.get(f, "")).strip().lower() == "yes" for f in ("valid_request", "promoted_satisfies", "action_ok"))
        )
        summary = {
            "n": n,
            "valid_request_yes": count("valid_request"),
            "promoted_satisfies_yes": count("promoted_satisfies"),
            "action_ok_yes": count("action_ok"),
            "successes": successes,
            **wilson(successes, n),
            "labels": str(args.labels_output),
            "protocol": str(args.protocol_output),
        }
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
