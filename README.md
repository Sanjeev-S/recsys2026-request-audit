# recsys2026-request-audit

Artifact for **"When the Label Ignores the Request: Auditing Policy-Selected Targets in Synthetic Conversational Music Recommendation"** (RecSys Challenge 2026 paper). Team: sanjeevsuresh.

This repository contains the final, cleaned scripts and ID-keyed evidence behind the paper's claims: the exact/version request detector and resolver, the training-target supplement, the rank and score readouts with bootstrap CIs, and the detector precision audit.

## What this is (and is not)

- **Is:** the audit detector/resolver, the target-remap sidecars (ID-keyed), the training and evaluation scripts, and the score/protocol files backing each number in the paper — everything needed to re-derive the audit against your own copy of the challenge data.
- **Is not:** a copy of the benchmark. TalkPlayData 2 / RecSys Challenge 2026 data is licensed CC-BY-NC-ND-4.0 and is **not redistributed here**. Released evidence is ID-keyed: it references session/turn/track identifiers so every verdict can be re-checked against your own copy of the data, but contains no dialogue text and no track/artist/album names. `src/redact_sidecar.py --check` is the enforcement tool.
- **Scope note:** the specialist and matched control are the paper's study models (Sections 6-7). They were not part of the team's leaderboard submission (paper Section 9).

## Layout

```
src/              all pipeline modules, flat (they import each other as siblings)
run_funnel.py     entrypoint: detector -> filter -> coverage -> changed-rows (Section 5 funnel)
run_bootstrap.sh  entrypoint: Table 2 bootstrap, frozen invocation (seed 20260701)
train.sh          entrypoint: specialist + matched-control training (Section 6)
evidence/         ID-keyed annotations, score summaries, protocols — grouped per claim
data/             train/val session-ID split (IDs only, seed 42)
```

## Data access

Scripts load the challenge data from HuggingFace (`talkpl-ai/TalkPlayData-Challenge-Dataset`, `talkpl-ai/TalkPlayData-Challenge-Track-Metadata`); obtain access and accept the license there. `data/train_val_split_seed42.json` fixes the train/val session split. Set `MCRS_EXPLORE_ROOT` to your data root for the training/eval scripts that read feature matrices (see below).

## Reproducing the paper's numbers

1. `pip install -r requirements.txt` (Python 3.12).
2. **Audit funnel (Section 5)** — `python run_funnel.py --split devset --prediction-path <production ranking>`: reproduces 82 visible directives, 41/82 conflicts, and 43 effective correction records. The detector chain is deterministic: the shipped devset sidecar re-derives byte-identically.
3. **Detector precision audit (Section 5)** — `src/score_detector_audit.py` scores the stratified 120-turn annotation packet; the shipped aggregate is `evidence/detector_validity/detector_audit_stratified120_score.json` (see provenance notes on the lost raw labels).
4. **Training (Section 6)** — `./train.sh` trains specialist + matched control with the frozen protocol (49 features + exact-request feature, LambdaRank, group weight 0.1, 50 rounds). **Scope:** this requires the production 49-feature matrices (~2 GB val / ~18 GB train parquet), which are built by the system pipeline outside this artifact's scope and are not redistributed. Given those matrices, training reproduces the shipped models; without them, the shipped model dumps (`evidence/models/`) plus gated evaluation reproduce every Table 1 and Table 2 number.
5. **Table 1, rank readout (Section 7)** — `python src/build_request_rank_table.py` aggregates the shipped per-turn ranks and asserts every printed cell (six buckets, rank-1 66 vs 46, turn-by-turn +35/-0, the single false-fire absence). Pass `--control-predictions/--specialist-predictions` (gated files regenerated via `src/run_request_fulltrain_eval.py` from the shipped models) to recompute the ranks from scratch and diff them against the shipped file.
6. **Table 2, score readout (Section 7)** — `src/evaluate_request_corrections.py` over the gated prediction files yields all six cells; `./run_bootstrap.sh` reproduces the CIs verbatim (10,000 paired session-level resamples, seed 20260701), including the exact-track slice CI (+0.277, [+0.163, +0.400], 29 sessions).
7. Cross-check each number against the claim map below.

## Claim map (paper number -> script -> evidence)

| Paper claim | Producing script (src/) | Evidence file (evidence/) |
|---|---|---|
| Detector audit: 37/40 positives confirmed; 3 misses in 80 sampled turns (Section 5) | `score_detector_audit.py` | `detector_validity/detector_audit_stratified120_score.json` |
| Dev funnel: 8,000 turns, 82 directives (1.0%), 41 conflicts (Section 5) | `request_correction_labels.py` -> `filter_request_corrections.py` -> `analyze_request_postrank_coverage.py` (via `run_funnel.py`) | `funnel/coverage_devset.json`, `funnel/scope_prevalence_card.json` |
| Conflicts deepen with turn depth: 1/6 of turn-1 directives to over 2/3 by turns 5-8 (Section 5) | `build_request_exact_robustness_table.py` | `funnel/turn_depth_robustness_table.json` |
| 43 effective cases: 42 exact + 1 version/duplicate (Section 6) | `filter_request_corrections.py` | `corrections/corrections_devset_exact_version.idkeyed.jsonl` + `.summary.json` |
| 1,873/121,592 turns supplemented; 1,878 added rows; 13 zero-add + 17 multi-add turns (Section 6) | `train_request_supplement.py` (label cache) | `fulltrain/label_cache_summaries.json`, `fulltrain/added_rows_reconciliation.json` |
| 49 features + exact-request feature; matched control; group weight 0.1 (Section 6) | `train_request_supplement.py` via `train.sh` | `models/specialist.json`, `models/matched_control.json` (feature_names embedded), `fulltrain/fulltrain_protocol.md` |
| Protocol freeze; preservation margin > -0.001 (Sections 6-7) | `verify_request_fulltrain_freeze.py`, `verify_request_freeze.py` (frozen records) | `fulltrain/freeze_manifest.json`, `fulltrain/freeze_protocol.md` |
| Inference-time dialogue gate (identical non-gated scores) (Section 6) | `request_corrected_rerank_blend_eval.py` | `results/table2_result_card.md` |
| Table 1: rank buckets 33/8/0 & 13/10/18 -> 41/0/0 & 25/15/1; rank-1 66 vs 46; +35/-0; one absence = detector false fire (Section 7) | `build_request_rank_table.py` | `results/table1_per_turn_ranks.idkeyed.jsonl`, `results/table1_rank_readout.json` |
| Table 2: 0.1908->0.1914; 0.1922->0.1943; 0.523->0.802 (Section 7) | `evaluate_request_corrections.py`, `build_request_fulltrain_result_card.py` | `results/table2_result_card.json` |
| Table 2 CIs: +0.0006 [+0.0002,+0.0012]; +1.09% [+0.0011,+0.0032]; slice +0.277 [+0.163,+0.400], 29 sessions, positive in all resamples (Section 7) | `bootstrap_request_corrected_dev.py` via `run_bootstrap.sh` | `results/bootstrap_control_vs_specialist_dev.json` (+ `results/bootstrap_retrain_replication.json`) |
| Hard artist: 74% action precision (37/50) on strict review (Section 8) | `build_request_hard_artist_selector_ladder.py` | `ledger/hard_artist_selector_ladder.json` |
| Switch-away: official loss -0.0009 [-0.0018,-0.0002]; 0/28 turns with admissible alternatives (Section 8) | `apply_request_switchaway_suppression.py`, `analyze_request_switchaway_oracle_bound.py` | `ledger/switchaway_suppression_bootstrap.json`, `ledger/switchaway_oracle_bound.json` |
| Broad semantic: 0/125 certifiable (Section 8) | `build_request_broad_semantic_screening_audit.py` | `ledger/family_search_ledger.json` (aggregate; raw screening file withheld — dialogue text) |
| Five screened-and-declined families (Section 8) | `build_request_family_search_ledger.py` | `ledger/family_search_ledger.json` |

The released-artifact promises of Section 9 map to: (i) ID-keyed audit annotations = `evidence/corrections/*.idkeyed.jsonl`; (ii) detector + precision audit = `src/request_correction_labels.py` + `src/score_detector_audit.py` + `evidence/detector_validity/`; (iii) target-remap script = `src/apply_request_exact_postrank.py` (rebuilds the request-satisfying slice via `src/evaluate_request_corrections.py`); (iv) training scripts = `train.sh` -> `src/train_request_supplement.py`.

## Additional validation (shipped, not cited in the camera-ready paper)

These packets supported earlier drafts or internal gates; they ship for completeness and are **not** cited by the camera-ready text:

- **Strict row-level review** of the 19 changed dev rows (19/19 confirmed; two model families, prompts + protocols on disk): `evidence/strict_review/`, `evidence/funnel/changed_rows_locked_table.json`.
- **Blinded LLM adjudication** of the 12 gold-differs dev rows (12/12 request-satisfying; concordant across families; no adjudication pair shares a canonical title): `evidence/adjudication/`.
- **Availability window** (65/82 requested tracks in the deployed top-100): `evidence/funnel/coverage_devset.json` category counts.
- **Validation-split replication** of the funnel (16,000 turns, 247 directives, 212 conflicts): `evidence/funnel/coverage_val.json`. The paper reports organizer train and development splits only.
- **Reduced-scale pilot readout** (retired in favor of the frozen full-train readout): `evidence/pilot/pilot_readout_tables.json`.
- **Auxiliary same-count training controls** (wrong-positive, cross-dialogue): `build_request_shuffled_positive_control.py`, `build_request_cross_dialogue_positive_control.py` -> `evidence/pilot/pilot_readout_tables.json`.
- **Internal audit capsules** (reproducibility, selection-pressure, transfer-evidence, goal-completion): `evidence/audits/`.

## Provenance notes (read before auditing)

- **Plain-name curation (2026-08-16):** evidence and script files were renamed from internal versioned names (`*_v09`, feature-set codes) to the descriptive names above. The freeze manifest and the two `verify_request_*freeze.py` scripts are byte-frozen records from the research repo at freeze time: the paths and sha256 hashes inside them refer to the original internal files, not to the renamed copies here.
- **Table numbering** follows the camera-ready paper: Table 1 = per-directive rank readout, Table 2 = nDCG readouts. (Earlier internal names used "table1" for the nDCG card.)
- **Per-turn ranks provenance:** `results/table1_per_turn_ranks.idkeyed.jsonl` was extracted 2026-07-24 from the frozen gated evaluation and re-verified from the July 12 retrain's gated prediction files (zero diffs); `base_rank` context comes from the deployed production ranking. Recompute it any time with `build_request_rank_table.py --control-predictions ... --specialist-predictions ...`.
- **1,878 vs 1,877 added rows:** canonical is 1,878 (1,663 train + 215 val), recomputed July 2026 on the canonical feature matrices; a July 12 retrain read 1,877 because a cached copy of the val feature parquet differed (input drift, not counting logic). Full reconciliation: `evidence/fulltrain/added_rows_reconciliation.json`.
- **Retrain provenance:** the shipped model dumps and gated predictions are from the July 12, 2026 retrain, which reproduces every Table 2 cell bit-identically (nDCG@20 is rank-discrete; the gated protocol confines differences to gated turns). Verification record: `evidence/results/repro_confirmation.json`.
- **Raw 120-turn detector-audit labels** (37/40 strata) were collected on a blinded packet whose raw annotation files were lost; the aggregate score file and the scorer ship, and the packet is regenerable, but the original labels are not re-derivable. The row-level validations that ship with raw labels are the July 2026 dev-only regenerations (strict review, adjudication), with prompts and protocols recorded.
- **Val availability buckets** in the coverage file were regenerated against a rescored production ranking (the original val top-500 ranking file was lost); detector-side numbers (247/212, turn buckets) are exact. Not paper-cited.
- **Scrubbed paths:** a few evidence files had absolute internal paths rewritten to `<repo-root>/...` placeholders; the freeze manifest's sha256 entries record the original bytes.

## License

Code: MIT (see LICENSE). Evidence files: released for research verification; identifiers and scores only, no benchmark content.
