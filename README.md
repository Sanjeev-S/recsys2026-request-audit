# recsys2026-request-audit

Artifact for **"When the Label Ignores the Request: Auditing Policy-Selected Targets in Synthetic Conversational Music Recommendation"** (RecSys Challenge 2026 paper). Team: sanjeevsuresh.

This repository contains the final, cleaned scripts and ID-keyed evidence behind the paper's claims: the exact/version request detector and resolver, the training-target supplement, the evaluation-slice readout with bootstrap CIs, and the row-level validity checks.

## What this is (and is not)

- **Is:** the audit detector/resolver, the target-remap sidecars (ID-keyed), the training and evaluation scripts, and the score/protocol files backing each number in the paper — everything needed to re-derive the audit against your own copy of the challenge data.
- **Is not:** a copy of the benchmark. TalkPlayData 2 / RecSys Challenge 2026 data is licensed CC-BY-NC-ND-4.0 and is **not redistributed here**. Released evidence is ID-keyed: it references session/turn/track identifiers so every verdict can be re-checked against your own copy of the data, but contains no dialogue text and no track/artist/album names. `src/redact_sidecar.py --check` is the enforcement tool.

## Layout

```
src/              all pipeline modules, flat (they import each other as siblings)
run_funnel.py     entrypoint: detector -> filter -> coverage -> changed-rows (Section 5 funnel)
run_bootstrap.sh  entrypoint: Table 1 bootstrap, frozen invocation (seed 20260701)
train.sh          entrypoint: specialist + matched-control training (Section 6)
evidence/         ID-keyed annotations, score summaries, protocols — grouped per claim
data/             train/val session-ID split (IDs only, seed 42)
```

## Data access

Scripts load the challenge data from HuggingFace (`talkpl-ai/TalkPlayData-Challenge-Dataset`, `talkpl-ai/TalkPlayData-Challenge-Track-Metadata`); obtain access and accept the license there. `data/train_val_split_seed42.json` fixes the train/val session split. Set `MCRS_EXPLORE_ROOT` to your data root for the training/eval scripts that read feature matrices (see below).

## Reproducing the paper's numbers

1. `pip install -r requirements.txt` (Python 3.12).
2. **Audit funnel (Section 5)** — `python run_funnel.py --split devset --prediction-path <production ranking>`: reproduces 82 visible directives, 41/82 conflicts, 43 effective correction records, and the availability window (65/82 present in top-100 with the deployed ranking). The detector chain is deterministic: the shipped devset sidecar re-derives byte-identically.
3. **Row-level validity (Section 5)** — `src/run_request_row_validation.py` re-runs the strict review and the blinded LLM adjudication over the changed rows; prompts and blinding protocol are recorded in `evidence/*/. ..protocol_*.json`. July 2026 dev-only regeneration: strict review 19/19 confirmed, adjudication 12/12 request-satisfying, concordant across two model families (Claude, GPT).
4. **Training (Section 6)** — `./train.sh` trains specialist + matched control with the frozen protocol (49 features + exact-request feature, LambdaRank, group weight 0.1, 50 rounds). **Scope:** this requires the production 49-feature matrices (~2 GB val / ~18 GB train parquet), which are built by the system pipeline outside this artifact's scope and are not redistributed. Given those matrices, training reproduces the shipped models; without them, the shipped model dumps (`evidence/models/`) plus gated evaluation reproduce every Table 1 number.
5. **Table 1 (Section 7)** — `src/evaluate_request_corrections.py` over the gated prediction files yields all six cells; `./run_bootstrap.sh` reproduces the three CIs verbatim (10,000 paired session-level resamples, seed 20260701).
6. Cross-check each number against the claim map below.

## Claim map (paper number -> script -> evidence)

| Paper claim | Producing script (src/) | Evidence file (evidence/) |
|---|---|---|
| Detector audit 37/40, 3/40, 0/40 on 120-turn stratified sample | `score_request_frame_ablation.py` | `detector_validity/request_frame_ablation_val_v08_score.json` |
| Strict review of changed rows (pooled 106/107; dev cell 19/19) | `run_request_row_validation.py --mode strict-review` | `funnel/request_exact_postrank_locked_table_v09.json`; `strict_review/request_strict_review_dev_v11_*` |
| LLM adjudication (pooled 94/94, CI [0.961,1.000]; dev cell 12/12) | `build_request_preference_annotation_set.py` + `run_request_row_validation.py --mode adjudication` + `score_request_preference_labels.py` | `funnel/request_exact_postrank_locked_table_v09.json`; `adjudication/request_preference_dev_v11_*` |
| No adjudication pair shares a canonical title (0/94) | `analyze_request_conflict_metadata.py` | `adjudication/request_conflict_metadata_baseline_v09.counts.json` |
| Dev funnel: 8,000 turns, 82 directives (1.0%), 41 conflicts | `request_correction_labels.py` -> `filter_request_corrections.py` -> `analyze_request_postrank_coverage.py` (via `run_funnel.py`) | `funnel/request_exact_postrank_coverage_v09_devset.json`, `funnel/request_scope_prevalence_card_v09.json` |
| Val replication: 16,000 turns, 247 directives, 212 conflicts (85.8%) | same chain, `--split val` | `funnel/request_exact_postrank_coverage_v09_val.json` |
| Turn-depth conflicts 5/30, 24/35, 12/17 (val 75%/87%/91%) | `build_request_exact_robustness_table.py` | `funnel/request_exact_robustness_table_v09.json` |
| Availability window: 65/82 in top-100 | `analyze_request_postrank_coverage.py` | `funnel/request_exact_robustness_table_v09.json` |
| 43 effective cases (42 exact + 1 version/duplicate) | `filter_request_corrections.py` | `corrections/request_corrections_devset_exact_version_v09.idkeyed.jsonl` + `.summary.json` |
| 1,873/121,592 turns corrected; 1,878 added positive rows | `request_hard_artist_rerank_train.py` (label cache) | `fulltrain/request_exact_version_fulltrain_label_cache_summaries_v09.json`, `fulltrain/request_added_rows_reconciliation_v11.json` |
| 49 features + exact-request feature; matched control | `request_hard_artist_rerank_train.py` via `train.sh` | `models/*.json` (feature_names embedded), `fulltrain/request_exact_version_fulltrain_protocol_v09.md` |
| Auxiliary same-count controls (wrong-positive, cross-dialogue) | `build_request_shuffled_positive_control.py`, `build_request_cross_dialogue_positive_control.py` | `pilot/request_paper_tables_v09.json` ("Training Causal Control") |
| Protocol freeze + preservation criterion (delta > -0.001) | `verify_request_fulltrain_freeze.py`, `verify_request_freeze.py` | `fulltrain/request_fulltrain_freeze_manifest_v09.json`, `fulltrain/request_fulltrain_freeze_protocol_v09.md` |
| Table 1 six cells (0.1908->0.1914; 0.1922->0.1943; 0.523->0.802) | `evaluate_request_corrections.py`, `build_request_fulltrain_result_card.py` | `table1/request_fulltrain_result_card_v09.json` |
| Table 1 CIs (+0.0006 [0.0002,0.0012]; +0.0020 [0.0011,0.0032]; +0.277 [0.163,0.400]) | `bootstrap_request_corrected_dev.py` via `run_bootstrap.sh` | `table1/request_exact_version_fulltrain_bootstrap_official_vs_specialist_dev_v09.json` (+ retrain replication) |
| Rank readout 40/82 -> 82/82, McNemar p=4.5e-13 (val 121/247 -> 246/247) | `request_exact_transfer_protocol.py`, `summarize_request_training_behavior_stats.py` | `pilot/request_paper_tables_v09.json` ("Paired Request-First Behavior") |
| Inference-time dialogue gate (identical non-gated scores) | `request_corrected_rerank_blend_eval.py` | `table1/request_fulltrain_result_card_v09.md` |
| Section 8: hard-artist 37/50 ladder | `build_request_hard_artist_selector_ladder.py` | `ledger/request_hard_artist_selector_ladder_v09.json` |
| Section 8: switch-away loss -0.0009 [-0.0018,-0.0002]; 28 turns, 0 admissible | `apply_request_switchaway_suppression.py`, `analyze_request_switchaway_oracle_bound.py` | `ledger/request_switchaway_strict_top100_bootstrap_*.json`, `ledger/request_switchaway_strict_oracle_top100_bound_v09.json` |
| Section 8: broad-semantic 0/125 certifiable | `build_request_broad_semantic_screening_audit.py` | `ledger/request_family_search_ledger_v09.json` (aggregate; raw screening file withheld — dialogue text) |
| Section 8: family search ledger | `build_request_family_search_ledger.py` | `ledger/request_family_search_ledger_v09.json` |

## Provenance notes (read before auditing)

- **1,878 vs 1,877 added rows:** canonical is 1,878 (1,663 train + 215 val), recomputed July 2026 on the canonical feature matrices; a July 12 retrain read 1,877 because a cached copy of the val feature parquet differed (input drift, not counting logic). Full reconciliation: `evidence/fulltrain/request_added_rows_reconciliation_v11.json`.
- **Retrain provenance:** the shipped model dumps and gated predictions are from the July 12, 2026 retrain, which reproduces every Table 1 cell bit-identically (nDCG@20 is rank-discrete; the gated protocol confines differences to gated turns). Verification record: `evidence/table1/request_repro_confirmation_v11.json`.
- **Raw 120-turn detector-audit labels** (37/40 strata) were collected on a blinded packet whose raw annotation files were lost; the aggregate score file and the scorer ship, and the packet is regenerable, but the original labels are not re-derivable. The row-level validations that ship with raw labels are the July 2026 dev-only regenerations (strict review, adjudication), with prompts and protocols recorded.
- **Val availability buckets** in the coverage file were regenerated against a rescored production ranking (the original val top-500 ranking file was lost); detector-side numbers (247/212, turn buckets) are exact. Not paper-cited.
- **Scrubbed paths:** a few evidence files had absolute internal paths rewritten to `<repo-root>/...` placeholders; the freeze manifest's sha256 entries record the original bytes.

## License

Code: MIT (see LICENSE). Evidence files: released for research verification; identifiers and scores only, no benchmark content.
