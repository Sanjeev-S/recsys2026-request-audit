# Where each number lives

Every number in the paper, the script that produces it, and the shipped
evidence file that records it. Scripts are in `src/` unless they sit at
the repo root; evidence paths are under `evidence/`.

| Paper claim | Script | Evidence |
|---|---|---|
| 82 directives, 41 conflicts, 1.0% of 8,000 turns (Section 5) | `run_audit.py`, `build_prevalence.py` | `audit/coverage_devset.json`, `audit/prevalence.json` |
| Detector audit: 37 of 40 confirmed, 3 misses in 80 (Section 5) | `score_detector_audit.py` | `detector/audit_score.json` |
| Conflicts deepen with turn depth (Section 5) | `build_turn_depth_table.py` | `audit/turn_depth.json` |
| 43 effective cases: 42 exact, 1 version (Section 6) | `select_effective_cases.py` | `annotations/devset_exact_version.idkeyed.jsonl` |
| 1,873 of 121,592 turns; 1,878 added rows; 13 zero-add, 17 multi-add (Section 6) | `train_request_supplement.py` | `training/label_cache_summaries.json`, `training/added_rows.json` |
| 49 features plus the exact-request feature; matched control; group weight 0.1 (Section 6) | `train.sh` | `models/specialist.json`, `models/matched_control.json`, `training/protocol.md` |
| Protocol freeze; preservation margin above -0.001 (Sections 6 and 7) | `verify_request_fulltrain_freeze.py` | `training/freeze_manifest.json`, `training/freeze_protocol.md` |
| Dialogue gate: identical scores outside gated turns (Section 6) | `gate_eval.py` | `results/repro_confirmation.json` (bit-identity note) |
| Table 1: rank buckets 33/8/0 and 13/10/18 to 41/0/0 and 25/15/1; rank 1 on 66 vs 46; higher on 35, lower on none; the one absence is a detector false fire (Section 7) | `reproduce_table1.py` | `results/table1_ranks.idkeyed.jsonl`, `results/table1_rank_readout.json` |
| Table 2: 0.1908 to 0.1914; 0.1922 to 0.1943; 0.523 to 0.802 (Section 7) | `evaluate_request_slice.py` | `results/table2_scores.json` |
| Table 2 intervals: +0.0006 [+0.0002, +0.0012]; [+0.0011, +0.0032]; slice +0.277 [+0.163, +0.400] over 29 sessions (Section 7) | `reproduce_table2.sh` | `results/table2_bootstrap.json` |
| Hard artist: 74% action precision, 37 of 50 (Section 8) | `hard_artist_ladder.py` | `ledger/hard_artist_ladder.json`, field `strict_intervention_precision` |
| Switch-away: official loss -0.0009 [-0.0018, -0.0002]; 0 of 28 turns admissible (Section 8) | `switchaway_suppression.py`, `switchaway_oracle_bound.py` | `ledger/switchaway_suppression_bootstrap.json`, `ledger/switchaway_oracle_bound.json` |
| Broad semantic: 0 of 125 certifiable (Section 8) | `semantic_screening.py` | `ledger/family_search_ledger.json` |
| Five screened and declined families: hard artist, rejection and switch-away, album, year, broad semantic (Section 8) | `family_ledger.py` | `ledger/family_search_ledger.json` (its extra rows are audit-only records, not families) |

The paper's Section 9 promises map to: (i) ID-keyed annotations,
`evidence/annotations/`; (ii) the detector and its precision audit,
`src/detect_requests.py` and `src/score_detector_audit.py`; (iii) the
target-remap script that rebuilds the request-satisfying slice,
`src/apply_request_override.py` with `src/evaluate_request_slice.py`;
(iv) the training scripts, `train.sh`.

## Shipped but not cited

These directories support the audit but back no number in the
camera-ready paper: `strict_review/` and `adjudication/` (row-level
reviews of the changed rows, 19/19 and 12/12), the availability counts
inside `audit/coverage_devset.json`, the validation-split replication
in `audit/coverage_val.json`, the retired reduced-scale readout in
`pilot/`, and the internal audit records in `internal/`.
