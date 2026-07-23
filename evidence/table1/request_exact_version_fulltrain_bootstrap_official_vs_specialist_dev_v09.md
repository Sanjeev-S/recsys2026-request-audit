# Full-Train Exact/Version Paired Bootstrap

Date: 2026-07-01

Question: Do the frozen full-train point estimates keep the same direction under a session-level paired bootstrap on organizer dev?

Answer: **yes** for the whole-split official and request-satisfying readouts, and for the exact-track subset of the exact/version slice.

## Inputs

- Baseline: `docs/evidence/dev_predictions/request_corrected_F10_R54SRC_blend_exact_version_v09_fulltrain_official_control_reqfeat_dialogue_w1_devset.json`
- Candidate: `docs/evidence/dev_predictions/request_corrected_F10_R54SRC_blend_exact_version_v09_fulltrain_exact_positive_weighted_reqfeat_dialogue_w1_devset.json`
- Corrections: `docs/evidence/request_corrections_devset_exact_version_v09.jsonl`
- Bootstrap output: `docs/evidence/request_exact_version_fulltrain_bootstrap_official_vs_specialist_dev_v09.json`
- Unit: session
- Samples: 10,000
- Seed: 20260701

## Result

| readout | observed delta | 95% CI | sessions |
|---|---:|---:|---:|
| official nDCG@20 | +0.00060 | [+0.00018, +0.00115] | 1,000 |
| request-satisfying nDCG@20 | +0.00205 | [+0.00107, +0.00321] | 1,000 |
| exact-track slice request-satisfying nDCG@20 | +0.27653 | [+0.16262, +0.40017] | 29 |

The version/duplicate slice has one session in this readout, so it is reported in the main point-estimate slice but not used as an uncertainty claim.
