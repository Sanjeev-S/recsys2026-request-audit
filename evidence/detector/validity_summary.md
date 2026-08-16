# Request Detector Validity Capsule v0.9

Date: 2026-06-22

Question: Does the exact/version detector identify an active request frame, and is the resulting conflict more than generic duplicate/version cleanup?

This is evidence for detector validity and simpler-baseline rejection, not a prevalence estimate or a human-user study.

## Frame Ablation

| readout | value |
|---|---:|
| stratified rows | 120 |
| minimum confidence | medium |
| strict detector-positive active exact request | 37/40 (0.925, CI [0.801, 0.974]) |
| control active exact request | 3/80 (0.037, CI [0.013, 0.105]) |
| title overlap among active strict positives | 37/37 (1.000, CI [0.906, 1.000]) |

## By Bucket

| bucket | n | active exact request | title overlap among active |
|---|---:|---:|---:|
| strict detected exact request | 40 | 37/40 (0.925, CI [0.801, 0.974]) | 37/37 (1.000, CI [0.906, 1.000]) |
| quoted but not strict-detected | 40 | 3/40 (0.075, CI [0.026, 0.199]) | 0/3 (0.000, CI [0.000, 0.561]) |
| unquoted request-like control | 40 | 0/40 (0.000, CI [0.000, 0.088]) | 0/0 (n/a) |

## Metadata Baseline

| relation between policy label and request target | count |
|---|---:|
| same canonical title / likely duplicate-version | 0/94 (0.0%) |
| same normalized title | 0/94 (0.0%) |
| same artist | 51/94 (54.3%) |
| same album | 26/94 (27.7%) |
| same release year | 31/94 (33.0%) |

## Reviewer Answer

- The detector is not merely finding quotation marks: strict positives are active exact requests far more often than quoted and unquoted controls.
- When the independent annotation marks strict positives active, it names the same requested title in 37/37 cases.
- Changed exact/version conflicts are 0/94 same-canonical-title cases, so the result is not explained by ordinary duplicate/version cleanup.

## Limitations

- The frame ablation is stratified and estimates detector validity, not population prevalence.
- The metadata baseline does not prove user preference; it rules out a simpler catalog-dedup explanation for changed conflicts.

## Sources

- `docs/evidence/request_frame_ablation_val_v08_score.json`
- `docs/evidence/request_frame_ablation_val_v08_score.md`
- `docs/evidence/request_conflict_metadata_baseline_v09.json`
- `docs/evidence/request_conflict_metadata_baseline_v09.md`
