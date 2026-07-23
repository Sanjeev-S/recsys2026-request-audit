# Request Full-Train Result Card v0.9

Date: 2026-06-23

Question: Does the frozen full-train exact/version readout preserve official dev score while improving request-positive dev score?

Answer: **yes_on_frozen_primary_readout**.

This card is pending-safe: before full-train eval outputs exist it reports readiness only; after they exist it computes the fixed primary readout deltas without tuning on repo devset.

## Exact/Version Primary Readout

| metric | request-positive minus official-control | pass criterion |
|---|---:|---|
| official nDCG@20 | +0.00060 | > -0.001 |
| auxiliary request-positive nDCG@20 | +0.00205 | > 0 |
| exact-slice auxiliary request-positive nDCG@20 | +0.27863 | > 0 |

## Paired Bootstrap Support

Unit: `session`. Samples: `10000`. Seed: `20260701`.

| readout | observed delta | 95% CI | sessions |
|---|---:|---:|---:|
| official nDCG@20 | +0.00060 | [+0.00018, +0.00115] | 1000 |
| request-satisfying nDCG@20 | +0.00205 | [+0.00107, +0.00321] | 1000 |
| exact-track slice request-satisfying nDCG@20 | +0.27653 | [+0.16262, +0.40017] | 29 |

The version/duplicate component has one session, so slice uncertainty is reported for exact-track requests only.

## Hard Violation-Drop Secondary Readout

Status: `pending_outputs`. Hard remains supporting and cannot replace the exact/version headline.

## Sources

- `docs/evidence/request_fulltrain_approval_packet_2026_06_23.md`
- `docs/evidence/request_fulltrain_eval_readiness_2026_06_23.md`
- `docs/evidence/request_exact_version_fulltrain_protocol_v09.md`
- `docs/evidence/request_exact_version_fulltrain_bootstrap_official_vs_specialist_dev_v09.json`
- `docs/evidence/request_exact_version_fulltrain_bootstrap_official_vs_specialist_dev_v09.md`
- `docs/evidence/request_hard_simple_nonneg_fulltrain_violation_drop_protocol_v09.md`
