# Request-Aware Strict Postrank Evidence Table

Date: 2026-06-21

This table is generated from frozen evidence artifacts. The postranker is non-oracle: it uses visible dialogue, catalog metadata, and the existing prediction list, not policy labels or request-positive sidecars, to decide interventions.

## Precision

| readout | count | estimate | Wilson 95% CI |
|---|---:|---:|---:|
| exact-track detector sample | 24/25 | 0.960 | [0.805, 0.993] |
| strict changed-row intervention | 106/107 | 0.991 | [0.949, 0.998] |
| blinded LLM adjudication chose request item | 94/94 | 1.000 | [0.961, 1.000] |

## Heldout Table

| split | rows | exact/version records | strict request turns | top-100 coverage | changed rows | changed precision | official delta | corrected delta | affected corrected delta | affected official delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| devset | 8000 | 43 | 82 | 65/82 (79.3%) | 19 | 19/19 | 0.00042 [0.00005, 0.00089] | 0.00132 [0.00048, 0.00240] | 0.11015 [0.03880, 0.19239] | -0.00744 [-0.02093, 0.00000] |
| val | 16000 | 215 | 247 | 227/247 (91.9%) | 88 | 87/88 | -0.00011 [-0.00034, 0.00014] | 0.00290 [0.00209, 0.00382] | 0.18060 [0.13765, 0.22696] | -0.02146 [-0.03398, -0.01078] |

## Request Funnel

| split | exact-request directives | already request-first | promoted by postranker | missing from top-100 | policy label differs |
|---|---:|---:|---:|---:|---:|
| devset | 82 | 46/82 (56.1%) | 19/82 (23.2%) | 17/82 (20.7%) | 41/82 (50.0%) |
| val | 247 | 139/247 (56.3%) | 88/247 (35.6%) | 20/247 (8.1%) | 212/247 (85.8%) |

## LLM Adjudication Check

Blinded independent LLM labels compare the policy-selected label against the request-satisfying item on changed rows where those differ.

| readout | count | Wilson 95% CI |
|---|---:|---:|
| request item chosen | 94/94 | [0.961, 1.000] |
| policy item chosen | 0/94 | [0.000, 0.039] |
| request item marked request-satisfying | 94/94 | [0.961, 1.000] |

## Read

- The global official readout is a preservation/small-loss check, not the main gain.
- The corrected objective captures request satisfaction when the visible exact request differs from the policy-selected label.
- The affected-session official loss is expected: satisfying the visible request can displace the policy-selected label.
- The request funnel separates already-satisfied requests, reachable postrank fixes, and retrieval misses.
- LLM adjudication supports the corrected target on exact-request conflict rows; it is not a broad user-preference study.
