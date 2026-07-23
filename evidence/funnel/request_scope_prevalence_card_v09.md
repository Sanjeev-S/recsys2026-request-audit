# Request Scope And Prevalence Card v0.9

Exact/version is deliberately narrow. The global auxiliary nDCG movement should be read together with per-1k prevalence, reachable coverage, and a unit-bounded prevalence ceiling; official preservation is whole-split, while exact-slice official deltas can be negative when request-satisfying items displace policy-selected labels.

The unit ceilings are simple prevalence ceilings for a unit-bounded per-turn score if only that slice could change; they are explanatory denominator math, not a replacement for paired bootstrap metrics.

## Exact/Version Scope

| split | turns | exact directives | per 1k turns | policy/request conflicts | reachable changes | request-first gain | visible-slice unit ceiling | gain-row unit ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| devset | 8000 | 82 | 10.25 | 41/82 (50.0%) | 19/82 (23.2%) | 40 -> 82 (+42) | +0.01025 | +0.00525 |
| val | 16000 | 247 | 15.44 | 212/247 (85.8%) | 88/247 (35.6%) | 121 -> 246 (+125) | +0.01544 | +0.00781 |

## Metric Readout Context

| split | specialist auxiliary request-positive delta | specialist exact-slice official delta | full-pool auxiliary request-positive delta | full-pool exact-slice official delta |
|---|---:|---:|---:|---:|
| devset | +0.00344 [+0.00213, +0.00494] | +0.00133 [-0.04448, +0.05919] | +0.00344 [+0.00213, +0.00494] | +0.00703 [-0.02280, +0.05535] |
| val | +0.00465 [+0.00350, +0.00594] | -0.03961 [-0.06140, -0.01877] | +0.00465 [+0.00350, +0.00594] | -0.03094 [-0.04554, -0.01800] |

Interpretation: the exact/version result is a conditional behavior result with a narrow denominator. That narrowness explains why whole-split auxiliary request-positive deltas are small even when request-first behavior changes sharply on exact/version turns. Exact-slice official deltas are reported because satisfying the visible request can trade off against the policy-selected single label.

## Sources

- `docs/evidence/request_exact_postrank_locked_table_v09.json`
- `docs/evidence/request_exact_robustness_table_v09.json`
- `docs/evidence/request_paper_tables_v09.json`
