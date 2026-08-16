# Request Family-Search Ledger v0.9

Date: 2026-06-23

Make family search and rejection decisions auditable so exact/version does not look like a cherry-picked survivor.

Selection-pressure verdict: `development_evidence_strong_controls_reduce_selection_risk_transfer_pending`.

The ledger records every request-family class currently used in the paper package, including negative and mined-but-not-admitted cases. It is an audit-order ledger reconstructed from dated evidence artifacts, not a claim of perfect wall-clock experiment chronology.

## Family Ledger

| order | family | selection stage | decision | first unresolved gate | pre/post readout status |
|---:|---|---|---|---|---|
| 1 | exact/version | main development family | `passes_development_gates_transfer_pending` | `frozen_transfer` | developed on dev/val; full-train and Blind B readouts frozen separately |
| 2 | version/duplicate equivalent | bundled low-prevalence subfamily | `proven_but_rare` | `coverage` | bundled with exact/version before final framing; not promoted as separate family |
| 3 | hard artist broad | stress test before simple selector | `not_admitted_action_precision_weak` | `action_precision` | broad failure analysis precedes the stricter simple/non-negated selector |
| 4 | hard artist simple/non-negated | supporting training evidence | `supporting_training_evidence_not_admitted` | `post_hoc_selector_and_transfer` | post-broad-failure selector; frozen-transfer candidate only |
| 5 | rejection/switch-away | negative diagnostic | `diagnostic_negative_no_main_fix` | `useful_action_and_official_preservation` | tested as conservative masking/suppression; never promoted to headline |
| 6 | hard album constraint | mined but not admitted | `mined_not_admitted` | `action_safety_and_dev_coverage` | additional-family mining after initial exact/hard/switch detectors |
| 7 | hard year/decade constraint | mined but not admitted | `mined_not_admitted` | `resolver_semantics` | additional-family mining; not used for training claims |
| 8 | genre/mood/instrumentation/language/popularity | mined but not admitted | `not_admitted_no_reusable_target_set` | `no_trusted_catalog_target_set` | additional-family mining; not converted to labels |
| 9 | synthetic thought traces | audit-only framing evidence | `audit_only_not_labels` | `not_visible_user_input` | offline qualitative audit; forbidden as detector or inference input |

## Gate Details

### 1. exact/version

- Attempt: visible exact-title and version/duplicate requests
- Why this reduces cherry-pick risk: matched official-label, no-feature, wrong-positive, and cross-dialogue controls are recorded separately
- Key thresholds/readouts:
  - 24/25 (0.960 [0.805, 0.993]); frame ablation 37/40 strict positives vs 3/80 controls; title overlap 37/37
  - 106/107 (0.991 [0.949, 0.998]); LLM adjudication chose request 94/94 (1.000 [0.961, 1.000]); duplicate baseline same canonical title 0/94
  - dev +0.00068 [+0.00017, +0.00130]; val -0.00013 [-0.00052, +0.00026]
  - dev +0.00344 [+0.00213, +0.00494]; val +0.00465 [+0.00350, +0.00594]
- Sources:
  - `docs/evidence/request_family_admission_audit_v09.json`
  - `docs/evidence/request_training_claim_ledger_v09.json`
  - `docs/evidence/request_selection_pressure_audit_v09.json`

### 2. version/duplicate equivalent

- Attempt: same resolver as exact/version, but allowing catalog versions/duplicates
- Why this reduces cherry-pick risk: reported as rare rather than inflated into a separate contribution
- Key thresholds/readouts:
  - devset has 1 version/duplicate record; train has 5 version records
  - same label contract as exact/version
- Sources:
  - `docs/evidence/request_goal_completion_audit_v09.json`
  - `docs/evidence/request_exact_robustness_table_v09.json`

### 3. hard artist broad

- Attempt: imperative named-artist constraints with broad parser
- Why this reduces cherry-pick risk: broad failure is retained in the selector ladder and limits the hard-family claim
- Key thresholds/readouts:
  - broad val postranking strict precision 37/50
  - learned broad hard specialist is useful stress evidence but action precision is weaker than exact/version
- Sources:
  - `docs/evidence/request_hard_artist_selector_ladder_v09.json`
  - `docs/evidence/request_family_admission_audit_v09.json`

### 4. hard artist simple/non-negated

- Attempt: abstaining hard-artist selector excluding exact-title, quoted-title, album/year/era, lyric/example, and local-negation cases
- Why this reduces cherry-pick risk: reported as supporting, not headline, because selector precision and transfer are weaker than exact/version
- Key thresholds/readouts:
  - selector ladder broad 37/50 -> simple/no-exact 12/13 strict precision, but simple changes only 13 val rows and independent labels cover postrank changed rows only
  - dev +0.00089 [+0.00031, +0.00160]; val +0.00047 [+0.00014, +0.00085]
  - top-1 satisfaction dev +9 (+18.4%); val +18 (+25.4%)
- Sources:
  - `docs/evidence/request_hard_artist_selector_ladder_v09.md`
  - `docs/evidence/request_hard_simple_nonneg_training_v09.md`
  - `docs/evidence/request_family_admission_audit_v09.json`

### 5. rejection/switch-away

- Attempt: strict requests to move away from a rejected item/source or avoid a repeated previous item
- Why this reduces cherry-pick risk: negative result is retained in the paper to show the contract rejects visible but unsafe actions
- Key thresholds/readouts:
  - violation@1 0.964 -> 0.214, but violation@20 1.000 -> 1.000; gold moved down 21/28; oracle sidecar top-100 clean-top20 feasible 0/28
  - official -0.00090 [-0.00181, -0.00021]
  - corrected -0.00099 [-0.00191, -0.00029]; switch-family -0.21058 [-0.31558, -0.11139]
- Sources:
  - `docs/evidence/request_switchaway_strict_v09.md`
  - `docs/evidence/request_switchaway_oracle_bound_v09.md`
  - `docs/evidence/request_family_admission_audit_v09.json`

### 6. hard album constraint

- Attempt: album-name constraints mined from the same sidecar workflow
- Why this reduces cherry-pick risk: literal salvage subset is disclosed but not promoted because dev coverage is 0
- Key thresholds/readouts:
  - train_lt: 77, val: 7, devset: 0
  - The current parser finds real album conflicts, but fuzzy album-name resolution also produces visible false actions. On val, only 3/7 album constraints resolve to an album string literally present in the request, and 4/7 have low token overlap with the visible request. On train, 7/77 album sidecar actions would mask a gold item whose metadata already satisfies the parsed album.
- Sources:
  - `docs/evidence/request_additional_family_mining_audit_v09.json`

### 7. hard year/decade constraint

- Attempt: release-year and decade constraints enabled in mining sidecar
- Why this reduces cherry-pick risk: broad coverage is reported, but unsafe range/date semantics block admission
- Key thresholds/readouts:
  - train_lt: 5110, val: 783, devset: 189
  - The enabled detector has broad coverage but weak action semantics: release_date can mean reissue or catalog date, open-ended era requests are treated as hard single intervals, and the current parser truncates ranges such as 2014-2015 to the first year.
- Sources:
  - `docs/evidence/request_additional_family_mining_audit_v09.json`

### 8. genre/mood/instrumentation/language/popularity

- Attempt: high-recall broad semantic request-language mining
- Why this reduces cherry-pick risk: frequent families are retained as rejected evidence because catalog fields cannot certify satisfaction
- Key thresholds/readouts:
  - train_lt: 87136, val: 13374, devset: 6678
  - screening request-like 23-25/25 per category; trusted target sets 0/125
  - These requests are frequent and important, but the current catalog does not expose high-precision fields that turn them into reusable request-satisfying target sets without model or human adjudication.
- Sources:
  - `docs/evidence/request_additional_family_mining_audit_v09.json`
  - `docs/evidence/request_broad_semantic_screening_audit_v09.json`

### 9. synthetic thought traces

- Attempt: inspect generator traces for policy/request boundary language
- Why this reduces cherry-pick risk: traces support vocabulary only; deployable detectors remain visible-dialogue/catalog based
- Key thresholds/readouts:
  - trace mentions policy title 251/258
  - trace request-corroborated 32/258
- Sources:
  - `docs/evidence/request_trace_corroboration_exact_version_v09.summary.json`
  - `docs/evidence/request_trace_boundary_examples_v09.json`
  - `docs/evidence/request_nonoracle_audit_v09.json`

## Sources

- `docs/evidence/request_family_admission_audit_v09.json`
- `docs/evidence/request_additional_family_mining_audit_v09.json`
- `docs/evidence/request_broad_semantic_screening_audit_v09.json`
- `docs/evidence/request_trace_corroboration_exact_version_v09.summary.json`
- `docs/evidence/request_selection_pressure_audit_v09.json`
