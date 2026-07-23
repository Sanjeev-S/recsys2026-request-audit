# Full-Train Exact/Version Request-Positive Protocol

Date: 2026-06-23

## Question

After freezing the exact/version detector, label action, request feature, and
deployment gate, can we fold repo `val` back into organizer train and use repo
`devset` as the held-out readout?

This is the clean paper protocol: train on organizer train (`train_a + val`),
then evaluate on the held-out organizer dev split/repo `devset`.

## Training Sources

| source | sessions/turns | candidate rows | role |
|---|---:|---:|---|
| `train_lt_105k` (`train_a`) | 105,592 turns | 177,937,551 | original train source |
| `val` | 16,000 turns | 25,283,719 | folded back after protocol freeze |
| combined full train | 121,592 turns | 203,221,270 | final model fitting |

## Correction Sidecars

| split | correction records | unique corrected turns | families |
|---|---:|---:|---|
| `train_a` | 1,685 | 1,661 | 1,680 exact-track request, 5 version/duplicate |
| `val` | 215 | 212 | 215 exact-track request |
| full train | 1,900 | 1,873 | 1,895 exact-track request, 5 version/duplicate |
| `devset` readout | 43 | 43 | 42 exact-track request, 1 version/duplicate |

## Training Cells

Both cells use the same visible `exact_request_match` feature:

| cell | label action |
|---|---|
| `official` | original policy-selected labels |
| `exact_positive_weighted` | original labels plus exact/version request-satisfying positives; corrected groups weighted `0.1` |

This isolates request-positive supervision from the presence of a visible
request-match feature.

## Local Cache Verification

| check | `train_a` | `val` | full train |
|---|---:|---:|---:|
| groups | 105,592 | 16,000 | 121,592 |
| rows | 177,937,551 | 25,283,719 | 203,221,270 |
| base positive rows | 78,497 | 12,075 | 90,572 |
| added request-positive rows | 1,663 | 215 | 1,878 |
| positive rows after correction | 80,160 | 12,290 | 92,450 |
| downweighted groups | 1,661 | 212 | 1,873 |
| dropped groups | 0 | 0 | 0 |
| exact-request feature directive groups | 1,952 | 247 | 2,199 |
| exact-request feature matched groups | 1,940 | 246 | 2,186 |
| exact-request feature matched rows | 1,957 | 250 | 2,207 |

Compact summaries:

- `docs/evidence/request_exact_version_fulltrain_label_cache_local_v09.json`
- `docs/evidence/request_exact_version_fulltrain_train_a_official_label_cache_v09.summary.json`
- `docs/evidence/request_exact_version_fulltrain_train_a_exact_positive_weighted_label_cache_v09.summary.json`
- `docs/evidence/request_exact_version_fulltrain_val_official_label_cache_v09.summary.json`
- `docs/evidence/request_exact_version_fulltrain_val_exact_positive_weighted_label_cache_v09.summary.json`
- `docs/evidence/request_exact_version_fulltrain_train_a_exact_request_feature_cache_v09.summary.json`
- `docs/evidence/request_exact_version_fulltrain_val_exact_request_feature_cache_v09.summary.json`

## Training Launch

Requires explicit approval because it uses paid RunPod compute and sends staged
repo files plus HF credentials to the pod.

```bash
.venv/bin/python scripts/request_exact_fulltrain_launch.py \
  --max-hours 5 \
  --num-boost-round 50 \
  --corrected-group-weight 0.1
```

Expected models:

- `docs/evidence/models/request_corrected_F10_R54SRC_official_exact_version_v09_fulltrain_trainval_reqfeat_gw0p1_medium50.json`
- `docs/evidence/models/request_corrected_F10_R54SRC_exact_positive_weighted_exact_version_v09_fulltrain_trainval_reqfeat_gw0p1_medium50.json`

## Devset Evaluation

Use devset as the held-out readout. Compare a matched official-label
exact-feature control against the request-positive specialist, each blended
under the same visible dialogue gate. The production reranker remains the
anchor, so non-gated turns keep production scores.

Primary readout is fixed at `lambda=1.0`, using the prior 105k development
result. Other blend weights, if run, are sensitivity analysis and must not be
used to select the reported full-train result.

Official-label exact-feature control:

```bash
.venv/bin/python scripts/request_corrected_rerank_blend_eval.py \
  --feature-set F10_R54SRC \
  --split devset \
  --official-model <repo-root>/exp/models/rerank_F10_R54SRC_r54src.json \
  --request-model docs/evidence/models/request_corrected_F10_R54SRC_official_exact_version_v09_fulltrain_trainval_reqfeat_gw0p1_medium50.json \
  --corrections docs/evidence/request_corrections_devset_exact_version_v09.jsonl \
  --gate-mode dialogue \
  --request-model-add-exact-request-feature \
  --blend-weight 1 \
  --prediction-dir docs/evidence/dev_predictions \
  --output-tag exact_version_v09_fulltrain_official_control_reqfeat_dialogue \
  --output docs/evidence/request_exact_version_fulltrain_official_control_blend_dev_v09.json
```

Request-positive specialist:

```bash
.venv/bin/python scripts/request_corrected_rerank_blend_eval.py \
  --feature-set F10_R54SRC \
  --split devset \
  --official-model <repo-root>/exp/models/rerank_F10_R54SRC_r54src.json \
  --request-model docs/evidence/models/request_corrected_F10_R54SRC_exact_positive_weighted_exact_version_v09_fulltrain_trainval_reqfeat_gw0p1_medium50.json \
  --corrections docs/evidence/request_corrections_devset_exact_version_v09.jsonl \
  --gate-mode dialogue \
  --request-model-add-exact-request-feature \
  --blend-weight 1 \
  --prediction-dir docs/evidence/dev_predictions \
  --output-tag exact_version_v09_fulltrain_exact_positive_weighted_reqfeat_dialogue \
  --output docs/evidence/request_exact_version_fulltrain_exact_positive_weighted_blend_dev_v09.json
```

## Acceptance Criterion

The result strengthens the paper if the request-positive specialist preserves
whole-devset official nDCG within the predeclared `0.001` small-loss threshold
and keeps the exact/version request-satisfaction gain over the official-label
control.

If the full-train effect is weaker than the 105k result, that does not kill the
paper, but it changes the training claim from "final fitting improves request
behavior" to "the correction contract is learnable under development fitting
and best deployed as a request-gated specialist/postranker."
