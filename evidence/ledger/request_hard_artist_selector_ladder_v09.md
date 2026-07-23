# Hard-Artist Selector Ladder v0.9

Date: 2026-06-22

Purpose: quantify the precision/coverage tradeoff that keeps hard-artist as supporting evidence rather than an admitted correction family.

## Selector Ladder

| selector | dev directives | dev changed | val directives | val changed | val labeled precision | val action ok | gold should drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| broad strict hard-artist | 135 | 8 | 387 | 50 | 37/50 (74.0%, 95% CI 60.4%-84.1%) | 40/50 | 46/50 |
| exclude exact-title turns | 92 | 8 | 244 | 33 | 21/33 (63.6%, 95% CI 46.6%-77.8%) | 24/33 | 30/33 |
| simple/no-exact action selector | 50 | 4 | 83 | 13 | 12/13 (92.3%, 95% CI 66.7%-98.6%) | 13/13 | 11/13 |

## Simple/Non-Negated Training Sidecar

| split | broad input records | selector keys before nonneg | selector keys after nonneg | kept conflict records | kept rate | dropped reasons |
|---|---:|---:|---:|---:|---:|---|
| train_lt | 1587 | 715 | 673 | 329 | 20.7% | negated_constraint_artist=7, not_simple_noexact_selector=1251 |
| devset | 31 | 50 | 49 | 16 | 51.6% | not_simple_noexact_selector=15 |
| val | 155 | 83 | 71 | 32 | 20.6% | negated_constraint_artist=2, not_simple_noexact_selector=121 |

## Read

The broad hard-artist family is learnable, but its changed-action precision is too weak for admission. Excluding exact-title turns alone does not fix precision. The simple/no-exact action selector reaches 12/13 labeled strict precision on val changed rows, but it changes only 13 val rows. The training sidecar applies one more non-negation filter and keeps 329/1,587 train conflict records, 16/31 dev records, and 32/155 val records.

Independent labels cover postrank selector changed rows only. The learned-specialist action audits are detector-coupled mechanism checks, not independent changed-row precision for the learned model.

Paper-safe claim: hard-artist is a useful learned stress test and frozen-transfer candidate. It is not an admitted correction family until an untouched split confirms the selector and independent action precision.

## Sources

- `docs/evidence/request_hard_artist_strict_postrank_top20_rerank_F10_scaled_devset.summary.json`
- `docs/evidence/request_hard_artist_strict_postrank_top20_rerank_F10_R54SRC_val.summary.json`
- `docs/evidence/request_hard_artist_strict_top20_changed_rows_val_v09.summary.json`
- `docs/evidence/request_hard_artist_strict_top20_changed_rows_val_labels_v09.summary.json`
- `docs/evidence/request_hard_artist_strict_postrank_top20_noexact_rerank_F10_scaled_devset.summary.json`
- `docs/evidence/request_hard_artist_strict_postrank_top20_noexact_rerank_F10_R54SRC_val.summary.json`
- `docs/evidence/request_hard_artist_strict_top20_noexact_changed_rows_val_v09.summary.json`
- `docs/evidence/request_hard_artist_strict_top20_noexact_changed_rows_val_labels_v09.summary.json`
- `docs/evidence/request_hard_artist_strict_postrank_top20_simple_noexact_rerank_F10_scaled_devset.summary.json`
- `docs/evidence/request_hard_artist_strict_postrank_top20_simple_noexact_rerank_F10_R54SRC_val.summary.json`
- `docs/evidence/request_hard_artist_strict_top20_simple_noexact_changed_rows_val_v09.summary.json`
- `docs/evidence/request_hard_artist_strict_top20_simple_noexact_changed_rows_val_labels_v09.summary.json`
- `docs/evidence/request_corrections_train_lt_v09_simple_nonneg_hard_artist_only.summary.json`
- `docs/evidence/request_corrections_devset_v09_simple_nonneg_hard_artist_only.summary.json`
- `docs/evidence/request_corrections_val_v09_simple_nonneg_hard_artist_only.summary.json`
