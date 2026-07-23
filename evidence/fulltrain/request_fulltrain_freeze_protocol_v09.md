# Request Full-Train Freeze Protocol v0.9

Date: 2026-06-23

## Purpose

This freezes the full-train request-aware training readout before any paid
full-train pod run is executed.

The methodological risk is simple: once repo `val` is folded back into
organizer train, repo `devset` must be a held-out readout, not another tuning
surface. Therefore the full-train protocol has one primary blend per family.

## Frozen Choices

| family | training cells | feature | primary dev blend | label action |
|---|---|---|---:|---|
| exact/version | `official`, `exact_positive_weighted` | `exact_request_match` | `1.0` | add request-satisfying positives; corrected groups weight `0.1` |
| hard simple/non-exact | `official`, `violation_drop` | `hard_artist_constraint_match` | `0.5` | original labels unchanged; zero-weight clear violation groups |

Shared choices:

- Train on `train_a + val`: 121,592 groups and 203,221,270 candidate rows.
- Train fixed `50` boosting rounds.
- Use `--skip-val-dmatrix`; no early stopping or hyperparameter selection on
  repo `val`.
- Use repo `devset` only for the predeclared readout.
- Treat any additional blend weights as sensitivity analysis, not as the
  reported full-train result.

## Machine Check

Manifest:

- `docs/evidence/request_fulltrain_freeze_manifest_v09.json`

Verifier:

```bash
.venv/bin/python scripts/verify_request_fulltrain_freeze.py
```

Current verification result:

```json
{
  "artifact_mismatches": [],
  "invariant_failures": [],
  "n_artifacts": 31,
  "n_scripts": 12,
  "ok": true,
  "protocol": "request-fulltrain-v0.9-frozen-readout",
  "script_mismatches": []
}
```

The verifier checks both file hashes and protocol invariants:

- exact full-train added positives: 1,878
- exact full-train downweighted groups: 1,873
- exact request-feature matched rows/groups: 2,207 / 2,186
- hard-drop full-train dropped groups: 361
- hard request-feature matched rows/groups: 34,153 / 760
- exact protocol has only `--blend-weight 1`
- hard protocol has only `--blend-weight 0.5`
- pod launchers use pinned dependency versions
- hard protocol no longer references the stale `105k` launcher

## What This Proves

This proves the full-train experiment is ready to run as a frozen protocol. It
does not prove the full-train models improve request satisfaction. That remains
pending until the approved pod training jobs are run and the frozen devset
readouts are evaluated.
