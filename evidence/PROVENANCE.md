# Provenance

Facts about where these files came from, for anyone auditing the
evidence rather than just reading it.

- Files were renamed to plain names in August 2026. The freeze manifest
  (`training/freeze_manifest.json`) and the two `verify_request_*.py`
  scripts are frozen records from the research repo: the paths and
  sha256 hashes inside them refer to the original internal files, and
  they are kept byte-for-byte rather than rewritten.
- Table numbering follows the camera-ready paper: Table 1 is the rank
  readout, Table 2 the score readout. Some frozen files use the older
  internal numbering in their text.
- The models in `models/` are a July 12, 2026 retrain, not the bytes
  from the frozen evaluation run. The retrain reproduces every Table 2
  cell exactly; `results/repro_confirmation.json` records the check.
- `results/table1_ranks.idkeyed.jsonl` was extracted from the frozen
  gated evaluation on July 24, 2026 and re-verified against the
  retrain's prediction files with zero differences.
- The canonical added-row count is 1,878. A July 12 retrain read 1,877
  because a cached copy of the validation feature file differed;
  `training/added_rows.json` has the reconciliation.
- The raw labels behind the 120-turn detector audit (37/40) were lost;
  the aggregate score and the scorer ship, and the packet can be
  regenerated, but the original labels cannot.
- The validation-split availability counts in `audit/coverage_val.json`
  were regenerated after the original ranking file was lost; the
  detector-side counts are exact. Nothing in that file is cited by the
  paper.
- A few files had absolute internal paths rewritten to `<repo-root>`
  placeholders before release.
