# recsys2026-request-audit

Code and evidence for the RecSys Challenge 2026 paper "When the Label
Ignores the Request". The paper audits turns where a user asks for an
exact song by name and the benchmark's official label points to a
different track, then trains a model that honors those requests without
hurting the official metric.

This repo lets you re-run that audit and reproduce the paper's numbers
against your own copy of the challenge data. The data itself is not
included: it is licensed CC-BY-NC-ND and stays on HuggingFace
(`talkpl-ai/TalkPlayData-Challenge-Dataset`,
`talkpl-ai/TalkPlayData-Challenge-Track-Metadata`). Evidence files here
use session, turn, and track IDs only, never dialogue text.

## What is here

- The request detector and resolver (Section 5).
- Training for the specialist and its matched control (Section 6).
- The readouts behind Table 1 and Table 2 (Section 7).
- ID-keyed annotations and score files for every number in the paper.
  [CLAIMS.md](CLAIMS.md) maps each claim to its script and evidence file.

## Reproduce

    pip install -r requirements.txt
    python run_audit.py --split devset --prediction-path <your ranking>
    ./train.sh
    python src/reproduce_table1.py
    ./reproduce_table2.sh <control predictions> <specialist predictions>

The commands assume a Python 3.12 environment with the requirements
installed and active. Training needs the feature matrices described in
the paper's Section 2; they are built outside this repo and are not
redistributed. The trained models are included (`evidence/models/`), so
the tables reproduce without retraining. The two prediction files for
`reproduce_table2.sh` are rebuilt from the shipped models with
`src/run_gated_eval.py`, which needs the same feature matrices. Set
`MCRS_EXPLORE_ROOT` to your data root for the scripts that read them.

`evidence/` also holds validation packets the paper does not cite, and
[evidence/PROVENANCE.md](evidence/PROVENANCE.md) records where each
file came from.

## License

Code: MIT (see LICENSE). Evidence: IDs and scores only, no benchmark
content.
