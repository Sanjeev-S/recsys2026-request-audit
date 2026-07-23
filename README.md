# recsys2026-request-audit

Artifact for **"When the Label Ignores the Request: Auditing Policy-Selected Targets in Synthetic Conversational Music Recommendation"** (RecSys Challenge 2026 paper). Team: sanjeevsuresh.

This repository contains the final, cleaned scripts and ID-keyed evidence needed to reproduce the paper's claims: the exact/version request detector and resolver, the training-target supplement, the evaluation slice readout, and the validation artifacts behind every number in the paper.

## What this is (and is not)

- **Is:** the audit detector/resolver, target-remap script, model training scripts, slice readout + bootstrap, and ID-keyed annotation/score files — everything needed to re-derive the paper's numbers against your own copy of the challenge data.
- **Is not:** a copy of the benchmark. TalkPlayData 2 / RecSys Challenge 2026 data is licensed CC-BY-NC-ND-4.0 and is **not redistributed here**. All evidence files are ID-keyed: they reference turn/track identifiers so every verdict can be re-checked against your own copy of the data, but contain no dialogue text or track metadata.

## Layout

```
audit/        exact/version request detector, catalog resolver, target-remap script
training/     reranker training scripts (specialist and matched control)
evaluation/   request-satisfying slice readout, paired session-level bootstrap
evidence/     ID-keyed annotations and score summaries backing each paper claim
```

The claim map — every number in the paper, the script that produces it, and the evidence file that records it — is the table at the end of this README.

## Reproducing the paper's numbers

1. Obtain the RecSys Challenge 2026 data from the organizers (not included here).
2. `audit/` — run the frozen detector over the development set to reproduce the 82-directive funnel and the 41/82 conflict count.
3. `training/` — build the supplemented training set (correction records) and train specialist + matched control.
4. `evaluation/` — run the frozen readout: official nDCG@20, request-satisfying nDCG@20, and the conflict-slice readout with bootstrap CIs.
5. Cross-check every number against the claim map at the end of this README.

Exact commands and environment pins: TODO (assembly in progress).

## Provenance and hygiene

Scripts here are cleaned copies of the research repo's production paths: no API keys, no internal infrastructure references, no experiment scratch.

## License

Code: MIT (see LICENSE). Evidence files: released for research verification; they contain identifiers and scores only, no benchmark content.
