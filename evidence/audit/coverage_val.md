# Request Postrank Coverage: val

Date: 2026-06-21

This analyzes the locked exact-request postranker before intervention. It uses visible dialogue, catalog metadata, and the baseline prediction list. Policy labels are summarized only after the request frame is resolved.

Prediction file: `/tmp/claude-0/-root-sanjeev-recsys-challenge-2026/fc45f04c-af58-4807-b3ab-916167538fbe/scratchpad/val_predictions/request_corrected_F10_R54SRC_blend_production_anchor_identity_w1_val.json`
k-search: `100`

## Funnel

| readout | count |
|---|---:|
| exact-request directive turns | 247 |
| changed by postranker | 68/247 (27.5%) |
| already request-satisfying at front | 139/247 (56.3%) |
| requested item absent from top-100 | 40/247 (16.2%) |

## Rank Of Best Requested Item

| rank bucket | count |
|---|---:|
| 1 | 139/247 (56.3%) |
| 2-5 | 28/247 (11.3%) |
| 6-20 | 40/247 (16.2%) |
| 21-100 | 0/247 (0.0%) |
| >100/missing | 40/247 (16.2%) |

## Policy Label Relation

| relation | count |
|---|---:|
| gold_requested | 35/247 (14.2%) |
| gold_differs | 212/247 (85.8%) |
| gold_missing | 0/247 (0.0%) |

## Read

- `promoted` is the reachable fix slice: the requested item is already in top-k but not at the front.
- `already_front` is not a failure: the existing system already puts a request-satisfying item first.
- `not_retrieved_topk` is the retrieval ceiling for this postranker.
- `gold_differs` is the observable proxy-boundary slice: the policy-selected label differs from the visible request-satisfying item.
