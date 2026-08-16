# Request Postrank Coverage: devset

Date: 2026-06-21

This analyzes the locked exact-request postranker before intervention. It uses visible dialogue, catalog metadata, and the baseline prediction list. Policy labels are summarized only after the request frame is resolved.

Prediction file: `exp/inference/devset/rerank_F10_scaled_devset.json`
k-search: `100`

## Funnel

| readout | count |
|---|---:|
| exact-request directive turns | 82 |
| changed by postranker | 19/82 (23.2%) |
| already request-satisfying at front | 46/82 (56.1%) |
| requested item absent from top-100 | 17/82 (20.7%) |

## Rank Of Best Requested Item

| rank bucket | count |
|---|---:|
| 1 | 46/82 (56.1%) |
| 2-5 | 8/82 (9.8%) |
| 6-20 | 11/82 (13.4%) |
| 21-100 | 0/82 (0.0%) |
| >100/missing | 17/82 (20.7%) |

## Policy Label Relation

| relation | count |
|---|---:|
| gold_requested | 41/82 (50.0%) |
| gold_differs | 41/82 (50.0%) |
| gold_missing | 0/82 (0.0%) |

## Read

- `promoted` is the reachable fix slice: the requested item is already in top-k but not at the front.
- `already_front` is not a failure: the existing system already puts a request-satisfying item first.
- `not_retrieved_topk` is the retrieval ceiling for this postranker.
- `gold_differs` is the observable proxy-boundary slice: the policy-selected label differs from the visible request-satisfying item.
