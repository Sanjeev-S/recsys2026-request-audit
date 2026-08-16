# Exact/Version Robustness Table v0.9

Purpose: address the reviewer concern that exact/version evidence is a tiny or cherry-picked slice.

Read: Exact/version requests are small but not a single cherry-picked row: they appear across turns, include reachable and missing-top100 cases, and version/duplicate equivalents are explicitly rare.

## Split Coverage

| split | visible exact directives | policy/request conflicts | sidecar family counts | requested present top-100 | reachable promotion slice | absent top-100 |
|---|---:|---:|---|---:|---:|---:|
| devset | 82 | 41/82 (50.0%) | exact_track_request: 42; version_duplicate_equivalence: 1 | 65/82 (79.3%) | 19/82 (23.2%) | 17/82 (20.7%) |
| val | 247 | 212/247 (85.8%) | exact_track_request: 215 | 227/247 (91.9%) | 88/247 (35.6%) | 20/247 (8.1%) |

## Turn Position

| split | turn bucket | directives | policy/request conflicts | already first | promoted | absent top-100 |
|---|---|---:|---:|---:|---:|---:|
| devset | turn 1 | 30 | 5/30 (16.7%) | 30/30 (100.0%) | 0/30 (0.0%) | 0/30 (0.0%) |
| devset | turns 2-4 | 35 | 24/35 (68.6%) | 12/35 (34.3%) | 7/35 (20.0%) | 16/35 (45.7%) |
| devset | turns 5-8 | 17 | 12/17 (70.6%) | 4/17 (23.5%) | 12/17 (70.6%) | 1/17 (5.9%) |
| val | turn 1 | 63 | 47/63 (74.6%) | 60/63 (95.2%) | 2/63 (3.2%) | 1/63 (1.6%) |
| val | turns 2-4 | 69 | 60/69 (87.0%) | 49/69 (71.0%) | 13/69 (18.8%) | 7/69 (10.1%) |
| val | turns 5-8 | 115 | 105/115 (91.3%) | 30/115 (26.1%) | 73/115 (63.5%) | 12/115 (10.4%) |

Interpretation: the exact/version slice is intentionally narrow, but the evidence is not a single-turn artifact. The main failure mode is retrieval availability: when a request-satisfying item is absent from top-100, neither postranking nor a reranker can promote it from the submitted list.
