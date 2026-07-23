# Request Preference Labels

Date: 2026-06-21

This scores blinded comparisons between the official policy-selected label and the request-satisfying item promoted by the locked exact-request postranker.

Minimum confidence: `medium`

## Summary

| readout | count |
|---|---:|
| request item preferred | 12/12 (1.000, CI [0.758, 1.000]) |
| policy item preferred | 0/12 (0.000, CI [0.000, 0.242]) |
| request item marked request-satisfying | 12/12 (1.000, CI [0.758, 1.000]) |

## Raw Counts

- preferred_item: `{'A': 9, 'B': 3}`
- request_satisfying_item: `{'A': 9, 'B': 3}`
- explicit_request_visible: `{'yes': 12}`
- confidence: `{'high': 12}`

## By Split

| split | counts |
|---|---|
| devset | `{'request_preferred': 12, 'request_satisfying': 12}` |

## Read

- This validates the external meaning of corrected nDCG: whether the request-positive target is preferred for the visible user request.
- It does not validate broad music preference, only request satisfaction in exact-request conflict rows.
