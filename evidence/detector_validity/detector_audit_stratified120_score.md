# Request-Frame Parser Ablation v0.8

Date: 2026-06-21

This is a stratified annotation check, not a prevalence estimate. A detector-positive row is counted as active when the independent label names the same requested title with sufficient confidence. A control row is counted as active when the independent label says `is_explicit_track_request=yes`, `is_reference_or_example` is not `yes`, and confidence meets the configured threshold.

Minimum confidence: `medium`

## Summary

| readout | count |
|---|---:|
| detector-positive active exact request | 37/40 (0.925, CI [0.801, 0.974]) |
| control active exact request | 3/80 (0.037, CI [0.013, 0.105]) |

## By Bucket

| bucket | n | active exact request | title overlap among active |
|---|---:|---:|---:|
| quoted_not_strict_detected | 40 | 3/40 (0.075, CI [0.026, 0.199]) | 0/3 (0.000, CI [0.000, 0.561]) |
| strict_detected_exact_request | 40 | 37/40 (0.925, CI [0.801, 0.974]) | 37/37 (1.000, CI [0.906, 1.000]) |
| unquoted_request_like | 40 | 0/40 (0.000, CI [0.000, 0.088]) | 0/0 (0.000, CI [0.000, 0.000]) |

## Read

- High detector-positive active rate supports that the regex detector is recovering a real request frame.
- Nonzero control active rate identifies detector recall gaps or ambiguous requests for follow-up inspection.
- Title overlap checks whether the independent parser names the same requested item as the detector.
