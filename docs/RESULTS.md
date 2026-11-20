# Results

**No language-model results yet.** Fill this file only from runs you executed, with the exact command, git commit, seed, hardware and library versions for each row.

## Run metadata
| Field | Value |
|---|---|
| Base model + revision | |
| Hardware | |
| transformers / peft / trl / bitsandbytes versions | |
| citegate commit (see `pyproject.toml`) | |
| Date | |

## citegate golden set (independent test, 744 cases)
Overall pass rate with Wilson 95% interval, then per `kind` and per `lang` (copy from `citegate compare`).

| System | Overall | injection | stale | out_of_scope | multihop | en | fr | ar |
|---|---|---|---|---|---|---|---|---|
| Base (untouched) | | | | | | | | |
| SFT | | | | | | | | |
| SFT + DPO | | | | | | | | |
| Hosted model (name, date) | | | | | | | | |

Paired deltas (`citegate compare`): record interval, not only the point estimate.

## Sanad dev world (`sanad dev-eval`, same metrics)
| System | Overall | injection | stale | out_of_scope | retrieval-miss-like | multihop |
|---|---|---|---|---|---|---|

## Serving (`sanad bench`)
| Setup | p50 | p95 | p99 | tok/s | $ per 1k answers (free = $0) | $ per 1k at rented-GPU price | Notes |
|---|---|---|---|---|---|---|---|

## Negative and surprising results
Write them here as they happen.
