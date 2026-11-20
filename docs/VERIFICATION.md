# What has and has not been verified

Last updated: 2026-11-18 (v0.3.0).

## Verified by running it
- `pytest`: 25 tests (one needs the `citegate` executable on PATH and is skipped if absent); line coverage floor 85% with GPU-only code excluded.
- `ruff check`, `ruff format --check`, strict `mypy`: clean.
- Prompt and answer parsing are identical to citegate's `LLMAgent` on real cases.
- Every chosen training completion passes, and every rejected completion fails, citegate's `score_case`.
- Leakage check against citegate's default and a larger world: no exact overlap; 4-gram overlap about 0.04%. The check demonstrably fires on citegate's own data and on a shared canary.
- Data builds are deterministic.
- End to end with citegate itself: Sanad data -> `sanad serve` machinery -> citegate's `LLMAgent` + `run_cases` -> metrics (fake oracle model, so 100% by design). Dynamic batching observed (batch size > 1). Backend errors return HTTP 500, not a hang.
- The real `citegate run --agent llm --provider ollama --base-url <server>` command line runs against the server.
- `sanad bench` percentiles, token counts and cost arithmetic.
- The Kaggle notebook is valid JSON.

## NOT verified (do these before claiming anything)
- [ ] `HFBackend` / `sanad serve` with a real model on a GPU (chat template, left padding, batching, memory).
- [ ] `train/sft_qlora.py`, `train/dpo.py`, `train/merge_adapter.py` on a GPU; TRL/PEFT argument names vary by version. Smoke-test 20 steps and pin versions.
- [ ] The base model loads, tokenises Arabic efficiently, and its licence fits your use.
- [ ] `notebooks/sanad_kaggle.ipynb` top to bottom.
- [ ] `deploy/serve_vllm.sh`, `deploy/serve_llamacpp.sh` against current tools.
- [ ] The GitHub Actions workflows on GitHub (written and mirrored locally, not run there).
- [ ] Any result for any real model.
