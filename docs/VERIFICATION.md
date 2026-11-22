# What has and has not been verified

Last updated: 2026-11-21 (v0.3.1, unreleased).

## Verified by running it
- First Windows run (Python 3.14, PowerShell): `mypy` clean and 24 of 25 tests passed; the one failure (a UTF-8 decoding bug in a test) is fixed and the suite is now 26 tests.
- First Kaggle run (T4-class GPU, Python 3.13): `HFBackend` / `sanad serve` served the **base model**; `citegate run`, `sanad dev-eval` and `sanad bench` completed against it (results in `docs/RESULTS.md`). The SFT smoke test crashed (bf16 on a GPU without native bf16); fixed in code and unit-tested, not yet re-run on a GPU.
- `pytest`: 30 tests (one needs the `citegate` executable on PATH and is skipped if absent); line coverage floor 85% with GPU-only code excluded.
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
- [ ] `HFBackend` serving a LoRA **adapter** and a merged model (the base-model path has run).
- [ ] `train/sft_qlora.py` after the fp16/TRL-fallback fix, `train/dpo.py` and `train/merge_adapter.py` on a GPU. Smoke-test 20 steps and pin library versions.
- [ ] The base model loads, tokenises Arabic efficiently, and its licence fits your use.
- [ ] `notebooks/sanad_kaggle.ipynb` sections 4 to 6 (sections 0 to 3 have run).
- [ ] `deploy/serve_vllm.sh`, `deploy/serve_llamacpp.sh` against current tools.
- [ ] The GitHub Actions workflows on GitHub (written and mirrored locally, not run there).
- [ ] SFT and DPO results (only the untouched-base baseline is measured).
