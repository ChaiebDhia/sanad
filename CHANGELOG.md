# Changelog

## 0.3.1 (unreleased)
- Added an end-to-end pipeline test on a tiny random model (`pytest -m e2e`) and a weekly `pipeline-smoke` workflow that runs it against the newest transformers, TRL and PEFT.
- Training: `precision` option (auto, fp16, bf16, fp32), NaN-loss guard with a clear message, first-step loss logging, smaller and faster evaluation during SFT (`eval_samples`, eval batch size 1), `load_4bit` option, DPO and SFT run on CPU for smoke tests, merge script uses the version-aware dtype helper.
- Notebook: one GPU, memory-fragmentation setting, optional pin to the tested library versions (`constraints/tested-versions.txt`).
- First measured baseline recorded in `docs/RESULTS.md` (untouched Qwen2.5-1.5B-Instruct: 3.8% on citegate's golden set, 11.4% on Sanad's dev world).
- Fix: training and serving pick fp16 on GPUs without native bf16 (T4 crash and slow serving), with a TRL fp32 matmul fallback.
- Fix: weights are loaded in the chosen dtype and trainable parameters are forced to fp32 (second T4 smoke-test failure: bf16 gradients in the fp16 gradient scaler); serving uses the same loader.
- Second baseline run (fp16 compute) recorded: same quality, 3.4x lower median latency.
- Fix: the Kaggle notebook stops on the first failed command and checks each step's output.

## 0.3.0 (2026-11-18): first public release (pre-release: no model results yet)
- Sanad is the training and serving half of a pipeline; evaluation is delegated to citegate (pinned commit).
- Data synthesis in citegate's exact prompt/answer contract with its own vocabulary and phrasing; zero-overlap leakage check against citegate's worlds.
- Verified SFT/DPO builders: chosen answers pass and rejected answers fail citegate's metrics.
- OpenAI-compatible batching server, serving benchmark (latency percentiles, tokens/s, cost per 1,000 answers), CLI.
- QLoRA SFT/DPO scripts and a Kaggle notebook (not yet run on a GPU).
- Decision records, free-stack guide and deployment playbook.

- Windows fixes found on the first Windows run: explicit UTF-8 in tests, console-safe printing of Arabic, SPDX licence string, `.gitattributes`, Python 3.14 in CI.

Known gaps: no language-model results; GPU paths unverified (`docs/VERIFICATION.md`).
