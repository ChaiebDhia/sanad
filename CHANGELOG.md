# Changelog

## 0.3.1 (unreleased)
- First measured baseline recorded in `docs/RESULTS.md` (untouched Qwen2.5-1.5B-Instruct: 3.8% on citegate's golden set, 11.4% on Sanad's dev world).
- Fix: training and serving pick fp16 on GPUs without native bf16 (T4 crash and slow serving), with a TRL fp32 matmul fallback.
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
