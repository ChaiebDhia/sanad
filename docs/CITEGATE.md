# How Sanad uses citegate

Sanad pins citegate to an exact commit in `pyproject.toml`.

## What is reused (imported, never copied)
`citegate.corpus` (World, Case), `citegate.retrieval` (BM25Index), `citegate.llm` (PROMPT, LLMAgent, completers), `citegate.agents`, `citegate.metrics.score_case`, `citegate.runner.run_cases`, `citegate.summary.summarise`, and the `citegate` command line.

## Commands
```bash
# the independent test (citegate's 744-case golden set)
citegate run --agent llm --provider ollama --base-url http://127.0.0.1:8000/v1 --model sft --concurrency 8 --out runs/sft.jsonl --html runs/sft.html
citegate compare runs/base.jsonl runs/sft.jsonl runs/dpo.jsonl          # leaderboard with intervals and paired deltas
# Sanad's own dev world, same metrics
sanad dev-eval --base-url http://127.0.0.1:8000/v1 --model sft --out runs/sft-dev.jsonl
```
`--provider ollama` is used only because it needs no API key; it simply speaks the OpenAI protocol to whatever `--base-url` points at.

## Updating the pin
Change the commit hash in `pyproject.toml`, reinstall, run the tests. The prompt-equivalence test fails if the prompt or parsing changed; rebuild data (`sanad data-build`) and re-run baselines before comparing with older results.

## Proposals to make upstream (in citegate, not here)
1. A **shift suite**: held-out wording for questions and documents, so a high score cannot come from template familiarity.
2. A second **domain** and longer contexts.
3. A small **real-document** subset with hand-written questions.
4. A `--served-by` field in result files so cost/latency from Sanad's benchmark can sit next to quality in one table.
