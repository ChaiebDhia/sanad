# Sanad (سند)

**Can a small open model, post-trained by one person on free hardware, pass citegate's grounding gate in English, French and Arabic, and what does it cost to serve?**

Sanad is the **training and serving half** of a two-repo pipeline. It deliberately contains **no evaluation logic**: [citegate](https://github.com/ChaiebDhia/citegate) is the evaluator (its retriever, prompt, adapter, metrics, statistics and reports). Sanad generates data in citegate's exact contract, trains small models (QLoRA SFT, then DPO), serves them behind an OpenAI-compatible endpoint, and measures latency and cost.

```
 Sanad styles ──► data-build ──► SFT / DPO files ──► train (free Kaggle GPU)
 (own vocabulary,   (verified by         │
  own phrasing)      citegate metrics)   ▼
                                    sanad serve  ◄── OpenAI-compatible, batching
                                         │
              citegate run / compare ◄───┤   judged by the independent test
              sanad dev-eval         ◄───┤   (citegate's metrics on Sanad's dev world)
              sanad bench            ◄───┘   latency, throughput, cost per 1,000 answers
```

## Status (read this first)

| Part | State |
|---|---|
| Data generation, leakage check against citegate, SFT/DPO builders, validators | Built; 26 tests; every training answer is verified by citegate's own metrics |
| Batching OpenAI-compatible server, serving benchmark, CLI | Built; tested end to end with citegate's real adapter, runner and command line (fake model) |
| QLoRA SFT / DPO scripts, `HFBackend` (GPU server), Kaggle notebook | Written; **not yet run on a GPU** |
| **Language-model results** | **None yet.** No model has been trained or evaluated |

Nothing in this repo claims a model result until a run produced it and `docs/RESULTS.md` records the command, seed, hardware and versions.

## No overlap with citegate, by construction

| Concern | Owner | Notes |
|---|---|---|
| Golden test set, retriever, prompt, adapters, metrics, Wilson/paired statistics, mutation testing, reports | **citegate** | Sanad imports it (pinned to an exact commit) and never re-implements it |
| Training-data synthesis in citegate's prompt format | Sanad | Own services, sentence patterns, currency, units, injection wording and canary tokens, so the test is genuinely held out |
| Verified preference pairs for the failures citegate measures | Sanad | Chosen answers must pass citegate's metrics; rejected answers must fail them |
| Leakage check between training data and citegate's worlds | Sanad | Zero exact overlap of names, questions, passages, sentences, canaries is a CI failure; 4-gram overlap is reported |
| Training, serving, cost benchmark | Sanad | Not in citegate |

A model that does well on citegate after training on Sanad data has **generalised across phrasing and vocabulary**; it cannot have memorised citegate's templates.

## Quickstart (works today, no GPU, no API key)

```bash
python -m venv .venv            # Windows: py -m venv .venv
# activate: Windows .venv\Scripts\Activate.ps1   |   macOS/Linux source .venv/bin/activate
pip install -e ".[dev]"         # also installs citegate from GitHub (needs git)
sanad data-build --out data     # builds data, runs the leakage check, writes data/build_report.json
sanad data-check --data data
sanad show --data data --file dpo.train.jsonl --index 3
pytest
```

## Train, evaluate, measure (free)

The fastest path is `notebooks/sanad_kaggle.ipynb` on a free Kaggle GPU. By hand on a GPU machine:

```bash
pip install -e ".[train]"
python train/sft_qlora.py --max-steps 20          # smoke test first
python train/sft_qlora.py                         # full run (--resume after an interrupted session)
sanad serve --model BASE_ID --adapter runs/sft --name sft --port 8000
citegate run --agent llm --provider ollama --base-url http://127.0.0.1:8000/v1 --model sft \
             --concurrency 8 --out runs/sft.jsonl --html runs/sft.html
sanad dev-eval --base-url http://127.0.0.1:8000/v1 --model sft --out runs/sft-dev.jsonl
sanad bench    --base-url http://127.0.0.1:8000/v1 --model sft --price-per-hour 0.40
citegate compare runs/base.jsonl runs/sft.jsonl
```

## Why the pipeline can be trusted

- **Same prompt as the test.** A test fails if Sanad's prompt or parsing ever differs from what citegate's `LLMAgent` sends and reads.
- **Training signal verified by the grader.** Every chosen completion passes citegate's metrics and every rejected completion fails them, or the pair is dropped and counted (`data/build_report.json`).
- **Held out by construction and by check.** See the table above and `tests/test_leakage.py`, which also proves the check fires on citegate's own data.
- **Hard cases on purpose.** Planted injections (different wording and canaries from citegate's), superseded guidance, out-of-scope questions, multi-hop questions, and "retrieval miss" examples where the passages are about the right topic but the wrong place (the model must refuse).
- **Reproducible.** Same seeds, same bytes (checked in CI).

## Limitations

- citegate's golden set is synthetic and templated, and its reference agent passes it. A high model score is therefore a floor, not proof of real-world readiness. Sanad's own dev world is a second, independent sample; a real-document set is planned.
- Sanad's worlds share citegate's *task structure* (fee, days, minimum age, office, plus traps). Success shows transfer across wording and vocabulary, not across task types.
- Retrieval is fixed (BM25, k=4) because that is what citegate uses.
- DPO negatives are synthetic defects, not human-rated model mistakes.
- Cost numbers for free hardware are $0 in cash; always also report the rented-GPU price (`--price-per-hour`).

## Layout

```
src/sanad/   styles, world, prompts, data, leakage, validate, serve, client, bench, cli
train/       sft_qlora.py, dpo.py, merge_adapter.py
configs/     sft.toml, dpo.toml
notebooks/   sanad_kaggle.ipynb
docs/        FREE_STACK, CITEGATE, DEPLOYMENT_PLAYBOOK, VERIFICATION, RESULTS,
             CORRECTIONS, MODEL_CARD, adr/ (decision records)
tests/       26 tests
```

MIT licensed. See `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`.
