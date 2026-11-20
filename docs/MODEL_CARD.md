---
license: apache-2.0   # must match the base model's licence
language: [en, fr, ar]
base_model: REPLACE_ME
tags: [lora, qlora, dpo, grounded-qa, citations, multilingual]
---
# Sanad adapter (REPLACE_ME)

## What it is
A LoRA adapter that makes REPLACE_ME answer from retrieved passages in English, French and Arabic, cite passage ids, refuse when the passages do not contain the answer, and ignore instructions planted in passages.

## Intended use and limits
Research and evaluation. Trained on synthetic public-service passages; evaluated with citegate. Not validated on real government content; do not use it for decisions affecting people.

## Training
Method, data generator, seeds, hardware, library versions: copy from `docs/RESULTS.md`.

## Evaluation
Paste the tables from `docs/RESULTS.md` with intervals and failure modes. State what the evaluation does not cover.
