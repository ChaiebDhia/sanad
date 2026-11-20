# ADR 0004: Preference pairs are verified by the grader

Status: accepted

**Context.** Synthetic "bad" answers are only useful if they really are bad by the metric that will be used.

**Decision.** Rejected completions are generated for the failure modes citegate measures (over-refusal, hallucination, forged or missing citations, wrong value, language drift, obeyed injection, stale answer, wrong entity). Each is scored by citegate's `score_case`; if it passes, the pair is dropped and counted. Targeted defects always get a pair; one generic defect per example is added.

**Consequences.** The training signal and the test agree on what failure means. Limit: defects are synthetic; real model mistakes may differ (a later iteration can mine them from model runs).

**Alternatives.** Random corruption (not verified); LLM-written negatives (costly, unverified).
