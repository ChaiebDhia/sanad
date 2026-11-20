# ADR 0001: Evaluation is delegated to citegate

Status: accepted. Supersedes the 0.2 prototype, which contained its own scorer, gate and mutants.

**Context.** The prototype re-implemented citegate's ideas (deterministic metrics, Wilson floors, mutation-tested evals, the same failure categories). Two evaluators in one portfolio is duplication, and a model judged by its own author's second grader is weaker evidence than one judged by a separate, already-published test.

**Decision.** Sanad imports citegate (pinned to a commit) and uses its retriever, prompt, `LLMAgent`, `run_cases`, `score_case`, summaries and CLI. Sanad contains no scoring code.

**Consequences.** One source of truth for what "grounded" means; citegate's improvements flow to Sanad; the independent test stays independent. Cost: Sanad depends on a git-pinned package and must be updated deliberately when citegate changes (the prompt-equivalence test will fail loudly).

**Alternatives.** Vendoring citegate (drifts); keeping a second evaluator (duplication).
