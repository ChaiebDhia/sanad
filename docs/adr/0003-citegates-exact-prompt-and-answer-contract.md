# ADR 0003: Train in citegate's exact prompt and answer contract

Status: accepted

**Context.** A fine-tune that learned a private JSON format would need an adapter at test time, and any mismatch would silently change what is measured.

**Decision.** The user message is built by citegate's own `PROMPT` and BM25 retrieval (k=4). The completion is quoted sentences followed by `[chunk-id]` citations, or `REFUSE`, exactly what citegate's `LLMAgent` parses. A test compares Sanad's prompt and parser with `LLMAgent` on real cases.

**Consequences.** No adapter, no format drift; if citegate changes, the test fails. Limit: tied to citegate's current contract.

**Alternatives.** A private JSON schema (needs a translation layer, hides mismatches).
