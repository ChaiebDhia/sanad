# ADR 0005: Serve the tuned model through an OpenAI-compatible batching server

Status: accepted

**Context.** citegate evaluates any model through the OpenAI chat protocol, and free GPUs are slow without batching.

**Decision.** `sanad serve` exposes `/v1/chat/completions` with dynamic batching (up to `max_batch`, waiting at most `max_wait_ms`). citegate and `sanad bench` talk to it exactly as they would to a hosted model. The protocol and batching are tested with a fake backend and citegate's real client; the Hugging Face backend needs a GPU and is verified manually.

**Consequences.** One code path for hosted and local models; a free T4 can evaluate hundreds of cases; the benchmark measures the same endpoint that is graded. Limit: no authentication, binds to localhost; not a production server.

**Alternatives.** Evaluate in-process (a second code path); vLLM only (heavier to run on free notebooks, kept as an option in `deploy/`).
