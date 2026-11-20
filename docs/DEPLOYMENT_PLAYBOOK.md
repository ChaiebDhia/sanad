# Deployment playbook: a grounded multilingual assistant for an institution

A working outline for forward-deployed and applied-AI conversations. It describes a method; it does not claim past client experience.

## 1. Discovery questions
Which documents are authoritative and who updates them? Which languages and dialects do users write in? What may never be answered (legal advice, personal data)? Where may data be processed (in-country, on-premise, a cloud region)? What is the cost of a wrong answer versus a refusal?

## 2. Architecture options
| Option | Data leaves the institution? | Quality | Cost | When |
|---|---|---|---|---|
| Hosted frontier API | yes | highest | per token | no residency constraint |
| Small tuned model on the institution's server | no | measured by the gate | hardware + ops | residency or sovereignty rules |
| Hybrid: small model first, escalate | partly | measured | mixed | cost control |

## 3. Evaluation protocol (before launch)
1. Build a client-specific golden set with citegate's structure: answerable, multi-hop, out-of-scope, stale guidance, planted injection, in each user language.
2. Set floors on the Wilson **lower bound** per group (not only the average) and a regression tolerance; record them in the contract.
3. Run the candidate models (hosted, small, tuned) through the same gate; compare paired, with intervals.
4. Add a small human-reviewed sample; record disagreement with the metrics.

## 4. Rollout
Shadow mode (answers logged, not shown) -> limited audience with a visible "sources" panel -> full launch behind the gate in CI. Every document update triggers the gate.

## 5. Monitoring
Refusal rate, citation validity, latency p50/p95, injection canary tests on a schedule, language mix, user "wrong answer" flags feeding new golden cases.

## 6. Cost model template
`cost per 1,000 answers = hardware price/hour x (seconds per answer x 1000 / 3600)` from `sanad bench` at the expected concurrency, compared with token-priced API cost. Report both, with the assumptions.

## 7. Incident handling
A failing gate blocks release; a user-reported wrong answer becomes a case; a provider outage falls back to refusal with a clear message rather than an unsupported answer.
