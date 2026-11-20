# Corrections log

Dated record of anything that was wrong, did not reproduce, or was retracted. Newest first.

| Date | What was claimed or built | What is true | Fix |
|---|---|---|---|
| 2026-11-18 | The 0.2 prototype shipped its own scorer, gate, mutants, baseline and results page | That duplicated citegate. It was removed; Sanad now imports citegate. | `docs/adr/0001-evaluation-is-delegated-to-citegate.md` |
| 2026-11-18 | A rule-based baseline scored 100% on the prototype's templated eval, and 94% before a case-sensitivity bug in the baseline was fixed | Templated evals are easy for rules. The lesson carries over: citegate's reference agent passes its golden set, so a high model score is a floor, not proof. | `README.md` limitations; a shift suite is a candidate upstream contribution to citegate |
| 2026-11-18 | Initial draft of this repository was assisted by an AI model | Reviewed, tested and extended by the author before any result is reported | `docs/VERIFICATION.md` |
