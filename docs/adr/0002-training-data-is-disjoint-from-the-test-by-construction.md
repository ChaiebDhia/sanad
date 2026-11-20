# ADR 0002: Training data is disjoint from the test, by construction and by check

Status: accepted

**Context.** citegate's corpus is templated. Training on its templates would make a high score meaningless.

**Decision.** Sanad uses its own services, three sentence variants per slot per language, a different currency and units, different injection sentences and canary tokens, and region/city names that exclude citegate's. `sanad.leakage` fails on any exact overlap of entity names, questions, passages, sentences or canaries against citegate's default and a larger world, and reports 4-gram overlap (currently about 0.04%).

**Consequences.** Success on citegate means transfer across wording and vocabulary. Limit: same task structure (fee, days, age, office, traps).

**Alternatives.** Train on citegate seeds with different values (identical templates: leakage in spirit).
