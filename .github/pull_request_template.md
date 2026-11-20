## What and why
<!-- One paragraph. Link the issue. -->

## Evidence
- [ ] `ruff check .`, `ruff format --check .`, `mypy` and `pytest` pass locally
- [ ] New behaviour has a test that fails without the change
- [ ] If data generation changed: `sanad data-build` still passes the leakage check against citegate
- [ ] If a number is quoted: the command, commit, seed and hardware are recorded in `docs/RESULTS.md`
- [ ] `CHANGELOG.md` updated; `docs/CORRECTIONS.md` updated if something earlier was wrong

## Risks and follow-ups
