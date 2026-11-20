# Contributing

Before opening a pull request run, in the project folder with the virtual environment active:

```
ruff check . && ruff format --check . && mypy && pytest --cov=sanad --cov-fail-under=85
sanad data-build --out data && sanad data-check --data data
```

- Evaluation logic belongs in **citegate**, not here. Sanad generates data, trains, serves and benchmarks.
- A change to `world.py` or `styles.py` must keep the leakage check against citegate green.
- Quote a number only with the command, commit, seed and hardware that produced it.
