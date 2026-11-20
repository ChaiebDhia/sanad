# The all-free stack

Free tiers change without notice: **check each provider's current limits and data policy in your own account.** Only synthetic data is used anywhere, so provider data terms are not a privacy risk here; never send anything private.

| Need | Free resource | How |
|---|---|---|
| Code, issues, CI, security scans | GitHub (public repos) + Actions + CodeQL + Dependabot | `.github/` |
| Training GPU | Kaggle Notebooks (T4 x2 / P100); Colab free as a backup | `notebooks/sanad_kaggle.ipynb`. Weekly quota and session length are set by the provider (commonly reported: ~30 GPU hours/week, 12-hour sessions; confirm in your account). Use `--resume` |
| Model and adapter hosting | Hugging Face Hub | upload cell in the notebook; respect the base model's licence |
| Experiment tracking | MLflow local files, or `report_to = "none"` | `configs/sft.toml`; commit result files, not the MLflow folder |
| Evaluation | citegate (zero dependencies) | `citegate run`, `citegate compare` |
| Hosted-model comparison | A free-tier API with an OpenAI-compatible endpoint (Groq, Gemini compatibility endpoint, or similar) | `citegate run --provider groq ...`; resumable; name the exact model and date |
| CPU serving test | Ollama or llama.cpp on your laptop | `sanad bench --base-url http://localhost:11434/v1 --model NAME` |
| Public report | citegate's HTML report on GitHub Pages | `citegate run --html`; publish through Pages if you want a link |

## Order of work that wastes the least quota
1. `--sample 30` smoke test of every new command. 2. Baseline. 3. 20-step training smoke test, then the full run. 4. Only then spend quota on hosted-model comparisons.
