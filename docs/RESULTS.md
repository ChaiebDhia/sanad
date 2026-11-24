# Results

Only measured runs are recorded here, each with its command and environment. SFT and DPO rows stay empty until those runs finish.

## Run metadata (baseline run)
| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (revision not recorded) |
| Hardware | Kaggle GPU with compute capability below 8.0 (T4 or P100 class) |
| Python / libraries | Python 3.13; transformers, peft, trl and bitsandbytes versions: add from the notebook's `pip list` cell output (not recorded) |
| citegate | commit `c3446ad820fc78e8b0b67d32dfd02d47bbafb672` (see `pyproject.toml`) |
| Date | 2026-11-21 |
| Known caveat | Run with the pre-fix dtype choice: bfloat16 computation that is **emulated** on this GPU class (see `docs/CORRECTIONS.md`). Quality numbers are valid for what was run; latency numbers are not representative |

## citegate golden set (independent test, 744 cases)
Command: `citegate run --agent llm --provider ollama --base-url http://127.0.0.1:8000/v1 --model base --concurrency 8 --out runs/base.jsonl` (served by `sanad serve`).

Pass rate, Wilson 95% interval, by group:

| System | Overall | injection | stale | out_of_scope | multihop | age | days | fee | office | en | fr | ar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Base, run 1 (emulated bf16, 2026-11-21) | **3.8%** (2.6% to 5.4%) | 0.0% | 0.0% | 95.8% | 0.0% | 0.0% | 0.0% | 0.0% | 3.5% | 3.2% | 4.4% | 3.6% |
| Base, run 2 (fp16 compute, 2026-11-22) | **3.6%** (2.5% to 5.2%) | 0.0% | 0.0% | 95.8% | 0.0% | 0.0% | 0.0% | 0.0% | 2.8% | 3.2% | 4.0% | 3.6% |
| SFT | | | | | | | | | | | | |
| SFT + DPO | | | | | | | | | | | | |
| Hosted model (name, date) | | | | | | | | | | | | |

Per-metric rates for the base model (all 744 cases):

| correct | faithful | citations_valid | cites_gold | refusal_ok | clean | lang_ok |
|---|---|---|---|---|---|---|
| 78.2% | 7.1% | 84.0% | 14.4% | 99.9% | 100.0% | 63.2% |

Per-language `lang_ok`: en 97.6%, ar 79.8%, **fr 12.1%**.

### What the two baseline runs show
- Switching from emulated bf16 to fp16 compute did **not** change quality (3.8% vs 3.6%; the intervals overlap almost entirely) and cut median latency from 13.4 s to 3.9 s. So the dtype mattered for speed, not for the answers.
- The final baseline will be re-run once more after the loader change (all weights in the chosen dtype), so base and tuned models are served identically.

### What the baseline shows (read this before training)
- The base model usually produces the **right value** (`correct` 78%) but almost never an answer the grounding metrics accept (`faithful` 7%, `cites_gold` 14%). On these results the gap looks like **citation and quoting behaviour**, not missing knowledge. This is a hypothesis: check examples in `runs/base.html` before relying on it.
- French answers rarely pass the language check (12%), so the model probably answers French questions in another language. Also a hypothesis to check in the report.
- It already **refuses** out-of-scope questions well on the golden set (95.8%) and never obeys the planted injection (`clean` 100%).
- Those are exactly the behaviours the SFT data targets, so a clear improvement is plausible, **but this is a prediction, not a result**.

## Sanad dev world (`sanad dev-eval`, 1,152 cases, same metrics)
Command: `sanad dev-eval --base-url http://127.0.0.1:8000/v1 --model base --concurrency 8 --out runs/base-dev.jsonl`.

| System | Overall | injection | stale | out_of_scope | multihop | age | days | fee | office | en | fr | ar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Base, run 1 (emulated bf16, 2026-11-21) | **11.4%** (9.7% to 13.3%) | 0.0% | 0.0% | 60.9% | 0.0% | 0.0% | 0.0% | 0.0% | 7.3% | 14.8% | 9.4% | 9.9% |
| Base, run 2 (fp16 compute, 2026-11-22) | **11.2%** (9.5% to 13.1%) | 0.0% | 0.0% | 61.5% | 0.0% | 0.0% | 0.0% | 0.0% | 5.7% | 14.8% | 9.1% | 9.6% |
| SFT | | | | | | | | | | | | |
| SFT + DPO | | | | | | | | | | | | |

The base model refuses much less reliably here (60.9%) than on citegate (95.8%): Sanad's out-of-scope questions come with same-region passages about *other* services, which is a harder test of staying grounded to the asked entity.

## Serving (`sanad bench`)
| Setup | p50 | p95 | p99 | completion tok/s | $ per 1k answers (free = $0) | $ per 1k at rented-GPU price | Notes |
|---|---|---|---|---|---|---|---|
| Base, run 1: emulated bf16, 4-bit, concurrency 8, 100 requests | 13.4 s | 41.4 s | 41.4 s | 9.6 | not computed (no price given) | not computed | **Not representative**: pre-fix emulated-bf16 compute on a T4-class GPU. Re-measure after the dtype fix |
| Base, run 2: fp16 compute, 4-bit, concurrency 8, 100 requests | 3.9 s | 15.3 s | 15.3 s | 28.6 | not computed (no price given) | not computed | Same GPU class. 3.4x lower median latency than run 1. Not yet the final setup: weights will be loaded in fp16 for both paths (see CORRECTIONS 2026-11-22) |
| SFT | | | | | | | |
| SFT + DPO | | | | | | | |

## Negative and surprising results
- The first SFT smoke test crashed on this GPU class (bf16 matmul needs compute capability 8.0 or higher). Cause and fix: `docs/CORRECTIONS.md`.
- Baseline serving was very slow (13 s median per answer) for the same reason.
