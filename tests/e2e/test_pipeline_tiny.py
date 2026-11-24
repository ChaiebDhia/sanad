"""End-to-end pipeline test on a tiny random model (CPU, offline).

Runs the real training, resume, merge and DPO scripts, then serves the tuned model with the real
HFBackend and evaluates it with citegate's agent and runner. The model cannot answer anything; the
point is that every library call in the pipeline still works with the installed versions of
transformers, TRL and PEFT. CI runs it weekly and on pull requests that touch the pipeline, so a
breaking upstream release is noticed before a long GPU run is wasted.

Run with: pytest -m e2e
"""

import asyncio
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
for _mod in ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "tokenizers"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(f"{_mod} is not installed", allow_module_level=True)


def run(*args: str) -> str:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1200,
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false"},
    )
    assert proc.returncode == 0, f"{' '.join(args)}\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}"
    return proc.stdout


def write_config(
    path: Path, model: str, base: str, data: dict[str, str], out: Path, kind: str
) -> None:
    train = (
        f'output_dir = "{out.as_posix()}"\nepochs = 1\nlearning_rate = 1e-3\n'
        "per_device_batch_size = 1\ngrad_accum = 2\nmax_length = 1024\nseed = 1\n"
        'precision = "auto"\nreport_to = "none"\nsave_steps = 2\n'
    )
    if kind == "sft":
        train += "eval_steps = 3\neval_samples = 4\n"
    else:
        train += "beta = 0.1\n"
    data_lines = "\n".join(f'{k} = "{v}"' for k, v in data.items())
    path.write_text(
        f'[model]\nbase_model = "{base}"\nload_4bit = false\n\n[data]\n{data_lines}\n\n'
        '[lora]\nr = 4\nalpha = 8\ndropout = 0.0\ntarget_modules = "all-linear"\n\n'
        f"[train]\n{train}",
        encoding="utf-8",
    )


def test_train_resume_merge_dpo_serve_and_evaluate_a_tiny_model(tmp_path: Path) -> None:
    from citegate.llm import LLMAgent, openai_compatible_completer
    from citegate.retrieval import BM25Index
    from citegate.runner import run_cases

    from sanad import data as sdata
    from sanad.serve import BatchingServer, HFBackend
    from sanad.world import build_sanad_world

    d = tmp_path / "data"
    run("-m", "sanad.cli", "data-build", "--out", str(d), "--services", "24", "--dev-services", "8")
    tiny = tmp_path / "tiny"
    run("tools/make_tiny_model.py", "--texts", str(d / "sft.train.jsonl"), "--out", str(tiny))

    sft_cfg, dpo_cfg = tmp_path / "sft.toml", tmp_path / "dpo.toml"
    sft_out, merged, dpo_out = tmp_path / "sft", tmp_path / "merged", tmp_path / "dpo"
    write_config(
        sft_cfg, "sft", tiny.as_posix(),
        {"train": (d / "sft.train.jsonl").as_posix(), "dev": (d / "sft.dev.jsonl").as_posix()},
        sft_out, "sft",
    )  # fmt: skip
    run("train/sft_qlora.py", "--config", str(sft_cfg), "--max-steps", "3")
    assert (sft_out / "adapter_config.json").exists()
    run("train/sft_qlora.py", "--config", str(sft_cfg), "--max-steps", "5", "--resume")

    run("train/merge_adapter.py", "--base", str(tiny), "--adapter", str(sft_out), "--out", str(merged),
        "--dtype", "float32")  # fmt: skip
    assert (merged / "config.json").exists()

    write_config(
        dpo_cfg,
        "dpo",
        merged.as_posix(),
        {"train": (d / "dpo.train.jsonl").as_posix()},
        dpo_out,
        "dpo",
    )
    run("train/dpo.py", "--config", str(dpo_cfg), "--max-steps", "2")
    assert (dpo_out / "adapter_config.json").exists()

    backend = HFBackend(str(merged), str(dpo_out), load_4bit=False, max_new_tokens=12)
    server = BatchingServer(backend, "tiny", port=0, max_batch=4, max_wait_ms=30)
    server.start()
    try:
        dev = build_sanad_world(8, sdata.DEV_SEED, tag="d")
        agent = LLMAgent(
            BM25Index(dev.world.chunks), openai_compatible_completer("tiny", None, server.url)
        )
        results = asyncio.run(run_cases(agent, dev.world.sample(12), concurrency=4))
        assert len(results) == 12
        assert all(getattr(r, "error", None) is None for r in results)
        assert server.requests >= 12
    finally:
        server.stop()
