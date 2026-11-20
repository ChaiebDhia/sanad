"""Command line, including the real citegate executable."""

import os
import shutil
import subprocess

import pytest

from sanad.cli import main


def test_cli_build_check_show_and_dev_eval(tmp_path, capsys, server):
    out = tmp_path / "d"
    assert main(["data-build", "--out", str(out), "--services", "60", "--dev-services", "20"]) == 0
    assert (out / "build_report.json").exists()
    assert main(["data-check", "--data", str(out), "--services", "60"]) == 0
    assert main(["show", "--data", str(out), "--file", "dpo.train.jsonl"]) == 0
    srv, _ = server
    capsys.readouterr()
    code = main(
        ["dev-eval", "--base-url", srv.url, "--model", "oracle", "--dev-services", "30",
         "--sample", "30", "--out", str(tmp_path / "r.jsonl")]
    )  # fmt: skip
    assert code == 0
    assert "ALL" in capsys.readouterr().out
    assert (tmp_path / "r.jsonl").exists()


def test_the_real_citegate_command_line_can_evaluate_the_server(server, tmp_path):
    """What the Kaggle notebook does: `citegate run --agent llm --base-url <server>`."""
    exe = shutil.which("citegate")
    if exe is None:
        pytest.skip("citegate executable not on PATH")
    srv, _ = server
    out = tmp_path / "run.jsonl"
    proc = subprocess.run(
        [exe, "run", "--agent", "llm", "--provider", "ollama", "--base-url", srv.url,
         "--model", "oracle", "--sample", "15", "--concurrency", "4", "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False,
        env={**os.environ, "PYTHONUTF8": "1"},
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert len(out.read_text(encoding="utf-8").splitlines()) >= 15
    assert "llm:ollama:oracle" in proc.stdout
