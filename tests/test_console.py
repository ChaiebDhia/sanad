"""Printing Arabic training text must not crash a legacy (cp1252) console or pipe."""

import io
import sys

from sanad.cli import main


def test_show_survives_a_cp1252_stdout(tmp_path, monkeypatch):
    out = tmp_path / "d"
    assert main(["data-build", "--out", str(out), "--services", "30", "--dev-services", "10"]) == 0
    legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", legacy)
    arabic_rows = [
        i
        for i, line in enumerate((out / "sft.train.jsonl").read_text(encoding="utf-8").splitlines())
        if "\u0627" in line or "ا" in line
    ]
    assert arabic_rows
    assert main(["show", "--data", str(out), "--index", str(arabic_rows[0])]) == 0
