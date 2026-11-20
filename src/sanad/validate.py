"""Format validation for training files. Runs without torch, so CI can check it."""

from __future__ import annotations

import json
from pathlib import Path

from citegate.llm import PROMPT


def _msgs(value: object, where: str, role: str) -> str:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        msg: dict[str, object] = value[0]
        content = msg.get("content")
        if msg.get("role") == role and isinstance(content, str):
            return content
    raise ValueError(f"{where}: expected exactly one {role} message")


def _prompt_ok(text: str, where: str) -> None:
    head = PROMPT.split("{context}")[0]
    if not text.startswith(head):
        raise ValueError(f"{where}: prompt does not start with citegate's current prompt")


def validate_sft(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            row = json.loads(line)
            _prompt_ok(_msgs(row.get("prompt"), f"{path}:{i} prompt", "user"), f"{path}:{i}")
            if not _msgs(row.get("completion"), f"{path}:{i} completion", "assistant").strip():
                raise ValueError(f"{path}:{i}: empty completion")
            n += 1
    if n == 0:
        raise ValueError(f"{path}: empty file")
    return n


def validate_dpo(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            row = json.loads(line)
            _prompt_ok(_msgs(row.get("prompt"), f"{path}:{i} prompt", "user"), f"{path}:{i}")
            chosen = _msgs(row.get("chosen"), f"{path}:{i} chosen", "assistant")
            rejected = _msgs(row.get("rejected"), f"{path}:{i} rejected", "assistant")
            if chosen == rejected:
                raise ValueError(f"{path}:{i}: chosen equals rejected")
            n += 1
    if n == 0:
        raise ValueError(f"{path}: empty file")
    return n
