"""Build prompts and completions in citegate's exact contract.

The prompt is produced by citegate's own template and retriever, so training sees what the model
will see at test time. If citegate changes its prompt, the equivalence test in this repo fails.
"""

from __future__ import annotations

import re

from citegate.agents import AgentAnswer, refusal
from citegate.llm import PROMPT
from citegate.retrieval import BM25Index, Chunk

K = 4  # citegate's LLMAgent default
_CITATION = re.compile(r"\[([^\]\s]+)\]")


def retrieve(index: BM25Index, question: str, k: int = K) -> list[Chunk]:
    return [chunk for chunk, _ in index.search(question, k)]


def build_prompt(index: BM25Index, question: str, k: int = K) -> tuple[str, list[Chunk]]:
    """The user message citegate's LLMAgent would send, and the chunks it contains."""
    hits = retrieve(index, question, k)
    context = "\n".join(f"[{c.id}] {c.text}" for c in hits)
    return PROMPT.format(context=context, question=question), hits


def render(parts: tuple[tuple[str, str], ...]) -> str:
    """Correct completion: each quoted sentence followed by the id of the chunk it came from."""
    return " ".join(f"{sentence} [{cid}]" for sentence, cid in parts)


def parse(raw: str, question: str) -> AgentAnswer:
    """Same parsing as citegate.llm.LLMAgent (equivalence is tested)."""
    if raw.strip().upper().startswith("REFUSE"):
        return refusal(question)
    cites = tuple(dict.fromkeys(_CITATION.findall(raw)))
    return AgentAnswer(_CITATION.sub("", raw).strip(), cites)
