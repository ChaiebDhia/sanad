"""Leakage checks between Sanad's training worlds and citegate's test worlds.

Zero exact overlap is required for: entity names, injection canaries, questions, passages,
sentences. An informational 4-gram overlap number is reported so template similarity is visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from citegate.corpus import CANARY, World, build_world
from citegate.text import normalize, tokens

from .world import SanadWorld

_SENT = re.compile(r"(?<=[.!?؟])\s+")
_NAME = re.compile(r"\(([^)]+)\)")


def _sentences(text: str) -> set[str]:
    return {normalize(s).strip() for s in _SENT.split(text) if s.strip()}


def _names(world: World) -> set[str]:
    out: set[str] = set()
    for c in world.chunks:
        out.update(normalize(m) for m in _NAME.findall(c.service))
    return out


def _grams(texts: list[str], n: int = 4) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    for t in texts:
        toks = tokens(t)
        out.update(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    return out


@dataclass
class LeakageReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    ngram_overlap: float = 0.0


def reference_worlds() -> list[World]:
    """citegate's default test world plus a larger one (same seed)."""
    return [build_world(48, 7), build_world(240, 7)]


def check(sw: SanadWorld, references: list[World] | None = None) -> LeakageReport:
    refs = references if references is not None else reference_worlds()
    problems: list[str] = []
    train = sw.world
    t_names = _names(train)
    t_questions = {normalize(c.question) for c in train.cases}
    t_chunks = {normalize(c.text) for c in train.chunks}
    t_sent = set().union(*(_sentences(c.text) for c in train.chunks))
    t_canary = {f for c in train.cases for f in c.forbidden if not f.isdigit()}
    train_grams = _grams([c.text for c in train.chunks])
    shared_grams = total_grams = 0
    for ref in refs:
        if x := t_names & _names(ref):
            problems.append(f"{len(x)} entity names shared with citegate, e.g. {sorted(x)[:3]}")
        if x := t_questions & {normalize(c.question) for c in ref.cases}:
            problems.append(f"{len(x)} questions identical to citegate questions")
        if x := t_chunks & {normalize(c.text) for c in ref.chunks}:
            problems.append(f"{len(x)} passages identical to citegate passages")
        if x := t_sent & set().union(*(_sentences(c.text) for c in ref.chunks)):
            problems.append(f"{len(x)} sentences identical to citegate sentences")
        ref_forbidden = {f for c in ref.cases for f in c.forbidden if not f.isdigit()}
        if x := t_canary & (ref_forbidden | {normalize(CANARY)}):
            problems.append(f"canary tokens shared with citegate: {sorted(x)}")
        ref_grams = _grams([c.text for c in ref.chunks])
        shared_grams = max(shared_grams, len(train_grams & ref_grams))
        total_grams = max(total_grams, len(train_grams))
    overlap = shared_grams / total_grams if total_grams else 0.0
    return LeakageReport(not problems, sorted(set(problems)), overlap)
