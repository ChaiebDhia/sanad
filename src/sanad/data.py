"""Turn a Sanad world into SFT and DPO training files.

Each example uses citegate's real prompt and retriever. Every chosen answer must pass citegate's
metrics and every rejected answer must fail them; anything else is dropped and counted, so the
training signal is verified by the same code that will later grade the model.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from citegate.corpus import Case
from citegate.metrics import score_case
from citegate.retrieval import BM25Index

from . import styles as st
from .prompts import K, build_prompt, parse, render
from .world import Gold, SanadWorld

TRAIN_SEED, DEV_SEED = 11, 9_001
_NUM = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class Example:
    case_id: str
    kind: str
    lang: str
    prompt: str
    chosen: str
    rejected: tuple[tuple[str, str], ...]  # (defect name, completion) pairs, all verified to fail


@dataclass(slots=True)
class BuildStats:
    examples: int = 0
    retrieval_miss_kept: int = 0
    retrieval_miss_skipped: int = 0
    dropped_chosen_fails: int = 0
    dropped_negative_passes: int = 0
    by_kind: Counter[str] = field(default_factory=Counter)
    by_defect: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, object]:
        return {
            "examples": self.examples,
            "retrieval_miss_kept": self.retrieval_miss_kept,
            "retrieval_miss_skipped": self.retrieval_miss_skipped,
            "dropped_chosen_fails": self.dropped_chosen_fails,
            "dropped_negative_passes": self.dropped_negative_passes,
            "by_kind": dict(self.by_kind),
            "by_defect": dict(self.by_defect),
        }


def _wrong_value(case: Case, parts: tuple[tuple[str, str], ...]) -> str:
    """The correct answer with its first fact replaced by an unsupported one."""
    sentence, cid = parts[0]
    value = case.values[0]
    if value.isdigit():
        sentence = _NUM.sub(
            lambda m: str(int(m.group()) + 13) if m.group() == value else m.group(),
            sentence,
            count=1,
        )
    else:
        sentence = re.sub(re.escape(value), "Xanadu", sentence, count=1, flags=re.IGNORECASE)
    return render(((sentence, cid), *parts[1:]))


def _negatives(case: Case, gold: Gold, hit_ids: set[str], first_hit: str) -> dict[str, str]:
    if case.must_refuse:
        fake = st.FABRICATION[case.lang].format(name=gold.name)
        return {"hallucinate": f"{fake} [{first_hit}]"}
    out = {
        "over_refuse": "REFUSE",
        "forge_citation": " ".join(f"{s} [ghost-001]" for s, _ in gold.parts),
        "no_citation": " ".join(s for s, _ in gold.parts),
        "wrong_value": _wrong_value(case, gold.parts),
    }
    if gold.drift:
        out["language_drift"] = render(gold.drift)
    if gold.stale and gold.stale[1] in hit_ids:
        out["stale_answer"] = f"{gold.stale[0]} [{gold.stale[1]}]"
    if gold.canary:
        out["obey_injection"] = gold.canary
    return out


MISS_KEEP_PERCENT = 35


def _keep_miss(case_id: str) -> bool:
    return int(hashlib.sha1(case_id.encode()).hexdigest(), 16) % 100 < MISS_KEEP_PERCENT


def _wrong_entity(hits_text: str, first_hit: str) -> str:
    """A plausible failure: answering from a passage about a different entity."""
    body = hits_text.split(". ", 1)[-1]
    return f"{body} [{first_hit}]"


def make_examples(sw: SanadWorld, k: int = K) -> tuple[list[Example], BuildStats]:
    index = BM25Index(sw.world.chunks)
    chunks = sw.world.chunk_map()
    stats = BuildStats()
    out: list[Example] = []
    for case in sw.world.cases:
        gold = sw.gold[case.id]
        prompt, hits = build_prompt(index, case.question, k)
        hit_ids = {h.id for h in hits}
        scored_case = case
        negatives: dict[str, str] | None = None
        if case.must_refuse:
            chosen = "REFUSE"
        elif not all(cid in hit_ids for _, cid in gold.parts):
            # The retriever missed the answer. The faithful behaviour is to refuse; keep a share
            # of these as hard negatives ("same topic, wrong place") and score them as refusals.
            if not _keep_miss(case.id):
                stats.retrieval_miss_skipped += 1
                continue
            stats.retrieval_miss_kept += 1
            chosen = "REFUSE"
            scored_case = replace(
                case,
                kind="retrieval_miss",
                must_refuse=True,
                values=(),
                gold_chunks=(),
                forbidden=(),
            )
            negatives = {"wrong_entity": _wrong_entity(hits[0].text, hits[0].id)} if hits else {}
        else:
            chosen = render(gold.parts)
        if not score_case(scored_case, parse(chosen, case.question), chunks, 0.0).passed:
            stats.dropped_chosen_fails += 1
            continue
        valid: list[tuple[str, str]] = []
        if negatives is None:
            negatives = _negatives(case, gold, hit_ids, hits[0].id if hits else "none")
        for name, raw in sorted(negatives.items()):
            if score_case(scored_case, parse(raw, case.question), chunks, 0.0).passed:
                stats.dropped_negative_passes += 1
            else:
                valid.append((name, raw))
        out.append(Example(case.id, scored_case.kind, case.lang, prompt, chosen, tuple(valid)))
        stats.examples += 1
        stats.by_kind[scored_case.kind] += 1
    return out, stats


def sft_row(ex: Example) -> dict[str, object]:
    return {
        "prompt": [{"role": "user", "content": ex.prompt}],
        "completion": [{"role": "assistant", "content": ex.chosen}],
    }


SPECIAL = ("obey_injection", "stale_answer", "wrong_entity", "hallucinate")


def dpo_rows(
    examples: Iterable[Example], stats: BuildStats | None = None
) -> list[dict[str, object]]:
    """Preference pairs. Targeted defects (obeyed injection, stale answer, wrong entity,
    hallucination) always get a pair; plus one generic defect per example, chosen
    deterministically, so common failures are covered without drowning the rare ones."""
    rows: list[dict[str, object]] = []
    for ex in examples:
        if not ex.rejected:
            continue
        special = [r for r in ex.rejected if r[0] in SPECIAL]
        generic = [r for r in ex.rejected if r[0] not in SPECIAL]
        picks = list(special)
        if generic:
            digest = int(hashlib.sha1(ex.case_id.encode()).hexdigest(), 16)
            picks.append(generic[random.Random(digest).randrange(len(generic))])
        for name, raw in picks:
            if stats is not None:
                stats.by_defect[name] += 1
            rows.append(
                {
                    "prompt": [{"role": "user", "content": ex.prompt}],
                    "chosen": [{"role": "assistant", "content": ex.chosen}],
                    "rejected": [{"role": "assistant", "content": raw}],
                }
            )
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
