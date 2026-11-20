import asyncio
import json
from pathlib import Path

import pytest
from citegate.agents import refusal
from citegate.corpus import World
from citegate.llm import LLMAgent
from citegate.metrics import score_case
from citegate.retrieval import BM25Index

from sanad import data, validate
from sanad.prompts import build_prompt, parse, render
from sanad.world import build_sanad_world


def test_prompt_is_identical_to_what_citegate_sends(train_world):
    """If citegate changes its prompt or retrieval, this fails and training must be rebuilt."""
    index = BM25Index(train_world.world.chunks)
    seen: list[str] = []

    def completer(prompt: str) -> str:
        seen.append(prompt)
        return "REFUSE"

    agent = LLMAgent(index, completer)
    for case in train_world.world.cases[:25]:
        asyncio.run(agent.answer(case.question))
        assert seen[-1] == build_prompt(index, case.question)[0]


def test_parse_matches_citegates_own_parsing(train_world):
    index = BM25Index(train_world.world.chunks)
    for case in train_world.world.cases[:40]:
        gold = train_world.gold[case.id]
        for raw in (render(gold.parts) if gold.parts else "x [a] y", "REFUSE", "plain text"):
            ours = parse(raw, case.question)
            theirs = asyncio.run(LLMAgent(index, lambda _p, r=raw: r).answer(case.question))
            assert ours == theirs
    assert parse("REFUSE", "q") == refusal("q")


def test_every_chosen_passes_and_every_rejected_fails_citegates_metrics(train_world, examples):
    exs, stats = examples
    assert stats.dropped_chosen_fails <= 0.01 * len(train_world.world.cases)
    assert stats.dropped_negative_passes == 0
    chunks = train_world.world.chunk_map()
    cases = {c.id: c for c in train_world.world.cases}
    checked = 0
    for ex in exs:
        if ex.kind == "retrieval_miss":
            continue
        case = cases[ex.case_id]
        assert score_case(case, parse(ex.chosen, case.question), chunks, 0.0).passed
        for _name, raw in ex.rejected:
            assert not score_case(case, parse(raw, case.question), chunks, 0.0).passed
        checked += 1
    assert checked > 500


def test_hard_cases_are_covered(examples):
    exs, stats = examples
    kinds = {e.kind for e in exs}
    assert {"injection", "stale", "out_of_scope", "retrieval_miss", "multihop"} <= kinds
    rows = data.dpo_rows(exs, stats)
    assert {"obey_injection", "stale_answer", "wrong_entity", "hallucinate"} <= set(stats.by_defect)
    assert len(rows) > len(exs)


def test_training_prompts_never_contain_citegates_canary(examples):
    assert all("pwned" not in e.prompt.lower() for e in examples[0])


def test_build_is_deterministic_and_files_validate(tmp_path: Path):
    sw = build_sanad_world(40, data.TRAIN_SEED, tag="t")
    a, _ = data.make_examples(sw)
    b, _ = data.make_examples(build_sanad_world(40, data.TRAIN_SEED, tag="t"))
    assert a == b
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    data.write_jsonl(sft, [data.sft_row(e) for e in a])
    data.write_jsonl(dpo, data.dpo_rows(a))
    assert validate.validate_sft(sft) == len(a)
    assert validate.validate_dpo(dpo) > 0


def test_validators_reject_bad_files(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "hello"}],
                "completion": [{"role": "assistant", "content": "x"}],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="citegate's current prompt"):
        validate.validate_sft(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError):
        validate.validate_sft(empty)


def test_dev_world_is_disjoint_from_training_world(train_world):
    dev = build_sanad_world(30, data.DEV_SEED, tag="d")
    train_q = {c.question for c in train_world.world.cases}
    assert not train_q & {c.question for c in dev.world.cases}
    assert isinstance(dev.world, World)
