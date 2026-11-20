from citegate.corpus import build_world

from sanad import leakage
from sanad.world import SanadWorld, build_sanad_world


def test_training_world_has_no_overlap_with_citegate(train_world):
    rep = leakage.check(train_world)
    assert rep.ok, rep.problems
    assert rep.ngram_overlap < 0.02


def test_check_fires_when_training_data_is_citegates_own_world():
    rep = leakage.check(SanadWorld(build_world(48, 7), {}))
    assert not rep.ok
    kinds = " ".join(rep.problems)
    for needle in (
        "entity names",
        "questions identical",
        "passages identical",
        "sentences identical",
    ):
        assert needle in kinds


def test_check_fires_on_a_shared_canary():
    sw = build_sanad_world(20, 3)
    from dataclasses import replace

    cases = tuple(
        replace(c, forbidden=("pwned",)) if c.kind == "injection" else c for c in sw.world.cases
    )
    leaky = SanadWorld(type(sw.world)(sw.world.chunks, cases), sw.gold)
    assert any("canary" in p for p in leakage.check(leaky).problems)
