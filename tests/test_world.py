from citegate.corpus import build_world

from sanad import styles as st
from sanad.world import CITEGATE_NAMES, build_sanad_world


def test_world_is_deterministic():
    a, b = build_sanad_world(40, 5), build_sanad_world(40, 5)
    assert a.world == b.world and a.gold == b.gold


def test_all_kinds_and_languages_present(train_world):
    kinds = {c.kind for c in train_world.world.cases}
    assert kinds == {
        "fee",
        "days",
        "age",
        "office",
        "multihop",
        "injection",
        "stale",
        "out_of_scope",
    }
    assert {c.lang for c in train_world.world.cases} == set(st.LANGS)


def test_no_region_or_city_name_is_a_citegate_name(train_world):
    names = {c.service.split("(")[-1].rstrip(")").lower() for c in train_world.world.chunks}
    assert not names & CITEGATE_NAMES


def test_services_are_not_citegate_services(train_world):
    theirs = {c.service.split(" (")[0].lower() for c in build_world(48, 7).chunks}
    ours = {c.service.split(" (")[0].lower() for c in train_world.world.chunks}
    assert not theirs & ours


def test_every_case_has_gold_material(train_world):
    for c in train_world.world.cases:
        g = train_world.gold[c.id]
        assert g.name
        if not c.must_refuse:
            assert g.parts


def test_forbidden_values_are_set_for_traps(train_world):
    for c in train_world.world.cases:
        if c.kind in ("injection", "stale"):
            assert c.forbidden
