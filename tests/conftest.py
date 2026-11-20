import pytest

from sanad.world import SanadWorld, build_sanad_world


@pytest.fixture(scope="session")
def train_world() -> SanadWorld:
    return build_sanad_world(120, 11, tag="t")
