import time

import pytest

from sanad import data
from sanad.world import SanadWorld, build_sanad_world


@pytest.fixture(scope="session")
def train_world() -> SanadWorld:
    return build_sanad_world(120, data.TRAIN_SEED, tag="t")


@pytest.fixture(scope="session")
def examples(train_world):
    return data.make_examples(train_world)


class OracleBackend:
    """Returns the verified correct completion for known prompts; records batch sizes."""

    def __init__(self, answers: dict[str, str], delay: float = 0.0) -> None:
        self.answers, self.delay = answers, delay
        self.sizes: list[int] = []

    def generate(self, batch):
        from sanad.serve import Generation

        self.sizes.append(len(batch))
        time.sleep(self.delay)
        return [Generation(self.answers.get(m[0]["content"], "REFUSE"), 120, 25) for m in batch]


@pytest.fixture(scope="module")
def dev():
    from citegate.corpus import World

    sw = build_sanad_world(30, data.DEV_SEED, tag="d")
    exs, _ = data.make_examples(sw)
    keep = {e.case_id for e in exs if e.kind != "retrieval_miss"}
    cases = tuple(c for c in sw.world.cases if c.id in keep)
    return sw, World(sw.world.chunks, cases), {e.prompt: e.chosen for e in exs}


@pytest.fixture
def server(dev):
    from sanad.serve import BatchingServer

    backend = OracleBackend(dev[2], delay=0.02)
    srv = BatchingServer(backend, "oracle", port=0, max_batch=8, max_wait_ms=30)
    srv.start()
    yield srv, backend
    srv.stop()
