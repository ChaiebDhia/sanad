"""Server, batching and benchmark, with citegate itself as the client."""

import asyncio
import json
import threading
import urllib.error
import urllib.request

import pytest
from citegate.llm import LLMAgent, openai_compatible_completer
from citegate.retrieval import BM25Index
from citegate.runner import run_cases
from citegate.summary import summarise

from sanad.bench import percentile, run_bench
from sanad.client import ChatClient
from sanad.serve import BatchingServer


def test_citegate_evaluates_the_served_model_end_to_end(dev, server):
    """The whole chain: Sanad data -> server -> citegate adapter -> citegate runner -> metrics."""
    srv, backend = server
    sw, world, _ = dev
    agent = LLMAgent(
        BM25Index(sw.world.chunks), openai_compatible_completer("oracle", None, srv.url)
    )
    results = asyncio.run(run_cases(agent, world, concurrency=8))
    s = summarise("oracle", results)
    assert s.groups["ALL"]["passed"].p == 1.0 and s.n == len(world.cases)
    assert max(backend.sizes) > 1  # concurrent requests really were batched


def test_protocol_errors_and_health(server):
    srv, _ = server
    base = srv.url.removesuffix("/v1")
    assert json.load(urllib.request.urlopen(base + "/health"))["status"] == "ok"
    assert json.load(urllib.request.urlopen(srv.url + "/models"))["data"][0]["id"] == "oracle"
    req = urllib.request.Request(srv.url + "/chat/completions", data=b"not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code == 400


def test_backend_failure_becomes_http_500_not_a_hang():
    class Boom:
        def generate(self, batch):
            raise RuntimeError("CUDA out of memory")

    srv = BatchingServer(Boom(), "boom", port=0)
    srv.start()
    try:
        with pytest.raises(RuntimeError):
            ChatClient(srv.url, "boom", retries=1).complete([{"role": "user", "content": "hi"}])
    finally:
        srv.stop()


def test_bench_reports_latency_tokens_and_cost(dev, server):
    srv, _ = server
    prompts = list(dev[2])[:10]
    client = ChatClient(srv.url, "oracle")
    res = run_bench(
        client.complete,
        prompts,
        16,
        4,
        price_per_hour=0.5,
        price_in_per_mtok=1.0,
        price_out_per_mtok=2.0,
    )
    assert res["completion_tokens"] == 16 * 25 and res["prompt_tokens"] == 16 * 120
    assert res["latency_s"]["p50"] <= res["latency_s"]["p99"]
    assert res["cost_per_1k_answers"]["token_priced_usd"] == pytest.approx(
        (120 + 25 * 2) / 1e6 * 1000
    )
    assert res["cost_per_1k_answers"]["self_hosted_usd"] > 0


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 5], 50) == 3 and percentile([], 50) == 0.0


def test_server_handles_parallel_clients(server):
    srv, _ = server
    errs: list[Exception] = []

    def hit():
        try:
            ChatClient(srv.url, "oracle", retries=1).complete([{"role": "user", "content": "x"}])
        except Exception as exc:
            errs.append(exc)

    threads = [threading.Thread(target=hit) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errs
