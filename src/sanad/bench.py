"""Latency / throughput / cost benchmark against any OpenAI-compatible endpoint."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .client import Completion


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def run_bench(
    complete: Callable[[list[dict[str, str]]], Completion],
    prompts_in: list[str],
    n_requests: int,
    concurrency: int,
    price_per_hour: float | None = None,
    price_in_per_mtok: float | None = None,
    price_out_per_mtok: float | None = None,
    warmup: int = 2,
) -> dict[str, Any]:
    prompts = [
        [{"role": "user", "content": prompts_in[i % len(prompts_in)]}] for i in range(n_requests)
    ]
    for p in prompts[:warmup]:
        complete(p)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        runs = list(ex.map(complete, prompts))
    wall = time.perf_counter() - t0
    lat = [r.latency_s for r in runs]
    p_tok = sum(r.prompt_tokens or 0 for r in runs)
    c_tok = sum(r.completion_tokens or 0 for r in runs)
    out: dict[str, Any] = {
        "n_requests": n_requests,
        "concurrency": concurrency,
        "wall_s": wall,
        "requests_per_s": n_requests / wall if wall else 0.0,
        "completion_tokens_per_s": c_tok / wall if wall else 0.0,
        "latency_s": {f"p{p}": percentile(lat, p) for p in (50, 95, 99)},
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "cost_per_1k_answers": {},
        "caveat": "Self-hosted cost assumes the server is saturated at this concurrency.",
    }
    cost: dict[str, float] = {}
    if price_per_hour is not None:
        cost["self_hosted_usd"] = price_per_hour * (wall / 3600) / n_requests * 1000
    if price_in_per_mtok is not None and price_out_per_mtok is not None:
        api = (p_tok * price_in_per_mtok + c_tok * price_out_per_mtok) / 1e6
        cost["token_priced_usd"] = api / n_requests * 1000
    out["cost_per_1k_answers"] = cost
    return out
