"""Command line: ``sanad <command>``. Evaluation itself is ``citegate`` (see docs/CITEGATE.md)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from citegate.agents import AgentAnswer
from citegate.llm import LLMAgent, openai_compatible_completer
from citegate.retrieval import BM25Index
from citegate.runner import run_cases
from citegate.summary import summarise

from . import data, leakage
from .bench import run_bench
from .client import ChatClient
from .prompts import build_prompt
from .world import SanadWorld, build_sanad_world

_ = AgentAnswer  # re-exported for type checkers of downstream users


def cmd_data_build(args: argparse.Namespace) -> int:
    out = Path(args.out)
    train = build_sanad_world(args.services, data.TRAIN_SEED, tag="t")
    dev = build_sanad_world(
        args.dev_services, data.DEV_SEED, tag="d", forbidden_names=_names(train)
    )
    rep = leakage.check(train)
    rep_dev = leakage.check(dev)
    if not (rep.ok and rep_dev.ok):
        print("LEAKAGE FAILED:", rep.problems + rep_dev.problems)
        return 1
    tr, tr_stats = data.make_examples(train)
    dv, dv_stats = data.make_examples(dev)
    counts = {
        "sft_train": data.write_jsonl(out / "sft.train.jsonl", [data.sft_row(e) for e in tr]),
        "sft_dev": data.write_jsonl(out / "sft.dev.jsonl", [data.sft_row(e) for e in dv]),
        "dpo_train": data.write_jsonl(out / "dpo.train.jsonl", data.dpo_rows(tr, tr_stats)),
    }
    report = {
        "files": counts,
        "train_stats": tr_stats.to_dict(),
        "dev_stats": dv_stats.to_dict(),
        "leakage": {"ok": True, "ngram_overlap_with_citegate": round(rep.ngram_overlap, 5)},
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


def _names(sw: SanadWorld) -> list[str]:
    return [n for c in sw.world.chunks for n in [c.service.split("(")[-1].rstrip(")")]]


def cmd_data_check(args: argparse.Namespace) -> int:
    from .validate import validate_dpo, validate_sft

    d = Path(args.data)
    n1, n2, n3 = (
        validate_sft(d / "sft.train.jsonl"),
        validate_sft(d / "sft.dev.jsonl"),
        validate_dpo(d / "dpo.train.jsonl"),
    )
    rep = leakage.check(build_sanad_world(args.services, data.TRAIN_SEED, tag="t"))
    print(
        f"files OK: {n1} SFT train, {n2} SFT dev, {n3} DPO pairs; leakage: {'OK' if rep.ok else rep.problems}"
    )
    return 0 if rep.ok else 1


def cmd_dev_eval(args: argparse.Namespace) -> int:
    """Score an endpoint on Sanad's own dev world using citegate's agent, runner and metrics."""
    dev = build_sanad_world(args.dev_services, data.DEV_SEED, tag="d")
    world = dev.world.sample(args.sample) if args.sample else dev.world
    agent = LLMAgent(
        BM25Index(dev.world.chunks),
        openai_compatible_completer(args.model, None, args.base_url),
        name=args.model,
    )
    out = Path(args.out) if args.out else None
    results = asyncio.run(run_cases(agent, world, concurrency=args.concurrency, out=out))
    s = summarise(agent.name, results)
    for group, metrics in s.groups.items():
        p = metrics["passed"]
        print(f"{group:22s} pass {p.p:6.1%}  (Wilson 95% {p.lo:.1%} to {p.hi:.1%}, n={p.n})")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    dev = build_sanad_world(args.dev_services, data.DEV_SEED, tag="d")
    index = BM25Index(dev.world.chunks)
    prompts = [build_prompt(index, c.question)[0] for c in dev.world.cases]
    client = ChatClient(args.base_url, args.model, max_tokens=args.max_tokens)
    res = run_bench(
        client.complete,
        prompts,
        args.n,
        args.concurrency,
        args.price_per_hour,
        args.price_in,
        args.price_out,
    )
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover  (needs a GPU)
    from .serve import BatchingServer, HFBackend

    backend = HFBackend(
        args.model, args.adapter, load_4bit=not args.no_4bit, max_new_tokens=args.max_new_tokens
    )
    srv = BatchingServer(backend, args.name or args.model, args.host, args.port, args.max_batch)
    print(f"serving on {srv.url} (model name: {srv.model_name})", flush=True)
    srv.start()
    try:
        srv._threads[0].join()
    except KeyboardInterrupt:
        srv.stop()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    rows = [
        json.loads(x)
        for x in (Path(args.data) / args.file).read_text(encoding="utf-8").splitlines()
    ]
    row = rows[args.index]
    for key in ("prompt", "completion", "chosen", "rejected"):
        if key in row:
            print(f"--- {key} ---\n{row[key][0]['content']}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sanad", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser(
        "data-build", help="generate SFT/DPO files in citegate's prompt format, with leakage check"
    )
    s.add_argument("--out", default="data")
    s.add_argument("--services", type=int, default=400, help="training services (~15 prompts each)")
    s.add_argument("--dev-services", type=int, default=64)
    s.set_defaults(fn=cmd_data_build)

    s = sub.add_parser("data-check", help="validate training files and re-run the leakage check")
    s.add_argument("--data", default="data")
    s.add_argument("--services", type=int, default=400)
    s.set_defaults(fn=cmd_data_check)

    s = sub.add_parser(
        "dev-eval", help="score an endpoint on Sanad's dev world with citegate's metrics"
    )
    s.add_argument("--base-url", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--dev-services", type=int, default=64)
    s.add_argument("--sample", type=int)
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--out", help="resumable JSONL, same format as `citegate run --out`")
    s.set_defaults(fn=cmd_dev_eval)

    s = sub.add_parser("bench", help="latency, throughput and cost per 1,000 answers")
    s.add_argument("--base-url", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--dev-services", type=int, default=64)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--concurrency", type=int, default=8)
    s.add_argument("--max-tokens", type=int, default=200)
    s.add_argument("--price-per-hour", type=float, help="USD/hour of the machine serving the model")
    s.add_argument("--price-in", type=float, help="USD per million input tokens")
    s.add_argument("--price-out", type=float, help="USD per million output tokens")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("serve", help="OpenAI-compatible server with dynamic batching (needs a GPU)")
    s.add_argument("--model", required=True, help="base model id or path")
    s.add_argument("--adapter", help="LoRA adapter folder")
    s.add_argument("--name", help="model name to report (default: --model)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--max-batch", type=int, default=8)
    s.add_argument("--max-new-tokens", type=int, default=200)
    s.add_argument("--no-4bit", action="store_true")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("show", help="print one training example")
    s.add_argument("--data", default="data")
    s.add_argument("--file", default="sft.train.jsonl")
    s.add_argument("--index", type=int, default=0)
    s.set_defaults(fn=cmd_show)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
