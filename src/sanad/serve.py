"""OpenAI-compatible inference server with dynamic batching.

Why it exists: citegate (and any other tool) evaluates a model through the OpenAI
``/chat/completions`` protocol. Serving the model this way means the tuned model is tested by
exactly the same code path as a hosted one, and ``sanad bench`` measures the same endpoint.
Concurrent requests are grouped into one GPU batch (up to ``max_batch``, waiting at most
``max_wait_ms``), which is what makes a free T4 usable for a few hundred evaluation cases.

The Hugging Face backend needs torch/transformers and a GPU and is NOT exercised in CI; the
server, batching and protocol are tested with a fake backend.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

Messages = list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Backend(Protocol):
    def generate(self, batch: list[Messages]) -> list[Generation]: ...


class HFBackend:  # pragma: no cover  (needs a GPU; see docs/VERIFICATION.md)
    """Greedy batched generation with an optional LoRA adapter."""

    def __init__(
        self,
        base_model: str,
        adapter: str | None = None,
        load_4bit: bool = True,
        max_new_tokens: int = 200,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tok: Any = AutoTokenizer.from_pretrained(base_model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            kwargs["torch_dtype"] = dtype
        model: Any = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self.model = model

    def generate(self, batch: list[Messages]) -> list[Generation]:
        texts = [
            self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in batch
        ]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        width = enc["input_ids"].shape[1]
        results: list[Generation] = []
        for i, row in enumerate(out):
            new = row[width:]
            n_new = int((new != self.tok.pad_token_id).sum())
            n_prompt = int(enc["attention_mask"][i].sum())
            results.append(
                Generation(self.tok.decode(new, skip_special_tokens=True), n_prompt, n_new)
            )
        return results


@dataclass
class _Job:
    messages: Messages
    future: Future[Generation]


class BatchingServer:
    def __init__(
        self,
        backend: Backend,
        model_name: str = "sanad",
        host: str = "127.0.0.1",
        port: int = 8000,
        max_batch: int = 8,
        max_wait_ms: float = 25.0,
    ) -> None:
        self.backend, self.model_name = backend, model_name
        self.max_batch, self.max_wait = max_batch, max_wait_ms / 1000
        self._queue: queue.Queue[_Job] = queue.Queue()
        self._stop = threading.Event()
        self.batches = 0
        self.requests = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path in ("/health", "/v1/health"):
                    self._send(200, {"status": "ok"})
                elif self.path == "/v1/models":
                    self._send(200, {"data": [{"id": outer.model_name, "object": "model"}]})
                elif self.path == "/stats":
                    mean = outer.requests / outer.batches if outer.batches else 0.0
                    self._send(
                        200,
                        {
                            "requests": outer.requests,
                            "batches": outer.batches,
                            "mean_batch": mean,
                        },
                    )
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    self._send(404, {"error": "not found"})
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    messages = [
                        {"role": str(m["role"]), "content": str(m["content"])}
                        for m in body["messages"]
                    ]
                except (ValueError, KeyError, TypeError):
                    self._send(400, {"error": "expected JSON with a messages list"})
                    return
                fut: Future[Generation] = Future()
                outer._queue.put(_Job(messages, fut))
                try:
                    gen = fut.result(timeout=900)
                except Exception as exc:  # backend failure becomes a 500, never a hang
                    self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                    return
                self._send(
                    200,
                    {
                        "id": "chatcmpl-sanad",
                        "object": "chat.completion",
                        "model": outer.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": gen.text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": gen.prompt_tokens,
                            "completion_tokens": gen.completion_tokens,
                            "total_tokens": gen.prompt_tokens + gen.completion_tokens,
                        },
                    },
                )

            def log_message(self, *args: object) -> None:
                return

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self._threads: list[threading.Thread] = []

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host!s}:{port}/v1"

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            jobs = [first]
            deadline = time.monotonic() + self.max_wait
            while len(jobs) < self.max_batch:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    jobs.append(self._queue.get(timeout=left))
                except queue.Empty:
                    break
            try:
                outs = self.backend.generate([j.messages for j in jobs])
                if len(outs) != len(jobs):
                    raise RuntimeError("backend returned the wrong number of generations")
                for j, o in zip(jobs, outs, strict=True):
                    j.future.set_result(o)
            except Exception as exc:
                for j in jobs:
                    j.future.set_exception(exc)
            self.batches += 1
            self.requests += len(jobs)

    def start(self) -> None:
        for target in (self._worker, self.httpd.serve_forever):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()
