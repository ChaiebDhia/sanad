"""Minimal OpenAI-compatible chat client (stdlib only).

Works with vLLM, llama.cpp's server, Ollama (/v1), LM Studio and hosted APIs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 120.0,
        retries: int = 3,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                dt = time.perf_counter() - t0
                usage = data.get("usage") or {}
                return Completion(
                    text=data["choices"][0]["message"]["content"] or "",
                    latency_s=dt,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
            except (
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                last = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"request failed after {self.retries} attempts: {last}")
