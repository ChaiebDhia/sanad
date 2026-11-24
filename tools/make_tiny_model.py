"""Build a tiny random Qwen2-style model and tokenizer offline (no downloads).

It cannot answer anything, but it exercises every code path of the training, merge, serving and
evaluation pipeline in seconds on a CPU. Used by tests/e2e and by the scheduled pipeline-smoke
workflow, which exists to catch breakage when transformers, TRL or PEFT change.

Usage: python tools/make_tiny_model.py --texts data/sft.train.jsonl --out /tmp/tiny
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path


def _texts(path: Path, limit: int = 400) -> Iterator[str]:
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                return
            row = json.loads(line)
            for key in ("prompt", "completion", "chosen", "rejected"):
                if key in row:
                    yield row[key][0]["content"]


def make_tiny_model(texts_file: Path, out: Path) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700,
        special_tokens=["<pad>", "<eos>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_texts(texts_file), trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    fast.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    config = Qwen2Config(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        pad_token_id=fast.pad_token_id,
        eos_token_id=fast.eos_token_id,
    )
    model = Qwen2ForCausalLM(config)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    fast.save_pretrained(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    print(make_tiny_model(args.texts, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
