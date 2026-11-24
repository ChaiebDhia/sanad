"""Merge a LoRA adapter into its base model so it can be served or converted (GGUF).

Usage:
  python train/merge_adapter.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/sft --out runs/sft-merged
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sanad.hw import model_dtype_kwargs

    model = AutoModelForCausalLM.from_pretrained(
        args.base, **model_dtype_kwargs(getattr(torch, args.dtype))
    )
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"merged model written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
