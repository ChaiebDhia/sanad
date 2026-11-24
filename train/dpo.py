"""Preference tuning (DPO) on verified failure-mode pairs: forged or missing citations, obeyed
injections, stale answers, hallucination, wrong entity, language drift, over-refusal.

Start from the MERGED SFT model (see merge_adapter.py). Same options as sft_qlora.py:
`load_4bit = false` trains in float32 on the CPU, which is only meant for a tiny-model smoke test.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from sanad.validate import validate_dpo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.toml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))

    n = validate_dpo(Path(cfg["data"]["train"]))
    print(f"data OK: {n} preference pairs")
    if args.dry_run:
        return 0

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from sanad.hw import (
        apply_trl_workarounds,
        cast_trainable_to_fp32,
        gpu_capability,
        model_dtype_kwargs,
        nan_guard,
        training_precision,
    )

    m, t, lo = cfg["model"], cfg["train"], cfg["lora"]
    capability = gpu_capability()
    dtype_name, use_bf16, use_fp16 = training_precision(capability, t.get("precision", "auto"))
    dtype = getattr(torch, dtype_name)
    print(f"GPU compute capability {capability}: training in {dtype}", flush=True)
    if apply_trl_workarounds(capability):
        print("TRL: using the fp32 matmul fallback for this GPU", flush=True)

    tok = AutoTokenizer.from_pretrained(m["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": "auto", **model_dtype_kwargs(dtype)}
    if m.get("load_4bit", True):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(m["base_model"], **kwargs)
    peft_cfg = LoraConfig(
        r=lo["r"],
        lora_alpha=lo["alpha"],
        lora_dropout=lo["dropout"],
        target_modules=lo["target_modules"],
        task_type="CAUSAL_LM",
    )
    data = load_dataset("json", data_files={"train": cfg["data"]["train"]})
    trainer = DPOTrainer(
        model=model,
        processing_class=tok,
        train_dataset=data["train"],
        peft_config=peft_cfg,
        args=DPOConfig(
            output_dir=t["output_dir"],
            num_train_epochs=t["epochs"],
            learning_rate=t["learning_rate"],
            beta=t["beta"],
            per_device_train_batch_size=t["per_device_batch_size"],
            gradient_accumulation_steps=t["grad_accum"],
            max_length=t["max_length"],
            gradient_checkpointing=True,
            bf16=use_bf16,
            fp16=use_fp16,
            logging_steps=10,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=t.get("save_steps", 100),
            save_total_limit=2,
            seed=t["seed"],
            report_to=t["report_to"],
            max_steps=args.max_steps,
        ),
        callbacks=[nan_guard()],
    )
    counts = cast_trainable_to_fp32(trainer.model, torch)
    print(f"trainable parameters by dtype: {counts}", flush=True)
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(t["output_dir"])
    print(f"adapter saved to {t['output_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
