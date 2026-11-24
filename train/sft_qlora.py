"""QLoRA SFT on the Sanad training split.

Needs `pip install -e ".[train]"`. A free Kaggle/Colab T4 is enough for a ~1.5B model. Without a GPU
(for a pipeline smoke test with a tiny model) set `load_4bit = false` in the config: the script then
trains in float32 on the CPU. In CI only `--dry-run` runs (data validation, no torch).
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from sanad.validate import validate_sft


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.toml")
    ap.add_argument("--dry-run", action="store_true", help="validate data, do not train")
    ap.add_argument("--max-steps", type=int, default=-1, help="use e.g. 20 for a smoke test")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))

    n_train = validate_sft(Path(cfg["data"]["train"]))
    n_dev = validate_sft(Path(cfg["data"]["dev"]))
    print(f"data OK: {n_train} train / {n_dev} dev examples")
    if args.dry_run:
        return 0

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

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
    data = load_dataset(
        "json", data_files={"train": cfg["data"]["train"], "dev": cfg["data"]["dev"]}
    )
    dev = data["dev"].select(range(min(len(data["dev"]), t.get("eval_samples", 200))))
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=data["train"],
        eval_dataset=dev,
        peft_config=peft_cfg,
        args=SFTConfig(
            output_dir=t["output_dir"],
            num_train_epochs=t["epochs"],
            learning_rate=t["learning_rate"],
            per_device_train_batch_size=t["per_device_batch_size"],
            per_device_eval_batch_size=t.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=t["grad_accum"],
            max_length=t["max_length"],
            gradient_checkpointing=True,
            bf16=use_bf16,
            fp16=use_fp16,
            logging_steps=10,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=t.get("eval_steps", 200),
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
