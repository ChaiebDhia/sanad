"""QLoRA SFT on the Sanad training split.

Needs `pip install -e ".[train]"` and a CUDA GPU (a free Kaggle/Colab T4 is enough for a ~1.5B
model). In CI only `--dry-run` runs: it validates the data files
without importing torch. The training path itself has NOT been executed in the repository's
CI; smoke-test it on a few steps before a long run and pin the library versions you used.
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    t, lo = cfg["train"], cfg["lora"]
    from sanad.hw import apply_trl_workarounds, gpu_capability, supports_bf16

    capability = gpu_capability()
    dtype = torch.bfloat16 if supports_bf16(capability) else torch.float16
    print(f"GPU compute capability {capability}: training in {dtype}", flush=True)
    if apply_trl_workarounds(capability):
        print("TRL: using the fp32 matmul fallback for this GPU", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        device_map="auto",
    )
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
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        peft_config=peft_cfg,
        args=SFTConfig(
            output_dir=t["output_dir"],
            num_train_epochs=t["epochs"],
            learning_rate=t["learning_rate"],
            per_device_train_batch_size=t["per_device_batch_size"],
            gradient_accumulation_steps=t["grad_accum"],
            max_length=t["max_length"],
            gradient_checkpointing=True,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            seed=t["seed"],
            report_to=t["report_to"],
            max_steps=args.max_steps,
        ),
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(t["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
