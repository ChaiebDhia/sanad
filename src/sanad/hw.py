"""GPU capability helpers. No torch import at module level, so the core stays dependency-free.

Why this exists: ``torch.cuda.is_bf16_supported()`` returns True on a T4 (compute capability 7.5)
because bfloat16 is *emulated* there. Training then hits kernels that need native bf16
(compute capability >= 8.0) and crashes, and inference is much slower. Native support is what
matters, so we decide from the capability itself.
"""

from __future__ import annotations

import re
import sys
from typing import Any


def supports_bf16(capability: tuple[int, int] | None) -> bool:
    """Native bfloat16 matrix multiplication exists from compute capability 8.0 (Ampere)."""
    return capability is not None and capability[0] >= 8


def gpu_capability() -> tuple[int, int] | None:  # pragma: no cover  (needs a GPU)
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return int(major), int(minor)


def pick_dtype(torch: Any) -> Any:  # pragma: no cover  (needs a GPU)
    return getattr(torch, training_precision(gpu_capability())[0])


def apply_trl_workarounds(capability: tuple[int, int] | None) -> bool:
    """On GPUs without native bf16, make TRL's chunked log-prob kernel use its plain fp32 path.

    Newer TRL versions call ``torch.addmm(..., out_dtype=float32)``, which can fail on older GPUs.
    TRL keeps a fallback (``acc.addmm_(a.float(), b.float())``) behind the module flag
    ``_MM_OUT_DTYPE``; setting it to False selects that path. Returns True if the flag was changed.
    Harmless if the module or flag does not exist (other TRL versions).
    """
    if supports_bf16(capability):
        return False
    module: Any = sys.modules.get("trl.kernels.chunked_logprob")
    if module is None:
        try:
            import importlib

            module = importlib.import_module("trl.kernels.chunked_logprob")
        except Exception:  # noqa: BLE001 - optional, version-dependent workaround
            return False
    if getattr(module, "_MM_OUT_DTYPE", None) is True:
        module._MM_OUT_DTYPE = False
        return True
    return False


def dtype_kwarg_name(transformers_version: str) -> str:
    """transformers renamed ``torch_dtype`` to ``dtype`` in 4.56; use whichever this version expects."""
    nums = [int(x) for x in re.findall(r"\d+", transformers_version)[:2]]
    major, minor = (nums + [0, 0])[:2]
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def model_dtype_kwargs(dtype: Any) -> dict[str, Any]:  # pragma: no cover  (needs transformers)
    """Keyword arguments that load every non-quantised weight in ``dtype`` (not the checkpoint's)."""
    from importlib.metadata import version

    return {dtype_kwarg_name(version("transformers")): dtype}


def cast_trainable_to_fp32(model: Any, torch: Any) -> dict[str, int]:
    """Make every trainable parameter float32 and return {dtype name: parameter count}.

    Mixed-precision training with fp16 needs fp32 trainable weights: the gradient scaler cannot
    unscale bf16 or fp16 gradients. LoRA adapters inherit the base model's dtype in some versions,
    so this is enforced explicitly instead of relying on library defaults.
    """
    counts: dict[str, int] = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
        n = int(p.numel()) if hasattr(p, "numel") else 1
        counts[str(p.dtype)] = counts.get(str(p.dtype), 0) + n
    return counts


def training_precision(
    capability: tuple[int, int] | None, override: str = "auto"
) -> tuple[str, bool, bool]:
    """(dtype name, use bf16 flag, use fp16 flag).

    ``auto``: bf16 on Ampere or newer, fp16 on older GPUs, fp32 on CPU. ``fp32`` forces float32
    (slower, no loss scaling) as the fallback if fp16 training produces NaN losses.
    """
    if override == "fp32" or capability is None:
        return "float32", False, False
    if override not in ("auto", "fp16", "bf16"):
        raise ValueError(f"unknown precision {override!r} (auto | fp16 | bf16 | fp32)")
    if override == "bf16" or (override == "auto" and supports_bf16(capability)):
        return "bfloat16", True, False
    return "float16", False, True


def nan_guard() -> Any:
    """A Trainer callback that stops with a clear error as soon as the logged loss is not finite."""
    import math

    from transformers import TrainerCallback

    class NanGuard(TrainerCallback):  # type: ignore[misc]
        def on_log(
            self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
        ) -> None:
            loss = (logs or {}).get("loss")
            if loss is not None and not math.isfinite(float(loss)):
                raise RuntimeError(
                    f"training loss is {loss} at step {state.global_step}. With fp16 this usually "
                    'means numerical overflow: set precision = "fp32" in the config and retry.'
                )

    return NanGuard()
