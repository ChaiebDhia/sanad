"""GPU capability helpers. No torch import at module level, so the core stays dependency-free.

Why this exists: ``torch.cuda.is_bf16_supported()`` returns True on a T4 (compute capability 7.5)
because bfloat16 is *emulated* there. Training then hits kernels that need native bf16
(compute capability >= 8.0) and crashes, and inference is much slower. Native support is what
matters, so we decide from the capability itself.
"""

from __future__ import annotations

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
    return torch.bfloat16 if supports_bf16(gpu_capability()) else torch.float16


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
