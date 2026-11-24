import sys
import types

from sanad import hw


def test_native_bf16_needs_ampere_or_newer():
    assert not hw.supports_bf16((7, 5))  # T4
    assert not hw.supports_bf16((6, 0))  # P100
    assert hw.supports_bf16((8, 0)) and hw.supports_bf16((8, 9)) and hw.supports_bf16((9, 0))
    assert not hw.supports_bf16(None)  # no GPU


def _fake_trl(monkeypatch, flag):
    for name in ("trl", "trl.kernels"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    mod = types.ModuleType("trl.kernels.chunked_logprob")
    if flag is not None:
        mod._MM_OUT_DTYPE = flag
    monkeypatch.setitem(sys.modules, "trl.kernels.chunked_logprob", mod)
    return mod


def test_workaround_switches_trl_to_the_fp32_path_on_a_t4(monkeypatch):
    mod = _fake_trl(monkeypatch, True)
    assert hw.apply_trl_workarounds((7, 5)) is True
    assert mod._MM_OUT_DTYPE is False


def test_workaround_leaves_modern_gpus_alone(monkeypatch):
    mod = _fake_trl(monkeypatch, True)
    assert hw.apply_trl_workarounds((8, 0)) is False
    assert mod._MM_OUT_DTYPE is True


def test_workaround_is_harmless_on_other_trl_versions(monkeypatch):
    _fake_trl(monkeypatch, None)  # module without the flag
    assert hw.apply_trl_workarounds((7, 5)) is False
    monkeypatch.delitem(sys.modules, "trl.kernels.chunked_logprob")
    assert hw.apply_trl_workarounds((7, 5)) is False  # module missing entirely


def test_dtype_keyword_follows_the_transformers_version():
    assert hw.dtype_kwarg_name("4.55.2") == "torch_dtype"
    assert hw.dtype_kwarg_name("4.56.0") == "dtype"
    assert hw.dtype_kwarg_name("5.0.1") == "dtype"
    assert hw.dtype_kwarg_name("4.9.0") == "torch_dtype"  # not a string comparison


class _Tensor:
    def __init__(self, dtype, n=10):
        self.dtype, self.n = dtype, n

    def to(self, dtype):
        return _Tensor(dtype, self.n)


class _Param:
    def __init__(self, dtype, trainable, n=10):
        self.data, self.requires_grad, self._n = _Tensor(dtype, n), trainable, n

    @property
    def dtype(self):
        return self.data.dtype

    def numel(self):
        return self._n


class _Model:
    def __init__(self, params):
        self._p = params

    def parameters(self):
        return self._p


def test_trainable_parameters_are_forced_to_fp32_and_frozen_ones_are_left_alone():
    torch = types.SimpleNamespace(float32="fp32")
    frozen = _Param("bf16", trainable=False)
    lora_a, lora_b = _Param("bf16", True, 4), _Param("fp32", True, 6)
    counts = hw.cast_trainable_to_fp32(_Model([frozen, lora_a, lora_b]), torch)
    assert counts == {"fp32": 10}
    assert frozen.dtype == "bf16"  # base weights untouched
    assert lora_a.dtype == "fp32" and lora_b.dtype == "fp32"


def test_training_precision_by_hardware():
    assert hw.training_precision(None) == ("float32", False, False)  # CPU
    assert hw.training_precision((7, 5)) == ("float16", False, True)  # T4
    assert hw.training_precision((8, 0)) == ("bfloat16", True, False)  # A100


def test_precision_override_and_validation():
    assert hw.training_precision((7, 5), "fp32") == ("float32", False, False)
    assert hw.training_precision((7, 5), "bf16") == ("bfloat16", True, False)
    assert hw.training_precision((8, 0), "fp16") == ("float16", False, True)
    assert hw.training_precision(None, "fp16") == ("float32", False, False)  # no GPU: always fp32
    try:
        hw.training_precision((7, 5), "half")
    except ValueError as exc:
        assert "unknown precision" in str(exc)
    else:
        raise AssertionError("an unknown precision must be rejected")
