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
