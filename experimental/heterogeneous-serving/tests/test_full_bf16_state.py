"""Full bf16 state (plan R0): bf16 activations, bf16 KV, default matmul precision.

The a48034a RPA widening patch is withdrawn: the native patch now restores the
pristine RPA v3 kernel on any host that still carries the widening.
"""
import hashlib
import json
from pathlib import Path

import pytest

from hetero import native_backend
from hetero.engine_actor import NUMERIC_PROFILES, EngineSettings
from hetero.native_backend import effective_activation_dtype

PATCH = json.loads(Path(native_backend.__file__).with_name("native_backend_patch.json").read_text())
PRISTINE_KERNEL_SHA = "bcd21fdaff6b333903a09991d0ad57f3683fef18bf8abef449c0f10d6879609c"
WIDENED_KERNEL_SHA = "b97aa3be868596ca813bd6cbb7423aa77f5ca923f3533c72b98984b7866265f9"  # a48034a
LOCAL_KERNEL = (Path.home() / "heterogeneous-serving-tpu/tpu-inference/tpu_inference/kernels"
                / "ragged_paged_attention/v3/kernel.py")


def test_default_settings_are_full_bf16_state():
    settings = EngineSettings()
    assert (settings.dtype, settings.activation_dtype, settings.kv_cache_dtype,
            settings.matmul_precision) == ("bfloat16", None, "bfloat16", "default")
    assert NUMERIC_PROFILES[(None, "bfloat16", "default")] == "bf16"
    assert EngineSettings.from_dict(settings.to_dict()) == settings


def test_fp32_state_profile_stays_selectable():
    settings = EngineSettings.from_dict(
        {"activation_dtype": "float32", "kv_cache_dtype": "float32", "matmul_precision": "highest"})
    assert NUMERIC_PROFILES[(settings.activation_dtype, settings.kv_cache_dtype,
                             settings.matmul_precision)] == "fp32"


@pytest.mark.parametrize("overrides", [
    {"activation_dtype": "float32"},                                  # fp32 q x bf16 KV: withdrawn
    {"activation_dtype": "float32", "matmul_precision": "highest"},   # the Q1/Q2 profile
    {"kv_cache_dtype": "float32"},                                    # bf16 q x fp32 KV
    {"matmul_precision": "highest"},
    {"activation_dtype": "float32", "kv_cache_dtype": "float32"},     # fp32 state needs highest
])
def test_mixed_numeric_profiles_are_refused(overrides):
    with pytest.raises(ValueError, match="numeric profile"):
        EngineSettings.from_dict(overrides)


@pytest.mark.parametrize("field,value", [("activation_dtype", "bfloat16"),
                                         ("matmul_precision", "high"),
                                         ("kv_cache_dtype", "auto")])
def test_unknown_values_are_refused(field, value):
    with pytest.raises(ValueError, match=f"unsupported {field}"):
        EngineSettings.from_dict({field: value})


@pytest.mark.parametrize("dtype", ["auto", "float32"])
def test_unset_activation_requires_explicit_16bit_model_dtype(dtype):
    with pytest.raises(ValueError, match="16-bit model dtype"):
        EngineSettings.from_dict({"dtype": dtype})


def test_effective_activation_dtype_follows_model_dtype_when_unset():
    assert effective_activation_dtype(None, "bfloat16") == "bfloat16"
    assert effective_activation_dtype(None, "half") == "float16"
    assert effective_activation_dtype("float32", "bfloat16") == "float32"


def test_native_patch_has_no_widening_and_requires_pristine_kernel():
    kernel = [f for f in PATCH["files"] if f["path"].endswith("ragged_paged_attention/v3/kernel.py")]
    assert len(kernel) == 1
    entry = kernel[0]
    assert entry["patched_sha256"] == PRISTINE_KERNEL_SHA
    assert entry["original_sha256"] == WIDENED_KERNEL_SHA
    for f in PATCH["files"]:
        for r in f["replacements"]:
            assert "k.astype(q.dtype)" not in r["new"]
    assert "k.astype(q.dtype)" in entry["replacements"][0]["old"]


@pytest.mark.skipif(not LOCAL_KERNEL.is_file()
                    or hashlib.sha256(LOCAL_KERNEL.read_bytes()).hexdigest() != PRISTINE_KERNEL_SHA,
                    reason="pristine tpu-inference RPA v3 kernel not available locally")
@pytest.mark.parametrize("state", ["widened", "pristine"])
def test_preparation_restores_pristine_kernel(tmp_path, monkeypatch, state):
    entry = next(f for f in PATCH["files"] if f["name"] == "rpa_v3_kernel_pristine")
    source = LOCAL_KERNEL.read_text()
    replacement = entry["replacements"][0]
    if state == "widened":
        source = source.replace(replacement["new"], replacement["old"], 1)
        assert hashlib.sha256(source.encode()).hexdigest() == WIDENED_KERNEL_SHA
    target = tmp_path / Path(entry["path"]).relative_to("/")
    target.parent.mkdir(parents=True)
    target.write_text(source)
    monkeypatch.setitem(native_backend._PATCH, "files", [entry])
    before = native_backend._inspect(tmp_path)
    assert before["files"][0]["state"] == ("original" if state == "widened" else "patched")
    evidence = native_backend._apply_local(tmp_path)
    assert evidence["ready"] is True
    assert hashlib.sha256(target.read_bytes()).hexdigest() == PRISTINE_KERNEL_SHA
