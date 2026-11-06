"""Full bf16 state (plan R0): bf16 activations, bf16 KV, default matmul precision.

The a48034a RPA widening patch is withdrawn. The jax 0.11.2 image's upstream RPA v3
kernel (tpu-inference 672546d) is already pristine, so the native patch has no kernel entry.
"""
import json
from pathlib import Path

import pytest

from hetero import native_backend
from hetero.engine_actor import NUMERIC_PROFILES, EngineSettings
from hetero.native_backend import effective_activation_dtype

PATCH = json.loads(Path(native_backend.__file__).with_name("native_backend_patch.json").read_text())



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


def test_native_patch_has_no_widening_and_no_kernel_entry():
    # tpu-inference 672546d ships a pristine RPA v3 kernel; float32 KV needs no kernel change.
    assert [f["name"] for f in PATCH["files"]] == ["cache", "gemma4"]
    for f in PATCH["files"]:
        assert not f["path"].endswith("ragged_paged_attention/v3/kernel.py")
        for r in f["replacements"]:
            assert "k.astype(q.dtype)" not in r["new"]
