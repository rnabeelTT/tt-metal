# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the precision policy mapping (no device): the baseline policy is the shipped runtime, candidate
overrides land on the knobs the runtime consumes, fixed fields raise, and check_effective flags mismatches."""
import pytest

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt import precision as P
from models.autoports.qwen_qwen3_5_122b_a10b.tt.multichip_decoder import MULTICHIP_SELECTED


def test_baseline_policy_is_identity():
    mc = P.apply_policy(MULTICHIP_SELECTED, P.BASELINE_POLICY)
    b0, b1 = MULTICHIP_SELECTED.base, mc.base
    assert (b1.routed_gate_up_dtype, b1.routed_down_dtype, b1.routed_fidelity) == (
        b0.routed_gate_up_dtype,
        b0.routed_down_dtype,
        b0.routed_fidelity,
    )
    assert (b1.shared_dtype, b1.shared_fidelity, b1.router_dtype, b1.router_fidelity) == (
        b0.shared_dtype,
        b0.shared_fidelity,
        b0.router_dtype,
        b0.router_fidelity,
    )
    assert mc.kv_dtype == MULTICHIP_SELECTED.kv_dtype == ttnn.bfloat8_b
    assert mc.gdn_state_dtype == MULTICHIP_SELECTED.gdn_state_dtype
    assert mc.mixer_fidelity == MULTICHIP_SELECTED.mixer_fidelity == "hifi2"
    assert mc.out_proj_decode_dtype is None and mc.precision_layer_exceptions == ()
    assert mc.attn_weight_dtype == mc.gdn_weight_dtype == mc.lm_head_dtype == ttnn.bfloat8_b
    d = P.describe_config(mc)
    assert d["routed_gate_up"] == "bfp4" and d["routed_fidelity"] == "lofi" and d["lm_head"] == "bfp8"
    # every other knob untouched
    for f in ("gdn_gqa_matmul", "moe_gate_fast", "sorted_min_tokens", "mixer_decode_mode", "mesh_shape"):
        assert getattr(mc, f) == getattr(MULTICHIP_SELECTED, f)


def test_candidate_overrides_map_to_knobs():
    pol = P.policy_from_overrides(
        "probe",
        weight_groups={
            "routed_moe": {"gate_up": "bfp4", "down": "bfp8"},
            "shared_moe": {"dtype": "bfp4"},
            "attention": {"dtype": "bfp4"},
            "linear_attention": {"dtype": "bfp4"},
            "lm_head": {"dtype": "bfp4"},
        },
        compute_fidelities={
            "attention": "lofi",
            "linear_attention": "lofi",
            "routed_moe": "hifi2",
            "shared_moe": "lofi",
        },
        ccl_dtype="bfp8",
        kv_cache_dtype="bf16",
        linear_state_dtype="fp32",
        layer_exceptions={"0": {"routed_moe": {"gate_up": "bfp8", "down": "bfp8"}, "routed_moe_fidelity": "hifi2"}},
    )
    mc = P.apply_policy(MULTICHIP_SELECTED, pol)
    assert mc.base.routed_gate_up_dtype == ttnn.bfloat4_b and mc.base.routed_down_dtype == ttnn.bfloat8_b
    assert mc.base.routed_fidelity == ttnn.MathFidelity.HiFi2
    assert mc.base.shared_dtype == ttnn.bfloat4_b and mc.base.shared_fidelity == ttnn.MathFidelity.LoFi
    assert mc.attn_weight_dtype == mc.gdn_weight_dtype == mc.lm_head_dtype == ttnn.bfloat4_b
    assert mc.mixer_fidelity == "lofi" and mc.out_proj_decode_dtype == "bfp8"
    assert mc.kv_dtype == ttnn.bfloat16
    assert mc.gdn_state_dtype == ttnn.float32 and mc.gdn_decode_high_precision is True
    assert mc.precision_layer_exceptions == (
        (
            0,
            {
                "routed_gate_up": ttnn.bfloat8_b,
                "routed_down": ttnn.bfloat8_b,
                "routed_fidelity": ttnn.MathFidelity.HiFi2,
            },
        ),
    )
    # the args hook hands the exceptions to MoEPrecision.from_args per layer
    from models.autoports.qwen_qwen3_5_122b_a10b.tt.moe import MoEPrecision

    class A:
        moe_precision = mc.to_optimization_config().moe_precision()
        precision_layer_exceptions = {int(k): dict(v) for k, v in mc.precision_layer_exceptions}

    assert MoEPrecision.from_args(A, 0).routed_gate_up == ttnn.bfloat8_b
    assert MoEPrecision.from_args(A, 0).routed_fidelity == ttnn.MathFidelity.HiFi2
    assert MoEPrecision.from_args(A, 3).routed_gate_up == ttnn.bfloat4_b
    assert MoEPrecision.from_args(A, 3).shared == ttnn.bfloat4_b


@pytest.mark.parametrize(
    "over",
    [
        dict(residual_dtype="bfp8"),
        dict(logits_dtype="bfp8"),
        dict(weight_groups={"embedding": {"dtype": "bfp8"}}),
        dict(compute_fidelities={"attention": "lofi"}),  # disagrees with linear_attention
        dict(compute_fidelities={"attention": "hifi4", "linear_attention": "hifi4"}),
        dict(ccl_dtype="bfp4"),
        dict(kv_cache_dtype="int8"),
        dict(layer_exceptions={"5": {"attention": {"dtype": "bfp4"}}}),
    ],
)
def test_unconstructible_values_raise(over, expect_error):
    with expect_error(
        ValueError, r"fixed at|must agree|hifi2 \| lofi|bf16 \| bfp8|unsupported dtype|only MoE-block exceptions"
    ):
        P.apply_policy(MULTICHIP_SELECTED, P.policy_from_overrides("bad", **over))


def test_check_effective_flags_mismatch():
    pol = P.policy_from_overrides("x", weight_groups={"routed_moe": {"gate_up": "bfp8", "down": "bfp8"}})
    eff = {
        "lm_head": "bfp8",
        "mixer_fidelity": "hifi2",
        "kv_cache_dtype": "bfp8",
        "linear_state_dtype": "bf16",
        "linear_state_high_precision": False,
        "lm_head_fidelity": "ttnn_default",
        "ccl_dtype": "bf16",
        "layers": {
            0: {
                "linear_attention": "bfp8",
                "linear_attention_out": "bfp8",
                "routed_gate_up": "bfp4",
                "routed_down": "bfp8",
                "routed_fidelity": "lofi",
                "shared": "bfp8",
                "shared_fidelity": "hifi2",
                "router": "bf16",
                "router_fidelity": "hifi4",
            }
        },
    }
    bad = P.check_effective(pol, eff)
    assert bad == ["layer 0 routed_gate_up: built bfp4 != policy bfp8"]
    eff["layers"][0]["routed_gate_up"] = "bfp8"
    assert P.check_effective(pol, eff) == []


def test_shipped_config_loads_by_default(monkeypatch):
    """No env override -> the shipped selected_precision_config.json is what create_model applies (default construction path)."""
    monkeypatch.delenv("QWEN35_PRECISION_CONFIG", raising=False)
    pol, path = P.load_policy()
    assert path == P.SELECTED_PRECISION_CONFIG and path.is_file()
    assert pol["config_id"] == pol["selected_from_config_id"]
    mc = P.apply_policy(MULTICHIP_SELECTED, pol)
    assert P.describe_config(mc)["lm_head"] == pol["weight_groups"]["lm_head"]["dtype"]
    assert P.describe_config(mc)["attention"] == pol["weight_groups"]["attention"]["dtype"]
    # 'none' forces the code defaults (= baseline_s7)
    monkeypatch.setenv("QWEN35_PRECISION_CONFIG", "none")
    pol2, path2 = P.load_policy()
    assert path2 is None and pol2["config_id"] == "baseline_s7"


def test_lm_head_hifi2_default_preset(expect_error):
    """'hifi2_default' spells the resolved ttnn default out (HiFi2, fp32 dest off, packer L1 acc on) and round-trips by name."""
    pol = P.policy_from_overrides("pin", compute_fidelities={"lm_head": "hifi2_default"})
    mc = P.apply_policy(MULTICHIP_SELECTED, pol)
    assert mc.lm_head_fidelity == "hifi2_default"
    assert P.describe_config(mc)["lm_head_fidelity"] == "hifi2_default"
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    assert P._lm_head_fid_name(ckc) == "hifi2_default"
    assert P._lm_head_fid_name(None) == "ttnn_default"
    ckc2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    assert P._lm_head_fid_name(ckc2) == "hifi2"
    with expect_error(ValueError, "compute_fidelities.lm_head"):
        P.apply_policy(MULTICHIP_SELECTED, P.policy_from_overrides("bad", compute_fidelities={"lm_head": "hifi3"}))
