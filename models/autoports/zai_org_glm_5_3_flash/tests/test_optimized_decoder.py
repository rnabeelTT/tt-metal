# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 3: the functional-decoder test suite (tests/test_functional_decoder.py: real / synthetic weights, lengths
around every boundary, batched users, determinism, host-fallback audit, decode-only) run on tt/optimized_decoder.py.

A separate instance of the functional test module is loaded with DECODER_CLS = OptimizedDecoder (SELECTED policy), so the
same cases, references and acceptance bars apply to the optimized path. GLM53_OPT=<candidate> runs the suite on one
policy of tt/optimized_decoder.CANDIDATES instead of SELECTED (candidate sweep)."""

import importlib.util
import os
from pathlib import Path

import pytest

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import CANDIDATES, SELECTED, OptimizedDecoder

_spec = importlib.util.spec_from_file_location(
    "glm53_optimized_decoder_suite", Path(__file__).with_name("test_functional_decoder.py")
)
_suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_suite)


def _decoder_cls():
    name = os.environ.get("GLM53_OPT")
    if not name:
        return OptimizedDecoder
    opt = CANDIDATES[name]

    class CandidateDecoder(OptimizedDecoder):
        @classmethod
        def from_state_dict(cls, *args, **kw):
            kw.setdefault("opt", opt)
            return super().from_state_dict(*args, **kw)

    return CandidateDecoder


_suite.DECODER_CLS = _decoder_cls()
# Reference: the quantization-aware HF layer (review #1 P1-2). The MoE layers' reference rounds the weights that the
# policy stores below bf16 (routed experts bfp4, shared expert bfp8: OptConfig.quant_spec) exactly as the port's host
# packing does (tests/hf_ref/hf_quant_ref.py, bfp.py; tools/bfp_check.py: bit-exact vs ttnn), and the functional bars
# apply unchanged (no BAR_OVERRIDES). Each MoE check also prints its numbers against the checkpoint reference (the bfp4
# quantization delta, summarized at module teardown).
# GLM53_REF_QUANT=0: the checkpoint reference with the earlier bfp4 bar shift instead (every PCC bar of the MoE layers
# down by 0.015, sequence scale 1 +- 0.03, token scale 1 +- 0.06; kept for comparison only).
_opt = CANDIDATES[os.environ["GLM53_OPT"]] if os.environ.get("GLM53_OPT") else SELECTED
if os.environ.get("GLM53_REF_QUANT", "1") != "0":
    _suite.QUANT_REF.update(_opt.quant_spec())
    print(f"optimized suite: quantization-aware reference {_suite.QUANT_REF}")
elif _opt.moe.routed_gate_up_dtype == ttnn.bfloat4_b:
    _suite.BAR_OVERRIDES.update(
        {layer: {"delta": 0.98, "seq": 0.03, "token": 0.06} for k, layer in _suite.KINDS.items() if k.endswith("moe")}
    )

pytestmark = _suite.pytestmark
mesh = _suite.mesh
rd = _suite.rd
decode_scale_mean = _suite.decode_scale_mean
suite_swap_cap = _suite.suite_swap_cap
globals().update({k: v for k, v in vars(_suite).items() if k.startswith("test_")})


def test_suite_runs_optimized_decoder():
    assert issubclass(_suite.DECODER_CLS, OptimizedDecoder)


def test_policy_reached_weights(mesh, rd):
    """Hard check of the policy (review #1, the OptConfig field-order bug): the built kda_moe / dsa_moe layers hold the
    policy's weight dtypes and routed / shared compute fidelity (a policy silently running bfp8 experts fails here)."""
    for kind in ("kda_moe", "dsa_moe"):
        dec = _suite.get_layer(mesh, rd, _suite.KINDS[kind])
        moe, m = dec.mlp, _opt.moe
        assert moe.up_gate.dtype == m.routed_gate_up_dtype, (moe.up_gate.dtype, m.routed_gate_up_dtype)
        assert moe.down.dtype == m.routed_down_dtype, (moe.down.dtype, m.routed_down_dtype)
        assert moe.prec.routed_fidelity == m.routed_fidelity and moe.prec.shared_fidelity == m.shared_fidelity
        assert moe.shared_w2.dtype == m.shared_dtype
        if kind == "kda_moe":
            w = dec.attn.weights
            assert w.input_projection.dtype == _opt.kda_in_dtype and w.output_projection.dtype == _opt.kda_out_dtype
        else:
            assert dec.attn.w_uk.dtype == _opt.dsa_bmm_dtype and dec.attn.bmm_reuse == _opt.dsa_bmm_reuse
        print(f"{kind}: policy {_opt.describe()} reached the weights")
    # review #2 hard check: the single-die program options are active (one die)
    if _opt.single_die_mc and tuple(mesh.shape)[1] == 1:
        for kind in ("kda_dense", "kda_moe", "dsa_moe"):
            dec = _suite.get_layer(mesh, rd, _suite.KINDS[kind])
            assert dec._norm_sharded, kind
            if dec.attn_kind == "dsa":
                assert type(dec.attn).__name__ == "MultichipDSA" and dec.attn.mc.dsa_pack, kind
                if _opt.prefill_cfg:
                    assert dec.attn.flash_prefill_pc is not None and dec.attn.prefill_o_2d, kind
                    assert dec.attn.bmm_prefill_pm == 0, "per-head prefill reuse config must stay off"
            if _opt.single_die_r2:
                import os as _os

                assert _os.environ.get("MHC_SINKHORN_PERM_FIDELITY") == "3"
                if dec.attn_kind == "kda":
                    assert type(dec.attn).__name__ == "SingleDieKDA" and dec.attn.mc.kda_conv_op, kind
                if dec.mlp_kind == "dense":
                    assert type(dec.mlp).__name__ == "MultichipMLP" and dec.mlp.mc.mlp_dn, kind
            if _opt.single_die_r3:
                assert os.environ.get("MHC_SINKHORN_FACES") == "1" and dec.mc.sinkhorn_faces, kind
            if _opt.single_die_specs:
                assert dec.mc.kda_out and dec.mc.shared_gu and dec.mc.shared_dn, kind
            print(f"{kind}: single-die options active ({type(dec.attn).__name__}, {type(dec.mlp).__name__})")


@pytest.mark.parametrize("kind", list(_suite.KINDS))
def test_single_slot_decode(mesh, rd, kind):
    """Review #3: a decoder built for ONE slot (the 1-user serving profile; the suite's other layers have 8 or 32 slots).
    One-slot-only paths (the KDA conv op, the 1-slot DSA programs) run here: the test_real_weights sequence (cached HF
    reference) in slot 0, prefill then eager + traced decode, against HF on the quantization-aware reference."""
    from models.autoports.zai_org_glm_5_3_flash.tests.harness import DecodeRunner, build_layer, prefill_user

    layer = _suite.KINDS[kind]
    for k in list(_suite._LAYERS):  # one layer resident (DRAM)
        del _suite._LAYERS[k]
    dec = build_layer(mesh, rd, layer, "real", batch=1, decoder_cls=_suite.DECODER_CLS)
    ref = _suite.reference(layer, "real", 1, 100, 3)
    x, y = ref
    out = prefill_user(dec, mesh, x, 100, 0)
    _suite.check(f"L{layer} 1-slot prefill 100", out, x[:100], y[:100])
    runner = DecodeRunner(dec, mesh, 1)
    for i in range(3):
        pos = [100 + i]
        if i == 0:
            o = runner.eager_step({0: x[100]}, pos)[0]
            runner.capture({0: x[101]}, [101])
        else:
            o = runner.traced_step({0: x[100 + i]}, pos)[0]
        _suite.check(
            f"L{layer} 1-slot decode pos {100 + i}",
            o,
            x[100 + i],
            y[100 + i],
            margin=ref.row_margin(100 + i),
            y_alt=ref.alt(100 + i),
        )
    runner.release()
    del dec
