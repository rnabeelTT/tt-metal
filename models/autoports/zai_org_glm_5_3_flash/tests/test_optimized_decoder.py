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
