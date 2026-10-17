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
# bfp4 routed experts: the MoE layers are held to the bfp4 quantization floor, measured on the routed output itself
# (tools/moe_scale_probe.py --policy opt, layer 4: the bfp4-rounded torch reference vs the checkpoint weights PCC 0.9826,
# rms 1.0275; the port vs that reference PCC 0.99997, rms 0.995). Layer 0 (dense MLP) keeps the functional bars.
_opt = CANDIDATES[os.environ["GLM53_OPT"]] if os.environ.get("GLM53_OPT") else SELECTED
if _opt.moe.routed_gate_up_dtype == ttnn.bfloat4_b:
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
