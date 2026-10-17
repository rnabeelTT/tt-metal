# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2: the functional-decoder test suite (tests/test_functional_decoder.py: real / synthetic weights, lengths
around every boundary, batched users, determinism, host-fallback audit, decode-only) run on tt/fused_decoder.py.

A separate instance of the functional test module is loaded with DECODER_CLS = FusedDecoder (all rewrites on), so the
same cases, references and acceptance bars apply to the fused path."""

import importlib.util
from pathlib import Path

from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder, FusedFlags

_spec = importlib.util.spec_from_file_location(
    "glm53_fused_decoder_suite", Path(__file__).with_name("test_functional_decoder.py")
)
_suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_suite)
_suite.DECODER_CLS = FusedDecoder

pytestmark = _suite.pytestmark
mesh = _suite.mesh
rd = _suite.rd
decode_scale_mean = _suite.decode_scale_mean
suite_swap_cap = _suite.suite_swap_cap
globals().update({k: v for k, v in vars(_suite).items() if k.startswith("test_")})


def test_suite_runs_fused_decoder():
    """The suite builds FusedDecoder with the default FusedFlags: every rewrite on except the rejected mhc_sumsq_op."""
    assert _suite.DECODER_CLS is FusedDecoder
    flags = vars(FusedFlags())
    assert {k for k, v in flags.items() if not v} == {"mhc_sumsq_op"}, flags
