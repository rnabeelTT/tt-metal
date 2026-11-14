# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: tt/sconv.host_reference (the oracle the device tests use) equals transformers' InklingShortConvolution,
whole-sequence and chunked with the carried state."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_P = Path(__file__).resolve().parents[1] / "tt" / "sconv.py"


def _load_host_reference():
    # tt/sconv.py imports ttnn at module level; the host test only needs host_reference, so stub ttnn if absent
    if "ttnn" not in sys.modules:
        try:
            import ttnn  # noqa: F401
        except ImportError:
            sys.modules["ttnn"] = type(sys)("ttnn")
    spec = importlib.util.spec_from_file_location("inkling_sconv", _P)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.host_reference


def test_matches_hf():
    M = pytest.importorskip("transformers.models.inkling.modeling_inkling")
    host_reference = _load_host_reference()
    torch.manual_seed(0)
    C, T = 64, 37
    mod = M.InklingShortConvolution(C, 4, layer_idx=0, conv_idx=0)
    torch.nn.init.normal_(mod.conv1d.weight)
    x = torch.randn(1, T, C)
    with torch.no_grad():
        want = mod(x)[0]
    w = mod.conv1d.weight.data[:, 0, :]  # [C, 4]
    got, _ = host_reference(x[0], w)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    # chunked with carried state == whole
    y1, s = host_reference(x[0, :10], w)
    y2, s = host_reference(x[0, 10:11], w, s)
    y3, _ = host_reference(x[0, 11:], w, s)
    torch.testing.assert_close(torch.cat([y1, y2, y3]), want, rtol=1e-5, atol=1e-5)
