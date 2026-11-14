# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the tile decomposition of the relative bias (tt/relbias.py) equals HF's InklingRelativeLogits, for
decode (1 row) and spec-verify (R consecutive rows) at positions around the tile, extent and window boundaries."""

import pytest
import torch

from models.autoports.thinkingmachines_inkling_small.tt import relbias


@pytest.mark.parametrize("E", [512, 1024])
@pytest.mark.parametrize("R", [1, 4, 8])
@pytest.mark.parametrize("pos0", [0, 5, 31, 32, 511, 512, 1023, 1024, 1500, 4097])
def test_tables_match_reference(E, R, pos0):
    torch.manual_seed(E + R + pos0)
    G, d_rel, scale = 4, 16, 1.0 / 128
    proj = torch.randn(d_rel, E)
    r = torch.randn(R, G, d_rel)
    pos = [pos0 + j for j in range(R)]
    kv_len = pos[-1] + 1
    T = relbias.build_tables(proj, scale)
    got = relbias.emulate_bias(r, pos, T, kv_len) * scale
    want = relbias.reference_bias(r, pos, proj, kv_len)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-4)


def test_against_hf_module():
    """Cross-check reference_bias itself against transformers' module (when transformers has inkling)."""
    M = pytest.importorskip("transformers.models.inkling.modeling_inkling")
    torch.manual_seed(0)
    E, d_rel, H = 512, 16, 4
    mod = M.InklingRelativeLogits(d_rel, E)
    torch.nn.init.normal_(mod.proj)
    T_q, kv = 6, 700
    qpos = torch.arange(kv - T_q, kv)
    rs = torch.randn(1, T_q, H, d_rel)
    hf = mod(rs, qpos, torch.arange(kv))[0]  # [H, T_q, kv]
    for t in range(T_q):
        ours = relbias.reference_bias(rs[0, t : t + 1], [int(qpos[t])], mod.proj.data, kv)  # [H, kv]
        torch.testing.assert_close(ours, hf[:, t], rtol=1e-5, atol=1e-5)
