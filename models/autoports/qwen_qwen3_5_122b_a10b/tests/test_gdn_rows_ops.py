# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN decode ops, ROWS mode (DFlash verify: R consecutive rows of ONE user in one call, per-row state snapshots) against
the same ops run R times with one row each (state / conv register updated in place): outputs and every snapshot must
match. 122B per-die shapes (Nk 4, Nv 16, Dk = Dv = 128) on the (1, 4) mesh, random weights.

    pytest models/autoports/qwen_qwen3_5_122b_a10b/tests/test_gdn_rows_ops.py
"""
import torch

try:
    import pytest
except ImportError:  # the serving image runs this file as a script
    pytest = None

import ttnn
from models.demos.blackhole.qwen36.tt.gdn_decode.ops import GdnDecode

L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_rows_match_sequential(mesh, R):
    torch.manual_seed(R)
    Nk, Nv, D = 4, 16, 128
    qkv = 2 * Nk * D + Nv * D
    P = qkv + Nv * D + 64
    rep = dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ops = GdnDecode(mesh, Nk, Nv, D, D, P, D**-0.5, max_batch=32)

    def dev(t, dtype=ttnn.bfloat16, mc=DR):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc, **rep)

    taps = [dev(torch.randn(1, qkv) * 0.5) for _ in range(4)]
    neg_a = dev(-torch.rand(1, Nv) * 2.0)
    dt = dev(torch.randn(1, Nv) * 0.5)
    nw = dev(1.0 + 0.1 * torch.randn(1, D))
    taps_t, params_t = ops.prepare(taps, neg_a, dt, nw)
    proj_h = torch.randn(1, R, P) * 0.5
    carry_h = [torch.randn(1, 1, qkv) * 0.5 for _ in range(4)]
    state_h = torch.randn(1, Nv, D, D) * 0.05

    # sequential reference: R one-row calls, state / register in place
    st = [dev(c) for c in carry_h]
    state = dev(state_h)
    y_ref, s_ref = [], []
    for r in range(R):
        proj = dev(proj_h[:, r : r + 1], mc=L1)
        y = ops(proj, st, taps_t, params_t, state, 1)
        y_ref.append(ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float().reshape(-1, Nv * D)[0])
        s_ref.append(ttnn.to_torch(ttnn.get_device_tensors(state)[0]).float().reshape(Nv, D, D))
        ttnn.deallocate(y)
        ttnn.deallocate(proj)
    # rows mode
    st2 = [dev(c) for c in carry_h]
    state2 = dev(state_h)
    snap = dev(torch.zeros(max(R, 1), Nv, D, D))
    proj = dev(proj_h, mc=L1)
    y = ops(proj, st2, taps_t, params_t, state2, R, snap=snap)
    y_rows = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float().reshape(-1, Nv * D)[:R]
    snaps = ttnn.to_torch(ttnn.get_device_tensors(snap)[0]).float().reshape(-1, Nv, D, D)
    state_after = ttnn.to_torch(ttnn.get_device_tensors(state2)[0]).float().reshape(Nv, D, D)
    carry_after = [ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float() for t in st2]
    print(f"shapes y_rows {tuple(y_rows.shape)} y_ref {tuple(y_ref[0].shape)} snaps {tuple(snaps.shape)}")
    for r in range(R):
        py = _pcc(y_rows[r], y_ref[r])
        ps = _pcc(snaps[r], s_ref[r])
        print(f"R {R} row {r}: y pcc {py:.6f} max err {(y_rows[r] - y_ref[r]).abs().max():.3e} | state pcc {ps:.6f}")
        assert py > 0.9999 and ps > 0.9999, (r, py, ps)
    # rows mode leaves the bound state and the conv register untouched (the commit sets them)
    assert torch.equal(state_after, ttnn.to_torch(ttnn.get_device_tensors(dev(state_h))[0]).float().reshape(Nv, D, D))
    for j in range(1, 4):
        assert torch.equal(carry_after[j], ttnn.to_torch(ttnn.get_device_tensors(dev(carry_h[j]))[0]).float())


if pytest is not None:
    mesh = pytest.fixture(scope="module")(_mesh)
    test_rows_match_sequential = pytest.mark.parametrize("R", [1, 4, 16])(test_rows_match_sequential)


if __name__ == "__main__":  # the serving image has no pytest
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    _m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576)
    for _R in (1, 4, 16):
        test_rows_match_sequential(_m, _R)
        print(f"GDN_ROWS_OK R={_R}", flush=True)
    ttnn.close_mesh_device(_m)
