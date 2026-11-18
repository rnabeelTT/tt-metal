# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device primitives the day-0 port relies on, against torch (no checkpoint needed).

    python -m pytest -q -s models/autoports/tencent_hy4_flash/tests/test_primitives_device.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn

SKF = getattr(ttnn.transformer, "SparseKVFormat", None) or ttnn.SparseKVFormat


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    m = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    yield m
    ttnn.close_mesh_device(m)


def rep(m):
    return ttnn.ReplicateTensorToMesh(m)


def t0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def test_all_reduce(mesh):
    x = torch.randn(4, 1, 32, 3072)
    d = ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    from models.autoports.tencent_hy4_flash.tt.common import AllReduce

    y = AllReduce(mesh)(d)
    ref = x.to(torch.bfloat16).float().sum(0)
    for i, yt in enumerate(ttnn.get_device_tensors(y)):
        got = ttnn.to_torch(yt).float().reshape(32, 3072)
        assert pcc(got, ref) > 0.9999, i


def test_slice_write(mesh):
    cache = ttnn.from_torch(torch.zeros(1, 1, 256, 608, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep(mesh))
    rows = torch.randn(1, 1, 5, 608).to(torch.bfloat16)
    r = ttnn.from_torch(rows, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep(mesh))
    ttnn.experimental.slice_write(r, cache, [0, 0, 37, 0], [1, 1, 42, 608], [1, 1, 1, 1])
    got = t0(cache)
    assert torch.equal(got[0, 0, 37:42], rows[0, 0])
    assert float(got[0, 0, :37].abs().sum()) == 0 and float(got[0, 0, 42:].abs().sum()) == 0


@pytest.mark.parametrize("H,S,K", [(32, 32, 640), (32, 64, 2176)])
def test_sparse_sdpa_sink(mesh, H, S, K):
    """608-wide rows, V = leading 512, scale 1/16, sink row via the extra columns, sentinel tails."""
    torch.manual_seed(0)
    Tk = 3000
    kv = torch.randn(1, 1, Tk + 32, 608) * 0.5
    kv[..., 576:] = 0
    sink_row = Tk
    kv[0, 0, sink_row] = 0
    kv[0, 0, sink_row, 576] = 16.0
    kv[0, 0, sink_row, 577] = 16.0
    kv = kv.to(torch.bfloat16)
    q = (torch.randn(1, H, S, 608) * 0.5)
    sinks = torch.randn(H) * 2
    s_hi = sinks.to(torch.bfloat16).float()
    s_lo = (sinks - s_hi).to(torch.bfloat16).float()
    q[..., 576:] = 0
    q[0, :, :, 576] = s_hi[:, None]
    q[0, :, :, 577] = s_lo[:, None]
    q = q.to(torch.bfloat16)
    idx = np.full((S, K), 0xFFFFFFFF, dtype=np.uint32)
    rng = np.random.default_rng(0)
    n_keys = []
    for s in range(S):
        n = int(rng.integers(1, K - 1))
        keys = rng.choice(Tk, size=n, replace=False)
        idx[s, 0] = sink_row
        idx[s, 1 : 1 + n] = keys
        n_keys.append(keys)
    qd = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep(mesh))
    kd = ttnn.from_torch(kv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep(mesh))
    idd = ttnn.from_torch(torch.from_numpy(idx.view(np.int32)).reshape(1, 1, S, K), dtype=ttnn.uint32,
                          layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep(mesh))
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    o = ttnn.transformer.sparse_sdpa(qd, kd, idd, 512, kv_format=SKF.BF16, scale=1 / 16, k_chunk_size=128,
                                     compute_kernel_config=ckc)
    got = t0(o).float().reshape(H, S, 512)
    # reference: softmax over [sink logit, selected keys]
    qf, kf = q.float()[0], kv.float()[0, 0]
    ref = torch.zeros(H, S, 512)
    for s in range(S):
        keys = torch.from_numpy(n_keys[s].astype(np.int64))
        sc = (qf[:, s, :576] @ kf[keys, :576].T) / 16.0  # [H, n]
        sc = torch.cat([sinks[:, None], sc], dim=1)
        p = sc.softmax(-1)[:, 1:]
        ref[:, s] = p @ kf[keys, :512]
    print(f"sparse_sdpa H={H} S={S} K={K}: pcc {pcc(got, ref):.6f} max {float((got - ref).abs().max()):.4g}")
    assert pcc(got, ref) > 0.999


def test_indexer_topk(mesh):
    torch.manual_seed(0)
    nh, S, G, D = 16, 32, 4096, 128
    q = torch.randn(1, nh, S, D).to(torch.bfloat16)
    k = torch.randn(1, 1, G, D).to(torch.bfloat16)
    w = torch.randn(1, 1, S, nh).to(torch.bfloat16) * 0.05
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep(mesh))  # noqa
    kp = torch.cat([k, torch.zeros(1, 1, 32, D, dtype=k.dtype)], dim=2)  # one spare tile: chunk_start = G (aligned)
    sc = ttnn.experimental.indexer_score_dsa(up(q), up(kp), up(w), chunk_start_idx=G, kv_len=G + 32)
    got = t0(sc).float().reshape(S, -1)[:, :G]
    ref = torch.einsum("hsd,gd->hsg", q.float()[0], k.float()[0, 0]).relu()
    ref = (ref * w.float()[0, 0].T[:, :, None]).sum(0)
    print(f"indexer_score_dsa: pcc {pcc(got, ref):.6f}")
    assert pcc(got, ref) > 0.999
    rm = ttnn.to_layout(ttnn.typecast(sc, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
    ids = ttnn.experimental.topk_large_indices(rm, k=512)
    gi = t0(ids).reshape(S, -1).numpy().astype(np.int64)
    refi = torch.topk(got.to(torch.bfloat16).float(), 512, dim=-1).indices.numpy()
    overlap = np.mean([len(set(gi[s]) & set(refi[s])) / 512 for s in range(S)])
    print(f"topk_large_indices: dtype {ids.dtype} shape {tuple(ids.shape)} overlap with torch.topk {overlap:.4f}")
    assert overlap > 0.99
