# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace-safe building blocks of the decode step:
- update_padded_kv_cache (tensor form) writing a 32-row slab of a row-major cache at a device-held offset,
  replicated cache on a (1, 4) mesh (cluster_axis 0 has extent 1);
- the one-hot staging update  S' = A @ S + P @ rows  (exact in bf16);
- uint32 group-id -> token-index expansion on the device.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn


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


def up(m, t, dtype, layout):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=m, mesh_mapper=rep(m))


def test_update_padded_rm(mesh):
    W = 608
    cache = up(mesh, torch.zeros(1, 1, 256, W, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    slab = torch.randn(1, 1, 64, W).to(torch.bfloat16)
    sd = up(mesh, slab, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    off = up(mesh, torch.tensor([96], dtype=torch.int32).reshape(1, 1, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    slot = up(mesh, torch.tensor([0], dtype=torch.int32).reshape(1, 1, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(cache, sd, slot, off, 0, 1, 0, input_row_offset=32,
                                                              input_rows=32)
    got = t0(cache)
    for d, dt in enumerate(ttnn.get_device_tensors(cache)):
        g = ttnn.to_torch(dt)
        assert torch.equal(g[0, 0, 96:128], slab[0, 0, 32:64]), d
        assert float(g[0, 0, :96].abs().sum()) == 0 and float(g[0, 0, 128:].abs().sum()) == 0, d
    print("update_padded_kv_cache RM tensor-form: ok on all dies")


def test_onehot_staging(mesh):
    S = torch.randn(1, 1, 64, 608).to(torch.bfloat16)
    rows = torch.randn(1, 1, 32, 608).to(torch.bfloat16)
    A = torch.zeros(64, 64)
    A[:32, 32:] = torch.eye(32)  # shift: next block becomes current
    P = torch.zeros(64, 32)
    P[5, 0] = 1
    P[6, 1] = 1
    keep = torch.ones(64)
    keep[5:7] = 0
    A = A * keep[:, None]
    ref = (A @ S.float()[0, 0] + P @ rows.float()[0, 0]).to(torch.bfloat16)
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    d = lambda t: up(mesh, t.to(torch.bfloat16).reshape(1, 1, *t.shape[-2:]), ttnn.bfloat16, ttnn.TILE_LAYOUT)  # noqa
    out = ttnn.add(ttnn.matmul(d(A), d(S), compute_kernel_config=ckc), ttnn.matmul(d(P), d(rows), compute_kernel_config=ckc))
    got = t0(out)[0, 0]
    assert torch.equal(got, ref), float((got.float() - ref.float()).abs().max())
    print("one-hot staging: exact")


def test_expand_ids(mesh):
    ids = torch.randint(0, 200000, (32, 512), dtype=torch.int64)
    d = up(mesh, ids.to(torch.int32).reshape(1, 1, 32, 512), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    dt = ttnn.to_layout(d, ttnn.TILE_LAYOUT)
    f = ttnn.typecast(dt, ttnn.float32)
    f = ttnn.reshape(ttnn.multiply(f, 4.0), (1, 32, 512, 1))
    offs = up(mesh, torch.arange(4, dtype=torch.float32).reshape(1, 1, 1, 4), ttnn.float32, ttnn.TILE_LAYOUT)
    e = ttnn.reshape(ttnn.add(f, offs), (1, 1, 32, 2048))
    u = ttnn.typecast(e, ttnn.uint32)
    u = ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT)
    got = t0(u).reshape(32, 2048).to(torch.int64) & 0xFFFFFFFF
    ref = (ids[:, :, None] * 4 + torch.arange(4)).reshape(32, 2048)
    assert torch.equal(got, ref), int((got != ref).sum())
    print("uint32 id expansion: exact")
