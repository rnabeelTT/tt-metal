# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/idx_pool_exact (HY4_IDX_POOL_K=1) vs the ttnn pooling chain of Hy4Indexer._pool_fused (HY4_IDX_POOL_K=0):
bitwise, per stage (pooled, keys, roped output), n = 32 / 64 / 256 rows, synthetic vg (wide gate range)."""

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.idx_pool_exact import IdxPoolExactOp
from models.autoports.tencent_hy4_flash.tt.rope import DeviceRope

EPS = 1e-6


def chain(vg, bias, dkn, rope, gcos, gsin, n, d, r, mem):
    vg = ttnn.add(ttnn.reshape(vg, (1, r, n, 2 * d)), bias, memory_config=mem)
    v = ttnn.slice(vg, (0, 0, 0, 0), (1, r, n, d), memory_config=mem)
    g = ttnn.slice(vg, (0, 0, 0, d), (1, r, n, 2 * d), memory_config=mem)
    m = ttnn.max(g, dim=1, keepdim=True, memory_config=mem)
    e = ttnn.exp(ttnn.subtract(g, m, memory_config=mem), memory_config=mem)
    num = ttnn.sum(ttnn.multiply(e, v, memory_config=mem), dim=1, keepdim=True, memory_config=mem)
    den = ttnn.sum(e, dim=1, keepdim=True, memory_config=mem)
    pooled = ttnn.div(num, den, memory_config=mem)
    nrm = ttnn.rms_norm(pooled, epsilon=EPS, compute_kernel_config=rope.ckc, memory_config=mem)
    keys = ttnn.typecast(ttnn.multiply(nrm, dkn, memory_config=mem), ttnn.bfloat16, memory_config=mem)
    kp = ttnn.slice(keys, (0, 0, 0, 0), (1, 1, n, d - 64), memory_config=mem)
    kr = ttnn.slice(keys, (0, 0, 0, d - 64), (1, 1, n, d), memory_config=mem)
    out = ttnn.concat([kp, rope(kr, gcos, gsin, mem)], dim=-1, memory_config=mem)
    return pooled, keys, out


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    bad = 0
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        up = lambda t, dt=ttnn.float32: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)  # noqa
        T = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa
        op = IdxPoolExactOp.get(mesh)
        rope = DeviceRope(mesh, 64)
        mem = ttnn.L1_MEMORY_CONFIG
        d, r = 128, 4
        for case, n in enumerate((32, 64, 256, 32)):
            torch.manual_seed(case)
            vg = torch.randn(1, 1, r * n, 2 * d) * (1 + 3 * case)
            ape = torch.randn(r, d)
            bias = torch.zeros(r, n, 2 * d)
            bias[:, :, d:] = ape[:, None, :]
            kn = 1 + 0.1 * torch.randn(d)
            ang = torch.rand(n, 64) * 6.28
            vgt, bt, knt = up(vg), up(bias.reshape(1, r, n, 2 * d)), up(kn.reshape(1, 1, 1, d))
            gcos, gsin = up(torch.cos(ang).reshape(1, 1, n, 64)), up(torch.sin(ang).reshape(1, 1, n, 64))
            p0, k0, o0 = chain(vgt, bt, knt, rope, gcos, gsin, n, d, r, mem)
            p1 = op.pool(vgt, bt, n, d, mem)
            nrm = ttnn.rms_norm(p1, epsilon=EPS, compute_kernel_config=rope.ckc, memory_config=mem)
            k1 = op.post(nrm, knt, mem)
            o1 = rope.tail(k1, gcos, gsin, mem, keep=True)
            res = []
            for name, a, b, i32 in (("pooled", p1, p0, True), ("keys", k1, k0, False), ("out", o1, o0, False)):
                a, b = T(a).reshape(T(b).shape), T(b)
                va, vb = (a.float().view(torch.int32), b.float().view(torch.int32)) if i32 else (a.view(torch.int16), b.view(torch.int16))
                nd = (va != vb).sum().item()
                res.append(f"{name} {nd}/{va.numel()}")
                bad += int(nd > 0)
            print(f"PXT case {case} n={n}: mismatches " + ", ".join(res), flush=True)
        print(f"PXT RESULT {'PASS' if bad == 0 else 'FAIL'}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
