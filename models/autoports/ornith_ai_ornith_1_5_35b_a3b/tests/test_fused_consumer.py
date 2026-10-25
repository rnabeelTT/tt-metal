# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fabric boundary fused with its consumer projection (FabricBoundary(..., consumer=DecodeLinear.consumer(w))) on the
(1, 4) mesh: y = RMSNorm(residual + sum of the devices' partials) @ W per device vs torch, and traced timing of
fused vs boundary + DecodeLinear vs boundary + the shipped ttnn 1D matmul, at Ornith's in-proj / QKV shapes.

    python test_fused_consumer.py
"""

import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear
from models.demos.blackhole.qwen36.tt.fabric_boundary.ops import FabricBoundary

H, EPS, ND = 2048, 1e-6, 4


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, ND), trace_region_size=64 << 20)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    try:
        fb = FabricBoundary(mesh, H, EPS, ring_topology=True)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        g = 1 + 0.1 * torch.randn(H)
        gam = fb.gamma(ttnn.from_torch(g.reshape(1, 1, 64, 32).bfloat16(), dtype=ttnn.bfloat16,
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep))
        e = torch.zeros(1, 1, 32, H)
        e[0, 0, 0] = torch.randn(H)
        emb = ttnn.from_torch(e, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                              memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=rep)
        res, _ = fb(emb, None, gam)
        p = torch.zeros(ND, 1, 32, H)
        p[:, 0, 0] = torch.randn(ND, H) * 0.5
        part = ttnn.from_torch(p.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        r = (e[0, 0, 0].bfloat16().float() + p.bfloat16().float()[:, 0, 0].sum(0)).bfloat16().float()
        xn = (r * torch.rsqrt(r.pow(2).mean() + EPS) * g.bfloat16().float()).bfloat16().float()
        ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                     math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        for name, N, cores1d, gw in (("gdn in-proj", 3136, 44, 11), ("attn qkv", 2560, 64, None)):
            ws = [torch.randn(H, N) * 0.02 for _ in range(ND)]
            dl = DecodeLinear(mesh, H, N, ttnn.bfloat8_b, readers=6, exclude=fb.reserved_cores())
            wd = dl.weight(ws)
            w_il = ttnn.from_torch(torch.stack(ws), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
            wq = ttnn.to_torch(w_il, mesh_composer=comp).float()
            r2, y = fb(res, part, gam, "mixer", consumer=dl.consumer(wd))
            got = ttnn.to_torch(y, mesh_composer=comp).float()
            ps = [pcc(got[d, 0, 0] if got.dim() == 4 else got[d], xn @ wq[d]) for d in range(ND)]
            print(f"{name}: fused y pcc per device {[round(v, 6) for v in ps]}", flush=True)
            assert min(ps) > 0.9995, ps
            for t in (r2, y):
                ttnn.deallocate(t)
            pc = tpc.create_matmul_1d_decode_progcfg(32, H, N, cores1d, **({"grid_w": gw} if gw else {}))
            sites = ["mixer", "mlp"]
            variants = {
                "fused boundary + stream": lambda i: fb(res, part, gam, sites[i % 2], consumer=dl.consumer(wd)),
                "boundary, stream": lambda i: (lambda rx: (rx[0], dl(rx[1], wd, 1), rx[1]))(fb(res, part, gam, sites[i % 2])),
                "boundary, ttnn 1D": lambda i: (lambda rx: (rx[0], ttnn.linear(rx[1], w_il, compute_kernel_config=ckc,
                                                                                 program_config=pc,
                                                                                 memory_config=ttnn.L1_MEMORY_CONFIG), rx[1]))(
                    fb(res, part, gam, sites[i % 2])),
            }
            line = f"{name}:"
            for label, fn in list(variants.items()) * 2:
                def run():
                    outs = [fn(i) for i in range(10)]
                    for o in outs:
                        for t in o:
                            ttnn.deallocate(t)
                run()
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                run()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                for _ in range(20):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                ttnn.release_trace(mesh, tid)
                line += f" | {label} {(time.perf_counter() - t0) / 200 * 1e6:5.1f} us"
            print(line, flush=True)
            for t in (wd, w_il):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
