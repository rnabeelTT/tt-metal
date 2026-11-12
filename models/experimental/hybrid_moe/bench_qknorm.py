# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Test + microbenchmark of the fused full-width QK norm (attn_decode QKStats + one all_gather + AttnDecodePrep
norm="full") against the stock chain (distributed RMSNorm of q and of k: slice / pre_all_gather / all_gather /
post_all_gather, concat, then the RoPE-only prep), at MiniMax-M2.7's per-die shapes on the 4-die mesh (12 q heads, 2
kv heads, head dim 128, 4 rows). Reports PCC / max error of q and k against an FP32 torch reference for both paths, and
the trace-replay time of each. usage (in the image): python bench_qknorm.py"""

import sys
import time

sys.path.insert(0, "/opt/tt-metal")
import torch  # noqa: E402

import ttnn  # noqa: E402
from models.demos.blackhole.qwen36.tt.attn_decode.ops import AttnDecodePrep  # noqa: E402
from models.demos.blackhole.qwen36.tt.attn_decode.qkstats import QKStats  # noqa: E402

ND, NH, NKV, HD, B, RD = 4, 12, 2, 128, 4, 64
EPS = 1e-6


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, ND), l1_small_size=24576, trace_region_size=64 << 20)
    try:
        g = torch.Generator().manual_seed(0)
        qd, kd = NH * HD, NKV * HD
        W = qd + 2 * kd
        qkv_t = torch.randn(ND, 1, B, W, generator=g) * 2.0  # per die [q_d | k_d | v_d]
        qkv_t = qkv_t.bfloat16().float()
        gq = (torch.rand(ND * qd, generator=g) * 2).bfloat16()
        gk = (torch.rand(ND * kd, generator=g) * 2).bfloat16()
        pos = torch.arange(1000, 1000 + B)
        inv = 1.0 / (10000 ** (torch.arange(0, RD, 2).float() / RD))
        ang = pos[:, None].float() * inv[None]
        cos_h, sin_h = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)  # [B, RD]
        # FP32 reference
        q_all = torch.cat([qkv_t[d, 0, :, :qd] for d in range(ND)], -1)
        k_all = torch.cat([qkv_t[d, 0, :, qd : qd + kd] for d in range(ND)], -1)
        qn = q_all * torch.rsqrt(q_all.pow(2).mean(-1, keepdim=True) + EPS) * gq.float()
        kn = k_all * torch.rsqrt(k_all.pow(2).mean(-1, keepdim=True) + EPS) * gk.float()

        def rope(x):  # x [B, heads, HD]
            r, p = x[..., :RD], x[..., RD:]
            r1, r2 = r[..., : RD // 2], r[..., RD // 2 :]
            rot = torch.cat([-r2, r1], -1)
            return torch.cat([r * cos_h[:, None] + rot * sin_h[:, None], p], -1)

        q_ref = rope(qn.reshape(B, ND * NH, HD))
        k_ref = rope(kn.reshape(B, ND * NKV, HD))

        sh0 = ttnn.ShardTensorToMesh(mesh, dim=0)
        qkv = ttnn.from_torch(
            qkv_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=sh0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rep = ttnn.ReplicateTensorToMesh(mesh)
        cs = lambda t: ttnn.from_torch(  # noqa: E731
            torch.cat([t, torch.zeros(B, 2 * 32 - RD)], -1).reshape(1, B, 1, 64),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos, sin = cs(cos_h), cs(sin_h)
        # stock: distributed rmsnorm of q and k (DRAM interleaved form), concat, RoPE-only prep
        gw = lambda gg: ttnn.as_tensor(  # noqa: E731
            gg.reshape(1, 1, -1, 32),
            dtype=ttnn.bfloat16,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 2), mesh_shape=[1, ND]),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gq_t, gk_t = gw(gq), gw(gk)
        prep0 = AttnDecodePrep(mesh, NH, NKV, HD, RD, max_batch=8, norm=False, gate=False)
        const0 = prep0.prepare(torch.zeros(HD), torch.zeros(HD))
        cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )

        def stock():
            parts = []
            for lo, hi, gamma in ((0, qd, gq_t), (qd, qd + kd, gk_t)):
                x = ttnn.slice(qkv, (0, 0, 0, lo), (1, 1, B, hi), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                st = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=cfg, dtype=ttnn.bfloat16)
                ga = ttnn.all_gather(
                    st, dim=-1, num_links=1, topology=ttnn.Topology.Linear, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                parts.append(ttnn.rms_norm_post_all_gather(x, ga, epsilon=EPS, weight=gamma, compute_kernel_config=cfg))
            v = ttnn.slice(qkv, (0, 0, 0, qd + kd), (1, 1, B, W), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.concat([parts[0], parts[1], v], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return prep0(out, cos, sin, const0, B)

        prep2 = AttnDecodePrep(mesh, NH, NKV, HD, RD, max_batch=8, norm="full", gate=False)
        const2 = prep2.prepare_full(gq, gk)
        qs = QKStats(mesh, qd, kd)

        def fused():
            st = qs(qkv)
            ga = ttnn.all_gather(
                st, dim=-1, num_links=1, topology=ttnn.Topology.Linear, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            return prep2(qkv, cos, sin, const2, B, stats=ga, stats_parts=qs.parts)

        def host(t, heads):  # [1, B, heads, HD] per die -> [B, ND * heads, HD]
            x = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            return x.reshape(ND, B, heads, HD).permute(1, 0, 2, 3).reshape(B, ND * heads, HD)

        for name, fn in (("stock", stock), ("fused", fused)):
            q, k, v, _ = fn()
            qh, kh = host(q, NH), host(k, NKV)
            print(
                f"{name}: q pcc {pcc(qh, q_ref):.6f} max {float((qh - q_ref).abs().max()):.4f} | k pcc "
                f"{pcc(kh, k_ref):.6f} max {float((kh - k_ref).abs().max()):.4f}",
                flush=True,
            )
        qa, ka = host(stock()[0], NH), host(stock()[1], NKV)
        qb, kb = host(fused()[0], NH), host(fused()[1], NKV)
        print(
            f"fused vs stock: q max {float((qa - qb).abs().max()):.4f}, k max {float((ka - kb).abs().max()):.4f}",
            flush=True,
        )

        def bench(fn, reps=20, iters=20):
            fn()
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            for _ in range(reps):
                fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            t0 = time.time()
            for _ in range(iters):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            ttnn.release_trace(mesh, tid)
            return (time.time() - t0) / (reps * iters) * 1e6

        print(f"QKNORM stock {bench(stock):.1f} us, fused {bench(fused):.1f} us per layer", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
