# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark (traced replays): the sparse-regime decode attention as the sibling ports do decode attention
(flash-decode, the key sequence split across cores, ttnn.transformer.flash_multi_latent_attention_decode) over the
selected keys gathered into a contiguous buffer (ttnn.embedding), vs sparse_sdpa (one core per query row, per-key
gathers). Same data, R real rows, 2044 selected keys + the tail (sink, own group); output difference vs sparse_sdpa."""

from __future__ import annotations

import os

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tools.bench_dsh import timed
from models.autoports.tencent_hy4_flash.tt.attention import SENTINEL, SKF


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100_000_000)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
    try:
        torch.manual_seed(0)
        T, W, H, K, N = 131072 + 256, 608, 32, 2176, 2176
        cache_t = (torch.randn(1, 1, T, W) * 0.5).to(torch.bfloat16)
        cache_t[..., 576:] = 0
        sink = T - 1
        cache_t[0, 0, sink] = 0
        cache_t[0, 0, sink, 576] = 16.0  # (the sink row: a key in the sink column)
        ct = ttnn.from_torch(cache_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                     math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        g = torch.Generator().manual_seed(1)
        for R in (1, 5):
            q = (torch.randn(1, H, 32, W, generator=g) * 0.5).to(torch.bfloat16)
            q[:, 8:] = 0  # (8 real heads per die)
            q[..., 576:] = 0
            q[..., 576] = 0.3  # the sink logit column
            idx = torch.full((1, 1, 32, K), SENTINEL, dtype=torch.int64)
            gidx = torch.full((R, N), sink, dtype=torch.int64)
            cur = torch.zeros(R, dtype=torch.int32)
            for r in range(32):
                idx[0, 0, r, 0] = sink
            for r in range(R):
                p = 120000 + r
                sel = torch.sort(torch.randperm(p // 4 - 1, generator=g)[:511]).values
                toks = (sel[:, None] * 4 + torch.arange(4)).flatten()
                own = torch.arange((p // 4) * 4, p + 1)
                lst = torch.cat([toks, torch.tensor([sink]), own])
                idx[0, 0, r, :] = SENTINEL
                idx[0, 0, r, : lst.numel()] = lst
                gidx[r, : lst.numel()] = lst
                cur[r] = lst.numel() - 1
            q_sp = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
            i_sp = ttnn.from_torch(idx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                   mesh_mapper=rep)
            t_sp, o_sp = timed(mesh, lambda: ttnn.transformer.sparse_sdpa(q_sp, ct, i_sp, 512, kv_format=SKF.BF16,
                                                                         scale=1 / 16, k_chunk_size=128), n=100)
            ref = d0(o_sp).float()[0, :, :R]  # [H, R, 512]
            print(f"FD_BENCH R {R}: sparse_sdpa {t_sp:.1f} us", flush=True)
            qf = q[0, :, :R].permute(1, 0, 2).reshape(1, R, H, W).contiguous()
            q_fd = ttnn.from_torch(qf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            gi = ttnn.from_torch(gidx.reshape(1, R * N).to(torch.int32), dtype=ttnn.uint32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
            cp = ttnn.from_torch(cur, dtype=ttnn.int32, device=mesh, mesh_mapper=rep)
            cache2d = ttnn.reshape(ct, (T, W))

            def gather():
                kk = ttnn.embedding(gi, cache2d, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
                return ttnn.reshape(kk, (R, 1, N, W))

            try:
                t_g, kb = timed(mesh, gather, n=100)
                print(f"FD_BENCH R {R}: gather (embedding) {t_g:.1f} us -> {list(kb.shape)}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"FD_BENCH R {R}: gather failed: {str(e)[:300]}", flush=True)
                continue
            for kc in (64, 128, 256):
                for mc in (0, 4, 8, 16):
                    try:
                        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
                                                    q_chunk_size=0, k_chunk_size=kc, exp_approx_mode=False,
                                                    **({"max_cores_per_head_batch": mc} if mc else {}))
                        fn = lambda: ttnn.transformer.flash_multi_latent_attention_decode(  # noqa: E731
                            q_fd, kb, None, 512, is_causal=True, cur_pos_tensor=cp, scale=1 / 16, program_config=pc,
                            compute_kernel_config=ckc)
                        t_fd, o_fd = timed(mesh, fn, n=100)
                        got = d0(o_fd).float()[0, :R, :H].permute(1, 0, 2)  # [H, R, 512]
                        d = (got[:8] - ref[:8]).abs().max().item()
                        rel = d / ref[:8].abs().max().item()
                        print(f"FD_BENCH R {R}: flash MLA decode kc {kc} max_cores {mc}: {t_fd:.1f} us (+ gather "
                              f"{t_g:.1f}); out {list(o_fd.shape)}; max |diff| vs sparse_sdpa {d:.4g} (rel {rel:.3g})",
                              flush=True)
                        ttnn.deallocate(o_fd)
                    except Exception as e:  # noqa: BLE001
                        print(f"FD_BENCH R {R}: flash MLA kc {kc} mc {mc} failed: {str(e)[:300]}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
