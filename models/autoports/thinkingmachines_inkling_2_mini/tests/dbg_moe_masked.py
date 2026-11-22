# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug: the MoE masked (prefill) path step by step on one 32-row chunk vs torch, layer 0, bf16 experts."""

import math
import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import moe as M
from models.autoports.thinkingmachines_inkling_2_mini.tt.weights import Checkpoint, load_config


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    g = torch.load("/cache/fixtures/golden_t256.pt")
    x = g["internals"]["layers.0.post_attention_layernorm"][0][0][:32].bfloat16()  # [32, H]
    cfg, ck = load_config(), Checkpoint()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    moe = M.MoE(mesh, cfg, ck, 0, expert_dtype=ttnn.bfloat16)
    v = os.environ.get("VAR", "")
    if "nofp32" in v:
        moe.ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                         math_approx_mode=False, fp32_dest_acc_en=False,
                                                         packer_l1_acc=False)
    if "k96" in v:
        moe.cfg_gu_m = M.sparse_matmul_program_config(32, 2 * moe.Il, moe.H, target_cores=32, in0_block_w_cap=96)
        moe.cfg_dn_m = M.sparse_matmul_program_config(32, moe.H, moe.Il, target_cores=32, in0_block_w_cap=96)
    if "sub1" in v:
        import dataclasses
        for n in ("cfg_gu_m", "cfg_dn_m"):
            c = getattr(moe, n)
            setattr(moe, n, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=c.compute_with_storage_grid_size, in0_block_w=c.in0_block_w,
                out_subblock_h=1, out_subblock_w=1, out_block_h=c.out_block_h, out_block_w=c.out_block_w,
                per_core_M=c.per_core_M, per_core_N=c.per_core_N, fuse_batch=False, fused_activation=None,
                mcast_in0=True))
    print("variant", v, moe.cfg_gu_m)
    E, Il, H, C = moe.E, moe.Il, moe.H, 32
    xd = ttnn.from_torch(x.reshape(1, 1, C, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    w, ids = moe.route(xd)
    wt, it = ttnn.to_torch(w).float().reshape(C, -1), ttnn.to_torch(ids).long().reshape(C, -1)
    r = moe._routing_matrix(w, ids, C)
    rt = ttnn.to_torch(r).float().reshape(C, E)
    want_r = torch.zeros(C, E).scatter_(1, it, wt)
    print("routing matrix pcc", pcc(rt, want_r), "nnz", (rt > 0).sum().item(), "want", (want_r > 0).sum().item())
    mask = ttnn.to_layout(ttnn.typecast(ttnn.sum(r, dim=2, keepdim=True), ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
    mt = ttnn.to_torch(mask).float().reshape(E)
    act_e = (want_r.sum(0) > 0).nonzero()[:, 0]
    print("mask active", (mt > 0).sum().item(), "want", len(act_e))
    fused = ttnn.sparse_matmul(xd, moe.up_gate, sparsity=mask, nnz=None, memory_config=M.DRAM, output_tile=moe.tile,
                               program_config=moe.cfg_gu_m, compute_kernel_config=moe.ckc, dtype=ttnn.bfloat16)
    print("fused shape", tuple(fused.shape))
    ft = ttnn.to_torch(fused).float().reshape(E, C, 2 * Il)
    W = ttnn.to_torch(moe.up_gate).float().reshape(E, H, 2 * Il)
    for e in act_e[:4].tolist():
        print(f"  expert {e}: gate|up pcc {pcc(ft[e], x.float() @ W[e]):.5f}")
    fr = ttnn.reshape(fused, (1, E, C, 2 * Il))
    din = M.inkling_act(ttnn.slice(fr, [0, 0, 0, 0], [1, E, C, Il]), ttnn.slice(fr, [0, 0, 0, Il], [1, E, C, 2 * Il]))
    dt = ttnn.to_torch(din).float().reshape(E, C, Il)
    for e in act_e[:2].tolist():
        g_, u_ = ft[e][:, :Il], ft[e][:, Il:]
        ref = 4 * torch.tanh(g_ / 4) * torch.sigmoid(g_) * 25 * torch.tanh(u_ / 25)
        print(f"  expert {e}: act pcc {pcc(dt[e], ref):.5f}")
    down = ttnn.sparse_matmul(din, moe.down, sparsity=mask, nnz=None, is_input_a_sparse=True, is_input_b_sparse=True,
                              memory_config=M.DRAM, output_tile=moe.tile, program_config=moe.cfg_dn_m,
                              compute_kernel_config=moe.ckc, dtype=ttnn.bfloat16)
    print("down shape", tuple(down.shape))
    dn = ttnn.to_torch(down).float().reshape(E, C, H)
    D = ttnn.to_torch(moe.down).float().reshape(E, Il, H)
    for e in act_e[:4].tolist():
        print(f"  expert {e}: down pcc {pcc(dn[e], dt[e] @ D[e]):.5f}")
    inactive = [e for e in range(E) if e not in set(act_e.tolist())][:3]
    print("inactive rows finite:", [bool(torch.isfinite(dn[e]).all()) for e in inactive],
          "absmax", [dn[e].abs().max().item() for e in inactive])
    out = ttnn.to_torch(moe._masked(xd, w, ids, C)).float().reshape(C, H)
    ref = torch.zeros(C, H)
    for t in range(C):
        for j in range(it.shape[1]):
            e = it[t, j].item()
            ref[t] += wt[t, j] * (dt[e][t] @ D[e])
    print("masked out vs recomposed torch", pcc(out, ref))
    ttnn.close_mesh_device(mesh)
    print("== done")


if __name__ == "__main__":
    main()
