# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2 sparse MoE on TT (transformers `Inkling2SparseMoeBlock`): 128 routed experts, top-8, no shared expert.

    scores  = sigmoid(x @ Wr^T)                    (fp32)
    ids     = topk(scores + bias, 8)               (the bias only selects)
    w       = scores[ids] / sum(scores[ids]) * 8   (route_scale)
    out     = sum_i w_i down_i(act(gate_i x, up_i x)),  act(g, u) = 4 tanh(g/4) * sigmoid(g) * 25 tanh(u/25)

Experts are tensor-parallel on the intermediate dim (each die holds every expert's I/N slice, gate|up packed per die)
and the dies' partial sums are all-reduced. route_scale (8) and the post-MLP conv's 1/sqrt(5) are folded into the down
weights (the conv is linear in its input). Expert matmuls are ttnn.sparse_matmul (the MiniMax / qwen36 forms): decode
rows one token at a time in gather mode (only the 8 selected experts' weights are read), prefill chunks of 32 tokens
over the union of their selected experts.
"""

from __future__ import annotations

import math
import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.ccl import all_reduce
from models.autoports.minimaxai_minimax_m2_7.tt.moe import sparse_matmul_program_config
from models.autoports.thinkingmachines_inkling_2_mini.tt.wload import cached

TILE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG
PF_CHUNK = int(os.environ.get("I2M_MOE_PF_CHUNK", "32"))


def softcap(t, c):
    return ttnn.multiply(ttnn.tanh(ttnn.multiply(t, 1.0 / c)), c)


def inkling_act(g, u):
    """softcap(g, 4) * sigmoid(g) * softcap(u, 25), fp32 (transformers Inkling2SoftcappedSwiGLU)."""
    g = ttnn.typecast(g, ttnn.float32) if g.dtype != ttnn.float32 else g
    u = ttnn.typecast(u, ttnn.float32) if u.dtype != ttnn.float32 else u
    a = ttnn.multiply(softcap(g, 4.0), ttnn.sigmoid(g))
    return ttnn.typecast(ttnn.multiply(a, softcap(u, 25.0)), ttnn.bfloat16)


class MoE:
    def __init__(self, mesh, cfg, ck, li: int, expert_dtype=ttnn.bfloat4_b, cache_dir: str | None = None,
                 tt_ccl=None):
        self.mesh, self.li, self.tt_ccl = mesh, li, tt_ccl
        self.N = N = mesh.get_num_devices()
        self.E, self.k, self.H, self.I = cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.hidden_size, \
            cfg.intermediate_size
        assert self.I % (N * TILE) == 0
        self.Il = Il = self.I // N
        E, H = self.E, self.H
        rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
        self.router_w = ttnn.from_torch(ck.L(li, "mlp.gate.weight").float().T.contiguous().reshape(1, 1, H, E),
                                        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        self.router_b = ttnn.from_torch(ck.L(li, "mlp.gate.bias").float().reshape(1, 1, 1, E), dtype=ttnn.float32,
                                        layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        self.route_scale = float(cfg.route_scale)
        dt_name = {ttnn.bfloat4_b: "bfp4", ttnn.bfloat8_b: "bfp8", ttnn.bfloat16: "bf16"}[expert_dtype]
        cname = (lambda n: os.path.join(cache_dir, f"L{li}.{n}.{dt_name}.tp{N}")) if cache_dir else (lambda n: None)
        mm = ttnn.ShardTensorToMesh(mesh, dim=-1) if N > 1 else None
        mm2 = ttnn.ShardTensorToMesh(mesh, dim=-2) if N > 1 else None

        def make_ug():
            w13 = ck.experts(li, "w13")  # [E, 2I, H] fp32, rows [gate | up]
            gate_t, up_t = w13[:, : self.I].transpose(1, 2), w13[:, self.I :].transpose(1, 2)  # [E, H, I]
            return torch.cat([torch.cat([gate_t[:, :, d * Il : (d + 1) * Il], up_t[:, :, d * Il : (d + 1) * Il]], -1)
                              for d in range(N)], -1).unsqueeze(0).to(torch.bfloat16)  # [1, E, H, N*2*Il]

        def make_dn():
            dn = ck.experts(li, "w2").transpose(1, 2) * (self.route_scale / math.sqrt(5.0))  # [E, I, H]
            return dn.unsqueeze(0).to(torch.bfloat16)

        self.up_gate = cached(make_ug, expert_dtype, ttnn.TILE_LAYOUT, mesh, cname("ug"), mesh_mapper=mm)
        self.down = cached(make_dn, expert_dtype, ttnn.TILE_LAYOUT, mesh, cname("dn"), mesh_mapper=mm2)
        self.ones_mask = ttnn.from_torch(torch.ones(1, 1, 1, E, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        self.tile = ttnn.Tile([TILE, TILE])
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2 if expert_dtype != ttnn.bfloat16 else
            ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        self.rckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=False)
        # K in ONE in0 block (cap = all K tiles): with fp32 dest accumulation, sparse_matmul's masked mode mis-accumulates
        # across in0 blocks (layer 0, 32 rows: pcc 0.85 per expert at in0_block_w = Kt / 2, 1.00000 at Kt; this image)
        kcap = max(H, Il) // TILE
        self.cfg_gu_g = sparse_matmul_program_config(1, 2 * Il, H, target_cores=64, in0_block_w_cap=kcap)
        self.cfg_dn_g = sparse_matmul_program_config(1, H, Il, target_cores=64, in0_block_w_cap=kcap)
        self.cfg_gu_m = sparse_matmul_program_config(PF_CHUNK, 2 * Il, H, target_cores=32, in0_block_w_cap=kcap)
        self.cfg_dn_m = sparse_matmul_program_config(PF_CHUNK, H, Il, target_cores=32, in0_block_w_cap=kcap)

    # -- routing -----------------------------------------------------------------------------------------------------
    def route(self, x):
        """x [1, 1, T, H] bf16 -> (weights [1, 1, T, k] fp32 normalized (no route_scale), ids [1, 1, T, k] uint16)."""
        xf = ttnn.typecast(x, ttnn.float32)
        logits = ttnn.matmul(xf, self.router_w, compute_kernel_config=self.rckc, dtype=ttnn.float32)
        sig = ttnn.sigmoid(logits)
        choice = ttnn.add(sig, self.router_b)
        _, ids = ttnn.topk(choice, k=self.k, dim=-1, largest=True, sorted=True)
        w = ttnn.gather(sig, dim=-1, index=ids)
        wn = ttnn.div(w, ttnn.add(ttnn.sum(w, dim=-1, keepdim=True), 1e-20))
        ids16 = ttnn.typecast(ids, ttnn.uint16) if ids.dtype != ttnn.uint16 else ids
        return wn, ids16

    # -- experts -----------------------------------------------------------------------------------------------------
    def forward(self, x):
        """x [1, 1, T, H] bf16 (normed) -> [1, 1, T, H] bf16 (all-reduced over the mesh; times 1/sqrt(5), see doc)."""
        T = int(x.shape[2])
        w, ids = self.route(x)
        if T <= 8:
            out = self._gather(x, w, ids, T)
        else:
            out = self._masked(x, w, ids, T)
        if self.N > 1:
            out = all_reduce(out, self.mesh, self.tt_ccl)
        return out

    def _gather(self, x, w, ids, T):
        k, Il, H = self.k, self.Il, self.H
        ids_rm = ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
        outs = []
        for u in range(T):
            ids_u = ids_rm if T == 1 else ttnn.slice(ids_rm, [0, 0, u, 0], [1, 1, u + 1, k])
            x_u = x if T == 1 else ttnn.slice(x, [0, 0, u, 0], [1, 1, u + 1, H])
            fused = ttnn.sparse_matmul(x_u, self.up_gate, sparsity=self.ones_mask, indices=ids_u,
                                       is_input_a_sparse=False, is_input_b_sparse=True, memory_config=DRAM,
                                       output_tile=self.tile, program_config=self.cfg_gu_g,
                                       compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)  # [1,1,1,k,1,2Il]
            rows = ttnn.reshape(fused, (1, k, 1, 2 * Il))
            g = ttnn.slice(rows, [0, 0, 0, 0], [1, k, 1, Il])
            up = ttnn.slice(rows, [0, 0, 0, Il], [1, k, 1, 2 * Il])
            din = inkling_act(g, up)
            down = ttnn.sparse_matmul(din, self.down, sparsity=self.ones_mask, indices=ids_u, is_input_a_sparse=True,
                                      is_input_b_sparse=True, memory_config=DRAM, output_tile=self.tile,
                                      program_config=self.cfg_dn_g, compute_kernel_config=self.ckc,
                                      dtype=ttnn.bfloat16)  # [1,k,1,H]
            w_u = w if T == 1 else ttnn.slice(w, [0, 0, u, 0], [1, 1, u + 1, k])
            w_u = ttnn.reshape(w_u, (1, k, 1, 1))
            weighted = ttnn.multiply(ttnn.typecast(ttnn.reshape(down, (1, k, 1, H)), ttnn.float32), w_u)
            outs.append(ttnn.sum(weighted, dim=1, keepdim=True))  # [1,1,1,H] fp32 (top-k accumulated in fp32)
        acc = outs[0] if T == 1 else ttnn.concat(outs, dim=2)
        return ttnn.typecast(ttnn.reshape(acc, (1, 1, T, H)), ttnn.bfloat16)

    def _routing_matrix(self, w, ids, T):
        """Dense [1, 1, T, E] fp32 routing weights (zero for unselected experts): a row-major scatter (ttnn.scatter
        rejects fp32 TILE; a one-hot over a tile-padded k dim leaked its padding rows into expert 0)."""
        z = ttnn.zeros((1, 1, T, self.E), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh)
        idx = ttnn.to_layout(ttnn.typecast(ids, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        src = ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT)
        r = ttnn.scatter(z, dim=-1, index=idx, src=src)
        return ttnn.to_layout(r, ttnn.TILE_LAYOUT)

    def _masked(self, x, w, ids, T):
        E, Il, H, C = self.E, self.Il, self.H, PF_CHUNK
        routing = self._routing_matrix(w, ids, T)  # [1,1,T,E] fp32
        pad = (-T) % C
        if pad:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            routing = ttnn.pad(routing, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
        Tp = T + pad
        outs = []
        for s0 in range(0, Tp, C):
            x_c = ttnn.slice(x, [0, 0, s0, 0], [1, 1, s0 + C, H])
            r_c = ttnn.slice(routing, [0, 0, s0, 0], [1, 1, s0 + C, E])
            mask = ttnn.to_layout(ttnn.typecast(ttnn.sum(r_c, dim=2, keepdim=True), ttnn.bfloat16),
                                  ttnn.ROW_MAJOR_LAYOUT)
            fused = ttnn.sparse_matmul(x_c, self.up_gate, sparsity=mask, nnz=None, memory_config=DRAM,
                                       output_tile=self.tile, program_config=self.cfg_gu_m,
                                       compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)  # [1,1,1,E,C,2Il]
            fused = ttnn.reshape(fused, (1, E, C, 2 * Il))
            din = inkling_act(ttnn.slice(fused, [0, 0, 0, 0], [1, E, C, Il]),
                              ttnn.slice(fused, [0, 0, 0, Il], [1, E, C, 2 * Il]))
            ttnn.deallocate(fused)
            down = ttnn.sparse_matmul(din, self.down, sparsity=mask, nnz=None, is_input_a_sparse=True,
                                      is_input_b_sparse=True, memory_config=DRAM, output_tile=self.tile,
                                      program_config=self.cfg_dn_m, compute_kernel_config=self.ckc,
                                      dtype=ttnn.bfloat16)  # [1,E,C,H]
            ttnn.deallocate(din)
            wc = ttnn.permute(r_c, (0, 3, 2, 1))  # [1,E,C,1]
            weighted = ttnn.multiply(ttnn.typecast(ttnn.reshape(down, (1, E, C, H)), ttnn.float32), wc)
            ttnn.deallocate(down)
            outs.append(ttnn.sum(weighted, dim=1, keepdim=True))  # [1,1,C,H] fp32
            ttnn.deallocate(weighted)
        acc = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)
        acc = ttnn.typecast(acc, ttnn.bfloat16)
        if pad:
            acc = ttnn.slice(acc, [0, 0, 0, 0], [1, 1, T, H])
        return acc
