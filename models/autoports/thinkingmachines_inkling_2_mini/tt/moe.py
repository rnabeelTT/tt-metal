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

from .dmm import dmm
from models.autoports.thinkingmachines_inkling_2_mini.tt.ccl import all_reduce, packed_rows
from models.autoports.minimaxai_minimax_m2_7.tt.moe import sparse_matmul_program_config
from models.autoports.thinkingmachines_inkling_2_mini.tt.wload import cached

TILE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG
PF_CHUNK = int(os.environ.get("I2M_MOE_PF_CHUNK", "32"))
# I2M_MOE_SORTED (default 1): prefill rows (> the fused decode width) through the sorted dispatch (MiniMax / Qwen3.5
# moe_dispatch: one grouped sparse matmul per projection, each 32-row group one expert) instead of the masked union
SORTED = os.environ.get("I2M_MOE_SORTED", "1") == "1"
# rows per expert group in the sorted dispatch (32 | 64 | 128): larger groups re-read each expert's weights fewer
# times (one weight read per group) at more padding; long prefill chunks favour larger groups
GROUP_ROWS = int(os.environ.get("I2M_MOE_GROUP", "32"))
# I2M_MOE_GMM (default 1): the sorted prefill's expert matmuls as tt/gmm.py (every core) instead of ttnn.sparse_matmul
# (its 1D config puts gate|up's 24 output tiles on 24 cores)
GMM = os.environ.get("I2M_MOE_GMM", "1") == "1"
_GMM = __import__("types").SimpleNamespace()
assert GROUP_ROWS in (32, 64), "the sorted dispatch groups: 32 or 64 rows (128 mis-dispatches: pcc 0.87)"
SPLIT_DECODE = os.environ.get("I2M_MOE_SPLIT", "1") == "1"  # 17-32 decode rows as two fused 16-row calls
# I2M_MOE_LAYOUT=fused (default): ONE copy of the experts in the fused decode kernels' layout (tt/moe_fused.py: per
# sub-shard [gate | up / 25], down x 100); decode rows (<= 16) run the fused kernels, prefill / wider batches the
# sparse_matmul union path over the same sub-shard tensors. sparse: the original layout and paths.
FUSED_LAYOUT = os.environ.get("I2M_MOE_LAYOUT", "fused") == "fused"


def softcap(t, c):
    return ttnn.multiply(ttnn.tanh(ttnn.multiply(t, 1.0 / c)), c)


def inkling_act_fused(g, u):
    """The fused layout's activation: tanh(g / 4) sigmoid(g) tanh(u') (u' = u / 25 in the weights, x 100 in down)."""
    g = ttnn.typecast(g, ttnn.float32) if g.dtype != ttnn.float32 else g
    u = ttnn.typecast(u, ttnn.float32) if u.dtype != ttnn.float32 else u
    a = ttnn.multiply(ttnn.tanh(ttnn.multiply(g, 0.25)), ttnn.sigmoid(g))
    return ttnn.typecast(ttnn.multiply(a, ttnn.tanh(u)), ttnn.bfloat16)


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
        # the router weight as bf16 hi / lo halves (the fused message gate's router round, tt/gate.py GateKernel2)
        rw = ck.L(li, "mlp.gate.weight").float().T.contiguous().reshape(1, 1, H, E)
        rhi = rw.bfloat16().float()
        self.router_hl = tuple(ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                               mesh_mapper=rep) for t in (rhi, (rw - rhi).bfloat16().float()))
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

        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2 if expert_dtype != ttnn.bfloat16 else
            ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        self.rckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=False)
        self.fused = FUSED_LAYOUT and expert_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b)
        if self.fused:
            from models.autoports.thinkingmachines_inkling_2_mini.tt.moe_fused import FusedExperts

            self.fe = FusedExperts(self, ck, li, dtype=expert_dtype, cache_dir=cache_dir)
            kc = lambda W: max(H, W) // TILE  # noqa: E731
            self.subs = [(ug, dn, W, sparse_matmul_program_config(PF_CHUNK, 2 * W, H, target_cores=32,
                                                                     in0_block_w_cap=kc(W)),
                          sparse_matmul_program_config(PF_CHUNK, H, W, target_cores=32, in0_block_w_cap=kc(W)))
                         for (op, ug, dn), (_, W) in zip(self.fe.subs, self.fe.widths)]
        else:
            self.up_gate = cached(make_ug, expert_dtype, ttnn.TILE_LAYOUT, mesh, cname("ug"), mesh_mapper=mm)
            self.down = cached(make_dn, expert_dtype, ttnn.TILE_LAYOUT, mesh, cname("dn"), mesh_mapper=mm2)
        self.ones_mask = ttnn.from_torch(torch.ones(1, 1, 1, E, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        self.tile = ttnn.Tile([TILE, TILE])
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
        logits = dmm(xf, self.router_w, self.rckc, dtype=ttnn.float32)
        sig = ttnn.sigmoid(logits)
        choice = ttnn.add(sig, self.router_b)
        _, ids = ttnn.topk(choice, k=self.k, dim=-1, largest=True, sorted=True)
        w = ttnn.gather(sig, dim=-1, index=ids)
        wn = ttnn.div(w, ttnn.add(ttnn.sum(w, dim=-1, keepdim=True), 1e-20))
        ids16 = ttnn.typecast(ids, ttnn.uint16) if ids.dtype != ttnn.uint16 else ids
        return wn, ids16

    # -- experts -----------------------------------------------------------------------------------------------------
    def forward(self, x, logits=None, dup=False, packed_out=False):
        """x [1, 1, T, H] bf16 (normed) -> [1, 1, T, H] bf16 (all-reduced over the mesh; times 1/sqrt(5), see doc).
        logits: the router logits computed upstream ([1, 1, max(T, 2), E] fp32; T = 1: dup, row 1 = row 0 in x too)."""
        T = int(x.shape[2])
        rows = None
        if self.fused and T <= self.fe.rows_max:
            # packed all-reduce: it reads the shared expert-output buffer's live rows directly (no clone / slice)
            raw = self.N > 1 and len(self.fe.subs) == 1 and packed_rows(T)
            out = self.fe.forward(x, logits=logits, dup=dup, raw=raw)
            rows = T if raw else None
        elif self.fused and T <= 2 * self.fe.rows_max and SPLIT_DECODE:
            # 17 .. 32 decode rows: two fused calls of <= 16 rows (the fused kernels' row limit) instead of the sorted
            # prefill path (32 rows: ~3.3 ms per application on TP4 vs ~2 x the 16-row fused call)
            R = self.fe.rows_max
            lo = self.fe.forward(ttnn.slice(x, [0, 0, 0, 0], [1, 1, R, self.H]))
            hi = self.fe.forward(ttnn.slice(x, [0, 0, R, 0], [1, 1, T, self.H]))
            out = ttnn.concat([lo, hi], dim=2)
        elif self.fused and SORTED:
            out = self._sorted(x, T)
        elif self.fused:
            w, ids = self.route(x)
            out = None
            for ug, dn, W, cgu, cdn in self.subs:
                part = self._masked(x, w, ids, T, ug=ug, dn=dn, Il=W, cfg_gu=cgu, cfg_dn=cdn, act=inkling_act_fused)
                out = part if out is None else ttnn.add(out, part)
        else:
            w, ids = self.route(x)
            if T <= 8:  # decode rows: per token, only its 8 experts are read (the union path computes all E slots)
                out = self._gather(x, w, ids, T)
            else:
                out = self._masked(x, w, ids, T)
        if self.N > 1:
            out = all_reduce(out, self.mesh, self.tt_ccl, site="mlp", rows=rows, packed_out=packed_out)
        return out

    def _sorted(self, x, T):
        """Prefill rows through the sorted dispatch: tokens' (token, expert) pairs sorted by expert into 32-row groups
        (one expert each), one grouped sparse matmul per projection and sub-shard, routing weights on the activations,
        a fused gather-add un-dispatch (bf16) per sub-shard; the sub-shards summed."""
        from models.autoports.minimaxai_minimax_m2_7.tt.moe_dispatch import SortedDispatch, gather_rows, undispatch_rows

        E, H, k = self.E, self.H, self.k
        gs = GROUP_ROWS if T >= 4 * GROUP_ROWS else 32
        self._dispatches = getattr(self, "_dispatches", {})
        if gs not in self._dispatches:
            self._dispatches[gs] = SortedDispatch(self.mesh, E, k, group=gs)
        if getattr(self, "_cfg_sorted", {}).get(("gs", gs)) is None:
            kc = lambda W: max(H, W) // TILE  # noqa: E731  (K in one in0 block, see the masked configs)
            self._cfg_sorted = getattr(self, "_cfg_sorted", {})
            self._cfg_sorted[("gs", gs)] = {
                W: (sparse_matmul_program_config(gs, 2 * W, H, target_cores=64, in0_block_w_cap=kc(W)),
                    sparse_matmul_program_config(gs, H, W, target_cores=64, in0_block_w_cap=kc(W)))
                for _, _, W, _, _ in self.subs}
        w, ids = self.route(x)
        m = self._dispatches[gs].build(ids, ttnn.typecast(w, ttnn.bfloat16), T)
        Rd, G = m["Rd"], m["G"]
        x_disp = gather_rows(x, m["src_tok"], Rd, gs)  # [1, G, 32, H]
        out = None
        for ug, dn, W, _, _ in self.subs:
            cgu, cdn = self._cfg_sorted[("gs", gs)][W]
            alloc = lambda n: ttnn.allocate_tensor_on_device(ttnn.Shape([1, G, gs, n]), ttnn.bfloat16,  # noqa: E731
                                                             ttnn.TILE_LAYOUT, self.mesh, DRAM)
            if GMM:  # the grouped expert matmul over the whole grid (tt/gmm.py)
                fused = self._gmm(x_disp, ug, m["group_ids"], alloc(2 * W), E)
            else:
                fused = ttnn.sparse_matmul(x_disp, ug, sparsity=m["group_mask"], nnz=G, memory_config=DRAM,
                                           program_config=cgu, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16,
                                           optional_output_tensor=alloc(2 * W))  # compact [1, G, 32, 2W]
            din = inkling_act_fused(ttnn.slice(fused, [0, 0, 0, 0], [1, G, gs, W]),
                                    ttnn.slice(fused, [0, 0, 0, W], [1, G, gs, 2 * W]))
            ttnn.deallocate(fused)
            din = ttnn.reshape(ttnn.multiply(ttnn.reshape(din, (1, 1, Rd, W)), m["weights"]), (1, G, gs, W))
            if GMM:
                down = self._gmm(din, dn, m["group_ids"], alloc(H), E)
            else:
                down = ttnn.sparse_matmul(din, dn, sparsity=m["group_mask"], nnz=G, is_input_a_sparse=False,
                                          is_input_b_sparse=True, memory_config=DRAM, program_config=cdn,
                                          compute_kernel_config=self.ckc, dtype=ttnn.bfloat16,
                                          optional_output_tensor=alloc(H))  # compact [1, G, 32, H]
            ttnn.deallocate(din)
            y = undispatch_rows(down, None, m["undispatch"], T, k, H, undispatch_rm=m.get("undispatch_rm"))
            ttnn.deallocate(down)
            out = y if out is None else ttnn.add(out, y)
        ttnn.deallocate(x_disp)
        for key in ("src_tok", "weights", "group_mask", "group_ids", "undispatch", "undispatch_rm"):
            if m.get(key) is not None:
                ttnn.deallocate(m[key])
        return out

    def _gmm(self, a, w, ids, out, E):
        if getattr(_GMM, "op", None) is None:
            from .gmm import GroupedMM

            _GMM.op = GroupedMM(self.mesh, ttnn.MathFidelity.HiFi2 if w.dtype != ttnn.bfloat16 else
                                ttnn.MathFidelity.HiFi4)
        return _GMM.op(a, w, ids, out, E)

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

    def _masked(self, x, w, ids, T, ug=None, dn=None, Il=None, cfg_gu=None, cfg_dn=None, act=None):
        E, H, C = self.E, self.H, PF_CHUNK
        Il = self.Il if Il is None else Il
        ug = self.up_gate if ug is None else ug
        dn = self.down if dn is None else dn
        cfg_gu = self.cfg_gu_m if cfg_gu is None else cfg_gu
        cfg_dn = self.cfg_dn_m if cfg_dn is None else cfg_dn
        act = inkling_act if act is None else act
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
            fused = ttnn.sparse_matmul(x_c, ug, sparsity=mask, nnz=None, memory_config=DRAM,
                                       output_tile=self.tile, program_config=cfg_gu,
                                       compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)  # [1,1,1,E,C,2Il]
            fused = ttnn.reshape(fused, (1, E, C, 2 * Il))
            din = act(ttnn.slice(fused, [0, 0, 0, 0], [1, E, C, Il]), ttnn.slice(fused, [0, 0, 0, Il], [1, E, C, 2 * Il]))
            ttnn.deallocate(fused)
            down = ttnn.sparse_matmul(din, dn, sparsity=mask, nnz=None, is_input_a_sparse=True,
                                      is_input_b_sparse=True, memory_config=DRAM, output_tile=self.tile,
                                      program_config=cfg_dn, compute_kernel_config=self.ckc,
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
