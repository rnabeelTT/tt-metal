# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 latent MoE (HF HYV4FlashMoE) and the dense MLP of layer 0, TP over the (1, N) mesh.

    logits = x @ W_r^T (fp32); s = sigmoid(logits); pick = top16(s + bias); w = s[pick] / sum * 3.998
    xl     = x @ W_down^T                                     (3072 -> 1536 latent, replicated)
    routed = sum_k w_k * D_k(clamp_silu(G_k xl) * clamp(U_k xl))   (latent 1536; expert intermediate split over TP)
    out    = combine_output_norm(W_up(routed) + shared(x))       (W_up 1536 -> 3072 applied to the per-die partial;
                                                                  one all-reduce of the sum)

Experts are packed per TP shard as [up_d | gate_d] (so a die's gate/up matmul is one sparse matmul), routed with
ttnn.sparse_matmul in "gather" mode for a few rows (decode / verify) or chunk-masked mode for prefill. SwiGLU with the
reference's clamps: gate <= 10, |up| <= 10.
"""

from __future__ import annotations

import os

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import AllReduce, RMSNorm, TILE, ckc, num_devices, upload

GATHER_MAX = int(os.environ.get("HY4_MOE_GATHER_MAX", "8"))
PREFILL_CHUNK = int(os.environ.get("HY4_MOE_CHUNK", "32"))


def _dt(name, default):
    return {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get(name, default)]


def _smm_cfg(M, N, K, target_cores=64, in0_block_w_cap=32, grid=(13, 10)):
    """A MatmulMultiCoreReuseMultiCast1DProgramConfig for sparse_matmul (one group of M rows x N cols)."""
    from models.autoports.minimaxai_minimax_m2_7.tt.moe import sparse_matmul_program_config

    return sparse_matmul_program_config(M, N, K, target_cores=target_cores, in0_block_w_cap=in0_block_w_cap)


class Hy4DenseMLP:
    """Layer 0: SwiGLU 3072 -> 13824 -> 3072 (gate/up column parallel, down row parallel), then post_fc2_norm."""

    def __init__(self, mesh, cfg, sd):
        self.mesh = mesh
        dt = _dt("HY4_DENSE_DTYPE", "bfp8")
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        I = cfg.dense_inter
        self.w1 = upload(mesh, bf(sd["gate_proj.weight"]).T.reshape(1, 1, cfg.hidden, I), dt, mapper=("shard", -1))
        self.w3 = upload(mesh, bf(sd["up_proj.weight"]).T.reshape(1, 1, cfg.hidden, I), dt, mapper=("shard", -1))
        self.w2 = upload(mesh, bf(sd["down_proj.weight"]).T.reshape(1, 1, I, cfg.hidden), dt, mapper=("shard", -2))
        self.norm = RMSNorm(mesh, sd["post_fc2_norm.weight"], cfg.rms_eps)
        self.ar = AllReduce(mesh)
        self.ckc = ckc(mesh)

    def forward(self, x, mem=None):
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        g = ttnn.matmul(x, self.w1, compute_kernel_config=self.ckc, memory_config=mem)
        u = ttnn.matmul(x, self.w3, compute_kernel_config=self.ckc, memory_config=mem)
        h = ttnn.multiply(u, g, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        part = ttnn.matmul(h, self.w2, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(h)
        out = self.ar(part, mem)
        ttnn.deallocate(part)
        return self.norm(out, mem)


class Hy4MoE:
    def __init__(self, mesh, cfg, sd, layer_idx: int, cache_dir=None):
        self.mesh, self.cfg, self.li = mesh, cfg, layer_idx
        self.N = tp = num_devices(mesh)
        self.E, self.k, self.L, self.I = cfg.n_experts, cfg.top_k, cfg.latent, cfg.moe_inter
        assert self.I % (tp * TILE) == 0
        self.Il = Il = self.I // tp
        self.limit = float(cfg.swiglu_limit)
        self.scale = float(cfg.routed_scale)
        self.ar = AllReduce(mesh)
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        cache = (lambda n: os.path.join(cache_dir, f"layers.{layer_idx}.mlp.{n}")) if cache_dir else (lambda n: None)
        gu_dt = _dt("HY4_EXPERT_GU_DTYPE", "bfp4")
        dn_dt = _dt("HY4_EXPERT_DN_DTYPE", "bfp4")
        sm_dt = _dt("HY4_MOE_SMALL_DTYPE", "bf16")
        # router (fp32 logits), selection bias
        self.wr = upload(mesh, sd["gate.weight"].float().T.reshape(1, 1, cfg.hidden, self.E), ttnn.float32,
                         cache=cache("router.fp32"))
        self.bias = upload(mesh, sd["gate.e_score_correction_bias"].float().reshape(1, 1, 1, self.E), ttnn.float32)
        # latent projections
        self.wd = upload(mesh, bf(sd["routed_expert_down_proj.weight"]).T.reshape(1, 1, cfg.hidden, self.L), sm_dt,
                         cache=cache(f"latent_down.{sm_dt}"))
        self.wu = upload(mesh, bf(sd["routed_expert_up_proj.weight"]).T.reshape(1, 1, self.L, cfg.hidden), sm_dt,
                         cache=cache(f"latent_up.{sm_dt}"))
        # routed experts: gate_up [E, 2I, L] (gate rows first), down [E, L, I]
        E, L, I = self.E, self.L, self.I
        have_cache = cache_dir and os.path.exists(cache(f"experts.up_gate.{gu_dt}") + ".tensorbin")
        if have_cache:
            gu_packed = torch.zeros(1)  # unused: as_tensor reads the cache file
            dn_t = torch.zeros(1)
        else:
            gu = sd["experts.gate_up_proj"]
            gate_t = gu[:, :I, :].transpose(1, 2)  # [E, L, I]
            up_t = gu[:, I:, :].transpose(1, 2)
            gu_packed = torch.cat(
                [torch.cat([up_t[:, :, d * Il : (d + 1) * Il], gate_t[:, :, d * Il : (d + 1) * Il]], dim=-1)
                 for d in range(tp)], dim=-1
            ).unsqueeze(0).contiguous()  # [1, E, L, tp * 2 Il]
            del gu, gate_t, up_t
            dn_t = sd["experts.down_proj"].transpose(1, 2).unsqueeze(0).contiguous()  # [1, E, I, L]
        self.up_gate = self._up(gu_packed, gu_dt, -1, cache(f"experts.up_gate.{gu_dt}"), have_cache, (1, E, L, tp * 2 * Il))
        self.down = self._up(dn_t, dn_dt, -2, cache(f"experts.down.{dn_dt}"), have_cache, (1, E, I, L))
        del gu_packed, dn_t
        # shared expert (TP)
        self.sw1 = upload(mesh, bf(sd["shared_experts.gate_proj.weight"]).T.reshape(1, 1, cfg.hidden, -1), sm_dt,
                          mapper=("shard", -1))
        self.sw3 = upload(mesh, bf(sd["shared_experts.up_proj.weight"]).T.reshape(1, 1, cfg.hidden, -1), sm_dt,
                          mapper=("shard", -1))
        self.sw2 = upload(mesh, bf(sd["shared_experts.down_proj.weight"]).T.reshape(1, 1, -1, cfg.hidden), sm_dt,
                          mapper=("shard", -2))
        self.norm = RMSNorm(mesh, sd["combine_output_norm.weight"], cfg.rms_eps)
        self.ones_mask = upload(mesh, torch.ones(1, 1, 1, E, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.iota = upload(mesh, torch.arange(E, dtype=torch.float32).reshape(1, 1, 1, E), ttnn.float32)
        self.tile = ttnn.Tile([TILE, TILE])
        self.ckc_r = ckc(mesh, ttnn.MathFidelity.HiFi4, fp32=True)
        self.ckc_e = ckc(mesh, ttnn.MathFidelity.HiFi2, fp32=False)  # tt-metal #49068: no fp32 dest on BH sparse mm
        self.ckc_s = ckc(mesh, ttnn.MathFidelity.HiFi4, fp32=True)
        self.cfg_gu_g = _smm_cfg(1, 2 * Il, L, target_cores=64, in0_block_w_cap=96)
        self.cfg_dn_g = _smm_cfg(1, L, Il, target_cores=64, in0_block_w_cap=32)
        self.cfg_gu_m = _smm_cfg(PREFILL_CHUNK, 2 * Il, L, target_cores=32, in0_block_w_cap=64)
        self.cfg_dn_m = _smm_cfg(PREFILL_CHUNK, L, Il, target_cores=32, in0_block_w_cap=32)

    def _up(self, t, dtype, dim, cache_name, have_cache, shape):
        kw = {"mesh_mapper": ttnn.ShardTensorToMesh(self.mesh, dim=dim)} if self.N > 1 else {}
        if have_cache:
            t = torch.zeros(shape, dtype=torch.bfloat16)  # shape only; the cache file supplies the data
        return ttnn.as_tensor(t if t.dtype == torch.bfloat16 else t.to(torch.bfloat16), dtype=dtype,
                              layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                              cache_file_name=cache_name, **kw)

    # ------------------------------------------------------------------ routing
    def route(self, x, mem):
        """-> (weights bf16 [1,1,T,16] incl. x 3.998, ids uint16 [1,1,T,16])."""
        T = int(x.shape[-2])
        xf = ttnn.typecast(x, ttnn.float32, memory_config=mem)
        logits = ttnn.matmul(xf, self.wr, dtype=ttnn.float32, compute_kernel_config=self.ckc_r, memory_config=mem)
        ttnn.deallocate(xf)
        s = ttnn.sigmoid(logits, memory_config=mem)
        ttnn.deallocate(logits)
        choice = ttnn.add(s, self.bias, memory_config=mem)
        _, ids = ttnn.topk(choice, k=self.k, dim=-1, largest=True, sorted=True, memory_config=mem)
        ttnn.deallocate(choice)
        # w[t, j] = s[t, ids[t, j]]: one-hot select (exact)
        idf = ttnn.typecast(ids, ttnn.float32, memory_config=mem)
        oh = ttnn.eq(ttnn.reshape(idf, (1, T, self.k, 1)), self.iota, memory_config=mem)  # [1,T,k,E]
        sel = ttnn.multiply(oh, ttnn.reshape(s, (1, T, 1, self.E)), memory_config=mem)
        ttnn.deallocate(oh)
        ttnn.deallocate(s)
        w = ttnn.reshape(ttnn.sum(sel, dim=-1, keepdim=True, memory_config=mem), (1, 1, T, self.k))
        ttnn.deallocate(sel)
        den = ttnn.add(ttnn.sum(w, dim=-1, keepdim=True, memory_config=mem), 1e-20, memory_config=mem)
        wn = ttnn.multiply(ttnn.div(w, den, memory_config=mem), self.scale, memory_config=mem)
        ttnn.deallocate(w)
        scores = ttnn.typecast(wn, ttnn.bfloat16, memory_config=mem)
        ids16 = ttnn.typecast(ids, ttnn.uint16, memory_config=mem) if ids.dtype != ttnn.uint16 else ids
        return scores, ids16

    def _swiglu(self, up, gate, mem):
        g = ttnn.clamp(gate, max=self.limit, memory_config=mem)
        u = ttnn.clamp(up, min=-self.limit, max=self.limit, memory_config=mem)
        return ttnn.multiply(u, g, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)

    # ------------------------------------------------------------------ experts
    def _gather(self, xl, scores, ids, T, mem):
        """Few rows: per row, the k picked experts' gate/up and down as two sparse matmuls."""
        k, Il, L = self.k, self.Il, self.L
        ids_rm = ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
        outs = []
        for u in range(T):
            ids_u = ttnn.slice(ids_rm, [0, 0, u, 0], [1, 1, u + 1, k])
            x_u = ttnn.slice(xl, [0, 0, u, 0], [1, 1, u + 1, L])
            fused = ttnn.sparse_matmul(x_u, self.up_gate, sparsity=self.ones_mask, indices=ids_u, is_input_a_sparse=False,
                                       is_input_b_sparse=True, memory_config=mem, output_tile=self.tile,
                                       program_config=self.cfg_gu_g, compute_kernel_config=self.ckc_e,
                                       dtype=ttnn.bfloat16)
            fused = ttnn.reshape(fused, (1, k, 1, 2 * Il))
            up = ttnn.slice(fused, [0, 0, 0, 0], [1, k, 1, Il])
            gate = ttnn.slice(fused, [0, 0, 0, Il], [1, k, 1, 2 * Il])
            ttnn.deallocate(fused)
            din = self._swiglu(up, gate, mem)
            ttnn.deallocate(up)
            ttnn.deallocate(gate)
            down = ttnn.sparse_matmul(din, self.down, sparsity=self.ones_mask, indices=ids_u, is_input_a_sparse=True,
                                      is_input_b_sparse=True, memory_config=mem, output_tile=self.tile,
                                      program_config=self.cfg_dn_g, compute_kernel_config=self.ckc_e,
                                      dtype=ttnn.bfloat16)
            ttnn.deallocate(din)
            down = ttnn.reshape(down, (1, k, 1, L))
            sc_u = ttnn.reshape(ttnn.slice(scores, [0, 0, u, 0], [1, 1, u + 1, k]), (1, k, 1, 1))
            weighted = ttnn.multiply(down, sc_u, memory_config=mem)
            ttnn.deallocate(down)
            outs.append(ttnn.sum(weighted, dim=1, keepdim=True, memory_config=mem))  # [1,1,1,L]
            ttnn.deallocate(weighted)
            ttnn.deallocate(ids_u)
            ttnn.deallocate(x_u)
        ttnn.deallocate(ids_rm)
        acc = ttnn.concat(outs, dim=2, memory_config=mem) if T > 1 else outs[0]
        if T > 1:
            for o in outs:
                ttnn.deallocate(o)
        return ttnn.reshape(acc, (1, 1, T, L))

    def _masked(self, xl, scores, ids, T, mem):
        """Many rows: chunks of PREFILL_CHUNK rows; each chunk runs every expert any of its rows picked."""
        E, Il, L, C = self.E, self.Il, self.L, PREFILL_CHUNK
        Tp = -(-T // C) * C
        # dense routing weights [1,1,T,E] (zero for unpicked)
        z = ttnn.zeros((1, 1, int(scores.shape[-2]), E), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                       memory_config=mem)
        idx = ttnn.typecast(ids, ttnn.uint32, memory_config=mem)
        routing = ttnn.scatter(z, dim=-1, index=idx, src=scores)
        ttnn.deallocate(z)
        ttnn.deallocate(idx)
        if Tp != int(routing.shape[-2]):
            routing = ttnn.pad(routing, [(0, 0), (0, 0), (0, Tp - int(routing.shape[-2])), (0, 0)], value=0.0)
        if Tp != int(xl.shape[-2]):
            xl = ttnn.pad(xl, [(0, 0), (0, 0), (0, Tp - int(xl.shape[-2])), (0, 0)], value=0.0)
        outs = []
        for s0 in range(0, Tp, C):
            x_c = ttnn.slice(xl, [0, 0, s0, 0], [1, 1, s0 + C, L], memory_config=mem)
            r_c = ttnn.slice(routing, [0, 0, s0, 0], [1, 1, s0 + C, E], memory_config=mem)
            mask = ttnn.to_layout(ttnn.sum(r_c, dim=2, keepdim=True, memory_config=mem), ttnn.ROW_MAJOR_LAYOUT,
                                  memory_config=mem)
            fused = ttnn.sparse_matmul(x_c, self.up_gate, sparsity=mask, nnz=None, memory_config=mem,
                                       output_tile=self.tile, program_config=self.cfg_gu_m,
                                       compute_kernel_config=self.ckc_e, dtype=ttnn.bfloat16)
            ttnn.deallocate(x_c)
            fused = ttnn.reshape(fused, (1, E, C, 2 * Il))
            up = ttnn.slice(fused, [0, 0, 0, 0], [1, E, C, Il], memory_config=mem)
            gate = ttnn.slice(fused, [0, 0, 0, Il], [1, E, C, 2 * Il], memory_config=mem)
            ttnn.deallocate(fused)
            din = self._swiglu(up, gate, mem)
            ttnn.deallocate(up)
            ttnn.deallocate(gate)
            down = ttnn.sparse_matmul(din, self.down, sparsity=mask, nnz=None, is_input_a_sparse=True,
                                      is_input_b_sparse=True, memory_config=mem, output_tile=self.tile,
                                      program_config=self.cfg_dn_m, compute_kernel_config=self.ckc_e,
                                      dtype=ttnn.bfloat16)
            ttnn.deallocate(din)
            w_c = ttnn.permute(r_c, (0, 3, 2, 1), memory_config=mem)  # [1,E,C,1]
            weighted = ttnn.multiply(ttnn.reshape(down, (1, E, C, L)), w_c, memory_config=mem)
            for t_ in (down, w_c, r_c, mask):
                ttnn.deallocate(t_)
            outs.append(ttnn.experimental.fast_reduce_nc(weighted, dims=[1]))  # [1,1,C,L]
            ttnn.deallocate(weighted)
        ttnn.deallocate(routing)
        acc = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2, memory_config=mem)
        if len(outs) > 1:
            for o in outs:
                ttnn.deallocate(o)
        acc = ttnn.reshape(acc, (1, 1, Tp, L))
        if Tp != T:
            acc = ttnn.slice(acc, [0, 0, 0, 0], [1, 1, T, L], memory_config=mem)
        return acc

    def _shared(self, x, mem):
        g = ttnn.matmul(x, self.sw1, compute_kernel_config=self.ckc_s, memory_config=mem)
        u = ttnn.matmul(x, self.sw3, compute_kernel_config=self.ckc_s, memory_config=mem)
        h = ttnn.multiply(u, g, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = ttnn.matmul(h, self.sw2, compute_kernel_config=self.ckc_s, memory_config=mem)
        ttnn.deallocate(h)
        return out

    def forward(self, x, T: int, mem=None):
        """x [1, 1, Tp, 3072] bf16 replicated (normed); rows >= T are padding. -> [1, 1, Tp, 3072] replicated."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        Tp = int(x.shape[-2])
        scores, ids = self.route(x, mem)
        xl = ttnn.matmul(x, self.wd, compute_kernel_config=self.ckc_s, memory_config=mem)  # [1,1,Tp,1536]
        if T <= GATHER_MAX:
            routed = self._gather(xl, scores, ids, T, mem)
            if Tp != T:
                routed = ttnn.pad(routed, [(0, 0), (0, 0), (0, Tp - T), (0, 0)], value=0.0)
        else:
            routed = self._masked(xl, scores, ids, Tp, mem)
        ttnn.deallocate(xl)
        ttnn.deallocate(scores)
        ttnn.deallocate(ids)
        up = ttnn.matmul(routed, self.wu, compute_kernel_config=self.ckc_s, memory_config=mem)  # partial [Tp,3072]
        ttnn.deallocate(routed)
        sh = self._shared(x, mem)
        part = ttnn.add(up, sh, memory_config=mem)
        ttnn.deallocate(up)
        ttnn.deallocate(sh)
        full = self.ar(part, mem)
        ttnn.deallocate(part)
        return self.norm(full, mem)
