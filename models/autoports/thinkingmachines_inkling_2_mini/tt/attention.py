# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2 attention on TT (transformers `Inkling2SlidingAttention` / `Inkling2FullAttention`), tensor-parallel over
the heads of a mesh of N dies (heads column-sharded, o_proj row-sharded + all-reduce; activations replicated).

Per layer weights (`AttnWeights`, shared by the 3 passes of a looped layer) and per application caches (`AttnCache`:
K / V caches and the conv states of one (layer, pass)).

    q   = L2clip(q_lin(xq) * q_scale, 22.6) [* tau on global layers]          (no RoPE)
    k   = L2clip(sconv(k_lin(xk)), 22.6)       sliding: static conv; global: dynamic conv driven by xk
    v   = rms(sconv(v_lin(xv)))                (unweighted per-head RMS norm)
    r   = L2clip(r_lin(xq) * r_scale, 8)    [* tau]   -> relative bias (r @ proj)[q - k], 0 <= q - k < 1024
    s   = sink_lin(xq)                         a zero-value extra key per (row, head)
    out = o_lin(sdpa(q, k, v; bias, sink, window) * sigmoid(gate_lin(xq)))

The short convs' output scales (1/sqrt(5) static, 1/3 dynamic) are folded into the preceding projection weights (k, v
here; o_proj for the post-attention conv): the convs are linear in their input. tau = 1 + 0.1 log(max((pos+1)/8192, 1)).
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.ccl import all_reduce
from models.autoports.thinkingmachines_inkling_2_mini.tt import relbias
from models.autoports.thinkingmachines_inkling_2_mini.tt.attn.inkling_sdpa import inkling_sdpa_decode
from models.autoports.thinkingmachines_inkling_2_mini.tt.sconv import DynShortConv, ShortConv

DRAM = ttnn.DRAM_MEMORY_CONFIG
FUSED_OPS = __import__("os").environ.get("I2M_OPS", "fused") == "fused"
# I2M_DRAM_SHARDED (default 1): the attention projections as DRAM-width-sharded weights (qwen36 tp_common
# sharded_decode_matmul: the DRAM-sharded kernel for decode rows, a 2D matmul for prefill)
# (measured: at one row these projections are compute-bound on the 32-row tile; DRAM-sharded only wins at LoFi without
# fp32 accumulation, 45 vs 60 us; at HiFi2 it is slower, 67-69 us: off by default)
DRAM_SHARDED = __import__("os").environ.get("I2M_DRAM_SHARDED", "0") == "1"
SQRT5 = math.sqrt(5.0)
R_ITEM = 8  # prefill: rows per SDPA work item (R * G <= 32 at G = 3)


def tau_of(positions: torch.Tensor, n_floor: int, alpha: float) -> torch.Tensor:
    pos = positions.double() + 1.0
    return (1.0 + alpha * torch.log(torch.clamp(pos / n_floor, min=1.0))).float()


class AttnWeights:
    def __init__(self, mesh, cfg, ck, li: int, wdtype=ttnn.bfloat16, ckc=None):
        self.mesh, self.li = mesh, li
        self.N = N = mesh.get_num_devices()
        self.sliding = li in cfg.local_layer_ids
        self.HD = HD = cfg.swa_head_dim if self.sliding else cfg.head_dim
        nh = cfg.swa_num_attention_heads if self.sliding else cfg.num_attention_heads
        nkv = cfg.swa_num_key_value_heads if self.sliding else cfg.num_key_value_heads
        assert nh % N == 0 and nkv % N == 0, (nh, nkv, N)
        self.NH, self.NKV, self.DR = NH, NKV, DR = nh // N, nkv // N, cfg.d_rel
        self.W = cfg.sliding_window_size if self.sliding else 0
        self.E = cfg.sliding_window_size if self.sliding else cfg.rel_extent
        self.scale = HD**-0.5
        self.qk_clip, self.rel_clip = cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold
        self.n_floor, self.alpha = cfg.log_scaling_n_floor, cfg.log_scaling_alpha
        self.eps = cfg.rms_norm_eps
        L = lambda n: ck.L(li, f"attn.{n}")  # noqa: E731
        wq, wr, ws, wg = L("wq_du.weight"), L("wr_du.weight"), L("attn_sink_proj.weight"), L("gate_attn.weight")
        conv_scale = 3.0 if not self.sliding else SQRT5
        wk, wv = L("wk_dv.weight").float() / conv_scale, L("wv_dv.weight").float() / conv_scale
        # the post-attention conv's 1/sqrt(5) folded into o_proj (the conv input is o_proj's output)
        wo = L("wo_ud.weight").float() / SQRT5
        # fused q-input projection per die: [q_d | r_d | s_d (padded to 32) | g_d], die-major columns
        self.qd, self.rd, self.sd, self.gd = NH * HD, NH * DR, 32, NH * HD
        self.kd, self.H = NKV * HD, cfg.hidden_size
        self.fq = self.qd + self.rd + self.sd + self.gd
        if FUSED_OPS:  # the per-(head, dim) q / r scales folded into the projection rows (exact up to weight rounding)
            wq = wq.float() * L("q_l2_clip.scale").float().reshape(-1, 1)
            wr = wr.float() * L("rel_l2_clip.scale").float().reshape(-1, 1)
        cols = []
        for d in range(N):
            s_d = torch.zeros(32, wq.shape[1])
            s_d[:NH] = ws[d * NH : (d + 1) * NH].float()
            cols += [wq[d * NH * HD : (d + 1) * NH * HD].float(), wr[d * NH * DR : (d + 1) * NH * DR].float(), s_d,
                     wg[d * NH * HD : (d + 1) * NH * HD].float()]
        shard_c = ttnn.ShardTensorToMesh(mesh, dim=-1) if N > 1 else None
        shard_r = ttnn.ShardTensorToMesh(mesh, dim=-2) if N > 1 else None
        rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
        up = lambda t, dt, mm: ttnn.as_tensor(t.to(torch.bfloat16).contiguous(), dtype=dt,  # noqa: E731
                                              layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                                              mesh_mapper=mm)
        self.dram_sharded = DRAM_SHARDED
        if DRAM_SHARDED:
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            def dsh(per_die):  # per_die: list of N [K, n_d] (input-major) blocks -> DRAM-sharded [K, N * n_pad]
                K, n = per_die[0].shape
                npad = -(-n // (32 * 8)) * (32 * 8)
                t = torch.cat([torch.nn.functional.pad(b, (0, npad - n)) for b in per_die], -1)
                mc = tpc.create_dram_sharded_mem_config(K, npad)
                w = ttnn.as_tensor(t.to(torch.bfloat16).reshape(1, 1, K, -1).contiguous(), dtype=wdtype,
                                   layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc,
                                   mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1) if N > 1 else None)
                return w, (tpc.create_dram_sharded_matmul_program_config(32, K, npad),
                           tpc.create_activation_shard_config(K), K)

            qcat = torch.cat(cols, 0)  # [N * fq, H] die-major rows
            fq = self.fq
            self.w_q, self.pc_q = dsh([qcat[d * fq : (d + 1) * fq].T for d in range(N)])
            kd = NKV * HD
            self.w_k, self.pc_k = dsh([wk[d * kd : (d + 1) * kd].T for d in range(N)])
            self.w_v, self.pc_v = dsh([wv[d * kd : (d + 1) * kd].T for d in range(N)])
            od = NH * HD  # o_proj: row-parallel (input dim split) -> per die [od, H]
            self.w_o, self.pc_o = dsh([wo[:, d * od : (d + 1) * od].T for d in range(N)])
        else:
            self.w_q = up(torch.cat(cols, 0).T.reshape(1, 1, wq.shape[1], -1), wdtype, shard_c)
            self.w_k = up(wk.T.reshape(1, 1, wk.shape[1], -1), wdtype, shard_c)
            self.w_v = up(wv.T.reshape(1, 1, wv.shape[1], -1), wdtype, shard_c)
            self.w_o = up(wo.T.reshape(1, 1, wo.shape[1], -1), wdtype, shard_r)
        # per-head scales (fp32): q [NH, HD], r [NH, DR] (head-sharded), as [1, 1, NH, HD] rows
        hs = lambda t, n: ttnn.from_torch(t.float().reshape(1, 1, n * N, -1), dtype=ttnn.float32,  # noqa: E731
                                          layout=ttnn.TILE_LAYOUT, device=mesh,
                                          mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2) if N > 1 else None)
        self.q_scale = None if FUSED_OPS else hs(L("q_l2_clip.scale"), NH)
        self.r_scale = None if FUSED_OPS else hs(L("rel_l2_clip.scale"), NH)
        T = relbias.build_tables(L("rel_logits_proj.proj").float(), self.scale)
        self.table = ttnn.as_tensor(T.reshape(1, 32, 32, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=mesh, memory_config=DRAM, mesh_mapper=rep)
        # sink -> [1, R, NH, 32]: sink / scale in every column (see inkling_sdpa)
        self.sink_bcast = ttnn.from_torch(torch.full((1, 1, 1, 32), 1.0 / self.scale), dtype=ttnn.float32,
                                          layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        # conv taps kept on the host for the per-application caches
        if self.sliding:
            self.k_taps = L("k_sconv.weight").reshape(nkv * HD, -1).float()
            self.v_taps = L("v_sconv.weight").reshape(nkv * HD, -1).float()
        else:
            self.k_static, self.k_wproj = L("k_sconv.static_w").float(), L("k_sconv.weight_proj").float()
            self.v_static, self.v_wproj = L("v_sconv.static_w").float(), L("v_sconv.weight_proj").float()
        # HiFi2 reads a bfp8 weight's whole mantissa; HiFi4 only for bf16 weights
        self.ckc = ckc or ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4 if wdtype == ttnn.bfloat16 else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)


def proj(w: "AttnWeights", which: str, x, dtype=None):
    """One attention projection (q | k | v | o) on x [1, 1, M, K] -> [1, 1, M, n(+pad)] in DRAM."""
    W = getattr(w, "w_" + which)
    if not w.dram_sharded:
        return ttnn.linear(x, W, compute_kernel_config=w.ckc, memory_config=DRAM, dtype=dtype)
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    pc, act, K = getattr(w, "pc_" + which)
    return tpc.sharded_decode_matmul(x, W, w.ckc, pc, act, tpc.create_prefill_matmul_program_config, K,
                                     decode_out_memory_config=DRAM, out_dtype=dtype)


class AttnCache:
    """K / V caches of one application (layer, pass) for B sequence slots, plus the conv states of a single-sequence
    "lane" (prefill / verify rows run one slot at a time on the lane; batched decode keeps its conv states in
    tt/batched.py and copies the lane's states in after a prefill).

    Global layers: a paged pool of B * max_blocks blocks, slot b owning blocks [b * mb, (b + 1) * mb). Sliding layers:
    one ring of n_ring blocks per slot (the window + one prefill chunk + a block), slot b's logical block j at physical
    b * n_ring + j % n_ring; writes go through an identity table per slot at position pos mod (n_ring * block)."""

    def __init__(self, w: AttnWeights, max_seq_len: int, block: int = 64, kv_dtype=ttnn.bfloat8_b,
                 prefill_chunk: int = 1024, slots: int = 1):
        mesh, N = w.mesh, w.N
        self.w, self.block, self.kv_dtype, self.prefill_chunk, self.B = w, block, kv_dtype, prefill_chunk, slots
        self.max_blocks = mb = math.ceil(max_seq_len / block)
        kv_shard = ttnn.ShardTensorToMesh(mesh, dim=-1) if N > 1 else None
        C = w.NKV * w.HD
        if w.sliding:
            self.k_conv = ShortConv(mesh, w.k_taps, kv_shard, C)
            self.v_conv = ShortConv(mesh, w.v_taps, kv_shard, C)
            self.n_ring = (w.W + prefill_chunk) // block + 1
            per = self.n_ring
        else:
            self.k_conv = DynShortConv(mesh, w.k_static, w.k_wproj, kv_shard, C, w.NKV, w.ckc)
            self.v_conv = DynShortConv(mesh, w.v_static, w.v_wproj, kv_shard, C, w.NKV, w.ckc)
            per = mb
        self.per_slot = per
        z = torch.zeros(per * slots, w.NKV * N, block, w.HD)
        mk = lambda: ttnn.from_torch(z, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                     memory_config=DRAM,
                                     mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1) if N > 1 else None)
        self.k_cache, self.v_cache = mk(), mk()
        self._mesh = mesh
        j = torch.arange(mb)
        if w.sliding:
            self.ring_period = self.n_ring * block
            read = torch.stack([b * per + j % per for b in range(slots)])  # [B, mb]
            write = torch.stack([b * per + torch.arange(per) for b in range(slots)])  # [B, n_ring]
        else:
            read = write = torch.stack([b * per + j for b in range(slots)])
        self.read_host = read
        self.read_table, self.write_table = self._up_i32(read), self._up_i32(write)  # [B, .] (batched decode)
        self.read_slot = [self._up_i32(read[b : b + 1]) for b in range(slots)]  # [1, .] per slot (one sequence)
        self.write_slot = [self._up_i32(write[b : b + 1]) for b in range(slots)]
        self._kv_shard = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, w.HD), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)

    def reset(self):
        self.k_conv.reset()
        self.v_conv.reset()

    def _up_i32(self, t):
        return ttnn.from_torch(t.to(torch.int32).contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self._mesh, memory_config=DRAM,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self._mesh) if self.w.N > 1 else None)


def _l2clip(x, thr, scale=None):
    """fp32 [.., HD]: x * scale (optional), then scaled down to L2 norm thr per row where longer:
    x * min(1, thr / ||x||) (transformers adds 1e-6 to the norm; immaterial at these norms)."""
    if scale is not None:
        x = ttnn.multiply(x, scale)
    ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
    if FUSED_OPS:
        return ttnn.multiply(x, ttnn.clamp(ttnn.multiply(ttnn.rsqrt(ss), thr), max=1.0))
    f = ttnn.clamp(ttnn.reciprocal(ttnn.add(ttnn.sqrt(ss), 1e-6)), max=1.0 / thr)  # min(1, thr/(n+eps)) / thr
    return ttnn.multiply(x, ttnn.multiply(f, thr))


def _rms_unweighted(x, eps):
    ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
    return ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ms, eps)))


def attention(w: AttnWeights, c: AttnCache, xq, xk, xv, p0: int, valid: int | None = None, tt_ccl=None, slot: int = 0):
    """One sequence, rows at positions p0 .. p0 + R - 1 (consecutive). xq / xk / xv [1, 1, R, H] bf16 (normed).
    R <= 32 // G: decode / verify rows (any p0); otherwise a prefill chunk (p0 % 64 == 0, R % 64 == 0; the first
    `valid` rows real). Returns [1, 1, R, H] bf16 (all-reduced over the mesh)."""
    NH, NKV, HD, DR, N = w.NH, w.NKV, w.HD, w.DR, w.N
    R = int(xq.shape[2])
    G = NH // NKV
    rows_mode = R * G <= 32
    if not rows_mode:
        assert p0 % c.block == 0 and R % 64 == 0 and R <= c.prefill_chunk, (p0, R)
    nvalid = R if valid is None else valid
    ckc = w.ckc
    pq = proj(w, "q", xq, ttnn.float32 if FUSED_OPS else None)  # [1, 1, R, fq(+pad)]
    pk = proj(w, "k", xk)
    pv = proj(w, "v", xv)
    if w.dram_sharded:  # drop the bank padding of k / v (the convs take exactly NKV * HD columns)
        pk = ttnn.slice(pk, (0, 0, 0, 0), (1, 1, R, w.kd))
        pv = ttnn.slice(pv, (0, 0, 0, 0), (1, 1, R, w.kd))
    o0 = 0
    q = ttnn.slice(pq, (0, 0, 0, o0), (1, 1, R, o0 + w.qd)); o0 += w.qd
    r = ttnn.slice(pq, (0, 0, 0, o0), (1, 1, R, o0 + w.rd)); o0 += w.rd
    s = ttnn.slice(pq, (0, 0, 0, o0), (1, 1, R, o0 + NH)); o0 += w.sd
    g = ttnn.slice(pq, (0, 0, 0, o0), (1, 1, R, o0 + w.gd))
    ttnn.deallocate(pq)
    # convs
    if w.sliding:
        if rows_mode:
            k = c.k_conv.rows(pk, commit_all=True)
            v = c.v_conv.rows(pv, commit_all=True)
        else:
            k = c.k_conv.prefill_commit(pk, nvalid)
            v = c.v_conv.prefill_commit(pv, nvalid)
    else:
        k = c.k_conv.rows(pk, xk, commit_n=nvalid)
        v = c.v_conv.rows(pv, xv, commit_n=nvalid)
    ttnn.deallocate(pk)
    ttnn.deallocate(pv)
    # per-head norms (fp32)
    f32 = lambda t: ttnn.typecast(t, ttnn.float32)  # noqa: E731
    q = _l2clip(f32(ttnn.reshape(q, (1, R, NH, HD))), w.qk_clip, w.q_scale)
    r = _l2clip(f32(ttnn.reshape(r, (1, R, NH, DR))), w.rel_clip, w.r_scale)
    k = _l2clip(f32(ttnn.reshape(k, (1, R, NKV, HD))), w.qk_clip)
    v = _rms_unweighted(f32(ttnn.reshape(v, (1, R, NKV, HD))), w.eps)
    if not w.sliding and p0 + R > w.n_floor - 1:  # log scaling (exact host tau; a no-op before n_floor)
        tau = tau_of(torch.arange(p0, p0 + R), w.n_floor, w.alpha).reshape(1, R, 1, 1)
        tau_t = ttnn.from_torch(tau, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=w.mesh,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(w.mesh) if N > 1 else None)
        q = ttnn.multiply(q, tau_t)
        r = ttnn.multiply(r, tau_t)
    bf = lambda t: ttnn.typecast(t, ttnn.bfloat16)  # noqa: E731
    q, k, v = bf(q), bf(k), bf(v)
    # KV write
    if rows_mode:
        for j in range(R):
            pos = c._up_i32(torch.tensor([p0 + j]))
            wpos = ttnn.remainder(pos, c.ring_period) if w.sliding else pos
            for cache, t in ((c.k_cache, k), (c.v_cache, v)):
                row = t if R == 1 else ttnn.slice(t, (0, j, 0, 0), (1, j + 1, NKV, HD))
                tp = ttnn.pad(row, [1, 1, 32, HD], [0, 0, 0, 0], 0.0)
                tsh = ttnn.to_memory_config(tp, c._kv_shard)
                ttnn.experimental.paged_update_cache(cache, tsh, update_idxs_tensor=wpos, page_table=c.write_slot[slot])
                ttnn.deallocate(tsh)
                ttnn.deallocate(tp)
    else:
        b0, nb = p0 // c.block, R // c.block
        fill_pt = c._up_i32(c.read_host[slot, b0 : b0 + nb].reshape(1, -1))
        for cache, t in ((c.k_cache, k), (c.v_cache, v)):
            tt_ = ttnn.transpose(t, 1, 2)  # [1, NKV, R, HD]
            t8 = ttnn.typecast(tt_, c.kv_dtype) if c.kv_dtype != tt_.dtype else tt_
            ttnn.experimental.paged_fill_cache(cache, t8, fill_pt, batch_idx=0)
            ttnn.deallocate(t8)
        ttnn.deallocate(fill_pt)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    # SDPA inputs
    rq = ttnn.pad(bf(r), [1, R, NH, 32], [0, 0, 0, 0], 0.0)
    sk = ttnn.multiply(f32(ttnn.reshape(s, (1, R, NH, 1))), w.sink_bcast)  # [1, R, NH, 32]
    sk = bf(sk)
    if rows_mode:
        positions = c._up_i32(torch.arange(p0, p0 + R).reshape(1, R) if R > 1 else torch.tensor([p0]))
    else:
        positions = c._up_i32(torch.arange(p0, p0 + R).reshape(-1, R_ITEM))
    o = inkling_sdpa_decode(q, c.k_cache, c.v_cache, positions, c.read_slot[slot], rq, w.table, w.scale, w.W, w.E,
                            sink=sk, out_memory_config=DRAM)
    for t in (q, rq, sk, positions):
        ttnn.deallocate(t)
    o = ttnn.reshape(o, (1, 1, R, NH * HD))
    gate = ttnn.sigmoid(f32(g))
    o = bf(ttnn.multiply(f32(o), gate))
    out = proj(w, "o", o)
    if w.dram_sharded and int(out.shape[-1]) != w.H:
        out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, R, w.H))
    if N > 1:
        out = all_reduce(out, w.mesh, tt_ccl)
    return out
