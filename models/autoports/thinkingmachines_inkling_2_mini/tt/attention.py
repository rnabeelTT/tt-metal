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
import os

import torch

import ttnn

from .dmm import dmm
from models.autoports.thinkingmachines_inkling_2_mini.tt.ccl import all_reduce
from models.autoports.thinkingmachines_inkling_2_mini.tt import relbias
from models.autoports.thinkingmachines_inkling_2_mini.tt.attn.inkling_sdpa import PAIR as SDPA_PAIR, inkling_sdpa_decode
from models.autoports.thinkingmachines_inkling_2_mini.tt.sconv import DynShortConv, ShortConv

DRAM = ttnn.DRAM_MEMORY_CONFIG
FUSED_OPS = __import__("os").environ.get("I2M_OPS", "fused") == "fused"
# I2M_DRAM_SHARDED (default 1): the attention projections as DRAM-width-sharded weights (qwen36 tp_common
# sharded_decode_matmul: the DRAM-sharded kernel for decode rows, a 2D matmul for prefill)
# (measured: at one row these projections are compute-bound on the 32-row tile; DRAM-sharded only wins at LoFi without
# fp32 accumulation, 45 vs 60 us; at HiFi2 it is slower, 67-69 us: off by default)
DRAM_SHARDED = __import__("os").environ.get("I2M_DRAM_SHARDED", "0") == "1"
# I2M_MSTREAM (default 1): the decode q | r | s | g, k and v projections in one program, each interleaved weight
# streamed per DRAM bank in place (tt/mstream.py; the q block zero-padded to a multiple of 8 tile columns)
MSTREAM = __import__("os").environ.get("I2M_MSTREAM", "1") == "1"
# I2M_ATTN_HIFI4 (default 1): the attention projections at HiFi4 on bfp8 weights too (HiFi2 drops activation mantissa
# bits: q vs the exact product mean |err| 2.7e-3 -> 2.1e-4; prefill eval KL 0.0139 -> 0.0123, decode 0.0184 -> 0.0157;
# one die +0.7 ms per step)
ATTN_HIFI4 = __import__("os").environ.get("I2M_ATTN_HIFI4", "1") == "1"
SQRT5 = math.sqrt(5.0)
R_ITEM = 8  # prefill: rows per SDPA work item (R * G <= 32 at G = 3)
# prefill SDPA items per work unit (inkling_sdpa_decode mi; I2M_SDPA_MI): each K / V chunk read once for 4 items' Q tiles
# (128K positions: the SDPA 377 -> 111 ms per 1024-row chunk; 8 items: the CBs clash with L1)
SDPA_MI = int(os.environ.get("I2M_SDPA_MI", "4"))
# I2M_RING_CHUNK (0: the prefill chunk): the sliding layers' rings hold the window + this many rows; a longer prefill
# chunk runs those layers' KV writes + SDPA in sub-chunks of this size (the MoE / projections still see the whole chunk:
# one pass over the experts per chunk), so one die keeps 256-row rings beside 8 slots with 1024-row chunks
RING_CHUNK = int(os.environ.get("I2M_RING_CHUNK", "0"))


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
            if MSTREAM and not DRAM_SHARDED and self.fq % 256:  # (zero columns: q | r | s | g, then the padding)
                cols[-1] = torch.cat([cols[-1], torch.zeros((-self.fq) % 256, wq.shape[1])], 0)
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
        self.ms = None
        if MSTREAM and not DRAM_SHARDED:  # (the interleaved weights themselves, streamed per bank: no copies)
            from .mstream import TP4 as MS_TP4
            from .mstream import MultiStream

            K, fqp = wq.shape[1], self.fq + (-self.fq) % 256
            # q readers per bank: 2 on one die (28 q columns per bank; 3 / 4: 22.4 / 23.1 vs 21.9 ms), 4 on more dies
            # (TP2 17.6 -> 17.2 ms, TP4 14.0 -> 13.9)
            qr = int(__import__("os").environ.get("I2M_MS_QREADERS", "4" if N > 1 else "2"))
            # I2M_MS_KV (default 1): k and v on one core per bank (one die: 2 q cores of 13 columns + one k | v core of 8
            # per bank, 64.5 -> 57.0 us per call with tt/mstream.py knobs(); TP2: 3 q + 1 k | v, 42.6 -> 37.3; TP4 (with
            # I2M_MS_TP4): 4 q cores of 2 / 2 / 2 / 1 columns + one k | v core of 2, traced one die at TP4's shapes 28.0 ->
            # 15.3 us: the separate k and v parts' extra kernels cost ~9 us of program launch per call)
            kv = __import__("os").environ.get("I2M_MS_KV", "1" if N <= 2 or MS_TP4 else "0") == "1"
            if kv:
                qr = int(__import__("os").environ.get("I2M_MS_QREADERS", "2" if N == 1 else "3" if N == 2 else "4"))
            self.ms = MultiStream(mesh, K, [(fqp, ttnn.float32, qr), (self.kd, ttnn.bfloat16, 1),
                                            (self.kd, ttnn.bfloat16, 1)], wdtype, interleaved=True,
                                  fidelity=ttnn.MathFidelity.HiFi4 if ATTN_HIFI4 else ttnn.MathFidelity.HiFi2,
                                  groups=[[0], [1, 2]] if kv else None)
            self.ms_w = [self.w_q, self.w_k, self.w_v]
        self.pf = self.opf = None
        if self.ms is not None:  # (I2M_PREFETCH: q streamed ahead into a global CB, tt/prefetch.py)
            from . import prefetch as PF

            if PF.ON:
                self.pf = PF.qkv_prefetch(mesh, wq.shape[1], fqp, self.kd,
                                          ttnn.MathFidelity.HiFi4 if ATTN_HIFI4 else ttnn.MathFidelity.HiFi2)
                self.opf = PF.o_prefetch(mesh, int(self.w_o.shape[-2]), int(self.w_o.shape[-1]),
                                         ttnn.MathFidelity.HiFi4 if ATTN_HIFI4 or wdtype == ttnn.bfloat16
                                         else ttnn.MathFidelity.HiFi2) if self.pf is not None else None
        # per-head scales (fp32): q [NH, HD], r [NH, DR] (head-sharded), as [1, 1, NH, HD] rows
        hs = lambda t, n: ttnn.from_torch(t.float().reshape(1, 1, n * N, -1), dtype=ttnn.float32,  # noqa: E731
                                          layout=ttnn.TILE_LAYOUT, device=mesh,
                                          mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2) if N > 1 else None)
        self.q_scale = None if FUSED_OPS else hs(L("q_l2_clip.scale"), NH)
        self.r_scale = None if FUSED_OPS else hs(L("rel_l2_clip.scale"), NH)
        T = relbias.build_tables(L("rel_logits_proj.proj").float(), self.scale)
        self.table = ttnn.as_tensor(T.reshape(1, 32, 32, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=mesh, memory_config=DRAM, mesh_mapper=rep)
        self.table2 = None  # (prefill: the row-pair tables, inkling_sdpa_decode table2)
        if SDPA_PAIR:
            self.table2 = ttnn.as_tensor(relbias.build_pair_tables(T).reshape(1, 32, 32, -1), dtype=ttnn.bfloat16,
                                         layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=rep)
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
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4 if wdtype == ttnn.bfloat16 or ATTN_HIFI4 else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)


# I2M_PF_PROJ_CFG (default 1): prefill rows (> 32) of the attention projections on an explicit 2D multicast config over
# the whole grid (pf_progcfg) instead of ttnn's default (tools/bench_pfproj.py, one die, 2048 rows, us: TP4 q 240 -> 136,
# k / v 99 -> 62, o 128 -> 73; TP2 q 395 -> 249, k / v 110 -> 65, o 215 -> 109)
PF_PROJ_CFG = __import__("os").environ.get("I2M_PF_PROJ_CFG", "1") == "1"
_PFCFG = {}


PF_PROJ_WHICH = __import__("os").environ.get("I2M_PF_PROJ_WHICH", "qkvo")
PF_PROJ_L1 = int(__import__("os").environ.get("I2M_PF_PROJ_L1", str(1100 << 10)))


def pf_progcfg(grid, M, K, N, wbytes=1088, obytes=2048):
    """2D multicast config for x [M, K] @ w [K, N] (M > 32): the grid shape with the fewest output tiles per core and no
    idle core row / column, K blocks of up to 8 tiles, the largest output subblock (<= 4 tiles: fp32 DST). None when
    its circular buffers (in0 / in1 double-buffered, the fp32 partials and the output) pass PF_PROJ_L1 bytes (TP1's
    q: ttnn's default config then)."""
    key = (grid.x, grid.y, M, K, N, wbytes, obytes)
    if key not in _PFCFG:
        Mt, Nt, Kt = -(-M // 32), -(-N // 32), K // 32
        best = None
        for gx in range(grid.x, 0, -1):
            for gy in range(grid.y, 0, -1):
                pm, pn = -(-Mt // gy), -(-Nt // gx)
                if pn * gx - Nt >= pn or pm * gy - Mt >= pm:
                    continue
                c = (pm * pn, -(gx * gy))
                if best is None or c < best[0]:
                    best = (c, gx, gy, pm, pn)
        _, gx, gy, pm, pn = best
        bw = max(d for d in range(1, 9) if Kt % d == 0)
        if 2 * pm * bw * 2048 + 2 * bw * pn * wbytes + pm * pn * (4096 + obytes) > PF_PROJ_L1:
            _PFCFG[key] = None
            return None
        sbs = [(h, ww) for h in (1, 2, 4) for ww in (1, 2, 4) if h * ww <= 4 and pn % ww == 0 and pm % h == 0]
        sh, sw = max(sbs, key=lambda s_: (s_[0] * s_[1], s_[1]))
        _PFCFG[key] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=sh, out_subblock_w=sw,
            per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None, fuse_batch=False)
    return _PFCFG[key]


def proj(w: "AttnWeights", which: str, x, dtype=None):
    """One attention projection (q | k | v | o) on x [1, 1, M, K] -> [1, 1, M, n(+pad)] in DRAM."""
    W = getattr(w, "w_" + which)
    M = int(x.shape[-2])
    if not w.dram_sharded and PF_PROJ_CFG and which in PF_PROJ_WHICH and M > 32 and int(x.shape[1]) == 1 \
            and not x.is_sharded():
        K, N = int(x.shape[-1]), int(W.shape[-1])
        for b in (1, 2, 4):  # (b > 1: the rows as b batches of M / b, the weight broadcast: smaller per-core blocks)
            if M % (32 * b):
                break
            cfg = pf_progcfg(w.mesh.compute_with_storage_grid_size(), M // b, K, N,
                             wbytes={ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}.get(W.dtype, 2048),
                             obytes=4096 if dtype == ttnn.float32 else 2048)
            if cfg is not None:
                kw = {} if dtype is None else {"dtype": dtype}
                xb = x if b == 1 else ttnn.reshape(x, (1, b, M // b, K))
                o = ttnn.linear(xb, W, compute_kernel_config=w.ckc, memory_config=DRAM, program_config=cfg, **kw)
                return o if b == 1 else ttnn.reshape(o, (1, 1, M, N))
    if not w.dram_sharded:
        return dmm(x, W, w.ckc, dtype=dtype, memory_config=DRAM, linear=True)
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    pc, act, K = getattr(w, "pc_" + which)
    return tpc.sharded_decode_matmul(x, W, w.ckc, pc, act, tpc.create_prefill_matmul_program_config, K,
                                     decode_out_memory_config=DRAM, out_dtype=dtype)


_TRUST_ALL = __import__("os").environ.get("I2M_PAGES_TRUST_ALL", "0") == "1"  # (debug: the pre-fix behaviour)


class GlobalPages:
    """Block tables of the global-attention layers, shared by every global application (one logical block maps to the
    same block id in each global layer's pool).

    Engine-owned (default): slot b owns blocks [b * mb, (b + 1) * mb). vLLM-paged serving (``set_row``): a slot's row
    is its request's vLLM block table (vLLM's prefix cache then shares the prompt blocks between requests); entries
    vLLM has not allocated (0) and every row of a free slot map to scratch blocks past vLLM's pool, so padded prefill
    rows and idle decode rows never write into a block vLLM may have handed to another request. Idle decode rows
    (position 0 of a free slot's row) write into their own block past the scratch blocks: their K / V are junk (inf)
    and the scratch blocks are read (masked) past a request's allocated blocks."""

    def __init__(self, mesh, B: int, mb: int, scratch: int = 8, pool_blocks: int = None):
        """pool_blocks: the KV pool (vLLM's block budget, shared by the slots; default B * mb, every slot a full
        context). Smaller than B * mb only with vLLM paging (the slots then start free)."""
        self.mesh, self.B, self.mb, self.scratch = mesh, B, mb, scratch
        pool = int(pool_blocks) if pool_blocks else B * mb
        self.scratch_base = pool
        self.idle_block = pool + scratch
        self.n_blocks = pool + scratch + 1
        self.rep = ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None
        j = torch.arange(mb)
        if pool >= B * mb:
            self.host = torch.stack([b * mb + j for b in range(B)]).to(torch.int32)  # [B, mb]
        else:  # (shared pool: free slots on the scratch blocks until vLLM assigns rows)
            self.host = torch.stack([pool + j % scratch for _ in range(B)]).to(torch.int32)
        self.slot_dev = [self._up(self.host[b : b + 1]) for b in range(B)]  # [1, mb] per slot (prefill)
        self.width_dev = {}  # W -> [W, mb] (batched decode of slots 0 .. W-1); built before any trace capture
        self.dirty = set()
        self.paged = False

    def _up(self, t):
        return ttnn.from_torch(t.to(torch.int32).contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.mesh, memory_config=DRAM, mesh_mapper=self.rep)

    def table(self, W):
        """W: a width (slots 0 .. W-1) or a tuple of slots, one per row (speculative verify rows)."""
        if W not in self.width_dev:
            self.width_dev[W] = self._up(self._rows(W))
        return self.width_dev[W]

    def _rows(self, W):
        return self.host[list(W)] if isinstance(W, tuple) else self.host[:W]

    def _scratch_row(self):
        return (self.scratch_base + torch.arange(self.mb) % self.scratch).to(torch.int32)

    def enable_vllm(self, num_blocks: int):
        """vLLM pages the global KV: its block ids [0, num_blocks) index this pool; every slot starts free."""
        assert num_blocks <= self.scratch_base, (num_blocks, self.scratch_base)
        self.paged = True
        for b in range(self.B):
            self.free(b)

    def set_row(self, b: int, row, n_valid: int) -> bool:
        """Slot b now serves the request whose vLLM block table row is `row`, of which the first n_valid entries are
        its blocks (vLLM does not clear a reused row past a request's allocation: the stale entries can name blocks
        vLLM has since given to another request, or to this one). Returns True if the row changed."""
        if not self.paged or row is None:
            return False
        r = torch.as_tensor(row).reshape(-1).to(torch.int32)
        m = self._scratch_row()
        k = min(self.mb, int(r.numel()), int(n_valid) if not _TRUST_ALL else self.mb)
        m[:k] = torch.where(r[:k] > 0, r[:k], m[:k])
        return self._put(b, m)

    def free(self, b: int):
        if self.paged:
            m = self._scratch_row()
            m[0] = self.idle_block
            self._put(b, m)

    def _put(self, b, m) -> bool:
        if torch.equal(self.host[b], m):
            return False
        self.host[b] = m
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(m.reshape(1, -1).contiguous(), dtype=ttnn.int32,
                                                        layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep),
                                        self.slot_dev[b])
        self.dirty.update(w for w in self.width_dev if (b in w if isinstance(w, tuple) else b < w))
        return True

    def sync(self, W: int):
        """Before a decode step of width W: its device table catches up with the host rows."""
        if W in self.dirty:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(self._rows(W).contiguous(), dtype=ttnn.int32,
                                                            layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep),
                                            self.width_dev[W])
            self.dirty.discard(W)


class AttnCache:
    """K / V caches of one application (layer, pass) for B sequence slots, plus the conv states of a single-sequence
    "lane" (prefill / verify rows run one slot at a time on the lane; batched decode keeps its conv states in
    tt/batched.py and copies the lane's states in after a prefill).

    Global layers: a paged pool of B * max_blocks blocks, slot b owning blocks [b * mb, (b + 1) * mb). Sliding layers:
    one ring of n_ring blocks per slot (the window + one prefill chunk + a block), slot b's logical block j at physical
    b * n_ring + j % n_ring; writes go through an identity table per slot at position pos mod (n_ring * block)."""

    def __init__(self, w: AttnWeights, max_seq_len: int, block: int = 64, kv_dtype=ttnn.bfloat8_b,
                 prefill_chunk: int = 1024, slots: int = 1, pages: GlobalPages = None):
        mesh, N = w.mesh, w.N
        self.pages = None if w.sliding else pages
        self.w, self.block, self.kv_dtype, self.prefill_chunk, self.B = w, block, kv_dtype, prefill_chunk, slots
        self.ring_chunk = min(prefill_chunk, RING_CHUNK) if RING_CHUNK else prefill_chunk
        self.max_blocks = mb = math.ceil(max_seq_len / block)
        kv_shard = ttnn.ShardTensorToMesh(mesh, dim=-1) if N > 1 else None
        C = w.NKV * w.HD
        if w.sliding:
            self.k_conv = ShortConv(mesh, w.k_taps, kv_shard, C)
            self.v_conv = ShortConv(mesh, w.v_taps, kv_shard, C)
            self.n_ring = (w.W + self.ring_chunk) // block + 1
            per = self.n_ring
        else:
            self.k_conv = DynShortConv(mesh, w.k_static, w.k_wproj, kv_shard, C, w.NKV, w.ckc)
            self.v_conv = DynShortConv(mesh, w.v_static, w.v_wproj, kv_shard, C, w.NKV, w.ckc)
            per = mb
        self.per_slot = per
        # (sliding: + one scratch block, where idle decode rows write: see the write table below)
        nblk = per * slots + int(w.sliding) if self.pages is None else self.pages.n_blocks
        z = torch.zeros(nblk, w.NKV * N, block, w.HD)
        mk = lambda: ttnn.from_torch(z, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                     memory_config=DRAM,
                                     mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1) if N > 1 else None)
        self.k_cache, self.v_cache = mk(), mk()
        self._mesh = mesh
        j = torch.arange(mb)
        if w.sliding:
            self.ring_period = self.n_ring * block
            read = torch.stack([b * per + j % per for b in range(slots)])  # [B, mb]
            # [B, n_ring + 1]: column n_ring is the scratch block, the write position of an idle decode row
            # (position ring_period): its K / V are junk (inf from the loop passes) and must not reach a slot's ring,
            # which a prefix-cache restore does not rewrite outside the window
            write = torch.stack([torch.cat([b * per + torch.arange(per), torch.tensor([per * slots])])
                                 for b in range(slots)])
        else:
            read = write = torch.stack([b * per + j for b in range(slots)])
        if self.pages is not None:  # the shared global tables (updated in place)
            pg = self.pages
            assert pg.mb == mb and pg.B == slots, (pg.mb, mb, pg.B, slots)
            self.read_host = self.write_host = pg.host
            self.read_table = self.write_table = pg.table(slots)
            self.read_slot = self.write_slot = pg.slot_dev
        else:
            self.read_host, self.write_host = read, write
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


# I2M_PREFILL_PREP (default 1): a prefill chunk's q / r / sink prep and K / V cache writes in one program (tt/aprep.py
# with the chunk's rows as its units): the slices, l2 clips, RMS, tau, casts, transposes and paged_fill_cache
PREFILL_PREP = __import__("os").environ.get("I2M_PREFILL_PREP", "1") == "1"
# I2M_PGATE (default 1): the packed prefill's output gate in one program reading the gate columns of the fp32 q
# projection output (tt/agate.py PrefillGate) instead of a slice, typecasts, sigmoid and multiply; bitwise
PGATE = __import__("os").environ.get("I2M_PGATE", "1") == "1"
_PGATE = {}
# I2M_PF_TAU1 (default 1): a packed chunk's log-scaling rows uploaded once per chunk (PackedCtx.tau), not per global
# application (8 x a [1, T, 1, 1] TILE fp32 host tilize + upload: ~4.7 ms each at 2048 rows); the same values
TAU1 = __import__("os").environ.get("I2M_PF_TAU1", "1") == "1"
_PREP = {}


def _tau_tensor(w, tau_rows):
    """the rows' log scaling [1, T, 1, 1] fp32 TILE (one tile per row: 8 MB of host tilize + upload at 2048 rows)"""
    T = int(tau_rows.numel())
    return ttnn.from_torch(tau_rows.reshape(1, T, 1, 1).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=w.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(w.mesh) if w.N > 1 else None)


def _prefill_prep(w, c, pq, k, v, tau_rows, fill):
    """-> (q [1, T, NH, HD], rq [1, T, NH, 32], sk [1, T, NH, 32]) bf16; the K / V rows written into the caches."""
    from .aprep import AttnPrep

    T = int(pq.shape[-2])
    key = (id(w), T)
    if key not in _PREP:
        _PREP[key] = AttnPrep(w, T, None)
    tau_t, own = None, False
    if not w.sliding and isinstance(tau_rows, torch.Tensor):  # (the log scaling: 1 before n_floor)
        tau_t, own = _tau_tensor(w, tau_rows), True
    elif not w.sliding:  # (a packed chunk's: built once per chunk, PackedCtx.tau)
        tau_t = tau_rows
    q, rq, _, _, sk = _PREP[key](pq, k, v, tau_t, fill=(c.k_cache, c.v_cache, fill))
    if own:
        ttnn.deallocate(tau_t)
    return q, rq, sk


def attention(w: AttnWeights, c: AttnCache, xq, xk, xv, p0: int, valid: int | None = None, tt_ccl=None, slot: int = 0):
    """One sequence, rows at positions p0 .. p0 + R - 1 (consecutive). xq / xk / xv [1, 1, R, H] bf16 (normed).
    R <= 32 // G: decode / verify rows (any p0); otherwise a prefill chunk (p0 % 64 == 0, R % 64 == 0; the first
    `valid` rows real). Returns [1, 1, R, H] bf16 (all-reduced over the mesh)."""
    NH, NKV, HD, DR, N = w.NH, w.NKV, w.HD, w.DR, w.N
    R = int(xq.shape[2])
    G = NH // NKV
    rows_mode = R * G <= 32
    if not rows_mode:
        assert p0 % c.block == 0 and R % 64 == 0 and R <= (c.ring_chunk if w.sliding else c.prefill_chunk), (p0, R)
    nvalid = R if valid is None else valid
    ckc = w.ckc
    pq = proj(w, "q", xq, ttnn.float32 if FUSED_OPS else None)  # [1, 1, R, fq(+pad)]
    pk = proj(w, "k", xk)
    pv = proj(w, "v", xv)
    if w.dram_sharded:  # drop the bank padding of k / v (the convs take exactly NKV * HD columns)
        pk = ttnn.slice(pk, (0, 0, 0, 0), (1, 1, R, w.kd))
        pv = ttnn.slice(pv, (0, 0, 0, 0), (1, 1, R, w.kd))
    pq_full = pq if (PREFILL_PREP and not rows_mode and R > 32 and c.kv_dtype == ttnn.bfloat16) else None
    if pq_full is None:  # (the prefill prep reads q / r / sink from pq itself)
        q = ttnn.slice(pq, (0, 0, 0, 0), (1, 1, R, w.qd))
        r = ttnn.slice(pq, (0, 0, 0, w.qd), (1, 1, R, w.qd + w.rd))
        s = ttnn.slice(pq, (0, 0, 0, w.qd + w.rd), (1, 1, R, w.qd + w.rd + NH))
    o0 = w.qd + w.rd + w.sd
    g = ttnn.slice(pq, (0, 0, 0, o0), (1, 1, R, o0 + w.gd))
    if pq_full is None:
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
    f32 = lambda t: ttnn.typecast(t, ttnn.float32)  # noqa: E731
    bf = lambda t: ttnn.typecast(t, ttnn.bfloat16)  # noqa: E731
    if pq_full is not None:
        b0, nb = p0 // c.block, R // c.block
        fill_pt = c._up_i32(c.read_host[slot, b0 : b0 + nb].reshape(1, -1))
        tau_rows = tau_of(torch.arange(p0, p0 + R), w.n_floor, w.alpha)
        q, rq, sk = _prefill_prep(w, c, pq_full, k, v, tau_rows, fill_pt)
        ttnn.deallocate(pq_full)
        ttnn.deallocate(fill_pt)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        return _attn_tail(w, c, q, rq, sk, g, p0, R, slot, rows_mode, tt_ccl)
    # per-head norms (fp32)
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
    return _attn_tail(w, c, q, rq, sk, g, p0, R, slot, rows_mode, tt_ccl)


def _attn_tail(w, c, q, rq, sk, g, p0, R, slot, rows_mode, tt_ccl):
    """attention()'s SDPA, output gate, o_proj and all-reduce."""
    NH, HD, N = w.NH, w.HD, w.N
    f32 = lambda t: ttnn.typecast(t, ttnn.float32)  # noqa: E731
    bf = lambda t: ttnn.typecast(t, ttnn.bfloat16)  # noqa: E731
    if rows_mode:
        positions = c._up_i32(torch.arange(p0, p0 + R).reshape(1, R) if R > 1 else torch.tensor([p0]))
    else:
        positions = c._up_i32(torch.arange(p0, p0 + R).reshape(-1, R_ITEM))
    o = inkling_sdpa_decode(q, c.k_cache, c.v_cache, positions, c.read_slot[slot], rq, w.table, w.scale, w.W, w.E,
                            sink=sk, out_memory_config=DRAM, mi=SDPA_MI, table2=w.table2)
    for t in (q, rq, sk, positions):
        ttnn.deallocate(t)
    if int(o.shape[-1]) != NH * HD:  # (the SDPA writes [1, 1, rows, NH * HD] directly when it can)
        o = ttnn.reshape(o, (1, 1, R, NH * HD))
    gate = ttnn.sigmoid(f32(g))
    o = bf(ttnn.multiply(f32(o), gate))
    out = proj(w, "o", o)
    if w.dram_sharded and int(out.shape[-1]) != w.H:
        out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, R, w.H))
    if N > 1:
        out = all_reduce(out, w.mesh, tt_ccl)
    return out


class PackedCtx:
    """One packed prefill chunk: several sequences' rows stacked (segments), each segment 64-row aligned.

    segs: [(row0, R, p0, valid, slot, lane)] (R % 64 == 0, p0 % 64 == 0). Built once per chunk: the SDPA's per-item
    positions and page-table rows (one 8-row item never straddles two segments), shared by every application."""

    dyn = None  # (StaticCtx: the traced chunk's per-call words)

    def __init__(self, mesh, segs, sliding_cache: "AttnCache", global_cache: "AttnCache"):
        self.segs = segs
        rep = ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None
        up = lambda t: ttnn.from_torch(t.to(torch.int32).contiguous(), dtype=ttnn.int32,  # noqa: E731
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=rep)
        from .blkcopy import BlockCopy

        pos, slot_of = [], []
        for row0, R, p0, valid, slot, lane in segs:
            assert R % 64 == 0 and p0 % 64 == 0 and row0 % 64 == 0, (row0, R, p0)
            pos.append(torch.arange(p0, p0 + R))
            slot_of += [slot] * (R // R_ITEM)
        self.T = sum(s[1] for s in segs)
        self._up = up
        self.pos_h = torch.cat(pos).reshape(-1, R_ITEM)
        self.positions = up(self.pos_h)
        self.pt, self.fill, self.pt_h, self.fill_h, self._subs, self._taus = {}, {}, {}, {}, {}, {}
        for c in (sliding_cache, global_cache):
            if c is not None:
                self.pt_h[c.w.sliding] = c.read_host[slot_of]
                self.pt[c.w.sliding] = up(self.pt_h[c.w.sliding])
                # one KV fill per chunk: the physical block of each 64-row block of the packed rows
                blk = [c.read_host[slot, p0 // 64 + j] for _, R, p0, _, slot, _ in segs for j in range(R // 64)]
                self.fill_h[c.w.sliding] = torch.stack(blk).reshape(1, -1)
                self.fill[c.w.sliding] = up(self.fill_h[c.w.sliding])
        self.p_hi = max(p0 + R for _, R, p0, _, _, _ in segs)
        self.bc = BlockCopy(mesh, sliding_cache.k_cache)

    def sub(self, sliding, r0, r1):
        """(positions, page-table rows, fill) of packed rows [r0, r1) (64-row aligned): a sub-chunk's SDPA / KV fill."""
        key = (sliding, r0, r1)
        if key not in self._subs:
            self._subs[key] = (self._up(self.pos_h[r0 // R_ITEM : r1 // R_ITEM]),
                               self._up(self.pt_h[sliding][r0 // R_ITEM : r1 // R_ITEM]),
                               self._up(self.fill_h[sliding][:, r0 // 64 : r1 // 64]))
        return self._subs[key]

    def tau(self, w):
        """the chunk's rows' log scaling (global layers) as the prep reads it: one upload per chunk (I2M_PF_TAU1), not
        one per global application"""
        key = (w.n_floor, w.alpha)
        if not TAU1:
            self._taus_once = getattr(self, "_taus_once", []) + [None]
        elif key in self._taus:
            return self._taus[key]
        tau = torch.cat([tau_of(torch.arange(p0, p0 + R), w.n_floor, w.alpha) for _, R, p0, _, _, _ in self.segs])
        t = _tau_tensor(w, tau)
        if TAU1:
            self._taus[key] = t
        else:
            self._taus_once[-1] = t
        return t

    def release(self):
        for t in [self.positions] + list(self.pt.values()) + list(self.fill.values()) \
                + [t for s in self._subs.values() for t in s] + list(self._taus.values()) \
                + list(getattr(self, "_taus_once", [])):
            ttnn.deallocate(t)


class StaticCtx(PackedCtx):
    """PackedCtx of a traced one-segment chunk of T rows (tt/model.py PrefillTrace): every per-call input (positions,
    page-table rows, KV fill blocks, log scaling, and the dyn words: the segment's valid / snapshot / last rows) lives
    in a device tensor allocated before any trace capture and rewritten from the host (copy_host_to_device_tensor)
    by set() before each replay. The same programs as the eager chunk, so the same results bit for bit.

    dyn (uint32 [1, 1, 1, 32]): 0 n (valid rows), 1 snapshot row or NONE (the pfseg conv kernel's words), 16 snapshot
    row or n (DynShortConv's always-run snapshot copy), 17 last row, 18 / 19 its RowSplit die-local row / gather row."""

    NONE = 0xFFFFFFFF

    def __init__(self, mesh, T, sliding_cache: "AttnCache", global_cache: "AttnCache", Tl=None):
        from .blkcopy import BlockCopy

        self.mesh, self.T, self.Tl = mesh, T, Tl
        self.rep = ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None
        self.caches = [c for c in (sliding_cache, global_cache) if c is not None]
        self.segs = [(0, T, 0, T, 0, 0)]
        self.pos_h = torch.arange(T).reshape(-1, R_ITEM)
        self.positions = self._alloc(self.pos_h)
        self.pt_h, self.fill_h, self.pt, self.fill, self._subs, self._taus = {}, {}, {}, {}, {}, {}
        for c in self.caches:
            self.pt_h[c.w.sliding] = c.read_host[[0] * (T // R_ITEM)].clone()
            self.pt[c.w.sliding] = self._alloc(self.pt_h[c.w.sliding])
            self.fill_h[c.w.sliding] = c.read_host[0, : T // 64].reshape(1, -1).clone()
            self.fill[c.w.sliding] = self._alloc(self.fill_h[c.w.sliding])
        self.gw = global_cache.w if global_cache is not None else None
        if self.gw is not None:
            self._tau_h = tau_of(torch.arange(T), self.gw.n_floor, self.gw.alpha)
            self._taus[(self.gw.n_floor, self.gw.alpha)] = _tau_tensor(self.gw, self._tau_h)
        self.dyn_h = torch.zeros(1, 1, 1, 32, dtype=torch.int64)
        self.dyn = ttnn.from_torch(self.dyn_h.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=mesh, memory_config=DRAM, mesh_mapper=self.rep)
        self.ids_t = ttnn.from_torch(torch.zeros(1, T, dtype=torch.int32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=self.rep)
        self.p_hi = T
        self.bc = BlockCopy(mesh, sliding_cache.k_cache)

    def _alloc(self, h):
        return ttnn.from_torch(h.to(torch.int32).contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.mesh, memory_config=DRAM, mesh_mapper=self.rep)

    def _write(self, h, dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(h.contiguous(), dtype=dtype, layout=layout,
                                                        mesh_mapper=self.rep), dev)

    @staticmethod
    def ok(seg, T, snap):
        """seg (row0, R, p0, valid, slot, lane) can run traced: one segment over the whole chunk on lane 0, the commit
        and snapshot rows past the state's (K - 1 = 3) rows."""
        row0, R, p0, valid, slot, lane = seg
        return row0 == 0 and R == T and lane == 0 and 3 <= valid <= T and (snap is None or 3 <= snap <= T)

    def set(self, ids, seg, snap=None):
        """the chunk's inputs: ids [T] (host), seg (0, T, p0, valid, slot, 0), snap (the armed snapshot row or None)"""
        row0, R, p0, valid, slot, lane = seg
        assert self.ok(seg, self.T, snap), (seg, snap)
        T = self.T
        self.segs = [tuple(seg)]
        self._write(ids.reshape(1, T).to(torch.int32), self.ids_t, dtype=ttnn.uint32)
        if int(self.pos_h[0, 0]) != p0:
            self.pos_h = torch.arange(p0, p0 + T).reshape(-1, R_ITEM)
            self._write(self.pos_h, self.positions)
            if self.gw is not None:
                tau = tau_of(torch.arange(p0, p0 + T), self.gw.n_floor, self.gw.alpha)
                if not torch.equal(tau, self._tau_h):
                    self._tau_h = tau
                    self._write(tau.reshape(1, T, 1, 1).float(), self._taus[(self.gw.n_floor, self.gw.alpha)],
                                dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        for c in self.caches:
            k = c.w.sliding
            pt = c.read_host[slot].reshape(1, -1).expand(T // R_ITEM, -1)
            if not torch.equal(pt, self.pt_h[k]):
                self.pt_h[k] = pt.clone()
                self._write(self.pt_h[k], self.pt[k])
            fill = c.read_host[slot, p0 // 64 : p0 // 64 + T // 64].reshape(1, -1)
            if not torch.equal(fill, self.fill_h[k]):
                self.fill_h[k] = fill.clone()
                self._write(self.fill_h[k], self.fill[k])
        last = valid - 1
        d = torch.zeros(1, 1, 1, 32, dtype=torch.int64)
        d[..., 0], d[..., 1] = valid, self.NONE if snap is None else snap
        d[..., 16], d[..., 17] = valid if snap is None else snap, last
        if self.Tl:
            d[..., 18], d[..., 19] = last % self.Tl, (last // self.Tl) * 32
        if not torch.equal(d, self.dyn_h):
            self.dyn_h = d
            self._write(torch.where(d >= 1 << 31, d - (1 << 32), d).to(torch.int32), self.dyn, dtype=ttnn.uint32)

    def sub(self, sliding, r0, r1):
        raise AssertionError("StaticCtx: chunks past the ring's chunk run eagerly")

    def tau(self, w):
        return self._taus[(w.n_floor, w.alpha)]

    def release(self):
        pass  # (persistent: kept across replays)


def attention_packed(w: AttnWeights, c: AttnCache, lanes, xq, xk, xv, ctx: PackedCtx, tt_ccl=None, scatter=None):
    """attention() for a packed prefill chunk (PackedCtx): the projections, norms, gate and o_proj on all rows at once;
    the K / V convs (per-sequence state: lanes[lane] = (k_conv, v_conv)) and the KV writes per segment; one SDPA over
    every segment (per-item page-table rows). xq / xk / xv [1, 1, T, H] bf16 -> [1, 1, T, H] bf16."""
    NH, NKV, HD, DR, N = w.NH, w.NKV, w.HD, w.DR, w.N
    T = int(xq.shape[2])
    pq = proj(w, "q", xq, ttnn.float32 if FUSED_OPS else None)
    pk = proj(w, "k", xk)
    pv = proj(w, "v", xv)
    if w.dram_sharded:
        pk = ttnn.slice(pk, (0, 0, 0, 0), (1, 1, T, w.kd))
        pv = ttnn.slice(pv, (0, 0, 0, 0), (1, 1, T, w.kd))
    prep = PREFILL_PREP and T > 32 and c.kv_dtype == ttnn.bfloat16
    o0 = 0
    if not prep:  # (the prep reads q / r / sink from pq itself)
        q = ttnn.slice(pq, (0, 0, 0, o0), (1, 1, T, o0 + w.qd))
        r = ttnn.slice(pq, (0, 0, 0, o0 + w.qd), (1, 1, T, o0 + w.qd + w.rd))
        s = ttnn.slice(pq, (0, 0, 0, o0 + w.qd + w.rd), (1, 1, T, o0 + w.qd + w.rd + NH))
    o0 = w.qd + w.rd + w.sd
    pg = PGATE and pq.dtype == ttnn.float32 and T % 32 == 0 and o0 % 32 == 0 and w.gd == NH * HD
    g = (pq, o0) if pg else ttnn.slice(pq, (0, 0, 0, o0), (1, 1, T, o0 + w.gd))  # (pg: pq lives until the gate)
    if not prep and not pg:
        ttnn.deallocate(pq)
    sk_ = [(row0, R, valid, lanes[lane][0]) for row0, R, p0, valid, slot, lane in ctx.segs]
    sv_ = [(row0, R, valid, lanes[lane][1]) for row0, R, p0, valid, slot, lane in ctx.segs]
    if w.sliding:
        k = c.k_conv.prefill_packed(pk, sk_, dyn=ctx.dyn)
        v = c.v_conv.prefill_packed(pv, sv_, dyn=ctx.dyn)
    else:
        k = c.k_conv.rows_packed(pk, xk, sk_, ctx.bc, dyn=ctx.dyn)
        v = c.v_conv.rows_packed(pv, xv, sv_, ctx.bc, dyn=ctx.dyn)
    ttnn.deallocate(pk)
    ttnn.deallocate(pv)
    f32 = lambda t: ttnn.typecast(t, ttnn.float32)  # noqa: E731
    bf = lambda t: ttnn.typecast(t, ttnn.bfloat16)  # noqa: E731
    if prep and w.sliding and T > c.ring_chunk:  # (the ring holds the window + ring_chunk rows: sub-chunks)
        S, os_ = c.ring_chunk, []
        rows = lambda t, r0, r1: ttnn.slice(t, (0, 0, r0, 0), (1, 1, r1, int(t.shape[-1])))  # noqa: E731
        for r0 in range(0, T, S):
            r1 = min(T, r0 + S)
            pos_s, pt_s, fill_s = ctx.sub(True, r0, r1)
            ps, ks, vs = rows(pq, r0, r1), rows(k, r0, r1), rows(v, r0, r1)
            q, rq, sk = _prefill_prep(w, c, ps, ks, vs, None, fill_s)
            for t in (ps, ks, vs):
                ttnn.deallocate(t)
            os_.append(_packed_sdpa(w, c, q, rq, sk, pos_s, pt_s, r1 - r0))
        for t in (k, v) if pg else (pq, k, v):
            ttnn.deallocate(t)
        return _packed_out(w, ttnn.concat(os_, dim=2), g, T, tt_ccl, scatter)
    if prep:  # the q / r / sink prep and the K / V writes in one program
        q, rq, sk = _prefill_prep(w, c, pq, k, v, None if w.sliding else ctx.tau(w), ctx.fill[w.sliding])
        for t in (k, v) if pg else (pq, k, v):
            ttnn.deallocate(t)
        return _packed_tail(w, c, q, rq, sk, g, ctx, T, tt_ccl, scatter)
    q = _l2clip(f32(ttnn.reshape(q, (1, T, NH, HD))), w.qk_clip, w.q_scale)
    r = _l2clip(f32(ttnn.reshape(r, (1, T, NH, DR))), w.rel_clip, w.r_scale)
    k = _l2clip(f32(ttnn.reshape(k, (1, T, NKV, HD))), w.qk_clip)
    v = _rms_unweighted(f32(ttnn.reshape(v, (1, T, NKV, HD))), w.eps)
    if not w.sliding and ctx.p_hi > w.n_floor - 1:  # log scaling (exact host tau per row)
        tau = torch.cat([tau_of(torch.arange(p0, p0 + R), w.n_floor, w.alpha) for _, R, p0, _, _, _ in ctx.segs])
        tau_t = ttnn.from_torch(tau.reshape(1, T, 1, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=w.mesh,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(w.mesh) if N > 1 else None)
        q = ttnn.multiply(q, tau_t)
        r = ttnn.multiply(r, tau_t)
    bf = lambda t: ttnn.typecast(t, ttnn.bfloat16)  # noqa: E731
    q, k, v = bf(q), bf(k), bf(v)
    # KV writes: one fill for the chunk (each 64-row block to its sequence's physical block)
    for cache, t in ((c.k_cache, k), (c.v_cache, v)):
        tt_ = ttnn.transpose(t, 1, 2)  # [1, NKV, T, HD]
        t8 = ttnn.typecast(tt_, c.kv_dtype) if c.kv_dtype != tt_.dtype else tt_
        ttnn.experimental.paged_fill_cache(cache, t8, ctx.fill[w.sliding], batch_idx=0)
        ttnn.deallocate(t8)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    rq = ttnn.pad(bf(r), [1, T, NH, 32], [0, 0, 0, 0], 0.0)
    sk = bf(ttnn.multiply(f32(ttnn.reshape(s, (1, T, NH, 1))), w.sink_bcast))
    return _packed_tail(w, c, q, rq, sk, g, ctx, T, tt_ccl, scatter)


def _packed_sdpa(w, c, q, rq, sk, positions, pt, T):
    """The packed SDPA over T rows -> [1, 1, T, NH * HD]."""
    o = inkling_sdpa_decode(q, c.k_cache, c.v_cache, positions, pt, rq, w.table, w.scale, w.W, w.E,
                            sink=sk, out_memory_config=DRAM, pt_per_item=True, mi=SDPA_MI, table2=w.table2, cargs=True)
    for t in (q, rq, sk):
        ttnn.deallocate(t)
    if int(o.shape[-1]) != w.NH * w.HD:  # (the SDPA writes [1, 1, rows, NH * HD] directly when it can)
        o = ttnn.reshape(o, (1, 1, T, w.NH * w.HD))
    return o


def _packed_tail(w, c, q, rq, sk, g, ctx, T, tt_ccl, scatter=None):
    """attention_packed()'s SDPA, output gate, o_proj and all-reduce."""
    return _packed_out(w, _packed_sdpa(w, c, q, rq, sk, ctx.positions, ctx.pt[w.sliding], T), g, T, tt_ccl, scatter)


def _packed_out(w, o, g, T, tt_ccl, scatter=None):
    """The output gate, o_proj and all-reduce of the SDPA output o [1, 1, T, NH * HD]. scatter (tt/model.py RowSplit,
    I2M_PF_RSX): reduce-scattered by rows instead (-> this die's rows)."""
    N = w.N
    if isinstance(g, tuple):  # (PGATE: g = (pq, the gate's first column))
        if w.mesh not in _PGATE:
            from .agate import PrefillGate

            _PGATE[w.mesh] = PrefillGate(w.mesh)
        go = _PGATE[w.mesh](o, *g)
        ttnn.deallocate(g[0])
        ttnn.deallocate(o)
        o = go
    else:
        f32 = lambda t: ttnn.typecast(t, ttnn.float32)  # noqa: E731
        bf = lambda t: ttnn.typecast(t, ttnn.bfloat16)  # noqa: E731
        o = bf(ttnn.multiply(f32(o), ttnn.sigmoid(f32(g))))
    out = proj(w, "o", o)
    if w.dram_sharded and int(out.shape[-1]) != w.H:
        out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, T, w.H))
    if N > 1 and scatter is not None:
        u = scatter(out)
        ttnn.deallocate(out)
        return u
    if N > 1:
        out = all_reduce(out, w.mesh, tt_ccl)
    return out
