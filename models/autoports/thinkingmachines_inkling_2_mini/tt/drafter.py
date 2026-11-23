# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative drafter (the checkpoint's mtp.safetensors), traceable, replicated on every die of the mesh.

Follows reference/dspark_ref.py (the spec) except for roundings that only move near-ties (the target verifies every
draft): the Markov head runs in fp32 without its bf16 roundings, greedy step logits skip the softcap (monotonic) and
the bf16 rounding of base + bias, RoPE runs in the fused bf16 op; the big matmuls use bfp8 weights (DRAFT_W_DT /
DRAFT_MLP_DT / DRAFT_HEAD_DT / DRAFT_FOLD_DT; bfp4 anywhere costs draft accuracy).

Device state (every shape fixed; positions live in device tensors, so the whole step replays from a trace):
  in_anchor / in_p0 / in_n   uint32 [1, 1] ROW_MAJOR inputs: the anchor token (at position p = p0 + n), the first
                             committed row's position, the number of committed rows
  ctrl program               p0, n -> RoPE row indices, f_rnn row, the SDPA's last key p + 7 (kernels/dsp_ctrl.cpp)
  caches                     per layer K / V, paged [NKV S / 32, 1, 32, 128] bf16 (KV head h = SDPA item h): the global
                             layer by absolute position (S = max_len + 512), the sliding layers (0, 2) as a ring of
                             S = 1568 rows (window 1024 + a 512-row prefill chunk + 32: rows written past the context
                             never evict a live row); written by kernels/dsp_kvw.cpp at (p0 + i) mod S
  commit(rows R)             mix_rmsnorm of the taps per layer -> k_proj_ctx / v_proj_ctx -> k_norm_ctx -> RoPE ->
                             cache rows p0 .. p0+R-1 (rows >= n land past the context: overwritten later);
                             f_rnn of row n-1 kept in a device buffer
  draft(d)                   8-row block (row 0 = embed_norm(embed(anchor)), rows 1..7 mask_embed) through the 3
                             layers: projections bank-streamed (DecodeLinear), heads split / merged by tile copies
                             (kernels/dsp_heads.cpp), the block's K / V written to cache rows p .. p+7, then paged
                             flash-decode SDPA with one item per KV head (its 3 query heads x 8 rows = one tile; keys
                             <= p+7, the sliding layers' window ending there); base logits of rows 1..d into a compact
                             FP32 buffer (the LM head streamed per DRAM bank, kernels/dsp_head_writer.cpp); d Markov
                             steps: the RNN on one row, then one program per step for [lat, 1] @ [fold W; fold b] + base
                             row -> argmax -> the next token (kernels/dsp_step_*.cpp, the fold weight streamed per bank)
  sampled drafts             capture(..., sample=True): per step the top TOPK = 32 of the softcapped step logits
                             (ids / logits in cand_id / cand_lg, host: cand_host()), token = argmax_j (logit_j + tg_j)
                             with tg = T * Gumbel noise the host writes before each replay (set_gumbel; zeros = top-1);
                             the drafter's q = softmax(logits / T) over the 32 candidates

Use: eager    commit(taps, p0, valid) / draft(anchor, ctx_len, d=..., sample=...)  (host values)
     traced   capture(taps_buf, d, sample=...) -> trace; set_inputs(anchor, p0, n) [+ set_gumbel(T g)]; replay();
              tokens_host() / self.tokens (uint32 [1, 1, 1, d] ROW_MAJOR on every die) [+ cand_host()]
     every eager shape used while a trace exists must be warmed before the capture (warm(); __init__ warms the
     prefill chunk sizes 64 .. 512 and d = 1 .. d_max)
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.demos.blackhole.qwen36.tt.decode_linear.ops import KERNEL_DIR as DL_DIR
from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear

from .dmm import dmm
from .model import Ops
from .weights import Checkpoint, load_config

KDIR = Path(__file__).parent / "kernels"
DRAM = ttnn.DRAM_MEMORY_CONFIG
F32, BF16, U32 = ttnn.float32, ttnn.bfloat16, ttnn.uint32
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
NEG = -1e9
_DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
RC = 4  # rows of the traced commit
CHUNK_MAX = 512  # largest prefill commit chunk
TOPK = 32  # sampled drafts: candidates per Markov step
CAP_L, CAP_INV_H = 128.0, 32.0  # the step softcap table: x in [-CAP_L, CAP_L], 32 points per unit
CAP_NTAB = int(2 * CAP_L * CAP_INV_H) + 2
DL = os.environ.get("DRAFT_DL", "1") == "1"  # block projections as bank-streamed DecodeLinear programs
RMS_BF = os.environ.get("DRAFT_RMS", "bf16") == "bf16"
ADD_BF = os.environ.get("DRAFT_ADD", "bf16") == "bf16"  # residual adds as one bf16 op
ROPE_BF = os.environ.get("DRAFT_ROPE", "bf16") == "bf16"


def _bits(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def _f32(t):
    return ttnn.typecast(t, F32) if t.dtype != F32 else t


def _rt(args):
    """{(x, y): [args]} -> RuntimeArgs."""
    rt = ttnn.RuntimeArgs()
    for (x, y), a in args.items():
        rt[x][y] = a
    return rt


def _hash(key):
    return int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)


def _cb(i, cores, dt, page, pages):
    return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])


def _crs(cores):
    rs = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
    for c in cores[1:]:
        rs = rs.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
    return rs


class DSparkDrafter:
    def __init__(self, mesh, model=None, max_len=8192, wdtype=None, head_dtype=None, fold_dtype=None, snap=None,
                 d_max=7):
        """model: an Inkling2Model on the same mesh: shares its embedding / embed_norm and, on one die with the streamed
        head, the LM head weight (same per-bank layout). None: the drafter loads them."""
        self.mesh = mesh
        self.N = N = mesh.get_num_devices()
        self.rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
        wdtype = wdtype or _DT[os.environ.get("DRAFT_W_DT", "bfp8")]
        mdtype = _DT[os.environ["DRAFT_MLP_DT"]] if os.environ.get("DRAFT_MLP_DT") else wdtype  # (MLP gate|up, down)
        head_dtype = head_dtype or _DT[os.environ.get("DRAFT_HEAD_DT", "bfp8")]
        fold_dtype = fold_dtype or _DT[os.environ.get("DRAFT_FOLD_DT", "bfp8")]
        cfg = model.cfg if model is not None else load_config(snap)
        m = cfg.mtp
        assert m["markov_head_type"] == "rnn_hidden"
        self.H, self.L, self.NH, self.NKV, self.D = (m["hidden_size"], m["num_hidden_layers"], m["num_attention_heads"],
                                                     m["num_key_value_heads"], m["head_dim"])
        self.G = self.NH // self.NKV
        self.eps = m["rms_norm_eps"]
        assert cfg.rms_norm_eps == self.eps
        self.theta, self.window, self.local = m["rope_theta"], m["sliding_window_size"], set(m["local_layer_ids"])
        self.B, self.R = m["block_size"], m["markov_rank"]
        self.d_max = d_max
        self.V = cfg.unpadded_vocab_size
        self.Vp = Vp = (self.V + 31) // 32 * 32
        self.NT = Vp // 32
        self.max_len = max_len
        self.ops = ops = Ops(mesh, self.eps)
        self.ckc = ops.ckc  # HiFi4, fp32 acc
        hifi4 = os.environ.get("DRAFT_HIFI4") == "1"  # bfp8 matmuls at HiFi4 (default HiFi2)
        fid = lambda dt: ttnn.MathFidelity.HiFi4 if dt == BF16 or hifi4 else ttnn.MathFidelity.HiFi2  # noqa: E731
        self.ckw = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=fid(wdtype),
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        H, D, R = self.H, self.D, self.R

        f = safe_open(os.path.join(cfg.snap, "mtp.safetensors"), "pt")
        w = {k: f.get_tensor(k).clone() for k in f.keys()}

        def mat(t, dt=wdtype):  # torch [N, K] (nn.Linear) -> device [1, 1, K, N]
            return ttnn.as_tensor(t.T.contiguous().reshape(1, 1, t.shape[1], t.shape[0]).to(torch.bfloat16), dtype=dt,
                                  layout=TILE, device=mesh, memory_config=DRAM, mesh_mapper=self.rep)

        dls = {}
        dl_readers = int(os.environ.get("DRAFT_DL_READERS", "6"))

        def lin(t, dt=wdtype):  # torch [N, K] -> (DecodeLinear, its bank-streamed weight) (DRAFT_DL=1) or mat(t, dt)
            if not DL:
                return mat(t, dt)
            key = (t.shape[1], t.shape[0], dt)
            if key not in dls:
                dls[key] = DecodeLinear(mesh, t.shape[1], t.shape[0], dt, readers=dl_readers, fidelity=fid(dt),
                                        buffers=2 if t.shape[1] > 4096 else 3)  # (L1: x + buffered weight)
            return dls[key], dls[key].weight([t.T.contiguous().to(torch.bfloat16)] * N)

        def vec(t):  # [n] -> fp32 [1, 1, 1, n]
            return ops.up(t.float().reshape(1, 1, 1, -1))

        alpha = torch.softmax(w["fuse_weight"].float(), dim=-1).to(torch.bfloat16).float()  # [L + 1, 4]
        self.alpha_f = [[float(v) for v in alpha[i]] for i in range(self.L + 1)]
        self.layers = []
        for i in range(self.L):
            p = f"layers.{i}."
            g = lambda n: w[p + n]  # noqa: E731
            self.layers.append(dict(
                dflare=vec(g("dflare_norm.weight")), in_norm=vec(g("input_layernorm.weight")),
                post_norm=vec(g("post_attention_layernorm.weight")),
                q_norm=vec(g("self_attn.q_norm.weight")), k_norm=vec(g("self_attn.k_norm.weight")),
                k_norm_ctx=vec(g("self_attn.k_norm_ctx.weight")),
                qkv=lin(torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                                   g("self_attn.v_proj.weight")])),
                kv_ctx=mat(torch.cat([g("self_attn.k_proj_ctx.weight"), g("self_attn.v_proj_ctx.weight")])),
                o=mat(g("self_attn.o_proj.weight")),  # (narrow N, wide K: the interleaved matmul is faster)
                gu=lin(torch.cat([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")]), mdtype),
                down=mat(g("mlp.down_proj.weight"), mdtype),
            ))
        self.norm = vec(w["norm.weight"])
        self.rnn_norm = vec(w["markov_head.dflare_norm.weight"])
        tmpl = torch.zeros(1, 1, self.B, H)
        tmpl[0, 0, 1:self.B] = w["mask_embed"].float()
        self.tmpl = ops.up(tmpl, dtype=BF16)
        rp = w["markov_head.rnn_proj.weight"]
        self.w_shh = mat(rp[:, :2 * R + H], BF16)  # [s, w1[prev], h] rows of rnn_proj
        sel = torch.zeros(self.B - 1, 1, 1, 1, self.B)
        for k in range(self.B - 1):
            sel[k, 0, 0, 0, k + 1] = 1.0
        self.sel = [ops.up(sel[k], dtype=BF16) for k in range(self.B - 1)]
        self.w_f = mat(rp[:, 2 * R + H:], BF16)
        self.b_rnn = vec(w["markov_head.rnn_proj.bias"])
        self.w1 = ttnn.as_tensor(w["markov_head.markov_w1.weight"], dtype=BF16, layout=RM, device=mesh,
                                 memory_config=DRAM, mesh_mapper=self.rep)
        self.zero_s = ops.up(torch.zeros(1, 1, 1, R))
        self.zero_sb = ops.up(torch.zeros(1, 1, 1, R), dtype=BF16)
        self.wp = [ops.up(torch.zeros(1, 1, 1, R), dtype=BF16) for _ in range(d_max + 1)]

        # shared modules: embedding + embed_norm, LM head
        ck = model.ck if model is not None else Checkpoint(cfg.snap)
        if model is not None:
            self.embed, self.embed_norm = model.embed, model.embed_norm
        else:
            self.embed = ttnn.as_tensor(ck.get("model.llm.embed.weight").to(torch.bfloat16), dtype=BF16, layout=RM,
                                        device=mesh, memory_config=DRAM, mesh_mapper=self.rep)
            self.embed_norm = vec(ck.get("model.llm.embed_norm.weight"))
        un = ck.get("model.llm.unembed.weight")[: self.V]
        if model is not None and N == 1 and model.shead is not None and model.shead.n == Vp:
            self.dl_head, self.lm_head = model.shead, model.lm_head
        else:
            self.dl_head = DecodeLinear(mesh, H, Vp, head_dtype, readers=int(os.environ.get("DRAFT_HEAD_READERS", "4")),
                                        fidelity=fid(head_dtype))
            wt = torch.nn.functional.pad(un.T.contiguous(), (0, Vp - self.V)).to(torch.bfloat16)
            self.lm_head = self.dl_head.weight([wt] * N)
        fw, fb = self._fold(un, w["markov_head.hidden_out.weight"], w["markov_head.hidden_out.bias"])
        wf = torch.zeros(R + 32, Vp)
        wf[:R, : self.V] = fw.T.float()
        wf[R, : self.V] = fb.float()
        self.dl_fold = DecodeLinear(mesh, R + 32, Vp, fold_dtype, readers=int(os.environ.get("DRAFT_FOLD_READERS", "8")),
                                    fidelity=fid(fold_dtype))
        self.fold = self.dl_fold.weight([wf.to(torch.bfloat16)] * N)
        del w

        # RoPE tables (fp32, neox): C = [c, c], S = [-s, s]; rope(x) = x C + [x2, x1] S
        self.SG = (max_len + CHUNK_MAX + 31) // 32 * 32
        self.SW = (self.window + CHUNK_MAX + 32 + 31) // 32 * 32
        npos = self.SG + 64
        inv = 1.0 / (self.theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
        ang = torch.arange(npos, dtype=torch.float64)[:, None] * inv[None, :]
        c, s = ang.cos().float(), ang.sin().float()
        # (ttnn.embedding gathers bf16 rows only: each table as hi + lo bf16 halves, summed in fp32 (~2^-17 relative))
        self.rope_t = []
        for t in (torch.cat([c, c], -1), torch.cat([-s, s], -1)):
            hi = t.to(torch.bfloat16)
            lo = (t - hi.float()).to(torch.bfloat16)
            self.rope_t.append([ttnn.as_tensor(x, dtype=BF16, layout=RM, device=mesh, memory_config=DRAM,
                                               mesh_mapper=self.rep) for x in (hi, lo)])
        # ROPE_BF (DRAFT_ROPE=bf16, default): the fused rotary_embedding op with bf16 cos / sin ([c, c], [s, s])
        self.rope_bf = ROPE_BF
        self.rope_b = [ttnn.as_tensor(torch.cat([t, t], -1).to(torch.bfloat16), dtype=BF16, layout=RM, device=mesh,
                                      memory_config=DRAM, mesh_mapper=self.rep) for t in (c, s)]

        # caches: per layer K / V, paged layout [NKV * S / 32, 1, 32, D] (= [1, NKV, S, D] tile order), KV head h =
        # SDPA item h (pages h S/32 ..): the G query heads x B block rows of a KV head are one item (one tile)
        self.S = [self.SW if i in self.local else self.SG for i in range(self.L)]
        self.kc = [self._zeros([self.NKV * S // 32, 1, 32, D], BF16) for S in self.S]
        self.vc = [self._zeros([self.NKV * S // 32, 1, 32, D], BF16) for S in self.S]
        self.ptab = {S: ttnn.from_torch(torch.arange(self.NKV * S // 32, dtype=torch.int32).reshape(self.NKV, S // 32),
                                        dtype=ttnn.int32, layout=RM, device=mesh, memory_config=DRAM,
                                        mesh_mapper=self.rep) for S in set(self.S)}
        grid = mesh.compute_with_storage_grid_size()
        self.sdpa_cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=32,
                                               k_chunk_size=int(os.environ.get("DRAFT_K_CHUNK", "128")),
                                               exp_approx_mode=False)

        # device inputs / control outputs / buffers (fixed addresses: traced)
        self.in_anchor, self.in_p0, self.in_n = (self._u32([[0]]) for _ in range(3))
        self.idx_c = self._u32([[0] * RC])
        self.idx_n = self._u32([[0]])
        self.idx_b = self._u32([[0] * self.B])
        self.cpos = ttnn.from_torch(torch.zeros(self.NKV, dtype=torch.int32), dtype=ttnn.int32, layout=RM,
                                    device=mesh, memory_config=DRAM, mesh_mapper=self.rep)
        self.base = self._zeros([1, 8, self.NT, 32], F32, RM)
        self.prev = [self._u32([[0]]) for _ in range(d_max)]
        self.f_rnn = self._zeros([1, 1, 1, H], BF16)
        self.chunk_in = {}  # T -> (p0 tensor, rope index tensor, last-row index tensor)
        # sampled drafts (capture(..., sample=True)): host-written T * Gumbel noise, per step the top TOPK ids / logits
        self.sample = False
        self.cap = cfg.final_logit_softcapping
        self.gum = self._zeros([d_max, TOPK], F32, RM)
        self.cand_id = ttnn.from_torch(torch.zeros(d_max, TOPK, dtype=torch.int32), dtype=U32, layout=RM, device=mesh,
                                       memory_config=DRAM, mesh_mapper=self.rep)
        self.cand_lg = self._zeros([d_max, TOPK], F32, RM)
        xs = torch.arange(CAP_NTAB, dtype=torch.float64) / CAP_INV_H - CAP_L
        self.cap_tab = ttnn.from_torch((self.cap * torch.tanh(xs / self.cap)).float().reshape(1, -1), dtype=F32,
                                       layout=RM, device=mesh, memory_config=DRAM, mesh_mapper=self.rep)
        self.tokens = None
        self.trace = None
        # every eager shape used while a trace exists must run once before the capture: buffers allocated after it
        # (program-cache tensors, chunk inputs) sit in memory the replays overwrite
        self.warm([64 * i for i in range(1, CHUNK_MAX // 64 + 1)], range(1, d_max + 1))

    # ------------------------------------------------------------------ helpers
    def _zeros(self, shape, dt, layout=TILE):
        return ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=layout, device=self.mesh, memory_config=DRAM,
                               mesh_mapper=self.rep)

    def _u32(self, rows):
        return ttnn.from_torch(torch.tensor(rows, dtype=torch.int32), dtype=U32, layout=RM, device=self.mesh,
                               memory_config=DRAM, mesh_mapper=self.rep)

    def _write(self, dev, rows):
        """host values -> an existing uint32 device tensor (same address)."""
        h = ttnn.from_torch(torch.tensor(rows, dtype=torch.int32), dtype=U32, layout=RM, mesh_mapper=self.rep)
        ttnn.copy_host_to_device_tensor(h, dev)

    def _host(self, t):
        if self.N > 1:
            t = ttnn.get_device_tensors(t)[0]
        return ttnn.to_torch(t)

    @staticmethod
    def _fold(un, proj, bias):
        """lm_head W @ hidden_out W / b in fp64, stored bf16 (Inkling2MTPHead.refresh_fold). Cached in I2M_WCACHE."""
        cdir = os.environ.get("I2M_WCACHE")
        path = os.path.join(cdir, f"dspark_fold_{un.shape[0]}.pt") if cdir else None
        if path and os.path.exists(path):
            d = torch.load(path)
            return d["w"], d["b"]
        V = un.shape[0]
        fw = torch.empty(V, proj.shape[1], dtype=torch.bfloat16)
        fb = torch.empty(V, dtype=torch.bfloat16)
        pd, bd = proj.double(), bias.double()
        for s in range(0, V, 8192):
            hd = un[s:s + 8192].to(torch.bfloat16).double()
            fw[s:s + 8192] = (hd @ pd).to(torch.bfloat16)
            fb[s:s + 8192] = (hd @ bd).to(torch.bfloat16)
        if path:
            torch.save({"w": fw, "b": fb}, path)
        return fw, fb

    def _lin(self, x, w):
        """x [1, 1, R, K] bf16 @ a lin() weight -> [1, 1, R, N] bf16."""
        if isinstance(w, tuple):
            return w[0](x, w[1], int(x.shape[2]), out_memory_config=DRAM)
        return self._mm(x, w, BF16)

    def _mm(self, x, wt, dtype, ck=None):
        return dmm(x, wt, ck or self.ckw, dtype=dtype, memory_config=DRAM)

    def _rope(self, idx):
        """idx uint32 [1, R] -> (C, S) [1, 1, R, D]: fp32 (sign-baked S), or bf16 for the fused op (ROPE_BF)."""
        R = int(idx.shape[-1])
        if self.rope_bf:
            return [ttnn.reshape(ttnn.embedding(idx, t, layout=TILE), (1, 1, R, self.D)) for t in self.rope_b]
        out = []
        for hi, lo in self.rope_t:
            t = ttnn.add(_f32(ttnn.embedding(idx, hi, layout=TILE)), _f32(ttnn.embedding(idx, lo, layout=TILE)))
            out.append(ttnn.reshape(t, (1, 1, R, self.D)))
        return out

    def _norm_rope(self, x, w, C, S):
        """x [1, nh, R, D] bf16 -> per-head RMSNorm (fp32 inside, bf16 out), neox RoPE in fp32 -> bf16."""
        h = self.D // 2
        if self.rope_bf:
            xn = ttnn.rms_norm(x, epsilon=self.eps, weight=w, compute_kernel_config=self.ckc)
            y = ttnn.experimental.rotary_embedding(xn, C, S, compute_kernel_config=self.ckc)
            nh, R = int(x.shape[1]), int(x.shape[2])
            return y if int(y.shape[2]) == R else ttnn.reshape(y, (1, nh, R, self.D), tuple(y.padded_shape))
        xn = _f32(ttnn.rms_norm(x, epsilon=self.eps, weight=w, compute_kernel_config=self.ckc))
        nh, R = int(x.shape[1]), int(x.shape[2])
        sw = ttnn.concat([ttnn.slice(xn, [0, 0, 0, h], [1, nh, R, self.D]), ttnn.slice(xn, [0, 0, 0, 0], [1, nh, R, h])],
                         dim=-1)
        return ttnn.typecast(ttnn.add(ttnn.multiply(xn, C), ttnn.multiply(sw, S)), BF16)

    def _mix(self, taps, alphas):
        """taps [1, 4, R, H] bf16 -> per alpha row [1, 1, R, H] fp32 ((a0 T0 + a1 T1) + a2 T2) + a3 T3 (one program,
        kernels/dsp_mix_*.cpp)."""
        R, NM = int(taps.shape[2]), len(alphas)
        assert NM <= 4
        TPT = (R + 31) // 32 * (self.H // 32)
        outs = [self._alloc([1, 1, R, self.H], F32) for _ in range(NM)]
        grid = self.mesh.compute_with_storage_grid_size()
        TPC = -(-TPT // (grid.x * grid.y))
        cores, dm_rt, cp_rt = [], {}, {}
        for t0 in range(0, TPT, TPC):
            j = len(cores)
            c = ttnn.CoreCoord(j % grid.x, j // grid.x)
            cores.append(c)
            nt = min(TPC, TPT - t0)
            dm_rt[(c.x, c.y)] = [taps.buffer_address(), t0, nt] + [o.buffer_address() for o in outs]
            cp_rt[(c.x, c.y)] = [nt]
        crs = _crs(cores)
        cb_in, cb_out = 0, 16
        ct = [cb_in, cb_out, NM, TPT] + ttnn.TensorAccessorArgs(taps).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(outs[0]).get_compile_time_args()
        bits = [_bits(float(a[j])) for a in alphas for j in range(4)]
        cp_ct = [cb_in, cb_out, NM] + bits + [0] * (16 - len(bits))
        kernels = [
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "dsp_mix_dm.cpp"), core_ranges=crs, compile_time_args=ct,
                                  runtime_args=_rt(dm_rt),
                                  config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                           noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "dsp_mix_compute.cpp"), core_ranges=crs,
                                  compile_time_args=cp_ct, runtime_args=_rt(cp_rt),
                                  config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                                      fp32_dest_acc_en=True, math_approx_mode=False)),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[],
                                      cbs=[_cb(cb_in, crs, BF16, 2048, 8), _cb(cb_out, crs, F32, 4096, 2 * NM)])
        prog.custom_program_hash = _hash(("dsp_mix", tuple(ct), tuple(cp_ct), TPC))
        ttnn.generic_op([taps] + outs, prog)
        return outs

    def _rms(self, x, w):
        """RMSNorm over the last dim -> bf16. RMS_BF: the width-sharded norm on x's own dtype (bf16 residual / fp32
        mix), the bf16 cast inside the reshard back; else Ops.rms (fp32 in, typecast out)."""
        H = int(x.shape[3])
        if not RMS_BF or int(x.shape[2]) > 32:
            return self.ops.rms(x, w, out_dtype=BF16)
        mc, pc = self.ops._rms_cfg(H)
        y = ttnn.rms_norm(ttnn.to_memory_config(x, mc), epsilon=self.eps, weight=w, program_config=pc,
                          memory_config=mc, compute_kernel_config=self.ckc)
        return ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG, dtype=BF16)

    def _add(self, a, b):
        if ADD_BF:
            return ttnn.add(a, b, dtype=BF16)
        return ttnn.typecast(ttnn.add(_f32(a), _f32(b)), BF16)

    # ------------------------------------------------------------------ programs
    def _ctrl(self):
        core = ttnn.CoreCoord(0, 0)
        crs = _crs([core])
        io = [self.in_p0, self.in_n, self.idx_c, self.idx_n, self.idx_b, self.cpos, self.in_anchor, self.w1, self.wp[0]]
        ct = [0, RC, self.B, self.NKV]
        for t in io:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        k = ttnn.KernelDescriptor(kernel_source=str(KDIR / "dsp_ctrl.cpp"), core_ranges=crs, compile_time_args=ct,
                                  runtime_args=_rt({(0, 0): [t.buffer_address() for t in io]}),
                                  config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                           noc=ttnn.NOC.NOC_0))
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(0, crs, F32, 4096, 1)])
        prog.custom_program_hash = _hash(("dsp_ctrl", tuple(ct)))
        ttnn.generic_op(io, prog)

    def _kvw(self, srcs, p0_t, R):
        """srcs: (layer, K, V), K / V [1, NKV, R, D] bf16 TILE -> cache rows (p0 + i) mod S (p0: element 0 of p0_t; one
        program, one core per (cache, head))."""
        grid = self.mesh.compute_with_storage_grid_size()
        jobs = []
        for i, k, v in srcs:
            for src, cache in ((k, self.kc[i]), (v, self.vc[i])):
                for h in range(self.NKV):
                    jobs.append((src, cache, h, self.S[i]))
        cores = [ttnn.CoreCoord(j % grid.x, j // grid.x) for j in range(len(jobs))]
        crs = _crs(cores)
        DT = self.D // 32
        # (one kernel per job group sharing the accessor compile args: all sources / caches are bf16 DRAM interleaved)
        ct = [0, R, DT] + ttnn.TensorAccessorArgs(jobs[0][0]).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(jobs[0][1]).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(p0_t).get_compile_time_args()
        rt = {(c.x, c.y): [src.buffer_address(), cache.buffer_address(), p0_t.buffer_address(), h, S]
              for c, (src, cache, h, S) in zip(cores, jobs)}
        k = ttnn.KernelDescriptor(kernel_source=str(KDIR / "dsp_kvw.cpp"), core_ranges=crs, compile_time_args=ct,
                                  runtime_args=_rt(rt),
                                  config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                           noc=ttnn.NOC.NOC_0))
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(0, crs, BF16, 2048, 2 * DT + 1)])
        prog.custom_program_hash = _hash(("dsp_kvw", tuple(ct), len(jobs)))
        io = [p0_t] + [t for s in srcs for t in s[1:]] + self.kc + self.vc
        ttnn.generic_op(io, prog)

    def _tcopy(self, jobs):
        """Head split / merge tile copies (kernels/dsp_heads.cpp), one program: jobs (src, dst, c0, RT, CT, ntiles, mode)
        with ntiles the head-major tensor's tile count."""
        grid = self.mesh.compute_with_storage_grid_size()
        NBUF = 4
        TPC = max(NBUF, -(-sum(j[5] for j in jobs) // (grid.x * grid.y - len(jobs))))  # tiles per core
        rt, cores = {}, []
        for src, dst, c0, RT, CT, nt, mode in jobs:
            for t0 in range(0, nt, TPC):
                j = len(cores)
                c = ttnn.CoreCoord(j % grid.x, j // grid.x)
                cores.append(c)
                rt[(c.x, c.y)] = [src.buffer_address(), dst.buffer_address(), c0, RT, CT, t0, min(TPC, nt - t0), mode]
        crs = _crs(cores)
        ct = [0, self.D // 32, NBUF] + ttnn.TensorAccessorArgs(jobs[0][0]).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(jobs[0][1]).get_compile_time_args()
        k = ttnn.KernelDescriptor(kernel_source=str(KDIR / "dsp_heads.cpp"), core_ranges=crs, compile_time_args=ct,
                                  runtime_args=_rt(rt),
                                  config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                           noc=ttnn.NOC.NOC_0))
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(0, crs, BF16, 2048, NBUF)])
        prog.custom_program_hash = _hash(("dsp_heads", tuple(ct), tuple(j[2:] for j in jobs), TPC))
        ttnn.generic_op([t for j in jobs for t in j[:2]], prog)

    def _alloc(self, shape, dt=BF16):
        return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, TILE, self.mesh, DRAM)

    def _split(self, x, heads):
        """x [1, 1, R, C] bf16 -> per (nh) of heads, consecutive column groups, [1, nh, R, D]."""
        R, CT = int(x.shape[2]), int(x.shape[3]) // 32
        RT, DTH = (R + 31) // 32, self.D // 32
        outs, jobs, c0 = [], [], 0
        for nh in heads:
            o = self._alloc([1, nh, R, self.D])
            jobs.append((x, o, c0, RT, CT, nh * RT * DTH, 0))
            outs.append(o)
            c0 += nh * DTH
        self._tcopy(jobs)
        return outs

    def _merge(self, x):
        """x [1, nh, R, D] bf16 -> [1, 1, R, nh*D]."""
        nh, R = int(x.shape[1]), int(x.shape[2])
        RT = (R + 31) // 32
        o = self._alloc([1, 1, R, nh * self.D])
        self._tcopy([(x, o, 0, RT, nh * self.D // 32, nh * RT * self.D // 32, 1)])
        return o

    def _stream(self, dl, kind, x, w, out, ct_w, rt_w, cp_src, cp_ct, cbs_extra, sems, u2d_cbs, defines=(),
                wdefines=()):
        """A DecodeLinear-placed program: dl_reader streams w per bank, writer `kind`, compute cp_src."""
        cb_x, cb_w = 0, 1
        cbs = [_cb(cb_x, dl.cores, BF16, 2048, dl.kt), _cb(cb_w, dl.cores, dl.weight_dtype, dl.tile_bytes,
                                                          dl.buffers * dl.kt)] + cbs_extra(dl.cores)
        rd_ct = [cb_w, dl.kt, dl.tile_bytes, dl.page_bytes, dl.buffers]
        rd, wr, cp = {}, {}, {}
        for slot, (c, b, c0, nc) in enumerate(dl.assign):
            rd[(c.x, c.y)] = [w.buffer_address(), b, c0, nc, (b + c0) & 0x3]
            wr[(c.x, c.y)] = rt_w(slot, b * dl.C + c0, nc)
            cp[(c.x, c.y)] = [nc]

        def sub(args, cores):
            return _rt({(c.x, c.y): args[(c.x, c.y)] for c in ttnn.corerange_to_cores(cores, row_wise=True)})

        dm = ttnn.DataMovementConfigDescriptor
        kernels = []
        for cores, rnoc, wnoc in ((dl.noc0_cores, ttnn.NOC.NOC_0, ttnn.NOC.NOC_1),
                                  (dl.noc1_cores, ttnn.NOC.NOC_1, ttnn.NOC.NOC_0)):
            if cores is None:
                continue
            kernels += [
                ttnn.KernelDescriptor(kernel_source=str(DL_DIR / "dl_reader.cpp"), core_ranges=cores,
                                      compile_time_args=rd_ct, runtime_args=sub(rd, cores),
                                      config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=rnoc)),
                ttnn.KernelDescriptor(kernel_source=str(KDIR / kind), core_ranges=cores, compile_time_args=ct_w,
                                      runtime_args=sub(wr, cores), defines=list(wdefines),
                                      config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=wnoc)),
            ]
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=dl.fidelity, fp32_dest_acc_en=True, math_approx_mode=False)
        if u2d_cbs:
            u2d = [ttnn.UnpackToDestMode.Default] * 64
            for i in u2d_cbs:
                u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
            cfg.unpack_to_dest_mode = u2d
        kernels.append(ttnn.KernelDescriptor(kernel_source=str(cp_src), core_ranges=dl.cores, compile_time_args=cp_ct,
                                             runtime_args=sub(cp, dl.cores), defines=list(defines), config=cfg))
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        prog.custom_program_hash = _hash((kind, tuple(ct_w), tuple(cp_ct), id(dl), tuple(defines), tuple(wdefines)))
        ttnn.generic_op([x, w, out], prog)

    def _base_logits(self, h, d):
        """rows 1..d of h [1, 1, 32, H] bf16 @ LM head -> self.base rows 1..d (FP32, compact)."""
        dl = self.dl_head
        cb_out = 16
        ct = [0, cb_out, dl.kt, d, self.NT] + ttnn.TensorAccessorArgs(h).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(self.base).get_compile_time_args()
        cp_ct = [0, 1, cb_out, dl.kt, 0, 0]
        self._stream(dl, "dsp_head_writer.cpp", h, self.lm_head, self.base, ct,
                     lambda slot, col0, nc: [h.buffer_address(), self.base.buffer_address(), col0, nc],
                     KDIR / "head_compute.cpp", cp_ct, lambda cores: [_cb(cb_out, cores, F32, 4096, 2)], [], [])

    def _step(self, lat, row, prev, ks=0):
        """lat [1, 1, 1, R] bf16 -> argmax_v ([lat, 1] @ [fold W; fold b] + base[row])_v into prev (uint32 [1, 1]).
        Sampled (self.sample): the softcapped logits' top TOPK -> row ks of cand_id / cand_lg, the token
        argmax_j (logit_j + gum[ks, j])."""
        dl = self.dl_fold
        cb_base, cb_out, cb_bseg, cb_cand = 2, 16, 3, 4
        n = len(dl.assign)
        red = self.mesh.worker_core_from_logical_core(dl.assign[0][0])
        ct = [0, cb_out, cb_base, cb_bseg, cb_cand, dl.kt, self.V, self.NT, row, n, red.x, red.y] \
            + ttnn.TensorAccessorArgs(lat).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(self.base).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(prev).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(self.w1).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(self.wp[ks + 1]).get_compile_time_args()
        cp_ct = [0, 1, cb_base, cb_out, dl.kt]
        extra, cdef, wdef, cand_bytes = [self.w1.buffer_address(), self.wp[ks + 1].buffer_address()], [], [], \
            (n + 2) * 16
        if self.sample:
            K = TOPK
            for t in (self.gum, self.cand_id, self.cand_lg, self.cap_tab):
                ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
            ct += [ks, CAP_NTAB]
            extra += [t.buffer_address() for t in (self.gum, self.cand_id, self.cand_lg, self.cap_tab)]
            wdef = [("SAMPLE", "1"), ("TOPK", str(K)), ("CAP_L", f"{CAP_L:.1f}f"), ("CAP_INV_H", f"{CAP_INV_H:.1f}f")]
            cand_bytes = (n + 2) * 8 * K + 4 * K + 4 * CAP_NTAB
        w1_off = -(-cand_bytes // 64) * 64  # (the reducer's w1-row scratch)
        wdef = wdef + [("W1_OFF", str(w1_off))]
        cand_bytes = w1_off + 1536
        cbs = lambda cores: [_cb(cb_base, cores, F32, 4096, 2), _cb(cb_out, cores, F32, 4096, 2),  # noqa: E731
                             _cb(cb_bseg, cores, F32, dl.cpc * 128, 1), _cb(cb_cand, cores, F32, cand_bytes, 1)]
        sems = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, dl.cores, 0)]
        self._stream(dl, "dsp_step_writer.cpp", lat, self.fold, self.base, ct,
                     lambda slot, col0, nc: [lat.buffer_address(), self.base.buffer_address(), prev.buffer_address(),
                                             col0, nc, slot] + extra,
                     KDIR / "dsp_step_compute.cpp", cp_ct, cbs, sems, [cb_base], defines=cdef, wdefines=wdef)
        # (prev is written by the program but is not its output tensor: keep it alive in the io list)

    # ------------------------------------------------------------------ commit
    def _commit_rows(self, taps, idx, p0_t, last):
        """taps [1, 4, R, H] bf16 TILE, idx uint32 [1, R] (RoPE rows), p0_t uint32 [1, 1], last uint32 [1, 1] (the row
        whose f_rnn is kept)."""
        R = int(taps.shape[2])
        mixes = self._mix(taps, self.alpha_f)
        C, S = self._rope(idx)
        srcs = []
        for i, Lw in enumerate(self.layers):
            ctx = self._rms(mixes[i], Lw["dflare"])
            kv = self._mm(ctx, Lw["kv_ctx"], BF16)
            k, v = self._split(kv, (self.NKV, self.NKV))
            srcs.append((i, self._norm_rope(k, Lw["k_norm_ctx"], C, S), v))
        self._kvw(srcs, p0_t, R)
        f = self._rms(mixes[self.L], self.rnn_norm)  # [1, 1, R, H]
        table = ttnn.reshape(ttnn.to_layout(f, RM), (R, self.H))
        ttnn.copy(ttnn.reshape(ttnn.embedding(last, table, layout=TILE), (1, 1, 1, self.H)), self.f_rnn)

    def commit(self, taps, p0: int, valid: int | None = None):
        """Eager / prefill commit: taps torch [T, 4, H] bf16 or ttnn [1, 4, T, H] bf16 TILE (T: any for torch; a multiple
        of 32 <= 512 for ttnn: one program set per T), rows 0 .. valid-1 at positions p0 ..; f_rnn from row valid-1."""
        if isinstance(taps, torch.Tensor):
            T = taps.shape[0]
            valid = T if valid is None else valid
            for s in range(0, valid, CHUNK_MAX):
                n = min(CHUNK_MAX, valid - s)
                Tp = (n + 63) // 64 * 64
                x = torch.zeros(1, 4, Tp, self.H, dtype=torch.bfloat16)
                x[0, :, :n] = taps[s:s + n].transpose(0, 1)
                t = ttnn.from_torch(x, dtype=BF16, layout=TILE, device=self.mesh, memory_config=DRAM,
                                    mesh_mapper=self.rep)
                self.commit(t, p0 + s, n)
            return
        T = int(taps.shape[2])
        valid = T if valid is None else valid
        assert T <= CHUNK_MAX and 1 <= valid <= T and p0 + T <= self.SG
        if T not in self.chunk_in:
            assert self.trace is None, f"commit of {T} rows not warmed before the capture (DSparkDrafter.warm)"
            self.chunk_in[T] = (self._u32([[0]]), self._u32([[0] * T]), self._u32([[0]]))
        p0_t, idx, last = self.chunk_in[T]
        self._write(p0_t, [[p0]])
        self._write(idx, [[p0 + i for i in range(T)]])
        self._write(last, [[valid - 1]])
        self._commit_rows(taps, idx, p0_t, last)

    def warm(self, Ts=(), ds=()):
        """Compile the eager commit for chunk sizes Ts and the eager draft for depths ds (zero taps at position 0:
        call before the first real commit; caches are positional, later commits overwrite the rows)."""
        assert self.trace is None, "warm before capture"
        for T in Ts:
            t = ttnn.from_torch(torch.zeros(1, 4, T, self.H), dtype=BF16, layout=TILE, device=self.mesh,
                                memory_config=DRAM, mesh_mapper=self.rep)
            self.commit(t, 0, T)
            ttnn.deallocate(t)
        for d in ds:
            self.draft(0, 64, d=d)
        if ds:
            self.draft(0, 64, d=max(ds), sample=True)
            self.sample = False
        ttnn.synchronize_device(self.mesh)

    def reset(self):
        pass  # (caches are positional: a new sequence overwrites them; masks never read past p)

    # ------------------------------------------------------------------ draft
    def _attention(self, i, q, k, v):
        """q [1, NH, B, D], k / v [1, NKV, B, D] bf16 (block rows): the block's K / V to cache rows p .. p+B-1, then
        flash-decode SDPA over the cache (item = KV head, its G x B query rows: one tile; keys <= p+B-1, the sliding
        layers' window W + B back from there) -> [1, 1, B, NH*D] bf16."""
        self._kvw([(i, k, v)], self.idx_b, self.B)
        local = i in self.local
        S = self.S[i]
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.reshape(q, (1, self.NKV, self.G * self.B, self.D)), self.kc[i], self.vc[i], self.ptab[S],
            cur_pos_tensor=self.cpos, scale=self.D**-0.5, sliding_window_size=self.window + self.B if local else None,
            cache_position_modulo=S if local else None, program_config=self.sdpa_cfg,
            compute_kernel_config=self.ckc, memory_config=DRAM)
        return self._merge(ttnn.reshape(o, (1, self.NH, self.B, self.D)))

    def block_hidden(self):
        """The block at position p (ctrl outputs) -> final normed hidden [1, 1, B, H] bf16."""
        ops, H = self.ops, self.H
        e = ttnn.embedding(self.in_anchor, self.embed, layout=TILE)
        e = ttnn.pad(ttnn.reshape(e, (1, 1, 1, H)), [(0, 0), (0, 0), (0, self.B - 1), (0, 0)], 0.0)
        hidden = ttnn.add(self._rms(e, self.embed_norm), self.tmpl)
        C, S = self._rope(self.idx_b)
        residual = None
        for i, Lw in enumerate(self.layers):
            residual = hidden if residual is None else self._add(hidden, residual)
            x = self._rms(residual, Lw["in_norm"])
            qkv = self._lin(x, Lw["qkv"])
            q, k, v = self._split(qkv, (self.NH, self.NKV, self.NKV))
            q, k = self._norm_rope(q, Lw["q_norm"], C, S), self._norm_rope(k, Lw["k_norm"], C, S)
            hidden = self._lin(self._attention(i, q, k, v), Lw["o"])
            residual = self._add(hidden, residual)
            x = self._rms(residual, Lw["post_norm"])
            gu = self._lin(x, Lw["gu"])
            I = int(gu.shape[-1]) // 2
            g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, self.B, I])
            u = ttnn.slice(gu, [0, 0, 0, I], [1, 1, self.B, 2 * I])
            act = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=BF16)
            hidden = self._lin(act, Lw["down"])
        residual = self._add(hidden, residual)
        return ttnn.to_memory_config(self._rms(residual, self.norm), DRAM)

    def _draft(self, d):
        """ctrl outputs + in_anchor + f_rnn -> d tokens (uint32 [1, 1, 1, d] ROW_MAJOR) and the block hidden."""
        R = self.R
        h = self.block_hidden()
        self._base_logits(h, d)
        cf = ttnn.add(self._mm(self.f_rnn, self.w_f, F32, self.ckc), self.b_rnn)  # [1, 1, 1, 3R]
        s, sb, prev = self.zero_s, self.zero_sb, self.in_anchor
        for k in range(d):
            wp = self.wp[k]  # (w1[prev]: written by ctrl / the previous step program)
            hk = ttnn.matmul(self.sel[k], h, compute_kernel_config=self.ckc, dtype=BF16)  # row k+1 of h (exact)
            pj = ttnn.add(self._mm(ttnn.concat([sb, wp, hk], dim=-1), self.w_shh, F32, self.ckc), cf)
            gate = ttnn.sigmoid(ttnn.slice(pj, [0, 0, 0, 0], [1, 1, 1, R]))
            tc = ttnn.tanh(ttnn.slice(pj, [0, 0, 0, R], [1, 1, 1, 2 * R]))
            lat = ttnn.typecast(ttnn.tanh(ttnn.slice(pj, [0, 0, 0, 2 * R], [1, 1, 1, 3 * R])), BF16)
            self._step(lat, k + 1, self.prev[k], k)
            prev = self.prev[k]
            if k + 1 < d:
                s = ttnn.lerp(tc, s, gate)
                sb = ttnn.typecast(s, BF16)
        toks = ttnn.concat([ttnn.reshape(t, (1, 1, 1, 1)) for t in self.prev[:d]], dim=-1) if d > 1 else \
            ttnn.reshape(self.prev[0], (1, 1, 1, 1))
        return toks, h

    def draft(self, anchor, ctx_len: int, last_taps=None, d: int = 3, return_hidden=False, sample=False):
        """Eager draft at position ctx_len (cache rows < ctx_len committed; f_rnn from the last commit unless last_taps
        (torch [4, H]) is given). Returns (ttnn uint32 [1, 1, 1, d], host list)."""
        assert 1 <= d <= self.d_max
        self.sample = sample
        self.set_inputs(anchor, ctx_len, 0)
        self._ctrl()
        if last_taps is not None:
            x = torch.zeros(1, 4, 32, self.H, dtype=torch.bfloat16)
            x[0, :, 0] = last_taps.reshape(4, self.H)
            t = ttnn.from_torch(x, dtype=BF16, layout=TILE, device=self.mesh, memory_config=DRAM, mesh_mapper=self.rep)
            f = self._rms(self._mix(t, [self.alpha_f[self.L]])[0], self.rnn_norm)
            ttnn.copy(ttnn.slice(f, [0, 0, 0, 0], [1, 1, 1, self.H]), self.f_rnn)
        toks, h = self._draft(d)
        host = self._host(toks).reshape(-1).long().tolist()
        return (toks, host, h) if return_hidden else (toks, host)

    # ------------------------------------------------------------------ traced step
    def set_inputs(self, anchor: int, p0: int, n: int):
        """Host write of the step inputs: anchor token at position p0 + n; commit rows p0 .. p0+n-1 (taps rows 0..n-1)."""
        self._write(self.in_anchor, [[int(anchor)]])
        self._write(self.in_p0, [[int(p0)]])
        self._write(self.in_n, [[int(n)]])

    def _traced_body(self, taps_buf, d, commit):
        self._ctrl()
        if commit:
            t = taps_buf if int(taps_buf.shape[2]) == RC else ttnn.slice(taps_buf, [0, 0, 0, 0], [1, 4, RC, self.H])
            self._commit_rows(t, self.idx_c, self.in_p0, self.idx_n)
        toks, _ = self._draft(d)
        return toks

    def set_gumbel(self, tg):
        """Sampled drafts: tg = T * Gumbel noise, torch [d, TOPK] (row k: step k's candidates in rank order; zeros = the
        top-1, i.e. greedy over the softcapped logits). Written in place: call before each replay()."""
        x = torch.zeros(self.d_max, TOPK)
        x[: tg.shape[0]] = tg.float()
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(x, dtype=F32, layout=RM, mesh_mapper=self.rep), self.gum)

    def cand_host(self, d=None):
        """Sampled drafts: (ids int64 [d, TOPK], softcapped FP32 logits [d, TOPK]) of each step's candidates (descending;
        the drafter's q at temperature T = softmax(logits / T) over them)."""
        d = d or self.d_max
        ids = self._host(self.cand_id).reshape(self.d_max, TOPK)[:d].long()
        return ids, self._host(self.cand_lg).reshape(self.d_max, TOPK)[:d].float()

    def capture(self, taps_buf=None, d: int = 3, commit: bool = True, sample: bool = False):
        """Trace ctrl [+ commit of RC rows of taps_buf ([1, 4, R >= RC, H] bf16 TILE, kept at its address)] + draft(d).
        The caller writes in_anchor / in_p0 / in_n (set_inputs) and the taps buffer before each replay(); sampled
        drafts (sample) also the noise (set_gumbel)."""
        assert not commit or taps_buf is not None
        self.sample = sample
        self._traced_body(taps_buf, d, commit)  # (compile every program before the capture)
        ttnn.synchronize_device(self.mesh)
        self.trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.tokens = self._traced_body(taps_buf, d, commit)
        ttnn.end_trace_capture(self.mesh, self.trace, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        return self.trace

    def replay(self, blocking=False):
        ttnn.execute_trace(self.mesh, self.trace, cq_id=0, blocking=blocking)
        return self.tokens

    def tokens_host(self):
        return self._host(self.tokens).reshape(-1).long().tolist()

    def release(self):
        if self.trace is not None:
            ttnn.release_trace(self.mesh, self.trace)
            self.trace = None
