# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative drafter (the checkpoint's mtp.safetensors), traceable, replicated on every die of the mesh.

Follows reference/dspark_ref.py (the spec) except for roundings that only move near-ties (the target verifies every
draft): the Markov head runs in fp32 without its bf16 roundings, the step logits skip the softcap (monotonic) and the
bf16 rounding of base + bias; the big matmuls use bfp8 weights (DRAFT_W_DT / DRAFT_HEAD_DT / DRAFT_FOLD_DT).

Device state (every shape fixed; positions live in device tensors, so the whole step replays from a trace):
  in_anchor / in_p0 / in_n   uint32 [1, 1] ROW_MAJOR inputs: the anchor token (at position p = p0 + n), the first
                             committed row's position, the number of committed rows
  ctrl program               p0, n -> RoPE row indices, f_rnn row, attention masks (kernels/dsp_ctrl.cpp)
  caches                     per layer K / V [1, 8, S, 128] bf16 TILE: the global layer by absolute position
                             (S = max_len + 512), the sliding layers (0, 2) as a ring of S = 1568 rows (window 1024 +
                             a 512-row prefill chunk + 32: rows written past the context never evict a live row);
                             written by kernels/dsp_kvw.cpp at (p0 + i) mod S
  commit(rows R)             mix_rmsnorm of the taps per layer -> k_proj_ctx / v_proj_ctx -> k_norm_ctx -> RoPE ->
                             cache rows p0 .. p0+R-1 (rows >= n land past the context: overwritten later, masked);
                             f_rnn of row n-1 kept in a device buffer
  draft(d)                   8-row block (row 0 = embed_norm(embed(anchor)), rows 1..7 mask_embed) through the 3
                             layers (non-causal block, all context keys < p, sliding layers >= p - 1024); base logits
                             of rows 1..d into a compact FP32 buffer (BaseHead: the LM head streamed per DRAM bank,
                             kernels/dsp_head_writer.cpp); d Markov steps: the RNN on one row, then one program per
                             step for [lat, 1] @ [fold W; fold b] + base row -> argmax -> the next token
                             (kernels/dsp_step_*.cpp, the fold weight streamed per bank)

Use: eager    commit(taps, p0, valid) / draft(anchor, ctx_len, d=...)  (host values; each call one program set)
     traced   capture(taps_buf, d) -> trace; set_inputs(anchor, p0, n) (or write in_* on device); replay();
              tokens_host() / self.tokens (uint32 [1, 1, 1, d] ROW_MAJOR on every die)
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

        def vec(t):  # [n] -> fp32 [1, 1, 1, n]
            return ops.up(t.float().reshape(1, 1, 1, -1))

        alpha = torch.softmax(w["fuse_weight"].float(), dim=-1).to(torch.bfloat16).float()  # [L + 1, 4]
        self.alpha = [ops.up(alpha[i].reshape(1, -1, 1, 1)) for i in range(self.L + 1)]
        self.layers = []
        for i in range(self.L):
            p = f"layers.{i}."
            g = lambda n: w[p + n]  # noqa: E731
            self.layers.append(dict(
                dflare=vec(g("dflare_norm.weight")), in_norm=vec(g("input_layernorm.weight")),
                post_norm=vec(g("post_attention_layernorm.weight")),
                q_norm=vec(g("self_attn.q_norm.weight")), k_norm=vec(g("self_attn.k_norm.weight")),
                k_norm_ctx=vec(g("self_attn.k_norm_ctx.weight")),
                qkv=mat(torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                                   g("self_attn.v_proj.weight")])),
                kv_ctx=mat(torch.cat([g("self_attn.k_proj_ctx.weight"), g("self_attn.v_proj_ctx.weight")])),
                o=mat(g("self_attn.o_proj.weight")),
                gu=mat(torch.cat([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")])),
                down=mat(g("mlp.down_proj.weight")),
            ))
        self.norm = vec(w["norm.weight"])
        self.rnn_norm = vec(w["markov_head.dflare_norm.weight"])
        tmpl = torch.zeros(1, 1, 32, H)
        tmpl[0, 0, 1:self.B] = w["mask_embed"].float()
        self.tmpl = ops.up(tmpl, dtype=BF16)
        rp = w["markov_head.rnn_proj.weight"]
        self.w_sw = mat(rp[:, :2 * R], BF16)
        self.w_h = mat(rp[:, 2 * R:2 * R + H], BF16)
        self.w_f = mat(rp[:, 2 * R + H:], BF16)
        self.b_rnn = vec(w["markov_head.rnn_proj.bias"])
        self.w1 = ttnn.as_tensor(w["markov_head.markov_w1.weight"], dtype=BF16, layout=RM, device=mesh,
                                 memory_config=DRAM, mesh_mapper=self.rep)
        self.zero_s = ops.up(torch.zeros(1, 1, 1, R))
        self.zero_sb = ops.up(torch.zeros(1, 1, 1, R), dtype=BF16)

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
        self.dl_fold = DecodeLinear(mesh, R + 32, Vp, fold_dtype, readers=int(os.environ.get("DRAFT_FOLD_READERS", "6")),
                                    fidelity=fid(fold_dtype))
        self.fold = self.dl_fold.weight([wf.to(torch.bfloat16)] * N)
        del w

        # RoPE tables (fp32, neox: cos / sin duplicated over the halves), rot(x) = x @ P = [-x2, x1]
        self.SG = (max_len + CHUNK_MAX + 31) // 32 * 32
        self.SW = (self.window + CHUNK_MAX + 32 + 31) // 32 * 32
        npos = self.SG + 64
        inv = 1.0 / (self.theta ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
        ang = torch.arange(npos, dtype=torch.float64)[:, None] * inv[None, :]
        c, s = ang.cos().float(), ang.sin().float()
        # (ttnn.embedding gathers bf16 rows only: each table as hi + lo bf16 halves, summed in fp32 (~2^-17 relative))
        self.rope_t = []
        for t in (torch.cat([c, c], -1), torch.cat([s, s], -1)):
            hi = t.to(torch.bfloat16)
            lo = (t - hi.float()).to(torch.bfloat16)
            self.rope_t.append([ttnn.as_tensor(x, dtype=BF16, layout=RM, device=mesh, memory_config=DRAM,
                                               mesh_mapper=self.rep) for x in (hi, lo)])
        P = torch.zeros(D, D)
        h2 = D // 2
        P[torch.arange(h2) + h2, torch.arange(h2)] = -1.0
        P[torch.arange(h2), torch.arange(h2) + h2] = 1.0
        self.rotP = ops.up(P.reshape(1, 1, D, D), dtype=BF16)

        # caches
        self.S = [self.SW if i in self.local else self.SG for i in range(self.L)]
        self.kc = [self._zeros([1, self.NKV, S, D], BF16) for S in self.S]
        self.vc = [self._zeros([1, self.NKV, S, D], BF16) for S in self.S]

        # device inputs / control outputs / buffers (fixed addresses: traced)
        self.in_anchor, self.in_p0, self.in_n = (self._u32([[0]]) for _ in range(3))
        self.idx_c = self._u32([[0] * RC])
        self.idx_n = self._u32([[0]])
        self.idx_b = self._u32([[0] * 32])
        self.gmask = self._zeros([1, 1, 1, self.SG + 32], F32)
        self.smask = self._zeros([1, 1, 1, self.SW + 32], F32)
        self.base = self._zeros([1, 8, self.NT, 32], F32, RM)
        self.prev = [self._u32([[0]]) for _ in range(d_max)]
        self.f_rnn = self._zeros([1, 1, 1, H], BF16)
        self.chunk_in = {}  # T -> (p0 tensor, rope index tensor, last-row index tensor)
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

    def _mm(self, x, wt, dtype, ck=None):
        return dmm(x, wt, ck or self.ckw, dtype=dtype, memory_config=DRAM)

    def _rope(self, idx):
        """idx uint32 [1, R] -> (C, S) fp32 [1, R, 1, D]."""
        R = int(idx.shape[-1])
        out = []
        for hi, lo in self.rope_t:
            t = ttnn.add(_f32(ttnn.embedding(idx, hi, layout=TILE)), _f32(ttnn.embedding(idx, lo, layout=TILE)))
            out.append(ttnn.reshape(t, (1, R, 1, self.D)))
        return out

    def _head_norm_rope(self, x, nh, w, C, S):
        """x [1, 1, R, nh*D] bf16 -> [1, nh, R, D] bf16: per-head fp32 RMSNorm, neox RoPE in fp32."""
        Rr = int(x.shape[2])
        x = ttnn.reshape(x, (1, Rr, nh, self.D))
        xn = ttnn.typecast(ttnn.rms_norm(_f32(x), epsilon=self.eps, weight=w, compute_kernel_config=self.ckc), BF16)
        rot = ttnn.matmul(xn, self.rotP, compute_kernel_config=self.ckc, dtype=F32)
        y = ttnn.add(ttnn.multiply(_f32(xn), C), ttnn.multiply(rot, S))
        return ttnn.permute(ttnn.typecast(y, BF16), (0, 2, 1, 3))

    def _heads(self, x, nh):
        return ttnn.permute(ttnn.reshape(x, (1, int(x.shape[2]), nh, self.D)), (0, 2, 1, 3))

    def _add(self, a, b):
        return ttnn.typecast(ttnn.add(_f32(a), _f32(b)), BF16)

    # ------------------------------------------------------------------ programs
    def _ctrl(self):
        core = ttnn.CoreCoord(0, 0)
        crs = _crs([core])
        io = [self.in_p0, self.in_n, self.idx_c, self.idx_n, self.idx_b, self.gmask, self.smask]
        ct = [0, RC, self.B, self.SG, self.SW, self.window, _bits(NEG)]
        for t in io:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        k = ttnn.KernelDescriptor(kernel_source=str(KDIR / "dsp_ctrl.cpp"), core_ranges=crs, compile_time_args=ct,
                                  runtime_args=_rt({(0, 0): [t.buffer_address() for t in io]}),
                                  config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                           noc=ttnn.NOC.NOC_0))
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(0, crs, F32, 1024, 1)])
        prog.custom_program_hash = _hash(("dsp_ctrl", tuple(ct)))
        ttnn.generic_op(io, prog)

    def _kvw(self, srcs, p0_t, R):
        """srcs: per layer (K, V) [1, NKV, R, D] bf16 TILE -> cache rows (p0 + i) mod S (one program, one core per
        (cache, head))."""
        grid = self.mesh.compute_with_storage_grid_size()
        jobs = []
        for i, (k, v) in enumerate(srcs):
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
        io = [p0_t] + [t for s in srcs for t in s] + self.kc + self.vc
        ttnn.generic_op(io, prog)

    def _stream(self, dl, kind, x, w, out, ct_w, rt_w, cp_src, cp_ct, cbs_extra, sems, u2d_cbs, defines=()):
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
                                      runtime_args=sub(wr, cores),
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
        prog.custom_program_hash = _hash((kind, tuple(ct_w), tuple(cp_ct), id(dl)))
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

    def _step(self, lat, row, prev):
        """lat [1, 1, 1, R] bf16 -> argmax_v ([lat, 1] @ [fold W; fold b] + base[row])_v into prev (uint32 [1, 1])."""
        dl = self.dl_fold
        cb_base, cb_out, cb_bseg, cb_cand = 2, 16, 3, 4
        n = len(dl.assign)
        red = self.mesh.worker_core_from_logical_core(dl.assign[0][0])
        ct = [0, cb_out, cb_base, cb_bseg, cb_cand, dl.kt, self.V, self.NT, row, n, red.x, red.y] \
            + ttnn.TensorAccessorArgs(lat).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(self.base).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(prev).get_compile_time_args()
        cp_ct = [0, 1, cb_base, cb_out, dl.kt]
        cbs = lambda cores: [_cb(cb_base, cores, F32, 4096, 2), _cb(cb_out, cores, F32, 4096, 2),  # noqa: E731
                             _cb(cb_bseg, cores, F32, dl.cpc * 128, 1), _cb(cb_cand, cores, F32, (n + 2) * 16, 1)]
        sems = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, dl.cores, 0)]
        self._stream(dl, "dsp_step_writer.cpp", lat, self.fold, self.base, ct,
                     lambda slot, col0, nc: [lat.buffer_address(), self.base.buffer_address(), prev.buffer_address(),
                                             col0, nc, slot],
                     KDIR / "dsp_step_compute.cpp", cp_ct, cbs, sems, [cb_base])
        # (prev is written by the program but is not its output tensor: keep it alive in the io list)

    # ------------------------------------------------------------------ commit
    def _commit_rows(self, taps, idx, p0_t, last):
        """taps [1, 4, R, H] bf16 TILE, idx uint32 [1, R] (RoPE rows), p0_t uint32 [1, 1], last uint32 [1, 1] (the row
        whose f_rnn is kept)."""
        R = int(taps.shape[2])
        T = _f32(taps)
        mixes = [ttnn.sum(ttnn.multiply(T, a), dim=1, keepdim=True) for a in self.alpha]  # [1, 1, R, H] fp32
        C, S = self._rope(idx)
        nkv = self.NKV * self.D
        srcs = []
        for i, Lw in enumerate(self.layers):
            ctx = self.ops.rms(mixes[i], Lw["dflare"], out_dtype=BF16)
            kv = self._mm(ctx, Lw["kv_ctx"], BF16)
            k = self._head_norm_rope(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, R, nkv]), self.NKV, Lw["k_norm_ctx"], C, S)
            v = self._heads(ttnn.slice(kv, [0, 0, 0, nkv], [1, 1, R, 2 * nkv]), self.NKV)
            srcs.append((k, v))
        self._kvw(srcs, p0_t, R)
        f = self.ops.rms(mixes[self.L], self.rnn_norm, out_dtype=BF16)  # [1, 1, R, H]
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
        ttnn.synchronize_device(self.mesh)

    def reset(self):
        pass  # (caches are positional: a new sequence overwrites them; masks never read past p)

    # ------------------------------------------------------------------ draft
    def _attention(self, i, q, k, v):
        """q [1, NH, 32, D], k / v [1, NKV, 32, D] bf16 (block rows) vs the whole cache of layer i (masked)."""
        G, D = self.G, self.D
        S = self.S[i]
        mask = self.smask if i in self.local else self.gmask
        qg = ttnn.reshape(q, (1, self.NKV, G * 32, D))
        sc = ttnn.matmul(qg, self.kc[i], transpose_b=True, compute_kernel_config=self.ckc, dtype=F32)
        sb = ttnn.matmul(qg, k, transpose_b=True, compute_kernel_config=self.ckc, dtype=F32)
        s = ttnn.multiply(ttnn.concat([sc, sb], dim=-1), D**-0.5)
        pr = ttnn.softmax(ttnn.add(s, mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ckc)
        o = ttnn.add(ttnn.matmul(ttnn.slice(pr, [0, 0, 0, 0], [1, self.NKV, G * 32, S]), self.vc[i],
                                 compute_kernel_config=self.ckc, dtype=F32),
                     ttnn.matmul(ttnn.slice(pr, [0, 0, 0, S], [1, self.NKV, G * 32, S + 32]), v,
                                 compute_kernel_config=self.ckc, dtype=F32))
        o = ttnn.reshape(o, (1, self.NH, 32, D))
        o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), (1, 1, 32, self.NH * D))
        return ttnn.typecast(o, BF16)

    def block_hidden(self):
        """The block at position p (ctrl outputs) -> final normed hidden [1, 1, 32, H] bf16 (rows 0..B-1 valid)."""
        ops, H = self.ops, self.H
        e = ttnn.embedding(self.in_anchor, self.embed, layout=TILE)
        e = ttnn.pad(ttnn.reshape(e, (1, 1, 1, H)), [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
        hidden = ttnn.add(ops.rms(_f32(e), self.embed_norm, out_dtype=BF16), self.tmpl)
        C, S = self._rope(self.idx_b)
        nq, nkv = self.NH * self.D, self.NKV * self.D
        residual = None
        for i, Lw in enumerate(self.layers):
            residual = hidden if residual is None else self._add(hidden, residual)
            x = ops.rms(residual, Lw["in_norm"], out_dtype=BF16)
            qkv = self._mm(x, Lw["qkv"], BF16)
            q = self._head_norm_rope(ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, 32, nq]), self.NH, Lw["q_norm"], C, S)
            k = self._head_norm_rope(ttnn.slice(qkv, [0, 0, 0, nq], [1, 1, 32, nq + nkv]), self.NKV, Lw["k_norm"],
                                     C, S)
            v = self._heads(ttnn.slice(qkv, [0, 0, 0, nq + nkv], [1, 1, 32, nq + 2 * nkv]), self.NKV)
            hidden = self._mm(self._attention(i, q, k, v), Lw["o"], BF16)
            residual = self._add(hidden, residual)
            x = ops.rms(residual, Lw["post_norm"], out_dtype=BF16)
            gu = self._mm(x, Lw["gu"], BF16)
            I = int(gu.shape[-1]) // 2
            g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 32, I])
            u = ttnn.slice(gu, [0, 0, 0, I], [1, 1, 32, 2 * I])
            act = ttnn.typecast(ttnn.multiply(ttnn.silu(_f32(g)), _f32(u)), BF16)
            hidden = self._mm(act, Lw["down"], BF16)
        residual = self._add(hidden, residual)
        return ttnn.to_memory_config(ops.rms(residual, self.norm, out_dtype=BF16), DRAM)

    def _draft(self, d):
        """ctrl outputs + in_anchor + f_rnn -> d tokens (uint32 [1, 1, 1, d] ROW_MAJOR) and the block hidden."""
        R = self.R
        h = self.block_hidden()
        self._base_logits(h, d)
        cf = ttnn.add(self._mm(self.f_rnn, self.w_f, F32, self.ckc), self.b_rnn)  # [1, 1, 1, 3R]
        cb = ttnn.add(self._mm(h, self.w_h, F32, self.ckc), cf)  # [1, 1, 32, 3R]
        s, sb, prev = self.zero_s, self.zero_sb, self.in_anchor
        for k in range(d):
            wp = ttnn.reshape(ttnn.embedding(prev, self.w1, layout=TILE), (1, 1, 1, R))
            sw = ttnn.concat([sb, wp], dim=-1)
            pj = ttnn.add(self._mm(sw, self.w_sw, F32, self.ckc), ttnn.slice(cb, [0, 0, k + 1, 0], [1, 1, k + 2, 3 * R]))
            gate = ttnn.sigmoid(ttnn.slice(pj, [0, 0, 0, 0], [1, 1, 1, R]))
            tc = ttnn.tanh(ttnn.slice(pj, [0, 0, 0, R], [1, 1, 1, 2 * R]))
            lat = ttnn.typecast(ttnn.tanh(ttnn.slice(pj, [0, 0, 0, 2 * R], [1, 1, 1, 3 * R])), BF16)
            self._step(lat, k + 1, self.prev[k])
            prev = self.prev[k]
            if k + 1 < d:
                s = ttnn.lerp(tc, s, gate)
                sb = ttnn.typecast(s, BF16)
        toks = ttnn.concat([ttnn.reshape(t, (1, 1, 1, 1)) for t in self.prev[:d]], dim=-1) if d > 1 else \
            ttnn.reshape(self.prev[0], (1, 1, 1, 1))
        return toks, h

    def draft(self, anchor, ctx_len: int, last_taps=None, d: int = 3, return_hidden=False):
        """Eager draft at position ctx_len (cache rows < ctx_len committed; f_rnn from the last commit unless last_taps
        (torch [4, H]) is given). Returns (ttnn uint32 [1, 1, 1, d], host list)."""
        assert 1 <= d <= self.d_max
        self.set_inputs(anchor, ctx_len, 0)
        self._ctrl()
        if last_taps is not None:
            x = torch.zeros(1, 4, 32, self.H, dtype=torch.bfloat16)
            x[0, :, 0] = last_taps.reshape(4, self.H)
            t = ttnn.from_torch(x, dtype=BF16, layout=TILE, device=self.mesh, memory_config=DRAM, mesh_mapper=self.rep)
            T = _f32(t)
            f = self.ops.rms(ttnn.sum(ttnn.multiply(T, self.alpha[self.L]), dim=1, keepdim=True), self.rnn_norm,
                             out_dtype=BF16)
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

    def capture(self, taps_buf=None, d: int = 3, commit: bool = True):
        """Trace ctrl [+ commit of RC rows of taps_buf ([1, 4, R >= RC, H] bf16 TILE, kept at its address)] + draft(d).
        The caller writes in_anchor / in_p0 / in_n (set_inputs) and the taps buffer before each replay()."""
        assert not commit or taps_buf is not None
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
