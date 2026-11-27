# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini (transformers `Inkling2LoopForCausalLM`, text) on TT: one sequence, eager (functional bring-up).

Structure (24 layers, hidden 2304; layers 6..17 run 3 passes -> 48 applications, each with its own KV / conv state):

  x = rms_w(embed(ids))                                       (embed_norm)
  blocks = []                                                 (completed attention-residual blocks: layer inputs at
                                                               layers 0, 6, 12, 18 -- every 6th layer)
  layer(x):                                                   x = the current block's partial sum (bf16)
    a = BlockAttnRes(blocks, x; 1 query sliding / 3 queries global (q, k, v inputs))     (fp32 softmax mixing)
    passes 2, 3 of loop layers: q / k / v inputs gated with messages (MessageGate)
    attn = attention(norms(a)) ; attn = sconv(attn)
    resid = attn if boundary (and blocks += [x]) else x + attn
    m = BlockAttnRes(blocks, resid) ; [MessageGate(m, mlp message)] ; mlp = sconv(MoE(rms_w(m)))
    return bf16(resid + mlp)
  after the 3rd pass and layers 18..23: h = rms_w(BlockAttnRes(blocks, x)) ; logits = 30 tanh(lm_head(h) / 30)

Messages for pass p > 1 come from pass p-1: sources [block 12 input of pass p-1, layer 17 output of pass p-1], mixed
by 4 learned queries per loop layer (MessageAttnRes). The short convs' scales are folded into the preceding weights.
"""

from __future__ import annotations

import copy
import math
import os

import torch

import ttnn

from .dmm import dmm
from models.autoports.thinkingmachines_inkling_2_mini.tt.attention import (AttnCache, AttnWeights, GlobalPages, PackedCtx,
                                                                          StaticCtx, attention, attention_packed)
from models.autoports.thinkingmachines_inkling_2_mini.tt.moe import MoE
from models.autoports.thinkingmachines_inkling_2_mini.tt.sconv import ShortConv
from models.autoports.thinkingmachines_inkling_2_mini.tt.weights import Checkpoint, load_config, schedule

DRAM = ttnn.DRAM_MEMORY_CONFIG
SQRT5 = math.sqrt(5.0)
FUSED_OPS = os.environ.get("I2M_OPS", "fused") == "fused"
_DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
ATTN_DT = _DT[os.environ.get("I2M_ATTN_DT", "bfp8")]  # attention projections (q|r|sink|gate, k, v, o)
GATE_FUSED = os.environ.get("I2M_GATE_FUSED", "1") == "1"
GATE2 = os.environ.get("I2M_GATE2", "1") == "1"  # the round-reduced gate on the post-attention rectangle (tt/gate.py GateKernel2)
GATE_KERNEL = os.environ.get("I2M_GATE_KERNEL", "1") == "1"  # fused message-gate tail (tt/gatet.py)
RMS_SHARD = os.environ.get("I2M_RMS_SHARD", "1") == "1"  # one-row norms width-sharded on 8 cores
# I2M_RMS_CAST (default 1): a one-row norm's input cast (bf16 -> fp32) inside its reshard (exact; one op fewer)
RMS_CAST = os.environ.get("I2M_RMS_CAST", "1") == "1"
MIX_KERNEL = os.environ.get("I2M_MIX_KERNEL", "1") == "1"  # fused block mix for one tile row (tt/mixw.py)
# I2M_MIX_PLAN (default 1): prefill block mixes take each block's logits from one matmul per block against every later
# query (tt/batched.py MixPlan), not one matmul per source per mix
MIX_PLAN = os.environ.get("I2M_MIX_PLAN", "1") == "1"
HEAD_DT = _DT[os.environ.get("I2M_HEAD_DT", "bfp8")]  # lm_head
# packed prefill: up to LANES sequences per prefill chunk, each with its own conv-state lane
# (default 8; 4 on one die: each lane's conv states take ~32 MB of DRAM over the 48 applications)
LANES = int(os.environ.get("I2M_PREFILL_LANES", "0"))
# I2M_PF_ROWN (default 1): prefill norms (> 32 rows) as one program per call reading the row once, with the input /
# output casts (tt/rown.py; bitwise equal to typecast + ttnn.rms_norm + typecast)
PF_ROWN = os.environ.get("I2M_PF_ROWN", "1") == "1"
# I2M_PF_GATEF (default 1, with PF_ROWN): prefill message gates: the two norms written straight into the halves of the
# down matmul's input (no casts / concat; the input's norm once for the three attention-side gates) and the forget /
# gain matmul with the tail in one program (tt/gatef.py); bitwise
PF_GATEF = os.environ.get("I2M_PF_GATEF", "1") == "1"
# I2M_PF_ROWN_LG (default 1, with PF_ROWN): a prefill mix's current-source norm computes that source's mix logits in
# the same program (tt/rown.py qrep; bitwise equal to the separate matmul)
PF_ROWN_LG = os.environ.get("I2M_PF_ROWN_LG", "1") == "1"
# I2M_PF_MIXC (default 1): prefill message logits once per pass: both sources against every loop layer's queries in one
# matmul each (the decode's msg_qcat), not two matmuls per application; bitwise
PF_MIXC = os.environ.get("I2M_PF_MIXC", "1") == "1"
# I2M_PF_ROWSPLIT (default on; 4K prefill TP4 773.0 -> 676.8 ms, TP2 1209.4 -> 1154.9): on > 1 die, a packed
# prefill chunk's replicated glue (norms, mixes, logits, gates, messages, the hidden-size convs) runs on each die's
# T / N rows instead of all T: the residual stream, blocks and messages are row-split, the all-reduces stay as they are (their full output gives the convs their halo and every
# segment's state rows), and only the attention inputs and the MoE input are all-gathered (RowSplit)
PF_ROWSPLIT = os.environ.get("I2M_PF_ROWSPLIT", "1") == "1"
# I2M_PF_RSX (with PF_ROWSPLIT; eager chunks): the attention / MoE outputs reduce-scattered by rows, not all-reduced:
# each conv reads its die's rows plus a few side tiles gathered from the others (the halo tile before its rows, the
# state points' tiles; RowSplit.side, sconv_kernels/pfseg_reader.cpp PF_RSX). I2M_PF_RS_LINKS: the reduce-scatter's
# fabric links
PF_RSX = os.environ.get("I2M_PF_RSX", "1") == "1"
RS_LINKS = int(os.environ.get("I2M_PF_RS_LINKS", "2"))


class RowSplit:
    """A chunk of T rows split over the mesh's N dies (die d: rows d T/N .. (d+1) T/N - 1)."""

    def __init__(self, mesh, T):
        N = mesh.get_num_devices()
        self.mesh, self.N, self.T, self.Tl = mesh, N, T, T // N
        k0 = torch.zeros(1, 1, 1, 16 * N, dtype=torch.int32)
        for d in range(N):
            k0[0, 0, 0, 16 * d] = d * self.Tl // 32
        # (each die's first row tile: tt/sconv_kernels/pfseg_reader.cpp PF_ROWOFF)
        self.k0 = ttnn.from_torch(k0, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                  memory_config=DRAM, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3))

    @staticmethod
    def ok(mesh, T):
        N = mesh.get_num_devices()
        return PF_ROWSPLIT and N > 1 and T % (32 * N) == 0 and T // N > 32

    def part(self, t):
        """[1, 1, T, W] (the same on every die) -> this die's rows"""
        return ttnn.mesh_partition(t, 2, cluster_axis=1, memory_config=DRAM)

    def gather(self, t):
        """this die's rows -> [1, 1, T, W] on every die"""
        return ttnn.all_gather(t, 2, cluster_axis=1, memory_config=DRAM, topology=ttnn.Topology.Linear)

    def conv(self):
        return (self.k0, self.Tl)

    def slots(self, segs):
        """PF_RSX: the local row tiles every die contributes to the conv's side tensor, the same list on every die:
        its last (the next die's halo), then every tile holding a state point's rows (segs: (row0, n, snap | None)),
        padded with the first to 4 / 8 / 16 (few gather programs). None: more than 16."""
        Rt = self.Tl // 32
        sl = [Rt - 1]
        for row0, n, snap in segs:
            for P in (n, snap):
                if P is None:
                    continue
                for r in range(3):
                    if P + r >= 3:
                        l = ((row0 + P + r - 3) // 32) % Rt
                        if l not in sl:
                            sl.append(l)
        m = next((m for m in (4, 8, 16) if len(sl) <= m), None)
        return None if m is None else sl + [sl[0]] * (m - len(sl))

    def side(self, x, slots, bc):
        """PF_RSX: this die's local row tiles `slots` of x [1, 1, Tl, C] (one copy program), gathered from every die
        -> [1, 1, N * 32 * len(slots), C] (die o's at tile rows o * len(slots) ..)"""
        C = int(x.shape[-1])
        cols = C // 32
        m = len(slots)
        c = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32 * m, C]), x.dtype, ttnn.TILE_LAYOUT, self.mesh, DRAM)
        bc([(x, l * cols, c, s * cols, cols) for s, l in enumerate(slots)])
        A = ttnn.all_gather(c, 2, cluster_axis=1, memory_config=DRAM, topology=ttnn.Topology.Linear)
        ttnn.deallocate(c)
        return A

    def scatter(self, t):
        """[1, 1, T, W] partial sums on every die -> the sum's rows of this die [1, 1, Tl, W] (PF_RSX)"""
        return ttnn.reduce_scatter(t, 2, cluster_axis=1, topology=ttnn.Topology.Linear, memory_config=DRAM,
                                   num_links=RS_LINKS)

    def cut(self, t, rows, bc, dyn=None):
        """rows (global row ids, <= 32) of a row-split [1, 1, Tl, W] -> [1, 1, 32, W] tile on every die: each die
        copies row r mod Tl of its slice for every r (the same copy list on every die), one gather of the picks, then
        row r from its owner's block. dyn (StaticCtx, one row): the two rows from its words 18 / 19."""
        W = int(t.shape[-1])
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        g = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, W]), t.dtype, ttnn.ROW_MAJOR_LAYOUT, self.mesh, DRAM)
        if dyn is not None:
            bc([(rm, 0, g, 0, 1, 18)], dyn=dyn)
        else:
            bc([(rm, r % self.Tl, g, i, 1) for i, r in enumerate(rows)])
        ttnn.deallocate(rm)
        gt = ttnn.to_layout(g, ttnn.TILE_LAYOUT)
        ttnn.deallocate(g)
        G = self.gather(gt)
        ttnn.deallocate(gt)
        Grm = ttnn.to_layout(G, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(G)
        o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, W]), t.dtype, ttnn.ROW_MAJOR_LAYOUT, self.mesh, DRAM)
        if dyn is not None:
            bc([(Grm, 0, o, 0, 1, 19)], dyn=dyn)
        else:
            bc([(Grm, (r // self.Tl) * 32 + i, o, i, 1) for i, r in enumerate(rows)])
        ttnn.deallocate(Grm)
        return ttnn.to_layout(o, ttnn.TILE_LAYOUT)


def _lane_bufs(cv):
    """A conv lane's packed-prefill buffers: its next state and its prefix-snapshot state (allocated before any
    trace capture)."""
    cv._pf_stn = ttnn.clone(cv._zero)
    cv.snap_buf = ttnn.clone(cv._zero)
    return cv


def _lane_copy(cv):
    """Another sequence lane of a conv: the same taps / programs, its own state (allocated now, before any capture)."""
    c = copy.copy(cv)
    c.st = ttnn.clone(cv._zero)
    c.state = [c.st]
    c._last_seq = None
    return _lane_bufs(c)


def _f32(t):
    return ttnn.typecast(t, ttnn.float32) if t.dtype != ttnn.float32 else t


def _bf(t):
    return ttnn.typecast(t, ttnn.bfloat16) if t.dtype != ttnn.bfloat16 else t


class Ops:
    """Small shared helpers bound to a mesh."""

    def __init__(self, mesh, eps):
        self.mesh, self.eps = mesh, eps
        self.N = mesh.get_num_devices()
        self.rep = ttnn.ReplicateTensorToMesh(mesh) if self.N > 1 else None
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=False)
        self.qrep = {}  # id(Q) -> Qrep [1,1,H,nq*32] (tt/mixw.py)
        self.lgx = None  # (tt/lgx.py LogitExpand, I2M_LG_PACK)
        self.mixk = None
        self.gatek = None
        self.gatefk = None
        self.gate2k = None  # (set by the decoder when the post-attention rectangle exists: tt/gate.py GateKernel2)
        self.rms_cfg = {}
        self.rown = None
        self.gatepf = None  # (tt/gatef.py, the prefill gate)

    def query(self, Qcols):
        """Qcols [H, nq] (pre-scaled) -> Q [1,1,H,32] fp32 (columns 0..nq-1); also builds the mix kernel's Qrep."""
        H, nq = Qcols.shape
        Q = torch.zeros(H, 32)
        Q[:, :nq] = Qcols
        t = self.up(Q.reshape(1, 1, H, 32))
        self.qrep[id(t)] = self.up(Qcols.repeat_interleave(32, dim=1).reshape(1, 1, H, nq * 32))
        return t

    def up(self, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=self.mesh, memory_config=DRAM,
                               mesh_mapper=self.rep)

    def rms(self, x, w=None, out_dtype=None):
        """fp32 RMS norm over the last dim (transformers Inkling2RMSNorm / UnweightedRMSNorm), w [1,1,1,H] fp32.
        I2M_OPS=fused (default): one ttnn.rms_norm; ref: the explicit fp32 ops. out_dtype: the result cast to it (one
        tile row: inside the reshard back, RMS_CAST; otherwise a typecast)."""
        if FUSED_OPS and RMS_SHARD and len(x.shape) == 4 and int(x.shape[0]) * int(x.shape[1]) == 1 \
                and int(x.shape[2]) <= 32 and int(x.shape[3]) % 256 == 0 and RMS_CAST:
            mc, pc = self._rms_cfg(int(x.shape[3]))
            xs = ttnn.to_memory_config(x, mc, dtype=ttnn.float32) if x.dtype != ttnn.float32 else \
                ttnn.to_memory_config(x, mc)
            y = ttnn.rms_norm(xs, epsilon=self.eps, weight=w, program_config=pc, memory_config=mc,
                              compute_kernel_config=self.ckc)
            y = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)
            # (the output cast stays a typecast: the reshard's fp32 -> bf16 conversion rounds differently, 1 ulp)
            return ttnn.typecast(y, out_dtype) if out_dtype is not None and out_dtype != ttnn.float32 else y
        if self.rown_ok(x):
            return self.rown(x, w, out_dtype=out_dtype or ttnn.float32)
        y = self._rms(x, w)
        return ttnn.typecast(y, out_dtype) if out_dtype is not None and y.dtype != out_dtype else y

    def rown_ok(self, x):
        """prefill rows (> 32, whole tiles) go through tt/rown.py (I2M_PF_ROWN)"""
        if not (PF_ROWN and FUSED_OPS and len(x.shape) == 4 and int(x.shape[2]) > 32):
            return False
        if self.rown is None:
            from .rown import RowNorm
            self.rown = RowNorm(self.mesh, int(x.shape[3]), self.eps)
        return self.rown.ok(x)

    def _rms_cfg(self, H):
        if H not in self.rms_cfg:
            mc = ttnn.create_sharded_memory_config(shape=(32, H // 8), core_grid=ttnn.CoreGrid(x=8, y=1),
                                                   strategy=ttnn.ShardStrategy.WIDTH,
                                                   orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                   use_height_and_width_as_shard_shape=True)
            bw = H // 8 // 32
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
                subblock_w=max(d for d in range(1, 5) if bw % d == 0), block_h=1, block_w=bw, inplace=False)
            self.rms_cfg[H] = (mc, pc)
        return self.rms_cfg[H]

    def _rms(self, x, w=None):
        x = _f32(x)
        if FUSED_OPS and RMS_SHARD and len(x.shape) == 4 and int(x.shape[0]) * int(x.shape[1]) == 1 \
                and int(x.shape[2]) <= 32 and int(x.shape[3]) % 256 == 0:
            # one tile row: width-sharded multi-core norm (8 cores) incl. the reshards, 29 -> 10 us (tools/bench_norm.py)
            H = int(x.shape[3])
            if H not in self.rms_cfg:
                mc = ttnn.create_sharded_memory_config(shape=(32, H // 8), core_grid=ttnn.CoreGrid(x=8, y=1),
                                                       strategy=ttnn.ShardStrategy.WIDTH,
                                                       orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                       use_height_and_width_as_shard_shape=True)
                bw = H // 8 // 32
                pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
                    subblock_w=max(d for d in range(1, 5) if bw % d == 0), block_h=1, block_w=bw, inplace=False)
                self.rms_cfg[H] = (mc, pc)
            mc, pc = self.rms_cfg[H]
            y = ttnn.rms_norm(ttnn.to_memory_config(x, mc), epsilon=self.eps, weight=w, program_config=pc,
                              memory_config=mc, compute_kernel_config=self.ckc)
            return ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)
        if FUSED_OPS:
            return ttnn.rms_norm(x, epsilon=self.eps, weight=w, compute_kernel_config=self.ckc)
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        y = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ms, self.eps)))
        return ttnn.multiply(y, w) if w is not None else y

    def mix(self, srcs_normed, Q, nq):
        """Block attention residual: srcs_normed list of n fp32 [1,1,T,H] (rms-normed sources), Q [1,1,H,32] fp32
        (nq queries in columns, pre-scaled by H^-0.5) -> nq fp32 outputs sum_i softmax_i(src_i . q) src_i."""
        T = int(srcs_normed[0].shape[2])
        n = len(srcs_normed)
        if MIX_KERNEL and FUSED_OPS and (T <= 32 or T % 32 == 0) and n <= 6 and id(Q) in self.qrep:
            if self.mixk is None:
                from .mixw import MixKernel
                self.mixk = MixKernel(self.mesh, int(srcs_normed[0].shape[3]))
            mc = ttnn.L1_MEMORY_CONFIG if T <= 32 else DRAM  # (prefill chunks: the stacked sources are tens of MB)
            src = [ttnn.sharded_to_interleaved(s, mc) if s.is_sharded() else s for s in srcs_normed]
            if T > 32 and len({repr(s.memory_config()) for s in src}) == 1:
                # prefill: per-source logits, the kernel reads the sources in place (no stacking copy)
                lgs = [ttnn.matmul(s, self.qrep[id(Q)], compute_kernel_config=self.ckc, dtype=ttnn.float32,
                                   memory_config=mc) for s in src]
                return self.mixk(src, lgs, n, nq, out_mc=mc)
            S = src[0] if n == 1 else ttnn.concat(src, dim=1, memory_config=mc)
            lg = dmm(S, self.qrep[id(Q)], self.ckc, dtype=ttnn.float32, memory_config=mc)  # [1, n, T, nq*32]
            return self.mixk(S, lg, n, nq, out_mc=mc)
        if FUSED_OPS:
            # Qrep [H, nq*32]: query q in all 32 columns of tile q -> every slice below starts on a tile boundary
            # (a width-1 slice at an unaligned column of a [1, n, T, 32] tile tensor returns wrong weights for n > 1)
            S = srcs_normed[0] if n == 1 else ttnn.concat(srcs_normed, dim=1)  # [1, n, T, H]
            lg = dmm(S, self.qrep[id(Q)], self.ckc, dtype=ttnn.float32)  # [1, n, T, nq*32]
            m = ttnn.max(lg, dim=1, keepdim=True)
            e = ttnn.exp(ttnn.subtract(lg, m))
            w = ttnn.multiply(e, ttnn.reciprocal(ttnn.sum(e, dim=1, keepdim=True)))
            outs = []
            for qi in range(nq):
                wq = ttnn.slice(w, [0, 0, 0, qi * 32], [1, n, T, qi * 32 + 1])  # [1, n, T, 1]
                outs.append(ttnn.sum(ttnn.multiply(S, wq), dim=1, keepdim=True))  # [1, 1, T, H]
            return outs
        logits = [ttnn.matmul(s, Q, compute_kernel_config=self.ckc, dtype=ttnn.float32) for s in srcs_normed]
        m = logits[0]
        for l in logits[1:]:
            m = ttnn.maximum(m, l)
        ex = [ttnn.exp(ttnn.subtract(l, m)) for l in logits]
        z = ex[0]
        for e in ex[1:]:
            z = ttnn.add(z, e)
        rz = ttnn.reciprocal(z)
        T = int(srcs_normed[0].shape[2])
        outs = []
        for qi in range(nq):
            acc = None
            for s, e in zip(srcs_normed, ex):
                wq = ttnn.slice(ttnn.multiply(e, rz), [0, 0, 0, qi], [1, 1, T, qi + 1])  # [1,1,T,1]
                term = ttnn.multiply(s, wq)
                acc = term if acc is None else ttnn.add(acc, term)
            outs.append(acc)
        return outs


# I2M_GATE_PC (default 1): prefill rows' gate down matmul [T, 2H] x [2H, r] with the rows split over the cores (1D, the
# weight multicast): the default config ran ~10 TFLOPS (512 rows 0.111 -> 0.038 ms; bitwise)
GATE_PC = os.environ.get("I2M_GATE_PC", "1") == "1"


def _gate_pc(mesh, T, N):
    g = mesh.compute_with_storage_grid_size()
    Mt = T // 32
    pm = -(-Mt // (g.x * g.y))
    while Mt % pm:
        pm += 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), in0_block_w=8, out_subblock_h=1, out_subblock_w=N // 32,
        per_core_M=pm, per_core_N=N // 32, fuse_batch=True, fused_activation=None, mcast_in0=False)


class MessageGate:
    """(1 - tanh(Wf g)) * h + tanh(Wg g) * m, g = Wd [rms(h) | rms(m)] (bf16 Linear in transformers)."""

    def __init__(self, ops: Ops, ck, li, name):
        L = lambda n: ck.L(li, f"controllers.{name}.{n}")  # noqa: E731
        d = L("down_weight").float()  # [r, 2H]
        H = d.shape[1] // 2
        self.ops, self.H = ops, H
        self.wd = ops.up(d.T.reshape(1, 1, 2 * H, -1), ttnn.bfloat16)  # [2H, r]
        # [forget | gain] in one fp32 matmul
        wfg = torch.cat([L("forget_weight").float().T, L("gain_weight").float().T], dim=1)  # [r, 2H]
        self.wfg = ops.up(wfg.reshape(1, 1, d.shape[0], 2 * H), ttnn.float32)
        if GATE_FUSED:  # the one-program gate (tt/gate.py): bf16 forget / gain weights (the checkpoint's, exact)
            self.wfg_bf = ops.up(wfg.reshape(1, 1, d.shape[0], 2 * H), ttnn.bfloat16)

    def fused_args(self, h, m, nw_rows):
        """(h, m, wd, wf|wg, nw) for GateKernel2.multi, or None (not available here)."""
        o = self.ops
        hf, mf = _f32(h), _f32(m)
        if not (GATE2 and o.gate2k is not None and len(hf.shape) == 4 and int(hf.shape[1]) == 1
                and int(hf.shape[-2]) <= 16):
            return None
        return hf, mf, self.wd, self.wfg_bf, nw_rows

    def fused_norm(self, h, m, nw_rows, router=None, dup=False):
        """The gate + the following RMS norm (nw_rows: [1, 1, 32, H] fp32) in one program, with the router logits
        (router: bf16 hi / lo [1, 1, H, E]) -> (bf16 normed x, logits | None). None: not available here."""
        o = self.ops
        hf, mf = _f32(h), _f32(m)
        if not (GATE2 and o.gate2k is not None and len(hf.shape) == 4 and int(hf.shape[1]) == 1
                and int(hf.shape[-2]) <= 16):
            return None
        if router is not None and int(hf.shape[-2]) > o.gate2k.ROUTER_ROWS:
            router = None
        return o.gate2k(hf, mf, self.wd, self.wfg_bf, nw=nw_rows, router=router, dup=dup)

    def pf_ok(self, h):
        """the prefill gate path (I2M_PF_GATEF): rows > 32 in whole tiles"""
        o = self.ops
        return PF_GATEF and GATE_FUSED and GATE_KERNEL and GATE_PC and o.rown_ok(h) and int(h.shape[1]) == 1

    def hm_buf(self, T):
        """the down matmul's input [1, 1, T, 2H] bf16: [rms(h) | rms(m)]"""
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, 2 * self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                              self.ops.mesh, DRAM)

    def __call__(self, h, m, hm=None, m_done=False):
        """hm (prefill path): the down matmul's input with its rms(h) half (m_done: and its rms(m) half) written"""
        o, H = self.ops, self.H
        hf, mf = _f32(h), _f32(m)
        if self.pf_ok(hf):
            T = int(hf.shape[-2])
            bf = ttnn.bfloat16
            if hm is None:  # (both norms in one program)
                hm = self.hm_buf(T)
                o.rown.multi([dict(x=hf, out_dtype=bf, dests=[(hm, 0)]), dict(x=mf, out_dtype=bf, dests=[(hm, H // 32)])])
            elif not m_done:
                o.rown(mf, None, out_dtype=bf, dests=[(hm, H // 32)])
            g = ttnn.matmul(hm, self.wd, compute_kernel_config=o.ckc, dtype=ttnn.bfloat16,
                            program_config=_gate_pc(o.mesh, T, int(self.wd.shape[-1])))
            ttnn.deallocate(hm)
            if o.gatepf is None:
                from .gatef import GateFused
                o.gatepf = GateFused(o.mesh, H, int(self.wd.shape[-1]))
            out = o.gatepf(g, self.wfg_bf, hf, mf)
            ttnn.deallocate(g)
            return out
        if GATE2 and o.gate2k is not None and int(hf.shape[-2]) <= 16 and len(hf.shape) == 4 and int(hf.shape[1]) == 1:
            return o.gate2k(hf, mf, self.wd, self.wfg_bf)
        if GATE_FUSED and int(hf.shape[-2]) <= 32 and len(hf.shape) == 4 and int(hf.shape[1]) == 1:
            if o.gatefk is None:
                from .gate import GateKernel
                o.gatefk = GateKernel(o.mesh, H, int(self.wd.shape[-1]), o.eps)
            return o.gatefk(hf, mf, self.wd, self.wfg_bf)
        hm = ttnn.concat([_bf(o.rms(hf)), _bf(o.rms(mf))], dim=-1)  # [.., 2H] bf16 (transformers: bf16 Linear input)
        T = int(hf.shape[-2])
        if GATE_PC and T % 32 == 0 and 64 <= T and len(hf.shape) == 4 and int(hf.shape[1]) == 1:
            g = ttnn.matmul(hm, self.wd, compute_kernel_config=o.ckc, dtype=ttnn.bfloat16,
                            program_config=_gate_pc(o.mesh, T, int(self.wd.shape[-1])))
        else:
            g = dmm(hm, self.wd, o.ckc, dtype=ttnn.bfloat16)  # bf16 Linear output
        if GATE_KERNEL and (T <= 32 or T % 32 == 0) and len(hf.shape) == 4 and int(hf.shape[1]) == 1:
            mc = ttnn.L1_MEMORY_CONFIG if T <= 32 else DRAM
            fg = dmm(_f32(g), self.wfg, o.ckc, dtype=ttnn.float32, memory_config=mc)  # pre-tanh [.., 2H]
            if o.gatek is None:
                from .gatet import GateTail
                o.gatek = GateTail(o.mesh, H)
            return o.gatek(fg, hf, mf, out_mc=mc)
        fg = ttnn.tanh(ttnn.matmul(_f32(g), self.wfg, compute_kernel_config=o.ckc, dtype=ttnn.float32))  # [.., 2H]
        lead = [int(d) for d in hf.shape][:-1]
        f = ttnn.slice(fg, [0] * (len(lead)) + [0], lead + [H])
        a = ttnn.slice(fg, [0] * (len(lead)) + [H], lead + [2 * H])
        return ttnn.add(ttnn.subtract(hf, ttnn.multiply(f, hf)), ttnn.multiply(a, mf))


def _res_add(resid, y, out_dtype):
    out = ttnn.add(_f32(resid), _f32(y))
    return ttnn.typecast(out, out_dtype) if out_dtype != ttnn.float32 else out


class Layer:
    """Weights of one decoder layer (shared by its passes)."""

    def __init__(self, ops: Ops, cfg, ck, li, expert_dtype, attn_dtype, cache_dir, tt_ccl):
        self.li, self.ops = li, ops
        self.sliding = li in cfg.local_layer_ids
        self.boundary = li % cfg.n_layers_per_attn_res_block == 0
        self.loop = cfg.loop_start <= li < cfg.loop_end
        H = cfg.hidden_size
        sc = H**-0.5
        wv = lambda n: ops.up(ck.L(li, n).float().reshape(1, 1, 1, -1))  # noqa: E731
        q = ck.attn_query(li)
        self.nq = 0 if q is None else q.shape[0]
        if q is not None:
            self.attn_Q = ops.query(q.float().T * sc)
        self.mlp_Q = ops.query(ck.mlp_query(li).float().reshape(-1, H).T * sc)
        if self.sliding:
            self.in_norm = wv("attn_norm.weight")
        else:
            self.q_norm, self.k_norm, self.v_norm = (wv(f"{n}_input_norm.weight") for n in "qkv")
        self.mlp_norm = wv("mlp_norm.weight")
        self.attn = AttnWeights(ops.mesh, cfg, ck, li, wdtype=attn_dtype)
        self.moe = MoE(ops.mesh, cfg, ck, li, expert_dtype=expert_dtype, cache_dir=cache_dir, tt_ccl=tt_ccl)
        self.attn_conv_w = ck.L(li, "attn_sconv.weight").reshape(H, -1).float()
        self.mlp_conv_w = ck.L(li, "mlp_sconv.weight").reshape(H, -1).float()
        if self.loop:
            self.gates = {n: MessageGate(ops, ck, li, n) for n in ("attn", "k", "v", "mlp")}


class AppState:
    """Per application (layer, pass): attention caches + the two hidden-size conv states."""

    def __init__(self, layer: Layer, max_seq_len, kv_dtype, prefill_chunk, slots=1, pages=None, lanes=1):
        mesh = layer.ops.mesh
        H = layer.attn_conv_w.shape[0]
        self.cache = AttnCache(layer.attn, max_seq_len, kv_dtype=kv_dtype, prefill_chunk=prefill_chunk, slots=slots,
                               pages=pages)
        rep = ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None
        self.attn_conv = ShortConv(mesh, layer.attn_conv_w, rep, H)
        self.mlp_conv = ShortConv(mesh, layer.mlp_conv_w, rep, H)

        self.lanes = [tuple(_lane_bufs(cv) for cv in (self.cache.k_conv, self.cache.v_conv, self.attn_conv,
                                                       self.mlp_conv))]
        for _ in range(lanes - 1):
            self.lanes.append(tuple(_lane_copy(cv) for cv in self.lanes[0]))

    def reset(self, lane=None):
        for i, ln in enumerate(self.lanes):
            if lane is None or i == lane:
                for cv in ln:
                    cv.reset()

    def conv(self, which, x, rows_mode, nvalid, resid=None, out_dtype=ttnn.bfloat16):
        """resid: -> out_dtype(f32(resid) + f32(conv(x))) (the residual add after the conv; in the prefill kernel)."""
        c = self.attn_conv if which == "attn" else self.mlp_conv
        if rows_mode:
            y = c.rows(x, commit_all=True)
            return y if resid is None else _res_add(resid, y, out_dtype)
        return c.prefill_commit(x, nvalid, resid=resid, out_dtype=out_dtype)


class Inkling2Model:
    def __init__(self, mesh, max_seq_len=8192, expert_dtype=ttnn.bfloat4_b, attn_dtype=None,
                 kv_dtype=ttnn.bfloat8_b, prefill_chunk=1024, cache_dir=None, tt_ccl=None, layers=None, slots=1,
                 kv_pool_tokens=None):
        self.mesh = mesh
        attn_dtype = ATTN_DT if attn_dtype is None else attn_dtype
        self.cfg = cfg = load_config()
        self.ck = ck = Checkpoint(cfg.snap)
        self.ops = ops = Ops(mesh, cfg.rms_norm_eps)
        self.H = H = cfg.hidden_size
        self.sched = schedule(cfg)
        self.prefill_chunk = prefill_chunk
        self._rs_by_T = {}
        if RowSplit.ok(mesh, prefill_chunk):  # (its per-die tile offsets allocated before any trace capture)
            self._rs_by_T[prefill_chunk] = RowSplit(mesh, prefill_chunk)
        if cache_dir is None:
            cache_dir = os.environ.get("I2M_WCACHE")
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        if tt_ccl is None and mesh.get_num_devices() > 1:
            from models.tt_transformers.tt.ccl import TT_CCL

            tt_ccl = TT_CCL(mesh)
        self.tt_ccl = tt_ccl
        from models.autoports.thinkingmachines_inkling_2_mini.tt import ccl as _ccl

        _ccl.prepare(mesh, cfg.hidden_size)  # (decode all-reduce scratch: before any trace capture)
        from models.autoports.thinkingmachines_inkling_2_mini.tt import prefetch as _pf

        _pf.SLOTS = slots  # (the o prefetch only up to I2M_PF_O_SLOTS slots: its L1)
        lids = range(cfg.num_hidden_layers) if layers is None else layers
        self.layers = {}
        for li in lids:
            t0 = __import__("time").time()
            self.layers[li] = Layer(ops, cfg, ck, li, expert_dtype, attn_dtype, cache_dir, tt_ccl)
            print(f"layer {li} loaded in {__import__('time').time() - t0:.0f}s", flush=True)
        self.slots = slots
        # the global layers' block tables (shared; vLLM's block ids when vLLM pages the global KV)
        self.pages = GlobalPages(mesh, slots, math.ceil(max_seq_len / 64),
                                 pool_blocks=math.ceil(kv_pool_tokens / 64) + 64 if kv_pool_tokens else None)
        # (+ 64 blocks: the TT plugin allocates a few blocks past the budget it is given)
        self.lanes = LANES or (4 if mesh.get_num_devices() == 1 else 8)
        self.apps = [AppState(self.layers[li], max_seq_len, kv_dtype, prefill_chunk, slots, self.pages, self.lanes)
                     if li in self.layers else None
                     for li, p in self.sched]
        print(f"application states up ({sum(a is not None for a in self.apps)})", flush=True)
        # message queries [n_loop, 4, H] -> per loop layer [1,1,H,32] (4 query columns, pre-scaled)
        mq = ck.message_queries()
        self.msg_Q = []
        for i in range(mq.shape[0]):
            self.msg_Q.append(ops.query(mq[i].float().T * H**-0.5))
        # every loop layer's message queries side by side ([1, 1, H, n_loop 4 32], each query in 32 columns): a decode
        # pass computes its two sources' logits against all of them in one matmul (tt/batched.py _messages)
        self.msg_qcat = ops.up(torch.cat([(mq[i].float().T * H**-0.5).repeat_interleave(32, dim=1)
                                          for i in range(mq.shape[0])], 1).reshape(1, 1, H, -1))
        from . import lgx

        if lgx.ON:  # (the same queries packed: tt/lgx.py)
            self.msg_qpk = lgx.pack_cols(torch.cat([mq[i].float().T * H**-0.5 for i in range(mq.shape[0])], 1),
                                         self.mesh, ops.rep)
            self.msg_nq = 4 * mq.shape[0]
        self.final_Q = self.ops.query(ck.final_query().float().reshape(-1, H).T * H**-0.5)
        self.final_norm = ops.up(ck.get("model.llm.norm.weight").float().reshape(1, 1, 1, -1))
        self.embed_norm = ops.up(ck.get("model.llm.embed_norm.weight").float().reshape(1, 1, 1, -1))
        V = cfg.unpadded_vocab_size
        self.V = V
        emb = ck.get("model.llm.embed.weight")
        self.embed = ttnn.as_tensor(emb.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                    device=mesh, memory_config=DRAM, mesh_mapper=ops.rep)  # no ttnn cache: large cached uploads hang
        # I2M_HEAD_SHARD (default 1 on > 1 die): the LM head split over the vocab (each die computes Vp / N logits; the
        # device top-k runs per die and the host merges the candidates; logits_host gathers full rows)
        N = mesh.get_num_devices()
        self.head_shard = N > 1 and os.environ.get("I2M_HEAD_SHARD", "1") == "1"
        q = 32 * (N if self.head_shard else 1)
        Vp = (V + q - 1) // q * q
        un = ck.get("model.llm.unembed.weight")[:V].float()
        un = torch.nn.functional.pad(un, (0, 0, 0, Vp - V))
        # I2M_HEAD_STREAM (default 1; one die, or each die's vocab slice): the head streamed per DRAM bank with the
        # softcap in the same program (tt/head.py; its per-bank weight copy replaces the interleaved one)
        self.shead = None
        if (N == 1 or self.head_shard) and os.environ.get("I2M_HEAD_STREAM", "1") == "1":
            from .head import StreamHead

            Vs = Vp // N
            self.shead = StreamHead(mesh, H, Vs, HEAD_DT, cfg.final_logit_softcapping,
                                    readers=int(os.environ.get("I2M_HEAD_READERS", "2")),
                                    # (I2M_HEAD_BUFFERS: weight columns per reader ring; 2 with the live-row softcap
                                    # streams fastest: P150 one die / TP2 / TP4 slice 1.30 / 0.66 / 0.34 -> 1.06 / 0.54 /
                                    # 0.28 ms)
                                    buffers=int(os.environ.get("I2M_HEAD_BUFFERS", "2")),
                                    fidelity=ttnn.MathFidelity.HiFi4 if os.environ.get("I2M_HEAD_HIFI4") == "1"
                                    else ttnn.MathFidelity.HiFi2)
            self.lm_head = self.shead.weight([un.T[:, d * Vs : (d + 1) * Vs].contiguous().to(torch.bfloat16)
                                              for d in range(N)])
        else:
            self.lm_head = ttnn.as_tensor(un.T.contiguous().reshape(1, 1, H, Vp).to(torch.bfloat16), dtype=HEAD_DT,
                                          layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                                          mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3) if self.head_shard else ops.rep)
        self.Vp = Vp  # all dies' columns
        self.Vs = Vp // N if self.head_shard else Vp  # one die's logits columns
        self.softcap = cfg.final_logit_softcapping
        print("embedding + lm head up", flush=True)

    # -- pieces ------------------------------------------------------------------------------------------------------
    def embed_tokens(self, ids: torch.Tensor, dev=None):
        """ids [T] -> [1, 1, T, H] bf16 (normed embedding). dev: the ids already on device ([1, T] uint32: a traced
        chunk's StaticCtx.ids_t)."""
        t = dev if dev is not None else \
            ttnn.from_torch(ids.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=self.mesh, mesh_mapper=self.ops.rep)
        e = ttnn.embedding(t, self.embed, layout=ttnn.TILE_LAYOUT)  # [1, T, H]
        e = ttnn.reshape(e, (1, 1, int(ids.numel()), self.H))
        return _bf(self.ops.rms(e, self.embed_norm, out_dtype=ttnn.bfloat16))

    def merge_rows(self, x, rows, emb):
        """x [1, 1, T, H] bf16 with rows `rows` replaced by emb [k, H]: x * keep + M (keep 0 / 1 per row, M zero
        elsewhere; exact). Its programs depend on T only."""
        T = int(x.shape[-2])
        M = torch.zeros(1, 1, T, self.H, dtype=torch.bfloat16)
        M[0, 0, rows] = emb.to(torch.bfloat16)
        keep = torch.ones(1, 1, T, 1, dtype=torch.bfloat16)
        keep[0, 0, rows] = 0
        up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,  # noqa: E731
                                       mesh_mapper=self.ops.rep, memory_config=DRAM)
        M_tt, k_tt = up(M), up(keep)
        xk = ttnn.multiply(x, k_tt, memory_config=DRAM)
        y = ttnn.add(xk, M_tt, memory_config=x.memory_config())
        for t in (x, xk, M_tt, k_tt):
            ttnn.deallocate(t)
        return y

    def _pnorm(self, x, Q):
        """the current source of a prefill mix: rms(x) fp32, with its logits against Q's queries when RowNorm can
        compute them (I2M_PF_ROWN_LG) -> (cur, logits | None)"""
        ops = self.ops
        if PF_ROWN_LG and ops.rown_ok(x):
            return ops.rown(x, None, qrep=ops.qrep[id(Q)])
        return ops.rms(x), None

    def _pmix(self, key, blocks_n, ids, D, cur, Q, nq, lg=None):
        """Prefill mix(blocks_n + [cur], Q, nq) with the blocks' logits from D (each block's against every later query,
        computed once: MixPlan) and only the current source's here (lg: already computed)."""
        ops = self.ops
        if lg is None:
            lg = ttnn.matmul(cur, ops.qrep[id(Q)], compute_kernel_config=ops.ckc, dtype=ttnn.float32,
                             memory_config=DRAM)
        plan = self._pf_plan
        lgs = [(D[b], plan.off[(b, key)], int(D[b].shape[-1]) // 32) for b in ids] + [lg]
        return ops.mixk(blocks_n + [cur], lgs, len(blocks_n) + 1, nq, out_mc=DRAM)

    def _block_logits(self, a, blk):
        """MixPlan: block a's logits against every later query, [1, 1, T, nq_all * 32] (each query's in its 32 columns)"""
        ops, plan = self.ops, self._pf_plan
        return ttnn.matmul(blk, plan.qcat[a], compute_kernel_config=ops.ckc, dtype=ttnn.float32, memory_config=DRAM)

    def layer_forward(self, a: int, x, blocks_n, p0, messages=None, valid=None, slot=0, ctx=None, ids=None, D=None):
        """Application a on x [1,1,T,H] bf16 (block partial sum) at positions p0..; blocks_n = rms-normed fp32 blocks.
        messages: 4 fp32 [1,1,T,H] (q, k, v, mlp) on passes > 1. Returns the bf16 output; at a block boundary the new
        block (the normed input) is left in self._blk. D (prefill MixPlan): the blocks' logits by block id (ids)."""
        li, _ = self.sched[a]
        L, st, ops = self.layers[li], self.apps[a], self.ops
        T = int(x.shape[2])
        rows_mode = T * (L.attn.NH // L.attn.NKV) <= 32
        nvalid = T if valid is None else valid
        xn = None
        if L.nq:
            if D is not None:
                xn, xlg = self._pnorm(x, L.attn_Q)
                ins = self._pmix(("attn", a), blocks_n, ids, D, xn, L.attn_Q, L.nq, lg=xlg)
            else:
                xn = ops.rms(x)
                ins = ops.mix(blocks_n + [xn], L.attn_Q, L.nq)
        else:
            ins = [_f32(x)]
        if L.sliding:
            h = ins[0]
            if messages is not None and L.gates["attn"].pf_ok(h):
                # rms(h) once into the three gates' inputs, the three messages' norms beside it: one program
                hms = [L.gates[n].hm_buf(T) for n in ("attn", "k", "v")]
                bf = ttnn.bfloat16
                ops.rown.multi([dict(x=h, out_dtype=bf, dests=[(t, 0) for t in hms])]
                               + [dict(x=_f32(m), out_dtype=bf, dests=[(t, self.H // 32)])
                                  for m, t in zip(messages[:3], hms)])
                qi, ki, vi = (L.gates[n](h, m, hm=t, m_done=True)
                              for n, m, t in zip(("attn", "k", "v"), messages[:3], hms))
            elif messages is not None:
                qi, ki, vi = (L.gates[n](h, m) for n, m in zip(("attn", "k", "v"), messages[:3]))
            else:
                qi = ki = vi = h
            bf16 = ttnn.bfloat16
            if ki is not qi and ops.rown_ok(qi):  # (the three norms in one program)
                qn, kn, vn = ops.rown.multi([dict(x=t, w=L.in_norm, out_dtype=bf16) for t in (qi, ki, vi)])
            else:
                qn = _bf(ops.rms(qi, L.in_norm, out_dtype=bf16))
                kn = qn if ki is qi else _bf(ops.rms(ki, L.in_norm, out_dtype=bf16))
                vn = qn if vi is qi else _bf(ops.rms(vi, L.in_norm, out_dtype=bf16))
        else:
            qi, ki, vi = ins if len(ins) == 3 else ins * 3
            if messages is not None:
                qi, ki, vi = (L.gates[n](h, m) for n, h, m in zip(("attn", "k", "v"), (qi, ki, vi), messages[:3]))
            bf16 = ttnn.bfloat16
            if ops.rown_ok(qi):  # (the three norms in one program)
                qn, kn, vn = ops.rown.multi([dict(x=t, w=w_, out_dtype=bf16)
                                             for t, w_ in ((qi, L.q_norm), (ki, L.k_norm), (vi, L.v_norm))])
            else:
                qn, kn, vn = (_bf(ops.rms(qi, L.q_norm, out_dtype=bf16)), _bf(ops.rms(ki, L.k_norm, out_dtype=bf16)),
                              _bf(ops.rms(vi, L.v_norm, out_dtype=bf16)))
        r_in = None if L.boundary else x  # (the residual add rides the attention-output conv)
        f32 = ttnn.float32
        rs = getattr(self, "_rs", None) if ctx is not None else None
        if rs is not None:  # (the head-sharded attention reads every row)
            qg = rs.gather(qn)
            kg = qg if kn is qn else rs.gather(kn)
            vg = qg if vn is qn else kg if vn is kn else rs.gather(vn)
            for t in {id(u): u for u in (qn, kn, vn)}.values():
                ttnn.deallocate(t)
            qn, kn, vn = qg, kg, vg
        # (I2M_PF_RSX: the attention / MoE outputs reduce-scattered; each conv's side tiles from RowSplit.side)
        sx = [self._rsx_slots(rs, st, i, ctx) for i in (2, 3)] if rs is not None and PF_RSX and ctx.dyn is None \
            else [None, None]
        if ctx is not None:
            att = attention_packed(L.attn, st.cache, [ln[:2] for ln in st.lanes], qn, kn, vn, ctx, tt_ccl=self.tt_ccl,
                                   scatter=rs.scatter if sx[0] is not None else None)
            att = self._seg_conv(st, 2, att, ctx, resid=r_in, out_dtype=f32, rs=rs, slots=sx[0])
        else:
            att = attention(L.attn, st.cache, qn, kn, vn, p0, valid=valid, tt_ccl=self.tt_ccl, slot=slot)
            att = st.conv("attn", att, rows_mode, nvalid, resid=r_in, out_dtype=f32)
        if L.boundary:
            resid = _f32(att)
            self._blk = blk = xn if xn is not None else ops.rms(x)
            blocks_n = blocks_n + [blk]
            if D is not None:
                D[a] = self._block_logits(a, blk)
                ids = list(ids) + [a]
        else:
            resid = att  # (x + conv(attention output), fp32)
        if D is not None:
            rn_, rlg = self._pnorm(resid, L.mlp_Q)
            m = self._pmix(("mlp", a), blocks_n, ids, D, rn_, L.mlp_Q, 1, lg=rlg)[0]
        else:
            m = ops.mix(blocks_n + [ops.rms(resid)], L.mlp_Q, 1)[0]
        if messages is not None:
            m = L.gates["mlp"](m, messages[3])
        mn = _bf(ops.rms(m, L.mlp_norm, out_dtype=ttnn.bfloat16))
        if rs is not None:  # (the expert-sharded MoE reads every row)
            mg = rs.gather(mn)
            ttnn.deallocate(mn)
            mn = mg
        mo = L.moe.forward(mn, scatter=rs.scatter if sx[1] is not None else None)
        if ctx is not None:
            return self._seg_conv(st, 3, mo, ctx, resid=resid, out_dtype=ttnn.bfloat16, rs=rs, slots=sx[1])
        return st.conv("mlp", mo, rows_mode, nvalid, resid=resid, out_dtype=ttnn.bfloat16)

    @staticmethod
    def _rsx_slots(rs, st, i, ctx):
        """PF_RSX: conv i's side tiles (RowSplit.slots) for the chunk's segments, or None (all-reduce as before)"""
        return rs.slots([(row0, valid, st.lanes[lane][i].snap_row) for row0, R, _, valid, _, lane in ctx.segs])

    @staticmethod
    def _seg_conv(st, i, x, ctx, resid=None, out_dtype=ttnn.bfloat16, rs=None, slots=None):
        """Conv i (2: attention output, 3: MoE output) of a packed chunk: each segment on its lane's state (resid: the
        following residual add, ShortConv.prefill_packed). rs (RowSplit): x all rows, resid / the output this die's.
        slots (PF_RSX): x is this die's rows too; the other rows the conv reads come from RowSplit.side."""
        x = _bf(x)
        rsa, A = (rs.conv() if rs is not None else None), None
        if slots is not None:
            A = rs.side(x, slots, ctx.bc)
            rsa = rsa + (A, slots)
        y = st.lanes[0][i].prefill_packed(x, [(row0, R, valid, st.lanes[lane][i])
                                              for row0, R, _, valid, _, lane in ctx.segs], resid=resid,
                                          out_dtype=out_dtype, rs=rsa, dyn=ctx.dyn)
        if A is not None:
            ttnn.deallocate(A)
        return y

    def messages(self, loop_idx, prev_src_n):
        """prev_src_n: rms-normed fp32 [block-12 input, layer-17 output] of the previous pass -> 4 fp32 messages.
        Prefill (I2M_PF_MIXC): both sources' logits against every loop layer's queries once per pass."""
        ops = self.ops
        T = int(prev_src_n[0].shape[-2])
        if not (PF_MIXC and MIX_KERNEL and FUSED_OPS and T > 32 and T % 32 == 0
                and len({repr(s.memory_config()) for s in prev_src_n}) == 1):
            return ops.mix(prev_src_n, self.msg_Q[loop_idx], 4)
        if getattr(self, "_msg_src", None) is not prev_src_n:
            for t in getattr(self, "_msg_lg", None) or []:
                ttnn.deallocate(t)
            self._msg_lg = [ttnn.matmul(s_, self.msg_qcat, compute_kernel_config=ops.ckc, dtype=ttnn.float32,
                                        memory_config=DRAM) for s_ in prev_src_n]
            self._msg_src = prev_src_n
        st = int(self._msg_lg[0].shape[-1]) // 32
        if ops.mixk is None:
            from .mixw import MixKernel
            ops.mixk = MixKernel(self.mesh, self.H)
        return ops.mixk(list(prev_src_n), [(lg, 4 * loop_idx, st) for lg in self._msg_lg], len(prev_src_n), 4,
                        out_mc=DRAM)

    def head(self, x, blocks_n, out=None):
        """x [1,1,T,H] bf16 (layer 23's output) -> logits [1,1,T,Vp] fp32 (softcapped)."""
        ops = self.ops
        return self.head_tail(ops.mix(blocks_n + [ops.rms(x)], self.final_Q, 1)[0], out=out)

    def head_tail(self, h, normed=None, out=None):
        """The final mix's output h -> logits (norm, LM head, softcap). normed: a list that gets the normed h. out: a
        logits tensor to write into (a trace's output allocated before any capture; the streamed head, <= 32 rows)."""
        ops = self.ops
        h = _bf(ops.rms(h, self.final_norm, out_dtype=ttnn.bfloat16))
        if normed is not None:
            normed.append(h)
        if self.shead is not None:  # (one call per tile row of 32)
            T = int(h.shape[-2])
            if T <= 32:
                return self.shead(h, self.lm_head, T, out=out)
            return ttnn.concat([self.shead(ttnn.slice(h, [0, 0, r, 0], [1, 1, min(T, r + 32), int(h.shape[-1])]),
                                           self.lm_head, min(32, T - r)) for r in range(0, T, 32)], dim=2)
        lg = dmm(h, self.lm_head, ops.ckc, dtype=ttnn.float32, memory_config=DRAM, linear=True)
        return ttnn.multiply(ttnn.tanh(ttnn.multiply(lg, 1.0 / self.softcap)), self.softcap)

    def logits_host(self, t, rows):
        """[1, 1, >= rows, Vs] fp32 TILE logits (per die: its vocab slice when the head is sharded) -> host [rows, V].
        Untilized on device first, so only the real rows cross PCIe."""
        dev = ttnn.get_device_tensors(t) if self.mesh.get_num_devices() > 1 else [t]
        if not self.head_shard:
            dev = dev[:1]
        parts = []
        for d in dev:
            u = ttnn.untilize_with_unpadding(d, [0, 0, rows - 1, int(d.shape[-1]) - 1])
            parts.append(ttnn.to_torch(u).float().reshape(-1, int(d.shape[-1]))[:rows])
        return torch.cat(parts, dim=-1)[:, : self.V]

    # -- whole model -------------------------------------------------------------------------------------------------
    def reset(self, lane=None):
        for s in self.apps:
            if s is not None:
                s.reset(lane)

    def forward_packed(self, ids: torch.Tensor, segs, taps=None, mm=None, all_rows=False):
        """A packed prefill chunk: ids [T] holding segments segs = [(row0, R, p0, valid, slot, lane)] (each its own
        sequence; R, p0 multiples of 64) -> logits [1, 1, U, Vp] of each segment's last valid row. taps, mm: see
        forward. A one-segment chunk with a captured trace of its size replays it (PrefillTrace, I2M_PF_TRACE).
        all_rows (evals): the logits of every row [1, 1, T, Vp] instead."""
        pft = getattr(self, "pft", None)
        if pft is not None and mm is None and not all_rows:
            r = pft.run(ids, segs, taps)
            if r is not None:
                return r
        slid = next(self.apps[a].cache for a, (li, _) in enumerate(self.sched) if self.layers[li].sliding)
        glob = next((self.apps[a].cache for a, (li, _) in enumerate(self.sched) if not self.layers[li].sliding), None)
        ctx = PackedCtx(self.mesh, segs, slid, glob)
        try:
            return self.forward(ids, 0, ctx=ctx, taps=taps, mm=mm,
                                last_rows=None if all_rows else [row0 + valid - 1 for row0, _, _, valid, _, _ in segs])
        finally:
            ctx.release()

    def forward(self, ids: torch.Tensor, p0: int, valid=None, capture=None, slot=0, last_only=False, last_row=None,
                ctx=None, last_rows=None, taps=None, mm=None, head_out=None):
        """Rows ids [T] at positions p0 .. p0+T-1 through all 48 applications -> logits [1,1,T,Vp] fp32.
        capture(a, out) is called with each application's output (debug). taps: a list that gets the speculative
        drafter's taps of every row, [1, 4, T, H] bf16 (the inputs of applications 5, 23, 41 + the final normed state).
        mm: (rows [k] int64, embeddings [k, H] bf16): these rows take the given embeddings instead of their tokens' (the
        image placeholders' rows get the vision tower's outputs, as transformers' masked_scatter)."""
        cfg, ops = self.cfg, self.ops
        x = self.embed_tokens(ids, dev=getattr(ctx, "ids_t", None))
        if mm is not None:
            x = self.merge_rows(x, *mm)
        if valid is not None and valid < int(ids.numel()):
            pass  # padded rows ride along; their K/V lands past the prompt and conv states stop at `valid`
        T = int(ids.numel())
        D = None
        if MIX_PLAN and MIX_KERNEL and FUSED_OPS and T > 32 and T % 32 == 0:  # each block's mix logits once (MixPlan)
            if getattr(self, "_pf_plan", None) is None:
                from .batched import MixPlan
                self._pf_plan = MixPlan(self)
            D = {}
        self._rs = None
        if ctx is not None and D is not None and RowSplit.ok(self.mesh, T):
            if T not in self._rs_by_T:
                self._rs_by_T[T] = RowSplit(self.mesh, T)
            self._rs = self._rs_by_T[T]
            xl = self._rs.part(x)
            ttnn.deallocate(x)
            x = xl
        rs = self._rs
        bids, pre_bids = [], None
        blocks_n = []
        pre_x = pre_blocks = None
        prev_src = None
        n_loop = cfg.loop_end - cfg.loop_start
        for a, (li, p) in enumerate(self.sched):
            L = self.layers[li]
            if li == cfg.loop_start:
                if p == 0:
                    pre_x, pre_blocks, pre_bids = x, list(blocks_n), list(bids)
                else:
                    x, blocks_n, bids = pre_x, list(pre_blocks), list(pre_bids)
            msgs = None
            if L.loop and p > 0:
                msgs = self.messages(li - cfg.loop_start, prev_src)
            x = self.layer_forward(a, x, blocks_n, p0, messages=msgs, valid=valid, slot=slot, ctx=ctx, ids=bids, D=D)
            if L.boundary:  # (the block layer_forward normed)
                blocks_n, bids = blocks_n + [self._blk], bids + [a]
            if capture is not None:
                capture(a, x)
            if taps is not None and a + 1 in (5, 23, 41):
                taps.append(rs.gather(x) if rs is not None else ttnn.to_memory_config(x, DRAM))
            if L.loop and li == cfg.loop_end - 1:
                # sources for the next pass's messages: the block appended at layer 12 of this pass + this output
                b12 = blocks_n[len(pre_blocks) + 1]
                prev_src = [b12, ops.rms(x)]
        for t in getattr(self, "_msg_lg", None) or []:  # (the per-pass message logits: chunk-sized, not kept)
            ttnn.deallocate(t)
        self._msg_lg = self._msg_src = None
        self._blk = None  # (the last block: a chunk temporary, not kept past the forward)
        if taps is not None:
            hn = _bf(ops.rms(ops.mix(blocks_n + [ops.rms(x)], self.final_Q, 1)[0], self.final_norm))
            if rs is not None:
                hn = rs.gather(hn)
            taps[:] = [ttnn.concat([_bf(t) for t in taps] + [hn], dim=1)]
        if last_rows is not None:  # the head on the given rows (each packed segment's last prompt token), rows 0..U-1
            def cut(t):  # row gather through row-major pages: every program's shape depends on T only
                if rs is not None:
                    return rs.cut(t, last_rows, ctx.bc, dyn=ctx.dyn)
                rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
                g = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, int(t.shape[-1])]), t.dtype,
                                                   ttnn.ROW_MAJOR_LAYOUT, self.mesh, DRAM)
                if ctx.dyn is not None:  # (StaticCtx: the last row from the dyn words)
                    ctx.bc([(rm, -1, g, 0, 1, 17)], dyn=ctx.dyn)
                else:
                    ctx.bc([(rm, r, g, i, 1) for i, r in enumerate(last_rows)])
                ttnn.deallocate(rm)
                return ttnn.to_layout(g, ttnn.TILE_LAYOUT)
            return self.head(cut(x), [cut(b) for b in blocks_n], out=head_out)
        if rs is not None:  # (the head on all rows / one row: the full rows back)
            x, blocks_n = rs.gather(x), [rs.gather(b) for b in blocks_n]
        if last_only:  # the head on one row (a prefill's last prompt token): [1, 1, 1, Vp]
            r = int(ids.numel()) - 1 if last_row is None else int(last_row)
            cut = lambda t: ttnn.slice(t, [0, 0, r, 0], [1, 1, r + 1, t.shape[-1]])  # noqa: E731
            return self.head(cut(x), [cut(b) for b in blocks_n])
        return self.head(x, blocks_n)


class PrefillTrace:
    """One-segment prefill chunks (one request: rows 0 .. T-1 of lane 0, any p0 / slot / valid) of the sizes Ts replayed
    from a trace per size. The chunk's per-call inputs are data in StaticCtx's device tensors (written before each
    replay), so a replay runs the eager chunk's programs on the same inputs: bitwise the eager chunk. Outputs go into
    buffers allocated before any capture (logits: one 32-row tile; taps: [1, 4, CHUNK, H], rows 0 .. T-1).

    Order (trace safety): __init__ (allocations) and warm() (every program, eager) before the first capture of the
    mesh; capture() after every other trace's (its intermediates then sit on no live tensor; its outputs predate them)."""

    def __init__(self, m: "Inkling2Model", Ts, taps=False):
        self.m, self.mesh, self.taps = m, m.mesh, taps
        mesh = m.mesh
        slid = next(m.apps[a].cache for a, (li, _) in enumerate(m.sched) if m.layers[li].sliding)
        glob = next((m.apps[a].cache for a, (li, _) in enumerate(m.sched) if not m.layers[li].sliding), None)
        N = mesh.get_num_devices()
        self.Ts = sorted(T for T in Ts if T <= slid.ring_chunk and T % 64 == 0)
        self.ctx = {}
        for T in self.Ts:
            if RowSplit.ok(mesh, T) and T not in m._rs_by_T:
                m._rs_by_T[T] = RowSplit(mesh, T)
            self.ctx[T] = StaticCtx(mesh, T, slid, glob, Tl=T // N if RowSplit.ok(mesh, T) else None)
        Tm = max(self.Ts) if self.Ts else 0
        # (the head's output: one tile row of logits, each die its vocab slice)
        self.lg = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, m.Vs]), ttnn.float32, ttnn.TILE_LAYOUT, mesh,
                                                 DRAM) if m.shead is not None else None
        self.taps_buf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, Tm, m.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                                       mesh, DRAM) if taps and Tm else None
        self.Tm = Tm
        self.tid, self.out = {}, {}
        self.lane0 = [cv for a in m.apps if a is not None for cv in a.lanes[0]]

    def _chunk(self, T):
        """the chunk's device work (eager or captured) -> logits"""
        ctx, m = self.ctx[T], self.m
        taps = [] if self.taps else None
        lg = m.forward(torch.zeros(T, dtype=torch.long), 0, ctx=ctx, last_rows=[T - 1], taps=taps, head_out=self.lg)
        if taps:
            t = taps[0]
            # (into the shared buffer: per tap, T / 32 x H / 32 tiles at the same tile offset in a T-row tensor)
            n = T // 32 * (m.H // 32)
            ctx.bc([(t, g * n, self.taps_buf, g * (self.Tm // 32) * (m.H // 32), n) for g in range(4)])
            ttnn.deallocate(t)
        return lg

    def warm(self):
        """every program of every size, eager (before any trace capture)"""
        for T in self.Ts:
            self.m.reset(0)
            self.ctx[T].set(torch.zeros(T, dtype=torch.long), (0, T, 0, T, 0, 0))
            lg = self._chunk(T)
            self.m.logits_host(lg, 1)
        self.m.reset(0)
        ttnn.synchronize_device(self.mesh)

    def capture(self):
        for T in self.Ts:
            ttnn.synchronize_device(self.mesh)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self.out[T] = self._chunk(T)
            finally:
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.tid[T] = tid
        ttnn.synchronize_device(self.mesh)

    def run(self, ids, segs, taps=None):
        """forward_packed by a replay, or None (not a traced shape)"""
        T = int(ids.numel())
        if len(segs) != 1 or T not in self.tid or (taps is not None and not self.taps):
            return None
        snap = self.lane0[0].snap_row
        if not StaticCtx.ok(segs[0], T, snap):
            return None
        self.ctx[T].set(ids, segs[0], snap)
        ttnn.execute_trace(self.mesh, self.tid[T], cq_id=0, blocking=False)
        if snap is not None:  # (the kernels wrote the snapshot rows into each lane-0 conv's snap_buf)
            for cv in self.lane0:
                cv.snap_out = cv.snap_buf
        if taps is not None:
            taps[:] = [self.taps_buf]
        return self.out[T]

    def release(self):
        for tid in self.tid.values():
            ttnn.release_trace(self.mesh, tid)
        self.tid = {}
