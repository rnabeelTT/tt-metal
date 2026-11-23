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
                                                                          attention, attention_packed)
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
HEAD_DT = _DT[os.environ.get("I2M_HEAD_DT", "bfp8")]  # lm_head
# packed prefill: up to LANES sequences per prefill chunk, each with its own conv-state lane
# (default 8; 4 on one die: each lane's conv states take ~32 MB of DRAM over the 48 applications)
LANES = int(os.environ.get("I2M_PREFILL_LANES", "0"))


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
        self.mixk = None
        self.gatek = None
        self.gatefk = None
        self.gate2k = None  # (set by the decoder when the post-attention rectangle exists: tt/gate.py GateKernel2)
        self.rms_cfg = {}

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
        y = self._rms(x, w)
        return ttnn.typecast(y, out_dtype) if out_dtype is not None and y.dtype != out_dtype else y

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

    def __call__(self, h, m):
        o, H = self.ops, self.H
        hf, mf = _f32(h), _f32(m)
        if GATE2 and o.gate2k is not None and int(hf.shape[-2]) <= 16 and len(hf.shape) == 4 and int(hf.shape[1]) == 1:
            return o.gate2k(hf, mf, self.wd, self.wfg_bf)
        if GATE_FUSED and int(hf.shape[-2]) <= 32 and len(hf.shape) == 4 and int(hf.shape[1]) == 1:
            if o.gatefk is None:
                from .gate import GateKernel
                o.gatefk = GateKernel(o.mesh, H, int(self.wd.shape[-1]), o.eps)
            return o.gatefk(hf, mf, self.wd, self.wfg_bf)
        hm = ttnn.concat([_bf(o.rms(hf)), _bf(o.rms(mf))], dim=-1)  # [.., 2H] bf16 (transformers: bf16 Linear input)
        g = dmm(hm, self.wd, o.ckc, dtype=ttnn.bfloat16)  # bf16 Linear output
        T = int(hf.shape[-2])
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

    def conv(self, which, x, rows_mode, nvalid):
        c = self.attn_conv if which == "attn" else self.mlp_conv
        if rows_mode:
            return c.rows(x, commit_all=True)
        return c.prefill_commit(x, nvalid)


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
        self.lm_head = ttnn.as_tensor(un.T.contiguous().reshape(1, 1, H, Vp).to(torch.bfloat16), dtype=HEAD_DT,
                                      layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                                      mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3) if self.head_shard else ops.rep)
        self.Vp = Vp  # all dies' columns
        self.Vs = Vp // N if self.head_shard else Vp  # one die's logits columns
        self.softcap = cfg.final_logit_softcapping
        print("embedding + lm head up", flush=True)

    # -- pieces ------------------------------------------------------------------------------------------------------
    def embed_tokens(self, ids: torch.Tensor):
        """ids [T] -> [1, 1, T, H] bf16 (normed embedding)."""
        t = ttnn.from_torch(ids.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=self.mesh, mesh_mapper=self.ops.rep)
        e = ttnn.embedding(t, self.embed, layout=ttnn.TILE_LAYOUT)  # [1, T, H]
        e = ttnn.reshape(e, (1, 1, int(ids.numel()), self.H))
        return _bf(self.ops.rms(e, self.embed_norm))

    def layer_forward(self, a: int, x, blocks_n, p0, messages=None, valid=None, slot=0, ctx=None):
        """Application a on x [1,1,T,H] bf16 (block partial sum) at positions p0..; blocks_n = rms-normed fp32 blocks.
        messages: 4 fp32 [1,1,T,H] (q, k, v, mlp) on passes > 1. Returns the bf16 output."""
        li, _ = self.sched[a]
        L, st, ops = self.layers[li], self.apps[a], self.ops
        T = int(x.shape[2])
        rows_mode = T * (L.attn.NH // L.attn.NKV) <= 32
        nvalid = T if valid is None else valid
        xn = None
        if L.nq:
            xn = ops.rms(x)
            ins = ops.mix(blocks_n + [xn], L.attn_Q, L.nq)
        else:
            ins = [_f32(x)]
        if L.sliding:
            h = ins[0]
            if messages is not None:
                qi, ki, vi = (L.gates[n](h, m) for n, m in zip(("attn", "k", "v"), messages[:3]))
            else:
                qi = ki = vi = h
            qn = _bf(ops.rms(qi, L.in_norm))
            kn = qn if ki is qi else _bf(ops.rms(ki, L.in_norm))
            vn = qn if vi is qi else _bf(ops.rms(vi, L.in_norm))
        else:
            qi, ki, vi = ins if len(ins) == 3 else ins * 3
            if messages is not None:
                qi, ki, vi = (L.gates[n](h, m) for n, h, m in zip(("attn", "k", "v"), (qi, ki, vi), messages[:3]))
            qn, kn, vn = _bf(ops.rms(qi, L.q_norm)), _bf(ops.rms(ki, L.k_norm)), _bf(ops.rms(vi, L.v_norm))
        if ctx is not None:
            att = attention_packed(L.attn, st.cache, [ln[:2] for ln in st.lanes], qn, kn, vn, ctx, tt_ccl=self.tt_ccl)
            att = self._seg_conv(st, 2, att, ctx)
        else:
            att = attention(L.attn, st.cache, qn, kn, vn, p0, valid=valid, tt_ccl=self.tt_ccl, slot=slot)
            att = st.conv("attn", att, rows_mode, nvalid)
        if L.boundary:
            resid = _f32(att)
            blocks_n = blocks_n + [xn if xn is not None else ops.rms(x)]
        else:
            resid = ttnn.add(_f32(x), _f32(att))
        m = ops.mix(blocks_n + [ops.rms(resid)], L.mlp_Q, 1)[0]
        if messages is not None:
            m = L.gates["mlp"](m, messages[3])
        mo = L.moe.forward(_bf(ops.rms(m, L.mlp_norm)))
        mo = self._seg_conv(st, 3, mo, ctx) if ctx is not None else st.conv("mlp", mo, rows_mode, nvalid)
        return _bf(ttnn.add(resid, _f32(mo)))

    @staticmethod
    def _seg_conv(st, i, x, ctx):
        """Conv i (2: attention output, 3: MoE output) of a packed chunk: each segment on its lane's state."""
        x = _bf(x)
        return st.lanes[0][i].prefill_packed(x, [(row0, R, valid, st.lanes[lane][i])
                                                 for row0, R, _, valid, _, lane in ctx.segs])

    def messages(self, loop_idx, prev_src_n):
        """prev_src_n: rms-normed fp32 [block-12 input, layer-17 output] of the previous pass -> 4 fp32 messages."""
        return self.ops.mix(prev_src_n, self.msg_Q[loop_idx], 4)

    def head(self, x, blocks_n):
        """x [1,1,T,H] bf16 (layer 23's output) -> logits [1,1,T,Vp] fp32 (softcapped)."""
        ops = self.ops
        return self.head_tail(ops.mix(blocks_n + [ops.rms(x)], self.final_Q, 1)[0])

    def head_tail(self, h):
        """The final mix's output h -> logits (norm, LM head, softcap)."""
        ops = self.ops
        h = _bf(ops.rms(h, self.final_norm))
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

    def forward_packed(self, ids: torch.Tensor, segs):
        """A packed prefill chunk: ids [T] holding segments segs = [(row0, R, p0, valid, slot, lane)] (each its own
        sequence; R, p0 multiples of 64) -> logits [1, 1, U, Vp] of each segment's last valid row."""
        slid = next(self.apps[a].cache for a, (li, _) in enumerate(self.sched) if self.layers[li].sliding)
        glob = next((self.apps[a].cache for a, (li, _) in enumerate(self.sched) if not self.layers[li].sliding), None)
        ctx = PackedCtx(self.mesh, segs, slid, glob)
        try:
            return self.forward(ids, 0, ctx=ctx, last_rows=[row0 + valid - 1 for row0, _, _, valid, _, _ in segs])
        finally:
            ctx.release()

    def forward(self, ids: torch.Tensor, p0: int, valid=None, capture=None, slot=0, last_only=False, last_row=None,
                ctx=None, last_rows=None):
        """Rows ids [T] at positions p0 .. p0+T-1 through all 48 applications -> logits [1,1,T,Vp] fp32.
        capture(a, out) is called with each application's output (debug)."""
        cfg, ops = self.cfg, self.ops
        x = self.embed_tokens(ids)
        if valid is not None and valid < int(ids.numel()):
            pass  # padded rows ride along; their K/V lands past the prompt and conv states stop at `valid`
        blocks_n = []
        pre_x = pre_blocks = None
        prev_src = None
        n_loop = cfg.loop_end - cfg.loop_start
        for a, (li, p) in enumerate(self.sched):
            L = self.layers[li]
            if li == cfg.loop_start:
                if p == 0:
                    pre_x, pre_blocks = x, list(blocks_n)
                else:
                    x, blocks_n = pre_x, list(pre_blocks)
            msgs = None
            if L.loop and p > 0:
                msgs = self.messages(li - cfg.loop_start, prev_src)
            xin = x
            x = self.layer_forward(a, x, blocks_n, p0, messages=msgs, valid=valid, slot=slot, ctx=ctx)
            if L.boundary:
                blocks_n = blocks_n + [ops.rms(xin)]
            if capture is not None:
                capture(a, x)
            if L.loop and li == cfg.loop_end - 1:
                # sources for the next pass's messages: the block appended at layer 12 of this pass + this output
                b12 = blocks_n[len(pre_blocks) + 1]
                prev_src = [b12, ops.rms(x)]
        if last_rows is not None:  # the head on the given rows (each packed segment's last prompt token), rows 0..U-1
            def cut(t):  # row gather through row-major pages: every program's shape depends on T only
                rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
                g = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, int(t.shape[-1])]), t.dtype,
                                                   ttnn.ROW_MAJOR_LAYOUT, self.mesh, DRAM)
                ctx.bc([(rm, r, g, i, 1) for i, r in enumerate(last_rows)])
                ttnn.deallocate(rm)
                return ttnn.to_layout(g, ttnn.TILE_LAYOUT)
            return self.head(cut(x), [cut(b) for b in blocks_n])
        if last_only:  # the head on one row (a prefill's last prompt token): [1, 1, 1, Vp]
            r = int(ids.numel()) - 1 if last_row is None else int(last_row)
            cut = lambda t: ttnn.slice(t, [0, 0, r, 0], [1, 1, r + 1, t.shape[-1]])  # noqa: E731
            return self.head(cut(x), [cut(b) for b in blocks_n])
        return self.head(x, blocks_n)
