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

import math
import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.attention import AttnCache, AttnWeights, attention
from models.autoports.thinkingmachines_inkling_2_mini.tt.moe import MoE
from models.autoports.thinkingmachines_inkling_2_mini.tt.sconv import ShortConv
from models.autoports.thinkingmachines_inkling_2_mini.tt.weights import Checkpoint, load_config, schedule

DRAM = ttnn.DRAM_MEMORY_CONFIG
SQRT5 = math.sqrt(5.0)


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

    def up(self, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=self.mesh, memory_config=DRAM,
                               mesh_mapper=self.rep)

    def rms(self, x, w=None):
        """fp32 RMS norm over the last dim (transformers Inkling2RMSNorm / UnweightedRMSNorm), w [1,1,1,H] fp32."""
        x = _f32(x)
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        y = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ms, self.eps)))
        return ttnn.multiply(y, w) if w is not None else y

    def mix(self, srcs_normed, Q, nq):
        """Block attention residual: srcs_normed list of n fp32 [1,1,T,H] (rms-normed sources), Q [1,1,H,32] fp32
        (nq queries in columns, pre-scaled by H^-0.5) -> nq fp32 outputs sum_i softmax_i(src_i . q) src_i."""
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
        self.ops = ops
        self.wd_h = ops.up(d[:, :H].T.reshape(1, 1, H, -1), ttnn.bfloat16)
        self.wd_m = ops.up(d[:, H:].T.reshape(1, 1, H, -1), ttnn.bfloat16)
        self.wf = ops.up(L("forget_weight").float().T.reshape(1, 1, d.shape[0], H), ttnn.float32)
        self.wg = ops.up(L("gain_weight").float().T.reshape(1, 1, d.shape[0], H), ttnn.float32)

    def __call__(self, h, m):
        o = self.ops
        hf, mf = _f32(h), _f32(m)
        g = ttnn.add(ttnn.matmul(_bf(o.rms(hf)), self.wd_h, compute_kernel_config=o.ckc, dtype=ttnn.float32),
                     ttnn.matmul(_bf(o.rms(mf)), self.wd_m, compute_kernel_config=o.ckc, dtype=ttnn.float32))
        g = _f32(_bf(g))  # the bf16 Linear's output rounding
        f = ttnn.tanh(ttnn.matmul(g, self.wf, compute_kernel_config=o.ckc, dtype=ttnn.float32))
        a = ttnn.tanh(ttnn.matmul(g, self.wg, compute_kernel_config=o.ckc, dtype=ttnn.float32))
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
            Q = torch.zeros(H, 32)
            Q[:, : self.nq] = q.T * sc
            self.attn_Q = ops.up(Q.reshape(1, 1, H, 32))
        Q = torch.zeros(H, 32)
        Q[:, :1] = ck.mlp_query(li).T * sc
        self.mlp_Q = ops.up(Q.reshape(1, 1, H, 32))
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

    def __init__(self, layer: Layer, max_seq_len, kv_dtype, prefill_chunk):
        mesh = layer.ops.mesh
        H = layer.attn_conv_w.shape[0]
        self.cache = AttnCache(layer.attn, max_seq_len, kv_dtype=kv_dtype, prefill_chunk=prefill_chunk)
        rep = ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None
        self.attn_conv = ShortConv(mesh, layer.attn_conv_w, rep, H)
        self.mlp_conv = ShortConv(mesh, layer.mlp_conv_w, rep, H)

    def reset(self):
        self.cache.reset()
        self.attn_conv.reset()
        self.mlp_conv.reset()

    def conv(self, which, x, rows_mode, nvalid):
        c = self.attn_conv if which == "attn" else self.mlp_conv
        if rows_mode:
            return c.rows(x, commit_all=True)
        return c.prefill_commit(x, nvalid)


class Inkling2Model:
    def __init__(self, mesh, max_seq_len=8192, expert_dtype=ttnn.bfloat4_b, attn_dtype=ttnn.bfloat16,
                 kv_dtype=ttnn.bfloat8_b, prefill_chunk=1024, cache_dir=None, tt_ccl=None, layers=None):
        self.mesh = mesh
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
        lids = range(cfg.num_hidden_layers) if layers is None else layers
        self.layers = {}
        for li in lids:
            t0 = __import__("time").time()
            self.layers[li] = Layer(ops, cfg, ck, li, expert_dtype, attn_dtype, cache_dir, tt_ccl)
            print(f"layer {li} loaded in {__import__('time').time() - t0:.0f}s", flush=True)
        self.apps = [AppState(self.layers[li], max_seq_len, kv_dtype, prefill_chunk) if li in self.layers else None
                     for li, p in self.sched]
        print(f"application states up ({sum(a is not None for a in self.apps)})", flush=True)
        # message queries [n_loop, 4, H] -> per loop layer [1,1,H,32] (4 query columns, pre-scaled)
        mq = ck.message_queries()
        self.msg_Q = []
        for i in range(mq.shape[0]):
            Q = torch.zeros(H, 32)
            Q[:, :4] = mq[i].T * H**-0.5
            self.msg_Q.append(ops.up(Q.reshape(1, 1, H, 32)))
        Q = torch.zeros(H, 32)
        Q[:, :1] = ck.final_query().T * H**-0.5
        self.final_Q = ops.up(Q.reshape(1, 1, H, 32))
        self.final_norm = ops.up(ck.get("model.llm.norm.weight").float().reshape(1, 1, 1, -1))
        self.embed_norm = ops.up(ck.get("model.llm.embed_norm.weight").float().reshape(1, 1, 1, -1))
        V = cfg.unpadded_vocab_size
        self.V = V
        emb = ck.get("model.llm.embed.weight")
        self.embed = ttnn.as_tensor(emb.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                    device=mesh, memory_config=DRAM, mesh_mapper=ops.rep)  # no ttnn cache: large cached uploads hang
        Vp = (V + 31) // 32 * 32
        un = ck.get("model.llm.unembed.weight")[:V].float()
        un = torch.nn.functional.pad(un, (0, 0, 0, Vp - V))
        self.lm_head = ttnn.as_tensor(un.T.contiguous().reshape(1, 1, H, Vp).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=ops.rep)
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

    def layer_forward(self, a: int, x, blocks_n, p0, messages=None, valid=None):
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
        att = attention(L.attn, st.cache, qn, kn, vn, p0, valid=valid, tt_ccl=self.tt_ccl)
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
        mo = st.conv("mlp", mo, rows_mode, nvalid)
        return _bf(ttnn.add(resid, _f32(mo)))

    def messages(self, loop_idx, prev_src_n):
        """prev_src_n: rms-normed fp32 [block-12 input, layer-17 output] of the previous pass -> 4 fp32 messages."""
        return self.ops.mix(prev_src_n, self.msg_Q[loop_idx], 4)

    def head(self, x, blocks_n):
        """x [1,1,T,H] bf16 (layer 23's output) -> logits [1,1,T,Vp] fp32 (softcapped)."""
        ops = self.ops
        h = ops.mix(blocks_n + [ops.rms(x)], self.final_Q, 1)[0]
        h = _bf(ops.rms(h, self.final_norm))
        lg = ttnn.linear(h, self.lm_head, compute_kernel_config=ops.ckc, dtype=ttnn.float32, memory_config=DRAM)
        return ttnn.multiply(ttnn.tanh(ttnn.multiply(lg, 1.0 / self.softcap)), self.softcap)

    # -- whole model -------------------------------------------------------------------------------------------------
    def reset(self):
        for s in self.apps:
            if s is not None:
                s.reset()

    def forward(self, ids: torch.Tensor, p0: int, valid=None, capture=None):
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
            x = self.layer_forward(a, x, blocks_n, p0, messages=msgs, valid=valid)
            if L.boundary:
                blocks_n = blocks_n + [ops.rms(xin)]
            if capture is not None:
                capture(a, x)
            if L.loop and li == cfg.loop_end - 1:
                # sources for the next pass's messages: the block appended at layer 12 of this pass + this output
                b12 = blocks_n[len(pre_blocks) + 1]
                prev_src = [b12, ops.rms(x)]
        return self.head(x, blocks_n)
