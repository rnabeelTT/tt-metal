# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug: tt/aprep.py vs the ttnn-op prep on the real activations of one decode step (every application): per output,
max / mean |err| relative to the reference's rms, and the mean signed relative error (a bias), plus fp32 torch
recomputation from the same inputs."""

import os

import torch

import ttnn
import models.autoports.thinkingmachines_inkling_2_mini.tt.aprep as ap
import models.autoports.thinkingmachines_inkling_2_mini.tt.batched as bm
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model

T = lambda t: ttnn.to_torch(t).float()  # noqa: E731


def l2clip(x, thr):
    return x * torch.clamp(thr / x.norm(dim=-1, keepdim=True), max=1.0)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=100 << 20)
    m = Inkling2Model(mesh, max_seq_len=512, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=256,
                      slots=int(os.environ.get("B", "4")))
    ref = torch.load("/cache/fixtures/ref_bf16.pt")
    ids = ref[list(ref)[1]]["ids"][:257]
    m.forward(ids[:256], 0, valid=256, slot=0, last_only=True, last_row=255)
    B = int(os.environ.get("B", "4"))
    dec = bm.BatchedDecoder(m, B)
    for b in range(B):
        if b:
            m.reset()
            m.forward(ids[:256], 0, valid=256, slot=b, last_only=True, last_row=255)
        dec.load_slot(b)
    stats = []
    # attention output, kernel path vs op path, same inputs (the cache write happens in both: same values)
    orig_attn = bm.attention_batched
    att_stats = []

    def attn_both(w, c, ba, xq, xk, xv, pos, pos_wr, pos_sdpa, tau, tt_ccl=None):
        bm.APREP = False
        st = [ttnn.clone(s) for s in (ba.k.s + ba.v.s)]
        old = T(orig_attn(w, c, ba, xq, xk, xv, pos, pos_wr, pos_sdpa, tau, tt_ccl))
        for dst, src in zip(ba.k.s + ba.v.s, st):  # undo the conv state advance
            ttnn.copy(src, dst)
        bm.APREP = os.environ.get("CMP_BASE") != "1"
        new_t = orig_attn(w, c, ba, xq, xk, xv, pos, pos_wr, pos_sdpa, tau, tt_ccl)
        new = T(new_t)
        Bn = int(xq.shape[2])
        d = (new - old)[0, 0, :Bn]
        sc = old[0, 0, :Bn].pow(2).mean().sqrt()
        att_stats.append((w.sliding, (d.abs().max() / sc).item(), (d.abs().mean() / sc).item()))
        return new_t

    bm.attention_batched = attn_both
    orig = ap.AttnPrep.__call__

    def wrapped(self, pq, k, v, tau=None):
        outs = orig(self, pq, k, v, tau)
        w, B = self.w, self.B
        P, K, V = T(pq)[0, 0, :B], T(k)[0, 0, :B], T(v)[0, 0, :B]
        NH, NKV, HD, DR = w.NH, w.NKV, w.HD, w.DR
        tf = T(tau).reshape(-1)[:B, None, None] if not w.sliding else 1.0
        rq = l2clip(P[:, : w.qd].reshape(B, NH, HD), w.qk_clip) * tf
        rr = l2clip(P[:, w.qd : w.qd + w.rd].reshape(B, NH, DR), w.rel_clip) * tf
        rk = l2clip(K.reshape(B, NKV, HD), w.qk_clip)
        rv = V.reshape(B, NKV, HD) * torch.rsqrt(V.reshape(B, NKV, HD).pow(2).mean(-1, keepdim=True) + w.eps)
        s0 = w.qd + w.rd
        rs = (P[:, s0 : s0 + NH] / w.scale)[:, :, None].expand(B, NH, 32)
        got = [T(t) for t in outs]
        got = [got[0].reshape(B, NH, HD), got[1][..., :DR].reshape(B, NH, DR), got[2][:, :, :NKV].reshape(B, NKV, HD),
               got[3][:, :, :NKV].reshape(B, NKV, HD), got[4].reshape(B, NH, 32)]
        row = []
        for g, r in zip(got, (rq, rr, rk, rv, rs)):
            rb = r.bfloat16().float()
            sc = r.pow(2).mean().sqrt()
            row.append(((g - r).abs().max() / sc).item())
            row.append(((g - rb).abs().mean() / sc).item())
            msk = r.abs() > 1e-3 * sc
            row.append(((g - r)[msk] / r[msk]).mean().item())
        stats.append((w.sliding, row))
        return outs

    from models.autoports.thinkingmachines_inkling_2_mini.tt.attention import _l2clip, _rms_unweighted

    def old_prep(w, pq, k, v, tau):
        B, NH, NKV, HD, DR = int(pq.shape[2]), w.NH, w.NKV, w.HD, w.DR
        _f32, _bf = bm._f32, bm._bf
        q = ttnn.slice(pq, (0, 0, 0, 0), (1, 1, B, w.qd))
        r = ttnn.slice(pq, (0, 0, 0, w.qd), (1, 1, B, w.qd + w.rd))
        s = ttnn.slice(pq, (0, 0, 0, w.qd + w.rd), (1, 1, B, w.qd + w.rd + NH))
        q = _l2clip(_f32(ttnn.reshape(q, (1, B, NH, HD))), w.qk_clip, w.q_scale)
        r = _l2clip(_f32(ttnn.reshape(r, (1, B, NH, DR))), w.rel_clip, w.r_scale)
        k = _l2clip(_f32(ttnn.reshape(k, (1, B, NKV, HD))), w.qk_clip)
        v = _rms_unweighted(_f32(ttnn.reshape(v, (1, B, NKV, HD))), w.eps)
        if not w.sliding:
            q, r = ttnn.multiply(q, tau), ttnn.multiply(r, tau)
        q, k, v = _bf(q), _bf(k), _bf(v)
        rq = ttnn.pad(_bf(r), [1, B, NH, 32], [0, 0, 0, 0], 0.0)
        sk = _bf(ttnn.multiply(_f32(ttnn.reshape(s, (1, B, NH, 1))), w.sink_bcast))
        return q, rq, k, v, sk

    def wrapped2(self, pq, k, v, tau=None):
        outs = wrapped(self, pq, k, v, tau)
        olds = old_prep(self.w, pq, k, v, tau)
        w, B = self.w, self.B
        cmp = []
        for n, a, o in zip("q r k v s".split(), outs, olds):
            a, o = T(a), T(o)
            if n in "kv":
                a = a[:, :, : w.NKV]
            a = a.reshape(o.shape)
            ne = (a != o).float().mean().item()
            sc = o.pow(2).mean().sqrt()
            cmp.append(f"{n}: neq {ne:.3f} max {((a - o).abs().max() / sc).item():.1e}")
        print(f"prep {len(stats) - 1:2d} {'sl' if w.sliding else 'gl'} " + "  ".join(cmp), flush=True)
        return outs

    ap.AttnPrep.__call__ = wrapped2
    dec.set_inputs(torch.tensor([int(ids[256])] * B), torch.tensor([256] * B))
    dec.step()
    names = "q r k v s".split()
    for i, (sl, row) in enumerate(stats):
        print(f"app {i:2d} {'sl' if sl else 'gl'} " + "  ".join(
            f"{n}: max {row[3 * j]:.1e} mean-vs-bf16 {row[3 * j + 1]:.1e} bias {row[3 * j + 2]:+.1e}"
            for j, n in enumerate(names)), flush=True)
    for i, (sl, mx, mn) in enumerate(att_stats):
        print(f"attn app {i:2d} {'sl' if sl else 'gl'}: kernel vs ops max {mx:.2e} mean {mn:.2e}", flush=True)
    ttnn.close_mesh_device(mesh)
    print("== dbg done")


if __name__ == "__main__":
    main()
