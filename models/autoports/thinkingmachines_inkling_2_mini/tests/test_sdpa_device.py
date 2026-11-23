# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test of tt/attn/inkling_sdpa.inkling_sdpa_decode with Inkling-2's additions (per-row sinks, per-item page
tables), against an fp32 torch reference of Inkling-2 attention on the same (dtype-rounded) cache:

  scores = q.k * scale + bias(r_q, proj)[q - k]  over k in [lo, pos] (lo = pos + 1 - W on sliding layers),
  softmax over [scores, sink] (the sink a zero-value extra key), @ V

Shapes: NH / NKV per die from TP (env TP, default 1: 24 q / 8 kv heads; TP 4: 6 / 2), head dim 128, scale 1/sqrt(128),
E = 1024 on both layer kinds, W = 1024 on sliding layers. Modes: decode / verify rows of one sequence (R = 1..8),
prefill (items of 8 rows of one sequence; env MI: items per work unit), multi-user (B items of 1 row, each its own page-table row). Controls: the
reference without the bias and without the sink must both be far from the op.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_2_mini/tests/test_sdpa_device.py
"""

import os
import sys

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import relbias
from models.autoports.thinkingmachines_inkling_2_mini.tt.attn.inkling_sdpa import inkling_sdpa_decode

TP = int(os.environ.get("TP", "1"))
NH, NKV, HD, DREL, BLK = 24 // TP, 8 // TP, 128, 16, 64
G = NH // NKV
SCALE = HD**-0.5
KV_DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("KV_DT", "bfp8")]


def lse(qf, Kpos, r, proj, pos, W):
    """Per (row, head) logsumexp of the biased scores: sinks drawn around it carry ~half the mass (a real test)."""
    R = qf.shape[0]
    E = proj.shape[1]
    out = torch.zeros(R, NH)
    for j in range(R):
        p = max(pos[j], 0)
        lo = 0 if W == 0 else max(0, p + 1 - W)
        ks = torch.arange(lo, p + 1)
        d = p - ks
        for h in range(NH):
            s = (Kpos[j][ks, h // G] @ qf[j, h]) * SCALE
            s = s + torch.where(d < E, (r[j, h] @ proj)[d.clamp(max=E - 1)], torch.zeros(()))
            out[j, h] = torch.logsumexp(s, 0)
    return out


def reference(qf, Kpos, Vpos, r, proj, sink, pos, W, use_bias=True, use_sink=True):
    """qf [R, NH, HD], Kpos/Vpos [R][P, NKV, HD] per row (its sequence's values), r [R, NH, DREL], sink [R, NH]."""
    R = qf.shape[0]
    E = proj.shape[1]
    out = torch.zeros(R, NH, HD)
    for j in range(R):
        p = pos[j]
        if p < 0:
            continue
        lo = 0 if W == 0 else max(0, p + 1 - W)
        ks = torch.arange(lo, p + 1)
        for h in range(NH):
            g = h // G
            s = (Kpos[j][ks, g] @ qf[j, h]) * SCALE
            if use_bias:
                d = p - ks
                rel = r[j, h] @ proj
                s = s + torch.where(d < E, rel[d.clamp(max=E - 1)], torch.zeros(()))
            if use_sink:
                a = torch.softmax(torch.cat([s, sink[j, h].reshape(1)]), 0)[:-1]
            else:
                a = torch.softmax(s, 0)
            out[j, h] = a @ Vpos[j][ks, g]
    return out


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _cache(dev, P, W, n_seq, extra_rows=0):
    """n_seq sequences of P positions in one paged pool: per sequence a shuffled block list (global) or a ring."""
    nlog = (P + BLK - 1) // BLK
    nring = (W + extra_rows) // BLK + 2 if W else 0
    per = nring if W else nlog + 1
    nphys = per * n_seq
    perm = torch.randperm(nphys)
    pts = []
    for s in range(n_seq):
        mine = perm[s * per : (s + 1) * per]
        pts.append(mine[torch.arange(nlog) % nring] if W else mine[:nlog])
    return nphys, nlog, pts


def run_case(dev, W, E, pos, seed, items_r=0, users=0):
    """items_r > 0: prefill mode (pos consecutive rows of one sequence, items of items_r rows).
    users > 0: multi-user decode, pos[i] = user i's position, each its own sequence (pt_per_item)."""
    torch.manual_seed(seed)
    R = len(pos)
    n_seq = users if users else 1
    P = max(pos) + 1
    proj = torch.randn(DREL, E) * 0.5
    r = torch.randn(R, NH, DREL) * 0.5
    q_h = torch.nn.functional.normalize(torch.randn(1, R, NH, HD), dim=-1) * 20.0  # L2-clipped scale (Inkling-2)
    Ks = [torch.nn.functional.normalize(torch.randn(P, NKV, HD), dim=-1) * 20.0 for _ in range(n_seq)]
    Vs = [torch.randn(P, NKV, HD) for _ in range(n_seq)]
    dram = ttnn.DRAM_MEMORY_CONFIG
    nphys, nlog, pts = _cache(dev, P, W, n_seq, extra_rows=len(pos) if items_r else 0)
    Kc = torch.zeros(nphys, NKV, BLK, HD)
    Vc = torch.zeros(nphys, NKV, BLK, HD)
    for s in range(n_seq):
        for p in range(P):  # in position order (a ring overwrites)
            b, o = divmod(p, BLK)
            Kc[pts[s][b], :, o] = Ks[s][p]
            Vc[pts[s][b], :, o] = Vs[s][p]
    K = ttnn.from_torch(Kc, dtype=KV_DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    V = ttnn.from_torch(Vc, dtype=KV_DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    Kq, Vq = ttnn.to_torch(K).float(), ttnn.to_torch(V).float()
    seq_of = (lambda j: j) if users else (lambda j: 0)
    Kpos = [torch.stack([Kq[pts[seq_of(j)][p // BLK], :, p % BLK] for p in range(P)]) for j in range(R)]
    Vpos = [torch.stack([Vq[pts[seq_of(j)][p // BLK], :, p % BLK] for p in range(P)]) for j in range(R)]
    page_table = torch.stack([t.to(torch.int32) for t in pts])  # [n_seq, nlog]
    q = ttnn.from_torch(q_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    qf = ttnn.to_torch(q).float()[0]
    rq_h = torch.zeros(1, R, NH, 32)
    rq_h[..., :DREL] = r
    rq = ttnn.from_torch(rq_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    rf = ttnn.to_torch(rq).float()[0, :, :, :DREL]
    sink = lse(qf, Kpos, ttnn.to_torch(rq).float()[0, :, :, :DREL], (proj / SCALE).bfloat16().float() * SCALE, pos, W)
    sink = sink + torch.randn(R, NH)
    sk_h = (sink / SCALE).reshape(1, R, NH, 1).expand(1, R, NH, 32).contiguous()
    sk = ttnn.from_torch(sk_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    sink_eff = ttnn.to_torch(sk).float()[0, :, :, 0] * SCALE
    T = relbias.build_tables(proj, SCALE)
    table = ttnn.from_torch(
        T.reshape(1, 32, 32, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram
    )
    proj_eff = (proj / SCALE).bfloat16().float() * SCALE  # the table holds bf16(proj / scale)
    pos_t = torch.tensor(pos, dtype=torch.int32)
    if items_r:
        pos_t = pos_t.reshape(-1, items_r)
    elif users:
        pos_t = pos_t.reshape(-1, 1)
    cur = ttnn.from_torch(pos_t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=dram)
    ptt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=dram)
    out = inkling_sdpa_decode(
        q, K, V, cur, ptt, rq, table, SCALE, W, E, sink=sk, pt_per_item=bool(users), k_chunk_size=256 if users else 0,
        mi=MI if items_r else 1,
    )
    got = ttnn.to_torch(out).float().reshape(R, NH, HD)  # ([1, 1, R, NH * HD] when mi items fill a tile row)
    want = reference(qf, Kpos, Vpos, rf, proj_eff, sink_eff, pos, W)
    nob = reference(qf, Kpos, Vpos, rf, proj_eff, sink_eff, pos, W, use_bias=False)
    nos = reference(qf, Kpos, Vpos, rf, proj_eff, sink_eff, pos, W, use_sink=False)
    keep = [j for j in range(R) if pos[j] >= 0]
    res = (pcc(got[keep], want[keep]), pcc(got[keep], nob[keep]), pcc(got[keep], nos[keep]))
    # discrimination: the op must be much closer to the full reference than to either control
    e_ok = (got[keep] - want[keep]).norm().item()
    res = res + (e_ok / (got[keep] - nob[keep]).norm().item(), e_ok / (got[keep] - nos[keep]).norm().item())
    err = (got[keep] - want[keep]).abs().max().item()
    for t in (q, K, V, rq, sk, table, cur, ptt, out):
        ttnn.deallocate(t)
    return res, err


W_SWA, E = 1024, 1024
CASES = [  # (W, positions)
    (0, [0]),
    (0, [5]),
    (0, [1023]),
    (0, [1024]),
    (0, [1500]),
    (0, [9000]),
    (0, [1497, 1498, 1499, 1500]),
    (0, [4093, -1, 4095, 4096]),
    (W_SWA, [0]),
    (W_SWA, [100]),
    (W_SWA, [1023]),
    (W_SWA, [1024]),
    (W_SWA, [1025]),
    (W_SWA, [5000]),
    (W_SWA, [1500, 1501, 1502]),
    (0, [2000 + j for j in range(32 // G)]),
    (W_SWA, [3000 + j for j in range(32 // G)]),
]
PREFILL = [(0, 0, 640), (W_SWA, 0, 640), (0, 1000, 256), (W_SWA, 2040, 256), (0, 4000, 256)]
MI = int(os.environ.get("MI", "1"))  # prefill items per SDPA work unit
MULTI = [(0, [0, 17, 300, 1100, 2047, 64, 5, 999]), (W_SWA, [0, 17, 300, 1100, 2047, 64, 5, 999, 3000, 1024])]


def main():
    dev = ttnn.open_device(device_id=0)
    worst, fails = 1.0, 0
    jobs = [("dec", W, pos, 0, 0) for W, pos in CASES]
    jobs += [("pf", W, list(range(p0, p0 + n)), 8, 0) for W, p0, n in PREFILL]
    jobs += [("mu", W, pos, 0, len(pos)) for W, pos in MULTI]
    for i, (kind, W, pos, items_r, users) in enumerate(jobs):
        (p, p_nob, p_nos, rb, rs), err = run_case(dev, W, E, pos, seed=i, items_r=items_r, users=users)
        ok = p > 0.998 and rb < 0.2 and rs < 0.2
        fails += not ok
        worst = min(worst, p)
        lbl = pos if len(pos) <= 4 else f"{pos[0]}..{pos[-1]} ({len(pos)})"
        print(f"{'ok ' if ok else 'BAD'} {kind} W={W} pos={lbl}: pcc {p:.5f} (no-bias {p_nob:.4f}, no-sink {p_nos:.4f})"
              f" max err {err:.4f} err ratio vs controls {rb:.3f} / {rs:.3f}", flush=True)
    ttnn.close_device(dev)
    print(f"{'PASS' if not fails else 'FAIL'}: {len(jobs) - fails}/{len(jobs)}, worst pcc {worst:.5f}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
