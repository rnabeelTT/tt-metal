# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test of tt/attn/inkling_sdpa.inkling_sdpa_decode on one die at Inkling's per-die shapes (8 q heads, 2 kv
heads, head dim 128, TP4), against an fp32 torch reference of Inkling attention on the same (dtype-rounded) cache:

  scores = q.k * scale + bias(r_q, proj)[q - k]  over k in [lo, pos]  (lo = pos + 1 - W on sliding layers), softmax, @ V

Cases: global layers (W 0, E 1024, a linear paged cache with a shuffled page table) and sliding layers (W = E = 512, a
ring cache: logical block b -> physical b mod n_ring, positions written in order so old ones are overwritten), R = 1
(decode) and R = 4 (verify: consecutive rows, one skipped), positions around tiles, chunks, E and W. A control case runs
the reference without the bias: the op must be far from it, so a silently dropped bias fails.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tests/test_sdpa_device.py
"""

import os
import sys

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tt import relbias
from models.autoports.thinkingmachines_inkling_small.tt.attn.inkling_sdpa import inkling_sdpa_decode

NH, NKV, HD, DREL, BLK = 8, 2, 128, 16, 64
G = NH // NKV
SCALE = 1.0 / HD  # Inkling: q/k RMS-normalised, scale 1/d
KV_DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("KV_DT", "bfp8")]


def reference(qf, Kpos, Vpos, r, proj, pos, W, use_bias=True):
    """qf [R, NH, HD], Kpos/Vpos [P, NKV, HD] (value at each position), r [R, NH, DREL] -> [R, NH, HD] fp32."""
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
            s = (Kpos[ks, g] @ qf[j, h]) * SCALE
            if use_bias:
                d = p - ks
                rel = r[j, h] @ proj  # [E]
                s = s + torch.where(d < E, rel[d.clamp(max=E - 1)], torch.zeros(()))
            a = torch.softmax(s, 0)
            out[j, h] = a @ Vpos[ks, g]
    return out


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run_case(dev, W, E, pos, seed):
    torch.manual_seed(seed)
    R = len(pos)
    P = max(pos) + 1
    proj = torch.randn(DREL, E) * 0.5
    r = torch.randn(R, NH, DREL) * 0.5
    q_h = torch.nn.functional.normalize(torch.randn(1, R, NH, HD), dim=-1) * HD**0.5  # RMS-normed scale
    Kp = torch.nn.functional.normalize(torch.randn(P, NKV, HD), dim=-1) * HD**0.5
    Vp = torch.randn(P, NKV, HD)
    dram = ttnn.DRAM_MEMORY_CONFIG
    nlog = (P + BLK - 1) // BLK
    if W == 0:  # linear cache, shuffled physical blocks
        nphys = nlog + 2
        perm = torch.randperm(nphys)[:nlog]  # logical b -> physical perm[b]
        pt = perm
    else:  # ring cache
        nphys = W // BLK + 1
        pt = torch.arange(nlog) % nphys
    Kc = torch.zeros(nphys, NKV, BLK, HD)
    Vc = torch.zeros(nphys, NKV, BLK, HD)
    for p in range(P):  # in position order (the ring overwrites)
        b, o = divmod(p, BLK)
        Kc[pt[b], :, o] = Kp[p]
        Vc[pt[b], :, o] = Vp[p]
    K = ttnn.from_torch(Kc, dtype=KV_DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    V = ttnn.from_torch(Vc, dtype=KV_DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    Kq = ttnn.to_torch(K).float()
    Vq = ttnn.to_torch(V).float()
    # the value each position holds in the (rounded) cache = what the op reads (in-window positions are live)
    Kpos = torch.stack([Kq[pt[p // BLK], :, p % BLK] for p in range(P)])
    Vpos = torch.stack([Vq[pt[p // BLK], :, p % BLK] for p in range(P)])
    page_table = torch.zeros(R, max(nlog, 1), dtype=torch.int32)
    page_table[:] = pt[:nlog].to(torch.int32)
    q = ttnn.from_torch(q_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    qf = ttnn.to_torch(q).float()[0]
    rq_h = torch.zeros(1, R, NH, 32)
    rq_h[..., :DREL] = r
    rq = ttnn.from_torch(rq_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    rf = ttnn.to_torch(rq).float()[0, :, :, :DREL]
    T = relbias.build_tables(proj, SCALE)
    table = ttnn.from_torch(
        T.reshape(1, 32, 32, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram
    )
    # the device table holds bf16(proj / scale) and 1/scale = 128 is a power of two: its proj is bf16(proj)
    proj_eff = proj.bfloat16().float()
    cur = ttnn.from_torch(
        torch.tensor(pos, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=dram
    )
    ptt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=dram)
    out = inkling_sdpa_decode(q, K, V, cur, ptt, rq, table, SCALE, W, E)
    got = ttnn.to_torch(out).float()[0]
    want = reference(qf, Kpos, Vpos, rf, proj_eff, pos, W)
    nob = reference(qf, Kpos, Vpos, rf, proj_eff, pos, W, use_bias=False)
    keep = [j for j in range(R) if pos[j] >= 0]
    p_ok = pcc(got[keep], want[keep])
    p_nob = pcc(got[keep], nob[keep])
    err = (got[keep] - want[keep]).abs().max().item()
    for t in (q, K, V, rq, table, cur, ptt, out):
        ttnn.deallocate(t)
    return p_ok, p_nob, err


CASES = [
    # (W, E, positions)
    (0, 1024, [0]),
    (0, 1024, [5]),
    (0, 1024, [31]),
    (0, 1024, [600]),
    (0, 1024, [1023]),
    (0, 1024, [1024]),
    (0, 1024, [1500]),
    (0, 1024, [5000]),
    (0, 1024, [20000]),
    (0, 1024, [1497, 1498, 1499, 1500]),
    (0, 1024, [4093, -1, 4095, 4096]),
    (512, 512, [0]),
    (512, 512, [100]),
    (512, 512, [511]),
    (512, 512, [512]),
    (512, 512, [513]),
    (512, 512, [1000]),
    (512, 512, [9000]),
    (512, 512, [700, 701, 702, 703]),
    (512, 512, [8190, 8191, 8192, 8193]),
    (0, 1024, [2000 + j for j in range(8)]),  # 8-row verify (the most rows a tile holds at G = 4)
    (512, 512, [3000 + j for j in range(8)]),
]
if os.environ.get("INK_LONG"):  # long context: the band on one core of many
    CASES += [(0, 1024, [131071]), (0, 1024, [131064 + j for j in range(8)]), (512, 512, [131071])]


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    fails = 0
    try:
        for i, (W, E, pos) in enumerate(CASES):
            p_ok, p_nob, err = run_case(dev, W, E, pos, seed=i)
            ok = p_ok > 0.999 and (p_nob < 0.995 or max(pos) == 0)
            fails += not ok
            print(f"{'OK  ' if ok else 'FAIL'} W={W:3d} E={E:4d} pos={pos}: pcc {p_ok:.5f} (no-bias {p_nob:.4f}) max err {err:.3e}", flush=True)
    finally:
        ttnn.close_device(dev)
    print("ALL OK" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
