# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does inkling_sdpa's decode / verify launch depend on where the KV blocks physically are? One die, Inkling's per-die
shapes (tests/test_sdpa_device.py): the same logical K / V (positions 0 .. P) stored under an identity page table and
under a shifted one (logical b -> physical b + 1, as vLLM's block ids start at 1) and a permuted one; the outputs of
the same q / Rq launch must be bit-identical. Global layers (W = 0, E = 1024), decode (1 row) and verify (3 rows).

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/sdpa_pt_check.py
"""

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tests import test_sdpa_device as T
from models.autoports.thinkingmachines_inkling_small.tt import relbias
from models.autoports.thinkingmachines_inkling_small.tt.attn.inkling_sdpa import inkling_sdpa_decode


def run(dev, W, E, pos, layout, seed=0):
    torch.manual_seed(seed)
    R, P = len(pos), max(pos) + 1
    NH, NKV, HD, DREL, BLK = T.NH, T.NKV, T.HD, T.DREL, T.BLK
    proj = torch.randn(DREL, E) * 0.5
    r = torch.randn(R, NH, DREL) * 0.5
    q_h = torch.nn.functional.normalize(torch.randn(1, R, NH, HD), dim=-1) * HD**0.5
    Kp = torch.nn.functional.normalize(torch.randn(P, NKV, HD), dim=-1) * HD**0.5
    Vp = torch.randn(P, NKV, HD)
    nlog = (P + BLK - 1) // BLK
    nphys = nlog + 2
    if layout == "identity":
        pt = torch.arange(nlog)
    elif layout == "shift":
        pt = torch.arange(nlog) + 1
    else:
        pt = torch.randperm(nphys, generator=torch.Generator().manual_seed(7))[:nlog]
    Kc = torch.zeros(nphys, NKV, BLK, HD)
    Vc = torch.zeros(nphys, NKV, BLK, HD)
    for p in range(P):
        b, o = divmod(p, BLK)
        Kc[pt[b], :, o] = Kp[p]
        Vc[pt[b], :, o] = Vp[p]
    dram = ttnn.DRAM_MEMORY_CONFIG
    K = ttnn.from_torch(Kc, dtype=T.KV_DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    V = ttnn.from_torch(Vc, dtype=T.KV_DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    width = 2048  # the served page table is as wide as the context, not as the sequence
    page_table = torch.zeros(1, width, dtype=torch.int32)
    page_table[0, :nlog] = pt.to(torch.int32)
    page_table[0, nlog:] = (torch.arange(nlog, width) % nphys).to(torch.int32)
    q = ttnn.from_torch(q_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    rq_h = torch.zeros(1, R, NH, 32)
    rq_h[..., :DREL] = r
    rq = ttnn.from_torch(rq_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
    table = ttnn.from_torch(
        relbias.build_tables(proj, T.SCALE).reshape(1, 32, 32, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=dram,
    )
    cur = ttnn.from_torch(torch.tensor(pos, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                          device=dev, memory_config=dram)
    ptt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=dram)
    o = inkling_sdpa_decode(q, K, V, cur, ptt, rq, table, T.SCALE, W, E)
    out = ttnn.to_torch(o).float()
    for t in (q, K, V, rq, table, cur, ptt, o):
        ttnn.deallocate(t)
    return out


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    ok = True
    try:
        for W, E in ((0, 1024), (512, 512)):
            for pos in ([299], [1499], [4499], [1499, 1500, 1501], [4499, 4500, 4501]):
                ref = run(dev, W, E, pos, "identity")
                for layout in ("shift", "perm"):
                    got = run(dev, W, E, pos, layout)
                    same = torch.equal(ref, got)
                    ok &= same or W != 0  # (sliding layers use their own ring table in the model)
                    print(f"W={W} E={E} pos={pos} {layout}: {'identical' if same else f'DIFFERS max |d| {float((ref - got).abs().max()):.4g}'}",
                          flush=True)
        print("RESULT", "INDEPENDENT OF PLACEMENT" if ok else "DEPENDS ON PLACEMENT", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
