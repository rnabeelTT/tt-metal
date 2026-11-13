# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Op test + bench of the shared-KV rows SDPA decode (models/demos/blackhole/qwen36/tt/attn_decode/rows_sdpa.py) on
one die at the MiniMax-M2.7 per-die shapes (12 q heads, 2 kv heads, head dim 128, bfp8 paged KV, block 64, R = 4).

Correctness: for each context and row placement (consecutive rows; rows straddling a 32-position tile, a K chunk and the
core-0 / core-1 split boundary, incl. a row with no attended position on core 0; a skipped row), the new op vs the stock
paged SDPA decode with the rows as 4 users sharing the page table (what TPAttention._forward_rows_ops runs), both vs an
FP32 torch reference on the same (bfp8-rounded) cache. The cache has a random block permutation, and the K / V at the
rows' positions are planted (K aligned with the rows' queries, distinct V) to take a large share of each row's softmax,
so a row seeing a position past its limit (a mask leak) is a gross error. The core-0 / core-1 straddle exists only while
core 0 holds one chunk (under ~28K context at 55 cores per kv head); past that, the rows always sit on core 0. Bench: trace replay us per call (stock 1 user / 4 users vs the new op).
usage (in the image, run_in_image.sh with FORK=1): POC_CTXS=100,1000,8000,40000  POC_BENCH=1  POC_KCHUNK=256"""

import json
import os
import sys
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.attn_decode.rows_sdpa import rows_sdpa_decode

NH, NKV, HD, BLK, R = 12, 2, 128, 64, 4
G = NH // NKV
CTXS = [int(c) for c in os.environ.get("POC_CTXS", "100,1000,8000,40000").split(",")]
KCH = int(os.environ.get("POC_KCHUNK", "256"))  # the model's sdpa_decode_k_chunk (optimized_decoder SELECTED)
BENCH = os.environ.get("POC_BENCH", "1") == "1"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
SCALE = HD**-0.5


def workload(cur_pos, core_num, ncores, chunk):
    """rt_args_common.hpp get_workload_for_core (no sliding window) -> (num_chunks, start, end)."""
    n = -(-(cur_pos + 1) // chunk)
    if ncores > n:
        per = 1 if core_num < n else 0
        return n, (n - core_num - 1) * per, (n - core_num) * per
    per, res = divmod(n, ncores)
    rev = ncores - core_num - 1
    s = rev * per + min(res, rev)
    return n, s, s + per + (1 if rev < res else 0)


def chunk_tiles(max_pos):
    if KCH:
        return KCH // 32
    t = max_pos // 32 + 1
    p = 1
    while p < t and p < 8:
        p *= 2
    return p


def placements(ctx, cpk):
    """[(name, positions)] near ctx: consecutive, tile / chunk / core-split straddles, a skipped row."""
    p0 = max(ctx - R, 0)
    out = [("consecutive", [p0 + j for j in range(R)])]
    t = (ctx // 32) * 32  # a tile boundary at or below ctx
    if t >= 2:
        out.append(("tile straddle", [t - 2 + j for j in range(R)]))
    ch = chunk_tiles(ctx) * 32
    c = (ctx // ch) * ch
    if c >= 1:
        out.append(("chunk straddle (row0 last in chunk)", [c - 1 + j for j in range(R)]))
    # the core-0 / core-1 boundary (core 0 holds the last chunks): row 0 just below it (it has no attended position on
    # core 0) and rows 0..1 below it
    for k, name in ((1, "core straddle (row0 none on core 0)"), (2, "core straddle (rows 0-1 below)")):
        for p in range(ctx, max(ctx - 4000, 0), -1):
            mx = p + R - 1
            ch = chunk_tiles(mx) * 32
            n, s, e = workload(mx, 0, cpk, ch)
            if n > 1 and s * ch == p + k:
                out.append((name, [p + j for j in range(R)]))
                break
    out.append(("skipped row 1", [p0, -1, p0 + 2, p0 + 3]))
    return out


def stats(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item() if a.std() > 0 and b.std() > 0 else float("nan")
    return pcc, (a - b).abs().max().item()


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 << 20)
    grid = dev.compute_with_storage_grid_size()
    cpk = min(grid.x * grid.y // NKV, 64)
    res = {"grid": [grid.x, grid.y], "cores_per_kv_head": cpk, "k_chunk": KCH, "cases": [], "bench": {}}
    dram = ttnn.DRAM_MEMORY_CONFIG
    i32 = lambda t: ttnn.from_torch(  # noqa: E731
        t.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=dram
    )
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=KCH
    )
    worst = {"pcc_vs_stock": 1.0, "pcc_vs_ref": 1.0, "err_ratio": 0.0}
    try:
        for ctx in CTXS:
            nblk = (ctx + 256) // BLK + 2
            perm = torch.randperm(nblk)
            q_h = torch.randn(1, R, NH, HD)
            Kv = torch.randn(nblk * BLK, NKV, HD)  # virtual positions
            Vv = torch.randn(nblk * BLK, NKV, HD)
            # every placement's row positions get a K aligned with the rows' queries (scaled score ~8 for the mean
            # query: a large share of each row's softmax) and a distinct V, so a row attending past its limit moves
            # its output far
            p0 = max(ctx - R, 0)
            cases = placements(ctx, cpk)
            planted = sorted({p for _, ps in cases for p in ps if p >= 0})
            for h in range(NKV):
                qd = q_h[0, :, h * G : (h + 1) * G].reshape(-1, HD).mean(0)
                qd = qd / qd.norm()
                a = 8.0 / (SCALE * (q_h[0, :, h * G : (h + 1) * G].reshape(-1, HD) @ qd).mean())
                for pos in planted:
                    Kv[pos, h] = a * qd
                    Vv[pos, h] = 3.0 * torch.randn(HD)
            phys = lambda X: X.reshape(nblk, BLK, NKV, HD).permute(0, 2, 1, 3)[torch.argsort(perm)]  # noqa: E731
            K = ttnn.from_torch(phys(Kv), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
            V = ttnn.from_torch(phys(Vv), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
            # the bfp8-rounded cache back in virtual order (the reference reads the same bytes)
            Kq = ttnn.to_torch(K).float()[perm].permute(0, 2, 1, 3).reshape(nblk * BLK, NKV, HD)
            Vq = ttnn.to_torch(V).float()[perm].permute(0, 2, 1, 3).reshape(nblk * BLK, NKV, HD)
            q = ttnn.from_torch(q_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
            qf = ttnn.to_torch(q).float()
            pt = i32(perm.repeat(R, 1))
            for name, positions in [] if os.environ.get("POC_ONLY_BENCH") else cases:
                pos_t = i32(torch.tensor(positions))
                new = ttnn.to_torch(rows_sdpa_decode(q, K, V, pos_t, pt, SCALE, KCH)).float()
                st = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q,
                    K,
                    V,
                    page_table_tensor=pt,
                    cur_pos_tensor=pos_t,
                    scale=SCALE,
                    program_config=cfg,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                stock = ttnn.to_torch(st).float()
                for r, p in enumerate(positions):
                    if p < 0:
                        continue
                    ref = torch.empty(NH, HD)
                    for hq in range(NH):
                        h = hq // G
                        s = (Kq[: p + 1, h] @ qf[0, r, hq]) * SCALE
                        ref[hq] = torch.softmax(s.double(), -1).float() @ Vq[: p + 1, h]
                    pn, en = stats(new[0, r, :NH], ref)
                    ps, es = stats(stock[0, r, :NH], ref)
                    pns, ens = stats(new[0, r, :NH], stock[0, r, :NH])
                    row = dict(
                        ctx=ctx,
                        case=name,
                        row=r,
                        pos=p,
                        pcc_new_ref=pn,
                        err_new_ref=en,
                        pcc_stock_ref=ps,
                        err_stock_ref=es,
                        pcc_new_stock=pns,
                        err_new_stock=ens,
                    )
                    res["cases"].append(row)
                    worst["pcc_vs_stock"] = min(worst["pcc_vs_stock"], pns)
                    worst["pcc_vs_ref"] = min(worst["pcc_vs_ref"], pn)
                    worst["err_ratio"] = max(worst["err_ratio"], en / max(es, 1e-6))
                    print(
                        f"ROWSDPA ctx {ctx:6d} {name:38s} row {r} pos {p:6d}: new-ref pcc {pn:.6f} err {en:.4f} | "
                        f"stock-ref pcc {ps:.6f} err {es:.4f} | new-stock pcc {pns:.6f} err {ens:.4f}",
                        flush=True,
                    )
                ttnn.deallocate(st)
            if BENCH:
                pos4 = i32(torch.tensor([p0 + j for j in range(R)]))
                pos1 = i32(torch.tensor([p0 + R - 1]))
                pt1 = i32(perm.reshape(1, -1))
                q1 = ttnn.from_torch(
                    q_h[:, :1], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram
                )

                def stock_call(qq, pp, tt):
                    def f():
                        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                            qq,
                            K,
                            V,
                            page_table_tensor=tt,
                            cur_pos_tensor=pp,
                            scale=SCALE,
                            program_config=cfg,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                        )
                        ttnn.deallocate(o)

                    return f

                def new_call():
                    ttnn.deallocate(rows_sdpa_decode(q, K, V, pos4, pt, SCALE, KCH))

                b = {
                    "stock 1 user": bench(dev, stock_call(q1, pos1, pt1)),
                    "stock 4 users": bench(dev, stock_call(q, pos4, pt)),
                    "rows shared": bench(dev, new_call),
                }
                res["bench"][ctx] = b
                print(f"ROWSDPA_BENCH ctx {ctx}: " + ", ".join(f"{k} {v:.1f} us" for k, v in b.items()), flush=True)
            for t in (K, V, q, pt):
                ttnn.deallocate(t)
        res["worst"] = worst
        print("ROWSDPA_WORST " + json.dumps(worst), flush=True)
        with open(os.path.join(OUT, "result.json"), "w") as f:
            json.dump(res, f, indent=1)
    finally:
        ttnn.close_device(dev)


def bench(dev, fn, reps=10, iters=20):
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.time()
    for _ in range(iters):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ttnn.release_trace(dev, tid)
    return (time.time() - t0) / (reps * iters) * 1e6


if __name__ == "__main__":
    main()
