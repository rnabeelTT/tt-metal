# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device cost of the decode indexer selection (tt/indexer.py Hy4Indexer._topk) vs the score width, per op, on random
inputs (no model load): indexer_score_dsa -> tilize -> fp32 -> visibility mask -> bf16 -> row-major -> top-512.
    python -m models.autoports.tencent_hy4_flash.tools.topk_bench [--widths 2048,4064,8192,16384,32768] [--rows 32]
Each op is timed alone (sync before and after, median of N): includes ~launch overhead per op.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import ttnn


def timed(mesh, fn, n):
    ts = []
    out = None
    for i in range(n):
        if out is not None:
            for t in (out if isinstance(out, (list, tuple)) else [out]):
                ttnn.deallocate(t)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh)
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e6, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="2048,4064,8192,16384,32768")
    ap.add_argument("--rows", default="32")
    ap.add_argument("--gcap", type=int, default=34816)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--score-sweep", action="store_true")
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import open_mesh

    mesh = open_mesh()
    rep = ttnn.ReplicateTensorToMesh(mesh)
    dev = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh, mesh_mapper=rep)  # noqa: E731
    L1 = ttnn.L1_MEMORY_CONFIG
    DR = ttnn.DRAM_MEMORY_CONFIG
    try:
        g = torch.Generator().manual_seed(0)
        gk = dev(torch.randn(1, 1, args.gcap, 128, generator=g).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        q = dev(torch.randn(1, 16, 32, 128, generator=g).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        w = dev(torch.randn(1, 1, 32, 16, generator=g).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        iota_all = dev(torch.arange(args.gcap + 32, dtype=torch.float32).reshape(1, 1, 1, -1), ttnn.float32,
                       ttnn.TILE_LAYOUT)
        for W in [int(x) for x in args.widths.split(",")]:
            vis = dev(torch.full((1, 1, 32, 1), float(W - 7)), ttnn.float32, ttnn.TILE_LAYOUT)
            one = ttnn.slice(iota_all, (0, 0, 0, 0), (1, 1, 1, W + 32))
            iota = ttnn.repeat(one, ttnn.Shape((1, 1, 32, 1)))
            res = {}
            res["score"], sc = timed(mesh, lambda: ttnn.experimental.indexer_score_dsa(
                q, gk, w, chunk_start_idx=W, kv_len=W + 32), args.n)
            shape = tuple(sc.shape)
            if args.score_sweep:  # the score op's program config (q chunk 32): head group size x k chunk size
                for hg in (1, 2, 4, 8, 16, 0):
                    for kc in (32, 64, 128, 256, 512):
                        try:
                            pc = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=kc, head_group_size=hg)
                            us, s_ = timed(mesh, lambda: ttnn.experimental.indexer_score_dsa(
                                q, gk, w, chunk_start_idx=W, kv_len=W + 32, program_config=pc), args.n)
                            a = ttnn.to_torch(ttnn.get_device_tensors(s_)[0]).float()[..., :W]
                            b = ttnn.to_torch(ttnn.get_device_tensors(sc)[0]).float()[..., :W]
                            print(f"TOPK_BENCH W={W} score hg {hg} kc {kc}: {us:.0f} us; same as default "
                                  f"{bool(torch.equal(a, b))} max diff {(a - b).abs().max().item():.4g}", flush=True)
                            ttnn.deallocate(s_)
                        except Exception as e:  # noqa: BLE001
                            print(f"TOPK_BENCH W={W} score hg {hg} kc {kc} failed: {str(e)[:150]}", flush=True)
            if shape[-1] != W + 32:
                res["slice"], sc2 = timed(mesh, lambda: ttnn.slice(sc, (0, 0, 0, 0), (1, 1, 32, W + 32),
                                                                     memory_config=L1), args.n)
            else:
                sc2 = sc
            lay = sc2.layout
            if lay != ttnn.TILE_LAYOUT:
                res["tilize"], st = timed(mesh, lambda: ttnn.to_layout(sc2, ttnn.TILE_LAYOUT, memory_config=L1), args.n)
            else:
                st = sc2
            res["tc32"], s32 = timed(mesh, lambda: ttnn.typecast(st, ttnn.float32, memory_config=L1), args.n)
            res["ge"], hide = timed(mesh, lambda: ttnn.ge(iota, vis, memory_config=L1), args.n)
            res["where"], m = timed(mesh, lambda: ttnn.where(hide, float("-inf"), s32, memory_config=L1), args.n)
            res["tc16"], mb = timed(mesh, lambda: ttnn.typecast(m, ttnn.bfloat16, memory_config=L1), args.n)
            res["to_rm"], rm = timed(mesh, lambda: ttnn.to_layout(mb, ttnn.ROW_MAJOR_LAYOUT, memory_config=DR), args.n)
            res["topk32"], ids = timed(mesh, lambda: ttnn.experimental.topk_large_indices(rm, k=512), args.n)
            for R in [int(x) for x in args.rows.split(",") if int(x) != 32]:
                r1 = ttnn.slice(rm, (0, 0, 0, 0), (1, 1, R, W + 32), memory_config=DR)
                res[f"topk{R}"], i1 = timed(mesh, lambda: ttnn.experimental.topk_large_indices(r1, k=512), args.n)
                ttnn.deallocate(i1)
                ttnn.deallocate(r1)
            # variants: k=1024 body (FusedSegmented) sliced to 512; valid_length_tensor (dynamic visible width)
            try:
                res["topk32_k1024"], i2 = timed(mesh, lambda: ttnn.experimental.topk_large_indices(rm, k=1024), args.n)
                ttnn.deallocate(i2)
            except Exception as e:  # noqa: BLE001
                print(f"TOPK_BENCH k1024 failed: {e}", flush=True)
            try:
                vl = ttnn.from_torch(torch.tensor([W // 2], dtype=torch.int32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
                res["topk32_valid_half"], i3 = timed(mesh, lambda: ttnn.experimental.topk_large_indices(
                    rm, k=512, valid_length_tensor=vl), args.n)
                ttnn.deallocate(i3)
            except Exception as e:  # noqa: BLE001
                print(f"TOPK_BENCH valid_length_tensor failed: {e}", flush=True)
            # the mask-free path (tt/indexer.Hy4Indexer._topk_vlen): time + exactness vs the masked top-512 (die 0)
            from types import SimpleNamespace

            from models.autoports.tencent_hy4_flash.tt.indexer import Hy4Indexer

            t0_ = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
            fake = SimpleNamespace(mesh=mesh, gkeys=gk, r=4)
            Hy4Indexer.decode_consts(mesh, SimpleNamespace(idx_group_topk=512, idx_ratio=4))
            for m0 in (W - 7, W // 3 + 5):
                for pat in ((0,), (0, 1, 1, 1, 1), (0, 0, 0, 0, 1), (1, 1, 1, 1, 1)):
                    R = len(pat)
                    vr = torch.tensor([m0 + pat[min(i, R - 1)] for i in range(32)], dtype=torch.float32)
                    if int(vr.max()) > W:
                        continue
                    vis_t = dev(vr.reshape(1, 1, 32, 1), ttnn.float32, ttnn.TILE_LAYOUT)
                    h2 = ttnn.ge(iota, vis_t, memory_config=L1)
                    m2 = ttnn.where(h2, float("-inf"), s32, memory_config=L1)
                    mb2 = ttnn.typecast(m2, ttnn.bfloat16, memory_config=L1)
                    rm2 = ttnn.to_layout(mb2, ttnn.ROW_MAJOR_LAYOUT, memory_config=DR)
                    ref = t0_(ttnn.experimental.topk_large_indices(rm2, k=512)).reshape(32, 512).long()[:, :511]
                    for t_ in (h2, m2, mb2, rm2):
                        ttnn.deallocate(t_)
                    vmax = int(vr.max())
                    si = SimpleNamespace(bucket=W, rows=R,
                                         vlen=ttnn.from_torch(torch.tensor([vmax], dtype=torch.int32).reshape(1, 1, 1, 1),
                                                              dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                              device=mesh, mesh_mapper=rep),
                                         vfix=dev(torch.where(vr < vmax, vr, torch.full_like(vr, -1.0)).reshape(1, 1, 32, 1),
                                                  ttnn.float32, ttnn.TILE_LAYOUT),
                                         vcol=ttnn.from_torch(torch.where(vr < vmax, vr, torch.full_like(vr, float(args.gcap - 1)))
                                                              .to(torch.int32).reshape(1, 1, 32, 1), dtype=ttnn.uint32,
                                                              layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep))
                    for vfix in (("scatter", "shift") if R > 1 else ("-",)):
                      os.environ["HY4_IDX_VFIX"] = vfix
                      try:
                        us, f = timed(mesh, lambda: Hy4Indexer._topk_vlen(fake, q, w, si, L1, R, 511), args.n)
                        got = t0_(f).reshape(32, 511)[:R].round().long()
                        same = bool(torch.equal(got, ref[:R]))
                        print(f"TOPK_BENCH W={W} vlen path rows {R} fix {vfix} vis {m0}+{pat}: {us:.0f} us (incl. score); "
                              f"identical to masked {same}", flush=True)
                        if not same:  # ties? the scores at each rank (die 0's masked fp32 scores), and the sets
                            sv = t0_(s32).float().reshape(32, -1)
                            vals_eq = all(torch.equal(sv[r][got[r]], sv[r][ref[r]]) for r in range(R))
                            sets_eq = all(set(got[r].tolist()) == set(ref[r].tolist()) for r in range(R))
                            print(f"TOPK_BENCH   W={W} rows {R} vis {m0}+{pat}: score at every rank equal {vals_eq}; "
                                  f"same sets {sets_eq}", flush=True)
                            for r in range(R):
                                d = (got[r] != ref[r]).nonzero().flatten()
                                if d.numel():
                                    print(f"TOPK_BENCH   row {r}: {d.numel()} diffs, first at {int(d[0])}: got "
                                          f"{got[r, int(d[0]):int(d[0]) + 4].tolist()} ref {ref[r, int(d[0]):int(d[0]) + 4].tolist()}",
                                          flush=True)
                        ttnn.deallocate(f)
                      except Exception as e:  # noqa: BLE001
                        print(f"TOPK_BENCH W={W} vlen rows {R} fix {vfix} failed: {e}", flush=True)
                    for t_ in (vis_t, si.vlen, si.vfix, si.vcol):
                        ttnn.deallocate(t_)
            base = ("score", "slice", "tilize", "tc32", "ge", "where", "tc16", "to_rm", "topk32")
            tot = sum(v for k, v in res.items() if k in base)
            print(f"TOPK_BENCH W={W} score_out={shape} layout={lay} total {tot:.0f} us: "
                  + " ".join(f"{k} {v:.0f}" for k, v in res.items()), flush=True)
            for t in (sc, s32, hide, m, mb, rm, ids, vis, one, iota):
                try:
                    ttnn.deallocate(t)
                except Exception:  # noqa: BLE001
                    pass
            if sc2 is not sc:
                ttnn.deallocate(sc2)
            if st is not sc2:
                ttnn.deallocate(st)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
