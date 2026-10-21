# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 10: which part of the sparse prefill selection depends on the build's max_seq_len T (synthetic keys).

For a bucket context T_b (scored width W = T_b/4 + 32) and an "ON" build of context T (P_alloc = T/4 + 32):
  A. indexer_score_dsa on the T-build pool view (length P_alloc) with kv_len = W, chunk_start W - 32, columns < W
     vs the T_b-build view (length W), no kv_len: bit-equal?
  B. the same indexer call with rb rows vs the first rb rows of a 2048-row call: bit-equal? (GlmDSA.index_row_block
     sizes rb from P_alloc)
  C. the selection (to_layout, typecast, iota mask, top-512) of identical [rows, W] scores, rb rows vs the first rb
     rows of 2048: ids equal?
Usage: python -m ...tools.dsa_bucket_probe2 --out rec.json
"""

import argparse
import json

import torch

import ttnn

HI, D, K = 32, 128, 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64 << 20)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    dev = lambda t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t, dtype=dt, layout=lay, device=mesh, mesh_mapper=rep
    )
    host = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    rec = {"A": [], "B": [], "C": []}
    g = torch.Generator().manual_seed(11)
    keys = torch.randn(65568 + 64, D, generator=g).bfloat16()  # pools of slot 0 (logical order)
    qi_h = (torch.randn(1, HI, 2048, D, generator=g) * 0.1).bfloat16()
    wi_h = (torch.randn(1, 1, 2048, HI, generator=g) * 0.2).bfloat16()

    def pool(T):
        """Pool-key cache of a T-token 1-slot build: [KB * P_alloc / 32, 1, 32, D], slot 0 owns blocks 0 .. T/128-1."""
        Pa = T // 4 + 32
        nb = T // 128
        NB = nb + 1
        KB = -(-NB * 32 // Pa)
        k = torch.zeros(KB * Pa, D).bfloat16()
        k[: nb * 32] = keys[: nb * 32]
        pt = torch.full((1, Pa // 32), NB - 1, dtype=torch.int32)
        pt[0, :nb] = torch.arange(nb, dtype=torch.int32)
        pooled = ttnn.from_torch(
            k.reshape(-1, 1, 32, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        return Pa, ttnn.reshape(pooled, (KB, 1, Pa, D)), dev(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), pooled

    def idx(view, pt, rows, W, kv_len):
        qi = dev(qi_h[:, :, :rows].float(), ttnn.bfloat16)
        wi = dev(wi_h[:, :, :rows].float(), ttnn.bfloat16)
        kw = {"kv_len": W} if kv_len else {}
        s = ttnn.experimental.indexer_score_dsa(
            qi, view, wi, chunk_start_idx=W - 32, cache_batch_idx=0, page_table=pt, **kw
        )
        return host(s).reshape(rows, -1)[:, :W]

    try:
        builds = {T: pool(T) for T in (8192, 16384, 65536, 262144)}
        for T_on, T_b in ((262144, 4096), (262144, 8192), (262144, 16384), (65536, 8192), (65536, 16384)):
            W = T_b // 4 + 32
            Pa_b, view_b, pt_b, _ = builds[T_b] if T_b in builds else pool(T_b)
            Pa, view, pt, _ = builds[T_on]
            for rows in (256, 2048):
                c = W - 64  # columns visible to every row (outside the causal band)
                ref = idx(view_b, pt_b, rows, W, False)[:, :c]
                on = idx(view, pt, rows, W, True)[:, :c]
                big = idx(view, pt, rows, W, False)[:, :c]
                r = {
                    "T_on": T_on,
                    "T_b": T_b,
                    "rows": rows,
                    "kvlen_bigview_eq_Tb_view": bool(torch.equal(ref, on)),
                    "bigview_eq_Tb_view": bool(torch.equal(ref, big)),
                    "kvlen_eq_nokvlen_bigview": bool(torch.equal(on, big)),
                    "max_abs_d_kvlen_vs_Tb": float((ref.float() - on.float()).abs().max()),
                    "max_abs_d_big_vs_Tb": float((ref.float() - big.float()).abs().max()),
                }
                rec["A"].append(r)
                print(f"A {r}", flush=True)
            import time

            for kv in (True, False):
                qi = dev(qi_h[:, :, :256].float(), ttnn.bfloat16)
                wi = dev(wi_h[:, :, :256].float(), ttnn.bfloat16)
                kw = {"kv_len": W} if kv else {}
                f = lambda: ttnn.experimental.indexer_score_dsa(
                    qi, view, wi, chunk_start_idx=W - 32, cache_batch_idx=0, page_table=pt, **kw
                )
                ttnn.deallocate(f())
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(5):
                    ttnn.deallocate(f())
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - t0) * 200
                rec["A"].append({"T_on": T_on, "W": W, "kv_len": kv, "rows": 256, "ms": ms})
                print(f"TIME indexer T_on {T_on} W {W} kv_len {kv}: {ms:.2f} ms", flush=True)
        for T in ():
            Pa, view, pt, _ = builds[T]
            W = Pa
            r2048 = idx(view, pt, 2048, W, False)
            for rows in (256, 1024):
                eq = bool(torch.equal(idx(view, pt, rows, W, False), r2048[:rows]))
                rec["B"].append({"T": T, "rows": rows, "equal": eq})
                print(f"B T {T} rows {rows} vs 2048: equal {eq}", flush=True)
            # C: selection of the same scores
            iota = dev(torch.arange(W).float().reshape(1, 1, 1, W))

            def sel(s):
                rows = s.shape[0]
                sc = ttnn.from_torch(
                    s.reshape(1, 1, rows, W),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    mesh_mapper=rep,
                )
                lim = dev(torch.full((1, 1, rows, 1), float(W - 32)))
                t = ttnn.to_layout(sc, ttnn.TILE_LAYOUT)
                ok = ttnn.typecast(ttnn.lt(iota, lim), ttnn.bfloat16)
                _, ids = ttnn.topk(ttnn.where(ok, t, float("-inf")), K, dim=-1, largest=True, sorted=True)
                return host(ids).reshape(rows, K)

            i2048 = sel(r2048)
            for rows in (256, 1024):
                got = sel(r2048[:rows])
                rec["C"].append(
                    {
                        "T": T,
                        "rows": rows,
                        "order_equal": bool(torch.equal(got, i2048[:rows])),
                        "set_equal": bool(torch.equal(got.sort(-1).values, i2048[:rows].sort(-1).values)),
                    }
                )
                print(f"C T {T} rows {rows} vs 2048: {rec['C'][-1]}", flush=True)
    finally:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
