# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 10: cost and width invariance of the DSA decode selection (dsa_decode_scores -> mask -> top-512) by scored
width, on a (1, 4) mesh with synthetic pooled keys (no model weights).

Questions it answers:
  1. time per traced replay of the selection at each candidate bucket width W (pools), 1 and 8 slots;
  2. are the top-512 pool ids (values AND order) at width W identical to those at the full P_alloc width, for every
     row whose limit <= W (tie-heavy bf16 scores)? (the bucketed decode is bit-identical only if so);
  3. the per-replay overhead of a small trace (the segmented decode replays ~23 traces per step);
  4. trace-region bytes of one selection trace.

Usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.dsa_bucket_probe --out rec.json
"""

import argparse
import json
import time

import torch

import ttnn

HI, D = 32, 128
K = 512
DRAM = ttnn.DRAM_MEMORY_CONFIG


def p_alloc(T):
    return T // 4 + 32


class Rig:
    def __init__(self, mesh, S, T, seed=0, dup_period=0):
        self.mesh, self.S, self.T = mesh, S, T
        self.rep = ttnn.ReplicateTensorToMesh(mesh)
        self.Pa = p_alloc(T)
        self.MB = self.Pa // 32
        nb_slot = T // 128
        NB = S * nb_slot + 1
        KB = -(-NB * 32 // self.Pa)
        g = torch.Generator().manual_seed(seed)
        nkeys = KB * self.Pa
        if dup_period:  # many exactly equal keys -> exactly tied scores
            base = torch.randn(dup_period, D, generator=g)
            keys = base.repeat(-(-nkeys // dup_period), 1)[:nkeys]
        else:
            keys = torch.randn(nkeys, D, generator=g)
        self.pooled = ttnn.from_torch(
            keys.reshape(nkeys // 32, 1, 32, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=self.rep,
        )
        pt = torch.full((S, self.MB), NB - 1, dtype=torch.int32)
        pt[:, :nb_slot] = torch.arange(S * nb_slot, dtype=torch.int32).reshape(S, nb_slot)
        # pooled block b = the 32 pool keys of page block b (128 tokens); column j of a slot's row = its logical block j
        self.pt = self.dev(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.qi = self.dev((torch.randn(1, 1, S * HI, D, generator=g) * 0.1), ttnn.bfloat16)
        self.wi = self.dev(torch.randn(1, 1, S, HI, generator=g) * 0.2)
        self.limit = self.dev(torch.zeros(1, 1, S, 1))
        self.limit_u32 = self.dev(torch.zeros(1, S, dtype=torch.int64), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.iota = {}

    def dev(self, t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dt, layout=lay, device=self.mesh, mesh_mapper=self.rep)

    def set_positions(self, positions):
        lim = torch.tensor([float((p + 1) // 4) if p >= 0 else 0.0 for p in positions])
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                lim.reshape(1, 1, self.S, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.rep
            ),
            self.limit,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                lim.to(torch.int64).reshape(1, self.S),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.rep,
            ),
            self.limit_u32,
        )
        return lim

    def iota_w(self, W):
        if W not in self.iota:
            self.iota[W] = self.dev(torch.arange(W).float().reshape(1, 1, 1, W))
        return self.iota[W]

    def select(self, W, part="all"):
        """GlmDSA.decode_forward selection at scored width W (pools): ids uint32 TILE [1,1,S,512]."""
        sc = ttnn.experimental.dsa_decode_scores(self.qi, self.wi, self.pooled, self.pt, self.limit_u32, W // 32)
        if part == "scores":
            return sc
        scores = ttnn.typecast(sc, ttnn.bfloat16)
        ttnn.deallocate(sc)
        ok = ttnn.typecast(ttnn.lt(self.iota_w(W), self.limit), ttnn.bfloat16)
        masked = ttnn.where(ok, scores, float("-inf"))
        ttnn.deallocate(ok)
        ttnn.deallocate(scores)
        _, ids = ttnn.topk(masked, K, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(masked)
        if ids.dtype != ttnn.uint32:
            ids32 = ttnn.typecast(ids, ttnn.uint32)
            ttnn.deallocate(ids)
            ids = ids32
        return ids


def host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def timed_trace(mesh, fn, reps):
    out = fn()  # compile
    ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)
    tb0 = int(ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE).total_bytes_allocated_per_bank)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    tb1 = int(ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE).total_bytes_allocated_per_bank)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) * 1e3 / reps
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)
    return dt, tb1 - tb0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=280576)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    rec = {"T": a.T, "P_alloc": p_alloc(a.T)}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=256 << 20)
    try:
        Pa = p_alloc(a.T)
        widths = [p_alloc(t) for t in (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144) if t < a.T] + [Pa]
        rec["widths"] = widths

        # ---- 0a. persistent-id buffer ops: uint16 -> uint32 typecast, uint32 TILE copy
        try:
            rep = ttnn.ReplicateTensorToMesh(mesh)
            src = ttnn.from_torch(
                torch.arange(512, dtype=torch.int32).reshape(1, 1, 1, 512).repeat(1, 1, 8, 1),
                dtype=ttnn.uint16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
            )
            buf = ttnn.from_torch(
                torch.zeros(1, 1, 8, 512, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
            )
            t32 = ttnn.typecast(src, ttnn.uint32)
            ttnn.copy(t32, buf)
            got = host(buf).reshape(8, 512).long()
            rec["u32_copy_ok"] = bool(torch.equal(got, torch.arange(512).repeat(8, 1)))
        except Exception as e:  # noqa: BLE001
            rec["u32_copy_ok"] = f"error: {e}"[:300]
        print("U32COPY", rec["u32_copy_ok"], flush=True)

        # ---- 0b. prefill indexer: kv_len = W (chunk_start W - 1) vs the full P_alloc scoring, columns < W
        try:
            rig = Rig(mesh, 1, a.T, seed=7)
            rb = 256
            g = torch.Generator().manual_seed(3)
            qi = rig.dev(torch.randn(1, HI, rb, D, generator=g) * 0.1, ttnn.bfloat16)
            wi = rig.dev(torch.randn(1, 1, rb, HI, generator=g) * 0.2, ttnn.bfloat16)
            view = ttnn.reshape(rig.pooled, (rig.pooled.shape[0] * 32 // Pa, 1, Pa, D))
            kw = dict(cache_batch_idx=0, page_table=rig.pt)
            full = host(ttnn.experimental.indexer_score_dsa(qi, view, wi, chunk_start_idx=a.T // 4, **kw))
            full = full.reshape(rb, -1)
            r0 = rec.setdefault("indexer_kv_len", {})
            for W in [w for w in widths if w < Pa][:4] + [widths[-2]]:
                f = lambda: ttnn.experimental.indexer_score_dsa(qi, view, wi, chunk_start_idx=W - 1, kv_len=W, **kw)
                part = host(f()).reshape(rb, -1)[:, :W]
                eq = bool(torch.equal(part, full[:, :W]))
                t0 = time.perf_counter()
                for _ in range(5):
                    ttnn.deallocate(f())
                ttnn.synchronize_device(mesh)
                dt = (time.perf_counter() - t0) * 1e3 / 5
                r0[W] = {"equal": eq, "ms": dt}
                print(f"IDXKV W {W}: equal {eq} {dt:.2f} ms", flush=True)
            t0 = time.perf_counter()
            for _ in range(5):
                ttnn.deallocate(ttnn.experimental.indexer_score_dsa(qi, view, wi, chunk_start_idx=a.T // 4, **kw))
            ttnn.synchronize_device(mesh)
            r0["full_ms"] = (time.perf_counter() - t0) * 1e3 / 5
            print(f"IDXKV full {r0['full_ms']:.2f} ms", flush=True)
            ttnn.deallocate(rig.pooled)
        except Exception as e:  # noqa: BLE001
            rec["indexer_kv_len_error"] = f"{e}"[:400]
            print("IDXKV error", e, flush=True)

        # ---- 3. small-trace replay overhead
        x = ttnn.from_torch(
            torch.ones(1, 1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        y = ttnn.add(x, 1.0)
        ttnn.deallocate(y)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        y = ttnn.add(x, 1.0)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        n = 230
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh)
        rec["tiny_trace_us"] = {"enqueue": (t1 - t0) * 1e6 / n, "total": (time.perf_counter() - t0) * 1e6 / n}
        ttnn.release_trace(mesh, tid)
        print("TINY", rec["tiny_trace_us"], flush=True)

        for S in (1, 8):
            for dup in (0, 37):
                rig = Rig(mesh, S, a.T, seed=S + dup, dup_period=dup)
                key = f"S{S}_dup{dup}"
                r = rec.setdefault(key, {"invariance": [], "time_ms": {}})
                # ---- 2. width invariance
                pos_sets = (
                    [[2100], [16000], [60000], [250000]]
                    if S == 1
                    else [
                        [2100, 16000, 3000, -1, 9000, 2051, 12000, 15000],
                        [60000, 2200, 250000, 40000, -1, 100000, 7000, 130000],
                    ]
                )
                for ps in pos_sets:
                    ps = (ps * S)[:S]
                    lim = rig.set_positions(ps)
                    ref = host(rig.select(Pa)).reshape(-1, K)[:S]
                    for W in widths[:-1]:
                        rows = [s for s in range(S) if 512 < lim[s] <= W]
                        if not rows:
                            continue
                        got = host(rig.select(W)).reshape(-1, K)[:S]
                        same_order = all(torch.equal(got[s], ref[s]) for s in rows)
                        same_set = all(torch.equal(got[s].sort().values, ref[s].sort().values) for s in rows)
                        r["invariance"].append(
                            {"positions": ps, "W": W, "rows": rows, "same_order": same_order, "same_set": same_set}
                        )
                        print(f"INV {key} pos {ps} W {W}: order {same_order} set {same_set}", flush=True)
                if dup:
                    continue
                # ---- 1. time per replay, positions 300 (the stage-7 decode point) and 60000
                for p0 in (300, 60000):
                    rig.set_positions([p0] * S)
                    for W in widths:
                        if (p0 + 1) // 4 > W:
                            continue
                        t_all, tb = timed_trace(mesh, lambda: rig.select(W), a.reps)
                        t_sc, _ = timed_trace(mesh, lambda: rig.select(W, "scores"), a.reps)
                        r["time_ms"][f"p{p0}_W{W}"] = {"select_ms": t_all, "scores_ms": t_sc, "trace_bytes": tb}
                        print(
                            f"TIME {key} p {p0} W {W}: select {t_all:.4f} ms scores {t_sc:.4f} ms trace {tb} B",
                            flush=True,
                        )
                ttnn.deallocate(rig.pooled)
    finally:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
