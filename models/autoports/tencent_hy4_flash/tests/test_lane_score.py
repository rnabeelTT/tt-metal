# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/lane_sel LaneSelOp.score vs the batched decode's reference path (indexer_score_dsa over every lane's keys, then
LaneSelOp.compact): bit-identical compact rows and top-512 ids, plus traced device time of both. Synthetic inputs,
8 lanes x gcap 6144 (the default serving profile's score width 49152)."""

import time

import numpy as np
import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.lane_sel import LaneSelOp

L, GCAP = 8, 6144


def cases():
    """(name, [(lane, vis, selects)] per row in row order)"""
    rng = np.random.default_rng(0)
    yield "tiny", [(0, 512, 1)]
    yield "8x1 short", [(l, 700 + 37 * l, 1) for l in range(8)]
    yield "8x1 long", [(l, 5100 - 3 * l, 1) for l in range(8)]
    yield "8x1 mixed", [(l, int(v), 1) for l, v in enumerate(rng.integers(512, 5200, 8))]
    yield "dense+pad", [(0, 600, 1), (1, 300, 0), (2, 4000, 1), (3, 100, 0)]
    yield "multi-row", [(0, 1000, 1), (0, 1001, 1), (0, 1001, 1), (2, 2500, 1), (2, 2501, 1), (5, 800, 1)]
    yield "2 lanes", [(3, 5000, 1), (6, 513, 1)]


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 << 20)
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        up = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh, mesh_mapper=rep)  # noqa: E731
        back = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
        torch.manual_seed(0)
        op = LaneSelOp.get(mesh)
        W = L * GCAP
        gk_h = torch.randn(1, 1, W + 64, 128).to(torch.bfloat16)
        gk = up(gk_h, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        all_ok = True
        for name, rows in cases():
            q_h = torch.randn(1, 16, 32, 128).to(torch.bfloat16)
            w_h = (0.05 * torch.randn(1, 1, 32, 16)).to(torch.bfloat16)
            meta = np.zeros((32, 16), dtype=np.uint32)
            for r, (l, v, s) in enumerate(rows):
                meta[r, :4] = [l * GCAP, v, 1000 * r, s]
            vmax = max(v for _, v, s in rows if s)
            VT = -(-vmax // 32)
            q = up(q_h, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            w = up(w_h, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            lm = up(torch.from_numpy(meta.view(np.int32)).reshape(1, 1, 32, 16), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            vm = up(torch.tensor([vmax], dtype=torch.int32).reshape(1, 1, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

            def ref():
                sc = ttnn.experimental.indexer_score_dsa(q, gk, w, chunk_start_idx=W, kv_len=W + 32)
                c = op.compact(sc, lm, GCAP)
                ttnn.deallocate(sc)
                return c

            def new():
                return op.score(q, gk, w, lm, GCAP)

            def poisoned_new():  # (a freed output buffer holds the reference's rows: poison it so unwritten = NaN)
                pz = up(torch.full((1, 1, 32, GCAP), float("nan")).to(torch.bfloat16), ttnn.bfloat16,
                        ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(pz)
                return new()

            def traced(fn):
                o = fn()
                ttnn.deallocate(o)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                o = fn()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                n = 50
                t0 = time.perf_counter()
                for _ in range(n):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                ttnn.release_trace(mesh, tid)
                return (time.perf_counter() - t0) / n * 1e6

            a = ref()
            ca = back(a).reshape(32, -1)[:, : VT * 32]
            ia = back(ttnn.experimental.topk_large_indices(a, k=512, valid_length_tensor=vm)).reshape(32, -1)
            t_ref = traced(ref)
            sel = [r for r, (_, _, s) in enumerate(rows) if s]
            b = poisoned_new()
            cb = back(b).reshape(32, -1)[:, : VT * 32]
            same = torch.equal(ca.view(torch.int16), cb.view(torch.int16))
            ib = back(ttnn.experimental.topk_large_indices(b, k=512, valid_length_tensor=vm)).reshape(32, -1)
            ids_same = all(torch.equal(ia[r], ib[r]) for r in sel)
            err = 0.0
            for r, (l, v, s_) in enumerate(rows):
                if not s_:
                    continue
                k = gk_h[0, 0, l * GCAP: l * GCAP + v].float()
                t = (torch.relu(torch.einsum("hd,gd->hg", q_h[0, :, r].float(), k))
                     * w_h[0, 0, r, :, None].float()).sum(0)
                err = max(err, float((cb[r, :v].float() - t).abs().max() / (t.abs().max() + 1e-6)))
            all_ok &= same and ids_same
            if not same:
                bad = (ca.view(torch.int16) != cb.view(torch.int16)).nonzero()
                print(f"LSCORE {name}: {bad.shape[0]} mismatches, first {bad[:4].tolist()}")
            ttnn.deallocate(b)
            t_new = traced(new)
            ttnn.deallocate(a)
            print(f"LSCORE {name}: VT {VT} scores exact {same} ids exact {ids_same} rel err vs torch {err:.4f}; traced "
                  f"ref (score+compact) {t_ref:.1f} us -> lane score {t_new:.1f} us", flush=True)
        print(f"LSCORE all exact {all_ok}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
