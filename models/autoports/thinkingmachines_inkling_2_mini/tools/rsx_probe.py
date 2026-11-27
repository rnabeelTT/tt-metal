# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Eager packed prefill chunks with a prefill switch off (reference) vs on: last-row logits, the drafter taps (every row),
every lane-0 conv state and snapshot rows compared (bitwise, else max |d| and top-1 agreement), then the chunk wall
time per size. SWITCH names a tt/model.py module flag (default PF_RSX).
env: MESH (4), CHUNK (2048), TS (1024,2048), SWITCH (PF_RSX), REPS (5)"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import model as M
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model


def main():
    N = int(os.environ.get("MESH", "4"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=64 << 20)
    chunk = int(os.environ.get("CHUNK", "2048"))
    Ts = [int(v) for v in os.environ.get("TS", "1024,2048").split(",")]
    sw = os.environ.get("SWITCH", "PF_RSX")
    if ":" in sw:  # (module:FLAG, a tt/ module)
        import importlib

        mod, sw = sw.split(":")
        globals()["M"] = importlib.import_module(f"models.autoports.thinkingmachines_inkling_2_mini.tt.{mod}")
    m = Inkling2Model(mesh, max_seq_len=8192, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=chunk,
                      slots=2)
    m.pft = None
    lane0 = [cv for st in m.apps if st is not None for cv in st.lanes[0]]
    g = torch.Generator().manual_seed(0)
    Tb = Ts[-1]
    cases = [[(Tb, 0, Tb, 0, None)], [(Tb, 0, Tb - 300, 1, None)], [(Ts[0], 0, Ts[0], 0, None), (Ts[0], Ts[0], min(777, Ts[0] - 29), 0, None)],
             [(Tb, 0, Tb - 40, 1, Tb // 2 + 64)], [(Tb, 0, 70, 0, None)]]
    ids = {}

    def host(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0] if N > 1 else t).clone()

    def run(ci):
        m.reset(0)
        m.reset(1)
        res = []
        for j, (T, p0, valid, slot, snap) in enumerate(cases[ci]):
            if (ci, j) not in ids:
                ids[(ci, j)] = torch.randint(1000, 100000, (T,), generator=g)
            x = ids[(ci, j)].clone()
            x[valid:] = 0
            if snap is not None:
                for cv in lane0:
                    cv.snap_row, cv.snap_out = snap, None
            taps = []
            lg = m.forward_packed(x, [(0, T, p0, valid, slot, 0)], taps=taps)
            out = {"lg": m.logits_host(lg, 1), "taps": host(taps[0])[:, :, :valid].float(),
                   "st": [host(cv.st) for cv in lane0]}
            if snap is not None:
                out["snap"] = [host(cv.snap_out) for cv in lane0]
                for cv in lane0:
                    cv.snap_row, cv.snap_out = None, None
            res.append(out)
        return res

    setattr(M, sw, False)
    ref = [run(i) for i in range(len(cases))]
    setattr(M, sw, True)
    got = [run(i) for i in range(len(cases))]
    bit = True
    for i, (a_, b_) in enumerate(zip(ref, got)):
        for j, (a, b) in enumerate(zip(a_, b_)):
            same = {k: (torch.equal(a[k], b[k]) if not isinstance(a[k], list)
                        else all(torch.equal(u, v) for u, v in zip(a[k], b[k]))) for k in a}
            bit &= all(same.values())
            d_lg = (a["lg"] - b["lg"]).abs().max().item()
            d_tp = (a["taps"] - b["taps"]).abs().max().item()
            rel = ((a["taps"] - b["taps"]).norm() / a["taps"].norm()).item()
            top1 = (a["lg"].argmax(-1) == b["lg"].argmax(-1)).float().mean().item()
            nf = lambda t: int((~torch.isfinite(t)).sum())  # noqa: E731
            print(f"case {i}.{j} {cases[i][j]}: {same}  lg max|d| {d_lg:.4g} top1 {top1:.3f}  taps max|d| {d_tp:.4g} "
                  f"rel {rel:.3g}  (off / on: taps max {a['taps'].abs().max():.3g} / {b['taps'].abs().max():.3g}, "
                  f"non-finite {nf(a['taps'])} / {nf(b['taps'])})", flush=True)
    print(f"== {sw}: {'BITWISE' if bit else 'NOT bitwise'}", flush=True)
    reps = int(os.environ.get("REPS", "5"))
    for T in Ts:
        x = torch.randint(1000, 100000, (T,), generator=g)
        for on in (False, True):
            setattr(M, sw, on)
            ws = []
            for _ in range(reps + 1):
                m.reset(0)
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                lg = m.forward_packed(x, [(0, T, 0, T, 0, 0)])
                ttnn.synchronize_device(mesh)
                ws.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(lg)
            ws = sorted(ws[1:])
            print(f"T={T} {sw}={int(on)}: chunk wall median {ws[len(ws) // 2]:.1f} ms (min {ws[0]:.1f})", flush=True)
    print("== rsx_probe done", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()
