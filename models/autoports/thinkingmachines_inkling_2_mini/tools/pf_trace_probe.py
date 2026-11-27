# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced one-segment prefill chunks (tt/model.py PrefillTrace) vs the eager chunk: logits (and taps, and every lane-0
conv state) bitwise for several (T, p0, slot, valid) incl. a continuation chunk and a prefix snapshot row; then chunk
wall time eager vs replayed per size.
env: MESH (4), CHUNK (2048), TS (256,512,1024), TRACE_MB (256), TAPS (1), REPS (5)"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model, PrefillTrace


def main():
    N = int(os.environ.get("MESH", "4"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=int(os.environ.get("TRACE_MB", "256")) << 20)
    chunk = int(os.environ.get("CHUNK", "2048"))
    Ts = [int(v) for v in os.environ.get("TS", "256,512,1024").split(",")]
    taps_on = os.environ.get("TAPS", "1") == "1"
    m = Inkling2Model(mesh, max_seq_len=8192, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=chunk,
                      slots=2)
    pft = PrefillTrace(m, Ts, taps=taps_on)
    pft.warm()
    g = torch.Generator().manual_seed(0)
    lane0 = pft.lane0
    T0, T1 = Ts[0], Ts[min(1, len(Ts) - 1)]
    # cases: a list of chunks (T, p0, valid, slot, snap) run in order from a reset lane 0
    cases = [[(T0, 0, T0, 0, None)], [(T0, 0, 130, 1, None)], [(T1, 0, T1, 0, None), (T0, T1, 77, 0, None)],
             [(Ts[-1], 0, Ts[-1] - 24, 1, 64)], [(T0, 0, 3, 0, None)]]
    ids = {}

    def host(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0] if N > 1 else t).clone()

    def run(case_i, traced):
        m.pft = pft if traced else None
        m.reset(0)
        res = []
        for j, (T, p0, valid, slot, snap) in enumerate(cases[case_i]):
            key = (case_i, j)
            if key not in ids:
                ids[key] = torch.randint(1000, 100000, (T,), generator=g)
            x = ids[key].clone()
            x[valid:] = 0
            if snap is not None:
                for cv in lane0:
                    cv.snap_row, cv.snap_out = snap, None
            taps = [] if taps_on else None
            lg = m.forward_packed(x, [(0, T, p0, valid, slot, 0)], taps=taps)
            out = {"lg": m.logits_host(lg, 1)}
            if taps_on:
                out["taps"] = host(taps[0])[:, :, :T]
            out["st"] = [host(cv.st) for cv in lane0]
            if snap is not None:
                out["snap"] = [host(cv.snap_out) for cv in lane0]
                for cv in lane0:
                    cv.snap_row, cv.snap_out = None, None
            res.append(out)
        m.pft = None
        return res

    ref = [run(i, False) for i in range(len(cases))]
    print("eager references done", flush=True)
    t0 = time.perf_counter()
    pft.capture()
    print(f"captured {pft.Ts} in {time.perf_counter() - t0:.1f}s", flush=True)
    try:
        v = ttnn.device.GetMemoryView(mesh, ttnn.BufferType.TRACE)
        print(f"trace region: {v.total_bytes_allocated_per_bank / 2**20:.1f} MB allocated per bank of "
              f"{v.total_bytes_per_bank / 2**20:.1f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"(trace memory view: {e})", flush=True)
    ok = True
    for i in range(len(cases)):
        got = run(i, True)
        for j, (r, o) in enumerate(zip(ref[i], got)):
            eq = {k: (torch.equal(r[k], o[k]) if not isinstance(r[k], list)
                      else all(torch.equal(a, b) for a, b in zip(r[k], o[k]))) for k in r}
            md = float((r["lg"] - o["lg"]).abs().max())
            print(f"case {i} chunk {j} {cases[i][j]}: " + " ".join(f"{k} {'==' if e else '!='}" for k, e in eq.items())
                  + f" (logits max|d| {md:.3g})", flush=True)
            ok = ok and all(eq.values())
    reps = int(os.environ.get("REPS", "5"))
    for T in pft.Ts:
        x = torch.randint(1000, 100000, (T,), generator=g)
        seg = [(0, T, 0, T, 0, 0)]
        tm = {}
        for traced in (False, True):
            m.pft = pft if traced else None
            w = []
            for _ in range(reps):
                m.reset(0)
                ttnn.synchronize_device(mesh)
                t = time.perf_counter()
                taps = [] if taps_on else None
                lg = m.forward_packed(x, seg, taps=taps)
                ttnn.synchronize_device(mesh)
                w.append(time.perf_counter() - t)
                del lg, taps
            tm[traced] = min(w)
        print(f"T={T:5d}: eager {tm[False] * 1e3:7.1f} ms  traced {tm[True] * 1e3:7.1f} ms", flush=True)
    m.pft = None
    print("PASS" if ok else "FAIL", flush=True)
    pft.release()
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()
