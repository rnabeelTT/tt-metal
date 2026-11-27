# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-die MoE timeline of one TRACED decode / verify pass (the trace_prof.py setup) from the rows kernels' device-profiler
marks (QWEN35_MOE_ZONES=1): per die and MoE layer, the gate|up end (G_WE, last core), the down compute end (D_WO: each
down core's output tiles packed, before the all-reduce push), and the down_push program's end (the all-reduce done).
The all-reduce ends on every die at about the same time (each die waits for the last arrival), so per layer the
arrival of die d is its last D_WO minus its program end (latest die closest to 0): the skew is their spread.
Also the expert union per MoE layer (route table read back after each layer of one eager step: the replays route
differently, the conv states advance) and, per layer of the profiled replay, ~ QWEN35_MOE_DN_CHUNK x the down's D_WA
marks, with a per-die fit of the gate|up / down time against it (us per union expert + fixed).

usage: DEV_ENV="-e TT_METAL_DEVICE_PROFILER=1 -e TT_METAL_PROFILER_MID_RUN_DUMP=1 -e TT_METAL_PROFILER_CPP_POST_PROCESS=1
       -e TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 -e QWEN35_MOE_ZONES=1 -e MESH=4 -e VERIFY=4" tools/run_dev.sh
       python .../tools/zones_verify.py      env: VERIFY (0: one row B=1), DPOS, TOK0 (100: first token id)
"""

import collections
import csv
import glob
import os
import subprocess

import torch

import ttnn
import ttnn._ttnn.profiler as P

from models.autoports.thinkingmachines_inkling_2_mini.tt import moe_fused as MF
from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder, TracedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model

UNION = {"on": False, "u": []}
_fwd = MF.FusedExperts.forward


def _forward(self, x, *a, **k):
    out = _fwd(self, x, *a, **k)
    if UNION["on"]:
        m = max(int(x.shape[-2]), 2)
        tab = self.subs[0][0].rows_ops(m)[3].table
        t = ttnn.to_torch(ttnn.get_device_tensors(tab)[0]).reshape(-1).to(torch.int64) & 0xFFFF
        rows = int(x.shape[-2])
        ids = {int(t[32 * r + j]) for r in range(rows) for j in range(self.moe.k)}
        UNION["u"].append(len(ids))
    return out


MF.FusedExperts.forward = _forward


def main():
    N = int(os.environ.get("MESH", "4"))
    vr = int(os.environ.get("VERIFY", "4"))
    dp, t0 = int(os.environ.get("DPOS", "300")), int(os.environ.get("TOK0", "100"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=int(os.environ.get("TRACE_MB", "300")) << 20)
    m = Inkling2Model(mesh, max_seq_len=1024, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=256,
                      slots=max(1, vr))
    if vr:
        dec = BatchedDecoder(m, vr, shared=BatchedDecoder(m, vr), verify=vr)
        dec.set_accept([0])
        dec.set_inputs(torch.arange(t0, t0 + vr), torch.arange(dp, dp + vr))
    else:
        dec = BatchedDecoder(m, 1)
        dec.set_inputs(torch.arange(t0, t0 + 1), torch.arange(dp, dp + 1))
    dec.step()
    UNION["on"] = True
    dec.step()
    UNION["on"] = False
    us = UNION["u"]
    print(f"union per MoE layer ({len(us)}): mean {sum(us) / max(1, len(us)):.2f} min {min(us)} max {max(us)}: {us}")
    tr = TracedDecoder(dec)
    tr.capture()
    for _ in range(5):
        tr.step()
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    P.get_latest_programs_perf_data()
    tr.step()
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    data = P.get_latest_programs_perf_data() or {}
    dies = sorted(data)
    progs = {}
    for d in dies:
        ps = []
        for p in data[d]:
            res = p.program_analyses_results
            kk = res.get("DEVICE KERNEL DURATION [ns]") or res[next(iter(res))]
            u = p.program_execution_uid
            ps.append(((u.trace_id_counter, u.runtime_id), kk.start_timestamp, kk.end_timestamp, p.core_count))
        ps.sort()
        progs[d] = [q[1:] for q in ps]
    tr.release()
    ttnn.close_mesh_device(mesh)
    # marks: (die, zone) -> per core sorted starts; the last replay's
    path = glob.glob("/opt/tt-metal/generated/profiler/.logs/profile_log_device.csv")[0]
    lines = subprocess.run(["grep", "-E", ",(D_WO|G_WE|D_WU|D_WA|G_RU),ZONE_START", path], capture_output=True, text=True).stdout
    with open(path) as f:
        f.readline()
        hdr = [c.strip() for c in f.readline().split(",")]
    ix = {n: i for i, n in enumerate(hdr)}
    mk = collections.defaultdict(lambda: collections.defaultdict(list))  # (slot, zone) -> core -> [t]
    for r in csv.reader(lines.splitlines()):
        mk[(int(r[0]), r[ix["zone name"]].strip())][(int(r[1]), int(r[2]))].append(int(r[ix["time[cycles since reset]"]]))
    slots = sorted({s for s, _ in mk})
    print(f"dies (perf data) {dies}, profiler slots {slots}")
    L = len(us)  # MoE applications per pass
    DNC = {int(v) for v in os.environ.get("DN_CORES", "29,35").split(",")}  # the down_push programs' core counts
    f = 1350.0
    per = {}
    for di, (d, s) in enumerate(zip(dies, slots)):
        def lastL(z):
            cs = mk[(s, z)]
            out = []
            for c, ts in cs.items():
                ts = sorted(ts)[-L:]
                if len(ts) == L:
                    out.append(ts)
            return out  # cores x L

        dwo, gwe, dwu, gru = lastL("D_WO"), lastL("G_WE"), lastL("D_WU"), lastL("G_RU")
        # union per layer from one down core's D_WA marks (one per chunk of QWEN35_MOE_DN_CHUNK slots)
        ch = int(os.environ.get("QWEN35_MOE_DN_CHUNK", "2"))
        c0 = next(c for c, ts in mk[(s, "D_WO")].items() if len(ts) >= L)
        wus, wos = sorted(mk[(s, "D_WU")][c0])[-L:], sorted(mk[(s, "D_WO")][c0])[-L:]
        was = sorted(mk[(s, "D_WA")][c0])
        nwa = [sum(1 for t in was if a <= t <= b) for a, b in zip(wus, wos)]
        ps = progs[d]
        # the down_push program of layer l: the first program whose kernel span holds the layer's D_WO marks
        dmax = [max(c[l] for c in dwo) for l in range(L)]
        dmin = [min(c[l] for c in dwo) for l in range(L)]
        gmax = [max(c[l] for c in gwe) for l in range(L)] if gwe else [0] * L
        umin = [min(c[l] for c in dwu) for l in range(L)]
        pe = []
        for l in range(L):
            hold = [q for q in ps if q[0] <= umin[l] and dmax[l] <= q[1] and q[2] in DNC]
            pe.append(max(q[1] for q in hold) if hold else dmax[l])
        per[d] = dict(dmax=dmax, dmin=dmin, gmax=gmax, umin=umin, pend=pe)
        grmin = [min(c[l] for c in gru) for l in range(L)] if gru else gmax
        gb = [(a - b) / f for a, b in zip(gmax, grmin)]  # gate|up: union known (G_RU first) -> last writer end
        db = [(a - b) / f for a, b in zip(dmax, umin)]  # down: D_WU first -> last D_WO
        if di == 0:
            print(f"die {d} chunks per layer (union ~ {ch} x): {nwa}")
        import statistics as st
        for nm, v in (("gate|up G_RU->G_WE", gb), ("down D_WU->D_WO", db)):
            xs = [ch * n for n in nwa]
            sl = st.covariance(xs, v) / st.variance(xs) if st.variance(xs) > 0 else 0.0
            print(f"die {d} {nm}: mean {st.mean(v):.2f} us at union ~{st.mean(xs):.1f}; fit {sl:.3f} us/expert + "
                  f"{st.mean(v) - sl * st.mean(xs):.2f}")
        cyc = lambda v: sum(v) / len(v) / f  # noqa: E731
        print(f"die {d}: per layer mean us: gate|up end -> down compute end {cyc([a - b for a, b in zip(dmax, gmax)]):.2f}"
              f" | down start (D_WU) -> compute end {cyc([a - b for a, b in zip(dmax, umin)]):.2f}"
              f" | down cores spread (last - first D_WO) {cyc([a - b for a, b in zip(dmax, dmin)]):.2f}"
              f" | compute end -> AR end {cyc([a - b for a, b in zip(pe, dmax)]):.2f}")
    # arrival per die: dmax - pend (latest die closest to 0)
    print("per layer arrival (us before the AR end; 0 = last) per die, and which die arrives last:")
    last = collections.Counter()
    spread = []
    for l in range(L):
        arr = {d: (per[d]["dmax"][l] - per[d]["pend"][l]) / f for d in dies}
        mx = max(arr.values())
        last[max(arr, key=arr.get)] += 1
        spread.append(mx - min(arr.values()))
        if l < 6 or l % 8 == 0:
            print(f"  L{l:2d} " + " ".join(f"d{d}:{arr[d]:7.2f}" for d in dies) + f"  spread {spread[-1]:.2f}")
    print(f"last-arriving die counts {dict(last)}; skew spread mean {sum(spread) / L:.2f} us, max {max(spread):.2f}, "
          f"sum {sum(spread):.1f} us/pass")
    mean_tail = {d: sum(a - b for a, b in zip(per[d]["pend"], per[d]["dmax"])) / L / f for d in dies}
    print("mean compute-end -> AR-end per die:", {d: round(v, 2) for d, v in mean_tail.items()})
    print("== done")


if __name__ == "__main__":
    main()
