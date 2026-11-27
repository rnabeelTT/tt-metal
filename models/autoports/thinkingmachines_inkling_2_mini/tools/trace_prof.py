# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program timeline of one TRACED batched decode step: every program's device span (first core start -> last core
end, profiler "DEVICE FW DURATION") and the idle gap before it (previous program's end -> this one's start: dispatch +
launch latency in a trace replay), per die, keyed by the call site of the eager op that made it (one eager step with
a per-op sync gives the site and program count of each op; the trace replays the same program sequence).

usage: DEV_ENV="-e TT_METAL_DEVICE_PROFILER=1 -e TT_METAL_PROFILER_MID_RUN_DUMP=1 -e TT_METAL_PROFILER_CPP_POST_PROCESS=1
       -e MESH=4 -e B=1" tools/run_dev.sh python .../tools/trace_prof.py
env: OUT (csv path, default /cache/trace_prof_B{B}_N{N}.csv), DIE (0), REPS (replays profiled, 1)
"""

import collections
import os
import traceback

import torch

import ttnn
import ttnn._ttnn.profiler as P
from ttnn import decorators as D

from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder, TracedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model

REC = {"on": False, "mesh": None, "rows": []}
PORT = "thinkingmachines_inkling_2_mini/"


def _site():
    for fr in reversed(traceback.extract_stack()[:-3]):
        if PORT in fr.filename and "trace_prof.py" not in fr.filename:
            return f"{fr.filename.split(PORT)[-1]}:{fr.lineno} {fr.name}"
    return "?"


def _progs():
    ttnn.synchronize_device(REC["mesh"])
    ttnn.ReadDeviceProfiler(REC["mesh"])
    return P.get_latest_programs_perf_data() or {}


_orig = D.FastOperation.__call__


def _call(self, *a, **k):
    if not REC["on"] or D.OPERATION_CALL_STACK:
        return _orig(self, *a, **k)
    site = _site()
    out = _orig(self, *a, **k)
    data = _progs()
    n = len(data[sorted(data)[0]]) if data else 0
    REC["rows"].append((self.python_fully_qualified_name, site, n))
    return out


D.FastOperation.__call__ = _call


def main():
    B = int(os.environ.get("B", "1"))
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=int(os.environ.get("TRACE_MB", "300")) << 20)
    REC["mesh"] = mesh
    m = Inkling2Model(mesh, max_seq_len=1024, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=256,
                      slots=B)
    dp = int(os.environ.get("DPOS", "300"))
    dec = BatchedDecoder(m, B)
    dec.set_inputs(torch.arange(100, 100 + B), torch.arange(dp, dp + B))
    dec.step()
    dec.step()
    _progs()
    REC["on"] = True
    dec.step()
    REC["on"] = False
    seq = []  # one (op, site) per program, in order
    for name, site, n in REC["rows"]:
        seq += [(name, site)] * n
    tr = TracedDecoder(dec)
    tr.capture()
    for _ in range(5):
        tr.step()
    _progs()
    reps = int(os.environ.get("REPS", "1"))
    for _ in range(reps):
        tr.step()
    data = _progs()
    die = sorted(data)[int(os.environ.get("DIE", "0"))]
    progs = []
    for p in data[die]:
        res = p.program_analyses_results
        k = "DEVICE FW DURATION [ns]" if "DEVICE FW DURATION [ns]" in res else next(iter(res))
        r = res[k]
        kk = res.get("DEVICE KERNEL DURATION [ns]")
        progs.append((r.start_timestamp, r.end_timestamp, r.duration, kk.duration if kk else r.duration, p.core_count))
    progs.sort()
    print(f"keys: {list(data[die][0].program_analyses_results.keys()) if data[die] else None}")
    print(f"== B={B} N={N}: eager {len(REC['rows'])} ops / {len(seq)} programs; traced profiled {len(progs)} programs "
          f"({reps} replays)")
    per = len(progs) // reps
    progs = progs[-per:]  # the last replay
    if per != len(seq):
        print(f"WARNING: program count mismatch traced {per} vs eager {len(seq)}: sites unaligned")
    # start / end are core cycles, duration ns: cycles -> ns at the chip clock (from the longest program)
    lp = max(progs, key=lambda q: q[2])
    mhz = (lp[1] - lp[0]) * 1000.0 / lp[2]
    progs = [(s * 1000.0 / mhz, e * 1000.0 / mhz, d, kd, cc) for s, e, d, kd, cc in progs]
    print(f"clock {mhz:.0f} MHz")
    t0, t1 = progs[0][0], progs[-1][1]
    span = t1 - t0
    busy = sum(p[2] for p in progs)
    gaps = [progs[i][0] - progs[i - 1][1] if i else 0 for i in range(len(progs))]
    print(f"replay span {span / 1e3:.1f} us, sum FW {busy / 1e3:.1f} us, sum gaps {sum(gaps) / 1e3:.1f} us, "
          f"programs {per}, mean gap {sum(gaps) / max(1, per - 1) / 1e3:.2f} us")
    out = os.environ.get("OUT", f"/cache/trace_prof_B{B}_N{N}.csv")
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    with open(out, "w") as f:
        f.write("i,site,op,start_us,fw_us,kernel_us,gap_us,cores\n")
        for i, (s, e, d, kd, cc) in enumerate(progs):
            name, site = seq[i] if per == len(seq) else ("?", "?")
            f.write(f"{i},{site},{name},{(s - t0) / 1e3:.3f},{d / 1e3:.3f},{kd / 1e3:.3f},{gaps[i] / 1e3:.3f},{cc}\n")
            a = agg[site]
            a[0] += 1
            a[1] += d / 1e3
            a[2] += kd / 1e3
            a[3] += gaps[i] / 1e3
    print(f"-- by site: programs, fw us, kernel us, gap-before us (csv {out})")
    for site, (c, d, kd, g) in sorted(agg.items(), key=lambda kv: -(kv[1][1] + kv[1][3])):
        print(f"{c:5d} {d:9.1f} {kd:9.1f} {g:8.1f}  {site}")
    tr.release()
    ttnn.close_mesh_device(mesh)
    print("== done")


if __name__ == "__main__":
    main()
