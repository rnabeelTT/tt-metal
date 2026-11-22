# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact device time per op of one batched decode step (eager), keyed by the op's call site in this port's code (the
Inkling-Small op_prof mechanism: each top-level ttnn op is followed by a device sync and a profiler read; die 0).
Prints the device total, the top call sites, and the time per component (attention / convs / moe / mix / gates / ...).

usage: DEV_ENV="-e TT_METAL_DEVICE_PROFILER=1 -e TT_METAL_PROFILER_MID_RUN_DUMP=1
       -e TT_METAL_PROFILER_CPP_POST_PROCESS=1 -e B=4" tools/run_dev.sh python .../tools/op_prof.py
"""

import collections
import os
import traceback

import torch

import ttnn
import ttnn._ttnn.profiler as P
from ttnn import decorators as D

from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedDecoder
from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model

REC = {"on": False, "mesh": None, "rows": []}
PORT = "thinkingmachines_inkling_2_mini/"


def _site():
    for fr in reversed(traceback.extract_stack()[:-3]):
        if PORT in fr.filename and "op_prof.py" not in fr.filename:
            return f"{fr.filename.split(PORT)[-1]}:{fr.lineno} {fr.name}"
    return "?"


def _read():
    ttnn.synchronize_device(REC["mesh"])
    ttnn.ReadDeviceProfiler(REC["mesh"])
    data = P.get_latest_programs_perf_data()
    if not data:
        return 0, 0.0
    chip = sorted(data)[0]
    n, us = 0, 0.0
    for prog in data[chip]:
        res = prog.program_analyses_results
        k = "DEVICE FW DURATION [ns]" if "DEVICE FW DURATION [ns]" in res else next(iter(res))
        us += res[k].duration / 1e3
        n += 1
    return n, us


_orig = D.FastOperation.__call__


def _call(self, *a, **k):
    if not REC["on"] or D.OPERATION_CALL_STACK:
        return _orig(self, *a, **k)
    site = _site()
    out = _orig(self, *a, **k)
    n, us = _read()
    REC["rows"].append((self.python_fully_qualified_name, site, n, us))
    return out


D.FastOperation.__call__ = _call
_orig_generic = ttnn.generic_op


def main():
    B = int(os.environ.get("B", "4"))
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    REC["mesh"] = mesh
    dt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
    m = Inkling2Model(mesh, max_seq_len=1024, expert_dtype=dt[os.environ.get("EXPERT_DT", "bfp8")],
                      kv_dtype=dt[os.environ.get("KV_DT", "bf16")], prefill_chunk=256, slots=B)
    T = int(os.environ.get("PREFILL", "0"))
    if T:  # one prefill chunk of T rows (slot 0) instead of a decode step
        ids = torch.arange(1000, 1000 + T)
        for _ in range(2):
            m.reset()
            m.forward(ids, 0, valid=T, slot=0, last_only=True)
        m.reset()
        _read()
        REC["on"] = True
        m.forward(ids, 0, valid=T, slot=0, last_only=True)
    else:
        dec = BatchedDecoder(m, B)
        dec.set_inputs(torch.arange(100, 100 + B), torch.arange(300, 300 + B))
        dec.step()
        dec.step()
        _read()
        REC["on"] = True
        dec.step()
    REC["on"] = False
    rows = REC["rows"]
    tot = sum(r[3] for r in rows)
    print(f"== {'PREFILL T=' + str(T) if T else 'B=' + str(B)}: {len(rows)} top-level ops, {sum(r[2] for r in rows)} programs, device total {tot / 1e3:.2f} ms")
    site_t = collections.defaultdict(lambda: [0.0, 0])
    comp = collections.defaultdict(lambda: [0.0, 0])
    for name, site, n, us in rows:
        s = site_t[(site, name)]
        s[0] += us
        s[1] += 1
        fn = site.split(" ")[-1] if site != "?" else "?"
        f = site.split(":")[0]
        key = f"{f}::{fn}"
        comp[key][0] += us
        comp[key][1] += 1
    print("-- by function (file::function): ms, ops")
    for k, (us, c) in sorted(comp.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"{us / 1e3:9.2f} ms {c:6d}  {k}")
    print("-- top call sites: ms, calls, op")
    for (site, name), (us, c) in sorted(site_t.items(), key=lambda kv: -kv[1][0])[: int(os.environ.get("TOP", "40"))]:
        print(f"{us / 1e3:9.2f} ms {c:5d}  {name:40s} {site}")
    ttnn.close_mesh_device(mesh)
    print("== done")


if __name__ == "__main__":
    main()
