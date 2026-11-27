# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host enqueue time of a prefill chunk per top-level ttnn call site (no device syncs: what the host spends issuing
the chunk; tools/bench_prefill.py's 'enqueue'). Prints the total, then sites by host ms and by op name.
usage: MESH=4 CHUNK=2048 T=256 tools/run_dev.sh python .../tools/host_prof.py   env: PACKED (1: forward_packed)"""

import collections
import os
import time
import traceback

import torch

import ttnn
from ttnn import decorators as D

from models.autoports.thinkingmachines_inkling_2_mini.tt.model import Inkling2Model

PORT = "thinkingmachines_inkling_2_mini/"
REC = {"on": False, "rows": []}
_orig = D.FastOperation.__call__


def _site():
    for fr in reversed(traceback.extract_stack()[:-3]):
        if PORT in fr.filename and "host_prof.py" not in fr.filename:
            return f"{fr.filename.split(PORT)[-1]}:{fr.lineno} {fr.name}"
    return "?"


def _call(self, *a, **k):
    if not REC["on"] or D.OPERATION_CALL_STACK:
        return _orig(self, *a, **k)
    site = _site()  # (outside the timed region)
    t0 = time.perf_counter()
    out = _orig(self, *a, **k)
    REC["rows"].append((self.python_fully_qualified_name, site, time.perf_counter() - t0))
    return out


D.FastOperation.__call__ = _call


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    chunk, T = int(os.environ.get("CHUNK", "1024")), int(os.environ.get("T", "256"))
    m = Inkling2Model(mesh, max_seq_len=8192, expert_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat16, prefill_chunk=chunk,
                      slots=int(os.environ.get("B", "1")))
    ids = torch.arange(1000, 1000 + T)
    packed = os.environ.get("PACKED", "1") == "1"

    def run():
        m.reset()
        if packed:
            m.forward_packed(ids, [(0, T, 0, T, 0, 0)])
        else:
            m.forward(ids, 0, valid=T, slot=0, last_only=True)

    for _ in range(2):
        run()
    ttnn.synchronize_device(mesh)
    m.reset()
    ttnn.synchronize_device(mesh)
    REC["on"] = True
    t0 = time.perf_counter()
    if packed:
        m.forward_packed(ids, [(0, T, 0, T, 0, 0)])
    else:
        m.forward(ids, 0, valid=T, slot=0, last_only=True)
    wall_enq = time.perf_counter() - t0
    REC["on"] = False
    ttnn.synchronize_device(mesh)
    rows = REC["rows"]
    tot = sum(r[2] for r in rows)
    print(f"== T={T}: enqueue {wall_enq * 1e3:.1f} ms, {len(rows)} top-level ops, {tot * 1e3:.1f} ms inside them "
          f"({(wall_enq - tot) * 1e3:.1f} ms Python between)", flush=True)
    by_name, by_site = collections.defaultdict(lambda: [0.0, 0]), collections.defaultdict(lambda: [0.0, 0])
    for name, site, dt in rows:
        by_name[name][0] += dt
        by_name[name][1] += 1
        by_site[(site, name)][0] += dt
        by_site[(site, name)][1] += 1
    print("-- by op: host ms, calls, us/call")
    for k, (t, c) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:30]:
        print(f"{t * 1e3:8.2f} {c:6d} {t / c * 1e6:7.1f}  {k}")
    print("-- by site: host ms, calls, us/call")
    for (s, k), (t, c) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[: int(os.environ.get("TOP", "50"))]:
        print(f"{t * 1e3:8.2f} {c:6d} {t / c * 1e6:7.1f}  {k:34s} {s}")
    print("== done", flush=True)


if __name__ == "__main__":
    main()
