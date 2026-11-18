# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact device time per op of one verify pass (R rows, eager) through every layer + the final norm / LM head: each
top-level ttnn op is followed by a device sync and a profiler read, and the programs that read returns (die 0) are
the op's. Ops are keyed by their call site in the model code. Prints the total per call site (all layers), one MoE
layer's op list, and the device total per layer.

usage (hybrid env as tools/eval_hybrid.sh, TOOL=op_prof.py,
       EXTRA="... -e TT_METAL_DEVICE_PROFILER=1 -e TT_METAL_PROFILER_MID_RUN_DUMP=1 -e TT_METAL_PROFILER_CPP_POST_PROCESS=1"):
       op_prof.py [R=3] [layer to list=20]
"""

import collections
import os
import sys
import traceback

import torch

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    import hybrid

    hybrid.install()

import ttnn  # noqa: E402
import ttnn._ttnn.profiler as P  # noqa: E402
from ttnn import decorators as D  # noqa: E402

from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402

REC = {"on": False, "mesh": None, "rows": [], "layer": -1, "keys": None}


def _site():
    for fr in reversed(traceback.extract_stack()[:-3]):
        f = fr.filename
        if "/models/" in f and "/ttnn/" not in f and "op_prof.py" not in f:
            return f"{f.split('/models/')[-1]}:{fr.lineno} {fr.name}"
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
        if REC["keys"] is None:
            REC["keys"] = list(res.keys())
            print("analyses:", REC["keys"], flush=True)
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
    REC["rows"].append((REC["layer"], self.python_fully_qualified_name, site, n, us))
    return out


D.FastOperation.__call__ = _call


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    show = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    REC["mesh"] = mesh
    try:
        args, model = create_model(mesh, max_seq_len=8192)
        hyb = sys.modules["hybrid.patch"].HYB if "hybrid.patch" in sys.modules else None
        if hyb is not None and hyb.enabled:  # as tt/inkling_serve.py warm(): library + pinned fetch copies
            for li in list(hyb.lib):
                hyb.lib_layer(li)
            if hyb.fetch.dir:
                hyb.fetch.finalize(hyb.loader, hyb.stream)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = model.layers[0].attention.max_blocks
        pt = ttnn.from_torch(torch.arange(nblk, dtype=torch.int32).reshape(1, -1), dtype=ttnn.int32,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        ids = [100 + i for i in range(R)]
        tok = ttnn.from_torch(torch.tensor(ids, dtype=torch.int32).reshape(1, R), dtype=ttnn.uint32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        pos = ttnn.from_torch(torch.arange(256, 256 + R, dtype=torch.int32), dtype=ttnn.int32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        idx = torch.full((R, R), -1, dtype=torch.int32)
        idx[torch.arange(R), torch.arange(R)] = torch.arange(256, 256 + R, dtype=torch.int32)
        kvi = ttnn.from_torch(idx, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        x0 = ttnn.reshape(model.embd(tok), (1, 1, R, -1))

        def run():
            x = ttnn.clone(x0)
            for i, layer in enumerate(model.layers):
                REC["layer"] = i
                x = layer.forward_rows(x, pos, pt, commit_all=False, kv_idx=kvi)
            REC["layer"] = 99  # the verify head as tt/inkling_serve.py _body: final norm, LM head, row-major logits
            h = model._final_norm_decode(x)
            lg = model._lm_head(h)
            if lg.is_sharded():
                lg = ttnn.sharded_to_interleaved(lg, ttnn.DRAM_MEMORY_CONFIG)
            lg = ttnn.untilize_with_unpadding(lg, [int(d) - 1 for d in lg.shape])
            ttnn.synchronize_device(mesh)

        model.reset_sequence_state()
        run()
        run()
        _read()
        REC["on"] = True
        run()
        REC["on"] = False
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        rows = REC["rows"]
        per_layer = collections.defaultdict(float)
        site_t = collections.defaultdict(lambda: [0.0, 0, 0])
        for L, name, site, n, us in rows:
            per_layer[L] += us
            s = site_t[(site, name)]
            s[0] += us
            s[1] += 1
            s[2] += n
        tot = sum(per_layer.values())
        print(f"== {len(rows)} top-level ops, {sum(r[3] for r in rows)} programs, device total {tot / 1e3:.2f} ms (die 0)")
        print("== device us per layer:", " ".join(f"{L}:{v:.0f}" for L, v in sorted(per_layer.items())))
        print("== by call site (all layers): us total | calls | programs")
        for (site, name), (us, c, n) in sorted(site_t.items(), key=lambda kv: -kv[1][0])[:70]:
            print(f"   {us:9.1f}  {c:4d}  {n:4d}  {name.replace('ttnn.', '')}  @ {site}")
        print(f"== layer {show} ops:")
        for L, name, site, n, us in rows:
            if L == show:
                print(f"   {us:7.1f} us  {n} prog  {name.replace('ttnn.', '')}  @ {site}")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
