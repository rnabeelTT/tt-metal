# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op DEVICE time of one Hy4 decode step, attributed to the Python call site that issued each program (the
Ornith / Qwen3.5 tools/decode_prof.py method), for the single path (1 user, Hy4Decoder) and the batched path
(N users, one row each, Hy4BatchDecoder), plus the traced step's wall time (sync'd replay) for the same config:
traced wall - device sum = dispatch / CCL-wait gaps; engine step - traced wall = host work.

A ttnn pre/post-operation hook records the device-operation ids each Python-level ttnn op issued (innermost op wins)
and the model frame that called it; after ReadDeviceProfiler the device data's runtime ids are joined to those ids.
Device time of a program = its kernel duration, max over chips.

    TT_METAL_DEVICE_PROFILER=1 python -m models.autoports.tencent_hy4_flash.tools.decode_prof --users 1,8 [--layers 0,1,2]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TTNN_CONFIG_OVERRIDES", '{"enable_fast_runtime_mode": false}')

import torch  # noqa: E402
import ttnn  # noqa: E402

OWN = collections.OrderedDict()
STACK = []
LAST = [0]
DUR = {}  # device op id -> us (accumulated over mid-step reads: the profiler buffer holds ~1/2 of a step)
FLUSH = {"mesh": None, "last": 0, "every": int(os.environ.get("DPROF_FLUSH", "300"))}


def _site():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "/ttnn/ttnn/" not in fn and "/decorators.py" not in fn and "contextlib" not in fn:
            short = fn.split("/tencent_hy4_flash/")[-1].split("/models/")[-1]
            return f"{short}:{f.f_code.co_name}:{f.f_lineno}"
        f = f.f_back
    return "?"


def _flush():
    cur = ttnn._ttnn.get_device_operation_id()
    if STACK:
        for i in range(LAST[0], cur):
            OWN.setdefault(i, STACK[-1])
    LAST[0] = cur


def pre_hook(op, args, kwargs):
    _flush()
    name = getattr(op, "python_fully_qualified_name", None) or getattr(op, "name", None) or str(op)
    extra = ""
    if os.environ.get("DPROF_SHAPES", "0") == "1" and "generic_op" in name and args and isinstance(args[0], (list, tuple)):
        try:  # the io tensors' shapes / dtypes (which linear / which expert op)
            extra = " [" + " ".join(f"{tuple(t.shape)}:{str(t.dtype).split('.')[-1]}" for t in args[0]) + "]"
        except Exception:  # noqa: BLE001
            pass
    STACK.append(f"{name.replace('ttnn.', '')} @ {_site()}{extra}")


def post_hook(op, args, kwargs, out):
    _flush()
    if STACK:
        STACK.pop()
    if not STACK and FLUSH["mesh"] is not None and LAST[0] - FLUSH["last"] >= FLUSH["every"]:
        _read(FLUSH["mesh"])


def _read(mesh):
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    for k, v in read_api().items():
        DUR[k] = max(DUR.get(k, 0.0), v)
    FLUSH["last"] = ttnn._ttnn.get_device_operation_id()


def read_api():
    dur = collections.defaultdict(float)
    data = ttnn.get_latest_programs_perf_data()
    for chip, progs in (data or {}).items():
        for p in progs:
            res = p.program_analyses_results
            r = res.get("DEVICE KERNEL DURATION [ns]") or res.get("DEVICE FW DURATION [ns]")
            if r is not None:
                rid = p.program_execution_uid.runtime_id >> 10
                dur[rid] = max(dur[rid], r.duration / 1e3)
    return dur


def profile(mesh, step, tag, top, layer_of):
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    OWN.clear()
    DUR.clear()
    LAST[0] = ttnn._ttnn.get_device_operation_id()
    first = LAST[0]
    FLUSH["mesh"], FLUSH["last"] = mesh, first
    with ttnn.decorators.register_pre_operation_hook(pre_hook), ttnn.decorators.register_post_operation_hook(post_hook):
        step()
    FLUSH["mesh"] = None
    last = ttnn._ttnn.get_device_operation_id()
    _read(mesh)
    dur = DUR
    by = collections.defaultdict(lambda: [0.0, 0])
    kind = collections.defaultdict(lambda: [0.0, 0])
    tot, miss = 0.0, 0
    for i in range(first, last):
        d = dur.get(i)
        if d is None:
            miss += 1
            continue
        lab = OWN.get(i, "<untracked>")
        by[lab][0] += d
        by[lab][1] += 1
        op = lab.split(" @ ")[0]
        k = ("CCL " if any(c in op for c in ("all_gather", "reduce_scatter", "all_reduce", "all_to_all")) else "") + op
        kind[k][0] += d
        kind[k][1] += 1
        tot += d
    n = last - first - miss
    print(f"DECODE_PROF {tag}: device sum {tot / 1e3:.2f} ms over {n} programs ({miss} without data)", flush=True)
    ccl = sum(v[0] for k, v in kind.items() if k.startswith("CCL"))
    print(f"DECODE_PROF {tag}: CCL {ccl / 1e3:.2f} ms ({100 * ccl / max(tot, 1e-9):.1f}%)", flush=True)
    print(f"DECODE_PROF {tag}: by op", flush=True)
    for k, (d, c) in sorted(kind.items(), key=lambda kv: -kv[1][0])[:40]:
        print(f"  {d:9.1f} us {100 * d / max(tot, 1e-9):5.1f}%  x{c:<5d} {k}", flush=True)
    # call sites with the line number stripped of the per-layer repetition: same file:func:line across layers
    print(f"DECODE_PROF {tag}: by call site (summed over layers)", flush=True)
    for lab, (d, c) in sorted(by.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"  {d:9.1f} us {100 * d / max(tot, 1e-9):5.1f}%  x{c:<5d} {lab}", flush=True)
    calls = [c for c in os.environ.get("DPROF_CALLS", "").split(",") if c]
    if calls:  # every matching program of the first layers, in order (which linear / op shape costs what)
        shown = 0
        for i in range(first, last):
            lab = OWN.get(i, "")
            if i in dur and any(c in lab for c in calls) and shown < int(os.environ.get("DPROF_CALLS_N", "40")):
                print(f"  CALL {tag} {dur[i]:8.1f} us {lab}", flush=True)
                shown += 1
    return tot, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="1,8")
    ap.add_argument("--layers", default="")
    ap.add_argument("--ctx", type=int, default=12288)
    ap.add_argument("--plen", type=int, default=200)
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--top", type=int, default=70)
    args = ap.parse_args()
    os.environ["HY4_SLOTS"] = "8"
    from types import SimpleNamespace

    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh
    from models.autoports.tencent_hy4_flash.tt.decode import Hy4Decoder
    from models.autoports.tencent_hy4_flash.tt.decode_batch import Hy4BatchDecoder

    mesh = open_mesh()
    try:
        model = build(mesh, SimpleNamespace(layers=args.layers, ctx=args.ctx, chunk=2048, no_engram=args.no_engram))
        g = torch.Generator().manual_seed(0)
        P = args.plen
        prompts = [torch.randint(1000, 100000, (P + 7 * u,), generator=g) for u in range(8)]
        for U in [int(x) for x in args.users.split(",")]:
            for trace in (False, True):
                if U == 1:
                    dec = Hy4Decoder(model, trace=trace)
                    model.set_slot(0)
                    model.reset()
                    model.prefill(prompts[0])
                    dec.start(P)
                    dec.set_history(prompts[0].tolist())
                    st = {"p": P, "i": 0}

                    def step():
                        dec.run([1000 + 7919 * st["i"] % 90000], p0=st["p"])
                        st["p"] += 1
                        st["i"] += 1
                else:
                    dec = Hy4BatchDecoder(model, lanes=8, trace=trace)
                    lanes = []
                    for u in range(U):
                        model.set_slot(u)
                        model.reset()
                        model.prefill(prompts[u])
                        lanes.append(dec.add(u, int(prompts[u].numel()), prompts[u].tolist()))
                    st = {"i": 0}

                    def step():
                        dec.run_batch({l: [1000 + (7919 * (st["i"] + 13 * l)) % 90000] for l in lanes})
                        st["i"] += 1
                if not trace:
                    for _ in range(3):
                        step()
                    tot, n = profile(mesh, step, f"{U} user(s) eager", args.top, None)
                    res = (tot, n)
                else:
                    for _ in range(4):
                        step()
                    ts = []
                    for _ in range(30):
                        ttnn.synchronize_device(mesh)
                        t0 = time.perf_counter()
                        step()
                        ttnn.synchronize_device(mesh)
                        ts.append(time.perf_counter() - t0)
                    med = sorted(ts)[len(ts) // 2] * 1e3
                    tr = dec.btraces if U > 1 else dec.traces
                    rp = []
                    for key, tid in tr.items():
                        for _ in range(10):  # replay alone (inputs of the last step): the device time of the step
                            ttnn.synchronize_device(mesh)
                            t0 = time.perf_counter()
                            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            rp.append(time.perf_counter() - t0)
                        print(f"DECODE_PROF {U} user(s): trace {key} replay alone median "
                              f"{sorted(rp)[len(rp) // 2] * 1e3:.2f} ms", flush=True)
                        rp = []
                    print(f"DECODE_PROF {U} user(s): traced step (inputs + replay + logits read, sync'd) median "
                          f"{med:.2f} ms; device op sum {res[0] / 1e3:.2f} ms over {res[1]} programs -> gaps "
                          f"{med - res[0] / 1e3:.2f} ms ({(med - res[0] / 1e3) * 1e3 / max(res[1], 1):.1f} us per program)",
                          flush=True)
                    dec.release()
                if U > 1:
                    for l in list(lanes):
                        dec.drop(l)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
