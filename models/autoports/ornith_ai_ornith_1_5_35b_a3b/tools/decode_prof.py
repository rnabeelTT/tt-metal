# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op DEVICE time of one decode step, attributed to the Python call site that issued each program.

Runs an eager (untraced) decode step of a reduced model with the device profiler on. A ttnn pre/post-operation hook
records which device-operation ids each Python-level ttnn op issued (innermost op wins) and the model frame that called
it; after ``ReadDeviceProfiler`` the device CSV's ``run host ID`` is joined to those ids. Device time of a program =
first kernel-zone start to last kernel-zone end over all cores of a chip (max over chips). Host gaps between programs
are not counted: the sum is the traced step's lower bound, the per-site split shows where the device time goes.

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \\
      python -m models.autoports.ornith_ai_ornith_1_5_35b_a3b.tools.decode_prof --port <port> --layers 0,1,2,3 --batch 1
Works for every Qwen3.5-family autoport (``--port``).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import importlib
import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
# fast runtime mode binds every op without the pre/post-operation hooks: off, before ttnn registers its operations
os.environ.setdefault("TTNN_CONFIG_OVERRIDES", '{"enable_fast_runtime_mode": false}')

import torch  # noqa: E402

import ttnn  # noqa: E402

OWN = collections.OrderedDict()  # device op id -> label
STACK = []
LAST = [0]


def _site():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "/ttnn/ttnn/" not in fn and "/decorators.py" not in fn and "contextlib" not in fn:
            short = fn.split("/models/")[-1].replace("demos/blackhole/qwen36/tt/", "q36/").replace("autoports/", "")
            return f"{short}:{f.f_code.co_name}"
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
    STACK.append(f"{name.replace('ttnn.', '')} @ {_site()}")


def post_hook(op, args, kwargs, out):
    _flush()
    if STACK:
        STACK.pop()


def read_csv():
    paths = sorted(glob.glob(os.path.join(os.environ.get("TT_METAL_HOME", "/opt/tt-metal"), "generated/profiler/.logs/profile_log_device.csv")))
    assert paths, "no profile_log_device.csv (device profiler off?)"
    f = open(paths[0])
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0]) / 1e3
    rows = csv.reader(f)
    hdr = [h.strip() for h in next(rows)]
    ix = {h: i for i, h in enumerate(hdr)}
    span = {}  # (chip, run) -> [min start, max end]
    for r in rows:
        zone = r[ix["zone name"]].strip()
        if not zone.endswith("-KERNEL"):
            continue
        run = int(r[ix["run host ID"]].strip() or 0) >> 10  # device op id << 10 | chip
        chip = r[ix["PCIe slot"]].strip()
        t = int(r[ix["time[cycles since reset]"]])
        s = span.setdefault((chip, run), [t, t])
        s[0] = min(s[0], t)
        s[1] = max(s[1], t)
    dur = collections.defaultdict(float)
    for (chip, run), (a, b) in span.items():
        dur[run] = max(dur[run], (b - a) / freq / 1e3)  # us, max over chips
    return dur


def read_api():
    """{runtime_id: device kernel us (max over chips)} from the in-process analysis of the last ReadDeviceProfiler."""
    dur = collections.defaultdict(float)
    try:
        data = ttnn.get_latest_programs_perf_data()
    except Exception as e:  # noqa: BLE001
        print(f"DECODE_PROF perf-data API unavailable: {e}", flush=True)
        return dur
    for chip, progs in (data or {}).items():
        for p in progs:
            res = p.program_analyses_results
            r = res.get("DEVICE KERNEL DURATION [ns]") or res.get("DEVICE FW DURATION [ns]")
            if r is not None:
                rid = p.program_execution_uid.runtime_id >> 10  # runtime id = device op id << 10 | chip
                dur[rid] = max(dur[rid], r.duration / 1e3)
    return dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="ornith_ai_ornith_1_5_35b_a3b")
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--prefill", type=int, default=0, help="profile one prefill of this many tokens instead of decode")
    ap.add_argument("--calls", default="", help="comma list of label substrings: print every matching program in order")
    args = ap.parse_args()
    B = args.batch
    layers = [int(x) for x in args.layers.split(",")]
    if args.port == "c30":
        # Qwen3-Coder-30B-A3B family (models/demos/blackhole/qwen3_coder_30b_a3b): all layers are attention + EP MoE
        gen_mod = importlib.import_module("models.demos.blackhole.qwen3_coder_30b_a3b.tt.generator")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=48 << 20)
        gen = gen_mod.build_generator("models/demos/blackhole/qwen3_coder_30b_a3b", mesh,
                                      override_num_layers=len(layers), max_context_len=4096, max_batch_size=B)
        kv = gen._ensure_kv_cache()
        pt = gen.make_page_table([128] * B)
        ids = torch.tensor([[1000 + (i * 7) % 9973 for i in range(64)]])
        gen.prefill_forward(ids, page_table=pt[:1], kv_cache=kv, prompt_lens=[64], sampling_mode="host")

        def step():
            gen.decode_forward(torch.full((B, 1), 1234, dtype=torch.int64), torch.full((B,), 64, dtype=torch.int64),
                               page_table=pt, kv_cache=kv, sampling_mode="host", enable_trace=False, active_batch=B)

        kinds = f"{len(layers)} attn+moe"
    else:
        base = f"models.autoports.{args.port}"
        gen_mod = importlib.import_module(f"{base}.tt.generator")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=256 << 20)
        gen = gen_mod.build_generator(f"models/autoports/{args.port}", mesh, max_seq_len=max(4096, args.prefill + 256),
                                      max_batch_size=B,
                                      warm_decode_trace=False, warm_prefill_trace=False, layer_indices=layers)
        m = gen.model
        ids = torch.tensor([[1000 + (i * 7) % 9973 for i in range(64)]])
        gen.prefill_forward(ids, page_table=gen.page_table[:1], prompt_lens=[64])
        tok = torch.full((B, 1), 1234, dtype=torch.int64)
        pos = torch.full((B,), 64, dtype=torch.int64)

        def step():
            gen.decode_forward(tok, pos, page_table=gen.page_table[:B], enable_trace=False)

        if args.prefill:
            pids = torch.tensor([[1000 + (i * 7) % 9973 for i in range(args.prefill)]])

            def step():
                gen.prefill_forward(pids, page_table=gen.page_table[:1], prompt_lens=[args.prefill])

        n_gdn = sum(1 for l in m.layers if not l.is_full_attention)
        kinds = f"{n_gdn} gdn + {len(m.layers) - n_gdn} attn"
    for _ in range(args.warm):
        step()
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    LAST[0] = ttnn._ttnn.get_device_operation_id()
    first_id = LAST[0]
    with ttnn.decorators.register_pre_operation_hook(pre_hook), ttnn.decorators.register_post_operation_hook(post_hook):
        step()
    ttnn.synchronize_device(mesh)
    last_id = ttnn._ttnn.get_device_operation_id()
    ttnn.ReadDeviceProfiler(mesh)
    dur = read_api()
    if not dur:
        getattr(gen, "teardown", lambda: None)()
        ttnn.close_mesh_device(mesh)  # the device CSV is written on close
        mesh = None
        dur = read_csv()
    print(f"DECODE_PROF port {args.port} {'prefill ' + str(args.prefill) if args.prefill else 'decode'} batch {B} layers {layers} ({kinds}), "
          f"device ops {first_id}..{last_id}", flush=True)
    by = collections.defaultdict(lambda: [0.0, 0])
    tot = 0.0
    missing = 0
    for i in range(first_id, last_id):
        d = dur.get(i)
        if d is None:
            missing += 1
            continue
        lab = OWN.get(i, "<untracked>")
        by[lab][0] += d
        by[lab][1] += 1
        tot += d
    print(f"DECODE_PROF total device {tot:.1f} us over {last_id - first_id - missing} programs ({missing} without "
          f"profile data)", flush=True)
    for lab, (d, n) in sorted(by.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"  {d:8.1f} us {100 * d / max(tot, 1e-9):5.1f}%  x{n:<3d} {lab}", flush=True)
    if args.calls:
        keys = args.calls.split(",")
        for i in range(first_id, last_id):
            lab = OWN.get(i, "<untracked>")
            if i in dur and any(k in lab for k in keys):
                print(f"  CALL {i} {dur[i]:8.1f} us {lab}", flush=True)
    if mesh is not None:
        getattr(gen, "teardown", lambda: None)()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
