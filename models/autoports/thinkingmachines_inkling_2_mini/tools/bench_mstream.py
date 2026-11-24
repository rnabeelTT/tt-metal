# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The streamed q | r | s | g, k, v call (tt/mstream.py) at one TP degree's per-die shard shapes, on one die: traced us
per call for each variant, outputs compared bitwise with the first variant (the reference) at rows 1 / 4 / 8.

env: TP (4: the shapes a TP4 die holds), VARIANTS ("name:k=v,k=v;..." keys qr, kv, and I2M_MS_* knobs as HMIN / G / KB /
NBLK / anything else; read at call time), REPS (50 calls per trace), ITERS (10)
usage: tools/run_dev.sh python .../tools/bench_mstream.py"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.mstream import MultiStream

REPS, ITERS = int(os.environ.get("REPS", "50")), int(os.environ.get("ITERS", "10"))


def parse(spec):
    out = []
    for v in spec.split(";"):
        name, _, kv = v.partition(":")
        out.append((name, dict(p.split("=") for p in kv.split(",") if p)))
    return out


def setup(TP):
    K, NH, kd = 2304, 24 // TP, 8 // TP * 128
    fq = NH * 128 * 2 + NH * 16 + 32
    fqp = fq + (-fq) % 256
    print(f"TP{TP}: K {K}, q {fq} -> {fqp}, k / v {kd}", flush=True)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    torch.manual_seed(0)
    wq = torch.randn(K, fqp) * K**-0.5
    wq[:, fq:] = 0
    up = lambda t: ttnn.as_tensor(t.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                  device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ws = [up(wq), up(torch.randn(K, kd) * K**-0.5), up(torch.randn(K, kd) * K**-0.5)]
    return dev, K, fqp, kd, ws


def make(dev, K, fqp, kd, kv):
    """A MultiStream for one variant: qr / kv here, the other keys set as I2M_MS_* (the kernels' knobs, read per call)."""
    for k in list(os.environ):
        if k.startswith("I2M_MS_"):
            del os.environ[k]
    for k, v in kv.items():
        if k not in ("qr", "kv", "kvr"):
            os.environ[f"I2M_MS_{k}"] = v
    qr, grp = int(kv.get("qr", 4)), kv.get("kv", "0") == "1"
    kvr = int(kv.get("kvr", 1))
    return MultiStream(dev, K, [(fqp, ttnn.float32, qr), (kd, ttnn.bfloat16, kvr), (kd, ttnn.bfloat16, kvr)],
                       ttnn.bfloat8_b, interleaved=True, fidelity=ttnn.MathFidelity.HiFi4,
                       groups=[[0], [1, 2]] if grp else None)


def main():
    TP = int(os.environ.get("TP", "4"))
    dev, K, fqp, kd, ws = setup(TP)
    variants = parse(os.environ.get("VARIANTS", "head:qr=4,kv=0,HMIN=8,G=1,KB=72,NBLK=3"))
    refs = {}
    for name, kv in variants:
        ms = make(dev, K, fqp, kd, kv)
        line = [f"{name:>10}"]
        ok = True
        for R in (1, 4, 8):
            g = torch.Generator().manual_seed(R)
            xs = [ttnn.from_torch(torch.randn(1, 1, R, K, generator=g).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
                  for _ in range(3)]
            outs = [ttnn.to_torch(o)[..., :R, :] for o in ms(xs, ws, R)]
            if R not in refs:
                refs[R] = outs
            else:
                ok &= all(torch.equal(a, b) for a, b in zip(outs, refs[R]))
            ttnn.synchronize_device(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            for _ in range(REPS):
                ms(xs, ws, R)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            line.append(f"R{R} {(time.perf_counter() - t0) / (REPS * ITERS) * 1e6:6.2f} us")
            ttnn.release_trace(dev, tid)
        cores = sum(len(ttnn.corerange_to_cores(dl.cores)) for dl, _, _, _ in ms.parts)
        rng = [(len(dl.noc0_cores.ranges()), len(dl.noc1_cores.ranges()) if dl.noc1_cores else 0) for dl, _, _, _ in ms.parts]
        kv = dict(kv, ranges=rng)
        print("  ".join(line) + f"  cores {cores}  {'bitwise' if ok else 'DIFFERS'}  {kv}", flush=True)
    nbytes = (fqp + 2 * kd) * K * 1088 // 1024
    print(f"bytes {nbytes / 1e6:.2f} MB: floor {nbytes / 512e3:.1f} us at 512 GB/s", flush=True)
    ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
