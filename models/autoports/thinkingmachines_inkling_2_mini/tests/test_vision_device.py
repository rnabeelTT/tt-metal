# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The vision tower (tt/vision.py) vs transformers' Inkling2VisionModel (reference/vision_ref.py fixtures): pcc of the
embeddings, layers 0 / 11 / 22 and the projected image embeddings against the fp32 reference, next to the bf16
reference's own pcc (its noise floor). Pass: projected embeddings pcc >= 0.999 vs fp32 on every image.

usage: MESH=1 tools/run_dev.sh python -m models.autoports.thinkingmachines_inkling_2_mini.tests.test_vision_device
env: IMAGES (cat,valley), TIMES (1: time a few forwards per image), STREAM (1: weights streamed from the host),
     BENCH (token counts, e.g. 1024,4096,10816: time synthetic images and the DRAM they need)"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.vision import HostInputs, VisionTower, vision_config
from models.autoports.thinkingmachines_inkling_2_mini.tt.weights import Checkpoint

FIX = os.environ.get("I2M_FIXTURES", "/cache/fixtures") + "/vision"


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def dram_free(mesh):
    d = mesh.get_devices()[0] if hasattr(mesh, "get_devices") else mesh
    s = ttnn.get_memory_view(d, ttnn.BufferType.DRAM)
    return s.total_bytes_free_per_bank * s.num_banks


def bench(mesh, tower):
    """Synthetic images (1 x n tokens): ms per image and the lowest free DRAM seen during the forward (per layer)."""
    from models.autoports.thinkingmachines_inkling_2_mini.tt import vision

    for n in [int(v) for v in os.environ["BENCH"].split(",")]:
        pv, grid = torch.rand(n, 40, 40, 3) * 2 - 1, torch.tensor([[1, 1, n]])
        hi = HostInputs(tower, pv, grid)
        base = dram_free(mesh)
        low = [base]
        tower.forward_device(hi, capture=lambda k, t: low.append(dram_free(mesh)))  # (compile)
        ttnn.synchronize_device(mesh)
        t1 = time.time()
        tower(pv, grid)
        ttnn.synchronize_device(mesh)
        print(f"[bench] {n} tokens (S_pad {vision.bucket(4 * n + 1)}): {(time.time() - t1) * 1e3:.0f} ms, transient DRAM "
              f">= {(base - min(low)) / 2**30:.3f} GiB", flush=True)


def main():
    N = int(os.environ.get("MESH", "1"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    ck = Checkpoint()
    t0 = time.time()
    stream = os.environ.get("STREAM", "0") == "1"
    free0 = dram_free(mesh)
    tower = VisionTower(mesh, ck, vision_config(ck.snap), stream=stream)
    print(f"tower weights on device: {(free0 - dram_free(mesh)) / 2**30:.3f} GiB (stream={stream})", flush=True)
    print(f"tower up in {time.time() - t0:.0f}s", flush=True)
    d0 = (lambda t: ttnn.get_device_tensors(t)[0]) if N > 1 else (lambda t: t)  # noqa: E731
    if os.environ.get("BENCH"):
        bench(mesh, tower)
    worst = 1.0
    for name in os.environ.get("IMAGES", "cat,valley").split(","):
        fx = torch.load(f"{FIX}/{name}.pt")
        pv, grid = fx["pixel_values"], fx["image_grid_thw"]
        hi = HostInputs(tower, pv, grid)
        P = 4 * hi.N
        got = {}

        def cap(k, t):
            if k == "embed" or k in (0, 11, 22):
                x = ttnn.to_torch(d0(t)).float().reshape(hi.S_pad, -1)
                got[k] = torch.cat([x[P : P + 1], x[:P]])  # (the reference's order: class token first)

        out = tower.forward_device(hi, capture=cap)
        o = ttnn.to_torch(d0(out)).float().reshape(-1, tower.Dout)[: hi.N]
        r32, r16 = fx["fp32"], fx["bf16"]
        rows = [("embed", got["embed"], r32["embed"], r16["embed"])]
        rows += [(f"layer {i}", got[i], r32["layers"][i], r16["layers"][i]) for i in (0, 11, 22)]
        rows += [("out", o, r32["out"], r16["out"])]
        for k, g, a, b in rows:
            print(f"[{name} S={hi.S} pad {hi.S_pad}] {k:9s} pcc vs fp32 {pcc(g, a):.6f}  (bf16 reference {pcc(b, a):.6f})  "
                  f"max|d| {(g - a).abs().max().item():.3g}", flush=True)
        worst = min(worst, pcc(o, r32["out"]))
        if os.environ.get("TIMES", "1") == "1":
            tower(pv, grid)
            ttnn.synchronize_device(mesh)
            t1 = time.time()
            for _ in range(3):
                tower(pv, grid)
            print(f"[{name}] {hi.N} tokens: {(time.time() - t1) / 3 * 1e3:.0f} ms per image (host in / out included)",
                  flush=True)
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst >= 0.999 else 'FAIL'}: worst image-embedding pcc vs fp32 {worst:.6f}")


if __name__ == "__main__":
    main()
