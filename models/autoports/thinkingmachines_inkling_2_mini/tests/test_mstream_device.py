# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The one-program streamed q | r | s | g, k, v projections (tt/mstream.py) vs the interleaved decode matmuls
(attention.proj) on real weights and real-scale inputs, layers 0 (sliding) and 5 (global), rows 1 / 4 / 8: max |d|
and the fraction of bitwise-equal outputs per projection (q fp32, k / v bf16).

usage: MESH=4 tools/run_dev.sh python .../tests/test_mstream_device.py"""

import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.attention import AttnWeights, proj
from models.autoports.thinkingmachines_inkling_2_mini.tt.weights import Checkpoint, load_config


def main():
    N = int(os.environ.get("MESH", "4"))
    if N > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
    cfg, ck = load_config(), Checkpoint()
    rep = ttnn.ReplicateTensorToMesh(mesh) if N > 1 else None
    d0 = (lambda t: ttnn.get_device_tensors(t)[0]) if N > 1 else (lambda t: t)  # noqa: E731
    worst = 0.0
    torch.manual_seed(0)
    for li in (0, 5):
        w = AttnWeights(mesh, cfg, ck, li, wdtype=ttnn.bfloat8_b)  # (the served attention dtype)
        assert w.ms is not None
        for R in (1, 4, 8):
            xs = [ttnn.from_torch(torch.randn(1, 1, R, cfg.hidden_size).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep) for _ in range(3)]
            got = w.ms(xs, w.ms_w, R)
            ref = [proj(w, "q", xs[0], ttnn.float32), proj(w, "k", xs[1]), proj(w, "v", xs[2])]
            # exact q: the bf16 input times the dequantized bfp8 weight in float64; each path's error against it
            wq = ttnn.to_torch(d0(w.w_q)).double().reshape(cfg.hidden_size, -1)
            xq = ttnn.to_torch(d0(xs[0])).double().reshape(-1, cfg.hidden_size)[:R]
            exact = xq @ wq
            gq = ttnn.to_torch(d0(got[0])).double().reshape(-1, wq.shape[1])[:R]
            rq = ttnn.to_torch(d0(ref[0])).double().reshape(-1, wq.shape[1])[:R]
            print(f"layer {li} R={R} q vs exact: streamed mean |err| {(gq - exact).abs().mean():.3e} max "
                  f"{(gq - exact).abs().max():.3e}; interleaved matmul mean {(rq - exact).abs().mean():.3e} max "
                  f"{(rq - exact).abs().max():.3e}", flush=True)
            for name, g, r in zip("qkv", got, ref):
                g = ttnn.to_torch(d0(g)).float()[..., :R, :]
                r = ttnn.to_torch(d0(r)).float()[..., :R, : g.shape[-1]]
                d = (g - r).abs().max().item()
                rel = d / max(r.abs().max().item(), 1e-30)
                worst = max(worst, rel)
                print(f"layer {li} R={R} {name}: max|d| {d:.3g} (rel {rel:.2e}), bitwise equal "
                      f"{(g == r).float().mean().item():.4f}", flush=True)
    ttnn.close_mesh_device(mesh)
    print(f"{'PASS' if worst < 1e-5 else 'FAIL'}: worst relative max|d| {worst:.2e}")


if __name__ == "__main__":
    main()
