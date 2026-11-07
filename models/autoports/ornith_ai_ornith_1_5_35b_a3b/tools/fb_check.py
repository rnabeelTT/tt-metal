# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN3_FABRIC_BOUNDARY bring-up check on the (1, 4) ring mesh.

1. the FabricBoundary op alone (H = 2048): entry boundary (norm of a replicated row) and an all-reduce boundary
   (residual + sum of 4 per-die partials, then the norm) against torch;
2. the reduced model: one eager decode step's logits with the fabric path off / on / off (same KV position).

    python -m models.demos.blackhole.qwen3_coder_30b_a3b.tests.fb_check [--layers 4]
"""
import argparse

import torch

import ttnn


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rms(x, g, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g.float()


def op_check(mesh, H=2048, eps=1e-6):
    from models.demos.blackhole.qwen36.tt.fabric_boundary.ops import FabricBoundary

    fb = FabricBoundary(mesh, H, eps, max_rows=1)
    torch.manual_seed(0)
    res = torch.randn(1, 1, 1, H).bfloat16()
    parts = (torch.randn(4, 1, 1, H) * 0.5).bfloat16()
    gw = (1 + 0.1 * torch.randn(H)).bfloat16()
    rep = ttnn.ReplicateTensorToMesh(mesh)
    res_tt = ttnn.from_torch(res, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                             memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=rep)  # entry source: L1 (see ops.py)
    g_tt = ttnn.from_torch(gw.reshape(1, 1, H // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
    gamma = fb.gamma(g_tt)
    flat, x0 = fb(res_tt, None, gamma)
    x0_h = ttnn.to_torch(x0, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    ref0 = rms(res, gw, eps)
    for d in range(4):
        print(f"FB_CHECK entry die {d}: pcc {pcc(x0_h[d].reshape(-1)[:H], ref0):.6f} "
              f"max|d| {float((x0_h[d].reshape(-1)[:H].float() - ref0.flatten()).abs().max()):.4f}", flush=True)
    part_tt = ttnn.from_torch(parts, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                              memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    flat2, x2 = fb(flat, part_tt, gamma, "mixer")
    x2_h = ttnn.to_torch(x2, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    ref2 = rms(res.float() + parts.float().sum(0, keepdim=True), gw, eps)
    for d in range(4):
        print(f"FB_CHECK reduce die {d}: pcc {pcc(x2_h[d].reshape(-1)[:H], ref2):.6f} "
              f"max|d| {float((x2_h[d].reshape(-1)[:H].float() - ref2.flatten()).abs().max()):.4f}", flush=True)


def model_check(mesh, layers):
    from models.demos.blackhole.qwen3_coder_30b_a3b.tt import multichip_decoder as md
    from models.demos.blackhole.qwen3_coder_30b_a3b.tt.generator import build_generator

    gen = build_generator("models/demos/blackhole/qwen3_coder_30b_a3b", mesh, override_num_layers=layers,
                          max_context_len=4096, max_batch_size=1)
    kv = gen._ensure_kv_cache()
    pt = gen.make_page_table([128])
    ids = torch.tensor([[1000 + (i * 7) % 9973 for i in range(32)]])
    gen.prefill_forward(ids, page_table=pt, kv_cache=kv, prompt_lens=[32], sampling_mode="host")
    out = {}
    for name, flag in (("off", False), ("on", True), ("off2", False), ("on2", True)):
        md.FABRIC_BOUNDARY = flag
        lg = gen.decode_forward(torch.tensor([[1234]]), torch.tensor([32]), page_table=pt, kv_cache=kv,
                                sampling_mode="host", enable_trace=False, active_batch=1)
        out[name] = torch.as_tensor(lg).float().reshape(-1)
        print(f"FB_CHECK model {name}: argmax {int(out[name].argmax())} finite {bool(torch.isfinite(out[name]).all())} "
              f"absmax {float(out[name].abs().max()):.3f}", flush=True)
    print(f"FB_CHECK model pcc off/off2 {pcc(out['off'], out['off2']):.6f} off/on {pcc(out['off'], out['on']):.6f} "
          f"on/on2 {pcc(out['on'], out['on2']):.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--fabric", default="FABRIC_1D_RING")
    a = ap.parse_args()
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, a.fabric))
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=48 << 20)
    op_check(mesh)
    if not a.skip_model:
        model_check(mesh, a.layers)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
