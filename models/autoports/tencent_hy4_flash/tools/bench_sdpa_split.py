# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark + exactness check (traced replays) of the sparse_sdpa key split (HY4_SDPA_SPLIT): kv_split C vs 1 on
random data (bf16 q/kv, 32 heads, scale 1/16, HiFi4 fp32 DEST like Hy4), R real rows (the rest all-sentinel + sink,
HY4_IDX_PADSENT), nkeys selected keys. Every index list starts with a sink-like row (Hy4's learned sink: a reserved
cache row scored against q's sink columns, SINK = logit scale) so the first chunk often holds the running max. Reports
traced us/call, max |err| vs an fp32 torch reference, and whether the split output is bit-identical to C=1 over the R
real rows (the key split must be exact). SPLITS / ROWS / KEYS / QSCALE / SINK env; exit status 1 on any mismatch."""
import os
import sys

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tools.bench_dsh import timed
from models.autoports.tencent_hy4_flash.tt.attention import SENTINEL, SKF

SPLITS = [int(x) for x in os.environ.get("SPLITS", "2,4,8,9,12,16").split(",")]
ROWS = [int(x) for x in os.environ.get("ROWS", "1,8,16").split(",")]
KEYS = [int(x) for x in os.environ.get("KEYS", "2177,1025,257,129,40").split(",")]


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100_000_000)
    bad = 0
    try:
        print("grid", mesh.compute_with_storage_grid_size(), flush=True)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        torch.manual_seed(0)
        rows_cache, W, H, V = 32768 + 256, 608, 32, 512
        kv = torch.randn(1, 1, rows_cache, W)
        sink_row = rows_cache - 2
        kv[0, 0, sink_row] = 0
        kv[0, 0, sink_row, 576:578] = 16.0  # the Hy4 sink trick: a unit key in the sink columns (scale 1/16)
        kv = kv.to(torch.bfloat16)
        ct = ttnn.from_torch(kv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        q = torch.randn(1, H, 32, W) * float(os.environ.get("QSCALE", "1.0"))
        q[..., 576:] = 0
        q[..., 576] = torch.randn(1, H, 1) * float(os.environ.get("SINK", "4.0"))  # per-head sink logit
        q = q.to(torch.bfloat16)
        qt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                     math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        g = torch.Generator().manual_seed(1)
        for nkeys in KEYS:
            K = -(-(nkeys + 4) // 128) * 128
            keys = torch.sort(torch.randperm(rows_cache - 256, generator=g)[:nkeys]).values
            keys[0] = sink_row
            for R in ROWS:
                idx = torch.full((1, 1, 32, K), SENTINEL, dtype=torch.int64)
                for r in range(R):  # different key counts per row (the last row the full count)
                    n = max(1, nkeys - 37 * (R - 1 - r))
                    idx[0, 0, r, :n] = keys[:n]
                idx[:, :, R:, 0] = sink_row
                it = ttnn.from_torch(idx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                     device=mesh, mesh_mapper=rep)
                ref = torch.zeros(H, R, V)
                for r in range(R):
                    sel = idx[0, 0, r]
                    sel = sel[sel != SENTINEL]
                    kk = kv[0, 0, sel].float()
                    s = q[0, :, r].float() @ kk.T / 16
                    ref[:, r] = torch.softmax(s, -1) @ kk[:, :V]
                res = {}
                for C in [1] + SPLITS:
                    kw = {} if C == 1 else {"kv_split": C, "active_tokens": R}
                    try:
                        t, o = timed(mesh, lambda: ttnn.transformer.sparse_sdpa(
                            qt, ct, it, V, kv_format=SKF.BF16, scale=1 / 16, k_chunk_size=128,
                            compute_kernel_config=ckc, **kw), n=100)
                        out = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float()[0, :, :R]
                        ttnn.deallocate(o)
                        res[C] = out
                        err = (out - ref).abs().max().item()
                        rel = ((out - ref).norm() / ref.norm()).item()
                        exact = C == 1 or torch.equal(out, res[1])
                        d = (out - res[1]).abs().max().item()
                        bad += not exact
                        print(f"SPLIT keys {nkeys:5d} R {R:2d} C {C:2d}: {t:7.1f} us  max|err| {err:.3e} rel {rel:.3e}"
                              f"  {'EXACT' if exact else 'MISMATCH'} max|C-1| {d:.3e}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        bad += 1
                        print(f"SPLIT keys {nkeys} R {R} C {C}: FAIL {str(e)[:300]}", flush=True)
                ttnn.deallocate(it)
    finally:
        ttnn.close_mesh_device(mesh)
    print(f"SPLIT_SUMMARY mismatches/fails {bad}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
