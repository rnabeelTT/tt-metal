# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/engram_exact (HY4_ENG_FUSED=1) vs the ttnn decode chain (HY4_ENG_FUSED=0): bitwise, several synthetic cases
(outlier stream dims, padding rows, tiny / large gates); eager and traced timing."""

import os
import time

import torch
import ttnn

import models.autoports.tencent_hy4_flash.tt.common as C
from models.autoports.tencent_hy4_flash.tt.config import Hy4Config
from models.autoports.tencent_hy4_flash.tt.engram import Hy4EngramDevice


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 << 20)
    bad = 0
    try:
        cfg = Hy4Config.load()
        H, hc, D = cfg.hidden, cfg.hc_mult, 1536
        rep = ttnn.ReplicateTensorToMesh(mesh)
        back = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(hc, 32, H)  # noqa
        for case in range(6):
            torch.manual_seed(case)
            sd = {"value_proj.weight": torch.randn(H, D) * 0.03, "value_proj.bias": torch.randn(H) * 0.1,
                  "unified_pre_gamma": 1 + torch.randn(H) * 0.1}
            for i in range(hc):
                sd[f"key_projs.{i}.weight"] = torch.randn(H, D) * 0.03 * (1 + case)
                sd[f"key_projs.{i}.bias"] = torch.randn(H) * 0.1
                sd[f"norm_key.{i}.weight"] = 1 + torch.randn(H) * 0.1
                sd[f"norm_query.{i}.weight"] = 1 + torch.randn(H) * 0.1
                sd[f"unified_per_hc_scales.{i}"] = torch.randn(H) * 0.05
            sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
            os.environ["HY4_ENG_FUSED"] = "1"
            eng = Hy4EngramDevice(mesh, cfg, sd)
            assert eng.dec is not None and eng.fused is not None
            X = torch.randn(hc, 32, H) * (0.3 + case)
            X[:, :, torch.randint(0, H, (6,))] *= 200  # massive-activation dims
            X = X.to(torch.bfloat16)
            E = (torch.randn(32, D)).to(torch.bfloat16)
            nreal = [1, 2, 8, 32, 5, 12][case]
            C.DECODE_ROWS[0] = nreal  # the pass's real rows (engram._row_sum: 8-row blocks past 8)
            E[nreal:] = 0
            Xt = ttnn.from_torch(X.reshape(1, hc, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 mesh_mapper=rep)
            Et = ttnn.from_torch(E.reshape(1, 1, 32, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 mesh_mapper=rep)
            outs = {}
            for mode, fold in (("0", "1"), ("1", "0"), ("1", "1")):
                os.environ["HY4_ENG_FUSED"], os.environ["HY4_ENG_FOLD"] = mode, fold
                o = eng.forward(Xt, Et, ttnn.DRAM_MEMORY_CONFIG)
                outs[mode + fold] = back(o)
                ttnn.deallocate(o)
            outs["0"], outs["1"] = outs["01"], outs["11"]
            nd0 = (outs["0"].view(torch.int16) != outs["10"].view(torch.int16)).sum().item()
            eq = torch.equal(outs["0"].view(torch.int16), outs["1"].view(torch.int16)) and nd0 == 0
            nd = (outs["0"].view(torch.int16) != outs["1"].view(torch.int16)).sum().item() + nd0
            bad += 0 if eq else 1
            print(f"ENGX case {case} rows {nreal}: bitwise {eq} mismatches {nd}/{outs['0'].numel()}", flush=True)
            if case == 0:
                for mode in ("0", "1"):
                    os.environ["HY4_ENG_FUSED"] = mode
                    ttnn.deallocate(eng.forward(Xt, Et, ttnn.DRAM_MEMORY_CONFIG))
                    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                    o = eng.forward(Xt, Et, ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    t0 = time.time()
                    for _ in range(50):
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    print(f"ENGX mode {mode}: traced {(time.time() - t0) / 50 * 1e6:.1f} us/call", flush=True)
                    if mode == "1":
                        tr = back(o)
                        print(f"ENGX traced bitwise vs chain {torch.equal(tr.view(torch.int16), outs['0'].view(torch.int16))}")
                    ttnn.release_trace(mesh, tid)
                    ttnn.deallocate(o)
        print(f"ENGX RESULT {'PASS' if bad == 0 else 'FAIL'} ({bad} cases differ)", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
