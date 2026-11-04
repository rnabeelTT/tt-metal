# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fused hybrid multi-row decode (FusedMoEDecode rows ops on crafted logits rows) against the single-row fused hybrid
path, row by row, on one MoE layer with random inputs. Run with HYB_MTP=3 (fused rows up to 4) and POC_LAYERS=0."""
import json
import os
import sys

import hybrid

HYB = hybrid.install()

import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator  # noqa: E402


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    out_dir = sys.argv[1]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    res = {}
    try:
        gen = build_generator("models/autoports/qwen_qwen3_5_122b_a10b", mesh, max_seq_len=2048,
                              warm_decode_trace=False, warm_prefill_trace=False, layer_indices=[0, 1, 2, 3])
        moe = gen.model.layers[0].feed_forward
        H = int(moe.hidden)
        torch.manual_seed(0)
        x = torch.randn(1, 1, 4, H) * 0.5
        rep = ttnn.ReplicateTensorToMesh(mesh)
        comp = ttnn.ConcatMeshToTensor(mesh, dim=-1)

        def run(xx):
            t = ttnn.from_torch(xx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            o = moe.forward(t, mode="decode")
            return ttnn.to_torch(o, mesh_composer=comp).float()

        HYB.skip = 0.0
        rows = run(x)[..., :4, :H].reshape(4, H)
        single = torch.stack([run(x[:, :, i : i + 1]).reshape(-1)[:H] for i in range(4)])
        same = run(x[:, :, :1].expand(1, 1, 4, H).contiguous())[..., :4, :H].reshape(4, H)
        moe._hyb_fused = False  # the sparse-matmul hybrid path (validated earlier), per row
        ref = torch.stack([run(x[:, :, i : i + 1]).reshape(-1)[:H] for i in range(4)])
        moe._hyb_fused = True
        r = {}
        for i in range(4):
            r[f"row{i}"] = {"rows_vs_single": round(pcc(rows[i], single[i]), 5),
                            "single_vs_unfused": round(pcc(single[i], ref[i]), 5),
                            "rows_vs_unfused": round(pcc(rows[i], ref[i]), 5),
                            "same_rows_vs_single0": round(pcc(same[i], single[0]), 5)}
        res["pcc"] = r
        print("ROWS", json.dumps(r), flush=True)
    finally:
        json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(1)
    os._exit(0)
