# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""HYB_DEVCRAFT (crafted logits rows built on the device) against the host-crafted fused hybrid path and the sparse-
matmul hybrid path, row by row, on one MoE layer with random inputs (4 rows and 1 row). Run with HYB_MTP=3."""
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
        gen = build_generator(
            "models/autoports/qwen_qwen3_5_122b_a10b",
            mesh,
            max_seq_len=2048,
            warm_decode_trace=False,
            warm_prefill_trace=False,
            layer_indices=[0, 1, 2, 3],
        )
        moe = gen.model.layers[0].feed_forward
        H = int(moe.hidden)
        torch.manual_seed(0)
        x = torch.randn(1, 1, 4, H) * 0.5
        rep = ttnn.ReplicateTensorToMesh(mesh)
        comp = ttnn.ConcatMeshToTensor(mesh, dim=-1)

        def run(xx):
            t = ttnn.from_torch(
                xx,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            o = moe.forward(t, mode="decode")
            return ttnn.to_torch(o, mesh_composer=comp).float()

        r = {}
        for skip in (0.0, 0.1):
            HYB.skip = skip
            for m in (4, 1):
                xm = x[:, :, :m].contiguous()
                HYB.devcraft = False
                host = run(xm)[..., :m, :H].reshape(m, H)
                HYB.devcraft = True
                dev = run(xm)[..., :m, :H].reshape(m, H)
                dev2 = run(xm)[..., :m, :H].reshape(m, H)
                HYB.devcraft = False
                moe._hyb_fused = False
                ref = torch.stack([run(xm[:, :, i : i + 1]).reshape(-1)[:H] for i in range(m)])
                moe._hyb_fused = True
                for i in range(m):
                    r[f"skip{skip}_m{m}_row{i}"] = {
                        "dev_vs_host": round(pcc(dev[i], host[i]), 5),
                        "dev_vs_unfused": round(pcc(dev[i], ref[i]), 5),
                        "host_vs_unfused": round(pcc(host[i], ref[i]), 5),
                        "dev_repeat": round(pcc(dev[i], dev2[i]), 6),
                    }
        res["pcc"] = r
        print("DEVCRAFT", json.dumps(r), flush=True)
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
