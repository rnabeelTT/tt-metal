# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One eager verify pass (R rows) through every layer + final norm + LM head after a short prefill, logits saved for an
A/B comparison of two environments (bit-exact changes must give identical logits).

usage (hybrid env as tools/eval_hybrid.sh, TOOL=pass_dump.py): pass_dump.py <out.pt> [R=3]
compare: python pass_dump.py --cmp a.pt b.pt
"""

import os
import sys

import torch

if len(sys.argv) > 1 and sys.argv[1] == "--cmp":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    for k in a:
        x, y = a[k].float(), b[k].float()
        d = (x - y).abs()
        top = (x.argmax(-1) == y.argmax(-1)).float().mean().item()
        print(f"{k}: identical {torch.equal(x, y)} | max |d| {d.max().item():.4g} mean {d.mean().item():.3g} | "
              f"argmax agree {top:.3f}")
    sys.exit(0)

if os.environ.get("HYB_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("HYB_POC_DIR", "/opt/tt-metal/models/experimental/hybrid_moe"))
    os.environ.setdefault("HYB_PORT", "thinkingmachines_inkling_small")
    import hybrid

    hybrid.install()

import ttnn  # noqa: E402

from models.autoports.thinkingmachines_inkling_small.tt.model import create_model  # noqa: E402


def main():
    out = sys.argv[1]
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    res = {}
    try:
        args, model = create_model(mesh, max_seq_len=8192)
        hyb = sys.modules["hybrid.patch"].HYB if "hybrid.patch" in sys.modules else None
        if hyb is not None and hyb.enabled:
            for li in list(hyb.lib):
                hyb.lib_layer(li)
            if hyb.fetch.dir:
                hyb.fetch.finalize(hyb.loader, hyb.stream)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        nblk = model.layers[0].attention.max_blocks
        pt_host = torch.arange(nblk, dtype=torch.int32).reshape(1, -1)
        pt = ttnn.from_torch(pt_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        g = torch.Generator().manual_seed(0)
        prompt = torch.randint(1000, 150000, (300,), generator=g)
        model.reset_sequence_state()
        model.prefill_tokens(prompt, pt_host)
        p0 = len(prompt)
        for step in range(2):
            ids = torch.randint(1000, 150000, (R,), generator=g)
            tok = ttnn.from_torch(ids.to(torch.int32).reshape(1, R), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=mesh, mesh_mapper=rep)
            pos = ttnn.from_torch(torch.arange(p0, p0 + R, dtype=torch.int32), dtype=ttnn.int32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
            idx = torch.full((R, R), -1, dtype=torch.int32)
            idx[torch.arange(R), torch.arange(R)] = torch.arange(p0, p0 + R, dtype=torch.int32)
            kvi = ttnn.from_torch(idx, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
            x = ttnn.reshape(model.embd(tok), (1, 1, R, -1))
            for layer in model.layers:
                x = layer.forward_rows(x, pos, pt, commit_all=False, kv_idx=kvi)
            h = model._final_norm_decode(x)
            lg = model._lm_head(h)
            if lg.is_sharded():
                lg = ttnn.sharded_to_interleaved(lg, ttnn.DRAM_MEMORY_CONFIG)
            full = ttnn.to_torch(lg, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
            res[f"step{step}"] = full.reshape(-1, full.shape[-1])[:R].clone()
            for layer in model.layers:
                layer.commit_rows(R)
            p0 += R
        torch.save(res, out)
        print(f"saved {out}: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in res.items()), flush=True)
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
