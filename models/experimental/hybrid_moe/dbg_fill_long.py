# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""DFlashDrafterPaged.fill_prompt at long contexts vs the CPU reference drafter: the context is the CPU target's taps
of the reference sequences tiled to P rows (the content does not matter for a device-vs-CPU check). Per P: PCC of
the final block rows and the drafts. usage (in the image): python dbg_fill_long.py <out dir>"""

import glob
import json
import os
import sys

sys.path.insert(0, "/poc")
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import Qwen3Config  # noqa: E402

import ttnn  # noqa: E402
import ref_m27 as R  # noqa: E402
from dflash_ref import DFlashDraftModel  # noqa: E402
from dflash_tt import DFlashDrafterPaged  # noqa: E402


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1])


@torch.no_grad()
def main():
    R.SNAP = glob.glob("/hf/hub/models--MiniMaxAI--MiniMax-M2.7/snapshots/*")[0]
    R.WMAP = json.load(open(f"{R.SNAP}/model.safetensors.index.json"))["weight_map"]
    snap = glob.glob("/hf/hub/models--nvidia--MiniMax-M2.7-DFlash/snapshots/*")[0]
    embed, head = R.get("model.embed_tokens.weight"), R.get("lm_head.weight")
    V, H = head.shape
    tp = torch.load("/poc/runs/m27/ref_dflash/taps.pt")
    rows = tp["taps"].reshape(-1, len(tp["layers"]) * H).float()  # all sequences' rows
    seq = json.load(open("/poc/runs/m27/dflash_seqs.json"))[tp["names"][0]]
    cfg = Qwen3Config.from_pretrained(snap)
    cfg._attn_implementation = "eager"
    cpu = DFlashDraftModel(cfg).to(torch.bfloat16).eval()
    cpu.load_state_dict(load_file(f"{snap}/model.safetensors"), strict=False)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    try:
        lm = ttnn.from_torch(
            head.T.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dr = DFlashDrafterPaged(mesh, snap, embed, lm, V, cmax=int(os.environ.get("CMAX", "8448")), tp=True)
        for P in [int(v) for v in os.environ.get("PS", "100,1000,2100,4100,8000").split(",")]:
            feats = rows.repeat(P // rows.shape[0] + 1, 1)[:P]
            tok = seq[60]
            dr.reset()
            dr.fill_prompt(feats)
            dr.stage_pass([], tok)
            dr.debug = []
            d = dr.run("pass", dr.pass_body)[1:4]
            dr.debug = None
            block = torch.tensor([tok] + [cpu.mask_token_id] * (cpu.block_size - 1))
            h = cpu(
                position_ids=torch.arange(P + cpu.block_size)[None],
                noise_embedding=torch.nn.functional.embedding(block, embed)[None],
                target_hidden=feats.to(torch.bfloat16)[None],
            )
            c = (h[0, 1:].float() @ head.float().T).argmax(-1).tolist()[:3]
            print(f"RESULT P={P}: dev {d} cpu {c}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
