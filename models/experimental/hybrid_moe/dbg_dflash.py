# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device DFlash drafter (dflash_tt) vs the CPU reference (dflash_ref) on the same context: the CPU target's taps
(runs/m27/ref_dflash/taps.pt). Per position: PCC of every drafter layer's block rows, and both draft lists.
usage (in the image): python dbg_dflash.py <out dir>"""

import glob
import json
import sys

sys.path.insert(0, "/poc")
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import Qwen3Config  # noqa: E402

import ttnn  # noqa: E402
import ref_m27 as R  # noqa: E402
from dflash_ref import DFlashDraftModel  # noqa: E402
from dflash_tt import DFlashDrafterTT  # noqa: E402


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@torch.no_grad()
def main():
    R.SNAP = glob.glob("/hf/hub/models--MiniMaxAI--MiniMax-M2.7/snapshots/*")[0]
    R.WMAP = json.load(open(f"{R.SNAP}/model.safetensors.index.json"))["weight_map"]
    snap = glob.glob("/hf/hub/models--nvidia--MiniMax-M2.7-DFlash/snapshots/*")[0]
    embed, head = R.get("model.embed_tokens.weight"), R.get("lm_head.weight")
    V, H = head.shape
    tp = torch.load("/poc/runs/m27/ref_dflash/taps.pt")
    seqs = json.load(open("/poc/runs/m27/dflash_seqs.json"))
    plens = json.load(open("/poc/runs/m27/dflash_prompt_lens.json"))
    name = tp["names"][0]
    seq, P = seqs[name], plens[name]
    feats = tp["taps"][0].reshape(len(seq), len(tp["layers"]), H)

    cfg = Qwen3Config.from_pretrained(snap)
    cfg._attn_implementation = "eager"
    cpu = DFlashDraftModel(cfg).to(torch.bfloat16).eval()
    cpu.load_state_dict(load_file(f"{snap}/model.safetensors"), strict=False)
    outs = []
    for L in cpu.layers:
        L.register_forward_hook(lambda m, a, kw_out: outs.append(kw_out.float()[0]))

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
    try:
        lm = ttnn.from_torch(
            head.T.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dr = DFlashDrafterTT(mesh, snap, embed, lm, V)
        bs = dr.bs

        def up(t):
            return ttnn.from_torch(
                t.reshape(1, 1, -1, H).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        dr.add_context([up(feats[:P, j]) for j in range(feats.shape[1])], list(range(P)))
        for p in range(P, P + 12):
            if p > P:  # one more context token (as the decode loop would add the accepted row)
                dr.add_context([up(feats[p - 1 : p, j]) for j in range(feats.shape[1])], [p - 1])
            outs.clear()
            block = torch.tensor([seq[p]] + [dr.mask_id] * (bs - 1))
            h = cpu(
                position_ids=torch.arange(p + bs)[None],
                noise_embedding=torch.nn.functional.embedding(block, embed)[None],
                target_hidden=feats[:p].reshape(1, p, -1),
            )
            cdr = (h[0, 1:].float() @ head.float().T).argmax(-1).tolist()
            dr.debug = []
            ddr = dr.draft(seq[p], bs - 1)
            dev = dr.debug
            dr.debug = None
            lay = " ".join(f"{pcc(a, b):.4f}" for a, b in zip(outs, dev[:-1]))
            print(
                f"p={p} layers [{lay}] final {pcc(h[0].float(), dev[-1]):.4f}  cpu {cdr[:4]} dev {ddr[:4]} "
                f"seq {seq[p + 1 : p + 5]}",
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
