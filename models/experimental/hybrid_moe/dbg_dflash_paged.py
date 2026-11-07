# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""dflash_tt.DFlashDrafterPaged (eager, then trace replays) vs the CPU reference drafter on the CPU target's taps.
usage (in the image): python dbg_dflash_traced.py <out dir>"""

import glob
import json
import sys
import time

sys.path.insert(0, "/poc")
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import Qwen3Config  # noqa: E402

import ttnn  # noqa: E402
import ref_m27 as R  # noqa: E402
from dflash_ref import DFlashDraftModel  # noqa: E402
from dflash_tt import DFlashDrafterPaged  # noqa: E402


@torch.no_grad()
def main():
    R.SNAP = glob.glob("/hf/hub/models--MiniMaxAI--MiniMax-M2.7/snapshots/*")[0]
    R.WMAP = json.load(open(f"{R.SNAP}/model.safetensors.index.json"))["weight_map"]
    snap = glob.glob("/hf/hub/models--nvidia--MiniMax-M2.7-DFlash/snapshots/*")[0]
    embed, head = R.get("model.embed_tokens.weight"), R.get("lm_head.weight")
    V, H = head.shape
    tp = torch.load("/poc/runs/m27/ref_dflash/taps.pt")
    name = tp["names"][0]
    seq = json.load(open("/poc/runs/m27/dflash_seqs.json"))[name]
    P = json.load(open("/poc/runs/m27/dflash_prompt_lens.json"))[name]
    feats = tp["taps"][0].reshape(len(seq), -1).float()
    cfg = Qwen3Config.from_pretrained(snap)
    cfg._attn_implementation = "eager"
    cpu = DFlashDraftModel(cfg).to(torch.bfloat16).eval()
    cpu.load_state_dict(load_file(f"{snap}/model.safetensors"), strict=False)

    def cpu_drafts(p):
        block = torch.tensor([seq[p]] + [cpu.mask_token_id] * (cpu.block_size - 1))
        h = cpu(
            position_ids=torch.arange(p + cpu.block_size)[None],
            noise_embedding=torch.nn.functional.embedding(block, embed)[None],
            target_hidden=feats[:p].to(torch.bfloat16)[None],
        )
        return (h[0, 1:].float() @ head.float().T).argmax(-1).tolist()

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
        import os

        dr = DFlashDrafterPaged(mesh, snap, embed, lm, V, cmax=4096, tp=os.environ.get("DFLASH_TP", "0") == "1")
        for mode in ("eager", "traced"):
            if mode == "traced":  # warm compile happened in the eager round
                dr.capture("pass", dr.pass_body)
            dr.reset()
            dr.fill_prompt(feats[:P])
            agree, n, t = 0, 0, []
            for p in range(P, P + 16):
                # pass: the previous position's row into the context (none at the first), then the block at p.
                # The context input rows come from the verify in the real loop; here: the CPU taps, uploaded.
                acc = [] if p == P else [p - 1]
                if acc:
                    x = torch.zeros(1, 1, dr.rows, feats.shape[1])
                    x[0, 0, 0] = feats[p - 1]
                    dr._put(dr.c_taps[dr.rows], x)
                dr.stage_pass(acc, seq[p])
                t0 = time.perf_counter()
                d = dr.run("pass", dr.pass_body)[1:]
                t.append(time.perf_counter() - t0)
                c = cpu_drafts(p)
                agree += sum(int(a == b) for a, b in zip(d[:3], c[:3]))
                n += 3
                if p < P + 4:
                    print(f"{mode} p={p} dev {d[:4]} cpu {c[:4]}", flush=True)
            print(
                f"RESULT {mode}: first-3 draft agreement {agree}/{n}; pass ms median "
                f"{1e3 * sorted(t)[len(t) // 2]:.1f}",
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
