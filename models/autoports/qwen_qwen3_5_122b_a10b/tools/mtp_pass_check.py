# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify pass vs plain eager decode on the full model, row by row (no traces anywhere, no MTP head).

Phase 1: eager all-logits prefill, then `steps` eager decode steps (greedy), snapshotting every GDN layer's state before
each step. Phase 2: for every step i and every R in `--rows`, restore the state before step i and run the verify pass
over the greedy tokens t_i .. t_{i+R-1}; row j must reproduce the decode logits of step i+j (PCC, argmax).
"""
import argparse
import time

import torch
from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tools.mtp_acceptance import PROMPTS
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_verify import (
    free_gdn_snapshot,
    restore_gdn_state,
    snapshot_gdn_state,
    verify_pass,
)
from models.autoports.qwen_qwen3_5_122b_a10b.tt.weights import resolve_checkpoint_dir
from models.common.utility_functions import comp_pcc

MODEL_DIR = "models/autoports/qwen_qwen3_5_122b_a10b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="prose")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--rows", default="1,2,5")
    ap.add_argument("--layers", default=None)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(resolve_checkpoint_dir()))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=256 * 1024 * 1024
    )
    layer_indices = [int(s) for s in args.layers.split(",")] if args.layers else None
    gen = build_generator(
        MODEL_DIR,
        mesh,
        max_seq_len=4096,
        warm_decode_trace=False,
        warm_prefill_trace=False,
        layer_indices=layer_indices,
    )
    model = gen.model
    V = gen.args.vocab_size
    page_table = gen.page_table[:1]
    ids = tok(PROMPTS[args.prompt], add_special_tokens=False)["input_ids"]
    P = len(ids)
    Rs = [int(r) for r in args.rows.split(",")]
    S = args.steps + max(Rs)  # decode steps needed so every check has its future rows

    # ---- phase 1: eager prefill + eager decode, snapshots before each step
    gen.reset()
    logits0 = gen.prefill_forward(
        torch.tensor(ids).reshape(1, -1), page_table=page_table, prompt_lens=[P], return_all_logits=True
    )
    toks = [int(torch.argmax(logits0[0, -1]).item())]
    dec_logits = []
    snaps = []
    t0 = time.time()
    for i in range(S):
        snaps.append(snapshot_gdn_state(model))
        out = gen.decode_forward(
            torch.tensor([[toks[i]]]),
            torch.tensor([P + i]),
            enable_trace=False,
            on_device_sampling=False,
            reload_inputs=True,
        )
        lg = out[0].float()[:V]
        dec_logits.append(lg)
        toks.append(int(lg.argmax().item()))
    logger.info(f"decode {S} steps in {time.time() - t0:.1f}s: {tok.decode(toks)!r}")

    # ---- phase 2: verify pass from the state before step i over R greedy tokens; rows vs the decode logits
    results = {}
    for R in Rs:
        agree, pccs, worst = 0, [], None
        for i in range(args.steps):
            restore_gdn_state(model, snaps[i])
            rows = toks[i : i + R]
            h, lg_tt, _ = verify_pass(model, rows, [P + i + j for j in range(R)], page_table)
            lg = ttnn.to_torch(lg_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
            lg = lg.reshape(-1, lg.shape[-1])[:R, :V]
            ttnn.deallocate(lg_tt)
            ttnn.deallocate(h)
            for j in range(R):
                _, pcc = comp_pcc(dec_logits[i + j], lg[j], 0.99)
                pccs.append(pcc)
                same = int(lg[j].argmax().item()) == toks[i + j + 1]
                agree += int(same)
                if not same:
                    d = dec_logits[i + j]
                    top = d.topk(2)
                    msg = (
                        f"R={R} step {i} row {j}: decode argmax {toks[i + j + 1]} ({tok.decode([toks[i + j + 1]])!r}, "
                        f"margin {top.values[0] - top.values[1]:.3f}) verify argmax {int(lg[j].argmax())} "
                        f"({tok.decode([int(lg[j].argmax())])!r}) PCC {pcc:.5f}"
                    )
                    logger.info(msg)
                    worst = msg
        results[R] = (agree, args.steps * R, min(pccs), sum(pccs) / len(pccs))
        print(
            f"PASS_CHECK R={R}: argmax agreement {agree}/{args.steps * R}, PCC min {min(pccs):.5f} mean {sum(pccs) / len(pccs):.5f}",
            flush=True,
        )
    for s in snaps:
        free_gdn_snapshot(s)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
