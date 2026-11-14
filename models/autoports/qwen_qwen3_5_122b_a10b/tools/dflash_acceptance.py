# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash 2 acceptance on the device target (eager, greedy): the drafter runs on the host (tt/dflash_ref.py), the target's
verify pass is the MTP rows pass over [t, d_1 .. d_{bs-1}] with the 8 context layers' outputs read back, the GDN state
committed exactly (`commit_gdn`). The tokens must equal the plain greedy decode (bf16 near-ties aside); reports the mean
acceptance length (tokens per verify pass) per prompt.

    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.dflash_acceptance --new 128 [--block 16] [--prompts a,b]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.dflash_ref import DFlash2Ref
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import layer_forward_rows, rows_pos_masks_host, rows_width
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_verify import (
    commit_gdn,
    free_gdn_snapshot,
    greedy_accept,
    snapshot_gdn_state,
)
from models.autoports.qwen_qwen3_5_122b_a10b.tt.weights import resolve_checkpoint_dir

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPTS = {
    "code": "Write a Python class implementing an LRU cache with get and put in O(1), with docstrings and a short test.",
    "math": "Explain step by step how to solve the equation 3x^2 - 12x + 9 = 0, then verify the roots.",
    "prose": "Write a detailed product description for a stainless steel water bottle.",
    "chat": "What are some good strategies for staying focused while working from home?",
    "c": "Implement quicksort in C and explain the partition scheme.",
    "explain": "Explain how a transformer language model generates text, for a high-school student.",
}


def drafter_dir():
    snaps = glob.glob(
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub",
                     "models--incoai--Ornith-1.5-122B-DFlash2", "snapshots", "*")
    )
    assert snaps, "drafter checkpoint incoai/Ornith-1.5-122B-DFlash2 not in the HF cache"
    return snaps[0]


def load_target_tables(ckpt):
    from safetensors import safe_open

    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for name, key in (("embed", "model.language_model.embed_tokens.weight"), ("lm_head", "lm_head.weight")):
        with safe_open(os.path.join(ckpt, idx[key]), "pt") as f:
            out[name] = f.get_tensor(key)
    return out["embed"], out["lm_head"]


def rows_to_host(x, n, mesh, H):
    """[1,1,W,H or H/TP] device -> host float [n, H]."""
    if x.shape[-1] == H:
        t = ttnn.to_torch(ttnn.get_device_tensors(x)[0])
    else:
        t = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
    return t.float().reshape(-1, t.shape[-2], t.shape[-1])[0, :n, :H]


def verify_rows(model, tokens, positions, page_table_row, layer_ids, H, saved):
    """mtp_verify.verify_pass with the context layers' outputs read back. Returns (logits [R, V] host, feats [R, 8H])."""
    R = len(tokens)
    BW = rows_width(R)
    mesh = model.mesh_device
    dev = ttnn.DRAM_MEMORY_CONFIG
    pos_list = list(positions) + [positions[-1] + 1 + j for j in range(BW - R)]
    tok_list = list(tokens) + [0] * (BW - R)
    tokens_tt, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
        torch.tensor(tok_list, dtype=torch.int32), torch.tensor(pos_list, dtype=torch.int32), page_table_row.repeat(BW, 1)
    )
    tokens_tt, cur_pos_tt, rot_idxs, pt_tt = (
        ttnn.to_device(t, mesh, memory_config=dev) for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt)
    )
    cos, sin = model._rope_from_idxs(rot_idxs)
    masks = ttnn.to_device(rows_pos_masks_host(pos_list, BW), mesh, memory_config=dev)
    x = model.embd(model._decode_tokens_view(tokens_tt, BW))
    x = ttnn.to_memory_config(ttnn.reshape(x, (1, 1, BW, x.shape[-1])), dev)
    feats = {}
    gi = 0
    for li, layer in enumerate(model.layers):
        y = layer_forward_rows(
            layer, x, R, cos=cos, sin=sin, position_tensor=cur_pos_tt, page_table=pt_tt, saved=saved,
            rows_pos_masks=masks, gdn_index=gi,
        )
        if not layer.is_full_attention:
            gi += 1
        ttnn.deallocate(x)
        x = y
        if li in layer_ids:
            feats[li] = rows_to_host(x, R, mesh, H)
    ttnn.deallocate(masks)
    h = model._final_norm_decode(x)
    logits = ttnn.linear(h, model.lm_head_weight, compute_kernel_config=model._lm_head_ckc())
    lg = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
    lg = lg.reshape(-1, lg.shape[-1])[:R]
    for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt, cos, sin, logits, h):
        ttnn.deallocate(t)
    return lg, torch.cat([feats[i] for i in layer_ids], dim=-1)


def prefill_features(gen, prompt_ids, layer_ids, H):
    """Eager all-logits prefill with the context layers' outputs captured. Returns (tok0, feats [P, 8H])."""
    model = gen.model
    mesh = model.mesh_device
    P = len(prompt_ids)
    seg = model.PREFILL_SEGMENT
    chunks = []

    def hidden(bi, bucket, chunk_start):
        x = model.embd(bi["tok"])
        x = ttnn.reshape(x, (1, 1, bucket, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = model._apply_vision_merge(x, length=bucket)
        n = min(seg, P - chunk_start)
        got = {}
        for li, layer in enumerate(model.layers):
            if layer.is_full_attention:
                x_new = layer.forward(
                    x, cos=bi["cos"], sin=bi["sin"], mode="prefill", page_table=bi["full_pt"],
                    chunk_page_table=bi["chunk_pt"], chunk_start_idx=chunk_start, chunk_start_idx_tensor=bi["csi"],
                    valid_len=bi["len"],
                )
            else:
                x_new = layer.forward(x, mode="prefill", chunk_size=model.args.gdn_chunk_size, valid_len=bi["len"])
            ttnn.deallocate(x)
            x = x_new
            if li in layer_ids:
                got[li] = rows_to_host(x, n, mesh, H)
        chunks.append(torch.cat([got[i] for i in layer_ids], dim=-1))
        return x

    orig = model._run_bucket_hidden
    model._run_bucket_hidden = hidden
    try:
        gen.reset()
        logits0 = gen.prefill_forward(
            torch.tensor(prompt_ids).reshape(1, -1), page_table=gen.page_table[:1], prompt_lens=[P],
            return_all_logits=True,
        )
    finally:
        model._run_bucket_hidden = orig
    return int(torch.argmax(logits0[0, -1]).item()), torch.cat(chunks, dim=0)[:P]


def dflash_generate(gen, drafter, prompt_ids, max_new, block, log=None):
    model = gen.model
    H = gen.args.dim
    V = gen.args.vocab_size
    lids = drafter.layer_ids
    tok0, feats = prefill_features(gen, prompt_ids, lids, H)
    P = len(prompt_ids)
    drafter.reset()
    drafter.add_context(feats, torch.arange(P))
    out = [tok0]
    pos = P
    lens = []
    t_draft = t_ver = 0.0
    while len(out) < max_new:
        t0 = time.time()
        drafter.bs = block
        path, _, _ = drafter.propose(out[-1], pos)
        drafts = path.tolist()
        t1 = time.time()
        rows = [out[-1]] + drafts
        positions = [pos + i for i in range(len(rows))]
        pre = snapshot_gdn_state(model)
        saved = []
        lg, rfeats = verify_rows(model, rows, positions, gen.page_table[:1], lids, H, saved)
        a, nxt = greedy_accept(lg[:, :V], drafts)
        if a < len(drafts):
            commit_gdn(model, pre, saved, a + 1)
        for sink in saved:
            for key, t in sink.items():
                if key != "meta":
                    ttnn.deallocate(t)
        free_gdn_snapshot(pre)
        t_draft += t1 - t0
        t_ver += time.time() - t1
        drafter.add_context(rfeats[: a + 1], torch.tensor(positions[: a + 1]))
        out.extend(drafts[:a])
        out.append(nxt)
        pos += a + 1
        lens.append(a + 1)
        if log:
            log(f"round {len(lens)}: accepted {a} -> +{a + 1} (total {len(out)})")
    return out[:max_new], lens, t_draft, t_ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--no-greedy", action="store_true", help="skip the plain greedy reference")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    ckpt = str(resolve_checkpoint_dir())
    tok = AutoTokenizer.from_pretrained(ckpt)
    embed, lm_head = load_target_tables(ckpt)
    drafter = DFlash2Ref(drafter_dir(), embed, lm_head)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=256 << 20)
    gen = build_generator(MODEL_DIR, mesh, max_seq_len=4096, warm_decode_trace=False, warm_prefill_trace=False)
    results = {}
    for name in args.prompts.split(","):
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPTS[name]}], tokenize=False, add_generation_prompt=True,
            enable_thinking=bool(args.thinking),
        )
        ids = tok(text, add_special_tokens=False)["input_ids"]
        greedy = None
        if not args.no_greedy:
            t0 = time.time()
            greedy = gen.generate(ids, args.new, enable_trace=False, on_device_sampling=False)
            logger.info(f"[{name}] greedy {args.new} in {time.time() - t0:.1f}s")
        t0 = time.time()
        toks, lens, td, tv = dflash_generate(gen, drafter, ids, args.new, args.block)
        wall = time.time() - t0
        tau = sum(lens) / len(lens)
        agree = None if greedy is None else sum(int(a == b) for a, b in zip(greedy, toks))
        first = None if greedy is None else next((i for i, (a, b) in enumerate(zip(greedy, toks)) if a != b), None)
        results[name] = dict(tau=tau, rounds=len(lens), lens=lens, agree=agree, first_diff=first)
        print(
            f"DFLASH_ACCEPT {name}: block {args.block} rounds {len(lens)} tau {tau:.2f} | equal to greedy "
            f"{agree}/{args.new} first diff {first} | host draft {td:.1f}s verify {tv:.1f}s wall {wall:.1f}s",
            flush=True,
        )
        print(f"  dflash: {tok.decode(toks)!r}", flush=True)
        if greedy is not None and first is not None:
            print(f"  greedy: {tok.decode(greedy)!r}", flush=True)
    taus = [r["tau"] for r in results.values()]
    print(f"DFLASH_ACCEPT mean tau {sum(taus) / len(taus):.2f} over {len(taus)} prompts", flush=True)
    print("DFLASH_JSON " + json.dumps(results), flush=True)
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
