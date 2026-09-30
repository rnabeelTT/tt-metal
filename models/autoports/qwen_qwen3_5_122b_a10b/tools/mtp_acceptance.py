# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP acceptance-rate measurement (offline, eager): expected accepted drafts per pass BEFORE building the verify path.

For every prompt: prefill, then the main model's greedy continuation step by step (eager, host argmax) while the
final-normed hidden of every step is captured; at every step the MTP head drafts k tokens (chained, greedy) from the
model's token + hidden, and the drafts are scored against the model's own later tokens (the greedy verify outcome).
Reports per prompt the mean number of accepted drafts per pass and the histogram, plus the MTP top-1 agreement.

    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.mtp_acceptance --k 4 --new 64 --out acceptance.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.generator import build_generator
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp import Qwen35MtpHead
from models.autoports.qwen_qwen3_5_122b_a10b.tt.weights import resolve_checkpoint_dir

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPTS = {
    "prose": "The old lighthouse keeper had watched the sea for forty years, and in all that time he had never seen",
    "code": 'def merge_sorted_lists(a, b):\n    """Merge two sorted lists into one sorted list."""\n    i, j = 0, 0\n    out = []\n',
    "json": 'Return the following as JSON with keys name, age, city, hobbies (a list of 5 strings).\nName: Ada Lovelace, age 36, city London.\n{"name": "Ada Lovelace",',
    "chat": "<|im_start|>user\nExplain in three short paragraphs why the sky is blue.<|im_end|>\n<|im_start|>assistant\n",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--prompts", default="prose,code,json,chat")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(resolve_checkpoint_dir()))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=24576,
        trace_region_size=int(os.environ.get("QWEN35_TRACE_REGION", 256 * 1024 * 1024)),
    )
    gen = build_generator(MODEL_DIR, mesh, max_seq_len=args.max_seq_len, warm_decode_trace=False)
    model = gen.model
    sd = gen.args.load_state_dict()
    head = Qwen35MtpHead(model, sd)
    kv_shape = None
    for layer in model.layers:
        if layer.is_full_attention and getattr(layer.attention, "paged_k", None) is not None:
            kv_shape = tuple(layer.attention.paged_k.shape)
            break
    assert kv_shape is not None, "main attention KV caches not bound"
    head.allocate_kv_cache(kv_shape)
    logger.info(f"MTP head ready (layer {head.layer_idx}); kv {kv_shape}")

    # capture the final-normed hidden of every decode step (the LM-head input) by wrapping the model's final norm
    captured = {}
    orig_norm = model._final_norm_decode

    def capturing_norm(x):
        h = orig_norm(x)
        captured["h"] = ttnn.clone(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return h

    model._final_norm_decode = capturing_norm
    rep = ttnn.ReplicateTensorToMesh(mesh)
    H = gen.args.dim
    page_table = gen.page_table[:1]
    results = {}
    for name in args.prompts.split(","):
        text = PROMPTS[name]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        P = len(ids)
        t0 = time.time()
        # 1) the model's own greedy continuation, eager, host logits (also fills the KV / GDN state step by step)
        preds = gen.generate(ids, args.new + args.k, enable_trace=False, on_device_sampling=False)
        logger.info(f"[{name}] P={P} greedy {len(preds)} tokens in {time.time()-t0:.0f}s: {tok.decode(preds[:24])!r}")
        # 2) replay the same continuation while drafting: reset, prefill, then decode step i feeds preds[i] (teacher
        #    forcing on the model's own tokens) and drafts k from (preds[i+1]... ) -- the hidden captured at step i is the
        #    state that produced preds[i+1]; the head extends token preds[i+1] with that hidden.
        gen.reset()
        gen.prefill_forward(torch.tensor(ids).reshape(1, -1), page_table=page_table, prompt_lens=[P])
        head.allocate_kv_cache(kv_shape)  # fresh MTP KV per prompt
        accepted = []
        top1_agree = 0
        pos = P
        for i in range(args.new):
            tok_i = preds[i]
            gen.decode_forward(
                torch.tensor([[tok_i]]),
                torch.tensor([pos]),
                enable_trace=False,
                on_device_sampling=False,
                reload_inputs=True,
            )
            hidden = captured["h"]  # replicated [1,1,B,H] -> the model's state after consuming preds[i] at pos
            # drafts extend preds[i+1] (the model's next token) at position pos+1
            cur_tok, cur_hid, cur_pos = preds[i + 1], hidden, pos + 1
            drafts = []
            for j in range(args.k):
                tokens_tt, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
                    torch.tensor([cur_tok]), cur_pos, page_table
                )
                tokens_tt, cur_pos_tt, rot_idxs, pt_tt = (
                    ttnn.to_device(x, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    for x in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt)
                )
                hid = cur_hid
                if hid.shape[2] != 1:
                    hid = ttnn.slice(hid, (0, 0, 0, 0), (1, 1, 1, H))
                logits, h_next = head.decode_step(tokens_tt, hid, cur_pos_tt, rot_idxs, pt_tt)
                lg = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
                d = int(lg.reshape(-1, lg.shape[-1])[0, : gen.args.vocab_size].argmax().item())
                drafts.append(d)
                cur_tok, cur_hid, cur_pos = d, h_next, cur_pos + 1
            truth = preds[i + 2 : i + 2 + args.k]
            n_ok = 0
            for d, t in zip(drafts, truth):
                if d != t:
                    break
                n_ok += 1
            accepted.append(n_ok)
            top1_agree += int(len(truth) > 0 and drafts[0] == truth[0])
            pos += 1
            ttnn.deallocate(hidden)
        hist = [accepted.count(v) for v in range(args.k + 1)]
        mean_acc = sum(accepted) / len(accepted)
        results[name] = {
            "prompt_tokens": P,
            "steps": len(accepted),
            "k": args.k,
            "mean_accepted": mean_acc,
            "expected_tokens_per_pass": 1 + mean_acc,
            "histogram": hist,
            "first_draft_top1_agreement": top1_agree / len(accepted),
            "continuation": tok.decode(preds[: args.new]),
        }
        print(
            f"MTP_ACCEPT {name}: k={args.k} mean accepted {mean_acc:.2f} -> {1 + mean_acc:.2f} tokens/pass; "
            f"first-draft agreement {top1_agree / len(accepted):.2f}; histogram {hist}",
            flush=True,
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
