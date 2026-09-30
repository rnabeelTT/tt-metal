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
    ap.add_argument(
        "--ref-check",
        type=int,
        default=0,
        help="steps on which the torch reference head also drafts from the captured hidden",
    )
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
    reference = None
    if args.ref_check:
        from models.autoports.qwen_qwen3_5_122b_a10b.tests import reference as ref
        from models.autoports.qwen_qwen3_5_122b_a10b.tests.reference_mtp import MtpReference

        text = ref.text_config()[1]
        reference = MtpReference(text, sd, embed_weight=sd["tok_embeddings.weight"], lm_head_weight=sd["output.weight"])
        logger.info("torch reference head loaded for the cross-check")

    # capture the final-normed hidden of every decode step (the LM-head input) by wrapping the model's final norm
    captured = {}
    orig_norm = model._final_norm_decode

    def capturing_norm(x):
        h = orig_norm(x)
        captured["h"] = ttnn.clone(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return h

    model._final_norm_decode = capturing_norm
    orig_bucket_hidden = model._run_bucket_hidden

    def capturing_bucket_hidden(bi, bucket, chunk_start):
        x = orig_bucket_hidden(bi, bucket, chunk_start)
        captured["x_all"] = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,bucket,H/TP] pre final norm
        return x

    model._run_bucket_hidden = capturing_bucket_hidden
    from models.tt_transformers.tt.common import Mode

    def head_rows(tok_ids_list, hidden_rows, positions):
        """Run the head on given (token, hidden row, position) triples, one decode row each (fills its KV cache)."""
        last_h = None
        for tk, hid_t, ps in zip(tok_ids_list, hidden_rows, positions):
            tokens_tt, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
                torch.tensor([tk]), ps, page_table
            )
            tokens_tt, cur_pos_tt, rot_idxs, pt_tt = (
                ttnn.to_device(x, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for x in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt)
            )
            _, last_h = head.decode_step(tokens_tt, hid_t, cur_pos_tt, rot_idxs, pt_tt)
        return last_h

    rep = ttnn.ReplicateTensorToMesh(mesh)
    H = gen.args.dim
    page_table = gen.page_table[:1]
    results = {}
    for name in args.prompts.split(","):
        text = PROMPTS[name]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        P = len(ids)
        t0 = time.time()
        # ONE greedy pass (eager, host argmax): prefill -> pred0, then decode step i consumes preds[i] at pos and yields
        # preds[i+1]; the hidden captured at step i is the model's state that produced preds[i+1], so the head drafts k
        # tokens extending preds[i+1] at pos+1 from it. Drafts are scored afterwards against the model's own later tokens.
        gen.reset()
        head.allocate_kv_cache(kv_shape)  # fresh MTP KV per prompt
        # the all-logits prefill runs the bucket program eagerly (the plain prefill replays a captured trace, which
        # bypasses the Python hook that captures the bucket hidden)
        logits0 = gen.prefill_forward(
            torch.tensor(ids).reshape(1, -1), page_table=page_table, prompt_lens=[P], return_all_logits=True
        )
        preds = [int(torch.argmax(logits0[0, -1]).item())]
        # the MTP head's own prefill: its layer must hold K/V for positions 1..P (vLLM runs the MTP layer over the whole
        # prompt): row t+1 = (token at t+1, the model's final-normed hidden at t); the token at position P is preds[0]
        x_all = captured.pop("x_all")
        h_all = model.norm(x_all, mode=Mode.PREFILL)  # replicated [1,1,bucket,H]
        ttnn.deallocate(x_all)
        rows_tok = [ids[t + 1] if t + 1 < P else preds[0] for t in range(P)]
        rows_hid = [ttnn.slice(h_all, (0, 0, t, 0), (1, 1, t + 1, H)) for t in range(P)]
        head_rows(rows_tok, rows_hid, [t + 1 for t in range(P)])
        for r in rows_hid:
            ttnn.deallocate(r)
        h_all_keep = h_all
        logger.info(f"[{name}] head prefilled over positions 1..{P} ({time.time()-t0:.0f}s)")
        if reference is not None:
            ref_cache = ref.DynamicCache()
            h_all_t = (
                ttnn.to_torch(ttnn.get_device_tensors(h_all_keep)[0]).float().reshape(-1, H)[:P].to(torch.bfloat16)
            )
            reference(
                torch.tensor([rows_tok]),
                h_all_t.reshape(1, P, H),
                torch.arange(1, P + 1).reshape(1, P),
                ref_cache,
                attention_mask=ref.causal_mask(1, P),
            )
        drafts_at = []
        pos = P
        n_steps = args.new + args.k + 1
        for i in range(n_steps):
            if i % 8 == 0:
                logger.info(f"[{name}] step {i}/{n_steps} pos {pos} ({time.time()-t0:.0f}s)")
            out = gen.decode_forward(
                torch.tensor([[preds[i]]]),
                torch.tensor([pos]),
                enable_trace=False,
                on_device_sampling=False,
                reload_inputs=(i == 0),
            )
            preds.append(int(torch.argmax(out[0]).item()))
            hidden = captured.pop("h")
            if i == 0:
                logger.info(
                    f"[{name}] first decode step done; hidden {tuple(hidden.shape)} {hidden.dtype}; drafting k={args.k}"
                )
            if i < args.new:
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
                    lg, h_next = head.decode_step(tokens_tt, hid, cur_pos_tt, rot_idxs, pt_tt)
                    lg = ttnn.to_torch(lg, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
                    d = int(lg.reshape(-1, lg.shape[-1])[0, : gen.args.vocab_size].argmax().item())
                    drafts.append(d)
                    cur_tok, cur_hid, cur_pos = d, h_next, cur_pos + 1
                drafts_at.append(drafts)
                if i == 0:
                    logger.info(f"[{name}] first drafts {drafts} ({time.time()-t0:.0f}s)")
                if reference is not None and i < args.ref_check:
                    h_t = (
                        ttnn.to_torch(ttnn.get_device_tensors(hidden)[0]).float().reshape(-1, H)[:1].to(torch.bfloat16)
                    )
                    r_logits, _ = reference(
                        torch.tensor([[preds[i + 1]]]), h_t.reshape(1, 1, H), torch.tensor([[pos + 1]]), ref_cache
                    )
                    r_draft = int(r_logits[0, -1].argmax().item())
                    logger.info(
                        f"[{name}] step {i}: hidden rms {h_t.float().pow(2).mean().sqrt():.3f} | device draft0 {drafts[0]} "
                        f"{tok.decode([drafts[0]])!r} | reference draft0 {r_draft} {tok.decode([r_draft])!r} | truth "
                        f"{preds[i + 2] if i + 2 < len(preds) else None} {tok.decode([preds[i + 2]]) if i + 2 < len(preds) else ''!r}"
                    )
            ttnn.deallocate(hidden)
            pos += 1
        logger.info(f"[{name}] P={P} greedy {len(preds)} tokens in {time.time()-t0:.0f}s: {tok.decode(preds[:24])!r}")
        accepted = []
        top1_agree = 0
        for i, drafts in enumerate(drafts_at):
            truth = preds[i + 2 : i + 2 + args.k]
            n_ok = 0
            for d, t in zip(drafts, truth):
                if d != t:
                    break
                n_ok += 1
            accepted.append(n_ok)
            top1_agree += int(len(truth) > 0 and drafts[0] == truth[0])
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
