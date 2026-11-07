# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""DFlash speculative decoding on the hybrid MiniMax-M2.7 (first version: eager, host acceptance).

Per pass: the drafter (dflash_tt.DFlashDrafterTT) drafts k tokens from [t, mask...]; one verify pass of the target over
the k + 1 rows [t, d_1..d_k] (the port's multi-row decode, the hybrid MoE per layer); greedy acceptance of the longest
matching prefix; the accepted rows' target features (the outputs of the drafter's tap layers) extend the drafter's
context. Also runs plain greedy decode on the same prompt for the token-match check.

env: POC_PROMPT_SET / POC_PROMPTS / POC_PROMPT_START / POC_NEW_TOKENS as run_generate.py, DFLASH_SNAPSHOT,
DFLASH_K (drafts per pass, default 3). usage (in the image): python run_dflash.py <out dir>
"""

import glob
import json
import os
import sys
import time

sys.path.insert(0, "/poc")
import hybrid  # noqa: E402

HYB = hybrid.install()

import torch  # noqa: E402

import ttnn  # noqa: E402
from dflash_tt import DFlashDrafterTT  # noqa: E402
from hybrid.patch import PORT, _port  # noqa: E402

rows = _port("mtp_rows")
K = int(os.environ.get("DFLASH_K", "3"))
NEW = int(os.environ.get("POC_NEW_TOKENS", "128"))


def replicated(mesh, x, H, n=None):
    """A device hidden ([1, 1, rows, H] replicated or [.., H / tp] fractured) -> replicated [1, 1, n, H] bf16."""
    t = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
    t = t.reshape(-1, t.shape[-1])
    if t.shape[-1] != H:  # replicated: every die's copy was concatenated
        t = t[:, :H]
    n = n or t.shape[0]
    return ttnn.from_torch(
        t[:n].reshape(1, 1, n, H).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def is_replicated_hidden(x, H):
    return int(ttnn.get_device_tensors(x)[0].shape[-1]) == H


def verify_with_taps(model, tokens, positions, page_table_row, taps):
    """mtp_verify.verify_pass with the outputs of the `taps` layers kept (device tensors, the R real rows)."""
    import ttnn as tt

    R = len(tokens)
    BW = rows.rows_width(R)
    mesh, dev = model.mesh_device, tt.DRAM_MEMORY_CONFIG
    pos_list = list(positions) + [positions[-1] + 1 + j for j in range(BW - R)]
    tok_list = list(tokens) + [0] * (BW - R)
    tokens_tt, cur_pos_tt, rot_idxs, pt_tt = model.prepare_decode_inputs_host(
        torch.tensor(tok_list, dtype=torch.int32),
        torch.tensor(pos_list, dtype=torch.int32),
        page_table_row.repeat(BW, 1),
    )
    tokens_tt, cur_pos_tt, rot_idxs, pt_tt = (
        tt.to_device(t, mesh, memory_config=dev) for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt)
    )
    cos, sin = model._rope_from_idxs(rot_idxs)
    masks = tt.to_device(rows.rows_pos_masks_host(pos_list, BW), mesh, memory_config=dev)
    x = model.embd(model._decode_tokens_view(tokens_tt, BW))
    x = tt.to_memory_config(tt.reshape(x, (1, 1, BW, x.shape[-1])), dev)
    kept = {}
    for li, layer in enumerate(model.layers):
        y = rows.layer_forward_rows(
            layer,
            x,
            R,
            cos=cos,
            sin=sin,
            position_tensor=cur_pos_tt,
            page_table=pt_tt,
            rows_pos_masks=masks,
            gdn_index=0,
        )
        if li in taps:
            kept[li] = tt.clone(y, memory_config=dev)
        tt.deallocate(x)
        x = y
    tt.deallocate(masks)
    h = model._final_norm_decode(x)
    logits = tt.linear(h, model.lm_head_weight, compute_kernel_config=model._lm_head_ckc())
    lg = tt.to_torch(logits, mesh_composer=tt.ConcatMeshToTensor(mesh, dim=-1)).float()
    for t in (tokens_tt, cur_pos_tt, rot_idxs, pt_tt, cos, sin, h, logits, x):
        tt.deallocate(t)
    return lg.reshape(-1, lg.shape[-1])[:R], [kept[li] for li in taps]


DEBUG = os.environ.get("DFLASH_DEBUG") == "1"
# The greedy reference (traced) is opt-in: an eager DFlash pass after a traced replay in the same process hung.
GREEDY = os.environ.get("DFLASH_GREEDY", "0") == "1"
TAPS = "/poc/runs/m27/ref_dflash/taps.pt"


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def tap_check(mesh, captured, taps, H, P, model, page_table, drafter):
    """DFLASH_DEBUG: the device target's tap features vs the CPU reference's, for the prefill and a teacher-forced
    verify pass of the next K + 1 reference tokens; then drafts from device-fed vs CPU-fed drafter context."""
    tp = torch.load(TAPS)
    name = tp["names"][0]
    seq = json.load(open("/poc/runs/m27/dflash_seqs.json"))[name]
    ref = tp["taps"][0].reshape(len(seq), len(taps), H).float()
    for j, li in enumerate(taps):
        x = captured[li][0]
        shapes = [tuple(t.shape) for t in ttnn.get_device_tensors(x)]
        t = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        t = t.reshape(-1, t.shape[-1])
        dies = [t[:P, d * H : (d + 1) * H] for d in range(t.shape[-1] // H)] if t.shape[-1] >= H else []
        dev = replicated_host(t, H)[:P]
        print(
            f"TAP prefill layer {li} per-die shapes {shapes} -> host {tuple(t.shape)}  pcc {_pcc(dev, ref[:P, j]):.4f}"
            f"  per-row min pcc {min(_pcc(dev[r], ref[r, j]) for r in range(P)):.4f}"
            f"  dies equal {[bool(torch.allclose(dies[0], d)) for d in dies[1:]]}",
            flush=True,
        )
    R = K + 1
    lg, kept = verify_with_taps(model, seq[P : P + R], list(range(P, P + R)), page_table, taps)
    for j, li in enumerate(taps):
        t = ttnn.to_torch(kept[j], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        dev = replicated_host(t.reshape(-1, t.shape[-1]), H)[:R]
        print(
            f"TAP verify layer {li} shapes {[tuple(x.shape) for x in ttnn.get_device_tensors(kept[j])]} "
            f"rows pcc {[round(_pcc(dev[r], ref[P + r, j]), 4) for r in range(R)]}",
            flush=True,
        )
    print(f"TAP verify argmax {lg[:, : model.args.vocab_size].argmax(-1).tolist()} seq {seq[P + 1 : P + R + 1]}")
    up = lambda t: ttnn.from_torch(
        t.reshape(1, 1, -1, H).to(torch.bfloat16),
        dtype=ttnn.bfloat16,  # noqa: E731
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    drafter.reset()
    drafter.add_context([up(ref[:P, j]) for j in range(len(taps))], list(range(P)))
    print(f"TAP drafts from CPU taps {drafter.draft(seq[P], 7)}")
    drafter.reset()
    drafter.add_context([replicated(mesh, captured[li][0], H, P) for li in taps], list(range(P)))
    print(f"TAP drafts from device taps {drafter.draft(seq[P], 7)}  seq {seq[P + 1 : P + 8]}", flush=True)


def replicated_host(t, H):
    return t[:, :H] if t.shape[-1] != H else t


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    res = {"runs": []}
    try:
        import ref_m27  # host access to the target checkpoint (embedding table for the drafter's block rows)

        ref_m27.SNAP = glob.glob("/hf/hub/models--MiniMaxAI--MiniMax-M2.7/snapshots/*")[0]
        ref_m27.WMAP = json.load(open(f"{ref_m27.SNAP}/model.safetensors.index.json"))["weight_map"]
        embed = ref_m27.get("model.embed_tokens.weight")
        snap = (
            os.environ.get("DFLASH_SNAPSHOT") or glob.glob("/hf/hub/models--nvidia--MiniMax-M2.7-DFlash/snapshots/*")[0]
        )
        # The drafter's weights go on the device BEFORE the target captures its decode trace: buffers allocated after a
        # capture can reuse the trace's freed intermediates, which every replay then overwrites.
        drafter = DFlashDrafterTT(mesh, snap, embed, None, None)
        gen = _port("generator").build_generator(
            f"models/autoports/{PORT}",
            mesh,
            max_seq_len=int(os.environ.get("POC_MAX_SEQ", "4096")),
            warm_decode_trace=True,
            warm_prefill_trace=False,
        )
        model, tok = gen.model, gen.tokenizer
        H, V = gen.args.dim, gen.args.vocab_size
        drafter.lm_head, drafter.vocab = model.lm_head_weight, V
        taps = drafter.taps
        import prompts_v2

        start = int(os.environ.get("POC_PROMPT_START", "0"))
        prompts = prompts_v2.HELDOUT[start : start + int(os.environ.get("POC_PROMPTS", "2"))]
        page_table = gen.page_table[:1]
        if DEBUG:  # the CPU reference's first sequence (its taps are in runs/m27/ref_dflash/taps.pt)
            name = torch.load(TAPS)["names"][0]
            prompts = [
                json.load(open("/poc/runs/m27/dflash_seqs.json"))[name][
                    : json.load(open("/poc/runs/m27/dflash_prompt_lens.json"))[name]
                ]
            ] + prompts[:1]
        for p_text in prompts:
            if isinstance(p_text, list):
                ids = p_text
            else:
                chat = tok.apply_chat_template(
                    [{"role": "user", "content": p_text}],
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=False,
                )
                ids = tok.encode(chat, add_special_tokens=False)
            P = len(ids)
            # ---- reference: plain greedy decode (eager) -----------------------------------------------------------
            # ---- DFlash -----------------------------------------------------------------------------------------
            gen.reset()
            drafter.reset()
            captured = {li: [] for li in taps}
            orig = {}
            for li in taps:
                layer = model.layers[li]
                orig[li] = layer.forward

                def hook(*a, _f=orig[li], _li=li, **kw):
                    y = _f(*a, **kw)
                    captured[_li].append(ttnn.clone(y, memory_config=ttnn.DRAM_MEMORY_CONFIG))  # the model frees y
                    return y

                layer.forward = hook
            try:
                logits0 = gen.prefill_forward(torch.tensor(ids).reshape(1, -1), page_table=page_table, prompt_lens=[P])
            finally:
                for li in taps:
                    model.layers[li].forward = orig[li]
            if any(len(captured[li]) != 1 for li in taps):
                raise RuntimeError(f"prefill tap capture: {[len(captured[li]) for li in taps]} chunks (one expected)")
            if isinstance(p_text, list):
                tap_check(mesh, captured, taps, H, P, model, page_table, drafter)
                drafter.reset()
            drafter.add_context([replicated(mesh, captured[li][0], H, P) for li in taps], list(range(P)))
            for li in taps:
                ttnn.deallocate(captured[li][0])
            t = int(torch.as_tensor(logits0).reshape(-1, logits0.shape[-1])[-1, :V].argmax())
            out, passes, acc_hist = [], 0, [0] * (K + 1)
            td = tv = tc = 0.0
            HYB.reset_stats()
            t1 = time.time()
            pos = P
            while len(out) < NEW:
                a0 = time.time()
                drafts = drafter.draft(t, K)
                a1 = time.time()
                lg, kept = verify_with_taps(model, [t] + drafts, list(range(pos, pos + K + 1)), page_table, taps)
                a2 = time.time()
                top = lg[:, :V].argmax(-1).tolist()
                a = 0
                while a < K and drafts[a] == top[a]:
                    a += 1
                acc_hist[a] += 1
                if DEBUG and passes < 20:
                    print(f"PASS {passes} pos {pos} t {t} drafts {drafts} top {top} accept {a}", flush=True)
                drafter.add_context([replicated(mesh, kv, H, a + 1) for kv in kept], list(range(pos, pos + a + 1)))
                for kv in kept:
                    ttnn.deallocate(kv)
                a3 = time.time()
                out += [t] + drafts[:a]
                t = top[a]
                pos += a + 1
                passes += 1
                td, tv, tc = td + a1 - a0, tv + a2 - a1, tc + a3 - a2
            dt = time.time() - t1
            out = out[:NEW]
            # ---- reference: plain greedy decode (traced), with the drafter's context cache freed first (the replays
            # overwrite buffers allocated after the capture: the cache could sit in one of them)
            drafter.reset()
            gen.reset()
            t0 = time.time()
            greedy = (
                []
                if isinstance(p_text, list) or not GREEDY
                else [int(t) for t in gen.generate(list(ids), NEW, enable_trace=True)]
            )
            t_greedy = time.time() - t0
            match = next((i for i, (x, y) in enumerate(zip(out, greedy)) if x != y), len(out))
            r = {
                "prompt": str(p_text)[:80],
                "prompt_tokens": P,
                "passes": passes,
                "tokens_per_pass": round(len(out) / passes, 3),
                "accept_hist": acc_hist,
                "dflash_ms_per_token": round(1e3 * dt / len(out), 1),
                "greedy_ms_per_token_traced_incl_prefill": round(1e3 * t_greedy / NEW, 1) if greedy else None,
                "ms_per_pass": {
                    "draft": round(1e3 * td / passes, 1),
                    "verify": round(1e3 * tv / passes, 1),
                    "context": round(1e3 * tc / passes, 1),
                },
                "first_mismatch_vs_greedy": match if greedy else None,
                "hyb_per_pass": {
                    k: round(HYB.st[k] / passes, 2)
                    for k in ("dec_picks", "dec_hits", "host_experts", "t_read", "t_host", "t_up", "t_route", "t_craft")
                },
                "text": tok.decode(out),
            }
            print("DFLASH " + json.dumps({k: v for k, v in r.items() if k != "text"}), flush=True)
            res["runs"].append(r)
            json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
