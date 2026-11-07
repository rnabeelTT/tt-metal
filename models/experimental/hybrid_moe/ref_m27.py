# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU reference for MiniMax-M2.7 (FP8 checkpoint, HF transformers decoder layers in bf16, streamed one layer at a
time with 128x128 block dequant), teacher-forced over a text set. Records every position's routing (top-8 expert ids
and their normalised weights, per layer) for routing-plan building and hit-rate simulation, and the reference's
next-token distribution (top-32 log-probs, NLL of the real next token) for later fidelity comparisons.

usage (host python_env, no device): python ref_m27.py <texts.json> <out dir>
Writes <out dir>/routing.npz (names, ids [B, L, T, 8] uint8, w [B, L, T, 8] fp16) and ref_<name>.npz per sequence.
env: REF_LAYERS (default all 62), REF_THREADS (default 16), HF_HUB (default ~/.cache/huggingface/hub),
REF_TAPS (comma list of layers whose outputs are saved as <out dir>/taps.pt [B, T, n, H] bf16, e.g. DFlash's taps).
"""

import glob
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.minimax_m2 import modeling_minimax_m2 as M

HUB = os.environ.get("HF_HUB", os.path.expanduser("~/.cache/huggingface/hub"))
SNAP = glob.glob(f"{HUB}/models--MiniMaxAI--MiniMax-M2.7/snapshots/*")[0]
WMAP = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
torch.set_num_threads(int(os.environ.get("REF_THREADS", "16")))
_files = {}


def dequant_fp8(w, s, block=128):
    """FP8 [O, I] with block scale_inv [ceil(O/128), ceil(I/128)] -> bf16 (as hybrid.state, without its imports)."""
    O, I = w.shape
    sf = s.float().repeat_interleave(block, 0)[:O].repeat_interleave(block, 1)[:, :I]
    return (w.float() * sf).to(torch.bfloat16)


def get_raw(key):
    shard = WMAP[key]
    f = _files.get(shard) or _files.setdefault(shard, safe_open(f"{SNAP}/{shard}", "pt"))
    return f.get_tensor(key)


def get(key):
    """bf16 tensor for a checkpoint key (FP8 weights dequantized with their block scale); fp32 tensors stay fp32."""
    t = get_raw(key)
    sk = key + "_scale_inv"
    if key.endswith(".weight") and sk in WMAP:
        return dequant_fp8(t, get_raw(sk))
    return t if t.dtype == torch.float32 else t.to(torch.bfloat16)


def config():
    cfg = AutoConfig.from_pretrained(SNAP)
    # transformers 5.12 drops the checkpoint's rotary_dim (64 of head_dim 128); the official modeling code rotates
    # only the first rotary_dim dims, so restore it as a partial rotary factor
    raw = json.load(open(f"{SNAP}/config.json"))
    cfg.rope_parameters = {**cfg.rope_parameters, "partial_rotary_factor": raw["rotary_dim"] / raw["head_dim"]}
    cfg._attn_implementation = "sdpa"
    return cfg


def load_layer(cfg, li):
    with torch.device("meta"):
        layer = M.MiniMaxM2DecoderLayer(cfg, li)
    layer = layer.to_empty(device="cpu").to(torch.bfloat16)
    pre = f"model.layers.{li}."
    sd = {}
    for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        sd[name] = get(pre + name)
    for p in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        sd[f"self_attn.{p}.weight"] = get(f"{pre}self_attn.{p}.weight")
    moe = pre + "block_sparse_moe."
    sd["mlp.gate.weight"] = get(moe + "gate.weight")
    sd["mlp.e_score_correction_bias"] = get_raw(moe + "e_score_correction_bias").float()
    E, I, H = cfg.num_local_experts, cfg.intermediate_size, cfg.hidden_size
    gu = torch.empty(E, 2 * I, H, dtype=torch.bfloat16)
    dn = torch.empty(E, H, I, dtype=torch.bfloat16)
    for e in range(E):
        ep = f"{moe}experts.{e}."
        gu[e, :I] = get(ep + "w1.weight")  # gate
        gu[e, I:] = get(ep + "w3.weight")  # up
        dn[e] = get(ep + "w2.weight")
    sd["mlp.experts.gate_up_proj"] = gu
    sd["mlp.experts.down_proj"] = dn
    for name, p in layer.named_parameters():  # params stored in fp32 keep fp32
        if name in sd and sd[name].dtype == torch.float32:
            p.data = p.data.float()
    layer.mlp.e_score_correction_bias = layer.mlp.e_score_correction_bias.float()
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    if unexpected or missing:
        raise RuntimeError(f"layer {li}: missing {missing} unexpected {unexpected}")
    return layer.eval()


@torch.no_grad()
def main():
    texts_path, out_dir = sys.argv[1:3]
    os.makedirs(out_dir, exist_ok=True)
    seqs = json.load(open(texts_path))
    names = list(seqs)
    ids = torch.tensor([seqs[n] for n in names])  # [B, T] (all the same length)
    B, T = ids.shape
    cfg = config()
    L = int(os.environ.get("REF_LAYERS", cfg.num_hidden_layers))
    K = cfg.num_experts_per_tok
    r_ids = np.zeros((B, L, T, K), dtype=np.uint8)
    r_w = np.zeros((B, L, T, K), dtype=np.float16)
    x = torch.nn.functional.embedding(ids, get("model.embed_tokens.weight"))
    pos = torch.arange(T).view(1, -1).expand(B, -1)
    pe = M.MiniMaxM2RotaryEmbedding(cfg)(x, pos)
    taps = [int(t) for t in os.environ.get("REF_TAPS", "").split(",") if t]  # layers whose outputs to save
    tap_out = torch.zeros(B, T, len(taps), cfg.hidden_size, dtype=torch.bfloat16) if taps else None
    for li in range(L):
        t0 = time.time()
        layer = load_layer(cfg, li)
        t1 = time.time()
        gate_fwd = layer.mlp.gate.forward

        def rec(h, bias, li=li, gate_fwd=gate_fwd):
            logits, w, idx = gate_fwd(h, bias)
            r_ids[:, li] = idx.reshape(B, T, K).numpy().astype(np.uint8)
            r_w[:, li] = w.reshape(B, T, K).float().numpy().astype(np.float16)
            return logits, w, idx

        layer.mlp.gate.forward = rec
        x = layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos)
        if li in taps:
            tap_out[:, :, taps.index(li)] = x
        print(
            f"layer {li} load {t1 - t0:.1f}s fwd {time.time() - t1:.1f}s |x| {x.float().norm(dim=-1).mean():.1f}",
            flush=True,
        )
        del layer
    np.savez(os.path.join(out_dir, "routing.npz"), names=np.array(names), ids=r_ids, w=r_w)
    if taps:
        torch.save({"names": names, "layers": taps, "taps": tap_out}, os.path.join(out_dir, "taps.pt"))
    if L < cfg.num_hidden_layers:
        return
    norm = M.MiniMaxM2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = get("model.norm.weight").float()
    x = norm(x.float())
    head = get("lm_head.weight").float()
    for b, n in enumerate(names):
        lp = torch.log_softmax(x[b] @ head.T, dim=-1)  # [T, V]
        nxt = ids[b, 1:]
        nll = -lp[:-1].gather(1, nxt[:, None])[:, 0]
        top = lp.topk(32, -1)
        np.savez(
            os.path.join(out_dir, f"ref_{n}.npz"),
            top_ids=top.indices.numpy().astype(np.int32),
            top_lp=top.values.numpy().astype(np.float16),
            nll=nll.numpy(),
            acc=(lp[:-1].argmax(-1) == nxt).numpy(),
        )
        print(
            f"REF {n} nll {float(nll.mean()):.3f} acc {float((lp[:-1].argmax(-1) == nxt).float().mean()):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
