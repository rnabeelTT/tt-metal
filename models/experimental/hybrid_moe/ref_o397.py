# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU reference for Ornith-1.5-397B (Qwen3.5-MoE: 60 layers, 512 experts top-10, GDN linear attention + gated full
attention; FP8 checkpoint, block or per-channel scales): HF transformers decoder layers in bf16, streamed one layer at a
time, teacher-forced over a text set. The ref_m27.py of this model: records every position's routing (top-10 expert ids
and their weights, per layer) for routing-plan building and hit-rate simulation, and the reference's next-token
distribution (top-32 log-probs, NLL of the real next token) for fidelity comparisons (compare_m27.py's format).

usage (host venv, no device): python ref_o397.py <texts.json> <out dir>
Writes <out dir>/routing.npz (names, ids [B, L, T, 10] uint16, w [B, L, T, 10] fp16) and ref_<name>.npz per sequence.
env: REF_REPO (default ornith-ai/Ornith-1.5-397B-FP8), REF_LAYERS (default all 60), REF_THREADS (default 16),
HF_HUB (default $HF_HUB_CACHE or ~/.cache/huggingface/hub), REF_TAPS (layers whose outputs go to <out dir>/taps.pt).
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
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

REPO = os.environ.get("REF_REPO", "ornith-ai/Ornith-1.5-397B-FP8")
HUB = os.environ.get("HF_HUB") or os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
)
_SNAPS = sorted(glob.glob(f"{HUB}/models--{REPO.replace('/', '--')}/snapshots/*"))
SNAP = _SNAPS[0] if _SNAPS else None  # importers that resolve the snapshot themselves set SNAP / WMAP
WMAP = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"] if SNAP else None
P = "model.language_model."
torch.set_num_threads(int(os.environ.get("REF_THREADS", "16")))
_files = {}


def dequant_fp8(w, s, block=128):
    """FP8 [O, I] times its scale -> bf16: per output channel [O, 1] (compressed-tensors) or 128x128 block scale_inv."""
    O, I = w.shape
    if s.numel() == O and (s.dim() == 1 or s.shape[-1] == 1):
        return (w.float() * s.float().reshape(O, 1)).to(torch.bfloat16)
    sf = s.float().repeat_interleave(block, 0)[:O].repeat_interleave(block, 1)[:, :I]
    return (w.float() * sf).to(torch.bfloat16)


def get_raw(key):
    shard = WMAP[key]
    f = _files.get(shard) or _files.setdefault(shard, safe_open(f"{SNAP}/{shard}", "pt"))
    return f.get_tensor(key)


def get(key):
    """bf16 tensor for a checkpoint key (FP8 weights dequantized); fp32 tensors (A_log, GDN norm) stay fp32."""
    t = get_raw(key)
    if key.endswith(".weight"):
        for sk in (key + "_scale", key + "_scale_inv"):
            if sk in WMAP:
                return dequant_fp8(t, get_raw(sk))
    return t if t.dtype == torch.float32 else t.to(torch.bfloat16)


def config():
    cfg = AutoConfig.from_pretrained(SNAP).text_config
    cfg._attn_implementation = "sdpa"
    return cfg


def load_layer(cfg, li):
    with torch.device("meta"):
        layer = M.Qwen3_5MoeDecoderLayer(cfg, li)
    layer = layer.to_empty(device="cpu").to(torch.bfloat16)
    pre = f"{P}layers.{li}."
    sd = {}
    for name, _ in layer.named_parameters():
        if not name.startswith("mlp.experts."):
            sd[name] = get(pre + name)
    E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    gu = torch.empty(E, 2 * I, H, dtype=torch.bfloat16)
    dn = torch.empty(E, H, I, dtype=torch.bfloat16)
    for e in range(E):
        ep = f"{pre}mlp.experts.{e}."
        gu[e, :I] = get(ep + "gate_proj.weight")
        gu[e, I:] = get(ep + "up_proj.weight")
        dn[e] = get(ep + "down_proj.weight")
    sd["mlp.experts.gate_up_proj"] = gu
    sd["mlp.experts.down_proj"] = dn
    for name, p in layer.named_parameters():  # params stored in fp32 keep fp32, like vLLM
        if name in sd and sd[name].dtype == torch.float32:
            p.data = p.data.float()
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
    r_ids = np.zeros((B, L, T, K), dtype=np.uint16)  # 512 experts: uint8 is too small
    r_w = np.zeros((B, L, T, K), dtype=np.float16)
    x = torch.nn.functional.embedding(ids, get(P + "embed_tokens.weight"))
    pos = torch.arange(T).view(1, 1, -1).expand(3, B, -1)  # text-only mRoPE: the same position on all 3 sections
    pe = M.Qwen3_5MoeTextRotaryEmbedding(cfg)(x, pos)
    taps = [int(t) for t in os.environ.get("REF_TAPS", "").split(",") if t]
    tap_out = torch.zeros(B, T, len(taps), cfg.hidden_size, dtype=torch.bfloat16) if taps else None
    for li in range(L):
        t0 = time.time()
        layer = load_layer(cfg, li)
        t1 = time.time()
        gate_fwd = layer.mlp.gate.forward

        def rec(h, li=li, gate_fwd=gate_fwd):
            logits, w, idx = gate_fwd(h)
            r_ids[:, li] = idx.reshape(B, T, K).numpy().astype(np.uint16)
            r_w[:, li] = w.reshape(B, T, K).float().numpy().astype(np.float16)
            return logits, w, idx

        layer.mlp.gate.forward = rec
        x = layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos[0])
        if li in taps:
            tap_out[:, :, taps.index(li)] = x
        print(
            f"layer {li} {cfg.layer_types[li]} load {t1 - t0:.1f}s fwd {time.time() - t1:.1f}s "
            f"|x| {x.float().norm(dim=-1).mean():.1f}",
            flush=True,
        )
        del layer
        # partial results survive an interrupted run (the routing of the layers done so far)
        np.savez(os.path.join(out_dir, "routing.npz"), names=np.array(names), ids=r_ids, w=r_w, layers_done=li + 1)
    if taps:
        torch.save({"names": names, "layers": taps, "taps": tap_out}, os.path.join(out_dir, "taps.pt"))
    if L < cfg.num_hidden_layers:
        return
    norm = M.Qwen3_5MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = get(P + "norm.weight").float()
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
        print(f"REF {n} nll {float(nll.mean()):.3f} acc {float((lp[:-1].argmax(-1) == nxt).float().mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
