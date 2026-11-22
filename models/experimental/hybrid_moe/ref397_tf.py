# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU reference for Qwen3.5-397B-A17B: the HF transformers decoder layers in bf16, streamed one layer at a time
from the FP8 checkpoint (block dequant), teacher-forced over the quality text set. Then compares the hybrid's saved
next-token log-probs (run_generate.py POC_TF_TEXTS) against it.

usage (in the image, no device): python ref397_tf.py <texts.json> <hybrid out dir> <out dir>
Writes <out dir>/ref_<name>.npz (reference top-1 ids, NLL per token) and <out dir>/compare.json.
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

sys.path.insert(0, "/poc")
from hybrid.state import dequant_fp8  # noqa: E402

SNAP = glob.glob("/hf/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/*")[0]
WMAP = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
P = "model.language_model."
torch.set_num_threads(int(os.environ.get("REF_THREADS", "16")))
_files = {}


def get(key):
    """bf16 tensor for a checkpoint key (FP8 weights dequantized with their block or per-channel scale)."""
    shard = WMAP[key]
    f = _files.get(shard) or _files.setdefault(shard, safe_open(f"{SNAP}/{shard}", "pt"))
    t = f.get_tensor(key)
    for sk in (key[: -len("weight")] + "weight_scale_inv", key[: -len("weight")] + "weight_scale"):
        if key.endswith(".weight") and sk in WMAP:  # block (Qwen3.5 FP8) or per-channel (compressed-tensors)
            return dequant_fp8(t, get_raw(sk))
    return t if t.dtype == torch.float32 else t.to(torch.bfloat16)  # A_log / GDN norm stay fp32, as stored


def get_raw(key):
    shard = WMAP[key]
    f = _files.get(shard) or _files.setdefault(shard, safe_open(f"{SNAP}/{shard}", "pt"))
    return f.get_tensor(key)


def load_layer(cfg, li):
    with torch.device("meta"):
        layer = M.Qwen3_5MoeDecoderLayer(cfg, li)
    layer = layer.to_empty(device="cpu").to(torch.bfloat16)
    pre = f"{P}layers.{li}."
    sd = {}
    for name, _ in layer.named_parameters():
        if name.startswith("mlp.experts."):
            continue
        sd[name] = get(pre + name)
    E = cfg.num_experts
    gu = torch.empty(E, 2 * cfg.moe_intermediate_size, cfg.hidden_size, dtype=torch.bfloat16)
    dn = torch.empty(E, cfg.hidden_size, cfg.moe_intermediate_size, dtype=torch.bfloat16)
    for e in range(E):
        ep = f"{pre}mlp.experts.{e}."
        gu[e, : cfg.moe_intermediate_size] = get(ep + "gate_proj.weight")
        gu[e, cfg.moe_intermediate_size :] = get(ep + "up_proj.weight")
        dn[e] = get(ep + "down_proj.weight")
    sd["mlp.experts.gate_up_proj"] = gu
    sd["mlp.experts.down_proj"] = dn
    # buffers that to_empty left uninitialised (e.g. GDN constants) are recomputed by the module's own init
    for name, p in layer.named_parameters():  # params stored in fp32 (A_log, GDN norm) keep fp32, like vLLM
        if name in sd and sd[name].dtype == torch.float32:
            p.data = p.data.float()
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    if unexpected or [m for m in missing if not m.startswith("mlp.experts")]:
        raise RuntimeError(f"layer {li}: missing {missing} unexpected {unexpected}")
    return layer.eval()


@torch.no_grad()
def main():
    texts_path, hyb_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    seqs = json.load(open(texts_path))
    names = list(seqs)
    ids = torch.tensor([seqs[n] for n in names])  # [B, T] (all the same length)
    B, T = ids.shape
    cfg = AutoConfig.from_pretrained(SNAP).text_config
    cfg._attn_implementation = "sdpa"
    x = torch.nn.functional.embedding(ids, get(P + "embed_tokens.weight"))
    rot = M.Qwen3_5MoeTextRotaryEmbedding(cfg)
    pos = torch.arange(T).view(1, 1, -1).expand(3, B, -1)
    pe = rot(x, pos)
    layers = range(int(os.environ.get("REF_LAYERS", cfg.num_hidden_layers)))
    for li in layers:
        t0 = time.time()
        layer = load_layer(cfg, li)
        t1 = time.time()
        x = layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos[0])
        print(
            f"layer {li} {cfg.layer_types[li]} load {t1 - t0:.1f}s fwd {time.time() - t1:.1f}s "
            f"|x| {x.float().norm(dim=-1).mean():.1f}",
            flush=True,
        )
        del layer
    norm = M.Qwen3_5MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = get(P + "norm.weight").float()
    x = norm(x.float())
    head = get("lm_head.weight").float()
    taus = sorted({f.split("_")[1] for f in os.listdir(hyb_dir) if f.startswith("lp_")})
    rows = []
    for b, n in enumerate(names):
        ref = torch.log_softmax(x[b] @ head.T, dim=-1)  # [T, V]
        nxt = ids[b, 1:]
        ref_nll = -ref[:-1].gather(1, nxt[:, None])[:, 0]
        np.savez(os.path.join(out_dir, f"ref_{n}.npz"), top1=ref.argmax(-1).numpy(), nll=ref_nll.numpy())
        for tau in taus:
            hyb = torch.from_numpy(np.load(os.path.join(hyb_dir, f"lp_{tau}_{n}.npy"))).float()
            hyb = torch.log_softmax(hyb, dim=-1)  # renormalise after the fp16 round trip
            hyb_nll = -hyb[:-1].gather(1, nxt[:, None])[:, 0]
            kl = (ref.exp() * (ref - hyb)).sum(-1)[:-1]
            top5r = ref.topk(5, -1).indices
            conf = (ref.exp().max(-1).values > 0.5)[:-1]  # positions where the reference is confident
            r = {
                "name": n,
                "tau": float(tau),
                "tokens": int(T - 1),
                "ref_nll": round(float(ref_nll.mean()), 4),
                "hyb_nll": round(float(hyb_nll.mean()), 4),
                "top1_agree": round(float((ref.argmax(-1) == hyb.argmax(-1))[:-1].float().mean()), 4),
                "hyb_top1_in_ref_top5": round(float((top5r == hyb.argmax(-1)[:, None]).any(-1)[:-1].float().mean()), 4),
                "kl_mean": round(float(kl.mean()), 4),
                "kl_p99": round(float(kl.quantile(0.99)), 4),
                "confident_frac": round(float(conf.float().mean()), 4),
                "top1_agree_confident": round(float((ref.argmax(-1) == hyb.argmax(-1))[:-1][conf].float().mean()), 4),
                "kl_confident": round(float(kl[conf].mean()), 4),
                "ref_acc": round(float((ref.argmax(-1)[:-1] == nxt).float().mean()), 4),
                "hyb_acc": round(float((hyb.argmax(-1)[:-1] == nxt).float().mean()), 4),
            }
            rows.append(r)
            print("CMP " + json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(out_dir, "compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
