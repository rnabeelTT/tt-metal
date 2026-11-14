# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU reference for thinkingmachines/Inkling-Small (BF16 checkpoint, HF transformers 5.17 `InklingDecoderLayer`s
streamed one layer at a time), teacher-forced over a text set. Same outputs as ref_m27.py:

  routing.npz      names, layers [Lm] (the MoE layer indices, 2..41), ids [B, Lm, T, 6] uint8, w [B, Lm, T, 6] fp16
                   (the routed weights as the model applies them: after the 8-way logsigmoid normalisation with the
                   2 shared experts, x route_scale x global_scale), shared [B, Lm, T, 2] fp16 (the shared gammas)
  ref_<name>.npz   top-32 log-probs / ids, NLL of the real next token, argmax==next
  taps.pt          (REF_TAPS) residual stream after the listed layers, e.g. DSpark's tap layers
  gold.pt          (REF_GOLD=1) the residual stream entering every layer and the final hidden state, for sequence 0:
                   the fixtures the TT port's layer tests compare against

Long texts are processed in query chunks (REF_CHUNK, default 2048) against a per-layer DynamicCache, which is how
the HF model itself does chunked prefill (sliding-window + full KV, the 4 sconv states, relative-bias offsets).
REF_CHUNK=0 runs each layer over the whole sequence at once (used to cross-check the chunked path).

usage (host venv with torch cpu + transformers>=5.17, no device): python ref_inkling.py <texts.json> <out dir>
env: REF_LAYERS (default all 42), REF_THREADS (default 16), HF_HUB, REF_TAPS, REF_GOLD, REF_CHUNK
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
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.inkling import modeling_inkling as M

REPO = os.environ.get("INKLING_REPO", "thinkingmachines/Inkling-Small")
HUB = os.environ.get("HF_HUB") or os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
)
_SNAPS = sorted(glob.glob(f"{HUB}/models--{REPO.replace('/', '--')}/snapshots/*"))
SNAP = _SNAPS[0] if _SNAPS else None
WMAP = (
    json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
    if SNAP and os.path.exists(f"{SNAP}/model.safetensors.index.json")
    else None
)
torch.set_num_threads(int(os.environ.get("REF_THREADS", "16")))
_files = {}


def get(key, keep_fp32=True):
    """Checkpoint tensor; bf16 unless stored in fp32 (router bias / global scales stay fp32)."""
    shard = WMAP[key]
    f = _files.get(shard) or _files.setdefault(shard, safe_open(f"{SNAP}/{shard}", "pt"))
    t = f.get_tensor(key)
    return t if (keep_fp32 and t.dtype == torch.float32) else t.to(torch.bfloat16)


def deinterleave(t, dim):
    """The checkpoint stores gate/up fused and row-interleaved (gate0, up0, gate1, ...) along `dim`; returns
    (gate, up). Same as transformers' Interleave + Chunk conversion for inkling_mm_model."""
    pairs = t.unflatten(dim, (t.shape[dim] // 2, 2))
    return pairs.select(dim + 1, 0).contiguous(), pairs.select(dim + 1, 1).contiguous()


def config():
    cfg = AutoConfig.from_pretrained(SNAP).text_config
    cfg._attn_implementation = "sdpa"  # transformers' sdpa path folds the relative position bias into the mask
    # transformers 5.17's InklingTextConfig moves the checkpoint's dense_intermediate_size (16384) into
    # intermediate_size, overwriting the checkpoint's intermediate_size (2048 = the expert width), and leaves
    # moe_intermediate_size at its class default 3072. The experts are 2048 wide (w13 [256, 4096, 4096]): pin both.
    raw = json.load(open(f"{SNAP}/config.json"))["text_config"]
    cfg.moe_intermediate_size = raw["intermediate_size"]
    cfg.intermediate_size = raw["dense_intermediate_size"]
    return cfg


def layer_state_dict(cfg, li, prefix="model.llm.layers"):
    """HF `InklingDecoderLayer` state dict for checkpoint layer `li` (key map = transformers conversion_mapping)."""
    pre = f"{prefix}.{li}."
    a = pre + "attn."
    sd = {
        "input_layernorm.weight": get(pre + "attn_norm.weight"),
        "post_attention_layernorm.weight": get(pre + "mlp_norm.weight"),
        "attn_sconv.conv1d.weight": get(pre + "attn_sconv.weight"),
        "mlp_sconv.conv1d.weight": get(pre + "mlp_sconv.weight"),
        "self_attn.q_proj.weight": get(a + "wq_du.weight"),
        "self_attn.k_proj.weight": get(a + "wk_dv.weight"),
        "self_attn.v_proj.weight": get(a + "wv_dv.weight"),
        "self_attn.r_proj.weight": get(a + "wr_du.weight"),
        "self_attn.o_proj.weight": get(a + "wo_ud.weight"),
        "self_attn.q_norm.weight": get(a + "q_norm.weight"),
        "self_attn.k_norm.weight": get(a + "k_norm.weight"),
        "self_attn.k_sconv.conv1d.weight": get(a + "k_sconv.weight"),
        "self_attn.v_sconv.conv1d.weight": get(a + "v_sconv.weight"),
        "self_attn.rel_logits_proj.proj": get(a + "rel_logits_proj.proj"),
    }
    m = pre + "mlp."
    if pre + "mlp.gate.weight" in WMAP:  # sparse
        sd["mlp.gate.weight"] = get(m + "gate.weight")
        sd["mlp.gate.e_score_correction_bias"] = get(m + "gate.bias")
        sd["mlp.gate.global_scale"] = get(m + "gate.global_scale")
        gate, up = deinterleave(get(m + "experts.w13_weight"), 1)  # [E, 2I, H]
        sd["mlp.experts.gate_up_proj"] = torch.cat([gate, up], dim=1)
        sd["mlp.experts.down_proj"] = get(m + "experts.w2_weight")
        sg, su = deinterleave(get(m + "shared_experts.shared_w13_weight"), 1)
        sd["mlp.shared_experts.gate_proj"] = sg
        sd["mlp.shared_experts.up_proj"] = su
        sd["mlp.shared_experts.down_proj"] = get(m + "shared_experts.shared_w2_weight")
    else:  # dense (layers < dense_mlp_idx)
        g, u = deinterleave(get(m + "w13_dn.weight"), 0)
        sd["mlp.gate_proj.weight"] = g
        sd["mlp.up_proj.weight"] = u
        sd["mlp.down_proj.weight"] = get(m + "w2_md.weight")
        sd["mlp.global_scale"] = get(m + "global_scale")
    return sd


def load_layer(cfg, li):
    with torch.device("meta"):
        layer = M.InklingDecoderLayer(cfg, li)
    layer = layer.to_empty(device="cpu").to(torch.bfloat16)
    sd = layer_state_dict(cfg, li)
    params = dict(layer.named_parameters()) | dict(layer.named_buffers())
    for name, t in sd.items():
        p = params[name]
        if t.numel() != p.numel():
            raise RuntimeError(f"layer {li} {name}: checkpoint {tuple(t.shape)} vs module {tuple(p.shape)}")
        sd[name] = t.reshape(p.shape)  # sconv weights: [C, K] in the checkpoint, [C, 1, K] in nn.Conv1d
        if t.dtype == torch.float32:
            p.data = p.data.float()
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k not in ("mlp.gate.e_score_correction_bias",)]
    if unexpected or missing:
        raise RuntimeError(f"layer {li}: missing {missing} unexpected {unexpected}")
    if hasattr(layer.mlp, "gate"):
        layer.mlp.gate.e_score_correction_bias = sd["mlp.gate.e_score_correction_bias"].float()
    return layer.eval()


def run_layer(cfg, layer, x, chunk):
    """x [B, T, H] through one decoder layer, in query chunks against a fresh per-layer cache."""
    B, T, _ = x.shape
    if chunk <= 0 or chunk >= T:
        chunk = T
    li = layer.self_attn.layer_idx
    cache = DynamicCache(config=cfg)
    # only this layer's cache entry is ever filled, but the mask helpers size masks from the first cache layer of
    # the matching type: alias every slot to this layer's entry
    cache.layers = [cache.layers[li]] * len(cache.layers)
    out = torch.empty_like(x)
    sliding = cfg.layer_types[li] == "hybrid_sliding"
    for s in range(0, T, chunk):
        xc = x[:, s : s + chunk]
        pos = torch.arange(s, s + xc.shape[1]).unsqueeze(0).expand(B, -1)
        mk = dict(config=cfg, inputs_embeds=xc, attention_mask=None, past_key_values=cache, position_ids=pos)
        mask = create_sliding_window_causal_mask(**mk) if sliding else create_causal_mask(**mk)
        out[:, s : s + xc.shape[1]] = layer(xc, attention_mask=mask, conv_mask=None, past_key_values=cache)
    return out


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
    chunk = int(os.environ.get("REF_CHUNK", "2048"))
    K, NS = cfg.num_experts_per_tok, cfg.n_shared_experts
    moe_layers = [i for i in range(L) if cfg.mlp_layer_types[i] == "sparse"]
    r_ids = np.zeros((B, len(moe_layers), T, K), dtype=np.uint8)
    r_w = np.zeros((B, len(moe_layers), T, K), dtype=np.float16)
    r_sh = np.zeros((B, len(moe_layers), T, NS), dtype=np.float16)

    norm = M.InklingRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = get("model.llm.embed_norm.weight").to(torch.bfloat16)
    x = norm(torch.nn.functional.embedding(ids, get("model.llm.embed.weight")))
    taps = [int(t) for t in os.environ.get("REF_TAPS", "").split(",") if t]
    tap_out = torch.zeros(B, T, len(taps), cfg.hidden_size, dtype=torch.bfloat16) if taps else None
    gold = {"names": names[0], "ids": ids[0].clone(), "x_in": []} if os.environ.get("REF_GOLD") else None
    for li in range(L):
        t0 = time.time()
        layer = load_layer(cfg, li)
        t1 = time.time()
        if gold is not None:
            gold["x_in"].append(x[0].clone())
        if li in moe_layers:
            j = moe_layers.index(li)
            gate_fwd = layer.mlp.gate.forward

            def rec(h, j=j, gate_fwd=gate_fwd):
                logits, w, idx, sh = gate_fwd(h)
                # rec runs once per query chunk: positions are filled chunk by chunk
                n = idx.shape[0] // B
                s = rec.pos
                r_ids[:, j, s : s + n] = idx.reshape(B, n, K).numpy().astype(np.uint8)
                r_w[:, j, s : s + n] = w.reshape(B, n, K).float().numpy().astype(np.float16)
                r_sh[:, j, s : s + n] = sh.reshape(B, n, NS).float().numpy().astype(np.float16)
                rec.pos += n
                return logits, w, idx, sh

            rec.pos = 0
            layer.mlp.gate.forward = rec
        x = run_layer(cfg, layer, x, chunk)
        if li in taps:
            tap_out[:, :, taps.index(li)] = x
        print(
            f"layer {li} load {t1 - t0:.1f}s fwd {time.time() - t1:.1f}s |x| {x.float().norm(dim=-1).mean():.1f}",
            flush=True,
        )
        del layer
    np.savez(
        os.path.join(out_dir, "routing.npz"),
        names=np.array(names),
        layers=np.array(moe_layers),
        ids=r_ids,
        w=r_w,
        shared=r_sh,
    )
    if taps:
        torch.save({"names": names, "layers": taps, "taps": tap_out}, os.path.join(out_dir, "taps.pt"))
    if L < cfg.num_hidden_layers:
        if gold is not None:
            gold["x_out"] = x[0].clone()
            torch.save(gold, os.path.join(out_dir, "gold.pt"))
        return
    fnorm = M.InklingRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    fnorm.weight.data = get("model.llm.norm.weight").to(torch.bfloat16)
    x = fnorm(x) / cfg.logits_mup_width_multiplier
    if gold is not None:
        gold["x_out"] = x[0].clone()  # after the final norm and the 1/16
        torch.save(gold, os.path.join(out_dir, "gold.pt"))
    head = get("model.llm.unembed.weight")[: cfg.unpadded_vocab_size]
    for b, n in enumerate(names):
        lp = torch.log_softmax((x[b] @ head.T).float(), dim=-1)  # [T, V]  (bf16 matmul, as HF's lm_head)
        nxt = ids[b, 1:]
        nll = -lp[:-1].gather(1, nxt[:, None])[:, 0]
        top = lp.topk(32, -1)
        acc = lp[:-1].argmax(-1) == nxt
        np.savez(
            os.path.join(out_dir, f"ref_{n}.npz"),
            top_ids=top.indices.numpy().astype(np.int32),
            top_lp=top.values.numpy().astype(np.float16),
            nll=nll.numpy(),
            acc=acc.numpy(),
        )
        print(f"REF {n} nll {float(nll.mean()):.3f} acc {float(acc.float().mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
