# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU reference of Inkling-Small's native MTP drafter (mtp.safetensors, 8 depths) and its teacher-forced acceptance
on a text set, from ref_inkling.py's outputs (final_hidden.pt: the target's final_norm(h); ref_<name>.npz: the target's
top ids). This decides whether native MTP is worth the device work and how many depths to run (Phase 4).

Depth k (vLLM's InklingMTPDepthLayer, mirroring the checkpoint's mtp_model.py; chain_hidden_post_norm = false):
    in_k[t]  = [hidden_norm_k(prev_k[t]) | embed_norm_k(embed_norm_backbone(embed(tok[t + k + 1])))]   (2H)
    out_k[t] = block_k(input_proj_k(in_k))        an Inkling decoder layer (dense MLP, sliding or global attention,
                                                  the 4 short convs) over positions t (causal, its own KV); no tau
    draft_k[t] = argmax(out_k[t] / 16 @ unembed^T)   predicts token t + k + 2
    prev_0 = the target's final_norm(h),  prev_{k+1} = out_k

Greedy acceptance (teacher-forced on the real text): depth k's draft at t is accepted when every depth j <= k matched
the target's argmax at t + j + 1 (ref top_ids[t + j + 1][0]). Prints acceptance per depth and the expected tokens per
verify pass for k = 1..8 drafts (1 + expected accepted run).

usage (host venv): python mtp_inkling_ref.py <texts.json> <ref dir (ref_inkling output)> [depths=8]
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_inkling as R  # noqa: E402
from transformers.models.inkling import modeling_inkling as M  # noqa: E402


def depth_layer(cfg, mtp_cfg, d, f):
    """HF InklingDecoderLayer for MTP depth d (dense MLP; sliding if d in mtp local_layer_ids) + its input parts."""
    import copy

    c = copy.deepcopy(cfg)
    local = set(mtp_cfg["local_layer_ids"])
    c.layer_types = ["hybrid_sliding" if d in local else "hybrid"]
    c.mlp_layer_types = ["dense"]
    c.num_hidden_layers = 1
    with torch.device("meta"):
        layer = M.InklingDecoderLayer(c, 0)
    layer = layer.to_empty(device="cpu").to(torch.bfloat16)
    pre = f"model.mtp.layers.{d}.transformer_block."
    g = lambda k: f.get_tensor(pre + k)  # noqa: E731
    a = "attn."
    sd = {
        "input_layernorm.weight": g("attn_norm.weight"),
        "post_attention_layernorm.weight": g("mlp_norm.weight"),
        "attn_sconv.conv1d.weight": g("attn_sconv.weight"),
        "mlp_sconv.conv1d.weight": g("mlp_sconv.weight"),
        "self_attn.q_proj.weight": g(a + "wq_du.weight"),
        "self_attn.k_proj.weight": g(a + "wk_dv.weight"),
        "self_attn.v_proj.weight": g(a + "wv_dv.weight"),
        "self_attn.r_proj.weight": g(a + "wr_du.weight"),
        "self_attn.o_proj.weight": g(a + "wo_ud.weight"),
        "self_attn.q_norm.weight": g(a + "q_norm.weight"),
        "self_attn.k_norm.weight": g(a + "k_norm.weight"),
        "self_attn.k_sconv.conv1d.weight": g(a + "k_sconv.weight"),
        "self_attn.v_sconv.conv1d.weight": g(a + "v_sconv.weight"),
        "self_attn.rel_logits_proj.proj": g(a + "rel_logits_proj.proj"),
        "mlp.down_proj.weight": g("mlp.w2_md.weight"),
        "mlp.global_scale": g("mlp.global_scale"),
    }
    sd["mlp.gate_proj.weight"], sd["mlp.up_proj.weight"] = R.deinterleave(g("mlp.w13_dn.weight"), 0)
    R.cast_like_hf(layer, sd, f"mtp{d}")
    p = f"model.mtp.layers.{d}."
    norm = lambda w: (lambda x, w=w.to(torch.bfloat16): w * (  # noqa: E731
        x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)).to(torch.bfloat16))
    return layer.eval(), norm(f.get_tensor(p + "hidden_norm.weight")), norm(f.get_tensor(p + "embed_norm.weight")), \
        f.get_tensor(p + "input_proj.weight").to(torch.bfloat16), c


@torch.no_grad()
def main():
    texts, ref_dir = sys.argv[1], sys.argv[2]
    n_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seqs = json.load(open(texts))
    fh = torch.load(os.path.join(ref_dir, "final_hidden.pt"))
    names, hidden = fh["names"], fh["hidden"]  # [B, T, H]
    ids = torch.tensor([seqs[n] for n in names])
    B, T = ids.shape
    cfg = R.config()
    mtp_cfg = json.load(open(os.path.join(R.SNAP, "config.json")))["mtp_config"]
    f = safe_open(os.path.join(R.SNAP, "mtp.safetensors"), "pt")
    bnorm = M.InklingRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    bnorm.weight.data = R.get("model.llm.embed_norm.weight").to(torch.bfloat16)
    emb = bnorm(torch.nn.functional.embedding(ids, R.get("model.llm.embed.weight")))  # backbone-normed [B, T, H]
    head = R.get("model.llm.unembed.weight")[: cfg.unpadded_vocab_size]
    target = torch.stack([torch.from_numpy(np.load(os.path.join(ref_dir, f"ref_{n}.npz"))["top_ids"][:, 0])
                          for n in names]).long()  # [B, T]: the target's argmax at t (predicts token t + 1)
    tconf = torch.stack([torch.from_numpy(np.load(os.path.join(ref_dir, f"ref_{n}.npz"))["top_lp"][:, 0].astype(np.float32))
                         for n in names]).exp()  # [B, T] the target's top-1 probability
    prev = hidden
    match = []  # match[k][b, t]: depth k's draft at t == target argmax at t + k + 1
    for d in range(n_depth):
        layer, hn, en, w_in, c = depth_layer(cfg, mtp_cfg, d, f)
        n = T - d - 1  # positions t with tok[t + d + 1] available
        x_in = torch.cat([hn(prev[:, :n]), en(emb[:, d + 1 : d + 1 + n])], -1) @ w_in.T  # [B, n, H]
        out = R.run_layer(c, layer, x_in, 0)
        logits = (out / cfg.logits_mup_width_multiplier) @ head.T
        draft = logits.argmax(-1)  # [B, n]: predicts token t + d + 2
        m = draft[:, : n - 1] == target[:, d + 1 : d + n]  # compare at t + d + 1
        match.append(m)
        prev = out
        print(f"depth {d}: own accuracy vs target argmax {m.float().mean():.3f}", flush=True)
        del layer
    # accepted run length at t with k drafts: leading depths that all match
    L = min(mm.shape[1] for mm in match)
    run = torch.zeros(B, L)
    alive = torch.ones(B, L, dtype=torch.bool)
    # teacher forcing on human text compares against a target that is itself unsure (its own top-1 accuracy on these
    # texts is ~0.5); greedy generation runs on the model's confident tokens. Also report the run restricted to
    # positions where the target is confident at every compared step (p > 0.5): a closer proxy for generation.
    conf_all = torch.ones(B, L, dtype=torch.bool)
    run_c = torch.zeros(B, L)
    print("drafts k : expected tokens per verify pass (1 + accepted run); [confident-target positions]")
    for k in range(n_depth):
        alive &= match[k][:, :L]
        run += alive.float()
        conf_all &= tconf[:, k + 1 : k + 1 + L] > 0.5
        print(f"  k={k + 1}: {1 + run.mean():.3f}   (P(accept depth {k}) {alive.float().mean():.3f})   "
              f"[{1 + run[conf_all].mean():.3f} over {conf_all.float().mean():.0%} of positions]")


if __name__ == "__main__":
    main()
