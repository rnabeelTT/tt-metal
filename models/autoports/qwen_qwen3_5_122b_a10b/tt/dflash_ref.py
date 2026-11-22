# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash 2 drafter, torch reference (host): the block-diffusion draft model of `incoai/Ornith-1.5-122B-DFlash2`.

    Follows z-lab/dflash `dflash/model.py` (DFlash2DraftModel), written without HF cache classes so it doubles as the
    reference of the device drafter:

    context  : the target's hidden states after layers `target_layer_ids` (8 x H, concatenated) of every verified token;
               fc -> hidden_norm gives one H row per token, whose K / V (k_norm, RoPE at the token's position) the drafter
               keeps per layer (`ctx_k` / `ctx_v`); only context rows are kept between rounds.
    block    : [last verified token, mask x (bs-1)] embedded with the TARGET's embedding, positions start .. start+bs-1.
    layer    : x = x + conv_a.finish(attn(conv_a.prepare(norm(x)))); x = x + conv_m.finish(mlp(conv_m.prepare(norm(x))))
               attention: q from the block rows; k / v = [context rows; block rows]; non-causal, sliding window 2048.
    proposal : rows 1..bs-1 -> final norm -> TARGET LM head -> top-16 candidates per row -> the selector's path (a
               first-order chain over the candidates with predecessor / successor codebooks).
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    """hidden [L, H], dynamic [L, K, G], base [K, H] -> [L, H] (causal along L, zero before row 0)."""
    L, H = hidden.shape
    G = H // group_size
    blocks = hidden.view(L, G, group_size)
    out = torch.zeros_like(blocks)
    for off in range(base.shape[0]):
        vals = blocks if off == 0 else F.pad(blocks[:-off], (0, 0, 0, 0, off, 0))
        out = out + base[off].view(1, G, group_size).to(hidden.dtype) * vals
        out = out + dynamic[:, off].view(L, G, 1) * vals
    return out.view(L, H)


class DFlash2Ref:
    def __init__(self, ckpt_dir, embed_weight, lm_head_weight, dtype=torch.float32):
        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
        dc = cfg["dflash_config"]
        self.cfg = cfg
        self.H = cfg["hidden_size"]
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.window = cfg["sliding_window"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.bs = dc["block_size"]
        self.mask_id = dc["mask_token_id"]
        self.layer_ids = dc["target_layer_ids"]
        self.K = dc["conv_kernel_size"]
        self.gs = dc["conv_group_size"]
        self.top_k = dc["selector_top_k"]
        self.L = cfg["num_hidden_layers"]
        self.dtype = dtype
        from safetensors.torch import load_file

        w = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        self.w = {k: v.to(dtype) for k, v in w.items()}
        self.embed = embed_weight  # bf16 [V, H] (target's)
        self.lm_head = lm_head_weight.to(dtype)  # [V, H]
        self.reset()

    def g(self, name):
        return self.w[name]

    def reset(self):
        self.ctx_k = [None] * self.L  # per layer [nkv, n_ctx, hd] (normed + roped)
        self.ctx_v = [None] * self.L
        self.ctx_pos = torch.zeros(0, dtype=torch.long)

    def _rope(self, positions):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        f = positions.float()[:, None] * inv[None]
        emb = torch.cat([f, f], dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def _conv(self, prefix, h):
        """prepare: (conv with dynamic half 0 + base 0, dynamic half 1 for finish)."""
        G = self.H // self.gs
        dyn = (h @ self.g(f"{prefix}.kernel_projection.weight").T).view(h.shape[0], 2, self.K, G)
        base = self.g(f"{prefix}.base_kernel")  # [2, K, H]
        return _grouped_dynamic_convolve(h, dyn[:, 0], base[0], self.gs), dyn[:, 1], base[1]

    def add_context(self, feats, positions):
        """feats [n, 8H] target features of verified tokens at `positions` [n]; appends their K / V to every layer."""
        if feats.shape[0] == 0:
            return
        x = feats.to(self.dtype) @ self.g("fc.weight").T
        x = _rms(x, self.g("hidden_norm.weight"), self.eps)
        cos, sin = self._rope(positions)
        for i in range(self.L):
            p = f"layers.{i}.self_attn"
            k = (x @ self.g(f"{p}.k_proj.weight").T).view(-1, self.nkv, self.hd)
            v = (x @ self.g(f"{p}.v_proj.weight").T).view(-1, self.nkv, self.hd)
            k = _rms(k, self.g(f"{p}.k_norm.weight"), self.eps)
            k = k * cos[:, None] + _rotate_half(k) * sin[:, None]
            k, v = k.transpose(0, 1), v.transpose(0, 1)
            self.ctx_k[i] = k if self.ctx_k[i] is None else torch.cat([self.ctx_k[i], k], dim=1)
            self.ctx_v[i] = v if self.ctx_v[i] is None else torch.cat([self.ctx_v[i], v], dim=1)
        self.ctx_pos = torch.cat([self.ctx_pos, positions.long()])
        # keep only what a sliding-window query can still see (the next block starts at positions[-1] + 1)
        keep = self.ctx_pos > (int(positions[-1]) + 1 - self.window)
        if not bool(keep.all()):
            self.ctx_pos = self.ctx_pos[keep]
            self.ctx_k = [k[:, keep] for k in self.ctx_k]
            self.ctx_v = [v[:, keep] for v in self.ctx_v]

    def block_hidden(self, anchor, start):
        """Draft hidden of rows 1..bs-1 for the block [anchor, mask...] at positions start..start+bs-1: [bs-1, H]."""
        bs = self.bs
        ids = torch.full((bs,), self.mask_id, dtype=torch.long)
        ids[0] = anchor
        x = F.embedding(ids, self.embed).to(self.dtype)
        pos = torch.arange(start, start + bs)
        cos, sin = self._rope(pos)
        # visibility: non-causal sliding window |q - k| < window (block rows all see each other)
        kpos = torch.cat([self.ctx_pos, pos])
        vis = (pos[:, None] - kpos[None, :]).abs() < self.window
        bias = torch.zeros(vis.shape, dtype=self.dtype).masked_fill(~vis, float("-inf"))
        rep = self.nh // self.nkv
        for i in range(self.L):
            pl = f"layers.{i}"
            pa = f"{pl}.self_attn"
            res = x
            h = _rms(x, self.g(f"{pl}.input_layernorm.weight"), self.eps)
            h, dyn, base1 = self._conv(f"{pl}.attention_conv", h)
            q = (h @ self.g(f"{pa}.q_proj.weight").T).view(bs, self.nh, self.hd)
            q = _rms(q, self.g(f"{pa}.q_norm.weight"), self.eps)
            k = (h @ self.g(f"{pa}.k_proj.weight").T).view(bs, self.nkv, self.hd)
            k = _rms(k, self.g(f"{pa}.k_norm.weight"), self.eps)
            v = (h @ self.g(f"{pa}.v_proj.weight").T).view(bs, self.nkv, self.hd)
            q = q * cos[:, None] + _rotate_half(q) * sin[:, None]
            k = k * cos[:, None] + _rotate_half(k) * sin[:, None]
            k, v, q = k.transpose(0, 1), v.transpose(0, 1), q.transpose(0, 1)
            if self.ctx_k[i] is not None:
                k = torch.cat([self.ctx_k[i], k], dim=1)
                v = torch.cat([self.ctx_v[i], v], dim=1)
            k = k.repeat_interleave(rep, dim=0)
            v = v.repeat_interleave(rep, dim=0)
            s = (q.float() @ k.float().transpose(1, 2)) / math.sqrt(self.hd) + bias.float()
            o = (torch.softmax(s, dim=-1) @ v.float()).to(self.dtype)  # [nh, bs, hd]
            o = o.transpose(0, 1).reshape(bs, self.nh * self.hd) @ self.g(f"{pa}.o_proj.weight").T
            o = _grouped_dynamic_convolve(o, dyn, base1, self.gs)
            x = res + o
            res = x
            h = _rms(x, self.g(f"{pl}.post_attention_layernorm.weight"), self.eps)
            h, dyn, base1 = self._conv(f"{pl}.mlp_conv", h)
            m = F.silu(h @ self.g(f"{pl}.mlp.gate_proj.weight").T) * (h @ self.g(f"{pl}.mlp.up_proj.weight").T)
            m = m @ self.g(f"{pl}.mlp.down_proj.weight").T
            m = _grouped_dynamic_convolve(m, dyn, base1, self.gs)
            x = res + m
        x = _rms(x, self.g("norm.weight"), self.eps)
        return x[1:]

    def select(self, hidden, logits, anchor, temperature=0.0, generator=None):
        """Selector path over the top-k candidates. Returns (path [n], candidates [n, k], q [n, k] or None)."""
        unary, cands = torch.topk(logits.float(), self.top_k, dim=-1, sorted=False)
        hp = hidden.float() @ self.g("candidate_selector.hidden_projection.weight").float().T  # [n, r]
        pre = self.g("candidate_selector.predecessor_codebook").float()
        suc = self.g("candidate_selector.successor_codebook").float()
        prev = int(anchor)
        path, qs = [], []
        for i in range(hidden.shape[0]):
            sc = unary[i] + (suc[cands[i]] @ (pre[prev] * hp[i]))
            if temperature > 0:
                q = torch.softmax(sc / temperature, dim=-1)
                j = int(torch.multinomial(q, 1, generator=generator))
                qs.append(q)
            else:
                j = int(sc.argmax())
            prev = int(cands[i, j])
            path.append(prev)
        return torch.tensor(path), cands, (torch.stack(qs) if qs else None)

    def propose(self, anchor, start, temperature=0.0, generator=None):
        h = self.block_hidden(anchor, start)
        logits = h.float() @ self.lm_head.float().T
        return self.select(h, logits, anchor, temperature, generator)
