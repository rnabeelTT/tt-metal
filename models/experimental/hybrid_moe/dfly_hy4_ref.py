# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFly2: Hy4-flash's built-in speculative drafter, as a self-contained CPU reference (torch, fp32 math).

Extracted from vLLM PR 68296 (jeejeelee/vllm@3f216719b67188e30f86a87212ab54dac93f7bf2):
`vllm/models/hy_v4/nvidia/{dfly.py, dfly_speculator.py, dfly_inputs.py, dfly_target_reuse.py, attention.py,
model.py}`, `vllm/models/hy_v4/dfly_config.py`, `vllm/models/hy_v4_flash/dfly_config.py`. The checkpoint's
`model.dfly_layers.*` weights (0.85B, BF16) live in the target checkpoint; the drafter borrows the target's embedding
table and LM head. There is no HF reference for it (transformers ignores `dfly_layers`).

What one DFly step does (after a target pass that accepted some drafts and sampled a "bonus" token):

  anchor  = position of the bonus token (= position of the last accepted target row + 1).
  queries = W = num_speculative_tokens + 1 = 6 rows at positions anchor .. anchor+5:
            input ids [bonus, MASK x 5]  (MASK = dfly_mask_token_id 120009).
  embed   = target embed_tokens(ids)  (plain token embedding: no Engram, no hyper-connection streams);
            row 0 only: e0 = e0 + gate * eh_proj(cat(enorm(e0), hnorm(h_target)))   (`mtp_fusion`, "slot0"),
            h_target = the target's FINAL hidden (model.norm(hc_head(streams)), the LM head's input) at the last
            accepted row.
  layers  = 5 pre-norm decoder layers (single residual stream, no iHC): x += attn(input_layernorm(x));
            x += mlp(post_attention_layernorm(x)); dense SwiGLU MLP 3072 -> 9216 -> 3072; then `final_norm`.
            Attention = the target's sparse-layer MLA (32 heads, q_lora 1024, kv_lora 512, qk 192 nope + 64 rope, v
            256, scale (192+64)^-0.5, interleaved RoPE with the sparse layers' theta 1e7), learned per-head sink,
            elementwise sigmoid output gate (`linear_gate`), `post_proj_norm` after `o_proj`.
            KV SHARE: draft layer i reads AND WRITES the MLA latent cache (kv_a_layernorm(kv_c) 512 + roped k_pe 64)
            of target layer dfly_target_kv_layers[i] = 25, 29, 33, 37, 41. Its own 6 query rows' latents are written
            at positions anchor..anchor+5 of that cache (vLLM aliases the storage; the next target verify pass
            overwrites those positions with the target's own latents before reading them).
            INDEX SHARE: query row j of draft layer i attends to
              (a) history: the token positions the target's compressed indexer recalled for the LAST ACCEPTED ROW at
                  target layer dfly_target_index_layer[i] (same list, 25..41): up to dfly_target_index_topk = 2048
                  positions (511 best complete 4-token groups + the row's own group, expanded to tokens), keeping
                  only positions < anchor (minus dfly_context_local_window, None -> 0);
              (b) local window: positions anchor-window .. anchor-1 (window 0 for Hy4-flash: empty);
              (c) causal block: positions anchor .. anchor+j (the draft's own rows).
            Below 2048 tokens of context the target attends densely; vLLM's indexer then recalls every complete
            group (+ the own group), i.e. every position < anchor: (a) is then "all positions < anchor".
  drafts  = sequential, cheap: prev = bonus; for k = 1..5:
              c_k = h[k] + down(silu(gate(z)) * up(z)),  z = cat(hidden_norm(h[k]), embed_norm(embed(prev)))
              (`hidden_correction`, 6144 -> 3072 -> 3072), logits_k = target lm_head(c_k) (fp32),
              d_k = argmax (greedy) or a sample at the request's temperature ("probabilistic"); prev = d_k.
            Row 0's output (the bonus position) is never read. d_k is the token at position anchor + k.
  verify  = the target runs [bonus, d1..d5] at positions anchor..anchor+5 (6 rows) and accepts with standard
            speculative sampling: greedy accepts while argmax matches; probabilistic accepts d_k with
            min(1, p_k(d_k) / q_k(d_k)), q_k = softmax(draft logits_k / T) (vLLM's gumbel draft sampler applies the
            temperature only, no top-k / top-p), else resamples from norm(max(p_k - q_k, 0)); all accepted -> a bonus
            from p_6. The first rejected / bonus token is the next step's `bonus`.
  confidence_head (Linear 3072 -> 1 + bias) is in the checkpoint but unused by vLLM / SGLang (it ignores it);
  presumably a per-draft confidence for D-Cut-style dynamic draft pruning. Loaded here, exposed as `confidence()`.

The drafter takes no part in prefill: after the prompt pass the first step needs only the last prompt row's final
hidden, the target latent caches of layers 25..41 and that row's recalled indices at those layers.

usage:
  ref = DFlyRef.from_checkpoint()               # needs model-00065/00095/00148 (dfly + embed + lm_head shards)
  out = ref.propose(bonus, h_target, anchor, target_latents, history, temperature=0.0)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

REPO = "tencent/Hy4-flash"
PREFIX = "model.dfly_layers."


# --------------------------------------------------------------------------------------------------- config


@dataclass
class DFlyConfig:
    hidden_size: int = 3072
    num_layers: int = 5
    num_heads: int = 32
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    intermediate_size: int = 9216  # read from the checkpoint's FFN weights (no config field)
    hidden_correction_intermediate_size: int = 3072
    vocab_size: int = 120832
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e7  # the target's sparse ("deepseek_sparse_attention") layers
    mask_token_id: int = 120009
    num_speculative_tokens: int = 5
    target_index_topk: int = 2048
    context_local_window: int = 0
    target_kv_layers: list[int] = field(default_factory=lambda: [25, 29, 33, 37, 41])
    target_index_layer: list[int] = field(default_factory=lambda: [25, 29, 33, 37, 41])

    @property
    def query_width(self) -> int:
        return self.num_speculative_tokens + 1

    @classmethod
    def from_hf(cls, cfg: dict) -> "DFlyConfig":
        t = cfg["text_config"]
        spec = cfg.get("speculative_config") or {}
        idx = cfg["dfly_target_index_layer"]
        n = cfg["dfly_num_hidden_layers"]
        if isinstance(idx, int):
            idx = [idx] * n
        assert cfg.get("dfly_mtp_fusion", "slot0") == "slot0" and cfg.get("dfly_enable_hidden_correction", True)
        rope = t["rope_parameters"]["deepseek_sparse_attention"]["rope_theta"]
        for li in list(idx) + list(cfg["dfly_target_kv_layers"]):
            assert t["layer_types"][li] == "deepseek_sparse_attention", f"DFly target layer {li} is not sparse"
        return cls(
            hidden_size=t["hidden_size"], num_layers=n, num_heads=t["num_attention_heads"],
            q_lora_rank=t["q_lora_rank"], kv_lora_rank=t["kv_lora_rank"], qk_nope_head_dim=t["qk_nope_head_dim"],
            qk_rope_head_dim=t["qk_rope_head_dim"], v_head_dim=t["v_head_dim"], vocab_size=t["vocab_size"],
            rms_norm_eps=t["rms_norm_eps"], rope_theta=float(rope), mask_token_id=cfg["dfly_mask_token_id"],
            num_speculative_tokens=spec.get("num_speculative_tokens", 5),
            target_index_topk=cfg.get("dfly_target_index_topk", t.get("index_topk", 2048)),
            context_local_window=cfg.get("dfly_context_local_window") or 0,
            target_kv_layers=list(cfg["dfly_target_kv_layers"]), target_index_layer=list(idx),
            hidden_correction_intermediate_size=cfg.get("dfly_hidden_correction_intermediate_size",
                                                        t["hidden_size"]),
        )


# --------------------------------------------------------------------------------------------------- math


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return y * w.float() if w is not None else y


def rope_cos_sin(positions: torch.Tensor, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved (GPT-J / GLM) RoPE tables, [T, dim]: pair (2i, 2i+1) rotates with frequency i."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    f = positions.double()[:, None] * inv[None, :]
    return f.cos().repeat_interleave(2, -1).float(), f.sin().repeat_interleave(2, -1).float()


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [..., T, (heads,) dim] with cos/sin [T, dim] broadcast over heads when x is [T, H, dim]."""
    if x.dim() == 3:
        cos, sin = cos[:, None, :], sin[:, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x.float() * cos + rot.float() * sin


# --------------------------------------------------------------------------------------------------- weights


def _snapshot_dir() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, allow_patterns=["config.json", "model.safetensors.index.json"])


def load_weights(path: str | None = None, dtype=torch.float32) -> tuple[dict, dict[str, torch.Tensor]]:
    """(config dict, {name: tensor}) with every `model.dfly_layers.*` tensor plus the target's embedding table and LM
    head. Opens only the shards that hold them (safetensors, lazily per tensor)."""
    from safetensors import safe_open

    path = path or _snapshot_dir()
    cfg = json.load(open(os.path.join(path, "config.json")))
    wmap = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    want = [k for k in wmap if k.startswith(PREFIX)] + ["model.embed_tokens.weight", "lm_head.weight"]
    out: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for k in want:
        by_shard.setdefault(wmap[k], []).append(k)
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(path, shard), framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                out[k] = t if k in ("model.embed_tokens.weight", "lm_head.weight") else t.to(dtype)
    return cfg, out


# --------------------------------------------------------------------------------------------------- model


class DFlyRef:
    """The drafter on CPU. All weights fp32 (embedding / LM head kept in their checkpoint dtype, upcast per use)."""

    def __init__(self, cfg: DFlyConfig, w: dict[str, torch.Tensor]):
        self.cfg = cfg
        self.w = w
        gp = w[PREFIX + "0.mlp.gate_proj.weight"]
        cfg.intermediate_size = gp.shape[0]
        hc = w[PREFIX + "hidden_correction.gate_proj.weight"]
        cfg.hidden_correction_intermediate_size = hc.shape[0]

    @classmethod
    def from_checkpoint(cls, path: str | None = None) -> "DFlyRef":
        cfg, w = load_weights(path)
        return cls(DFlyConfig.from_hf(cfg), w)

    @classmethod
    def random(cls, cfg: DFlyConfig | None = None, seed: int = 0, std: float = 0.02) -> "DFlyRef":
        """Random weights with the checkpoint's names and shapes (cfg dims may be shrunk for tests)."""
        c = cfg or DFlyConfig()
        g = torch.Generator().manual_seed(seed)
        d, H = c.hidden_size, c.num_heads
        qk = c.qk_nope_head_dim + c.qk_rope_head_dim

        def r(*s):
            return torch.randn(*s, generator=g) * std

        def one(n):
            return 1.0 + torch.randn(n, generator=g) * 0.05

        w = {"model.embed_tokens.weight": r(c.vocab_size, d), "lm_head.weight": r(c.vocab_size, d)}
        for i in range(c.num_layers):
            p = f"{PREFIX}{i}."
            w.update({
                p + "input_layernorm.weight": one(d), p + "post_attention_layernorm.weight": one(d),
                p + "mlp.gate_proj.weight": r(c.intermediate_size, d), p + "mlp.up_proj.weight": r(c.intermediate_size, d),
                p + "mlp.down_proj.weight": r(d, c.intermediate_size),
                p + "self_attn.q_a_proj.weight": r(c.q_lora_rank, d), p + "self_attn.q_a_layernorm.weight": one(c.q_lora_rank),
                p + "self_attn.q_b_proj.weight": r(H * qk, c.q_lora_rank),
                p + "self_attn.kv_a_proj_with_mqa.weight": r(c.kv_lora_rank + c.qk_rope_head_dim, d),
                p + "self_attn.kv_a_layernorm.weight": one(c.kv_lora_rank),
                p + "self_attn.kv_b_proj.weight": r(H * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank),
                p + "self_attn.linear_gate.weight": r(H * c.v_head_dim, d),
                p + "self_attn.o_proj.weight": r(d, H * c.v_head_dim),
                p + "self_attn.post_proj_norm.weight": one(d),
                p + "self_attn.learnable_sink_param": torch.randn(H, generator=g),
            })
        hci = c.hidden_correction_intermediate_size
        w.update({
            PREFIX + "final_norm.weight": one(d),
            PREFIX + "mtp_fusion.enorm.weight": one(d), PREFIX + "mtp_fusion.hnorm.weight": one(d),
            PREFIX + "mtp_fusion.eh_proj.weight": r(d, 2 * d), PREFIX + "mtp_fusion.gate": r(d) * 10,
            PREFIX + "hidden_correction.hidden_norm.weight": one(d), PREFIX + "hidden_correction.embed_norm.weight": one(d),
            PREFIX + "hidden_correction.gate_proj.weight": r(hci, 2 * d),
            PREFIX + "hidden_correction.up_proj.weight": r(hci, 2 * d),
            PREFIX + "hidden_correction.down_proj.weight": r(d, hci),
            PREFIX + "confidence_head.proj.weight": r(1, d), PREFIX + "confidence_head.proj.bias": torch.zeros(1),
        })
        return cls(c, w)

    # ---------------------------------------------------------------- pieces

    def _p(self, name: str) -> torch.Tensor:
        return self.w[PREFIX + name].float()

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.w["model.embed_tokens.weight"][ids].float()

    def lm_head(self, h: torch.Tensor) -> torch.Tensor:
        return h.float() @ self.w["lm_head.weight"].float().T  # the target computes its head in fp32

    def fusion(self, e0: torch.Tensor, h_target: torch.Tensor) -> torch.Tensor:
        eps = self.cfg.rms_norm_eps
        cat = torch.cat((rms_norm(e0, self._p("mtp_fusion.enorm.weight"), eps),
                         rms_norm(h_target, self._p("mtp_fusion.hnorm.weight"), eps)), -1)
        return e0 + self._p("mtp_fusion.gate") * (cat @ self._p("mtp_fusion.eh_proj.weight").T)

    def hidden_correction(self, h: torch.Tensor, prev_ids: torch.Tensor) -> torch.Tensor:
        eps = self.cfg.rms_norm_eps
        z = torch.cat((rms_norm(h, self._p("hidden_correction.hidden_norm.weight"), eps),
                       rms_norm(self.embed(prev_ids), self._p("hidden_correction.embed_norm.weight"), eps)), -1)
        a = F.silu(z @ self._p("hidden_correction.gate_proj.weight").T) * (z @ self._p("hidden_correction.up_proj.weight").T)
        return h + a @ self._p("hidden_correction.down_proj.weight").T

    def confidence(self, h: torch.Tensor) -> torch.Tensor:
        """The checkpoint's (unused) confidence head: one logit per row."""
        return (h.float() @ self._p("confidence_head.proj.weight").T + self._p("confidence_head.proj.bias")).squeeze(-1)

    def latents(self, layer: int, x_normed: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The MLA latent a layer writes to the (shared) cache: kv_a_layernorm(kv_c) [T, 512], roped k_pe [T, 64]."""
        c = self.cfg
        p = f"{layer}.self_attn."
        kv = x_normed @ self._p(p + "kv_a_proj_with_mqa.weight").T
        kv_c, k_pe = kv.split((c.kv_lora_rank, c.qk_rope_head_dim), -1)
        cos, sin = rope_cos_sin(positions, c.qk_rope_head_dim, c.rope_theta)
        return rms_norm(kv_c, self._p(p + "kv_a_layernorm.weight"), c.rms_norm_eps), apply_rope_interleaved(k_pe, cos, sin)

    def attention(self, layer: int, x_normed: torch.Tensor, positions: torch.Tensor,
                  kv_c_all: torch.Tensor, k_pe_all: torch.Tensor, key_index: list[torch.Tensor]) -> torch.Tensor:
        """One draft layer's attention for W query rows.

        kv_c_all / k_pe_all: the shared latent cache rows [S, 512] / [S, 64] by POSITION (the target's latents for
        positions < anchor, this layer's own block latents at anchor .. anchor+W-1). key_index[j]: the positions query
        row j attends to (history + window + causal block), any order, no duplicates."""
        c = self.cfg
        H, dn, dr, dv = c.num_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
        p = f"{layer}.self_attn."
        q_c = rms_norm(x_normed @ self._p(p + "q_a_proj.weight").T, self._p(p + "q_a_layernorm.weight"), c.rms_norm_eps)
        q = (q_c @ self._p(p + "q_b_proj.weight").T).view(-1, H, dn + dr)
        cos, sin = rope_cos_sin(positions, dr, c.rope_theta)
        q_nope, q_pe = q[..., :dn], apply_rope_interleaved(q[..., dn:], cos, sin)
        wkv = self._p(p + "kv_b_proj.weight").view(H, dn + dv, c.kv_lora_rank)
        w_uk, w_uv = wkv[:, :dn], wkv[:, dn:]  # [H, dn, 512], [H, dv, 512]
        sink = self._p(p + "learnable_sink_param")
        scale = (dn + dr) ** -0.5
        outs = []
        for j in range(q.shape[0]):
            idx = key_index[j]
            lat, pe = kv_c_all[idx], k_pe_all[idx]  # [K, 512], [K, 64]
            k_nope = torch.einsum("hdl,kl->hkd", w_uk, lat)  # [H, K, dn]
            v = torch.einsum("hdl,kl->hkd", w_uv, lat)  # [H, K, dv]
            s = (torch.einsum("hd,hkd->hk", q_nope[j], k_nope) + q_pe[j] @ pe.T) * scale  # [H, K]
            s = torch.cat((s, sink[:, None]), -1)  # learned sink: an extra logit per head, value 0
            pr = torch.softmax(s, -1)[:, :-1]
            outs.append(torch.einsum("hk,hkd->hd", pr, v).reshape(-1))
        o = torch.stack(outs) * torch.sigmoid(x_normed @ self._p(p + "linear_gate.weight").T)
        o = o @ self._p(p + "o_proj.weight").T
        return rms_norm(o, self._p(p + "post_proj_norm.weight"), c.rms_norm_eps)

    def mlp(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        p = f"{layer}.mlp."
        return (F.silu(x @ self._p(p + "gate_proj.weight").T) * (x @ self._p(p + "up_proj.weight").T)) @ self._p(
            p + "down_proj.weight").T

    # ---------------------------------------------------------------- one DFly step

    def key_positions(self, anchor: int, history: torch.Tensor) -> list[torch.Tensor]:
        """Per query row j: target-recalled history (< anchor - window), the local window, the causal block."""
        c = self.cfg
        start = anchor - c.context_local_window
        h = history[(history >= 0) & (history < start)].long()
        win = torch.arange(max(start, 0), anchor, dtype=torch.long)
        return [torch.cat((h, win, torch.arange(anchor, anchor + j + 1, dtype=torch.long)))
                for j in range(c.query_width)]

    @torch.no_grad()
    def propose(self, bonus: int, h_target: torch.Tensor, anchor: int,
                target_latents: list[tuple[torch.Tensor, torch.Tensor]], history: list[torch.Tensor],
                temperature: float = 0.0, generator: torch.Generator | None = None) -> dict:
        """One DFly step.

        bonus: the token at position `anchor` (the target's last sampled token). h_target [3072]: the target's final
        normalized hidden at position anchor-1 (its last accepted row). target_latents[i]: (kv_c [S, 512], k_pe [S, 64])
        of target layer target_kv_layers[i] for positions 0..S-1 (S >= anchor; rows >= anchor are ignored: the draft
        writes its own there). history[i]: the positions the target's indexer recalled for its last accepted row at
        layer target_index_layer[i] (int, -1 padded; < 2048 context: every position < anchor).

        Returns {"drafts" [5], "draft_logits" [5, V] (fp32, before temperature), "block_hidden" [6, 3072] (after
        final_norm), "corrected" [5, 3072], "block_latents": [(kv_c [6,512], k_pe [6,64]) per layer]}."""
        c = self.cfg
        W = c.query_width
        pos = torch.arange(anchor, anchor + W)
        ids = torch.tensor([bonus] + [c.mask_token_id] * (W - 1))
        x = self.embed(ids)
        x[0] = self.fusion(x[0], h_target.float())
        keys = self.key_positions(anchor, history[0]) if c.num_layers else []
        block_latents = []
        for i in range(c.num_layers):
            keys = self.key_positions(anchor, history[i])
            xn = rms_norm(x, self._p(f"{i}.input_layernorm.weight"), c.rms_norm_eps)
            lat, pe = self.latents(i, xn, pos)
            block_latents.append((lat, pe))
            kv_c_t, k_pe_t = target_latents[i]
            kv_c_all = torch.cat((kv_c_t[:anchor].float(), lat), 0)
            k_pe_all = torch.cat((k_pe_t[:anchor].float(), pe), 0)
            x = x + self.attention(i, xn, pos, kv_c_all, k_pe_all, keys)
            x = x + self.mlp(i, rms_norm(x, self._p(f"{i}.post_attention_layernorm.weight"), c.rms_norm_eps))
        h = rms_norm(x, self._p("final_norm.weight"), c.rms_norm_eps)
        prev = torch.tensor([bonus])
        drafts, logits_all, corrected = [], [], []
        for k in range(1, W):
            ck = self.hidden_correction(h[k:k + 1], prev)
            lg = self.lm_head(ck)[0]
            if temperature > 0:
                tok = torch.multinomial(torch.softmax(lg / temperature, -1), 1, generator=generator)
            else:
                tok = lg.argmax(-1, keepdim=True)
            drafts.append(int(tok))
            logits_all.append(lg)
            corrected.append(ck[0])
            prev = tok.view(1)
        return {"drafts": torch.tensor(drafts), "draft_logits": torch.stack(logits_all), "block_hidden": h,
                "corrected": torch.stack(corrected), "block_latents": block_latents}


# --------------------------------------------------------------------------------------------------- verification


def speculative_accept(drafts: torch.Tensor, draft_logits: torch.Tensor, target_logits: torch.Tensor,
                       temperature: float, generator: torch.Generator | None = None) -> list[int]:
    """Standard speculative acceptance for one block. target_logits [K+1, V] for rows [bonus, d1..dK] (row k predicts
    position anchor+k+1); returns the emitted tokens (accepted drafts + one corrected / bonus token). Greedy when
    temperature == 0. The target distribution here is temperature-only; the serving path applies the request's
    top-k / top-p / penalties to p (vLLM's rejection sampler), q stays temperature-only."""
    out = []
    K = drafts.shape[0]
    for k in range(K):
        if temperature <= 0:
            t = int(target_logits[k].argmax())
            out.append(t)
            if t != int(drafts[k]):
                return out
            continue
        p = torch.softmax(target_logits[k].float() / temperature, -1)
        q = torch.softmax(draft_logits[k].float() / temperature, -1)
        d = int(drafts[k])
        u = torch.rand(1, generator=generator).item()
        if u <= min(1.0, (p[d] / q[d].clamp_min(1e-30)).item()):
            out.append(d)
            continue
        resid = (p - q).clamp_min(0)
        resid = resid / resid.sum() if resid.sum() > 0 else p
        out.append(int(torch.multinomial(resid, 1, generator=generator)))
        return out
    last = target_logits[K]
    out.append(int(last.argmax()) if temperature <= 0 else
               int(torch.multinomial(torch.softmax(last.float() / temperature, -1), 1, generator=generator)))
    return out


if __name__ == "__main__":  # smoke test on the real weights with a random target state
    import time

    t0 = time.time()
    ref = DFlyRef.from_checkpoint(os.environ.get("HY4_PATH"))
    c = ref.cfg
    print(f"loaded in {time.time() - t0:.0f} s: {c}")
    S, anchor = 300, 280
    g = torch.Generator().manual_seed(0)
    lat = [(torch.randn(S, c.kv_lora_rank, generator=g), torch.randn(S, c.qk_rope_head_dim, generator=g))
           for _ in range(c.num_layers)]
    hist = [torch.arange(anchor) for _ in range(c.num_layers)]
    out = ref.propose(1234, torch.randn(c.hidden_size, generator=g), anchor, lat, hist)
    print("drafts", out["drafts"].tolist(), "finite", bool(torch.isfinite(out["draft_logits"]).all()))
