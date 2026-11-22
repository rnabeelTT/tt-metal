# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Plain-PyTorch CPU reference of Inkling-2-Mini's native speculative drafter
(DSpark / "MTP", weights in mtp.safetensors) plus an acceptance-length
measurement against the HF target.

Mirrors SGLang PR #51926 (zcnrex/sglang@chunan/inkling-2-mini-day0):
  srt/models/inkling2_mtp.py        Inkling2ForConditionalGenerationMTP, Inkling2MTPHead
  kernels/ops/speculative/inkling2_mtp.py   mix_rmsnorm / rnn_inputs / rnn_gates
  srt/models/dflash.py              DFlashDraftModel / DFlashDecoderLayer / DFlashAttention
  srt/models/dspark.py, srt/speculative/dspark_components/dspark_draft.py
                                    draft_markov_block / _draft_step, compute_base_logits
  kernels/ops/speculative/dspark/markov_sample.py  MarkovSampleStep.torch (step-logit rounding)
  srt/configs/inkling2.py           Inkling2MTPConfig.from_target
  srt/models/inkling2.py            target tap capture (_capture / _run_loop)

Algorithm (one draft block; "p" = number of tokens the target has committed):
  * Target taps: for every committed position t < p the target exposes 4 hidden
    states, at target_layer_ids = [5, 23, 41, 48] *layer visits* of the looped
    target (24 physical layers, core [6, 18) run 3 passes -> 48 visits; visit v of
    pass k of core layer l is 6 + 12 k + (l - 6)).  Tap v < 48 is the residual
    "partial" stream that is the INPUT of visit v (SGLang captures state.partial
    before running the layer; = the hidden_states argument of the HF decoder
    layer call).  Visit 5 = input of layer 5 (prelude), 23 = input of layer 11 in
    pass 1, 41 = input of layer 17 in pass 2.  48 is the sentinel for the final
    normed hidden state (target `norm` output, i.e. what feeds lm_head), bf16.
  * Context K/V: per draft layer i, ctx_i = mix_rmsnorm(taps, softmax(fuse_weight)[i],
    layers[i].dflare_norm) (fp32 mix + rms, single bf16 rounding); K = rope(k_norm_ctx(
    k_proj_ctx(ctx_i))) at the token's own position t, V = v_proj_ctx(ctx_i).
  * Block: 8 rows at positions p .. p+7.  Row 0 = target normed embedding of the
    anchor token (the bonus token the target just emitted, not yet processed by the
    target), rows 1..7 = mask_embed.  3 pre-norm decoder layers (input_layernorm ->
    q/k/v proj -> per-head q_norm/k_norm -> neox RoPE (theta 1e6) -> attention ->
    o_proj -> residual; post_attention_layernorm -> SiLU MLP -> residual), final norm.
    Attention: every block row attends to ALL context keys (positions < p) and to
    ALL 8 block keys (non-causal inside the block, is_causal=False / ENCODER_ONLY).
    Layers 0 and 2 are sliding-window (1024, window anchored at the block start);
    at the lengths measured here (< 1024) the window never cuts anything.
  * Draft step k = 1..7 uses the final hidden of block row k (the anchor row's
    output is discarded: sample_from_anchor=False):
      base_k  = bf16(h_k @ lm_head^T)                              (no muP divisor)
      Markov RNN head (markov_head_type "rnn_hidden"), state s_0 = 0,
      f_rnn = mix_rmsnorm(taps[p-1], softmax(fuse_weight)[-1], markov_head.dflare_norm):
        x     = [s_{k-1}, markov_w1[prev_tok], h_k, f_rnn]          (bf16, 5120)
        g,c,o = rnn_proj(x).chunk(3)                                 (bf16)
        gate  = bf16(sigmoid(g)); s_k = bf16(bf16(gate*s) + bf16(bf16(1-gate)*bf16(tanh c)))
        lat   = bf16(tanh(o))
        bias  = bf16(lat @ (lm_head W @ hidden_out W)^T + lm_head W @ hidden_out b)
                (the lm_head-folded hidden_out, folded in fp64 then stored bf16)
      logits_k = bf16(30 tanh(bf16(base_k + bias)/30)); vocab >= 200058 masked
      d_k = argmax(logits_k) (lowest index on ties); prev_tok_1 = anchor, prev_tok_k = d_{k-1}.
    The 7 drafts are then verified by the target in one step that also yields one
    bonus token, so tokens / verify step = accepted + 1.

Known deviations from the GPU reference (all at bf16-rounding level):
  * Linear layers use torch CPU bf16 matmul (fp32 accumulate, bf16 output) like cuBLAS,
    but reduction order differs.
  * Residual add + RMSNorm: the residual is rounded to bf16 and the norm reads the
    rounded value (SGLang's fused_add_rmsnorm may normalise the unrounded fp32 sum).
  * RoPE is applied in fp32 and rounded once (SGLang's bf16 cos/sin cache rounds
    cos/sin first).  q_norm/k_norm/RMSNorm compute in fp32 and round once.
  * Attention softmax in fp32 (FA4 also accumulates in fp32).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

BF16 = torch.bfloat16
MODEL_ID = "thinkingmachines/Inkling-2-Mini"
DEFAULT_TEXTS = "/home/raahem/.cache/inkling2/fixtures/texts.json"


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(BF16).float()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 RMSNorm with a single bf16 rounding (SGLang RMSNorm / flashinfer)."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * weight.float()).to(BF16)


def mix_rmsnorm(taps: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """kernels/ops/speculative/inkling2_mtp.py::_mix_rmsnorm. taps [..., n_taps, H] bf16,
    alpha [n_taps] bf16. fp32 weighted sum, fp32 rms, one bf16 store."""
    mixed = (taps.float() * alpha.float()[:, None]).sum(-2)
    return rms_norm(mixed, weight, eps)


def rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [T, heads, D] bf16; cos/sin [T, D/2] fp32. Neox (rotate-half) style."""
    xf = x.float()
    half = xf.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(BF16)


@dataclass
class MTPConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    local_layer_ids: list
    sliding_window: int
    block_size: int
    markov_rank: int
    target_layer_ids: list
    final_logit_softcapping: float
    vocab_limit: int  # unpadded vocab; step logits from here on are -inf

    @classmethod
    def from_hf(cls, config) -> "MTPConfig":
        m = config.mtp_config
        text = config.text_config
        assert m["markov_head_type"] == "rnn_hidden"
        return cls(
            hidden_size=m["hidden_size"],
            num_layers=m["num_hidden_layers"],
            num_heads=m["num_attention_heads"],
            num_kv_heads=m["num_key_value_heads"],
            head_dim=m["head_dim"],
            rms_norm_eps=m["rms_norm_eps"],
            rope_theta=m["rope_theta"],
            local_layer_ids=list(m["local_layer_ids"]),
            sliding_window=m["sliding_window_size"],
            block_size=m["block_size"],
            markov_rank=m["markov_rank"],
            target_layer_ids=list(m["target_layer_ids"]),
            final_logit_softcapping=text.final_logit_softcapping,
            vocab_limit=text.unpadded_vocab_size,
        )


class DSparkDrafter:
    """Inkling2 MTP / DSpark draft model (plain torch, CPU, bf16 weights).

    Shared modules borrowed from the target (as SGLang's attach_shared_modules):
      embed_weight [V_pad, H] + embed_norm_weight [H]  (get_draft_input_embeddings
      = embed_norm(embed_tokens(ids))), lm_head_weight [V, H].
    """

    def __init__(
        self,
        cfg: MTPConfig,
        mtp_path: str,
        embed_weight: torch.Tensor,
        embed_norm_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_eps: float,
        max_positions: int = 4096,
    ):
        self.cfg = cfg
        w = load_file(mtp_path)
        for k, v in w.items():
            assert v.dtype == BF16, k
        self.w = w
        self.gamma = cfg.block_size - 1
        self.embed_weight = embed_weight
        self.embed_norm_weight = embed_norm_weight
        self.target_eps = target_eps
        self.lm_head = lm_head_weight.to(BF16).contiguous()
        # alpha = softmax(fuse_weight.float()).to(bf16): row i for draft layer i, row -1 for f_rnn.
        self.fuse_alpha = torch.softmax(w["fuse_weight"].float(), dim=-1).to(BF16)
        assert self.fuse_alpha.shape == (cfg.num_layers + 1, len(cfg.target_layer_ids))
        # Inkling2MTPHead.refresh_fold: fold hidden_out into the lm_head in fp64, store bf16.
        proj = w["markov_head.hidden_out.weight"].double()  # [H, R]
        bias = w["markov_head.hidden_out.bias"].double()  # [H]
        V = self.lm_head.shape[0]
        self.fold_weight = torch.empty(V, cfg.markov_rank, dtype=BF16)
        self.fold_bias = torch.empty(V, dtype=BF16)
        for s in range(0, V, 8192):
            head = self.lm_head[s : s + 8192].double()
            self.fold_weight[s : s + 8192] = (head @ proj).to(BF16)
            self.fold_bias[s : s + 8192] = (head @ bias).to(BF16)
        # RoPE tables (fp32), neox, full rotary dim.
        d = cfg.head_dim
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
        ang = torch.arange(max_positions, dtype=torch.float64)[:, None] * inv_freq[None, :]
        self.cos = ang.cos().float()
        self.sin = ang.sin().float()

    # ------------------------------------------------------------------ context
    def context_kv(self, taps: torch.Tensor, positions: torch.Tensor):
        """write_target_hidden_kv. taps [T, n_taps, H] bf16 -> per layer (K [T, kv, D] roped, V)."""
        cfg, w = self.cfg, self.w
        out = []
        for i in range(cfg.num_layers):
            p = f"layers.{i}."
            ctx = mix_rmsnorm(taps, self.fuse_alpha[i], w[p + "dflare_norm.weight"], cfg.rms_norm_eps)
            k = F.linear(ctx, w[p + "self_attn.k_proj_ctx.weight"]).view(-1, cfg.num_kv_heads, cfg.head_dim)
            v = F.linear(ctx, w[p + "self_attn.v_proj_ctx.weight"]).view(-1, cfg.num_kv_heads, cfg.head_dim)
            k = rms_norm(k, w[p + "self_attn.k_norm_ctx.weight"], cfg.rms_norm_eps)
            k = rope_neox(k, self.cos[positions], self.sin[positions])
            out.append((k, v))
        return out

    def f_rnn(self, last_taps: torch.Tensor) -> torch.Tensor:
        """compute_f_rnn: taps of the last committed position [n_taps, H] -> [H] bf16."""
        return mix_rmsnorm(
            last_taps, self.fuse_alpha[-1], self.w["markov_head.dflare_norm.weight"], self.cfg.rms_norm_eps
        )

    # ------------------------------------------------------------------- block
    def block_hidden(self, ctx_kv, ctx_len: int, anchor_token: int) -> torch.Tensor:
        """DFlashDraftModel.forward for one block at positions ctx_len .. ctx_len+7.
        ctx_kv: per-layer (K, V) for positions >= 0 (only the first ctx_len rows are used).
        Returns the final normed hidden [block_size, H] bf16 (row 0 = anchor row)."""
        cfg, w = self.cfg, self.w
        B = cfg.block_size
        # forward_embed: row 0 = normed target embedding of the anchor, rest = mask_embed.
        hidden = w["mask_embed"].expand(B, -1).clone()
        hidden[0] = rms_norm(self.embed_weight[anchor_token].float(), self.embed_norm_weight, self.target_eps)
        pos = torch.arange(ctx_len, ctx_len + B)
        cos, sin = self.cos[pos], self.sin[pos]
        rep = cfg.num_heads // cfg.num_kv_heads
        residual = None
        for i in range(cfg.num_layers):
            p = f"layers.{i}."
            if residual is None:
                residual = hidden
                x = rms_norm(hidden, w[p + "input_layernorm.weight"], cfg.rms_norm_eps)
            else:
                residual = (hidden.float() + residual.float()).to(BF16)
                x = rms_norm(residual, w[p + "input_layernorm.weight"], cfg.rms_norm_eps)
            q = F.linear(x, w[p + "self_attn.q_proj.weight"]).view(B, cfg.num_heads, cfg.head_dim)
            k = F.linear(x, w[p + "self_attn.k_proj.weight"]).view(B, cfg.num_kv_heads, cfg.head_dim)
            v = F.linear(x, w[p + "self_attn.v_proj.weight"]).view(B, cfg.num_kv_heads, cfg.head_dim)
            q = rope_neox(rms_norm(q, w[p + "self_attn.q_norm.weight"], cfg.rms_norm_eps), cos, sin)
            k = rope_neox(rms_norm(k, w[p + "self_attn.k_norm.weight"], cfg.rms_norm_eps), cos, sin)
            kc, vc = ctx_kv[i]
            start = 0
            if i in cfg.local_layer_ids:
                # Sliding window anchored at the block start (window_anchor_at_start):
                # every block row sees context positions >= ctx_len - window.
                start = max(0, ctx_len - cfg.sliding_window)
            K = torch.cat([kc[start:ctx_len], k], 0).float()  # [S, kv, D]
            V = torch.cat([vc[start:ctx_len], v], 0).float()
            K = K.repeat_interleave(rep, dim=1).transpose(0, 1)  # [heads, S, D]
            V = V.repeat_interleave(rep, dim=1).transpose(0, 1)
            Q = q.float().transpose(0, 1)  # [heads, B, D]
            scores = (Q @ K.transpose(-1, -2)) * (cfg.head_dim**-0.5)
            attn = torch.softmax(scores, dim=-1) @ V  # non-causal: block rows see all block keys
            attn = attn.transpose(0, 1).reshape(B, cfg.num_heads * cfg.head_dim).to(BF16)
            hidden = F.linear(attn, w[p + "self_attn.o_proj.weight"])
            residual = (hidden.float() + residual.float()).to(BF16)
            x = rms_norm(residual, w[p + "post_attention_layernorm.weight"], cfg.rms_norm_eps)
            gate = F.linear(x, w[p + "mlp.gate_proj.weight"]).float()
            up = F.linear(x, w[p + "mlp.up_proj.weight"]).float()
            hidden = F.linear((F.silu(gate) * up).to(BF16), w[p + "mlp.down_proj.weight"])
        residual = (hidden.float() + residual.float()).to(BF16)
        return rms_norm(residual, w["norm.weight"], cfg.rms_norm_eps)

    # ------------------------------------------------------------------ markov
    def base_logits(self, draft_hidden: torch.Tensor) -> torch.Tensor:
        """compute_base_logits: bf16 [gamma, V] (no muP multiplier for Inkling-2-Mini)."""
        return F.linear(draft_hidden.to(BF16), self.lm_head)

    def rnn_step(self, state, prev_token: int, hidden, f_rnn):
        """Inkling2MTPHead._rnn_step (rnn_inputs + rnn_proj + rnn_gates), returns (state', latent)."""
        w = self.w
        R = self.cfg.markov_rank
        inputs = torch.cat([state, w["markov_head.markov_w1.weight"][prev_token], hidden, f_rnn], -1)
        proj = F.linear(inputs[None], w["markov_head.rnn_proj.weight"], w["markov_head.rnn_proj.bias"])[0].float()
        gate_raw, cand_raw, out_raw = proj[:R], proj[R : 2 * R], proj[2 * R :]
        s = state.float()
        gate = _bf(torch.sigmoid(gate_raw))
        kept = _bf(gate * s)
        blend = _bf(_bf(1.0 - gate) * _bf(torch.tanh(cand_raw)))
        new_state = (kept + blend).to(BF16)
        latent = torch.tanh(out_raw).to(BF16)
        return new_state, latent

    def step_bias(self, latent: torch.Tensor) -> torch.Tensor:
        """addmm(fold_bias, latent, fold_weight.T) in bf16 (fp32 accumulate, one rounding)."""
        return (self.fold_bias.float() + latent.float() @ self.fold_weight.float().T).to(BF16)

    def step_logits(self, base: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        """MarkovSampleStep.torch: bf16(base+bias) -> bf16(softcap) -> vocab mask. fp32 out."""
        value = base.float()
        if bias is not None:
            value = value + bias.float()
        value = _bf(value)
        cap = self.cfg.final_logit_softcapping
        if cap:
            value = _bf(torch.tanh(value / cap) * cap)
        if value.shape[-1] > self.cfg.vocab_limit:
            value[..., self.cfg.vocab_limit :] = float("-inf")
        return value

    def draft_from_hidden(self, block_hidden, anchor_token: int, f_rnn, use_markov: bool = True):
        """draft_markov_block (greedy). Returns (tokens [gamma] long, step logits [gamma, V] fp32)."""
        draft_hidden = block_hidden[1:]  # select_draft_hidden_without_anchor
        base = self.base_logits(draft_hidden)
        if not use_markov:
            logits = self.step_logits(base, None)
            return logits.argmax(-1), logits
        state = torch.zeros(self.cfg.markov_rank, dtype=BF16)
        prev = int(anchor_token)
        tokens, all_logits = [], []
        for k in range(self.gamma):
            state, latent = self.rnn_step(state, prev, draft_hidden[k], f_rnn)
            logits = self.step_logits(base[k], self.step_bias(latent))
            prev = int(logits.argmax())  # torch argmax: first max = lowest index on ties
            tokens.append(prev)
            all_logits.append(logits)
        return torch.tensor(tokens), torch.stack(all_logits)

    # ---------------------------------------------------------------- top-level
    def draft(self, taps: torch.Tensor, anchor_token: int, use_markov: bool = True, ctx_kv=None):
        """taps: [p, n_taps, H] bf16 target taps of the committed positions 0..p-1.
        anchor_token: the token at position p (target's bonus token).
        Returns (7 greedy draft tokens for positions p+1..p+7, per-step logits [7, V])."""
        p = taps.shape[0]
        if ctx_kv is None:
            ctx_kv = self.context_kv(taps, torch.arange(p))
        h = self.block_hidden(ctx_kv, p, anchor_token)
        return self.draft_from_hidden(h, anchor_token, self.f_rnn(taps[p - 1]), use_markov)


# =============================================================================
# Measurement
# =============================================================================


def tap_visit_map(text_cfg, target_layer_ids):
    """visit -> (physical layer, pass index) for the looped target."""
    start, end, passes = text_cfg.loop_layer_index_start, text_cfg.loop_layer_index_end, text_cfg.loop_num_passes
    core = end - start
    out = {}
    for v in target_layer_ids:
        if v < start:
            out[v] = (v, 0)
        elif v < start + passes * core:
            out[v] = (start + (v - start) % core, (v - start) // core)
        elif v < text_cfg.num_hidden_layers + (passes - 1) * core:
            out[v] = (v - (passes - 1) * core, 0)
        # v == num_visits: final normed state, handled separately
    return out


class TargetRunner:
    def __init__(self, model, target_layer_ids):
        self.model = model
        self.lm = model.model.language_model
        text_cfg = model.config.text_config
        self.n_visits = text_cfg.num_hidden_layers + (text_cfg.loop_num_passes - 1) * (
            text_cfg.loop_layer_index_end - text_cfg.loop_layer_index_start
        )
        assert target_layer_ids[-1] == self.n_visits, (target_layer_ids, self.n_visits)
        self.visit_map = tap_visit_map(text_cfg, target_layer_ids[:-1])
        self.target_layer_ids = target_layer_ids
        self.calls = {}
        self.captured = {}
        want = {}
        for v, (layer, pidx) in self.visit_map.items():
            want.setdefault(layer, {})[pidx] = v
        for layer, passes in want.items():
            self.lm.layers[layer].register_forward_pre_hook(self._make_hook(layer, passes), with_kwargs=True)

    def _make_hook(self, layer, passes):
        def hook(module, args, kwargs):
            idx = self.calls.get(layer, 0)
            self.calls[layer] = idx + 1
            if idx in passes:
                hs = args[0] if args else kwargs["hidden_states"]
                self.captured[passes[idx]] = hs.detach()[0].to(BF16).clone()

        return hook

    @torch.no_grad()
    def run(self, ids: torch.Tensor):
        """ids [T]. Returns (taps [T, 4, H] bf16, greedy argmax [T] (prediction of token t+1))."""
        self.calls, self.captured = {}, {}
        out = self.lm(input_ids=ids[None], use_cache=False)
        final = out.last_hidden_state[0].to(BF16)
        taps = torch.stack([self.captured[v] for v in self.target_layer_ids[:-1]] + [final], dim=1)
        logits = self.model.lm_head(final).float()
        logits = self.model.logit_softcap(logits)
        self.last_logits = logits  # [T, V] fp32 (softcapped): the sampled-acceptance estimate
        return taps, logits.argmax(-1)

    @torch.no_grad()
    def _step(self, ids: torch.Tensor, cache):
        self.calls, self.captured = {}, {}
        out = self.lm(input_ids=ids[None], cache=cache, use_cache=True)
        final = out.last_hidden_state[0].to(BF16)
        taps = torch.stack([self.captured[v] for v in self.target_layer_ids[:-1]] + [final], dim=1)
        logits = self.model.logit_softcap(self.model.lm_head(final).float())
        return taps, logits.argmax(-1), out.cache

    @torch.no_grad()
    def run_greedy_windows(self, ids: torch.Tensor, anchors, window: int, rewrite: bool = True):
        """Incremental (KV-cached) target pass that splices the target's own greedy
        decode into the text after every anchor: chunk-prefill text up to position a,
        then greedy-decode positions a+1 .. a+window-1 one token at a time (position
        a+window gets the greedy token too and starts the next chunk).  Every compared
        target token is then exactly the target's greedy continuation of its prefix.
        rewrite=False feeds the original text through the same chunking (consistency check).
        Returns (seq, taps [T, 4, H], greedy [T])."""
        seq = ids.clone()
        T = seq.shape[0]
        taps = [None] * T
        greedy = torch.zeros(T, dtype=torch.long)
        cache, pos = None, 0

        def feed(lo, hi):
            nonlocal cache
            t, g, cache = self._step(seq[lo:hi], cache)
            for i in range(hi - lo):
                taps[lo + i] = t[i]
            greedy[lo:hi] = g

        for a in anchors:
            feed(pos, a + 1)
            for j in range(window - 1):  # positions a+1 .. a+window-1
                if rewrite:
                    seq[a + 1 + j] = greedy[a + j]
                feed(a + 1 + j, a + 2 + j)
            if rewrite:
                seq[a + window] = greedy[a + window - 1]
            pos = a + window
        if pos < T:
            feed(pos, T)
        return seq, torch.stack(taps), greedy


def accepted_len(draft: torch.Tensor, target: torch.Tensor) -> int:
    n = 0
    for d, t in zip(draft.tolist(), target.tolist()):
        if d != t:
            break
        n += 1
    return n


def summarize(name, accs, gamma):
    hist = [0] * (gamma + 1)
    for a in accs:
        hist[a] += 1
    n = len(accs)
    mean = sum(accs) / n
    # per-depth conditional acceptance P(d_k ok | d_1..d_{k-1} ok)
    cond = []
    for k in range(1, gamma + 1):
        reach = sum(1 for a in accs if a >= k - 1)
        ok = sum(1 for a in accs if a >= k)
        cond.append(ok / reach if reach else float("nan"))
    return mean, hist, cond


def fmt_row(name, accs, gamma):
    mean, hist, cond = summarize(name, accs, gamma)
    return (
        f"  {name:<24s} n={len(accs):4d}  accepted={mean:5.3f}  tokens/step={mean + 1:5.3f}  "
        f"hist[0..{gamma}]={hist}  P(k|k-1)=[{', '.join(f'{c:.2f}' for c in cond)}]"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=DEFAULT_TEXTS)
    ap.add_argument("--only", default=None, help="comma-separated text names")
    ap.add_argument("--num-anchors", type=int, default=40)
    ap.add_argument("--min-anchor", type=int, default=64)
    ap.add_argument("--tf-only", action="store_true", help="skip the greedy-window measurement")
    ap.add_argument("--check-incremental", action="store_true", help="compare cached chunked pass with the full forward")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/opt/tenstorrent/models")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch.set_num_threads(args.threads)
    from transformers import Inkling2LoopForConditionalGeneration

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    model = Inkling2LoopForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.eval()
    log(f"target loaded in {time.time() - t0:.0f}s")
    cfg = MTPConfig.from_hf(model.config)
    snap = glob.glob(f"{os.environ['HF_HOME']}/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/*/mtp.safetensors")
    assert len(snap) == 1, snap
    lm = model.model.language_model
    drafter = DSparkDrafter(
        cfg,
        snap[0],
        embed_weight=lm.embed_tokens.weight.data,
        embed_norm_weight=lm.embed_tokens.embed_norm.weight.data,
        lm_head_weight=model.lm_head.weight.data,
        target_eps=model.config.text_config.rms_norm_eps,
    )
    runner = TargetRunner(model, cfg.target_layer_ids)
    gamma = drafter.gamma
    B = cfg.block_size
    log(f"mtp: {cfg}")
    log(f"tap visits -> (layer, pass): {runner.visit_map}, {cfg.target_layer_ids[-1]} = final norm")

    texts = json.load(open(args.texts))
    if args.only:
        texts = {k: texts[k] for k in args.only.split(",")}

    # results[mode][text] = list of accepted lengths
    modes = ["tf_markov", "tf_base", "greedy_markov", "greedy_base"]
    samp_all = {}
    results = {m: {} for m in modes}

    def sampled_accept(d_logits, t_logits):
        """Expected accepted drafts under speculative sampling at temperature 1 (rejection sampling, draft q vs target
        p): sum_k prod_{j<=k} alpha_j, alpha_j = sum_x min(p_j(x), q_j(x)). An estimate: alpha_j uses the target
        conditioned on the text prefix and the draft conditioned on its greedy path."""
        q = torch.softmax(d_logits.float(), -1)
        pt = torch.softmax(t_logits.float(), -1)
        alpha = torch.minimum(q, pt).sum(-1)
        return float(torch.cumprod(alpha, 0).sum())

    def eval_anchors(ids, taps, greedy, anchors, tag, consistent_mask=None, t_logits=None, samp=None):
        """Context = taps of positions 0..a, anchor token = ids[a+1] at position a+1,
        drafts for positions a+2..a+8 compared with greedy[a+1..a+7]."""
        ctx_kv = drafter.context_kv(taps, torch.arange(taps.shape[0]))
        acc_m, acc_b = [], []
        for j, a in enumerate(anchors):
            if consistent_mask is not None and not consistent_mask[j]:
                continue
            p = a + 1
            anchor = int(ids[p])
            h = drafter.block_hidden(ctx_kv, p, anchor)
            f_rnn = drafter.f_rnn(taps[p - 1])
            target = greedy[p : p + gamma]
            d_m, lg_m = drafter.draft_from_hidden(h, anchor, f_rnn, use_markov=True)
            if t_logits is not None and samp is not None:
                samp.append(sampled_accept(lg_m, t_logits[p : p + gamma]))
            d_b, _ = drafter.draft_from_hidden(h, anchor, f_rnn, use_markov=False)
            acc_m.append(accepted_len(d_m, target))
            acc_b.append(accepted_len(d_b, target))
        return acc_m, acc_b

    for name, toks in texts.items():
        ids = torch.tensor(toks, dtype=torch.long)
        T = ids.shape[0]
        anchors = torch.linspace(args.min_anchor, T - B - 2, args.num_anchors).round().long().tolist()
        assert all(b - a > B for a, b in zip(anchors, anchors[1:])), "windows must not overlap"
        t1 = time.time()
        taps, greedy = runner.run(ids)
        log(f"[{name}] pass 1 (teacher forced) forward {time.time() - t1:.0f}s")
        # --- teacher-forced (one forward, as specified): anchor = text token
        samp = []
        acc_m, acc_b = eval_anchors(ids, taps, greedy, anchors, "tf", t_logits=runner.last_logits, samp=samp)
        results["tf_markov"][name], results["tf_base"][name] = acc_m, acc_b
        samp_all[name] = samp
        log(f"  {name}/tf markov sampled-T1 estimate: accepted={sum(samp) / len(samp):.3f}")
        log(fmt_row(f"{name}/tf markov", acc_m, gamma))
        log(fmt_row(f"{name}/tf base", acc_b, gamma))
        if args.tf_only:
            continue
        if args.check_incremental:
            t1 = time.time()
            _, taps_i, greedy_i = runner.run_greedy_windows(ids, anchors, B, rewrite=False)
            dt = (taps_i.float() - taps.float()).abs().amax(dim=(0, 2))
            log(
                f"[{name}] incremental-vs-full check ({time.time() - t1:.0f}s): greedy agreement "
                f"{(greedy_i == greedy).float().mean():.4f}, max |tap diff| per tap {dt.tolist()}"
            )
        # --- greedy: splice the target's own greedy continuation after every anchor
        # (anchor = bonus token argmax at a, then the greedy chain), KV-cached, so every
        # compared target token is exactly the target's greedy decode of its prefix.
        t1 = time.time()
        seq, taps, greedy = runner.run_greedy_windows(ids, anchors, B)
        exact = [bool((seq[a + 1 : a + 1 + gamma] == greedy[a : a + gamma]).all()) for a in anchors]
        log(f"[{name}] greedy-window pass {time.time() - t1:.0f}s, {sum(exact)}/{len(anchors)} windows consistent")
        acc_m, acc_b = eval_anchors(seq, taps, greedy, anchors, "greedy", exact)
        results["greedy_markov"][name], results["greedy_base"][name] = acc_m, acc_b
        log(fmt_row(f"{name}/greedy markov", acc_m, gamma))
        log(fmt_row(f"{name}/greedy base", acc_b, gamma))

    log("")
    log("=" * 100)
    log("SUMMARY (accepted = leading drafts equal to target greedy; tokens/step = accepted + 1)")
    for m in modes:
        if not results[m]:
            continue
        log(f"mode {m}:")
        allacc = []
        for name, accs in results[m].items():
            log(fmt_row(name, accs, gamma))
            allacc += accs
        log(fmt_row("ALL (pooled)", allacc, gamma))
        per_text = [sum(a) / len(a) for a in results[m].values() if a]
        log(f"  mean of per-text means: accepted={sum(per_text) / len(per_text):.3f}")
    if samp_all:
        pooled = [x for v in samp_all.values() for x in v]
        log(f"sampled (temperature 1) estimate, tf markov, pooled: accepted={sum(pooled) / len(pooled):.3f}")
    log(f"total time {time.time() - t0:.0f}s")
    if args.report:
        with open(args.report, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
