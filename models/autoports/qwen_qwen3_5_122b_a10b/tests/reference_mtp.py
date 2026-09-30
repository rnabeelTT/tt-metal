# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch (CPU, bf16) reference of the Qwen3.5-122B-A10B multi-token-prediction head, for the MTP bring-up tests.

Math (vLLM ``qwen3_5_mtp.py``; transformers ignores ``mtp.*``):

    h0 = fc(cat(pre_fc_norm_embedding(embed(tok)), pre_fc_norm_hidden(hidden)))      # hidden = main model's FINAL-NORMED state
    h1 = decoder_layer(h0, position)     # one full-attention MoE layer with its own KV cache (mtp.layers.0.*)
    logits = lm_head(norm(h1))            # embeddings and LM head shared with the main model

Weights come from :meth:`StreamingStateDict.mtp_state` (experts fused like the main layers). The layer is the HF
``Qwen3_5MoeDecoderLayer`` built as virtual layer index ``n_layers`` of a text config whose ``layer_types`` gained one
``full_attention`` entry, so the stage-1 reference harness (rotary tables, DynamicCache) applies unchanged.
"""
from __future__ import annotations

import copy

import torch

from models.autoports.qwen_qwen3_5_122b_a10b.tests import reference as ref
from models.autoports.qwen_qwen3_5_122b_a10b.tt.weights import StreamingStateDict, resolve_checkpoint_dir


class RMSNormZeroCentered(torch.nn.Module):
    """Qwen3.5 RMSNorm: x_normed * (1 + weight)."""

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (1.0 + self.weight.float())).to(x.dtype)


def mtp_text_config(text):
    """Text config copy with the MTP layer appended as layer index ``n_layers`` (a full-attention layer)."""
    cfg = copy.deepcopy(text)
    cfg.layer_types = list(text.layer_types) + ["full_attention"]
    cfg.num_hidden_layers = len(cfg.layer_types)
    return cfg


class MtpReference(torch.nn.Module):
    def __init__(self, text, state_dict: StreamingStateDict, embed_weight=None, lm_head_weight=None):
        super().__init__()
        self.text = mtp_text_config(text)
        self.layer_idx = len(text.layer_types)
        st = state_dict.mtp_state(fuse_experts=True)
        prefix = "mtp.layers.0."
        layer_state = {k[len(prefix) :]: v for k, v in st.items() if k.startswith(prefix)}
        self.layer = ref.torch_layer(self.text, self.layer_idx, layer_state)
        eps = text.rms_norm_eps
        self.fc = torch.nn.Linear(2 * text.hidden_size, text.hidden_size, bias=False)
        self.fc.weight = torch.nn.Parameter(st["mtp.fc.weight"].to(torch.bfloat16), requires_grad=False)
        self.pre_fc_norm_embedding = RMSNormZeroCentered(st["mtp.pre_fc_norm_embedding.weight"], eps)
        self.pre_fc_norm_hidden = RMSNormZeroCentered(st["mtp.pre_fc_norm_hidden.weight"], eps)
        self.norm = RMSNormZeroCentered(st["mtp.norm.weight"], eps)
        self.embed_weight = state_dict["tok_embeddings.weight"] if embed_weight is None else embed_weight
        self.lm_head_weight = state_dict["output.weight"] if lm_head_weight is None else lm_head_weight
        self.eval()

    def embed(self, tok_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(tok_ids, self.embed_weight).to(torch.bfloat16)

    @torch.no_grad()
    def mix(self, tok_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """[B, T] ids + [B, T, H] final-normed hidden -> the MTP layer input [B, T, H]."""
        e = self.pre_fc_norm_embedding(self.embed(tok_ids))
        h = self.pre_fc_norm_hidden(hidden.to(torch.bfloat16))
        return self.fc(torch.cat([e, h], dim=-1))

    @torch.no_grad()
    def forward(self, tok_ids, hidden, positions, cache, attention_mask=None):
        """One MTP step over T rows: returns (logits [B, T, vocab] float32, mtp_hidden [B, T, H] bf16 post-norm)."""
        x = self.mix(tok_ids, hidden)
        pe = ref.rotary(self.text, x, positions)
        out = self.layer(x, position_embeddings=pe, attention_mask=attention_mask, past_key_values=cache)
        out = out[0] if isinstance(out, tuple) else out
        h = self.norm(out)
        logits = h.float() @ self.lm_head_weight.float().T
        return logits, h

    @torch.no_grad()
    def draft(self, tok_ids, hidden, positions, cache, k: int):
        """Chained greedy drafts: k tokens from (tok, hidden) at `positions` (each step feeds its own hidden)."""
        drafts = []
        tok, hid, pos = tok_ids, hidden, positions
        for _ in range(k):
            logits, hid = self.forward(tok, hid, pos, cache)
            tok = logits[:, -1:].argmax(-1)
            drafts.append(tok)
            hid = hid[:, -1:]
            pos = pos[:, -1:] + 1
        return torch.cat(drafts, dim=1)


def load_reference(text=None, checkpoint_dir=None) -> MtpReference:
    text = text or ref.text_config()[1]
    sd = StreamingStateDict(
        checkpoint_dir or resolve_checkpoint_dir(),
        linear_q_dim=text.linear_num_key_heads * text.linear_key_head_dim,
        linear_k_dim=text.linear_num_key_heads * text.linear_key_head_dim,
    )
    return MtpReference(text, sd)
