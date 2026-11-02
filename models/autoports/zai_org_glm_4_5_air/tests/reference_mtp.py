# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch (CPU, bf16) reference of the GLM-4.5-Air multi-token-prediction head, for the MTP bring-up tests.

Math (vLLM ``glm4_moe_mtp.py``; transformers ignores checkpoint layer 46):

    h0 = eh_proj(cat(enorm(embed(tok)), hnorm(hidden)))      # hidden = main model's FINAL-NORMED state
    h1 = decoder_layer(h0, position)     # one full-attention MoE layer with its own KV cache (layers.46.*)
    logits = lm_head(shared_head.norm(h1))   # embeddings and LM head = the main model's (identical in the checkpoint)

Weights come from :meth:`StreamingStateDict.mtp_state` in the Qwen3.5 autoport's MTP scheme (``mtp.fc`` = eh_proj,
``mtp.pre_fc_norm_embedding`` = enorm, ``mtp.pre_fc_norm_hidden`` = hnorm, ``mtp.norm`` = shared_head.norm; experts
fused like the main layers). The norms are PLAIN RMSNorm (``weight * x_normed``, HF ``Glm4MoeRMSNorm``). The layer is
the HF ``Glm4MoeDecoderLayer`` built as virtual layer index ``n_layers`` (= 46, the checkpoint index too) of a text
config whose ``layer_types`` gained one ``full_attention`` entry, so the stage-1 reference harness (rotary tables,
DynamicCache) applies unchanged.
"""
from __future__ import annotations

import copy

import torch
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeRMSNorm

from models.autoports.zai_org_glm_4_5_air.tests import reference as ref
from models.autoports.zai_org_glm_4_5_air.tt.weights import StreamingStateDict, resolve_checkpoint_dir


def rms_norm(weight: torch.Tensor, eps: float) -> Glm4MoeRMSNorm:
    """GLM RMSNorm (plain: weight * x_normed) holding ``weight`` in bf16."""
    norm = Glm4MoeRMSNorm(weight.shape[-1], eps=eps)
    norm.weight = torch.nn.Parameter(weight.to(torch.bfloat16).clone(), requires_grad=False)
    return norm


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
        self.layer = ref.torch_layer(self.text, self.layer_idx, st)  # mtp.layers.0.* only (mtp.fc / norms skipped)
        eps = text.rms_norm_eps
        self.fc = torch.nn.Linear(2 * text.hidden_size, text.hidden_size, bias=False)
        self.fc.weight = torch.nn.Parameter(st["mtp.fc.weight"].to(torch.bfloat16), requires_grad=False)
        self.pre_fc_norm_embedding = rms_norm(st["mtp.pre_fc_norm_embedding.weight"], eps)  # enorm
        self.pre_fc_norm_hidden = rms_norm(st["mtp.pre_fc_norm_hidden.weight"], eps)  # hnorm
        self.norm = rms_norm(st["mtp.norm.weight"], eps)  # shared_head.norm
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
    sd = StreamingStateDict(checkpoint_dir or resolve_checkpoint_dir())
    return MtpReference(text, sd)
