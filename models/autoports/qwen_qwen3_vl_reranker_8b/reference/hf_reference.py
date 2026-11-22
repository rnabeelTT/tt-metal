# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF reference scorer: ``Qwen3VLForConditionalGeneration`` on CPU, score = h_last . (w_yes - w_no).

Same as the official reranker's ``score_linear`` (a 1-row linear built from the two lm_head rows); the relevance
probability is ``sigmoid(score)``.
"""
from __future__ import annotations

import torch

from models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs import NO_TOKEN, YES_TOKEN, PairInputs, model_path


class HFReranker:
    def __init__(self, path: str | None = None, dtype=torch.float32, num_layers: int | None = None):
        from transformers import AutoConfig, Qwen3VLForConditionalGeneration

        path = path or model_path()
        cfg = AutoConfig.from_pretrained(path)
        if num_layers is not None:
            cfg.text_config.num_hidden_layers = num_layers
        lm = Qwen3VLForConditionalGeneration.from_pretrained(path, config=cfg, dtype=dtype).eval()
        self.config = cfg
        self.model = lm.model
        w = lm.lm_head.weight.data
        self.score_vec = (w[YES_TOKEN].float() - w[NO_TOKEN].float()).to(dtype)
        del lm.lm_head
        self.dtype = dtype

    @torch.no_grad()
    def hidden(self, p: PairInputs, output_hidden_states=False):
        kw = p.hf_kwargs()
        if kw.get("pixel_values") is not None:
            kw["pixel_values"] = kw["pixel_values"].to(self.dtype)
        if kw.get("pixel_values_videos") is not None:
            kw["pixel_values_videos"] = kw["pixel_values_videos"].to(self.dtype)
        return self.model(**kw, output_hidden_states=output_hidden_states)

    @torch.no_grad()
    def score(self, p: PairInputs) -> float:
        h = self.hidden(p).last_hidden_state[0, -1]
        return float(h.float() @ self.score_vec.float())
