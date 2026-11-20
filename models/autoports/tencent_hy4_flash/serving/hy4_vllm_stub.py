# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A vLLM model-interface stub for HYV4FlashForConditionalGeneration (vLLM 0.26 has no implementation of hy_v4_flash).

vLLM validates the checkpoint's architecture (ModelConfig: a text-generation model class) before the TT plugin maps
the request to its TT-prefixed registration (models.autoports.tencent_hy4_flash.tt.generator_vllm, the device engine).
This class only satisfies that inspection; it is never instantiated on a TT platform."""

from __future__ import annotations

import torch
from torch import nn


class HYV4FlashForConditionalGeneration(nn.Module):
    supports_multimodal = False  # HY4_VISION=1 (default): image inputs (hy4_mm.register below)

    def __init__(self, *, vllm_config=None, prefix: str = "") -> None:
        super().__init__()
        raise RuntimeError("hy_v4_flash runs on the TT platform only (the TT plugin's TTHYV4FlashForConditionalGeneration)")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_tensors=None,
                inputs_embeds=None) -> torch.Tensor:
        raise NotImplementedError

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


try:
    import hy4_mm
except ImportError:  # imported as a package module
    from models.autoports.tencent_hy4_flash.serving import hy4_mm
if hy4_mm.enabled():
    HYV4FlashForConditionalGeneration = hy4_mm.register(HYV4FlashForConditionalGeneration)
