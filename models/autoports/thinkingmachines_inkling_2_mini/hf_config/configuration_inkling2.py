# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A config class vLLM 0.26 / transformers 5.12 can load for Inkling-2-Mini (``--hf-config-path`` this directory,
``--trust-remote-code``). Only what vLLM's scheduler reads (sizes, vocab, max length, architecture); the TT model reads
the checkpoint's own config.json."""

from transformers import PretrainedConfig


class Inkling2LoopTTConfig(PretrainedConfig):
    model_type = "inkling2_loop_tt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
