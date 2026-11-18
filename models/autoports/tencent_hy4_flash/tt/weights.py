# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lazy reads of the official Hy4-flash safetensors (one tensor at a time, no whole-shard loads).

`Hy4Weights(ckpt_dir).get("model.layers.3.self_attn.q_a_proj.weight")` returns a torch tensor in the checkpoint's
dtype. `layer(li)` returns a dict view of one decoder layer's tensors with the `model.layers.{li}.` prefix stripped.
The Engram embedding tables are never read here (the host store, models/experimental/hybrid_moe/engram_store.py,
serves them); asking for one raises.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

import torch
from safetensors import safe_open

from models.autoports.tencent_hy4_flash.tt.config import checkpoint_dir


class Hy4Weights:
    def __init__(self, ckpt_dir: str | None = None):
        self.dir = ckpt_dir or checkpoint_dir()
        self.map = json.load(open(os.path.join(self.dir, "model.safetensors.index.json")))["weight_map"]
        self._open = {}

    def _f(self, shard):
        f = self._open.get(shard)
        if f is None:
            if len(self._open) > 16:  # bounded number of open mmaps
                self._open.clear()
            f = self._open[shard] = safe_open(os.path.join(self.dir, shard), framework="pt")
        return f

    def has(self, name: str) -> bool:
        return name in self.map

    def get(self, name: str) -> torch.Tensor:
        if name.endswith("multi_head_embedding.embedding.weight"):
            raise RuntimeError(f"{name}: Engram tables are served by the host store, not loaded as tensors")
        return self._f(self.map[name]).get_tensor(name)

    def get_slice(self, name: str):
        return self._f(self.map[name]).get_slice(name)

    def prefixed(self, prefix: str) -> "Prefixed":
        return Prefixed(self, prefix)

    def layer(self, li: int) -> "Prefixed":
        return Prefixed(self, f"model.layers.{li}.")


class Prefixed(Mapping):
    """A read-only mapping of the checkpoint names under `prefix` (prefix stripped); values load on access."""

    def __init__(self, w: Hy4Weights, prefix: str):
        self.w, self.prefix = w, prefix
        self.keys_ = [k[len(prefix) :] for k in w.map if k.startswith(prefix)]

    def __getitem__(self, k):
        return self.w.get(self.prefix + k)

    def __iter__(self):
        return iter(self.keys_)

    def __len__(self):
        return len(self.keys_)

    def __contains__(self, k):
        return (self.prefix + k) in self.w.map

    def sub(self, prefix: str) -> "Prefixed":
        return Prefixed(self.w, self.prefix + prefix)
