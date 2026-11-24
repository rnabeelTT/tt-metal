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
import mmap
import os
import struct
from collections.abc import Mapping

import torch
from safetensors import safe_open

from models.autoports.tencent_hy4_flash.tt.config import checkpoint_dir

# HY4_FAST_LOAD=1 (default): checkpoint tensors >= HY4_FAST_LOAD_MIN_MB (16) read with O_DIRECT (Hy4Weights._get_direct)
_FAST = os.environ.get("HY4_FAST_LOAD", "1") == "1"
_FAST_MIN = int(float(os.environ.get("HY4_FAST_LOAD_MIN_MB", "16")) * 2**20)
_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I64": torch.int64, "I32": torch.int32,
       "U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn}


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
        if _FAST:
            t = self._get_direct(name)
            if t is not None:
                return t
        return self._f(self.map[name]).get_tensor(name)

    def _header(self, shard):
        """(header dict, data start) of a shard (cached)."""
        hd = self._hdr.get(shard) if hasattr(self, "_hdr") else None
        if hd is None:
            self._hdr = getattr(self, "_hdr", {})
            with open(os.path.join(self.dir, shard), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hd = self._hdr[shard] = (json.loads(fh.read(n)), 8 + n)
        return hd

    def _get_direct(self, name):
        """A large tensor read with O_DIRECT (tt/fastload.py: this host's page-cache reads run at ~0.2-1 GB/s; the
        safetensors memory map reads through it). The same bytes as safe_open's get_tensor; None: use safe_open."""
        shard = self.map[name]
        hdr, data0 = self._header(shard)
        meta = hdr.get(name)
        dt = _DT.get(meta["dtype"]) if meta else None
        if dt is None:
            return None
        a, b = meta["data_offsets"]
        if b - a < _FAST_MIN:
            return None
        try:
            from models.autoports.tencent_hy4_flash.tt import fastload

            start, end = data0 + a, data0 + b
            off0 = (start // fastload.ALIGN) * fastload.ALIGN
            mm = mmap.mmap(-1, -(-(end - off0) // fastload.ALIGN) * fastload.ALIGN)
            fastload.read_direct(os.path.join(self.dir, shard), mm, off0, end - off0)
            t = torch.frombuffer(mm, dtype=torch.uint8, count=b - a, offset=start - off0)
            return t.view(dt).reshape(meta["shape"])
        except Exception:  # noqa: BLE001
            return None

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
