# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint reader for GLM-5.3-Flash E192 (FP8 e4m3 with 128x128 block scales).

Pure torch + safetensors: the HF reference venv and the TTNN environment both import it.

Two layouts are supported:
  * pruned: the E192 checkpoint built by bringup `build_e192_ckpt.py` (routed experts renumbered 0..191, the MTP layer
    with its 288 experts under `mtp/`);
  * parent: the zai-org/GLM-5.3-Flash FP8 snapshot plus `e192_mask.json` ({layer: [parent expert ids]}); logical
    expert e of layer L reads parent expert mask[L][e].
Names are the checkpoint names (`model.language_model.layers.{L}.<suffix>`). FP8 tensors are dequantized as
W = fp8 * scale_inv[i // 128, j // 128] in fp32 and returned in the requested dtype (bf16 by default), which is the
HF convention (transformers integrations/finegrained_fp8.py).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

TEXT_PREFIX = "model.language_model."
LAYER_PREFIX = TEXT_PREFIX + "layers.{}."
MTP_LAYER = 45
BLOCK = 128
_EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(.*)$")


def default_checkpoint_dir() -> Path:
    path = os.environ.get("GLM53_CKPT")
    if not path:
        raise RuntimeError("GLM53_CKPT is not set (path of the E192 FP8 checkpoint or of the parent snapshot)")
    return Path(path)


def dequant_fp8(weight: torch.Tensor, scale_inv: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """Block dequantization: W[i, j] = float(W_fp8[i, j]) * scale_inv[i // 128, j // 128]."""
    out_f, in_f = weight.shape
    s = scale_inv.float().repeat_interleave(BLOCK, dim=0)[:out_f].repeat_interleave(BLOCK, dim=1)[:, :in_f]
    return (weight.float() * s).to(dtype)


class CheckpointReader:
    """Lazy tensor access by checkpoint name; safetensors files stay memory-mapped."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path is not None else default_checkpoint_dir()
        self.config = json.loads((self.path / "config.json").read_text())
        self.text_config = self.config["text_config"]
        index_path = self.path / "model.safetensors.index.json"
        streaming = not index_path.exists() and (self.path / "_done").is_dir()
        if not streaming:
            self._files = dict(json.loads(index_path.read_text())["weight_map"])
            mtp_index = self.path / "mtp" / "mtp.safetensors.index.json"
            if mtp_index.exists():
                self._files.update(json.loads(mtp_index.read_text())["weight_map"])
        else:  # E192 build still streaming: index the shards finished so far (build_e192_ckpt.py _done records)
            self._files = {}
            for rec_path in sorted((self.path / "_done").glob("*.json")):
                rec = json.loads(rec_path.read_text())
                self._files.update({n: rec["shard"] for n in rec["main"]})
                self._files.update({n: "mtp/" + rec["shard"] for n in rec["mtp"]})
        mask_path = self.path / "e192_mask.json"
        n_ckpt = 192 if streaming else self.text_config["n_routed_experts"]
        self.mask = None
        if n_ckpt != 192:
            if not mask_path.exists():
                raise RuntimeError(f"{self.path}: {n_ckpt} routed experts and no e192_mask.json")
            self.mask = {int(k): v for k, v in json.loads(mask_path.read_text())["mask"].items()}
        self.num_experts = 192
        # Router rows of the E192 layers: a checkpoint whose router still has the parent's 288 rows (the parent
        # snapshot, or a pruned build made before routers were pruned) is indexed with the mask on read.
        self.router_mask = None
        if mask_path.exists():
            self.router_mask = {int(k): v for k, v in json.loads(mask_path.read_text())["mask"].items()}
        self._handles: dict[str, object] = {}

    # ----------------------------------------------------------------------------------------------------- raw access
    def _handle(self, fname: str):
        h = self._handles.get(fname)
        if h is None:
            h = safe_open(str(self.path / fname), "pt")
            self._handles[fname] = h
        return h

    def has(self, name: str) -> bool:
        return name in self._files

    def raw(self, name: str) -> torch.Tensor:
        return self._handle(self._files[name]).get_tensor(name)

    def get(self, name: str, dtype=torch.bfloat16) -> torch.Tensor:
        """Tensor by checkpoint name; FP8 weights come back dequantized in `dtype`."""
        t = self._router_rows(name, self.raw(name))
        if t.dtype == torch.float8_e4m3fn:
            return dequant_fp8(t, self.raw(name + "_scale_inv"), dtype)
        return t.to(dtype) if dtype is not None else t

    def _router_rows(self, name: str, t: torch.Tensor) -> torch.Tensor:
        if self.router_mask is None or not (
            name.endswith("mlp.gate.weight") or name.endswith("mlp.gate.e_score_correction_bias")
        ):
            return t
        layer = int(name.split(".layers.")[1].split(".")[0])
        if layer == MTP_LAYER or t.shape[0] == self.num_experts:
            return t
        return t[torch.tensor(self.router_mask[layer])]

    # ------------------------------------------------------------------------------------------------- name helpers
    def _expert_name(self, layer: int, expert: int, suffix: str) -> str:
        e = expert
        if self.mask is not None and layer != MTP_LAYER:
            e = self.mask[layer][expert]
        return LAYER_PREFIX.format(layer) + f"mlp.experts.{e}.{suffix}"

    def layer_names(self, layer: int) -> list[str]:
        """Checkpoint suffixes of one decoder layer, routed experts excluded, scale tensors excluded."""
        prefix = LAYER_PREFIX.format(layer)
        out = []
        for name in self._files:
            if not name.startswith(prefix) or name.endswith("_scale_inv"):
                continue
            suffix = name[len(prefix) :]
            if _EXPERT_RE.match(suffix):
                continue
            out.append(suffix)
        return sorted(out)

    def layer_tensor(self, layer: int, suffix: str, dtype=torch.bfloat16) -> torch.Tensor:
        return self.get(LAYER_PREFIX.format(layer) + suffix, dtype)

    def expert(self, layer: int, expert: int, proj: str, dtype=torch.bfloat16) -> torch.Tensor:
        """Routed expert weight [out, in] for proj in gate_proj / up_proj / down_proj."""
        return self.get(self._expert_name(layer, expert, f"{proj}.weight"), dtype)

    def num_layer_experts(self, layer: int) -> int:
        if layer == MTP_LAYER:
            return int(self.config["text_config"].get("mtp_n_routed_experts", 288))
        return self.num_experts

    def _layer_item(self, layer: int, suffix: str, dtype) -> torch.Tensor:
        t = self.raw(LAYER_PREFIX.format(layer) + suffix)
        if t.dtype == torch.float32:
            return self._router_rows(LAYER_PREFIX.format(layer) + suffix, t)
        return self.layer_tensor(layer, suffix, dtype)

    def layer_state(self, layer: int, dtype=torch.bfloat16, experts: bool = True, lazy: bool = False):
        """All tensors of one layer keyed by checkpoint suffix (FP8 dequantized; fp32 params keep fp32).
        lazy: a LazyMap that reads / dequantizes a tensor only when it is looked up (the weight cache skips the reads
        of every weight it already holds)."""
        sd = {}
        for suffix in self.layer_names(layer):
            sd[suffix] = lambda s=suffix: self._layer_item(layer, s, dtype)
        if experts and self.has(self._expert_name(layer, 0, "gate_proj.weight")):
            for e in range(self.num_layer_experts(layer)):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    sd[f"mlp.experts.{e}.{proj}.weight"] = lambda e=e, p=proj: self.expert(layer, e, p, dtype)
        lz = LazyMap(sd)
        return lz if lazy else {k: lz[k] for k in lz}

    def top(self, name: str, dtype=torch.bfloat16) -> torch.Tensor:
        """embed_tokens.weight / norm.weight (text model) or lm_head.weight."""
        full = name if name.startswith("lm_head") else TEXT_PREFIX + name
        return self.get(full, dtype)


class LazyMap(Mapping):
    """Read-only mapping name -> tensor whose values are computed by a thunk on every lookup (nothing is kept)."""

    def __init__(self, thunks: dict):
        self._thunks = thunks

    def __getitem__(self, key):
        return self._thunks[key]()

    def __iter__(self):
        return iter(self._thunks)

    def __len__(self):
        return len(self._thunks)

    def __contains__(self, key):
        return key in self._thunks


@lru_cache(maxsize=None)
def reader(path: str | None = None) -> CheckpointReader:
    return CheckpointReader(path)
