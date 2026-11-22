# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access and HF -> device layout conversion for the text tower.

Weights are read tensor by tensor from the safetensors shards (no full state dict in host memory). Linear weights
are transposed to ``[in, out]``; Q and K channels (and the per-head ``q_norm`` / ``k_norm`` scales) are permuted to
the Meta pairing that ``rotary_embedding_llama`` rotates with. Device copies go through ``ttnn.as_tensor`` with a
cache file per tensor and dtype, so later boots skip the conversion.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import ttnn
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

TEXT_PREFIX = "model.language_model."
LAYOUT_VERSION = 1  # bump when a tensor's device layout (permutation, fusion, padding) changes
VISUAL_PREFIX = "model.visual."


class Checkpoint:
    """Lazy tensor access over the sharded safetensors checkpoint."""

    def __init__(self, path: str | os.PathLike):
        from safetensors import safe_open

        self.path = Path(path)
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._open = {}
        self._safe_open = safe_open

    def _file(self, name):
        f = self.weight_map[name]
        if f not in self._open:
            self._open[f] = self._safe_open(str(self.path / f), framework="pt")
        return self._open[f]

    def get(self, name: str) -> torch.Tensor:
        return self._file(name).get_tensor(name)

    def keys(self, prefix: str = ""):
        return [k for k in self.weight_map if k.startswith(prefix)]

    def visual_state_dict(self) -> dict[str, torch.Tensor]:
        return {k[len(VISUAL_PREFIX) :]: self.get(k) for k in self.keys(VISUAL_PREFIX)}


def hf_to_meta_channels(head_dim: int) -> torch.Tensor:
    """``meta[2i] = hf[i]``, ``meta[2i+1] = hf[i + head_dim/2]`` (from the Qwen3-Coder-30B Blackhole port)."""
    half = head_dim // 2
    return torch.stack([torch.arange(half), torch.arange(half) + half], dim=1).reshape(-1)


def permute_rows_to_meta(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Permute the output rows (HF ``[out, in]`` layout) of a Q or K projection, head by head."""
    perm = hf_to_meta_channels(head_dim)
    return w.reshape(n_heads, head_dim, -1)[:, perm].reshape(n_heads * head_dim, -1)


def layer_tensors(ckpt: Checkpoint, layer: int, cfg) -> dict[str, torch.Tensor]:
    """Host tensors for one decoder layer in device layout (``[in, out]`` linears, Meta Q/K)."""
    p = f"{TEXT_PREFIX}layers.{layer}."
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    q = permute_rows_to_meta(ckpt.get(p + "self_attn.q_proj.weight"), nh, hd)
    k = permute_rows_to_meta(ckpt.get(p + "self_attn.k_proj.weight"), nkv, hd)
    v = ckpt.get(p + "self_attn.v_proj.weight")
    perm = hf_to_meta_channels(hd)
    return {
        "wqkv": torch.cat([q, k, v], dim=0).T.contiguous(),
        "wo": ckpt.get(p + "self_attn.o_proj.weight").T.contiguous(),
        "q_norm": ckpt.get(p + "self_attn.q_norm.weight")[perm],
        "k_norm": ckpt.get(p + "self_attn.k_norm.weight")[perm],
        # [gate | up] packed and tile-pair interleaved for minimal_matmul's fused SwiGLU (silu(gate) * up)
        "w13": prepare_for_fused_swiglu(
            torch.cat([ckpt.get(p + "mlp.gate_proj.weight").T, ckpt.get(p + "mlp.up_proj.weight").T], dim=-1),
            1,
            gate_is_first=True,
        ).contiguous(),
        "w2": ckpt.get(p + "mlp.down_proj.weight").T.contiguous(),
        "attn_norm": ckpt.get(p + "input_layernorm.weight"),
        "mlp_norm": ckpt.get(p + "post_attention_layernorm.weight"),
    }


def cache_dir() -> Path:
    root = (
        os.environ.get("TT_DIT_CACHE_DIR")
        or os.environ.get("TT_CACHE_PATH")
        or Path.home() / ".cache/ttnn/qwen3_vl_reranker"
    )
    # namespaced by checkpoint revision and device layout version: a shared cache dir, a new revision or a layout
    # change never picks up stale tensors
    from models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs import HF_REVISION

    rev = os.environ.get("HF_REVISION", HF_REVISION)[:12]
    d = Path(root) / f"qwen3-vl-reranker-8b-{rev}-L{LAYOUT_VERSION}" / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def as_device(
    t: torch.Tensor | None,
    device,
    dtype,
    name: str,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    cache=True,
    loader=None,
):
    """``ttnn.as_tensor`` with a per-(name, dtype, layout) cache file. ``loader`` builds the host tensor lazily, so a
    warm cache never touches the checkpoint."""
    fname = None
    if cache:
        fname = str(
            cache_dir() / f"{name}.{str(dtype).split('.')[-1]}.{'tile' if layout == ttnn.TILE_LAYOUT else 'rm'}"
        )
        if t is None and loader is not None and not _cached(fname):
            t = loader()
        elif t is None and loader is not None:
            t = torch.zeros(1)  # placeholder: as_tensor reads the cache file and ignores the host tensor's data
    elif t is None:
        t = loader()
    return ttnn.as_tensor(
        t,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
        cache_file_name=fname,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if hasattr(device, "get_num_devices") else None,
    )


def _cached(fname: str) -> bool:
    return any(Path(fname).parent.glob(Path(fname).name + "*"))
