# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Streaming weight access for Qwen3.5-122B-A10B.

The checkpoint is 250 GB of bf16 safetensors (39 shards) on a host with 249 GB of RAM, so the
whole state dict can never be materialized. ``StreamingStateDict`` is a read-only mapping from the
qwen36 INTERNAL key scheme to tensors that are read from the safetensors shards on first access
(``safe_open`` is mmap-backed, so a read costs page cache, not resident memory).

Internal key scheme (matches ``models/demos/blackhole/qwen36/tt/weight_mapping.py``):

    tok_embeddings.weight                       <- model.language_model.embed_tokens.weight
    output.weight                               <- lm_head.weight
    norm.weight                                 <- model.language_model.norm.weight
    layers.N.linear_attn.qkv_proj.weight        <- ...layers.N.linear_attn.in_proj_qkv.weight
    layers.N.linear_attn.{q,k,v}_conv.weight    <- slices of ...layers.N.linear_attn.conv1d.weight
    layers.N.<everything else>                  <- model.language_model.layers.N.<everything else>

``mtp.*`` (multi-token prediction head) is not part of the layer mapping; :meth:`StreamingStateDict.mtp_state`
materializes it on demand in the same internal scheme (``mtp.layers.0.*`` with the per-expert tensors fused into the
main layers' ``mlp.experts.gate_up_proj`` / ``down_proj`` layout, plus ``mtp.fc``, ``mtp.norm``, ``mtp.pre_fc_norm_*``).
``model.visual.*`` (vision tower) is NOT part of
the text mapping but stays reachable through :meth:`StreamingStateDict.visual_keys` /
:meth:`StreamingStateDict.visual_tensor` for the later vision enablement.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Iterator

import torch
from safetensors import safe_open

HF_MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
_LM_PREFIX = "model.language_model."
_VISUAL_PREFIX = "model.visual."


def resolve_checkpoint_dir(hf_model: str | None = None) -> Path:
    """Local snapshot directory for ``hf_model`` (hub id or directory). Offline by default."""
    hf_model = hf_model or os.environ.get("HF_MODEL") or HF_MODEL_ID
    if os.path.isfile(os.path.join(hf_model, "config.json")):
        return Path(hf_model)
    from huggingface_hub import snapshot_download

    offline = os.getenv("HF_HUB_OFFLINE", "1") == "1"
    return Path(snapshot_download(hf_model, local_files_only=offline))


class _Entry:
    __slots__ = ("shard", "hf_key", "transform")

    def __init__(self, shard: str, hf_key: str, transform: Callable[[torch.Tensor], torch.Tensor] | None = None):
        self.shard = shard
        self.hf_key = hf_key
        self.transform = transform


class StreamingStateDict(Mapping):
    """Lazy read-only mapping of internal weight keys to tensors read from safetensors shards.

    ``items()``/``values()`` are deliberately NOT lazy-safe for whole-dict iteration: use
    :meth:`layer` to materialize one layer, or index single keys.
    """

    def __init__(self, checkpoint_dir: str | Path, *, linear_q_dim: int, linear_k_dim: int):
        self.checkpoint_dir = Path(checkpoint_dir)
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        self._entries: dict[str, _Entry] = {}
        self._visual: dict[str, str] = {}
        self._layer_keys: dict[int, list[str]] = {}
        q, k = linear_q_dim, linear_k_dim
        self._mtp: dict[str, str] = {}
        for hf_key, shard in weight_map.items():
            if hf_key.startswith("mtp."):
                self._mtp[hf_key] = shard
                continue
            if hf_key.startswith(_VISUAL_PREFIX):
                self._visual[hf_key[len(_VISUAL_PREFIX) :]] = shard
                continue
            if hf_key == "lm_head.weight":
                self._entries["output.weight"] = _Entry(shard, hf_key)
                continue
            if not hf_key.startswith(_LM_PREFIX):
                raise KeyError(f"unexpected checkpoint key {hf_key!r}")
            key = hf_key[len(_LM_PREFIX) :]
            if key == "embed_tokens.weight":
                self._entries["tok_embeddings.weight"] = _Entry(shard, hf_key)
            elif key == "norm.weight":
                self._entries["norm.weight"] = _Entry(shard, hf_key)
            elif key.startswith("layers."):
                layer_idx = int(key.split(".")[1])
                sub = key.split(".", 2)[2]
                prefix = f"layers.{layer_idx}."
                if sub == "linear_attn.in_proj_qkv.weight":
                    self._add(prefix + "linear_attn.qkv_proj.weight", _Entry(shard, hf_key), layer_idx)
                elif sub == "linear_attn.conv1d.weight":
                    self._add(prefix + "linear_attn.q_conv.weight", _Entry(shard, hf_key, lambda t: t[:q]), layer_idx)
                    self._add(
                        prefix + "linear_attn.k_conv.weight", _Entry(shard, hf_key, lambda t: t[q : q + k]), layer_idx
                    )
                    self._add(
                        prefix + "linear_attn.v_conv.weight", _Entry(shard, hf_key, lambda t: t[q + k :]), layer_idx
                    )
                else:
                    self._add(key, _Entry(shard, hf_key), layer_idx)
            else:
                raise KeyError(f"unexpected language-model key {hf_key!r}")
        self._open: dict[str, object] = {}

    def _add(self, key: str, entry: _Entry, layer_idx: int) -> None:
        self._entries[key] = entry
        self._layer_keys.setdefault(layer_idx, []).append(key)

    # -- Mapping protocol -------------------------------------------------------------------
    def _shard(self, name: str):
        handle = self._open.get(name)
        if handle is None:
            handle = safe_open(str(self.checkpoint_dir / name), framework="pt", device="cpu")
            self._open[name] = handle
        return handle

    def __getitem__(self, key: str) -> torch.Tensor:
        entry = self._entries[key]
        tensor = self._shard(entry.shard).get_tensor(entry.hf_key)
        return entry.transform(tensor) if entry.transform is not None else tensor

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    # -- Convenience ------------------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        return len(self._layer_keys)

    def layer_keys(self, layer_idx: int) -> list[str]:
        return list(self._layer_keys[layer_idx])

    def layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Materialize one decoder layer's tensors (internal keys, ``layers.N.`` prefix kept)."""
        return {key: self[key] for key in self._layer_keys[layer_idx]}

    def shard_names(self) -> set[str]:
        return {e.shard for e in self._entries.values()}

    # -- MTP head -----------------------------------------------------------------------------
    def has_mtp(self) -> bool:
        return bool(self._mtp)

    def mtp_keys(self) -> list[str]:
        """Checkpoint names of the multi-token-prediction head (``mtp.*``)."""
        return sorted(self._mtp)

    def mtp_state(self, fuse_experts: bool = True) -> dict[str, torch.Tensor]:
        """The MTP head in the internal scheme: ``mtp.fc.weight`` [H, 2H], ``mtp.pre_fc_norm_embedding.weight``,
        ``mtp.pre_fc_norm_hidden.weight``, ``mtp.norm.weight`` and ONE full-attention MoE decoder layer under
        ``mtp.layers.0.`` whose routed experts are fused like the main layers' checkpoint tensors:
        ``mlp.experts.gate_up_proj`` [E, 2I, H] (gate rows first, then up) and ``mlp.experts.down_proj`` [E, H, I]
        (the checkpoint stores the MTP experts per expert: ``experts.N.{gate,up,down}_proj``). Reads ~5 GB bf16 once."""
        if not self._mtp:
            raise KeyError("checkpoint has no mtp.* tensors")
        out: dict[str, torch.Tensor] = {}
        per_expert: dict[tuple[int, int, str], str] = {}  # (layer, expert, proj) -> hf key
        for hf_key in self._mtp:
            parts = hf_key.split(".")
            if (
                len(parts) >= 7
                and parts[1] == "layers"
                and parts[3] == "mlp"
                and parts[4] == "experts"
                and parts[5].isdigit()
            ):
                per_expert[(int(parts[2]), int(parts[5]), parts[6])] = hf_key
                if fuse_experts:
                    continue
            out[hf_key] = self._shard(self._mtp[hf_key]).get_tensor(hf_key)
        if fuse_experts and per_expert:
            layers = sorted({li for li, _, _ in per_expert})
            for li in layers:
                experts = sorted({e for l2, e, _ in per_expert if l2 == li})
                gate_up = []
                down = []
                for e in experts:
                    g = self._shard(self._mtp[per_expert[(li, e, "gate_proj")]]).get_tensor(
                        per_expert[(li, e, "gate_proj")]
                    )
                    u = self._shard(self._mtp[per_expert[(li, e, "up_proj")]]).get_tensor(
                        per_expert[(li, e, "up_proj")]
                    )
                    d = self._shard(self._mtp[per_expert[(li, e, "down_proj")]]).get_tensor(
                        per_expert[(li, e, "down_proj")]
                    )
                    gate_up.append(torch.cat([g, u], dim=0))  # [2I, H]
                    down.append(d)  # [H, I]
                out[f"mtp.layers.{li}.mlp.experts.gate_up_proj"] = torch.stack(gate_up, dim=0)
                out[f"mtp.layers.{li}.mlp.experts.down_proj"] = torch.stack(down, dim=0)
        return out

    def mtp_shapes(self) -> dict[str, list[int]]:
        """Shapes of the mtp.* tensors without reading them (safetensors slices)."""
        return {k: list(self._shard(s).get_slice(k).get_shape()) for k, s in sorted(self._mtp.items())}

    def visual_keys(self) -> list[str]:
        """Vision-tower tensor names (without the ``model.visual.`` prefix). Not loaded by the text port."""
        return sorted(self._visual)

    def visual_tensor(self, key: str) -> torch.Tensor:
        return self._shard(self._visual[key]).get_tensor(_VISUAL_PREFIX + key)

    def close(self) -> None:
        self._open.clear()


def remap_hf_layer_state(state: Mapping[str, torch.Tensor], layer_idx: int, *, linear_q_dim: int, linear_k_dim: int):
    """HF-format tensors for ONE layer -> internal keys with the ``layers.{layer_idx}.`` prefix.

    Accepts layer-local keys (``self_attn.q_proj.weight``) or full-model keys with any of the
    ``model.language_model.layers.N.`` / ``model.layers.N.`` / ``layers.N.`` prefixes.
    """
    prefixes = (f"model.language_model.layers.{layer_idx}.", f"model.layers.{layer_idx}.", f"layers.{layer_idx}.")
    out: dict[str, torch.Tensor] = {}
    q, k = linear_q_dim, linear_k_dim
    for key, tensor in state.items():
        local = key
        for p in prefixes:
            if key.startswith(p):
                local = key[len(p) :]
                break
        else:
            if key.startswith(("model.", "layers.", "lm_head", "mtp.")):
                continue  # a different layer or a non-layer tensor
        dst = f"layers.{layer_idx}."
        if local == "linear_attn.in_proj_qkv.weight":
            out[dst + "linear_attn.qkv_proj.weight"] = tensor
        elif local == "linear_attn.conv1d.weight":
            out[dst + "linear_attn.q_conv.weight"] = tensor[:q]
            out[dst + "linear_attn.k_conv.weight"] = tensor[q : q + k]
            out[dst + "linear_attn.v_conv.weight"] = tensor[q + k :]
        else:
            out[dst + local] = tensor
    return out


def hf_layer_state(checkpoint_dir: str | Path, layer_idx: int) -> dict[str, torch.Tensor]:
    """Real HF-format tensors (layer-local keys) for one decoder layer, read from the shards."""
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "model.safetensors.index.json", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"{_LM_PREFIX}layers.{layer_idx}."
    wanted = {key: shard for key, shard in weight_map.items() if key.startswith(prefix)}
    if not wanted:
        raise KeyError(f"no tensors for layer {layer_idx} in {checkpoint_dir}")
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(str(checkpoint_dir / shard), framework="pt", device="cpu") as f:
            for key, s in wanted.items():
                if s == shard:
                    out[key[len(prefix) :]] = f.get_tensor(key)
    return out
