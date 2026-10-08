# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Streaming weight access for zai-org/GLM-4.5-Air (``glm4_moe``).

The checkpoint is ~220 GB of bf16 safetensors (47 shards) on a host with 249 GB of RAM, so the whole state dict is
never materialized. ``StreamingStateDict`` is a read-only mapping from the INTERNAL key scheme of the Qwen3.5 autoport
code (qwen36 layer / MoE block) to tensors read from the shards on first access (``safe_open`` is mmap-backed).

Internal key scheme:

    tok_embeddings.weight                         <- model.embed_tokens.weight
    output.weight                                 <- lm_head.weight
    norm.weight                                   <- model.norm.weight
    layers.N.self_attn.{q,k,v}_proj.{weight,bias} <- model.layers.N.self_attn.*  (o_proj.weight too)
    layers.N.{input,post_attention}_layernorm.weight
    layers.0.mlp.{gate,up,down}_proj.weight       <- the dense first layer (first_k_dense_replace = 1)
    layers.N.mlp.gate.weight                      <- router [E, H]
    layers.N.mlp.gate.e_score_correction_bias     <- router selection bias [E] (fp32)
    layers.N.mlp.experts.gate_up_proj             <- FUSED from experts.{0..E-1}.{gate,up}_proj: [E, 2I, H] (gate rows
                                                     first, then up: the HF ``Glm4MoeNaiveMoe`` layout)
    layers.N.mlp.experts.down_proj                <- FUSED from experts.{0..E-1}.down_proj: [E, H, I]
    layers.N.mlp.shared_expert.{gate,up,down}_proj.weight <- model.layers.N.mlp.shared_experts.*

Layer ``num_hidden_layers`` (46) is the multi-token-prediction layer; it is not a main layer. :meth:`mtp_state` maps
it to the MTP head scheme of the Qwen3.5 autoport (``mtp.fc`` <- ``eh_proj``, ``mtp.pre_fc_norm_embedding`` <-
``enorm``, ``mtp.pre_fc_norm_hidden`` <- ``hnorm``, ``mtp.norm`` <- ``shared_head.norm``, ``mtp.layers.0.*`` <- the
layer's attention / MoE / norms; vLLM's ``glm4_moe_mtp`` concatenates ``[enorm(embed), hnorm(hidden)]`` exactly like
Qwen3.5's ``fc``).
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

HF_MODEL_ID = "zai-org/GLM-4.5-Air"
_PREFIX = "model."


def resolve_checkpoint_dir(hf_model: str | None = None) -> Path:
    """Local snapshot directory for ``hf_model`` (hub id or directory). Offline by default."""
    hf_model = hf_model or os.environ.get("HF_MODEL") or HF_MODEL_ID
    if os.path.isfile(os.path.join(hf_model, "config.json")):
        return Path(hf_model)
    from huggingface_hub import snapshot_download

    offline = os.getenv("HF_HUB_OFFLINE", "1") == "1"
    return Path(snapshot_download(hf_model, local_files_only=offline))


def _internal_layer_key(sub: str) -> str:
    """GLM layer-local key -> internal layer-local key (only the shared expert is renamed)."""
    if sub.startswith("mlp.shared_experts."):
        return "mlp.shared_expert." + sub[len("mlp.shared_experts.") :]
    return sub


class StreamingStateDict(Mapping):
    """Lazy read-only mapping of internal weight keys to tensors read from safetensors shards.

    The fused routed-expert entries (``layers.N.mlp.experts.gate_up_proj`` / ``down_proj``) are assembled from the
    per-expert checkpoint tensors on access (~1.7 GB bf16 per layer). Use :meth:`layer` to materialize one layer.
    """

    def __init__(
        self, checkpoint_dir: str | Path, *, num_layers: int | None = None, dense_pad_to: int | None = None, **_unused
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.dense_pad_to = dense_pad_to
        with open(self.checkpoint_dir / "model.safetensors.index.json", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        if num_layers is None:
            with open(self.checkpoint_dir / "config.json", encoding="utf-8") as f:
                num_layers = int(json.load(f)["num_hidden_layers"])
        self.n_main_layers = num_layers
        self._weight_map = weight_map
        self._plain: dict[str, tuple[str, str]] = {}  # internal key -> (shard, hf key)
        self._experts: dict[int, dict[tuple[int, str], str]] = {}  # layer -> (expert, proj) -> hf key
        self._layer_keys: dict[int, list[str]] = {}
        self._mtp: dict[str, str] = {}  # layer-local key of the MTP layer -> hf key
        for hf_key in weight_map:
            if hf_key == "lm_head.weight":
                self._plain["output.weight"] = (weight_map[hf_key], hf_key)
                continue
            if not hf_key.startswith(_PREFIX):
                raise KeyError(f"unexpected checkpoint key {hf_key!r}")
            key = hf_key[len(_PREFIX) :]
            if key == "embed_tokens.weight":
                self._plain["tok_embeddings.weight"] = (weight_map[hf_key], hf_key)
            elif key == "norm.weight":
                self._plain["norm.weight"] = (weight_map[hf_key], hf_key)
            elif key.startswith("layers."):
                layer_idx = int(key.split(".")[1])
                sub = key.split(".", 2)[2]
                if layer_idx >= num_layers:
                    self._mtp[sub] = hf_key
                    continue
                parts = sub.split(".")
                if parts[:2] == ["mlp", "experts"] and parts[2].isdigit():
                    self._experts.setdefault(layer_idx, {})[(int(parts[2]), parts[3])] = hf_key
                    continue
                self._add(f"layers.{layer_idx}." + _internal_layer_key(sub), weight_map[hf_key], hf_key, layer_idx)
            else:
                raise KeyError(f"unexpected model key {hf_key!r}")
        for layer_idx in self._experts:
            for name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                self._layer_keys.setdefault(layer_idx, []).append(f"layers.{layer_idx}.{name}")
        self._open: dict[str, object] = {}

    def _add(self, key: str, shard: str, hf_key: str, layer_idx: int) -> None:
        self._plain[key] = (shard, hf_key)
        self._layer_keys.setdefault(layer_idx, []).append(key)

    # -- raw access -------------------------------------------------------------------------
    def _shard(self, name: str):
        handle = self._open.get(name)
        if handle is None:
            handle = safe_open(str(self.checkpoint_dir / name), framework="pt", device="cpu")
            self._open[name] = handle
        return handle

    def _hf(self, hf_key: str) -> torch.Tensor:
        return self._shard(self._weight_map[hf_key]).get_tensor(hf_key)

    @staticmethod
    def _fuse(per: dict[tuple[int, str], str], get) -> tuple[torch.Tensor, torch.Tensor]:
        n = 1 + max(e for e, _ in per)
        gate_up = torch.stack([torch.cat([get(per[(e, "gate_proj")]), get(per[(e, "up_proj")])], 0) for e in range(n)])
        down = torch.stack([get(per[(e, "down_proj")]) for e in range(n)])
        return gate_up, down  # [E, 2I, H], [E, H, I]

    def _fused_experts(self, layer_idx: int, which: str) -> torch.Tensor:
        """One half of the fused routed experts (reads only that half's per-expert tensors)."""
        per = self._experts[layer_idx]
        n = 1 + max(e for e, _ in per)
        if which == "gate_up_proj":
            return torch.stack(
                [torch.cat([self._hf(per[(e, "gate_proj")]), self._hf(per[(e, "up_proj")])], 0) for e in range(n)]
            )
        return torch.stack([self._hf(per[(e, "down_proj")]) for e in range(n)])

    # -- Mapping protocol -------------------------------------------------------------------
    def __getitem__(self, key: str) -> torch.Tensor:
        ent = self._plain.get(key)
        if ent is not None:
            t = self._shard(ent[0]).get_tensor(ent[1])
            return pad_dense_mlp(key, t, self.dense_pad_to) if self.dense_pad_to else t
        if key.startswith("layers.") and ".mlp.experts." in key:
            layer_idx = int(key.split(".")[1])
            which = key.rsplit(".", 1)[1]
            if layer_idx in self._experts and which in ("gate_up_proj", "down_proj"):
                return self._fused_experts(layer_idx, which)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for keys in self._layer_keys.values():
            yield from keys
        for key in ("tok_embeddings.weight", "norm.weight", "output.weight"):
            if key in self._plain:
                yield key

    def __len__(self) -> int:
        return sum(len(v) for v in self._layer_keys.values()) + 3

    def __contains__(self, key: object) -> bool:
        if key in self._plain:
            return True
        if isinstance(key, str) and key.startswith("layers.") and ".mlp.experts." in key:
            return any(key in v for v in self._layer_keys.values())
        return False

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
        return set(self._weight_map.values())

    # -- MTP head -----------------------------------------------------------------------------
    def has_mtp(self) -> bool:
        return bool(self._mtp)

    def mtp_keys(self) -> list[str]:
        return sorted(self._mtp.values())

    def mtp_state(self, fuse_experts: bool = True) -> dict[str, torch.Tensor]:
        """The MTP layer in the Qwen3.5 autoport's MTP scheme (see the module docstring). ``mtp.head.weight`` is the
        layer's own ``shared_head.head`` (GLM ships one; the main ``lm_head`` is used when they are identical)."""
        if not self._mtp:
            raise KeyError("checkpoint has no MTP layer")
        rename = {
            "eh_proj.weight": "mtp.fc.weight",
            "enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "shared_head.norm.weight": "mtp.norm.weight",
            "shared_head.head.weight": "mtp.head.weight",
            "embed_tokens.weight": "mtp.embed_tokens.weight",
        }
        out: dict[str, torch.Tensor] = {}
        per: dict[tuple[int, str], str] = {}
        for sub, hf_key in self._mtp.items():
            parts = sub.split(".")
            if parts[:2] == ["mlp", "experts"] and parts[2].isdigit() and fuse_experts:
                per[(int(parts[2]), parts[3])] = hf_key
                continue
            dst = rename.get(sub, "mtp.layers.0." + _internal_layer_key(sub))
            out[dst] = self._hf(hf_key)
        if per:
            gate_up, down = self._fuse(per, self._hf)
            out["mtp.layers.0.mlp.experts.gate_up_proj"] = gate_up
            out["mtp.layers.0.mlp.experts.down_proj"] = down
        return out

    def mtp_shapes(self) -> dict[str, list[int]]:
        return {
            sub: list(self._shard(self._weight_map[k]).get_slice(k).get_shape()) for sub, k in sorted(self._mtp.items())
        }

    def visual_keys(self) -> list[str]:
        return []  # text-only checkpoint

    def close(self) -> None:
        self._open.clear()


def pad_dense_mlp(key: str, t: torch.Tensor, width: int) -> torch.Tensor:
    """Zero-pad a dense-MLP weight (``mlp.{gate,up}_proj`` rows / ``mlp.down_proj`` columns) to ``width`` (exact)."""
    if key.endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight")) and t.shape[0] < width:
        return torch.nn.functional.pad(t, (0, 0, 0, width - t.shape[0]))
    if key.endswith(".mlp.down_proj.weight") and t.shape[1] < width:
        return torch.nn.functional.pad(t, (0, width - t.shape[1]))
    return t


def remap_hf_layer_state(
    state: Mapping[str, torch.Tensor], layer_idx: int, *, dense_pad_to: int | None = None, **_unused
):
    """HF-format tensors for ONE layer -> internal keys with the ``layers.{layer_idx}.`` prefix.

    Accepts layer-local keys (``self_attn.q_proj.weight``) or full-model keys (``model.layers.N.`` / ``layers.N.``).
    Per-expert tensors (``mlp.experts.E.{gate,up,down}_proj.weight``) are fused into the internal
    ``mlp.experts.gate_up_proj`` / ``down_proj``; already-fused tensors pass through.
    """
    prefixes = (f"model.layers.{layer_idx}.", f"layers.{layer_idx}.")
    out: dict[str, torch.Tensor] = {}
    per: dict[tuple[int, str], torch.Tensor] = {}
    dst = f"layers.{layer_idx}."
    for key, tensor in state.items():
        local = key
        for p in prefixes:
            if key.startswith(p):
                local = key[len(p) :]
                break
        else:
            if key.startswith(("model.", "layers.", "lm_head")):
                continue  # a different layer or a non-layer tensor
        parts = local.split(".")
        if parts[:2] == ["mlp", "experts"] and parts[2].isdigit():
            per[(int(parts[2]), parts[3])] = tensor
            continue
        key_out = dst + _internal_layer_key(local)
        out[key_out] = pad_dense_mlp(key_out, tensor, dense_pad_to) if dense_pad_to else tensor
    if per:
        gate_up, down = StreamingStateDict._fuse({k: k for k in per}, lambda k: per[k])
        out[dst + "mlp.experts.gate_up_proj"] = gate_up
        out[dst + "mlp.experts.down_proj"] = down
    return out


def hf_layer_state(checkpoint_dir: str | Path, layer_idx: int) -> dict[str, torch.Tensor]:
    """Real HF-format tensors (layer-local keys, per-expert as in the checkpoint) for one decoder layer."""
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "model.safetensors.index.json", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"{_PREFIX}layers.{layer_idx}."
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
