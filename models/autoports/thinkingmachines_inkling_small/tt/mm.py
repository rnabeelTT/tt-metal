# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-Small image / audio inputs: the two towers on the host CPU and the placement of their embeddings in a
prompt.

Inkling has no cross-modal fusion: the vision tower (HMLPPatchEncoder, 4 linear layers over 40 x 40 x 2-frame patches)
emits one decoder-width embedding per patch, the audio tower (a dMel-bin embedding sum + RMSNorm) one per 50 ms
frame, and each embedding replaces the input embedding of one placeholder token. vLLM's processor (vllm/models/
inkling/common/mm_preprocess.py) expands every item to `marker, placeholder x N`; only the placeholder positions take
tower rows (the marker stays a text token), in prompt order. Text embeddings are the normed table (embed_norm); the
tower rows carry their own final norm and replace them as they are (vLLM's _merge_multimodal_embeddings).

The towers follow vLLM's CPU path (vllm/models/inkling/common/towers.py: the same scale plan, folds, RMSNorm eps 1e-6,
exact GELU) with bf16 weights, fp32 activations and the outputs rounded to bf16 (the GPU path's fused kernels
accumulate in fp32 and round to bf16 between layers).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_TOKEN_ID = AUDIO_TOKEN_ID = None  # resolved lazily from vLLM's processor module (the ids are its constants)


def placeholder_ids():
    """(image placeholder id, audio placeholder id), as vLLM's Inkling processor writes them."""
    from vllm.transformers_utils.processors import inkling as P

    return int(P.IMAGE_TOKEN_ID), int(P.AUDIO_TOKEN_ID)


def _prime_factors(n):
    f, p = [], 2
    while p * p <= n:
        while n % p == 0:
            f.append(p)
            n //= p
        p += 1
    return f + ([n] if n > 1 else [])


def plan_out_scales(temporal_patch_size, patch_size, n_layers, n_channels=3):
    """vLLM's plan_out_scales (vllm/models/inkling/common/towers.py: the (t, h, w, c) after each HMLP layer), with its
    scipy linear_sum_assignment replaced by an exhaustive search (a handful of candidate scales; scipy is not in the
    serving image)."""
    import itertools

    def up(x):
        return int(np.ceil(x / 64)) * 64

    h, scales = 1, [(1, 1, 1, n_channels)]
    for p in _prime_factors(patch_size)[::-1]:
        h *= p
        scales.append((1, h, h, up(h * h * n_channels)))
    t = 1
    for p in _prime_factors(temporal_patch_size)[::-1]:
        t *= p
        scales.append((t, h, h, up(h * h * n_channels * t)))
    red = np.prod(np.array(scales)[:, :-1], 1)
    ideal = np.linspace(0, np.log(patch_size * patch_size * temporal_patch_size * n_channels), n_layers + 1)
    cost = np.abs(ideal[:, None] - np.log(red)[None])
    if n_layers >= len(scales):
        idx = list(np.argmin(cost, axis=1))
    else:  # min-cost assignment of the n_layers + 1 rows to distinct scales
        rows = range(n_layers + 1)
        idx = list(min(itertools.permutations(range(len(scales)), n_layers + 1),
                       key=lambda c: sum(cost[r, c[r]] for r in rows)))
    idx[0], idx[-1] = 0, len(scales) - 1
    return [scales[i] for i in idx]


def _rms(x, w, eps=1e-6):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()


class InklingTowers:
    """The vision and audio towers' weights from the checkpoint (bf16, host RAM: ~0.2 GB) and their forward passes."""

    def __init__(self, ckpt_dir, cfg=None):
        ckpt_dir = Path(ckpt_dir)
        cfg = cfg or json.loads((ckpt_dir / "config.json").read_text())
        self.vcfg, self.acfg = cfg.get("vision_config"), cfg.get("audio_config")
        index = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())["weight_map"]
        names = [k for k in index if k.startswith(("model.visual.", "model.audio."))]
        from safetensors import safe_open

        self.w = {}
        for shard in sorted({index[k] for k in names}):
            with safe_open(str(ckpt_dir / shard), framework="pt") as f:
                for k in names:
                    if index[k] == shard:
                        self.w[k] = f.get_tensor(k)
        if self.vcfg is not None:
            v = self.vcfg
            self.scales = plan_out_scales(v["temporal_patch_size"], v["patch_size"], v["n_layers"], v["n_channels"])
            for i, (s0, s1) in enumerate(zip(self.scales[:-1], self.scales[1:])):  # the plan must fit the weights
                mult = (s1[0] // s0[0]) * (s1[1] // s0[1]) * (s1[2] // s0[2])
                got = tuple(self.w[f"model.visual.layers.linear_{i}.weight"].shape)
                assert got[1] == s0[3] * mult, f"vision layer {i}: plan {s0} -> {s1} vs weight {got}"

    # ---- vision -------------------------------------------------------------------------------------------------------
    def image(self, patches):
        """patches [P, T, H, W, C] (vision_patches_bthwc, normalized) -> [P, D] bf16."""
        from vllm.models.inkling.common.towers import fold_timespace_to_depth

        v, w = self.vcfg, self.w
        x = patches.float()
        P = x.shape[0]
        n = v["n_layers"]
        for i, (s0, s1) in enumerate(zip(self.scales[:-1], self.scales[1:])):
            tf, hf = s1[0] // s0[0], s1[1] // s0[1]
            if hf > 1 or tf > 1:
                x = fold_timespace_to_depth(x, tf, hf)
            x = F.linear(x, w[f"model.visual.layers.linear_{i}.weight"].float())
            if i < n - 1:
                x = F.gelu(_rms(x, w[f"model.visual.layers.norm_{i}.weight"])).to(torch.bfloat16).float()
            else:
                x = x.to(torch.bfloat16).float()
        if v.get("use_vision_norm", True):
            x = _rms(x, w["model.visual.final_norm.weight"])
        return x.reshape(P, -1).to(torch.bfloat16)

    # ---- audio --------------------------------------------------------------------------------------------------------
    def audio(self, dmel):
        """dmel [T, n_mel_bins] integer bins -> [T, D] bf16."""
        a, w = self.acfg, self.w
        bins, vocab = a["n_mel_bins"], a["mel_vocab_size"]
        idx = torch.arange(bins).unsqueeze(0) * vocab + dmel.to(torch.int64)
        h = w["model.audio.encoder.weight"][idx.reshape(-1)].float().reshape(dmel.shape[0], bins, -1).sum(1)
        if a.get("use_audio_norm", True):
            h = _rms(h, w["model.audio.final_norm.weight"])
        return h.to(torch.bfloat16)


def _tensor(v):
    if isinstance(v, (list, tuple)):
        v = [_tensor(t) for t in v]
        return torch.cat(v, 0) if v and isinstance(v[0], torch.Tensor) else v
    return torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v


def build(items, prompt_ids, towers):
    """The prompt's multimodal rows: items = the plugin's mm_items for this request (dicts with modality, identifier,
    offset, length, data) -> {"pos": int64 [M] absolute positions, "rows": bf16 [M, D], "keys": [(first position,
    identifier)]}, or None without items. Each item's rows go to the placeholder positions inside its range (or, without
    offsets, the next placeholders of its modality in prompt order)."""
    if not items:
        return None
    img_id, aud_id = placeholder_ids()
    ids = np.asarray(prompt_ids, dtype=np.int64)
    items = sorted(items, key=lambda it: (it.get("offset") is None, it.get("offset") or 0))
    used = {img_id: 0, aud_id: 0}
    all_pos = {t: np.nonzero(ids == t)[0] for t in used}
    pos, rows, keys = [], [], []
    for it in items:
        mod, d = it["modality"], it.get("data") or {}
        if mod.startswith("image"):
            emb, tid = towers.image(_tensor(d["pixel_values"])), img_id
        elif mod.startswith("audio"):
            emb, tid = towers.audio(_tensor(d["input_audio_features"])), aud_id
        else:
            raise NotImplementedError(f"Inkling serving takes image and audio inputs, not {mod!r}")
        n = int(emb.shape[0])
        if it.get("offset") is not None:
            o, L = int(it["offset"]), int(it["length"])
            p = o + np.nonzero(ids[o : o + L] == tid)[0]
        else:
            p = all_pos[tid][used[tid] : used[tid] + n]
            used[tid] += n
        if len(p) != n:
            raise ValueError(f"{mod} item: {n} embeddings for {len(p)} placeholder tokens")
        pos.append(p)
        rows.append(emb)
        keys.append((int(p[0]) if n else -1, str(it.get("identifier"))))
    pos = np.concatenate(pos)
    order = np.argsort(pos, kind="stable")
    return {"pos": torch.as_tensor(pos[order]), "rows": torch.cat(rows, 0)[torch.as_tensor(order)], "keys": keys}


def chunk_rows(mm, a, b):
    """The multimodal rows at positions [a, b): (row indices within the chunk int64, rows bf16) or None."""
    if mm is None:
        return None
    p = mm["pos"]
    sel = (p >= a) & (p < b)
    if not bool(sel.any()):
        return None
    return p[sel] - a, mm["rows"][sel]


def merge(x, mm, a, n, Tp, mesh, H):
    """x [1, 1, Tp, w] (the chunk's embeddings; w = H or H / dies, fractured on the hidden dim) with the multimodal
    rows at positions [a, a + n) substituted: x * keep + M (keep 0 / 1 per row, M zero outside the rows: exact)."""
    import ttnn

    cr = chunk_rows(mm, a, a + n)
    if cr is None:
        return x
    r, rows = cr
    M = torch.zeros(1, 1, Tp, H, dtype=torch.bfloat16)
    M[0, 0, r] = rows
    keep = torch.ones(1, 1, Tp, 1, dtype=torch.bfloat16)
    keep[0, 0, r] = 0
    w = int(x.shape[-1])
    mapper = ttnn.ShardTensorToMesh(mesh, dim=-1) if w != H else ttnn.ReplicateTensorToMesh(mesh)
    M_tt = ttnn.from_torch(M, dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_tt = ttnn.from_torch(keep, dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y = ttnn.add(ttnn.multiply(x, k_tt), M_tt, memory_config=x.memory_config())
    for t in (x, M_tt, k_tt):
        ttnn.deallocate(t)
    return y


def default_checkpoint():
    """The HF snapshot directory of the served checkpoint (HF_MODEL: a path or a hub id in the local cache)."""
    hf = os.environ.get("HF_MODEL", "thinkingmachines/Inkling-Small")
    if os.path.isdir(hf):
        return hf
    from huggingface_hub import snapshot_download

    return snapshot_download(hf, local_files_only=True)
