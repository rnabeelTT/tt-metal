# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The chip-sharded Qwen3-VL text tower: one ``TextScorer`` over the whole mesh, each weight holding a different
checkpoint per chip (see the package docstring).

Every weight is built as ``torch.cat([w(lane_0), w(lane_1), ...], dim=0)`` and placed with ``ShardTensorToMesh(dim=0)``,
so chip ``i`` holds exactly the tensor the single-chip build of lane ``i``'s checkpoint would hold, at the same shape.
The programs are the single-chip programs (no collective anywhere), compiled once for the mesh.

The head computes both outputs on every chip: the final-normed last-token rows (the embedding) and their product
with the ``[hidden, 32]`` yes-minus-no column (the rerank score). Each lane reads the output of its own task.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List

import torch

import ttnn
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs import NO_TOKEN, YES_TOKEN
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.text_model import (
    TILE,
    LayerWeights,
    Precision,
    TextScorer,
    _norm_w,
    rope_trans_mat,
)
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.weights import Checkpoint, layer_tensors

LAYOUT_VERSION = 1  # bump when a split tensor's layout changes


def split_cache_dir(ckpts: Dict[str, Checkpoint], revisions: Dict[str, str], lanes: List[str]) -> Path:
    root = os.environ.get("TT_DIT_CACHE_DIR") or os.environ.get("TT_CACHE_PATH") or str(Path.home() / ".cache/ttnn")
    tag = "-".join(f"{task}{revisions[task][:10]}" for task in sorted(ckpts))
    d = Path(root) / f"qwen3vl-cohost-split-{tag}-{''.join(t[0] for t in lanes)}-L{LAYOUT_VERSION}"
    d.mkdir(parents=True, exist_ok=True)
    return d


class SplitScorer(TextScorer):
    """``lanes[i]`` names the checkpoint (a key of ``ckpts``) chip ``i`` runs."""

    def __init__(self, device, ckpts: Dict[str, Checkpoint], revisions: Dict[str, str], lanes: List[str], hf_config,
                 precision: Precision, num_layers=None):
        n = device.get_num_devices()
        if len(lanes) != n:
            raise ValueError(f"QWEN3VL_COHOST_LANES names {len(lanes)} chips; the mesh has {n}")
        unknown = set(lanes) - set(ckpts)
        if unknown:
            raise ValueError(f"lanes {sorted(unknown)} name no checkpoint; known: {sorted(ckpts)}")
        self.lanes = list(lanes)
        self.ckpts = ckpts
        self.cache = split_cache_dir(ckpts, revisions, lanes)
        first = next(iter(ckpts.values()))
        super().__init__(device, first, hf_config, precision=precision, num_layers=num_layers)

    # ---- weights ------------------------------------------------------------------------------------------------
    def _split(self, name: str, dtype, loader: Callable[[Checkpoint], torch.Tensor], layout=ttnn.TILE_LAYOUT):
        """One mesh tensor whose chip ``i`` shard is ``loader(ckpts[lanes[i]])`` (built once per distinct task)."""
        fname = str(self.cache / f"{name}.{str(dtype).split('.')[-1]}.{'tile' if layout == ttnn.TILE_LAYOUT else 'rm'}")
        if any(Path(fname).parent.glob(Path(fname).name + "*")):
            host = torch.zeros(len(self.lanes))  # as_tensor reads the cache file and ignores the host tensor
        else:
            per_task = {task: loader(self.ckpts[task]) for task in dict.fromkeys(self.lanes)}
            host = torch.cat([per_task[t] for t in self.lanes], dim=0)
        return ttnn.as_tensor(
            host,
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=fname,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
        )

    def _load(self, ckpt):  # the ``ckpt`` argument (the first checkpoint) is unused: every tensor is split
        d, P = self.device, self.precision
        self.embed = self._split(
            "embed_tokens_plus_zero",
            ttnn.bfloat16,
            lambda ck: torch.cat(
                [ck.get(ck.text_prefix + "embed_tokens.weight"), torch.zeros(1, self.hidden, dtype=torch.bfloat16)]
            ),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.layers = []
        for i in range(self.num_layers):
            memo: Dict[int, dict] = {}

            def lt(ck, key, i=i, memo=memo):
                k = id(ck)
                if k not in memo:
                    memo[k] = layer_tensors(ck, i, self.cfg)
                return memo[k][key]

            def W(key, dtype, rm=False, norm=False, i=i, lt=lt):
                if norm:
                    loader = lambda ck: _norm_w(lt(ck, key))  # [1, 1, n, 32]: cat on dim 0 = one per chip
                else:
                    loader = lambda ck: lt(ck, key)  # [in, out]: cat on dim 0 = [chips * in, out]
                return self._split(f"L{i}.{key}", dtype, loader, ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT)

            self.layers.append(
                LayerWeights(
                    wqkv=W("wqkv", P.attn_weights),
                    wo=W("wo", P.attn_weights),
                    q_norm=W("q_norm", ttnn.bfloat16, rm=True, norm=True),
                    k_norm=W("k_norm", ttnn.bfloat16, rm=True, norm=True),
                    w13=W("w13", P.mlp_weights),
                    w2=W("w2", P.mlp_weights),
                    attn_norm=W("attn_norm", ttnn.bfloat16, rm=True, norm=True),
                    mlp_norm=W("mlp_norm", ttnn.bfloat16, rm=True, norm=True),
                )
            )
            memo.clear()
        # The final RMSNorm and the yes-minus-no score run on the host, in fp32, on the selected rows: on device they
        # need a whole 4096-wide fp32 row per core in L1, which an LLM's resident L1 leaves no room for. Host fp32
        # matches the single-chip server's fp32 / HiFi4 device math.
        self.final_norm_host, self.score_vec = {}, {}
        for task in dict.fromkeys(self.lanes):
            ck = self.ckpts[task]
            self.final_norm_host[task] = ck.get(ck.text_prefix + "norm.weight").float()
            w = ck.get("lm_head.weight") if ck.has("lm_head.weight") else ck.get(ck.text_prefix + "embed_tokens.weight")
            # the official score_linear holds (w_yes - w_no) in model dtype
            self.score_vec[task] = (w[YES_TOKEN].float() - w[NO_TOKEN].float()).to(torch.bfloat16).float()
        self.trans_mat = ttnn.from_torch(
            rope_trans_mat(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d,
            mesh_mapper=ttnn.ReplicateTensorToMesh(d),
        )

    # ---- matmul: the tuned blocking where it fits, else the default ------------------------------------------------
    def matmul(self, x, w, ckc, swiglu=False):
        """The single-chip server's tuned minimal_matmul blocking assumes the whole L1. Next to an LLM that keeps L1
        resident, some blockings' circular buffers do not fit; those shapes fall back to the tt_dit default blocking
        (remembered per shape). The clash is a host-side check before dispatch, so the retry is safe."""
        M, K, N = int(x.shape[-2]), int(w.shape[-2]), int(w.shape[-1])
        key = (M, K, N, swiglu, ckc is self.ckc_mlp)
        if key in self._fallback:
            from models.tt_dit.utils.matmul import get_matmul_config

            cfg = get_matmul_config(M, K, N, self.device.compute_with_storage_grid_size())
            return ttnn.experimental.minimal_matmul(x, w, config=cfg, compute_kernel_config=ckc,
                                                     dtype=self.precision.activations, fuse_swiglu=swiglu)
        try:
            return super().matmul(x, w, ckc, swiglu)
        except RuntimeError as e:
            if "clash" not in str(e) and "circular buffer" not in str(e).lower():
                raise
            self._fallback.add(key)
            return self.matmul(x, w, ckc, swiglu)

    _fallback: set = set()

    # ---- head: the selected rows only (norm and score on the host) ------------------------------------------------
    def head(self, x, select):
        rows = ttnn.matmul(select, x, compute_kernel_config=self.ckc_head, dtype=ttnn.float32)
        ttnn.deallocate(x)
        return rows  # [1, 1, 32, hidden] fp32: row b = pair b's last-token hidden state

    def read_lane(self, rows, chip: int, B: int) -> list:
        """Chip ``chip``'s results for rows 0..B-1: the final-normed row (embedder) or its score (reranker)."""
        task = self.lanes[chip]
        t = ttnn.to_torch(ttnn.get_device_tensors(rows)[chip]).float()[0, 0, :B]
        n = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps) * self.final_norm_host[task]
        if task == "embed":
            return [n[b].numpy().copy() for b in range(B)]
        return [float(v) for v in n @ self.score_vec[task]]
