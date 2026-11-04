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
        self.final_norm = self._split(
            "final_norm",
            ttnn.bfloat16,
            lambda ck: _norm_w(ck.get(ck.text_prefix + "norm.weight")),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        def head(ck):
            # tied embeddings: the head rows are the embedding rows (the embedder's column is computed and unused)
            w = ck.get("lm_head.weight") if ck.has("lm_head.weight") else ck.get(ck.text_prefix + "embed_tokens.weight")
            v = w[YES_TOKEN].float() - w[NO_TOKEN].float()
            h = torch.zeros(self.hidden, TILE, dtype=torch.float32)
            h[:, 0] = v.to(torch.bfloat16).float()
            return h

        self.score_head = self._split("score_head_yes_minus_no", ttnn.float32, head)
        self.trans_mat = ttnn.from_torch(
            rope_trans_mat(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d,
            mesh_mapper=ttnn.ReplicateTensorToMesh(d),
        )

    # ---- head: both tasks' outputs on every chip -----------------------------------------------------------------
    def head(self, x, select):
        rows = ttnn.matmul(select, x, compute_kernel_config=self.ckc_head, dtype=ttnn.float32)
        ttnn.deallocate(x)
        n = ttnn.rms_norm(rows, weight=self.final_norm, epsilon=self.eps, compute_kernel_config=self.ckc_norm)
        ttnn.deallocate(rows)
        s = ttnn.matmul(n, self.score_head, compute_kernel_config=self.ckc_head, dtype=ttnn.float32)
        return n, s  # [1, 1, 32, hidden] embeddings, [1, 1, 32, 32] scores (row b, column 0)


def read_lane(out, chip: int, task: str, B: int) -> list:
    """Lane ``chip``'s results from a pass's ``(embeddings, scores)``: rows 0..B-1."""
    n, s = out
    t = ttnn.to_torch(ttnn.get_device_tensors(n if task == "embed" else s)[chip]).float()
    if task == "embed":
        return [t[0, 0, b].clone().numpy() for b in range(B)]
    return [float(t[0, 0, b, 0]) for b in range(B)]
