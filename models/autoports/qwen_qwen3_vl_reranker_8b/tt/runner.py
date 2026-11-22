# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host driver: pairs -> bucketed device inputs -> scores (eager; traced replay lives in ``traced.py``)."""
from __future__ import annotations

import torch

import ttnn
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs import PairInputs
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.text_model import TILE, TextScorer

PAD_ID = 151643


def bucket_for(n: int, buckets) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} tokens exceeds the largest bucket {buckets[-1]}")


class HostBatch:
    """Host tensors for one pass of B pairs at bucket length T."""

    def __init__(self, scorer: TextScorer, pairs: list[PairInputs], T: int, vis_index: torch.Tensor | None = None):
        B = len(pairs)
        assert B <= TILE
        self.B, self.T = B, T
        ids = torch.full((B, T), PAD_ID, dtype=torch.int32)
        sel = torch.zeros(1, 1, TILE, B * T, dtype=torch.float32)
        for b, p in enumerate(pairs):
            n = p.length
            assert n <= T, (n, T)
            row = p.input_ids.clone()
            if p.has_vision:
                cfg = scorer.hf_config
                row[(row == cfg.image_token_id) | (row == cfg.video_token_id)] = scorer.zero_row
            ids[b, :n] = row.int()
            sel[0, 0, b, b * T + n - 1] = 1.0
        pos = pairs[0].position_ids
        if B > 1:
            assert all(not p.has_vision for p in pairs), "batched passes are text only (shared arange positions)"
            pos = torch.arange(T)[None].expand(3, T)
        else:
            n = pos.shape[1]
            if n < T:  # padding rows continue the last position (never read: causal + one-hot select)
                tail = pos[:, -1:] + torch.arange(1, T - n + 1)[None]
                pos = torch.cat([pos, tail], dim=1)
        cos, sin = scorer.rope(pos)
        self.ids = ids.reshape(1, B * T)
        self.cos = cos[None, None].to(torch.bfloat16)
        self.sin = sin[None, None].to(torch.bfloat16)
        self.select = sel.to(torch.bfloat16)
        self.vis_index = vis_index

    def to_device(self, device):
        rep = ttnn.ReplicateTensorToMesh(device)
        kw = dict(device=device, mesh_mapper=rep)
        out = {
            "ids": ttnn.from_torch(self.ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, **kw),
            "cos": ttnn.from_torch(self.cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **kw),
            "sin": ttnn.from_torch(self.sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **kw),
            "select": ttnn.from_torch(self.select, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **kw),
        }
        if self.vis_index is not None:
            out["vis_index"] = ttnn.from_torch(
                self.vis_index.reshape(1, -1).int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, **kw
            )
        return out


def read_scores(score_tile, B) -> list[float]:
    t = ttnn.to_torch(ttnn.get_device_tensors(score_tile)[0]).float()
    return [float(t[0, 0, b, 0]) for b in range(B)]


def visual_index(scorer: TextScorer, p: PairInputs, tables, T: int) -> torch.Tensor:
    """[T] table row per position: 0 for text, the image / video token's table row otherwise (sequence order)."""
    idx = torch.zeros(T, dtype=torch.int32)
    ids = p.input_ids
    img = (ids == scorer.hf_config.image_token_id).nonzero().flatten()
    vid = (ids == scorer.hf_config.video_token_id).nonzero().flatten()
    assert len(img) == len(tables.image_rows) and len(vid) == len(tables.video_rows), (
        len(img),
        len(tables.image_rows),
        len(vid),
        len(tables.video_rows),
    )
    idx[img] = tables.image_rows.int()
    idx[vid] = tables.video_rows.int()
    return idx


class EagerRunner:
    def __init__(self, scorer: TextScorer, vision=None, buckets=(128, 256, 512, 1024, 2048, 4096, 8192, 16384)):
        self.scorer = scorer
        self.vision = vision
        self.buckets = tuple(buckets)

    def score_pair(self, p: PairInputs, T: int | None = None, cache=None) -> float:
        if not p.has_vision:
            return self.score_text([p], T)[0]
        T = T or bucket_for(p.length, self.buckets)
        tables = self.vision.tables(
            p.pixel_values, p.image_grid_thw, p.pixel_values_videos, p.video_grid_thw, cache=cache
        )
        hb = HostBatch(self.scorer, [p], T, vis_index=visual_index(self.scorer, p, tables, T))
        d = hb.to_device(self.scorer.device)
        s = self.scorer.forward(
            d["ids"], d["cos"], d["sin"], d["select"], 1, T, vis_index=d["vis_index"], vis_tables=tables.all()
        )
        out = read_scores(s, 1)
        for t in (s, *d.values()):
            ttnn.deallocate(t)
        tables.deallocate()
        return out[0]

    def score_text(self, pairs: list[PairInputs], T: int | None = None) -> list[float]:
        T = T or bucket_for(max(p.length for p in pairs), self.buckets)
        hb = HostBatch(self.scorer, pairs, T)
        d = hb.to_device(self.scorer.device)
        s = self.scorer.forward(d["ids"], d["cos"], d["sin"], d["select"], hb.B, T)
        out = read_scores(s, hb.B)
        for t in (s, *d.values()):
            ttnn.deallocate(t)
        return out
