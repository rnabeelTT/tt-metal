# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL vision tower on one die: the Blackhole qwen36 ViT blocks + merger (vendored from the Ornith-1.5 bundle,
``tt/vision``) plus the three DeepStack mergers Qwen3-VL adds.

Per image, or per temporal frame group of a video (attention is block-diagonal per group, so a t-group video is
exactly t images): the bilinear position embedding, the vision rotary and the window bounds depend on the grid alone
and are cached on host per grid; on device run the patch embedding (the stride == kernel Conv3d as a linear), 27 ViT
blocks over the patch sequence padded to a 1024 / 2048 multiple with block-diagonal attention windows (the real
patches, then the padding), the main merger, and a DeepStack merger tapped after blocks 8, 16 and 24.

The outputs stay on device. ``VisionTables`` stacks, per feature stream, a 32-row zero block followed by every
encoded piece's padded merger output, as ROW_MAJOR ``[rows, hidden]`` tables; the text tower gathers them with
``ttnn.embedding`` through a per-position index (0 = text, else the row of that visual token), so no visual rows are
read back or rewritten on host.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import torch

import ttnn
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.weights import Checkpoint, as_device

PATCH_BLOCK = 2048  # the qwen36 QKV projection chunks rows by 2048 above 2048
SMALL_BLOCK = 1024  # below 2048 the blocks only need 1024-row multiples
ZERO_ROWS = 32


def padded_len(n: int) -> int:
    if n <= PATCH_BLOCK:
        return max(SMALL_BLOCK, -(-n // SMALL_BLOCK) * SMALL_BLOCK)
    return -(-n // PATCH_BLOCK) * PATCH_BLOCK


def padded_buckets(max_patches: int = 8192) -> list[int]:
    return sorted({padded_len(n) for n in range(1, max_patches + 1, 256)} | {padded_len(max_patches)})


def build_hf_visual(ckpt: Checkpoint, hf_config, dtype=torch.float32):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    visual = Qwen3VLVisionModel._from_config(hf_config.vision_config)
    sd = {k: v.to(dtype) for k, v in ckpt.visual_state_dict().items()}
    missing, unexpected = visual.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return visual.to(dtype).eval()


def _vision_args(mesh, hf_model_path: str, max_seq_len: int):
    """qwen36 ``VisionModelArgs`` over this checkpoint (tt_transformers ``ModelArgs`` reads ``HF_MODEL``)."""
    from models.autoports.qwen_qwen3_vl_reranker_8b.tt.vision.vision_model_config import VisionModelArgs

    prev = os.environ.get("HF_MODEL")
    os.environ["HF_MODEL"] = hf_model_path
    try:
        return VisionModelArgs(mesh, max_batch_size=1, max_seq_len=max_seq_len, cache_hf=False)
    finally:
        if prev is not None:
            os.environ["HF_MODEL"] = prev


class DeepstackMerger:
    """Qwen3VLVisionPatchMerger with ``use_postshuffle_norm=True``: merge 2x2 patches, LayerNorm(4608), fc1+GELU, fc2."""

    def __init__(self, mesh, sd: dict, prefix: str, dtype, eps=1e-6, name=""):
        self.mesh = mesh
        self.eps = eps
        self.mlp_size = int(sd[f"{prefix}.linear_fc1.weight"].shape[1])

        def W(key, t, dt, layout=ttnn.TILE_LAYOUT):
            return as_device(t, mesh, dt, f"vision.{name}.{key}", layout=layout)

        # TILE-layout affine params (32 rows): the only layer_norm variant whose L1 buffers fit a 4608-wide row
        tile_rows = lambda v: v.reshape(1, 1, 1, -1).expand(1, 1, 32, -1).contiguous()
        self.norm_w = W("norm_w_tile", tile_rows(sd[f"{prefix}.norm.weight"]), ttnn.bfloat16)
        self.norm_b = W("norm_b_tile", tile_rows(sd[f"{prefix}.norm.bias"]), ttnn.bfloat16)
        self.w1 = W("fc1_w", sd[f"{prefix}.linear_fc1.weight"].T.contiguous(), dtype)
        self.b1 = W("fc1_b", sd[f"{prefix}.linear_fc1.bias"].reshape(1, -1), ttnn.bfloat16)
        self.w2 = W("fc2_w", sd[f"{prefix}.linear_fc2.weight"].T.contiguous(), dtype)
        self.b2 = W("fc2_b", sd[f"{prefix}.linear_fc2.bias"].reshape(1, -1), ttnn.bfloat16)
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # a 4608-wide row does not fit the norm's L1 buffers with fp32 dest accumulation
        self.ckc_norm = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (x.shape[0], x.shape[1], -1, self.mlp_size))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        n = ttnn.layer_norm(
            x, weight=self.norm_w, bias=self.norm_b, epsilon=self.eps, compute_kernel_config=self.ckc_norm
        )
        ttnn.deallocate(x)
        h = ttnn.linear(n, self.w1, bias=self.b1, activation="gelu", compute_kernel_config=self.ckc)
        ttnn.deallocate(n)
        o = ttnn.linear(h, self.w2, bias=self.b2, compute_kernel_config=self.ckc)
        ttnn.deallocate(h)
        return o


@dataclass
class VisionTables:
    """Device tables [rows, hidden] ROW_MAJOR: index 0 of ``main`` / ``deepstack[i]`` is a zero row."""

    main: ttnn.Tensor
    deepstack: list
    image_rows: torch.Tensor  # table row of each image token, in sequence order
    video_rows: torch.Tensor

    def all(self):
        return [self.main, *self.deepstack]

    def deallocate(self):
        for t in self.all():
            ttnn.deallocate(t)


class VisionTower:
    def __init__(self, mesh, ckpt: Checkpoint, hf_config, model_path: str, dtype=ttnn.bfloat8_b, max_patches=16384):
        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.vision.model import DropInVisionTransformer

        self.mesh = mesh
        self.cfg = hf_config.vision_config
        self.merge2 = self.cfg.spatial_merge_size**2
        self.out_hidden = self.cfg.out_hidden_size
        self.taps = list(self.cfg.deepstack_visual_indexes)
        self.hf_visual = build_hf_visual(ckpt, hf_config)
        self.args = _vision_args(mesh, model_path, max_seq_len=padded_len(max_patches))
        self.drop_in = DropInVisionTransformer(self.hf_visual, self.args, dtype=dtype)
        self.tt = self.drop_in.tt_model
        sd = {k: v.float() for k, v in self.hf_visual.state_dict().items() if k.startswith("deepstack_merger_list.")}
        self.deepstack = [
            DeepstackMerger(mesh, sd, f"deepstack_merger_list.{i}", dtype, name=f"ds{i}") for i in range(len(self.taps))
        ]
        # Conv3d patch embedding with stride == kernel is a [1536 -> 1152] linear: run it on device
        pe = self.hf_visual.patch_embed.proj
        self.patch_w = as_device(
            pe.weight.detach().reshape(pe.weight.shape[0], -1).T.contiguous().float(),
            mesh,
            ttnn.bfloat16,
            "vision.patch_w",
        )
        self.patch_b = as_device(pe.bias.detach().reshape(1, -1).float(), mesh, ttnn.bfloat16, "vision.patch_b")
        self.ckc_patch = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._grid_cache = {}  # (t, h, w) -> per-grid device inputs (position embedding, rotary, windows)
        self.zero_rows = ttnn.from_torch(
            torch.zeros(1, 1, ZERO_ROWS, self.out_hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def grid_inputs(self, grid_thw):
        """Device position embedding [1, 1, seq, 1152] (zero past n), rotary cos / sin and the frame windows of one
        (t, h, w) grid; cached, since they depend on the grid alone."""
        key = tuple(int(v) for v in grid_thw.reshape(-1))
        hit = self._grid_cache.get(key)
        if hit is not None:
            return hit
        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.vision.functional import (
            qwen3_5_vision_transformer_preprocess,
        )
        from models.ttt_compat.tt.load_checkpoints import convert_rope_style_hf_to_meta

        g = torch.tensor([key])
        n = int(g.prod())
        seq = padded_len(n)
        _, (cos, sin) = qwen3_5_vision_transformer_preprocess(
            seq_len=n, grid_thw=g, head_dim=self.args.head_dim, spatial_merge_size=self.cfg.spatial_merge_size
        )
        with torch.no_grad():
            pos = self.hf_visual.fast_pos_embed_interpolate(g).float()
        pos = torch.nn.functional.pad(pos, (0, 0, 0, seq - n))[None, None]
        cos, sin = convert_rope_style_hf_to_meta(cos, sin)
        cos = torch.nn.functional.pad(cos, (0, 0, 0, seq - n), value=1)[None, None]
        sin = torch.nn.functional.pad(sin, (0, 0, 0, seq - n), value=0)[None, None]
        # Host tensors (bf16, tile-converted once): a device tensor kept alive across calls would be allocated after
        # the text traces were captured and could sit on memory their replays reuse.
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        host = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep
        )
        t, h, w = key
        hit = {
            "pos": host(pos),
            "cos": host(cos),
            "sin": host(sin),
            "n": n,
            "seq": seq,
            "bounds": [i * h * w for i in range(t + 1)],
        }
        if len(self._grid_cache) >= 64:  # small LRU: drop the oldest grid
            self._grid_cache.pop(next(iter(self._grid_cache)))
        self._grid_cache[key] = hit
        return hit

    def encode_one(self, pixel_values, grid_thw):
        """-> (main, [deepstack...]) device tensors [1, 1, seq/4, hidden] TILE (rows past n/4 are padding)."""
        gi = self.grid_inputs(grid_thw)
        n, seq = gi["n"], gi["seq"]
        px = torch.nn.functional.pad(pixel_values.to(torch.bfloat16), (0, 0, 0, seq - n))[None, None]
        px_t = ttnn.from_torch(
            px,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        emb = ttnn.linear(px_t, self.patch_w, bias=self.patch_b, compute_kernel_config=self.ckc_patch)
        ttnn.deallocate(px_t)
        pos, cos, sin = (ttnn.to_device(gi[k], self.mesh) for k in ("pos", "cos", "sin"))
        x_t = ttnn.add(emb, pos)
        ttnn.deallocate(emb)
        ttnn.deallocate(pos)
        taps = {}

        def tap(i, h):
            taps[i] = self.deepstack[self.taps.index(i)](h)

        main = self.tt(
            x_t,
            unpadded_seq_len=n,
            rot_mats=[cos, sin],
            window_bounds=gi["bounds"],
            tap_fn=tap,
            tap_layers=self.taps,
        )
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        return main, [taps[i] for i in self.taps], n // self.merge2

    def tables(self, pixel_values=None, image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None, cache=None):
        """Encode every image, then every video, of one pair and build the gather tables. ``image_rows`` /
        ``video_rows`` are the table rows of the image / video tokens in sequence order. ``cache`` (a dict the caller
        owns and frees with ``free_cache``) reuses an image already encoded in the same call, e.g. a query image
        shared by every document."""
        mains, deeps, owned = [], [[] for _ in self.taps], []
        rows = {"image": [], "video": []}
        offset = ZERO_ROWS
        for kind, pv, grids in (
            ("image", pixel_values, image_grid_thw),
            ("video", pixel_values_videos, video_grid_thw),
        ):
            if grids is None:
                continue
            start = 0
            pieces = []
            for g in grids:
                npatch = int(g.prod())
                t, h, w = (int(v) for v in g)
                # Attention is block-diagonal per temporal frame group, and every other op is per token, with the
                # position embedding and rotary repeated per group: a t-group video is exactly t [1, h, w] images.
                # Encoding them one by one keeps every tower pass inside the image buckets.
                for k in range(t):
                    pieces.append((pv[start + k * h * w : start + (k + 1) * h * w], torch.tensor([1, h, w])))
                start += npatch
            for chunk, g in pieces:
                key = None
                if cache is not None:
                    key = (
                        hashlib.sha1(chunk.to(torch.bfloat16).view(torch.int16).numpy().tobytes()).hexdigest(),
                        tuple(int(v) for v in g),
                    )
                if key is not None and key in cache:
                    m, ds, ntok = cache[key]
                else:
                    m, ds, ntok = self.encode_one(chunk, g)
                    if key is not None:
                        cache[key] = (m, ds, ntok)
                    else:
                        owned += [m, *ds]
                mains.append(m)
                for k, d in enumerate(ds):
                    deeps[k].append(d)
                rows[kind].append(torch.arange(ntok) + offset)
                offset += int(m.shape[2])

        def stack(parts):
            t = ttnn.concat([self.zero_rows, *parts], dim=2) if parts else self.zero_rows
            rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
            if parts:
                ttnn.deallocate(t)
            return ttnn.reshape(rm, (rm.shape[2], rm.shape[3]))

        out = VisionTables(
            main=stack(mains),
            deepstack=[stack(d) for d in deeps],
            image_rows=torch.cat(rows["image"]) if rows["image"] else torch.zeros(0, dtype=torch.long),
            video_rows=torch.cat(rows["video"]) if rows["video"] else torch.zeros(0, dtype=torch.long),
        )
        for t in owned:
            ttnn.deallocate(t)
        return out

    @staticmethod
    def free_cache(cache: dict):
        for m, ds, _ in cache.values():
            ttnn.deallocate(m)
            for d in ds:
                ttnn.deallocate(d)
        cache.clear()
