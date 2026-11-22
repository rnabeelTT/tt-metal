# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL text tower as a prefill-only relevance scorer on one Blackhole die.

One pass takes ``B`` pairs padded to a bucket length ``T`` (right padding: causal attention keeps padding from
reaching any real row), runs the 36 decoder layers, picks each pair's last real row with a one-hot matmul, applies
the final RMSNorm and a ``[hidden, 32]`` head whose column 0 is ``lm_head[yes] - lm_head[no]``. Column 0 of row ``b``
is pair ``b``'s score; ``sigmoid(score)`` is the relevance probability.

Vision features enter as two gathers (trace-safe, no host scatter): ``ttnn.embedding`` over the token ids against
the text table with one extra zero row (image/video positions point at it), plus ``ttnn.embedding`` over a per-row
visual index against ``[zero row; visual rows]``. The DeepStack features enter the same way after decoder layers
0..2. MRoPE needs no special kernel: cos/sin rows are built on host from the [3, T] positions and fed as inputs.

Attention is Qwen3's (fused QKV, per-head QK-norm, RoPE in the Meta channel pairing, causal GQA SDPA); the QKV
permutation and per-head norm follow the Qwen3-Coder-30B Blackhole port.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.inputs import NO_TOKEN, YES_TOKEN
from models.autoports.qwen_qwen3_vl_reranker_8b.tt.weights import TEXT_PREFIX, Checkpoint, as_device, layer_tensors
from models.tt_dit.utils.matmul import get_matmul_config

TILE = 32


@dataclass
class Precision:
    attn_weights: object = ttnn.bfloat8_b
    mlp_weights: object = ttnn.bfloat8_b
    activations: object = ttnn.bfloat16
    attn_fidelity: object = ttnn.MathFidelity.HiFi2
    mlp_fidelity: object = ttnn.MathFidelity.HiFi2
    sdpa_fidelity: object = ttnn.MathFidelity.HiFi4
    fp32_acc: bool = True
    tag: str = "default"

    def key(self):
        return f"a{str(self.attn_weights).split('.')[-1]}_m{str(self.mlp_weights).split('.')[-1]}"

    _DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
    _MF = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}

    @classmethod
    def from_dict(cls, d: dict) -> "Precision":
        return cls(
            attn_weights=cls._DT[d.get("attn_weights", "bfp8")],
            mlp_weights=cls._DT[d.get("mlp_weights", "bfp8")],
            activations=cls._DT[d.get("activations", "bf16")],
            attn_fidelity=cls._MF[d.get("attn_fidelity", "hifi2")],
            mlp_fidelity=cls._MF[d.get("mlp_fidelity", "hifi2")],
            sdpa_fidelity=cls._MF[d.get("sdpa_fidelity", "hifi4")],
            fp32_acc=bool(d.get("fp32_acc", True)),
            tag=d.get("tag", "custom"),
        )

    @classmethod
    def load(cls, name_or_path: str | None = None) -> "Precision":
        """``QWEN3VL_RERANKER_PRECISION``: a JSON file, or a policy name in ``config/precision.json``
        (whose ``selected`` entry is the default)."""
        import os

        name_or_path = name_or_path or os.environ.get("QWEN3VL_RERANKER_PRECISION")
        if name_or_path and Path(name_or_path).is_file():
            return cls.from_dict(json.loads(Path(name_or_path).read_text()))
        table = json.loads(PRECISION_PATH.read_text()) if PRECISION_PATH.exists() else {}
        name = name_or_path or table.get("selected")
        if not name:
            return cls()
        d = dict(table["policies"][name])
        d.setdefault("tag", name)
        return cls.from_dict(d)


def _ckc(device, fidelity, fp32_acc=True, packer_l1=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1,
    )


def rope_trans_mat() -> torch.Tensor:
    m = torch.zeros(1, 1, TILE, TILE)
    m[..., torch.arange(0, TILE, 2), torch.arange(1, TILE, 2)] = 1
    m[..., torch.arange(1, TILE, 2), torch.arange(0, TILE, 2)] = -1
    return m


class MRoPE:
    """Host cos/sin for MRoPE positions, in the Meta channel pairing ``[c0, c0, c1, c1, ...]``."""

    def __init__(self, text_config):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

        self.rope = Qwen3VLTextRotaryEmbedding(text_config)
        hd = text_config.head_dim
        from models.autoports.qwen_qwen3_vl_reranker_8b.tt.weights import hf_to_meta_channels

        self.perm = hf_to_meta_channels(hd)

    @torch.no_grad()
    def __call__(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids [3, T] -> cos, sin [T, head_dim] float32 (Meta order)."""
        x = torch.zeros(1, dtype=torch.float32)
        cos, sin = self.rope(x, position_ids[:, None, :].long())
        return cos[0][:, self.perm], sin[0][:, self.perm]


@dataclass
class LayerWeights:
    wqkv: ttnn.Tensor
    wo: ttnn.Tensor
    q_norm: ttnn.Tensor
    k_norm: ttnn.Tensor
    w13: ttnn.Tensor
    w2: ttnn.Tensor
    attn_norm: ttnn.Tensor
    mlp_norm: ttnn.Tensor


def _norm_w(t: torch.Tensor):
    return t.reshape(1, 1, -1, TILE)


class TextScorer:
    def __init__(self, device, ckpt: Checkpoint, hf_config, precision: Precision | None = None, num_layers=None):
        self.device = device
        self.cfg = hf_config.text_config
        self.hf_config = hf_config
        self.precision = precision or Precision.load()
        self.num_layers = num_layers or self.cfg.num_hidden_layers
        self.hidden = self.cfg.hidden_size
        self.nh, self.nkv, self.hd = self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim
        self.eps = self.cfg.rms_norm_eps
        self.vocab = self.cfg.vocab_size
        self.zero_row = self.vocab  # extra zero row of the embedding table: image / video positions
        self.deepstack_layers = len(hf_config.vision_config.deepstack_visual_indexes)
        P = self.precision
        self.ckc_attn = _ckc(device, P.attn_fidelity, P.fp32_acc)
        self.ckc_mlp = _ckc(device, P.mlp_fidelity, P.fp32_acc)
        self.ckc_sdpa = _ckc(device, P.sdpa_fidelity, False, False)
        self.ckc_norm = _ckc(device, ttnn.MathFidelity.HiFi4, True, False)
        self.ckc_head = _ckc(device, ttnn.MathFidelity.HiFi4, True, False)
        self.rope = MRoPE(self.cfg)
        self._mm_cfg = {}
        self._load(ckpt)

    # ---- weights ----------------------------------------------------------------------------------------------
    def _load(self, ckpt: Checkpoint):
        d, P = self.device, self.precision
        self.embed = as_device(
            None,
            d,
            ttnn.bfloat16,
            "embed_tokens_plus_zero",
            layout=ttnn.ROW_MAJOR_LAYOUT,
            loader=lambda: torch.cat(
                [ckpt.get(TEXT_PREFIX + "embed_tokens.weight"), torch.zeros(1, self.hidden, dtype=torch.bfloat16)]
            ),
        )
        self.layers = []
        for i in range(self.num_layers):
            host = {}

            def lt(key, i=i, host=host):
                if not host:
                    host.update(layer_tensors(ckpt, i, self.cfg))
                return host[key]

            def W(key, dtype, i=i, lt=lt, rm=False, norm=False):
                return as_device(
                    None,
                    d,
                    dtype,
                    f"L{i}.{key}",
                    layout=ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT,
                    loader=(lambda: _norm_w(lt(key))) if norm else (lambda: lt(key)),
                )

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
            host.clear()
        self.final_norm = as_device(
            None,
            d,
            ttnn.bfloat16,
            "final_norm",
            layout=ttnn.ROW_MAJOR_LAYOUT,
            loader=lambda: _norm_w(ckpt.get(TEXT_PREFIX + "norm.weight")),
        )

        def head():
            w = ckpt.get("lm_head.weight")
            v = w[YES_TOKEN].float() - w[NO_TOKEN].float()
            h = torch.zeros(self.hidden, TILE, dtype=torch.float32)
            h[:, 0] = v.to(torch.bfloat16).float()  # the official score_linear holds (w_yes - w_no) in model dtype
            return h

        self.score_head = as_device(None, d, ttnn.float32, "score_head_yes_minus_no", loader=head)
        self.trans_mat = ttnn.from_torch(
            rope_trans_mat(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=d,
            mesh_mapper=ttnn.ReplicateTensorToMesh(d),
        )

    # ---- one decoder layer ------------------------------------------------------------------------------------
    def _per_head_norm(self, t, w):
        s = t.shape
        flat = ttnn.reshape(t, (1, 1, s[0] * s[1] * s[2], s[3]))
        out = ttnn.rms_norm(flat, weight=w, epsilon=self.eps, compute_kernel_config=self.ckc_norm)
        ttnn.deallocate(flat)
        return ttnn.reshape(out, s)

    def attention(self, x, lw: LayerWeights, cos, sin, B, T):
        act = self.precision.activations
        qkv = self.matmul(x, lw.wqkv, self.ckc_attn)
        if B > 1:
            qkv = ttnn.reshape(qkv, (B, 1, T, qkv.shape[-1]))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.nh,
            num_kv_heads=self.nkv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q = self._per_head_norm(q, lw.q_norm)
        k = self._per_head_norm(k, lw.k_norm)
        q2 = ttnn.experimental.rotary_embedding_llama(q, cos, sin, self.trans_mat, is_decode_mode=False)
        k2 = ttnn.experimental.rotary_embedding_llama(k, cos, sin, self.trans_mat, is_decode_mode=False)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q2, k2, v, is_causal=True, program_config=self._sdpa_cfg(T), compute_kernel_config=self.ckc_sdpa
        )
        for t in (q2, k2, v):
            ttnn.deallocate(t)
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if B > 1:
            attn = ttnn.reshape(attn, (1, 1, B * T, attn.shape[-1]))
        out = self.matmul(attn, lw.wo, self.ckc_attn)
        ttnn.deallocate(attn)
        return out

    def _sdpa_cfg(self, T):
        chunk = 256 if T >= 2048 else (128 if T >= 128 else 32)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )

    def matmul(self, x, w, ckc, swiglu=False):
        """Prefill projection: minimal_matmul with the tt_dit Blackhole blocking for this (M, K, N) and grid
        (2.3x the default ``ttnn.linear`` program on the 4096 x 12288 projections at 4K rows)."""
        M, K, N = int(x.shape[-2]), int(w.shape[-2]), int(w.shape[-1])
        key = (M, K, N, swiglu, ckc is self.ckc_mlp)
        cfg = self._mm_cfg.get(key)
        if cfg is None:
            cfg = self._mm_cfg[key] = tuned_matmul_config(
                M, K, N, swiglu, self.device.compute_with_storage_grid_size(), self.fidelity_name(ckc)
            )
        return ttnn.experimental.minimal_matmul(
            x, w, config=cfg, compute_kernel_config=ckc, dtype=self.precision.activations, fuse_swiglu=swiglu
        )

    def fidelity_name(self, ckc):
        f = self.precision.mlp_fidelity if ckc is self.ckc_mlp else self.precision.attn_fidelity
        return "lofi" if f == ttnn.MathFidelity.LoFi else "hifi2"

    def mlp(self, x, lw: LayerWeights):
        h = self.matmul(x, lw.w13, self.ckc_mlp, swiglu=True)
        out = self.matmul(h, lw.w2, self.ckc_mlp)
        ttnn.deallocate(h)
        return out

    def layer(self, x, i, cos, sin, B, T):
        lw = self.layers[i]
        n = ttnn.rms_norm(x, weight=lw.attn_norm, epsilon=self.eps, compute_kernel_config=self.ckc_norm)
        a = self.attention(n, lw, cos, sin, B, T)
        ttnn.deallocate(n)
        h = ttnn.add(x, a)
        ttnn.deallocate(a)
        n = ttnn.rms_norm(h, weight=lw.mlp_norm, epsilon=self.eps, compute_kernel_config=self.ckc_norm)
        m = self.mlp(n, lw)
        ttnn.deallocate(n)
        out = ttnn.add(h, m)
        ttnn.deallocate(h)
        ttnn.deallocate(m)
        return out

    # ---- full pass --------------------------------------------------------------------------------------------
    def embed_tokens(self, ids):
        """ids: [1, B*T] uint32 ROW_MAJOR on device -> [1, 1, B*T, hidden] TILE."""
        x = ttnn.embedding(ids, self.embed, layout=ttnn.TILE_LAYOUT, dtype=self.precision.activations)
        return ttnn.reshape(x, (1, 1, x.shape[-2], x.shape[-1]))

    def gather_visual(self, vis_index, table):
        """vis_index [1, B*T] uint32 (0 = text row), table [n+1, hidden] ROW_MAJOR with row 0 zero."""
        g = ttnn.embedding(vis_index, table, layout=ttnn.TILE_LAYOUT, dtype=self.precision.activations)
        return ttnn.reshape(g, (1, 1, g.shape[-2], g.shape[-1]))

    def forward(self, ids, cos, sin, select, B, T, vis_index=None, vis_tables=None):
        """ids [1, B*T]; cos/sin [B? 1, 1, T, hd]; select [1, 1, 32, B*T] one-hot rows; vis_tables =
        (main, deepstack_0, deepstack_1, deepstack_2) each [n+1, hidden] ROW_MAJOR, or None (text only).
        Returns the [1, 1, 32, 32] score tile (row b, col 0 = pair b's score)."""
        x = self.embed_tokens(ids)
        if vis_tables is not None:
            v = self.gather_visual(vis_index, vis_tables[0])
            x2 = ttnn.add(x, v)
            ttnn.deallocate(x)
            ttnn.deallocate(v)
            x = x2
        for i in range(self.num_layers):
            x2 = self.layer(x, i, cos, sin, B, T)
            ttnn.deallocate(x)
            x = x2
            if vis_tables is not None and i < self.deepstack_layers:
                v = self.gather_visual(vis_index, vis_tables[1 + i])
                x2 = ttnn.add(x, v)
                ttnn.deallocate(x)
                ttnn.deallocate(v)
                x = x2
        return self.head(x, select)

    def head(self, x, select):
        rows = ttnn.matmul(select, x, compute_kernel_config=self.ckc_head, dtype=ttnn.float32)
        ttnn.deallocate(x)
        n = ttnn.rms_norm(rows, weight=self.final_norm, epsilon=self.eps, compute_kernel_config=self.ckc_norm)
        ttnn.deallocate(rows)
        s = ttnn.matmul(n, self.score_head, compute_kernel_config=self.ckc_head, dtype=ttnn.float32)
        ttnn.deallocate(n)
        return s


PRECISION_PATH = Path(__file__).resolve().parents[1] / "config" / "precision.json"
BLOCKING_PATH = Path(__file__).resolve().parents[1] / "config" / "matmul_blocking.json"
_BLOCKING = None


def tuned_matmul_config(M, K, N, swiglu, grid, fidelity="hifi2"):
    """minimal_matmul blocking from the autotuned table (``config/matmul_blocking.json``, keyed
    ``MxKxN[s]@GXxGY:fidelity``); an untuned M takes the tuned entry with the nearest M for the same K, N, else the
    other fidelity's, else the tt_dit Blackhole default."""
    global _BLOCKING
    if _BLOCKING is None:
        _BLOCKING = json.loads(BLOCKING_PATH.read_text()) if BLOCKING_PATH.exists() else {}
    g = f"{grid.x}x{grid.y}"
    tag = f"x{K}x{N}{'s' if swiglu else ''}@{g}"
    for fid in (fidelity, "hifi2", "lofi"):
        cands = [
            (int(k.split("x")[0]), v)
            for k, v in _BLOCKING.items()
            if k.endswith(f"{tag}:{fid}") and k.startswith(tuple("0123456789"))
        ]
        if cands:
            m, v = min(cands, key=lambda mv: (abs(math.log2(mv[0] / M)), -mv[0]))
            mb, kb, nb, sh, sw = v["cfg"]
            mb = max(1, min(mb, M // TILE))
            while mb % sh:
                sh //= 2
            return ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=max(sh, 1),
                subblock_w=sw,
                compute_with_storage_grid_size=grid,
            )
    return get_matmul_config(M, K, N, grid)


def pad_to(n: int, m: int) -> int:
    return int(math.ceil(n / m) * m)
