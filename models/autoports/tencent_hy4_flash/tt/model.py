# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-flash text model on a (1, N) Blackhole mesh (HF HYV4FlashTextModel + lm_head), day-0 eager form.

    X_i = embed(ids)                         (4 identical streams)
    per layer: [engram] -> attn HC -> input_layernorm -> attention -> HC post -> ffn HC -> post_attention_layernorm
               -> dense MLP / MoE -> HC post
    h = norm(HyperHead(X)) ; logits = lm_head(h)   (vocab-sharded, fp32 accumulation, gathered)

`forward(ids, p0)` runs T tokens at positions p0 .. p0 + T - 1 of the one sequence (prefill chunk or decode rows) and
returns the logits of the requested rows. The KV / indexer state advances with every call; `reset()` starts over.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import ttnn
from loguru import logger

from models.autoports.tencent_hy4_flash.tt.attention import Hy4Attention
from models.autoports.tencent_hy4_flash.tt.common import (
    AllGather,
    RMSNorm,
    ckc,
    num_devices,
    to_torch_cat,
    upload,
)
from models.autoports.tencent_hy4_flash.tt.config import Hy4Config
from models.autoports.tencent_hy4_flash.tt.engram import Hy4EngramDevice
from models.autoports.tencent_hy4_flash.tt.hc import head_hc, sublayer_hc
from models.autoports.tencent_hy4_flash.tt.moe import Hy4DenseMLP, Hy4MoE
from models.autoports.tencent_hy4_flash.tt.rope import DeviceRope
from models.autoports.tencent_hy4_flash.tt.weights import Hy4Weights


class Hy4Layer:
    def __init__(self, mesh, cfg, w: Hy4Weights, li: int, max_ctx: int, chunk: int, cache_dir=None):
        self.li, self.cfg = li, cfg
        sd = w.layer(li)
        self.attn_hc = sublayer_hc(mesh, sd, "hc_attn_layer", cfg)
        self.ffn_hc = sublayer_hc(mesh, sd, "hc_mlp_layer", cfg)
        self.in_norm = RMSNorm(mesh, sd["input_layernorm.weight"], cfg.rms_eps)
        self.post_norm = RMSNorm(mesh, sd["post_attention_layernorm.weight"], cfg.rms_eps)
        self.attn = Hy4Attention(mesh, cfg, sd.sub("self_attn."), li, max_ctx, chunk)
        self.mlp = Hy4MoE(mesh, cfg, sd.sub("mlp."), li, cache_dir) if cfg.is_moe(li) else Hy4DenseMLP(mesh, cfg, sd.sub("mlp."))
        self.engram = Hy4EngramDevice(mesh, cfg, sd.sub("engram_layer.")) if li in cfg.engram_layers else None

    def forward(self, xs, p0, T, cos_sin, engram_emb=None, mem=None):
        if self.engram is not None and engram_emb is not None:
            xs = self.engram.forward(xs, engram_emb, mem)
        a_in, post = self.attn_hc.pre(xs, mem)
        a = self.attn.forward(self.in_norm(a_in, mem), p0, T, cos_sin, mem)
        ttnn.deallocate(a_in)
        xs = self.attn_hc.post(xs, a, post, mem)
        ttnn.deallocate(a)
        f_in, post = self.ffn_hc.pre(xs, mem)
        f_n = self.post_norm(f_in, mem)
        ttnn.deallocate(f_in)
        f = self.mlp.forward(f_n, T, mem) if isinstance(self.mlp, Hy4MoE) else self.mlp.forward(f_n, mem)
        ttnn.deallocate(f_n)
        xs = self.ffn_hc.post(xs, f, post, mem)
        ttnn.deallocate(f)
        return xs

    def reset(self):
        if self.attn.indexer is not None:
            self.attn.indexer.reset()


class Hy4Model:
    def __init__(self, mesh, cfg: Hy4Config | None = None, w: Hy4Weights | None = None, max_ctx: int = 32768,
                 chunk: int = 2048, layers: list[int] | None = None, engram=None, cache_dir=None):
        """layers: a subset of decoder layers (probes / tests); engram: a host Engram store (None: zero rows, tests)."""
        self.mesh = mesh
        self.cfg = cfg = cfg or Hy4Config.load()
        self.w = w = w or Hy4Weights()
        self.N = num_devices(mesh)
        self.max_ctx, self.chunk = max_ctx, chunk
        self.layer_ids = list(range(cfg.n_layers)) if layers is None else list(layers)
        self.engram = engram
        self.ag = AllGather(mesh)
        t0 = time.time()
        # embedding (replicated bf16 table, row-major for ttnn.embedding)
        self.emb = upload(mesh, w.get("model.embed_tokens.weight").to(torch.bfloat16), ttnn.bfloat16,
                          ttnn.ROW_MAJOR_LAYOUT)
        self.layers = []
        for li in self.layer_ids:
            self.layers.append(Hy4Layer(mesh, cfg, w, li, max_ctx, chunk, cache_dir))
            logger.info(f"layer {li} loaded ({time.time() - t0:.0f} s)")
        self.head_hc = head_hc(mesh, w, cfg)
        self.norm = RMSNorm(mesh, w.get("model.norm.weight"), cfg.rms_eps)
        V = cfg.vocab
        self.Vp = -(-V // (32 * self.N)) * (32 * self.N)
        lm = torch.zeros(cfg.hidden, self.Vp, dtype=torch.bfloat16)
        lm[:, :V] = w.get("lm_head.weight").to(torch.bfloat16).T
        lm_dt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("HY4_LM_DTYPE", "bf16")]
        self.lm = upload(mesh, lm.reshape(1, 1, cfg.hidden, self.Vp), lm_dt, mapper=("shard", -1))
        del lm
        self.ckc = ckc(mesh, fp32=True)
        self.rope = DeviceRope(mesh, cfg.qk_rope)
        self.pos = 0  # next position of the sequence
        logger.info(f"Hy4Model ready: {len(self.layers)} layers in {time.time() - t0:.0f} s")

    def reset(self):
        self.pos = 0
        for layer in self.layers:
            layer.reset()
        if self.engram is not None:
            self.engram.reset()

    def _engram_rows(self, ids: torch.Tensor, p0: int, Tp: int) -> dict:
        """{layer: [1,1,Tp,1536] device rows} for this pass (zeros past the real rows)."""
        out = {}
        if self.engram is None:
            return out
        for layer in self.layers:
            if layer.engram is None:
                continue
            rows = self.engram.rows(layer.li, ids, p0)  # [T, 1536] bf16 (host store, continues the sequence)
            full = torch.zeros(Tp, rows.shape[-1], dtype=torch.bfloat16)
            full[: rows.shape[0]] = rows
            out[layer.li] = upload(self.mesh, full.reshape(1, 1, Tp, -1), ttnn.bfloat16)
        return out

    def forward(self, ids: torch.Tensor, logits_rows: str = "last", mem=None) -> torch.Tensor:
        """ids: [T] int (the next T tokens of the sequence). -> logits fp32 [V] (last row) or [T, V] (all rows)."""
        cfg, mesh = self.cfg, self.mesh
        T = int(ids.numel())
        p0 = self.pos
        assert p0 + T <= self.max_ctx, f"context {p0 + T} > max_ctx {self.max_ctx}"
        Tp = -(-T // 32) * 32
        tok = torch.zeros(1, Tp, dtype=torch.int32)
        tok[0, :T] = ids.reshape(-1).to(torch.int32)
        tok_t = upload(mesh, tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.embedding(tok_t, self.emb, layout=ttnn.TILE_LAYOUT, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok_t)
        x = ttnn.reshape(x, (1, 1, Tp, cfg.hidden))
        xs = [x] + [ttnn.clone(x) for _ in range(cfg.hc_mult - 1)]
        pos = torch.arange(p0, p0 + Tp)
        cos_sin = {k: self.rope.tables(pos, cfg.rope_theta[k]) for k in ("sparse", "swa")}
        eng = self._engram_rows(ids.reshape(-1), p0, Tp)
        for layer in self.layers:
            xs = layer.forward(xs, p0, T, cos_sin, eng.get(layer.li), mem)
        for t_ in list(eng.values()) + [c for cs in cos_sin.values() for c in cs]:
            ttnn.deallocate(t_)
        self.pos = p0 + T
        h, _ = self.head_hc.pre(xs, mem)
        for x_ in xs:
            ttnn.deallocate(x_)
        if logits_rows == "last":
            r = ((T - 1) // 32) * 32
            h = ttnn.slice(h, (0, 0, r, 0), (1, 1, r + 32, cfg.hidden))
        h = self.norm(h, mem)
        lg = ttnn.matmul(h, self.lm, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
        out = to_torch_cat(lg, dim=-1).reshape(-1, self.Vp)[:, : cfg.vocab].float()
        ttnn.deallocate(lg)
        ttnn.deallocate(h)
        if logits_rows == "last":
            return out[(T - 1) - ((T - 1) // 32) * 32]
        return out[:T]
