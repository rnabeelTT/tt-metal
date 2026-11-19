# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 gated MLA with learned sinks (HF HYV4FlashAttention), sparse (indexed) and sliding-window kinds.

TP over the (1, N) mesh by heads (sparse: 32 -> 8 per die, SWA: 64 -> 16). Absorbed MLA: one cached latent row per
token, [c_kv (512, normed) | k_rope (64, roped) | sink columns (32)] bf16 ROW-MAJOR, replicated on every die:

    score(h, t) = scale * (q_nope_h . W_uk_h c_t + q_rope_h . k_rope_t) = scale * ([W_uk_h^T q_nope_h | q_rope_h] . row_t)
    out_h       = W_uv_h (sum_t p_ht c_t)                      (V = the leading 512 columns of the row)

`ttnn.transformer.sparse_sdpa` computes softmax over an explicit index list per query row (the sparse layers' indexer
selection, the window of the SWA layers, or every earlier token while dense). The learned sink of head h is one more
key with score s_h and value 0: a reserved cache row whose sink columns hold 16 (with scale 1/16 that is a unit key)
and zeros elsewhere, scored against q's sink columns [s_hi, s_lo] (s_h split into two bf16 parts, so the sink logit
keeps ~16 bits). Every index list starts with that row.

Then: out * sigmoid(gate_proj(x)) -> o_proj (row parallel) -> all-reduce -> post_proj_norm (replicated).

Day-0 execution is eager: the host builds each pass's index lists (positions are host knowledge), the device scores
the compressed indexer and takes its top-512 groups (read back), and the indexer's group keys are pooled on the host
in fp32 from the read-back token keys (bit-for-bit the reference's pooling). See tt/indexer.py.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import (
    AllReduce,
    Boundary,
    RMSNorm,
    StreamLinear,
    TILE,
    ckc,
    mm,
    num_devices,
    stream_enabled,
    wfid,
    upload,
)
from models.autoports.tencent_hy4_flash.tt.rope import DeviceRope

SKF = getattr(ttnn.transformer, "SparseKVFormat", None) or getattr(ttnn, "SparseKVFormat", None)
SENTINEL = 0xFFFFFFFF
K_CHUNK = 128  # sparse_sdpa key chunk; index lists are padded to a multiple of it
SINK_COLS = 32
SINK_KEY = 16.0


def _dt(name, default):
    return {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get(name, default)]


class Hy4Attention:
    def __init__(self, mesh, cfg, sd, layer_idx: int, max_ctx: int, chunk: int, share=None):
        """sd: the layer's `self_attn.` tensors (prefix stripped). max_ctx: cache capacity (sparse layers); chunk: the
        largest rows per forward (prefill chunk), which sizes the SWA ring."""
        self.mesh, self.cfg, self.li = mesh, cfg, layer_idx
        self.kind = cfg.layer_kind[layer_idx]
        self.N = N = num_devices(mesh)
        self.H = H = cfg.heads[self.kind]
        assert H % N == 0
        self.Hd = Hd = H // N
        self.Hp = max(32, -(-Hd // 32) * 32)  # sparse_sdpa wants a multiple of 32 heads
        self.nope, self.rope_d, self.vd, self.kvl = cfg.qk_nope, cfg.qk_rope, cfg.v_head, cfg.kv_lora
        self.width = cfg.latent_width + SINK_COLS  # 608
        self.scale = (cfg.qk_nope + cfg.qk_rope) ** -0.5
        assert abs(self.scale * SINK_KEY - 1.0) < 1e-12, "the sink trick assumes scale = 1/16"
        self.theta = cfg.rope_theta[self.kind]
        self.ar = AllReduce(mesh)
        self.ckc = ckc(mesh, fp32=True)
        # sparse_sdpa (bf16 q / cache): HY4_SDPA_FID (default HiFi4)
        self.ckc_sdpa = ckc(mesh, getattr(ttnn.MathFidelity, os.environ.get("HY4_SDPA_FID", "HiFi4")), fp32=True)
        self.ckc_lofi = ckc(mesh, ttnn.MathFidelity.HiFi2, fp32=True)
        self.rope = DeviceRope(mesh, self.rope_d)
        wdt = _dt("HY4_ATTN_DTYPE", "bfp8")
        sdt = _dt("HY4_ATTN_SMALL_DTYPE", "bf16")
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731

        # q: q_a (replicated) -> q_a_layernorm -> q_b (column parallel by heads)
        self.wqa = upload(mesh, bf(sd["q_a_proj.weight"]).T.reshape(1, 1, cfg.hidden, cfg.q_lora), sdt)
        self.q_norm = RMSNorm(mesh, sd["q_a_layernorm.weight"], cfg.rms_eps)
        self.wqb = upload(mesh, bf(sd["q_b_proj.weight"]).T.reshape(1, 1, cfg.q_lora, H * 256), wdt, mapper=("shard", -1))
        # kv: kv_a (replicated) -> [c (512) | k_rope (64)]
        self.wkva = upload(mesh, bf(sd["kv_a_proj_with_mqa.weight"]).T.reshape(1, 1, cfg.hidden, cfg.latent_width), sdt)
        self.kv_norm = RMSNorm(mesh, sd["kv_a_layernorm.weight"], cfg.rms_eps)
        # absorbed kv_b: W_uk [H, 192, 512], W_uv^T [H, 512, 256] (sharded by heads)
        kvb = bf(sd["kv_b_proj.weight"]).reshape(H, self.nope + self.vd, self.kvl)
        wuk = kvb[:, : self.nope, :].contiguous()  # [H, 192, 512]
        wuvT = kvb[:, self.nope :, :].transpose(1, 2).contiguous()  # [H, 512, 256]
        udt = _dt("HY4_ATTN_UKV_DTYPE", "bf16")  # the absorbed per-head W_uk / W_uv (batched matmuls over the heads)
        self.wuk = upload(mesh, wuk.reshape(1, H, self.nope, self.kvl), udt, mapper=("shard", 1))
        self.wuvT = upload(mesh, wuvT.reshape(1, H, self.kvl, self.vd), udt, mapper=("shard", 1))
        self.ckc_ukv = self.ckc_lofi if udt != ttnn.bfloat16 else self.ckc  # bfp8: HiFi2
        # output gate (column parallel), o_proj (row parallel), post_proj_norm
        self.wg = upload(mesh, bf(sd["linear_gate.weight"]).T.reshape(1, 1, cfg.hidden, H * self.vd), wdt, mapper=("shard", -1))
        self.wo = upload(mesh, bf(sd["o_proj.weight"]).T.reshape(1, 1, H * self.vd, cfg.hidden), wdt, mapper=("shard", -2))
        self.post_norm = RMSNorm(mesh, sd["post_proj_norm.weight"], cfg.rms_eps)
        # sinks: q's sink columns per (local, padded) head: [s_hi, s_lo, 0, ...]
        s = sd["learnable_sink_param"].float().reshape(H)
        # the flash prefill path (first chunk): ttnn SDPA scales the sink with the logits, so it takes sink / scale
        self.sink_sdpa = upload(mesh, (s / self.scale).reshape(1, H, 1, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                mapper=("shard", 1))
        s_hi = s.to(torch.bfloat16).float()
        s_lo = (s - s_hi).to(torch.bfloat16).float()
        qs = torch.zeros(N, self.Hp, SINK_COLS)
        for d in range(N):
            qs[d, :Hd, 0] = s_hi[d * Hd : (d + 1) * Hd]
            qs[d, :Hd, 1] = s_lo[d * Hd : (d + 1) * Hd]
        self.q_sink = upload(mesh, qs[:, :Hd].reshape(N, Hd, 1, SINK_COLS), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                             mapper=("shard", 0))
        # decode passes: the sink columns repeated over the 32 rows once (constant; a lazy copy made inside a trace
        # capture would hold nothing until that trace's first replay)
        self._qs32 = ttnn.repeat(self.q_sink, ttnn.Shape((1, 1, TILE, 1)), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.stage = None  # the decode staging window (init_stage)
        self.indexer = None
        self.dec = None
        if stream_enabled(mesh):  # (the DFly2 drafter's layers too: q_a | kv_a, gate, q_b, o_proj)
            self._decode_linears(sd, wdt, layer_idx, share is not None)
        if share is not None:  # the DFly2 drafter: KV share with a target layer (its cache, sink and scratch rows)
            self.cache, self.rows = share.cache, share.rows
            self.sink_row, self.scratch_row = share.sink_row, share.scratch_row
            return
        # caches (row-major bf16, replicated): sparse layers hold every position + a sink block at the end; SWA layers
        # a ring of `ring` rows + the sink block
        if self.kind == "sparse":
            self.rows = -(-max_ctx // 32) * 32 + 64  # + the decode staging window's lookahead (two 32-row slabs)
        else:
            self.rows = -(-(chunk + cfg.window + 32) // 32) * 32
        self.sink_row = self.rows
        self.scratch_row = self.rows + 32  # 32 scratch rows: the DFly2 drafter's block latents (never target rows)
        cache = torch.zeros(1, 1, self.rows + 64, self.width, dtype=torch.bfloat16)
        cache[0, 0, self.sink_row, cfg.latent_width] = SINK_KEY
        cache[0, 0, self.sink_row, cfg.latent_width + 1] = SINK_KEY
        self.cache = upload(mesh, cache, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        del cache
        self.indexer = None
        if self.kind == "sparse":
            from models.autoports.tencent_hy4_flash.tt.indexer import Hy4Indexer

            self.indexer = Hy4Indexer(mesh, cfg, sd, max_ctx)

    def _decode_linears(self, sd, wdt, li, drafter):
        """Decode-only streamed copies (tt/common.StreamLinear): x -> [q_a | kv_a | (indexer token key | head weights)]
        in one bf16 linear, the output gate, q_b and o_proj (per-die shards, wdt)."""
        cfg, N, Hd = self.cfg, self.N, self.Hd
        import os as _os

        cdir = _os.environ.get("HY4_TENSOR_CACHE")
        tag = f"dfly{li}" if drafter else f"layers.{li}"
        cache = (lambda n: _os.path.join(cdir, f"{tag}.attn.{n}")) if cdir else (lambda n: None)
        T_ = lambda k: sd[k].to(torch.bfloat16).T.contiguous()  # noqa: E731  [in, out]
        xa = [T_("q_a_proj.weight"), T_("kv_a_proj_with_mqa.weight")]
        if self.kind == "sparse" and not drafter:
            p = "compressed_indexer."
            hw = T_(p + "head_weight_projection.weight")
            xa += [T_(p + "token_key_projection.weight"), torch.nn.functional.pad(hw, (0, TILE - hw.shape[1]))]
        xa = torch.cat(xa, dim=1)
        q, l, d = cfg.q_lora, cfg.latent_width, cfg.idx_dim
        self.xa_cols = [(0, q), (q, q + l)] + ([(q + l, q + l + d), (q + l + d, q + l + d + cfg.idx_heads)]
                                              if xa.shape[1] > q + l else [])
        wg, wqb, wo = T_("linear_gate.weight"), T_("q_b_proj.weight"), T_("o_proj.weight")
        w = Hd * 256
        self.dec = {
            "xa": StreamLinear(self.mesh, [xa] * N, ttnn.bfloat16, cache=cache("xa.bf16")),
            "g": StreamLinear(self.mesh, [wg[:, i * w : (i + 1) * w] for i in range(N)], wdt, cache=cache(f"g.{wdt}")),
            "qb": StreamLinear(self.mesh, [wqb[:, i * w : (i + 1) * w] for i in range(N)], wdt,
                               cache=cache(f"qb.{wdt}")),
            "o": StreamLinear(self.mesh, [wo[i * w : (i + 1) * w, :] for i in range(N)], wdt, cache=cache(f"o.{wdt}")),
        }
        if self.kind == "sparse" and not drafter:
            wq = T_("compressed_indexer.query_projection.weight")
            self.dec["iq"] = StreamLinear(self.mesh, [wq] * N, ttnn.bfloat16, cache=cache("iq.bf16"))

    def _xa(self, x, mem):
        """Decode: the fused x-projections -> [q_a, kv_a, (token key, head weights)] column slices."""
        y = self.dec["xa"](x, mem)
        outs = [ttnn.slice(y, (0, 0, 0, a), (1, 1, 32, b), memory_config=mem) for a, b in self.xa_cols]
        ttnn.deallocate(y)
        return outs

    # ------------------------------------------------------------------ host-side index lists
    def _cache_row(self, pos: np.ndarray) -> np.ndarray:
        return pos if self.kind == "sparse" else pos % self.rows

    def index_lists(self, p0: int, T: int, selected: np.ndarray | None, Tp: int | None = None) -> torch.Tensor:
        """[1, 1, Tp, K] uint32 per query row p = p0 + s: [sink | keys... | sentinels]. keys: every position <= p
        (dense: sparse layers with p < 2048), the window (p - 511 .. p) for SWA, or the indexer's 511 groups x 4 +
        the query's own group up to p (selected[s] = the row's group ids, sorted by score)."""
        cfg = self.cfg
        Tp = Tp or -(-T // 32) * 32
        key = None
        if selected is None or self.kind == "swa":  # position-only lists: shared by every layer of the kind
            key = (self.kind, p0, T, Tp, self.rows, self.sink_row)
            hit = Hy4Attention._IDX_CACHE.get(key)
            if hit is not None:
                return hit
        out = self._index_lists(p0, T, selected, Tp)
        if key is not None:
            if len(Hy4Attention._IDX_CACHE) > 8:
                Hy4Attention._IDX_CACHE.clear()
            Hy4Attention._IDX_CACHE[key] = out
        return out

    _IDX_CACHE = {}

    def _index_lists(self, p0, T, selected, Tp):
        cfg = self.cfg
        if self.kind == "swa":
            K = -(-(cfg.window + 1) // K_CHUNK) * K_CHUNK
        else:
            K = -(-(cfg.idx_candidate_topk + 1) // K_CHUNK) * K_CHUNK
        out = np.full((Tp, K), SENTINEL, dtype=np.uint32)
        out[:, 0] = self.sink_row
        for s in range(T):
            p = p0 + s
            if self.kind == "swa":
                keys = np.arange(max(0, p - cfg.window + 1), p + 1)
            elif selected is None or p < cfg.idx_candidate_topk:
                keys = np.arange(0, p + 1)
            else:
                g = selected[s][: cfg.idx_group_topk - 1].astype(np.int64)
                own = (p // cfg.idx_ratio) * cfg.idx_ratio
                keys = np.concatenate(
                    [(g[:, None] * cfg.idx_ratio + np.arange(cfg.idx_ratio)[None, :]).reshape(-1), np.arange(own, p + 1)]
                )
            out[s, 1 : 1 + keys.size] = self._cache_row(keys)
        # padding rows (s >= T) attend to the sink only: a defined (zero) output
        return torch.from_numpy(out.view(np.int32)).reshape(1, 1, Tp, K)

    # ------------------------------------------------------------------ shared pieces
    _ZEROS = {}

    def _zeros(self, shape):
        """A persistent zero tensor of this shape (shared by the layers; created outside any trace capture: the decode
        path's first pass of each shape is eager)."""
        key = (id(self.mesh), tuple(shape))
        z = Hy4Attention._ZEROS.get(key)
        if z is None:
            z = Hy4Attention._ZEROS[key] = ttnn.zeros(tuple(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                       device=self.mesh)
        return z
    def _q(self, x, cos, sin, mem, qa=None):
        """-> (q_lat [1,Hd,Tp,512], q_rope [1,Hd,Tp,64] roped, q_resid [1,1,Tp,1024]). qa given: the decode path's
        fused projection (and q_b streamed)."""
        Tp = int(x.shape[-2])
        dec = qa is not None
        if qa is None:
            qa = mm(x, self.wqa, wfid(self.ckc, self.wqa), mem)
        q_resid = self.q_norm(qa, mem)
        ttnn.deallocate(qa)
        if dec:
            q = self.dec["qb"](q_resid, mem)
        else:
            q = mm(q_resid, self.wqb, wfid(self.ckc, self.wqb), mem)  # [1,1,Tp,Hd*256]
        if Tp == TILE and os.environ.get("HY4_HEAD_VIEWS", "1") == "1":
            # one tile row: [1, 1, 32, Hd * 256] and [1, Hd, 32, 256] share their tile pages (head h, tile c -> page
            # 8 h + c): a zero-copy view instead of reshape + permute (~21 us)
            q = ttnn.experimental.view(q, (1, self.Hd, Tp, 256))
        elif os.environ.get("HY4_PF_NLP_HEADS", "1") == "1":
            # [1, 1, Tp, Hd * 256] -> [1, Hd, Tp, 256] (the reshape + permute went through a 32-padded head dim: ~1 ms at
            # 16 heads x 2K rows vs 84 us)
            q2 = ttnn.experimental.nlp_create_qkv_heads(q, num_heads=self.Hd, num_kv_heads=0, transpose_k_heads=False,
                                                        memory_config=mem)[0]
            ttnn.deallocate(q)
            q = q2
        else:
            q = ttnn.permute(ttnn.reshape(q, (1, Tp, self.Hd, 256)), (0, 2, 1, 3), memory_config=mem)  # [1,Hd,Tp,256]
        q_nope = ttnn.slice(q, (0, 0, 0, 0), (1, self.Hd, Tp, self.nope), memory_config=mem)
        q_rope = ttnn.slice(q, (0, 0, 0, self.nope), (1, self.Hd, Tp, 256), memory_config=mem)
        ttnn.deallocate(q)
        q_lat = ttnn.matmul(q_nope, self.wuk, compute_kernel_config=self.ckc_ukv, memory_config=mem)  # [1,Hd,Tp,512]
        ttnn.deallocate(q_nope)
        q_rope = self.rope(q_rope, cos, sin, mem)
        return q_lat, q_rope, q_resid

    def _kv_row(self, x, cos, sin, mem, kv=None):
        """-> the cache rows of these tokens [1,1,Tp,608] TILE: [norm(c) | rope(k_rope) | 0]."""
        Tp = int(x.shape[-2])
        if kv is None:
            kv = mm(x, self.wkva, wfid(self.ckc, self.wkva), mem)  # [1,1,Tp,576]
        c = self.kv_norm(ttnn.slice(kv, (0, 0, 0, 0), (1, 1, Tp, self.kvl), memory_config=mem), mem)
        kr = self.rope(ttnn.slice(kv, (0, 0, 0, self.kvl), (1, 1, Tp, self.cfg.latent_width), memory_config=mem),
                       cos, sin, mem)
        ttnn.deallocate(kv)
        row = ttnn.concat([c, kr, self._zeros((1, 1, Tp, SINK_COLS))], dim=-1, memory_config=mem)
        ttnn.deallocate(c)
        ttnn.deallocate(kr)
        return row

    def _attend(self, x, q_lat, q_rope, idx, mem, dec=False, R=None):
        """sparse_sdpa over idx [1,1,Tp,K] uint32, then V-up, output gate, o_proj, all-reduce, post norm."""
        Tp = int(x.shape[-2])
        if dec and os.environ.get("HY4_QSINK_CACHE", "1") == "1":
            qs = self._qs32
            qcat = ttnn.concat([q_lat, q_rope, qs], dim=-1, memory_config=mem)  # [1,Hd,Tp,608]
        else:
            qs = ttnn.repeat(self.q_sink, ttnn.Shape((1, 1, Tp, 1)), memory_config=mem)  # [1,Hd,Tp,32]
            qcat = ttnn.concat([q_lat, q_rope, qs], dim=-1, memory_config=mem)  # [1,Hd,Tp,608]
            ttnn.deallocate(qs)
        if self.Hp != self.Hd:
            q2 = ttnn.concat([qcat, self._zeros((1, self.Hp - self.Hd, Tp, self.width))], dim=1, memory_config=mem)
            ttnn.deallocate(qcat)
            qcat = q2
        q_rm = ttnn.to_layout(qcat, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # sparse_sdpa: DRAM
        ttnn.deallocate(qcat)
        o = ttnn.transformer.sparse_sdpa(q_rm, self.cache, idx, self.kvl, kv_format=SKF.BF16, scale=self.scale,
                                         k_chunk_size=K_CHUNK, compute_kernel_config=self.ckc_sdpa)  # [1,Hp,Tp,512] RM
        ttnn.deallocate(q_rm)
        o2 = ttnn.slice(o, (0, 0, 0, 0), (1, self.Hd, Tp, self.kvl), memory_config=mem)
        ttnn.deallocate(o)
        o = ttnn.to_layout(o2, ttnn.TILE_LAYOUT, memory_config=mem)
        ttnn.deallocate(o2)
        ov = ttnn.matmul(o, self.wuvT, compute_kernel_config=self.ckc_ukv, memory_config=mem)  # [1,Hd,Tp,256]
        ttnn.deallocate(o)
        if Tp == TILE and os.environ.get("HY4_HEAD_VIEWS", "1") == "1":
            ov = ttnn.experimental.view(ov, (1, 1, Tp, self.Hd * self.vd))  # (the same page identity as in _q)
        else:
            ov = ttnn.reshape(ttnn.permute(ov, (0, 2, 1, 3), memory_config=mem), (1, 1, Tp, self.Hd * self.vd))
        if dec:
            g = self.dec["g"](x, mem)
        else:
            g = mm(x, self.wg, wfid(self.ckc, self.wg), mem)
        gated = ttnn.multiply(ov, ttnn.sigmoid(g, memory_config=mem), memory_config=mem)
        ttnn.deallocate(ov)
        ttnn.deallocate(g)
        if dec:
            part = self.dec["o"](gated, mem)
        else:
            part = mm(gated, self.wo, wfid(self.ckc, self.wo), mem)
        ttnn.deallocate(gated)
        if dec and R is not None and Boundary.enabled(self.mesh, R):
            if getattr(self, "bnd", None) is None:
                self.bnd = Boundary(self.mesh, self.cfg.hidden, self.cfg.rms_eps)
            out = self.bnd(part, R, self.post_norm, "mixer")
            ttnn.deallocate(part)
            return out
        full = self.ar(part, mem)
        ttnn.deallocate(part)
        return self.post_norm(full, mem)

    # ------------------------------------------------------------------ eager (prefill)
    def forward_part(self, x, p0: int, T: int, cos_sin, mem=None):
        """forward() without its all-reduce and post norm: the per-die o_proj partial [1, 1, Tp, 3072] (the row-split
        prefill reduce-scatters it over rows and norms each die's rows)."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        cos, sin = cos_sin[self.kind]
        q_lat, q_rope, q_resid = self._q(x, cos, sin, mem)
        row = self._kv_row(x, cos, sin, mem)
        row_rm = ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
        Tp = int(x.shape[-2])
        # the first chunk attends causally to its own rows only (dense regime / SWA window): flash attention over the
        # chunk's rows (keys read once per query block) instead of sparse_sdpa's per-row key gathers (HY4_PF_FLASH)
        flash = (p0 == 0 and os.environ.get("HY4_PF_FLASH", "1") == "1"
                 and (self.kind == "swa" or Tp <= self.cfg.idx_candidate_topk))
        kv_t = row if flash else None
        if not flash:
            ttnn.deallocate(row)
        self._write_rows(row_rm, p0, self._write_n(row_rm, T))
        ttnn.deallocate(row_rm)
        selected = None
        if self.indexer is not None:
            tail = self._idx_tail(p0, T, Tp) if p0 >= self.cfg.idx_candidate_topk else None
            selected = self.indexer.forward(x, q_resid, p0, T, mem, tail=tail)
        ttnn.deallocate(q_resid)
        if flash:
            o = self._flash_first_chunk(q_lat, q_rope, kv_t, mem)
            ttnn.deallocate(kv_t)
            return self._out_part(x, o, mem)
        if isinstance(selected, tuple):  # the device-built lists of a selecting chunk
            idx, shared = selected[1], False
        else:
            idx, shared = self._idx_device(p0, T, selected, Tp)
        qs = ttnn.repeat(self.q_sink, ttnn.Shape((1, 1, Tp, 1)), memory_config=mem)
        qcat = ttnn.concat([q_lat, q_rope, qs], dim=-1, memory_config=mem)
        for t_ in (qs, q_lat, q_rope):
            ttnn.deallocate(t_)
        if self.Hp != self.Hd:
            q2 = ttnn.concat([qcat, self._zeros((1, self.Hp - self.Hd, Tp, self.width))], dim=1, memory_config=mem)
            ttnn.deallocate(qcat)
            qcat = q2
        q_rm = ttnn.to_layout(qcat, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qcat)
        o = ttnn.transformer.sparse_sdpa(q_rm, self.cache, idx, self.kvl, kv_format=SKF.BF16, scale=self.scale,
                                         k_chunk_size=K_CHUNK, compute_kernel_config=self.ckc_sdpa)
        ttnn.deallocate(q_rm)
        if not shared:
            ttnn.deallocate(idx)
        o2 = ttnn.slice(o, (0, 0, 0, 0), (1, self.Hd, Tp, self.kvl), memory_config=mem)
        ttnn.deallocate(o)
        o = ttnn.to_layout(o2, ttnn.TILE_LAYOUT, memory_config=mem)
        ttnn.deallocate(o2)
        return self._out_part(x, o, mem)

    def _out_part(self, x, o, mem):
        """Prefill: the attention output o [1, Hd, Tp, 512] TILE (consumed) -> V-up, output gate, o_proj partial."""
        Tp = int(x.shape[-2])
        ov = ttnn.matmul(o, self.wuvT, compute_kernel_config=self.ckc_ukv, memory_config=mem)
        ttnn.deallocate(o)
        if os.environ.get("HY4_PF_NLP_HEADS", "1") == "1":
            ov2 = ttnn.experimental.nlp_concat_heads(ov, memory_config=mem)  # [1, 1, Tp, Hd * 256]
        else:
            ov2 = ttnn.reshape(ttnn.permute(ov, (0, 2, 1, 3), memory_config=mem), (1, 1, Tp, self.Hd * self.vd))
        ttnn.deallocate(ov)
        ov = ov2
        g = mm(x, self.wg, wfid(self.ckc, self.wg), mem)
        gated = ttnn.multiply(ov, ttnn.sigmoid(g, memory_config=mem), memory_config=mem)
        ttnn.deallocate(ov)
        ttnn.deallocate(g)
        part = mm(gated, self.wo, wfid(self.ckc, self.wo), mem)
        ttnn.deallocate(gated)
        return part

    def _flash_first_chunk(self, q_lat, q_rope, row, mem):
        """Causal (SWA: windowed) MQA flash attention of a first prefill chunk over its own rows, with the sinks:
        q [1, Hd, Tp, 576] = [q_lat | roped q], K = V = row[..., :576] -> o [1, Hd, Tp, 512] TILE (V's first 512).
        (q_lat / q_rope consumed.)"""
        Tp = int(row.shape[-2])
        q = ttnn.concat([q_lat, q_rope], dim=-1, memory_config=mem)
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rope)
        lw = self.cfg.latent_width
        k = ttnn.slice(row, (0, 0, 0, 0), (1, 1, Tp, lw), memory_config=mem)
        v = k  # (ttnn SDPA wants V's head dim = Q's: V = the whole 576-wide row, output sliced to its 512 latents)
        qc = int(os.environ.get("HY4_PF_FLASH_Q", "128"))  # q 128 x k 128 overflowed L1 at head dim 576
        kc = int(os.environ.get("HY4_PF_FLASH_K", "64"))
        while Tp % qc:
            qc //= 2
        while Tp % kc:
            kc //= 2
        grid = self.mesh.compute_with_storage_grid_size()
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y), exp_approx_mode=False,
                                    q_chunk_size=max(qc, 32), k_chunk_size=max(kc, 32))
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scale,
            sliding_window_size=self.cfg.window + int(os.environ.get("HY4_PF_FLASH_WIN_DELTA", "0")) if self.kind == "swa" else None,
            attention_sink=self.sink_sdpa, program_config=pc, compute_kernel_config=self.ckc_sdpa, memory_config=mem)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        o2 = ttnn.slice(o, (0, 0, 0, 0), (1, self.Hd, Tp, self.kvl), memory_config=mem)
        ttnn.deallocate(o)
        return o2

    _IDX_DEV = {}

    def _idx_tail(self, p0, T, Tp):
        """[1, 1, Tp, K - 2044] uint32: per row [sink, own group's positions up to p, sentinels] (device, shared by the
        sparse layers of the chunk)."""
        cfg = self.cfg
        key = (id(self.mesh), "tail", p0, T, Tp, self.sink_row)
        t = Hy4Attention._IDX_DEV.get(key)
        if t is None:
            for k in [k for k in Hy4Attention._IDX_DEV if k[1] == "tail"]:
                ttnn.deallocate(Hy4Attention._IDX_DEV.pop(k))
            K = -(-(cfg.idx_candidate_topk + 1) // K_CHUNK) * K_CHUNK
            n_sel = (cfg.idx_group_topk - 1) * cfg.idx_ratio
            tail = np.full((Tp, K - n_sel), SENTINEL, dtype=np.uint32)
            tail[:, 0] = self.sink_row
            for s_ in range(Tp):
                p = p0 + min(s_, T - 1)
                own = np.arange((p // cfg.idx_ratio) * cfg.idx_ratio, p + 1)
                tail[s_, 1 : 1 + own.size] = own
            t = Hy4Attention._IDX_DEV[key] = upload(self.mesh, torch.from_numpy(tail.view(np.int32)).reshape(1, 1, Tp, -1),
                                                    ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        return t

    def _idx_device(self, p0, T, selected, Tp):
        """The chunk's index lists on the device. Position-only lists (SWA, dense) are uploaded once per chunk and shared
        by every layer of the kind (-> (tensor, True)); selection lists are per layer (-> (tensor, False))."""
        if selected is not None and self.kind != "swa":
            return upload(self.mesh, self.index_lists(p0, T, selected, Tp), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), False
        key = (id(self.mesh), self.kind, p0, T, Tp, self.rows, self.sink_row)
        t = Hy4Attention._IDX_DEV.get(key)
        if t is None:
            stale = [k for k in Hy4Attention._IDX_DEV if k[1] == self.kind and k[2] != 0]  # p0 = 0: traced chunks
            for k in stale:  # one chunk at a time per kind
                ttnn.deallocate(Hy4Attention._IDX_DEV.pop(k))
            t = Hy4Attention._IDX_DEV[key] = upload(self.mesh, self.index_lists(p0, T, None, Tp), ttnn.uint32,
                                                    ttnn.ROW_MAJOR_LAYOUT)
        return t, True

    def forward(self, x, p0: int, T: int, cos_sin, mem=None):
        """x: [1, 1, Tp, 3072] bf16 replicated (normed attention input), rows p0 .. p0 + T - 1 real (Tp = T padded to
        32). cos_sin: {kind: (cos, sin)} fp32 [1, 1, Tp, 64] device tables of these positions (shared by the layers).
        Returns [1, 1, Tp, 3072] bf16 replicated."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        cos, sin = cos_sin[self.kind]
        q_lat, q_rope, q_resid = self._q(x, cos, sin, mem)
        row = self._kv_row(x, cos, sin, mem)
        row_rm = ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
        ttnn.deallocate(row)
        self._write_rows(row_rm, p0, self._write_n(row_rm, T))
        ttnn.deallocate(row_rm)
        selected = self.indexer.forward(x, q_resid, p0, T, mem) if self.indexer is not None else None
        ttnn.deallocate(q_resid)
        idx = upload(self.mesh, self.index_lists(p0, T, selected), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        out = self._attend(x, q_lat, q_rope, idx, mem)
        for t_ in (q_lat, q_rope, idx):
            ttnn.deallocate(t_)
        return out

    # ------------------------------------------------------------------ trace-safe step (decode / verify rows)
    def init_stage(self, base: int):
        """After a prefill: the 64-row staging window starting at position `base` (32-aligned) from the cache."""
        r0 = base if self.kind == "sparse" else base % self.rows
        if os.environ.get("HY4_STAGE_GATHER", "1") == "1":
            # the 64 window rows gathered by a runtime index tensor (ttnn.embedding over the row-major cache): a slice
            # at the prompt's position bakes the offset into its program (one new program per 32-token prompt
            # position, compiled at serve time, re-capturing every decode trace)
            pos = torch.arange(base, base + 64)
            rows = (pos if self.kind == "sparse" else pos % self.rows).to(torch.int32).reshape(1, 64)
            idx = upload(self.mesh, rows, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            table = ttnn.reshape(self.cache, (int(self.cache.shape[-2]), self.width))
            g = ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT)
            ttnn.deallocate(idx)
            wt = ttnn.reshape(g, (1, 1, 64, self.width))
        else:
            if r0 + 64 <= self.rows:
                w = ttnn.slice(self.cache, (0, 0, r0, 0), (1, 1, r0 + 64, self.width))
            else:  # the SWA ring wraps inside the window
                a = ttnn.slice(self.cache, (0, 0, r0, 0), (1, 1, self.rows, self.width))
                b = ttnn.slice(self.cache, (0, 0, 0, 0), (1, 1, r0 + 64 - self.rows, self.width))
                w = ttnn.concat([a, b], dim=2)
                ttnn.deallocate(a)
                ttnn.deallocate(b)
            wt = ttnn.to_layout(w, ttnn.TILE_LAYOUT)
            ttnn.deallocate(w)
        if self.stage is None:
            self.stage = wt
        else:
            ttnn.copy(wt, self.stage)
            ttnn.deallocate(wt)

    def step(self, x, si, mem=None):
        """Trace-safe rows pass. si: the step inputs (tt/decode.StepInputs): cos/sin, staging maps, slab offsets,
        index lists. The KV rows go through the staging window and two 32-row slab writes."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        cos, sin = si.cos_sin[self.kind]
        dec = self.dec is not None and int(x.shape[-2]) == 32
        parts = self._xa(x, mem) if dec else [None, None]
        q_lat, q_rope, q_resid = self._q(x, cos, sin, mem, qa=parts[0])
        row = self._kv_row(x, cos, sin, mem, kv=parts[1])
        a = ttnn.matmul(si.A, self.stage, compute_kernel_config=self.ckc, memory_config=mem)
        b = ttnn.matmul(si.P, row, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(row)
        new = ttnn.add(a, b, memory_config=mem)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.copy(new, self.stage)
        ttnn.deallocate(new)
        st_rm = ttnn.to_layout(self.stage, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        offs = si.off_sparse if self.kind == "sparse" else si.off_swa
        for j in range(2):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                self.cache, st_rm, si.slot, offs[j], 0, 1, 0, input_row_offset=32 * j, input_rows=32)
        ttnn.deallocate(st_rm)
        if self.kind == "swa":
            idx = si.idx_swa
        elif self.indexer is not None:
            # the selection, or si.idx_dense in the dense regime
            idx = self.indexer.step(x, q_resid, si, mem, tk=parts[2] if dec else None, hw=parts[3] if dec else None,
                                    iq=self.dec["iq"] if dec else None)
        else:
            idx = si.idx_dense
        ttnn.deallocate(q_resid)
        out = self._attend(x, q_lat, q_rope, idx, mem, dec=dec, R=getattr(si, "R", None))
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rope)
        if idx is not si.idx_swa and idx is not si.idx_dense:
            ttnn.deallocate(idx)
        return out

    @staticmethod
    def _write_n(row_rm, T):
        """Rows a prefill chunk writes: all of its padded rows (HY4_PF_WRITE_PAD=1, default): their KV sits past the
        prompt and decode / the next chunk rewrites it, and the slice / slice_write programs then depend on the bucket
        only (a per-length program would be compiled at serve time and re-capture every decode trace)."""
        return int(row_rm.shape[-2]) if os.environ.get("HY4_PF_WRITE_PAD", "1") == "1" else T

    def _write_rows(self, row_rm, p0: int, T: int):
        """Write rows [0, T) of row_rm [1,1,Tp,608] RM to the cache at positions p0.. (ring rows for SWA)."""
        start = p0 if self.kind == "sparse" else p0 % self.rows
        first = min(T, self.rows - start)
        segs = [(0, start, first)]
        if first < T:  # SWA ring wrap
            segs.append((first, 0, T - first))
        for src, dst, n in segs:
            part = row_rm if (src == 0 and n == int(row_rm.shape[-2])) else ttnn.slice(
                row_rm, (0, 0, src, 0), (1, 1, src + n, self.width))
            ttnn.experimental.slice_write(part, self.cache, [0, 0, dst, 0], [1, 1, dst + n, self.width], [1, 1, 1, 1])
            if part is not row_rm:
                ttnn.deallocate(part)
