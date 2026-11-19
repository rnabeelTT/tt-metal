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
    StreamLinear,
    ckc,
    num_devices,
    stream_enabled,
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
        self.engram = Hy4EngramDevice(mesh, cfg, sd.sub("engram_layer."), li) if li in cfg.engram_layers else None

    def forward(self, X, p0, T, cos_sin, engram_emb=None, mem=None, engram_keep=None):
        """X: the residual streams [1, hc, Tp, H] bf16 (replicated). Returns the updated streams."""
        if self.engram is not None and engram_emb is not None:
            X2 = self.engram.forward(X, engram_emb, mem, engram_keep)
            ttnn.deallocate(X)
            X = X2
        a_in, post = self.attn_hc.pre(X, mem)
        a = self.attn.forward(self.in_norm(a_in, mem), p0, T, cos_sin, mem)
        ttnn.deallocate(a_in)
        X2 = self.attn_hc.post(X, a, post, mem)
        for t_ in (X, a, post):
            ttnn.deallocate(t_)
        f_in, post = self.ffn_hc.pre(X2, mem)
        f_n = self.post_norm(f_in, mem)
        ttnn.deallocate(f_in)
        f = self.mlp.forward(f_n, T, mem) if isinstance(self.mlp, Hy4MoE) else self.mlp.forward(f_n, mem)
        ttnn.deallocate(f_n)
        X3 = self.ffn_hc.post(X2, f, post, mem)
        for t_ in (X2, f, post):
            ttnn.deallocate(t_)
        return X3

    def forward_split(self, X, p0, T, cos_sin, engram_emb=None, mem=None, engram_keep=None):
        """Row-split prefill (tt-metal PR 61548 row 22): X [1, hc, Tp/N, H] holds THIS die's quarter of the rows; the
        hyper-connections, norms and Engram run on it; the sublayer inputs are all-gathered over rows, the sublayer
        partials reduce-scattered over rows (the same bytes as the all-reduce they replace)."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        ag = lambda t: ttnn.all_gather(t, dim=2, cluster_axis=1, topology=ttnn.Topology.Linear, memory_config=mem)  # noqa
        rs = lambda t: ttnn.reduce_scatter(t, 2, cluster_axis=1, topology=ttnn.Topology.Linear, memory_config=mem)  # noqa
        if self.engram is not None and engram_emb is not None:
            X2 = self.engram.forward(X, engram_emb, mem, engram_keep)
            ttnn.deallocate(X)
            X = X2
        a_in, post = self.attn_hc.pre(X, mem)
        a_n = self.in_norm(a_in, mem)
        ttnn.deallocate(a_in)
        a_full = ag(a_n)
        ttnn.deallocate(a_n)
        part = self.attn.forward_part(a_full, p0, T, cos_sin, mem)
        ttnn.deallocate(a_full)
        a_loc = rs(part)
        ttnn.deallocate(part)
        a = self.attn.post_norm(a_loc, mem)
        ttnn.deallocate(a_loc)
        X2 = self.attn_hc.post(X, a, post, mem)
        for t_ in (X, a, post):
            ttnn.deallocate(t_)
        f_in, post = self.ffn_hc.pre(X2, mem)
        f_n = self.post_norm(f_in, mem)
        ttnn.deallocate(f_in)
        f_full = ag(f_n)
        ttnn.deallocate(f_n)
        part = self.mlp.forward_part(f_full, T, mem)
        ttnn.deallocate(f_full)
        f_loc = rs(part)
        ttnn.deallocate(part)
        f = self.mlp.norm(f_loc, mem)
        ttnn.deallocate(f_loc)
        X3 = self.ffn_hc.post(X2, f, post, mem)
        for t_ in (X2, f, post):
            ttnn.deallocate(t_)
        return X3

    def step(self, X, si, R, mem=None):
        """Trace-safe rows pass (tt/decode.py): every position-dependent input in si. HY4_L1_PARTS (default all):
        which parts keep their intermediates in `mem` (the rest DRAM)."""
        import os as _os

        parts = _os.environ.get("HY4_L1_PARTS", "hc,attn,moe,norm").split(",")  # engram in L1 drifts (PCC 0.993): DRAM
        m = lambda k: mem if k in parts else ttnn.DRAM_MEMORY_CONFIG  # noqa: E731
        if self.engram is not None and self.li in si.engram:
            X2 = self.engram.forward(X, si.engram[self.li], m("engram"))
            ttnn.deallocate(X)
            X = X2
        a_in, post = self.attn_hc.pre(X, m("hc"))
        a = self.attn.step(self.in_norm(a_in, m("norm")), si, m("attn"))
        ttnn.deallocate(a_in)
        X2 = self.attn_hc.post(X, a, post, m("hc"))
        for t_ in (X, a, post):
            ttnn.deallocate(t_)
        f_in, post = self.ffn_hc.pre(X2, m("hc"))
        f_n = self.post_norm(f_in, m("norm"))
        ttnn.deallocate(f_in)
        f = self.mlp.forward_decode(f_n, R, m("moe")) if isinstance(self.mlp, Hy4MoE) else self.mlp.forward(f_n, m("moe"), R=R)
        ttnn.deallocate(f_n)
        X3 = self.ffn_hc.post(X2, f, post, m("hc"))
        for t_ in (X2, f, post):
            ttnn.deallocate(t_)
        return X3

    def step_fused(self, X, pend, si, R, hcf, mem=None):
        """step() with the hyper-connections as fused programs (tt/hc_fused): pend = (o, w) of the previous sublayer's
        HC post (None at layer 0); returns (X, pend) with this layer's MLP output pending."""
        import os as _os

        parts = _os.environ.get("HY4_L1_PARTS", "hc,attn,moe,norm").split(",")
        m = lambda k: mem if k in parts else ttnn.DRAM_MEMORY_CONFIG  # noqa: E731

        def run(X, pend, hc):
            o, wp = pend if pend is not None else (None, None)
            Xn, inp, w = hcf(X, o, wp, hc)
            if pend is not None:
                ttnn.deallocate(X)
                ttnn.deallocate(o)
                ttnn.deallocate(wp)
            return Xn, inp, w

        if self.engram is not None and self.li in si.engram:
            if pend is not None:
                X, _, _ = run(X, pend, None)
                pend = None
            X2 = self.engram.forward(X, si.engram[self.li], m("engram"))
            ttnn.deallocate(X)
            X = X2
        X, a_in, w_a = run(X, pend, self.attn_hc)
        a = self.attn.step(self.in_norm(a_in, m("norm")), si, m("attn"))
        ttnn.deallocate(a_in)
        X, f_in, w_f = run(X, (a, w_a), self.ffn_hc)
        f_n = self.post_norm(f_in, m("norm"))
        ttnn.deallocate(f_in)
        f = self.mlp.forward_decode(f_n, R, m("moe")) if isinstance(self.mlp, Hy4MoE) else self.mlp.forward(f_n, m("moe"), R=R)
        ttnn.deallocate(f_n)
        return X, (f, w_f)

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
        if cache_dir:  # the attention / dense modules' decode copies cache next to the other tensors
            os.environ.setdefault("HY4_TENSOR_CACHE", cache_dir)
        self.ag = AllGather(mesh)
        t0 = time.time()
        # embedding (replicated bf16 table, row-major for ttnn.embedding; tensor-cached: a plain upload of the 742 MB
        # table took minutes)
        self.emb = upload(mesh, w.get("model.embed_tokens.weight").to(torch.bfloat16), ttnn.bfloat16,
                          ttnn.ROW_MAJOR_LAYOUT, cache=os.path.join(cache_dir, "embed_tokens.rm") if cache_dir else None)
        logger.info(f"embedding loaded ({time.time() - t0:.0f} s)")
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
        self.lm = upload(mesh, lm.reshape(1, 1, cfg.hidden, self.Vp), lm_dt, mapper=("shard", -1),
                         cache=os.path.join(cache_dir, f"lm_head.{lm_dt}") if cache_dir else None)
        self.lm_dec = None
        if stream_enabled(mesh):  # decode LM head: streamed copy, fp32 logits (bit-identical to the interleaved matmul)
            n = self.Vp // self.N
            self.lm_dec = StreamLinear(mesh, [lm[:, i * n : (i + 1) * n] for i in range(self.N)], lm_dt,
                                       cache=os.path.join(cache_dir, f"lm_head.{lm_dt}") if cache_dir else None,
                                       out_dtype=ttnn.float32)
        del lm
        self.ckc = ckc(mesh, fp32=True)
        self.rope = DeviceRope(mesh, cfg.qk_rope)
        self.pos = 0  # next position of the sequence
        logger.info(f"Hy4Model ready: {len(self.layers)} layers in {time.time() - t0:.0f} s; {self.dram_report()}")
        store = getattr(self.engram, "store", None)
        if store is not None and hasattr(store, "rewarm") and store.loaders:
            store.rewarm()  # the tensor-cache load (~90 GB through the page cache) evicted the Engram tables

    def dram_report(self) -> str:
        mv = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        free = mv.total_bytes_free_per_bank * mv.num_banks / 2**30
        used = mv.total_bytes_allocated_per_bank * mv.num_banks / 2**30
        return f"DRAM per die: {used:.2f} GiB used, {free:.2f} GiB free (largest block {mv.largest_contiguous_bytes_free_per_bank / 2**20:.0f} MiB/bank)"

    def engram_li(self):
        return [l.li for l in self.layers if l.engram is not None] if self.engram is not None else []

    def step(self, si, R, mem=None):
        """Trace-safe rows pass over si (tt/decode.StepInputs) -> logits [1, 1, 32, Vp / N] fp32 per die."""
        from models.autoports.tencent_hy4_flash.tt.common import real_rows

        with real_rows(R):
            return self._step(si, R, mem)

    def _step(self, si, R, mem=None):
        cfg = self.cfg
        si.R = R  # the real rows (the decode boundary all-reduces only those)
        x = ttnn.embedding(si.tok, self.emb, layout=ttnn.TILE_LAYOUT, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.reshape(x, (1, 1, 32, cfg.hidden))
        fused_hc = self.head_hc.fused and int(x.shape[-2]) == 32
        X = ttnn.repeat(x, ttnn.Shape((1, cfg.hc_mult, 1, 1)),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG if fused_hc else (mem or ttnn.DRAM_MEMORY_CONFIG))
        ttnn.deallocate(x)
        if fused_hc:
            from models.autoports.tencent_hy4_flash.tt.hc_fused import HCFusedOp

            hcf = HCFusedOp.get(self.mesh, cfg.hidden)
            pend = None
            for layer in self.layers:
                X, pend = layer.step_fused(X, pend, si, R, hcf, mem)
            o, wp = pend
            X2, h, _w = hcf(X, o, wp, self.head_hc)
            for t_ in (X, X2, o, wp, _w):
                ttnn.deallocate(t_)
        else:
            for layer in self.layers:
                X = layer.step(X, si, R, mem)
            h, _ = self.head_hc.pre(X, mem)
            ttnn.deallocate(X)
        hn = self.norm(h, mem)
        ttnn.deallocate(h)
        if self.lm_dec is not None:
            lg = self.lm_dec(hn, mem)
        else:
            lg = ttnn.matmul(hn, self.lm, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
        bufs = getattr(self, "step_bufs", None)
        if bufs is not None:
            # persistent outputs, allocated before any trace capture: a trace's outputs allocated during its capture
            # can sit in another trace's freed intermediates and be overwritten by that trace's replays
            if R <= self.LG_ROWS and "lg_rows" in bufs:
                # only the real rows leave the device: row-major [LG_ROWS, Vp / N] per die (~120 KB) instead of the
                # padded fp32 tile block [32, Vp / N] (~3.9 MB per die, 4.2 ms to read); same values
                lr = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ls = ttnn.slice(lr, (0, 0, 0, 0), (1, 1, self.LG_ROWS, int(lg.shape[-1])))
                ttnn.deallocate(lr)
                ttnn.copy(ls, bufs["lg_rows"])
                ttnn.deallocate(ls)
                out = bufs["lg_rows"]
            else:
                ttnn.copy(lg, bufs["lg"])
                out = bufs["lg"]
            ttnn.copy(hn, bufs["hn"])
            ttnn.deallocate(lg)
            ttnn.deallocate(hn)
            lg, hn = out, bufs["hn"]
        self.last_hn = hn  # the final normalized hidden of the rows (kept: the DFly2 drafter's target hidden)
        return lg

    LG_ROWS = 8  # passes of up to this many rows return their logits row-major (tt/decode persistent "lg_rows")

    def read_logits(self, lg, R) -> torch.Tensor:
        return to_torch_cat(lg, dim=-1).reshape(-1, self.Vp)[:R, : self.cfg.vocab].float()

    def reset(self):
        self.pos = 0
        for layer in self.layers:
            layer.reset()
        if self.engram is not None:
            self.engram.reset()

    def _engram_rows(self, ids: torch.Tensor, p0: int, Tp: int, split: bool = False) -> dict:
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
            out[layer.li] = upload(self.mesh, full.reshape(1, 1, Tp, -1), ttnn.bfloat16,
                                   mapper=("shard", 2) if split else "rep")
        keep = self.engram.mask(ids).reshape(-1)
        if out and not bool(keep.all()):
            k = torch.zeros(Tp, 1, dtype=torch.bfloat16)
            k[: keep.numel(), 0] = keep.to(torch.bfloat16)
            out["keep"] = upload(self.mesh, k.reshape(1, 1, Tp, 1), ttnn.bfloat16, mapper=("shard", 2) if split else "rep")
        return out

    def prefill(self, ids: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """A whole prompt (continuing the sequence) -> the last row's logits. Chunks: up to the 2048-token dense regime
        boundary first (so every later chunk selects with device-built index lists), then HY4_PREFILL_CHUNK tokens
        (default: the model's chunk), each a multiple of 128 so the next chunk starts on a 32-group boundary."""
        ids = ids.reshape(-1)
        if (self.pos == 0 and os.environ.get("HY4_PF_TRACE", "0") == "1" and self.N > 1
                and 0 < ids.numel() <= self.cfg.idx_candidate_topk and self.engram_keep_all(ids)):
            return self._prefill_traced(ids)
        big = min(int(chunk or os.environ.get("HY4_PREFILL_CHUNK", self.chunk)), self.chunk)  # the SWA ring holds self.chunk
        dense_end = self.cfg.idx_candidate_topk
        lg, i = None, 0
        bucketed = os.environ.get("HY4_PF_BUCKETS", "1") == "1" and self.N > 1
        while i < ids.numel():
            p = self.pos
            n = dense_end - p if p < dense_end else big
            n = max(128, (n // 128) * 128) if i + n < ids.numel() else n
            rows = 0
            if bucketed:  # the chunk's rows padded to a bucket (compiled at warm-up: no programs at serve time)
                m = min(n, ids.numel() - i)
                rows = next((b for b in self.PF_BUCKETS if b >= m and p + b <= self.max_ctx), 0)
            lg = self.forward(ids[i : i + n], min_rows=rows)
            i += n
        return lg

    # ------------------------------------------------------------------ traced first chunk (TTFT of short prompts)
    PF_BUCKETS = (128, 256, 512, 1024, 2048)

    def engram_keep_all(self, ids) -> bool:
        return self.engram is None or bool(self.engram.mask(ids.reshape(-1)).all())

    def _chunk_layers(self, X, p0, T, cos_sin, eng, mem=None):
        for layer in self.layers:
            X = layer.forward_split(X, p0, T, cos_sin, eng.get(layer.li), mem, eng.get("keep"))
        return X

    def _prefill_traced(self, ids: torch.Tensor) -> torch.Tensor:
        """A prompt of <= 2048 tokens at position 0 as one captured trace per bucket (the chunk's every position-
        dependent input is constant at p0 = 0: RoPE tables, index lists, slab offsets; the token ids and Engram rows go
        through persistent buffers). Padding rows run as real rows (their KV / group keys sit past the prompt and are
        rewritten by decode). The head and the indexers' host views follow the replay, at the real length."""
        cfg, mesh = self.cfg, self.mesh
        T = int(ids.numel())
        Tb = next(b for b in self.PF_BUCKETS if b >= T)
        st = self.__dict__.setdefault("_pf_traces", {}).get(Tb)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        tok = torch.zeros(1, Tb, dtype=torch.int32)
        tok[0, :T] = ids.to(torch.int32)
        tok_h = ttnn.from_torch(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
        eng_h = {}
        for layer in self.layers:
            if layer.engram is not None and self.engram is not None:
                rows = self.engram.rows(layer.li, ids, 0)
                full = torch.zeros(1, 1, Tb, rows.shape[-1], dtype=torch.bfloat16)
                full[0, 0, :T] = rows
                eng_h[layer.li] = ttnn.from_torch(full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                  mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))
        for layer in self.layers:
            if layer.attn.indexer is not None:
                layer.attn.indexer._traced_pf = True
        try:
            if st is None:
                st = {"tok": ttnn.to_device(tok_h, mesh),
                      "eng": {li: ttnn.to_device(t, mesh) for li, t in eng_h.items()},
                      "cos_sin": {k: self.rope.tables(torch.arange(0, Tb), cfg.rope_theta[k]) for k in ("sparse", "swa")}}
                body = lambda: self._chunk_layers(self._embed_split(st["tok"], Tb), 0, Tb, st["cos_sin"], st["eng"])  # noqa
                X = body()  # the eager pass: compiles every program, creates every cached constant
                ttnn.deallocate(X)
                self._reset_for_trace()
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                st["X"] = body()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                st["tid"] = tid
                self._pf_traces[Tb] = st
                self._reset_for_trace()
                for layer in self.layers:  # the engram store advanced in the eager pass: rewind for the replay
                    pass
            ttnn.copy_host_to_device_tensor(tok_h, st["tok"])
            for li, t in eng_h.items():
                ttnn.copy_host_to_device_tensor(t, st["eng"][li])
            ttnn.execute_trace(mesh, st["tid"], cq_id=0, blocking=False)
        finally:
            for layer in self.layers:
                if layer.attn.indexer is not None:
                    layer.attn.indexer._traced_pf = False
        self.pos = T
        for layer in self.layers:
            if layer.attn.indexer is not None:
                layer.attn.indexer.after_traced_prefill(T, Tb)
        return self._head_split(st["X"], T, Tb, "last", None)

    def _embed_split(self, tok_t, Tp):
        cfg = self.cfg
        x = ttnn.embedding(tok_t, self.emb, layout=ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, (1, 1, Tp // self.N, cfg.hidden))
        X = ttnn.repeat(x, ttnn.Shape((1, cfg.hc_mult, 1, 1)))
        ttnn.deallocate(x)
        return X

    def _reset_for_trace(self):
        for layer in self.layers:
            if layer.attn.indexer is not None:
                layer.attn.indexer.reset()

    def forward(self, ids: torch.Tensor, logits_rows: str = "last", mem=None, min_rows: int = 0) -> torch.Tensor:
        """ids: [T] int (the next T tokens of the sequence). -> logits fp32 [V] (last row) or [T, V] (all rows).
        From HY4_ROW_SPLIT_MIN tokens (default 128) the rows are split over the dies between layers (forward_split).
        min_rows: pad the chunk to at least this many rows (a prefill bucket: every prompt length then runs one of a few
        compiled program sets; padding rows run as real rows past the prompt, their KV / keys rewritten by decode)."""
        cfg, mesh = self.cfg, self.mesh
        T = int(ids.numel())
        p0 = self.pos
        assert p0 + T <= self.max_ctx, f"context {p0 + T} > max_ctx {self.max_ctx}"
        split = self.N > 1 and max(T, min_rows) >= int(os.environ.get("HY4_ROW_SPLIT_MIN", "128"))
        align = 32 * self.N if split else 32
        Tp = -(-max(T, min_rows) // align) * align
        assert p0 + Tp <= self.max_ctx + 64 or Tp == -(-T // align) * align, "bucket padding past the cache"
        tok = torch.zeros(1, Tp, dtype=torch.int32)
        tok[0, :T] = ids.reshape(-1).to(torch.int32)
        tok_t = upload(mesh, tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mapper=("shard", 1) if split else "rep")
        x = ttnn.embedding(tok_t, self.emb, layout=ttnn.TILE_LAYOUT, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok_t)
        x = ttnn.reshape(x, (1, 1, Tp // self.N if split else Tp, cfg.hidden))
        X = ttnn.repeat(x, ttnn.Shape((1, cfg.hc_mult, 1, 1)))  # the 4 streams start equal
        ttnn.deallocate(x)
        pos = torch.arange(p0, p0 + Tp)
        cos_sin = {k: self.rope.tables(pos, cfg.rope_theta[k]) for k in ("sparse", "swa")}
        eng = self._engram_rows(ids.reshape(-1), p0, Tp, split=split)
        for layer in self.layers:
            if split:
                X = layer.forward_split(X, p0, T, cos_sin, eng.get(layer.li), mem, eng.get("keep"))
            else:
                X = layer.forward(X, p0, T, cos_sin, eng.get(layer.li), mem, eng.get("keep"))
        for t_ in list(eng.values()) + [c for cs in cos_sin.values() for c in cs]:
            ttnn.deallocate(t_)
        self.pos = p0 + T
        if split:
            out = self._head_split(X, T, Tp, logits_rows, mem)
            ttnn.deallocate(X)
            return out
        h, _ = self.head_hc.pre(X, mem)
        ttnn.deallocate(X)
        if logits_rows == "last":
            r = ((T - 1) // 32) * 32
            h = ttnn.slice(h, (0, 0, r, 0), (1, 1, r + 32, cfg.hidden))
        out = self._logits(h, mem)
        ttnn.deallocate(h)
        if logits_rows == "last":
            return out[(T - 1) - ((T - 1) // 32) * 32]
        return out[:T]

    def _logits(self, h, mem):
        hn = self.norm(h, mem)
        lg = ttnn.matmul(hn, self.lm, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
        out = to_torch_cat(lg, dim=-1).reshape(-1, self.Vp)[:, : self.cfg.vocab].float()
        ttnn.deallocate(lg)
        ttnn.deallocate(hn)
        return out

    def warm_head_blocks(self, Tp):
        """Compile the head's block slices of a Tp-row bucket (the last row's 32-row block offset depends on the prompt
        length; ttnn.slice bakes its offsets into the program)."""
        loc = Tp // self.N
        X = ttnn.zeros((1, self.cfg.hc_mult, loc, self.cfg.hidden), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                       device=self.mesh)
        for b in range(0, loc, 32):
            xs = ttnn.slice(X, (0, 0, b, 0), (1, self.cfg.hc_mult, b + 32, self.cfg.hidden))
            ttnn.deallocate(xs)
        ttnn.deallocate(X)

    def _head_split(self, X, T, Tp, logits_rows, mem):
        """Head of a row-split chunk: X [1, hc, Tp/N, H] per die. Each die takes the same local 32-row block, the blocks
        are gathered and die d's block kept: only the rows asked for cross the dies and run the fp32 head HC."""
        cfg, N = self.cfg, self.N
        loc = Tp // N
        if logits_rows == "last":
            d, lr = (T - 1) // loc, (T - 1) % loc
            blocks = [(d, (lr // 32) * 32)]
        else:
            blocks = [(None, b) for b in range(0, loc, 32)]
        rows = {}
        for d_, b in blocks:
            xs = ttnn.slice(X, (0, 0, b, 0), (1, cfg.hc_mult, b + 32, cfg.hidden), memory_config=mem)
            xg = ttnn.all_gather(xs, dim=2, cluster_axis=1, topology=ttnn.Topology.Linear)  # [1, hc, 32 N, H]
            ttnn.deallocate(xs)
            h, _ = self.head_hc.pre(xg, mem)
            ttnn.deallocate(xg)
            lg = self._logits(h, mem)  # [32 N, V]: die d's rows at d * 32
            ttnn.deallocate(h)
            for dd in range(N) if d_ is None else [d_]:
                for j in range(32):
                    g = dd * loc + b + j
                    if g < T:
                        rows[g] = lg[dd * 32 + j]
        if logits_rows == "last":
            return rows[T - 1]
        return torch.stack([rows[g] for g in range(T)])
