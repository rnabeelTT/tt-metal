# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-flash vision tower (HF HYV4FlashVisionTransformer + HYV4FlashVisionPatchMerger) on the (1, N) mesh.

    patch embed: Conv2d(3, 1152, 16, 16) on flat patches = a [768 -> 1152] linear, + the learned 128 x 128 position
    table bilinearly resized to the grid (HF: scale_factor (g + 0.1) / 128, align_corners False; host, fp32 -> bf16)
    27 x pre-LN block: LN -> q/k/v (bias, 16 heads x 72) -> full (non-causal) attention per image -> o_proj (bias)
                       -> residual; LN -> fc1 (4304, bias) -> GELU (erf) -> fc2 (bias) -> residual
    merger: RMSNorm -> Conv2d(1152, 4608, 2, 2) -> GELU -> Conv2d(4608, 3072, 1) -> + image_newline after every row
    -> features [(gh / 2) x (gw / 2 + 1), 3072] (the embeddings of the prompt's image-pad tokens, in order).

Layout on the device:
- Patch rows go in 2 x 2 block order (block row, block col, dy, dx), position rows permuted alike (attention is
  permutation-equivariant; no rotary in this tower), so the merger's 2 x 2 conv is a reshape [Np, 1152] -> [Np / 4,
  4608] (4 consecutive rows side by side) and a linear.
- Tensor parallel over the N dies (weights 1/N per die; the LM leaves little DRAM): die d owns heads 4d .. 4d + 3 of
  q / k / v (column parallel, head dim 72 zero-padded to 96: tile aligned), its heads' o_proj rows (row parallel ->
  all-reduce), fc1 columns 1088 d .. (4304 padded to 4352) and fc2 rows (row parallel -> all-reduce); merger conv
  columns / out-proj rows alike. Activations [1, 1, Np, 1152] replicated between blocks.
- Rows padded to Np (multiple of 32): padding keys are masked through the spare head dims: q[72] = 1 (bias), the
  padding rows' k[72] = -3e4 (an added per-row constant), so their scores are ~ -3500 (exp -> 0). Real rows add 0.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.autoports.tencent_hy4_flash.tt.common import AllReduce, ckc, mm, num_devices, upload
from models.autoports.tencent_hy4_flash.tt.vision_host import IMAGE_PAD, preprocess, prompt_positions  # noqa: F401
from models.autoports.tencent_hy4_flash.tt.weights import Hy4Weights

D, HEADS, HD, HDP, FF, OUT = 1152, 16, 72, 96, 4304, 3072
EDGE = 128  # max_image_size 2048 / patch 16
EPS = 1e-5
KMASK = -30000.0


def block_order(gh: int, gw: int) -> np.ndarray:
    """Row-major patch index of every row in 2 x 2 block order."""
    bh, bw, dy, dx = np.meshgrid(np.arange(gh // 2), np.arange(gw // 2), np.arange(2), np.arange(2), indexing="ij")
    return ((2 * bh + dy) * gw + (2 * bw + dx)).reshape(-1)


def _dt(name):
    return {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[name]


class Hy4Vision:
    def __init__(self, mesh, w: Hy4Weights | None = None, wdtype: str | None = None, layers: int | None = None):
        self.mesh = mesh
        self.N = num_devices(mesh)
        assert HEADS % self.N == 0
        self.H = HEADS // self.N  # heads per die
        w = w or Hy4Weights()
        wd = _dt(wdtype or os.environ.get("HY4_VIT_WDTYPE", "bf16"))  # bfp8: same latency, PCC 0.9994 -> 0.9988
        self.ck = ckc(mesh, ttnn.MathFidelity.HiFi4 if wd == ttnn.bfloat16 else ttnn.MathFidelity.HiFi2, fp32=True)
        self.ck_norm = ckc(mesh)
        self.ck_sdpa = ckc(mesh, ttnn.MathFidelity.HiFi4, fp32=True)
        self.ar = AllReduce(mesh)
        t0 = time.time()
        g = lambda n: w.get("vit." + n)  # noqa: E731
        col = lambda t: upload(mesh, t.to(torch.bfloat16).reshape(1, 1, *t.shape), wd, mapper=("shard", 3))  # noqa
        row = lambda t: upload(mesh, t.to(torch.bfloat16).reshape(1, 1, *t.shape), wd, mapper=("shard", 2))  # noqa
        vec = lambda t, dt=ttnn.bfloat16: upload(mesh, t.to(torch.bfloat16).reshape(1, 1, 1, -1), dt)  # noqa
        cvec = lambda t: upload(mesh, t.to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.bfloat16, mapper=("shard", 3))  # noqa
        pw = g("embeddings.patch_embedding.weight").float().reshape(D, -1)  # [1152, 3 * 16 * 16] (c, y, x)
        self.patch_w = upload(mesh, pw.T.contiguous().to(torch.bfloat16).reshape(1, 1, -1, D), ttnn.bfloat16)
        self.patch_b = vec(g("embeddings.patch_embedding.bias"))
        self.pos_table = g("embeddings.position_embedding.weight")[1:].float()  # [128 * 128, 1152] (row 0: cls)
        nl = 27 if layers is None else int(layers)
        self.layers = []
        H, N = self.H, self.N
        for li in range(nl):
            p = f"layers.{li}."
            Wqkv = torch.zeros(D, N, 3, H, HDP)
            bqkv = torch.zeros(N, 3, H, HDP)
            for j, nm in enumerate(("q", "k", "v")):
                Wt = g(p + f"self_attn.{nm}_proj.weight").float().T.reshape(D, N, H, HD)  # in x (die, head, 72)
                Wqkv[:, :, j, :, :HD] = Wt
                bqkv[:, j, :, :HD] = g(p + f"self_attn.{nm}_proj.bias").float().reshape(N, H, HD)
            bqkv[:, 0, :, HD] = 1.0  # q[72] = 1: the padding-key mask channel
            Wo = torch.zeros(N, H, HDP, D)
            Wo[:, :, :HD] = g(p + "self_attn.o_proj.weight").float().T.reshape(N, H, HD, D)
            W1 = torch.zeros(D, N * (self._ffp() // N))
            W1[:, :FF] = g(p + "mlp.dense_h_to_4h.weight").float().T
            b1 = torch.zeros(W1.shape[1])
            b1[:FF] = g(p + "mlp.dense_h_to_4h.bias").float()
            W2 = torch.zeros(W1.shape[1], D)
            W2[:FF] = g(p + "mlp.dense_4h_to_h.weight").float().T
            self.layers.append({
                "ln1_w": vec(g(p + "input_layernorm.weight")), "ln1_b": vec(g(p + "input_layernorm.bias")),
                "ln2_w": vec(g(p + "post_attention_layernorm.weight")),
                "ln2_b": vec(g(p + "post_attention_layernorm.bias")),
                "qkv": col(Wqkv.reshape(D, -1)), "qkv_b": cvec(bqkv.reshape(-1)),
                "o": row(Wo.reshape(-1, D)), "o_b": vec(g(p + "self_attn.o_proj.bias")),
                "fc1": col(W1), "fc1_b": cvec(b1), "fc2": row(W2), "fc2_b": vec(g(p + "mlp.dense_4h_to_h.bias")),
            })
        self.m_norm = vec(g("perceive.norm.weight"))
        c0 = g("perceive.proj.0.weight").float()  # [4608, 1152, 2, 2] -> in (dy, dx, c)
        self.m_w0 = col(c0.permute(0, 2, 3, 1).reshape(c0.shape[0], -1).T.contiguous())
        self.m_b0 = cvec(g("perceive.proj.0.bias"))
        c2 = g("perceive.proj.2.weight").float().reshape(OUT, -1)  # [3072, 4608]
        self.m_w2 = row(c2.T.contiguous())
        self.m_b2 = vec(g("perceive.proj.2.bias"))
        self.newline = g("perceive.image_newline").to(torch.bfloat16)
        self._pos_cache = {}
        logger.info(f"Hy4Vision: {nl} blocks ({wd}) loaded in {time.time() - t0:.1f} s")

    BUCKETS = (256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384)

    def bucket(self, n: int) -> int:
        """Padded patch rows of an image of n patches: the smallest bucket (HY4_VIT_BUCKETS=0: the next multiple of 32;
        every program then depends on the image size and compiles at its first use)."""
        if os.environ.get("HY4_VIT_BUCKETS", "1") == "1":
            b = next((b for b in self.BUCKETS if b >= n), None)
            if b is not None:
                return b
        return -(-n // 32) * 32

    def warm(self, model=None, max_patches: int | None = None):
        """Compile every bucket's programs (one zero image per bucket up to max_patches, default the serving cap) and
        the LM's image-row splice at every prefill bucket (before the engine captures its traces)."""
        from models.autoports.tencent_hy4_flash.tt.vision_host import grid_for, max_pixels

        cap = max_patches or (grid_for(int(max_pixels() ** 0.5), int(max_pixels() ** 0.5))[1] ** 2)
        t0 = time.time()
        done = []
        for b in self.BUCKETS:
            side = 2 * int((b // 4) ** 0.5)
            while side * side > b:
                side -= 2
            if side * side > cap or (done and b > cap):
                break
            self.forward(np.zeros((side * side, 768), dtype=np.float32), (1, side, side))
            done.append(b)
        if model is not None:
            import ttnn as _t

            for Tb in getattr(model, "PF_BUCKETS", ()):
                for split in (False, True):
                    rows = Tb // model.N if split else Tb
                    x = upload(self.mesh, torch.zeros(1, 1, Tb, OUT, dtype=torch.bfloat16), _t.bfloat16,
                               mapper=("shard", 2) if split else "rep")
                    model._img = (model.slot, torch.zeros(1, OUT, dtype=torch.bfloat16), torch.tensor([0]))
                    try:
                        y = model._inject_images(x, 0, Tb, Tb, split)
                    finally:
                        model._img = None
                    _t.deallocate(y)
                    del rows
        logger.info(f"Hy4Vision warm: buckets {done} + splice in {time.time() - t0:.1f} s")

    def _ffp(self):
        q = 32 * self.N
        return -(-FF // q) * q

    # ------------------------------------------------------------------ host constants
    def pos_embed(self, gh: int, gw: int) -> torch.Tensor:
        """HF interpolate_pos_encoding: [gh * gw, 1152] bf16 (row-major patches)."""
        key = (gh, gw)
        if key not in self._pos_cache:
            t = self.pos_table.reshape(1, EDGE, EDGE, D).permute(0, 3, 1, 2)
            r = F.interpolate(t, scale_factor=((gh + 0.1) / EDGE, (gw + 0.1) / EDGE), mode="bilinear",
                              align_corners=False)
            if len(self._pos_cache) > 8:
                self._pos_cache.clear()
            self._pos_cache[key] = r.permute(0, 2, 3, 1).reshape(-1, D).to(torch.bfloat16)
        return self._pos_cache[key]

    # ------------------------------------------------------------------ device
    def _sdpa_cfg(self, Np):
        qc = next(c for c in (256, 128, 64, 32) if Np % c == 0)
        kc = next(c for c in (512, 256, 128, 64, 32) if Np % c == 0)
        grid = self.mesh.compute_with_storage_grid_size()
        return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y), q_chunk_size=qc,
                                      k_chunk_size=kc, exp_approx_mode=False)

    def _block(self, x, L, kmask, sdpa_cfg, mem):
        H = self.H
        h = ttnn.layer_norm(x, epsilon=EPS, weight=L["ln1_w"], bias=L["ln1_b"], compute_kernel_config=self.ck_norm,
                            memory_config=mem)
        qkv = mm(h, L["qkv"], self.ck, mem)
        ttnn.deallocate(h)
        qkv2 = ttnn.add(qkv, L["qkv_b"], memory_config=mem)
        ttnn.deallocate(qkv)
        if kmask is not None:
            qkv3 = ttnn.add(qkv2, kmask, memory_config=mem)
            ttnn.deallocate(qkv2)
            qkv2 = qkv3
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv2, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
                                                         memory_config=mem)
        ttnn.deallocate(qkv2)
        o = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, scale=HD ** -0.5,
                                                          program_config=sdpa_cfg,
                                                          compute_kernel_config=self.ck_sdpa)
        for t_ in (q, k, v):
            ttnn.deallocate(t_)
        oc = ttnn.experimental.nlp_concat_heads(o, memory_config=mem)  # [1, 1, Np, H * 96]
        ttnn.deallocate(o)
        a = mm(oc, L["o"], self.ck, mem)
        ttnn.deallocate(oc)
        a2 = self.ar(a, mem)
        ttnn.deallocate(a)
        x2 = ttnn.add(x, a2, memory_config=mem)
        ttnn.deallocate(a2)
        x3 = ttnn.add(x2, L["o_b"], memory_config=mem)
        ttnn.deallocate(x2)
        h = ttnn.layer_norm(x3, epsilon=EPS, weight=L["ln2_w"], bias=L["ln2_b"], compute_kernel_config=self.ck_norm,
                            memory_config=mem)
        f = mm(h, L["fc1"], self.ck, mem)
        ttnn.deallocate(h)
        f2 = ttnn.gelu(ttnn.add(f, L["fc1_b"], memory_config=mem), fast_and_approximate_mode=False, memory_config=mem)
        ttnn.deallocate(f)
        y = mm(f2, L["fc2"], self.ck, mem)
        ttnn.deallocate(f2)
        y2 = self.ar(y, mem)
        ttnn.deallocate(y)
        x4 = ttnn.add(x3, y2, memory_config=mem)
        ttnn.deallocate(x3)
        ttnn.deallocate(y2)
        out = ttnn.add(x4, L["fc2_b"], memory_config=mem)
        ttnn.deallocate(x4)
        return out

    def embed(self, pixel_values, grid):
        """-> (x [1, 1, Np, 1152] bf16 replicated in block order, n, Np, order)."""
        _, gh, gw = grid
        n = gh * gw
        Np = self.bucket(n)
        order = block_order(gh, gw)
        pv = torch.zeros(Np, pixel_values.shape[-1], dtype=torch.bfloat16)
        pv[:n] = torch.as_tensor(np.asarray(pixel_values))[order].to(torch.bfloat16)
        pos = torch.zeros(Np, D, dtype=torch.bfloat16)
        pos[:n] = self.pos_embed(gh, gw)[order]
        pv_t = upload(self.mesh, pv.reshape(1, 1, Np, -1), ttnn.bfloat16)
        pos_t = upload(self.mesh, pos.reshape(1, 1, Np, D), ttnn.bfloat16)
        e = ttnn.matmul(pv_t, self.patch_w, compute_kernel_config=self.ck_norm)
        e2 = ttnn.add(e, self.patch_b)  # HF: conv output (bf16, bias inside) + pos (bf16)
        x = ttnn.add(e2, pos_t)
        for t_ in (pv_t, pos_t, e, e2):
            ttnn.deallocate(t_)
        return x, n, Np, order

    def _kmask(self, n, Np):
        if Np == n:
            return None
        m = torch.zeros(Np, self.N, 3, self.H, HDP)
        m[n:, :, 1, :, HD] = KMASK
        return upload(self.mesh, m.reshape(1, 1, Np, -1).to(torch.bfloat16), ttnn.bfloat16, mapper=("shard", 3))

    def forward(self, pixel_values, grid, states: list | None = None, mem=None) -> torch.Tensor:
        """One image: pixel_values [gh * gw, 768] (row-major, vision_host.preprocess), grid (1, gh, gw) -> features
        bf16 [(gh / 2) * (gw / 2 + 1), 3072]. states: append every block input (+ the merger input), row-major, host."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        _, gh, gw = grid
        assert grid[0] == 1 and gh % 2 == 0 and gw % 2 == 0, grid
        x, n, Np, order = self.embed(pixel_values, grid)
        kmask = self._kmask(n, Np)
        cfg = self._sdpa_cfg(Np)
        inv = np.argsort(order)
        for L in self.layers:
            if states is not None:
                states.append(ttnn.to_torch(ttnn.get_device_tensors(x)[0]).reshape(Np, D)[:n][inv])
            x = self._block(x, L, kmask, cfg, mem)
        if states is not None:
            states.append(ttnn.to_torch(ttnn.get_device_tensors(x)[0]).reshape(Np, D)[:n][inv])
        if kmask is not None:
            ttnn.deallocate(kmask)
        feat = self.merge(x, Np, mem)
        ttnn.deallocate(x)
        rows = ttnn.to_torch(ttnn.get_device_tensors(feat)[0]).reshape(Np // 4, OUT)[: n // 4]
        ttnn.deallocate(feat)
        H2, W2 = gh // 2, gw // 2
        out = torch.empty(H2, W2 + 1, OUT, dtype=torch.bfloat16)
        out[:, :W2] = rows.reshape(H2, W2, OUT)
        out[:, W2] = self.newline
        return out.reshape(-1, OUT)

    def merge(self, x, Np, mem):
        h = ttnn.rms_norm(x, epsilon=EPS, weight=self.m_norm, compute_kernel_config=self.ck_norm, memory_config=mem)
        h4 = ttnn.reshape(h, (1, 1, Np // 4, 4 * D))
        a = mm(h4, self.m_w0, self.ck, mem)
        ttnn.deallocate(h)
        a2 = ttnn.gelu(ttnn.add(a, self.m_b0, memory_config=mem), fast_and_approximate_mode=False, memory_config=mem)
        ttnn.deallocate(a)
        y = mm(a2, self.m_w2, self.ck, mem)
        ttnn.deallocate(a2)
        y2 = self.ar(y, mem)
        ttnn.deallocate(y)
        out = ttnn.add(y2, self.m_b2, memory_config=mem)
        ttnn.deallocate(y2)
        return out

    def encode(self, images) -> list:
        """images [(pixel_values, grid)] -> [features]."""
        return [self.forward(pv, g) for pv, g in images]

    def prepare(self, ids, images) -> dict:
        """A prompt with its image-pad tokens expanded (vision_host.expand_image_tokens) and its images [(pixel_values,
        grid)] in order -> Hy4Model.set_images kwargs (the engines' prefill(..., mm=))."""
        ids = [int(t) for t in ids]
        pos3, delta = prompt_positions(ids, [g for _, g in images])
        feats = torch.cat(self.encode(images))
        idx = torch.nonzero(torch.tensor(ids) == IMAGE_PAD).reshape(-1)
        assert idx.numel() == feats.shape[0], (idx.numel(), feats.shape)
        return {"pos3": pos3, "delta": delta, "embeds": feats, "idx": idx}


def request_images(kwargs: dict, u: int):
    """The plugin's multimodal kwargs (pixel_values / image_grid_thw: per batch row a list per image, or None) -> row
    u's [(pixel_values float32 [gh * gw, 768], grid (1, gh, gw))], [] for a text request."""
    pvs = (kwargs.get("pixel_values") or [None] * (u + 1))
    grids = (kwargs.get("image_grid_thw") or [None] * (u + 1))
    if u >= len(pvs) or pvs[u] is None:
        return []
    out = []
    for pv, g in zip(pvs[u], grids[u]):
        if pv is None:
            continue
        g = [int(v) for v in torch.as_tensor(g).reshape(-1).tolist()]
        out.append((torch.as_tensor(pv).float().numpy(), tuple(g)))
    return out
