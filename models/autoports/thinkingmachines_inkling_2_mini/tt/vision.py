# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini's vision tower on TT (transformers `Inkling2VisionModel` + the merger), and its image preprocessing.

Preprocessing (`preprocess`, transformers `Inkling2ImageProcessor`): no resize; the image is padded right / bottom to a
multiple of 40 px with -1 (on the 0-255 scale), scaled to [-1, 1] and cut into one 40 x 40 x 3 patch per language-model
image token (raster order): pixel_values [N, 40, 40, 3] fp32, image_grid_thw [1, rows, cols].

Tower (one image per call; the weights replicated on every die):
  each token = 2 x 2 native 20 px patches -> S = 4 N patches + one class token
  x = patches @ W_pe + pos_emb (the 32 x 32 table bilinearly resampled to the patch grid; host, fp32);
      the class token: class_embedding + cls_position_embedding
  x = LayerNorm(x); 23 x [x += l1 * attn(LN(x)); x += l2 * fc2(gelu(fc1(LN(x))))]   (residual fp32)
      attn: 16 heads x 64, 2-D axial RoPE on interleaved channel pairs (w frequencies, then h), positions (row + 1,
      col + 1), the class token at (0, 0); non-causal over the image
  out = proj_out(gelu(proj_in(concat of each token's 4 patch states)))  -> [N, 2304]

Device layout: rows [patches (4 N) | class token | padding]; attention is order-free (explicit positions, no mask), so
the patch rows start tile-aligned and the merger reads the padded sequence as [S_pad / 4, 4096] (token i = rows 4i..4i+3).
Every program depends on the padded length S_pad only (BUCKETS; warm each before any trace capture): the SDPA's
block-diagonal windows [0, S) [S, S_pad) keep the real rows and the padding apart.
"""

from __future__ import annotations

import math
import os

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
PATCH = 40  # pixels per language-model token side (native patch 20 x merge 2)
NATIVE = 20
MERGE = 2
# padded sequence lengths (patches + class token): ~1.5x steps; each one is a set of programs to warm up
BUCKETS = [256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 45056]
MM_CHUNK = 1024  # matmul rows per batch entry past this length (the long sequences run as [1, S / 1024, 1024, K])
_DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
WDT = _DT[os.environ.get("I2M_VISION_DT", "bf16")]  # the tower's matmul weights


def bucket(S):
    for b in BUCKETS:
        if b > S:  # (always at least one padding row: the window [S, S_pad) is never empty)
            return b
    raise ValueError(f"image too large for the vision tower: {S} patches > {BUCKETS[-1] - 1}")


def max_tokens():
    """The most image tokens one image may have: I2M_VISION_MAX_TOKENS (default 4,095: the 16,384-row padded length,
    whose transients fit beside the one-die p150 profile's KV pool), at most the largest padded length's (10,815)."""
    return min((BUCKETS[-1] - 2) // 4, int(os.environ.get("I2M_VISION_MAX_TOKENS", "4095")))


# ------------------------------------------------------------------------------------------------------------- host
def preprocess(image):
    """PIL image -> (pixel_values [N, 40, 40, 3] fp32, image_grid_thw [1, 3] int64): transformers'
    Inkling2ImageProcessor (do_convert_rgb, pad with -1 to a multiple of 40, (x - 127.5) / 127.5, one patch per token)."""
    import numpy as np

    has_alpha = image.has_transparency_data
    image = image.convert("RGBA" if has_alpha else "RGB")
    img = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)  # [C, H, W] uint8
    if img.shape[0] == 4:  # alpha composited onto an 8 px grey checkerboard (204 / 153)
        H, W = img.shape[-2:]
        rows = torch.arange(H)[:, None] // 8
        cols = torch.arange(W)[None, :] // 8
        bg = torch.where((rows + cols) % 2 == 0, 204, 153)
        rgba = img.to(torch.int32)
        a = rgba[3:4]
        img = ((rgba[:3] * a + bg * (255 - a) + 127) // 255).to(torch.uint8)
    C, H, W = img.shape
    rows, cols = math.ceil(H / PATCH), math.ceil(W / PATCH)
    x = torch.full((C, rows * PATCH, cols * PATCH), -1.0)
    x[:, :H, :W] = img.float()
    mean = torch.tensor([0.5] * 3) * (1.0 / 0.00392156862745098)
    std = torch.tensor([0.5] * 3) * (1.0 / 0.00392156862745098)
    x = x.sub_(mean[:, None, None]).div_(std[:, None, None])
    x = x.reshape(C, rows, PATCH, cols, PATCH).permute(1, 3, 2, 4, 0).flatten(0, 1)
    return x.contiguous(), torch.tensor([[1, rows, cols]])


def _interp(index, size, side):
    """bilinear, align_corners=False, border clamp: per-axis taps [n, 2] and weights [n, 2] (fp32)."""
    src = (index.float() + 0.5) * side / size - 0.5
    fl = torch.floor(src)
    off = torch.arange(2)
    taps = (fl.long()[:, None] + off).clamp(0, side - 1)
    wts = (1 - (src[:, None] - fl[:, None] - off).abs()).clamp(min=0)
    return taps, wts


def patch_grid(grid):
    """(rows, cols) of each patch in the token-major order (token raster, then the 2 x 2 patches inside it)."""
    _, R, C = [int(v) for v in grid.reshape(-1)]
    h, w = R * MERGE, C * MERGE
    hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    shape = (h // MERGE, MERGE, w // MERGE, MERGE)
    return hh.reshape(shape).transpose(1, 2).flatten(), ww.reshape(shape).transpose(1, 2).flatten(), h, w


class HostInputs:
    """One image's host-side tower inputs at its padded length."""

    def __init__(self, tower, pixel_values, grid):
        N = int(pixel_values.shape[0])
        P = 4 * N
        self.N, self.S = N, P + 1
        self.S_pad = bucket(self.S)
        pv = pixel_values.float().reshape(-1, MERGE, NATIVE, MERGE, NATIVE, 3).transpose(2, 3)
        patches = torch.zeros(self.S_pad, tower.K_pe)
        patches[:P, : 3 * NATIVE * NATIVE] = pv.reshape(P, -1)
        self.patches = patches
        r, c, h, w = patch_grid(grid)
        g = tower.grid_side
        ht, hw_ = _interp(r, h, g)
        wt, ww_ = _interp(c, w, g)
        idx = (ht[:, :, None] * g + wt[:, None, :]).reshape(-1, 4)
        wts = (hw_[:, :, None] * ww_[:, None, :]).reshape(-1, 4)
        add = torch.zeros(self.S_pad, tower.D)
        add[:P] = (tower.pos_table[idx] * wts[:, :, None]).sum(1)
        add[P] = tower.cls
        self.add = add
        pos = torch.zeros(self.S_pad, 2)
        pos[:P, 0], pos[:P, 1] = r.float() + 1, c.float() + 1
        f = pos[..., None] * tower.inv_freq  # [S, 2, hd / 4]
        cos, sin = f.cos(), f.sin()
        self.cos = torch.cat([cos[:, 1], cos[:, 0]], -1).repeat_interleave(2, -1)  # (w frequencies, then h)
        self.sin = torch.cat([sin[:, 1], sin[:, 0]], -1).repeat_interleave(2, -1)


# ----------------------------------------------------------------------------------------------------------- device
BIG = ("wqkv", "wo", "w1", "w2")  # the per-layer matmul weights (streamed from the host in stream mode)
MLP_ROWS = 4096  # past 8192 rows the MLP runs in row blocks of this many rows (its [rows, 4096] intermediates)


class VisionTower:
    """stream=True: the 23 layers' and the merger's matmul weights stay on the host (tilized) and are written to the
    device one layer at a time during a forward (~0.6 GB per image; for a die without room for them resident)."""

    def __init__(self, mesh, ck, vcfg: dict, stream=False):
        self.mesh, self.stream = mesh, stream
        self.rep = ttnn.ReplicateTensorToMesh(mesh)
        self.D = D = int(vcfg["hidden_size"])
        self.nh = int(vcfg["num_attention_heads"])
        self.hd = D // self.nh
        self.L = int(vcfg["num_hidden_layers"])
        self.eps = float(vcfg.get("layer_norm_eps", 1e-5))
        self.grid_side = int(vcfg.get("posemb_grid_size", 32))
        theta = float((vcfg.get("rope_parameters") or {}).get("rope_theta", 10000.0))
        sd = self.hd // 2
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, sd, 2, dtype=torch.float) / sd))
        pfx = "model.visual.backbone."
        g = lambda k: ck.get(pfx + k)  # noqa: E731
        self.pos_table = g("embeddings.position_embedding").float()
        self.cls = g("embeddings.class_embedding").float() + g("embeddings.cls_position_embedding").float()
        pe = g("embeddings.patch_embedding.weight").float()  # [D, 1200]
        self.K_pe = (pe.shape[1] + 31) // 32 * 32
        self.w_pe = self._w(torch.nn.functional.pad(pe.T, (0, 0, 0, self.K_pe - pe.shape[1])), ttnn.bfloat16, False)
        self.ln_pre = self._ln(g("ln_pre.weight"), g("ln_pre.bias"))
        self.layers = []
        for i in range(self.L):
            p = f"layers.{i}."
            a = p + "self_attn."
            wqkv = torch.cat([g(a + n + "_proj.weight") for n in "qkv"], 0)  # [3D, D]
            bqkv = torch.cat([g(a + n + "_proj.bias") for n in "qkv"], 0)
            self.layers.append(dict(
                ln1=self._ln(g(p + "ln_1.weight"), g(p + "ln_1.bias")),
                ln2=self._ln(g(p + "ln_2.weight"), g(p + "ln_2.bias")),
                wqkv=self._w(wqkv.T), bqkv=self._b(bqkv),
                wo=self._w(g(a + "out_proj.weight").T), bo=self._b(g(a + "out_proj.bias")),
                w1=self._w(g(p + "mlp.c_fc.weight").T), b1=self._b(g(p + "mlp.c_fc.bias")),
                w2=self._w(g(p + "mlp.c_proj.weight").T), b2=self._b(g(p + "mlp.c_proj.bias")),
                l1=self._b(g(p + "ls_1.gamma"), ttnn.float32), l2=self._b(g(p + "ls_2.gamma"), ttnn.float32),
            ))
        self.merger = dict(w_in=self._w(ck.get("model.visual.proj_in.weight").T),
                           w_out=self._w(ck.get("model.visual.proj_out.weight").T))
        self.Dout = int(ck.get("model.visual.proj_out.weight").shape[0])
        r = torch.zeros(1, 1, self.hd, self.hd)  # x @ r: each channel pair (x1, x2) -> (-x2, x1)
        r[..., torch.arange(0, self.hd, 2), torch.arange(1, self.hd, 2)] = 1
        r[..., torch.arange(1, self.hd, 2), torch.arange(0, self.hd, 2)] = -1
        self.rot = self._up(r, ttnn.bfloat16)
        hifi4 = dict(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
                     packer_l1_acc=True)
        self.ck_mm = ttnn.init_device_compute_kernel_config(mesh.arch(), **hifi4)
        self.ck_ln = ttnn.init_device_compute_kernel_config(mesh.arch(), **{**hifi4, "packer_l1_acc": False})
        self.ck_sdpa = ttnn.init_device_compute_kernel_config(mesh.arch(), **{**hifi4, "packer_l1_acc": False})

    # -- weights
    def _w(self, w, dtype=None, big=True):
        t = ttnn.from_torch(w.contiguous().to(torch.bfloat16).reshape(1, 1, *w.shape), dtype=dtype or WDT,
                            layout=ttnn.TILE_LAYOUT, mesh_mapper=self.rep)
        return t if (big and self.stream) else t.to(self.mesh, DRAM)

    def _b(self, b, dtype=ttnn.bfloat16):
        return self._up(b.float().reshape(1, 1, 1, -1), dtype)

    def _ln(self, w, b):
        D = w.numel()
        return tuple(self._up(t.float().reshape(1, 1, D // 32, 32), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT) for t in (w, b))

    def _on_device(self, w):
        """A layer's weights on the device (stream mode: its matmul weights written now; free them with _release)."""
        if not self.stream:
            return w
        return {k: (v.to(self.mesh, DRAM) if k in BIG or k.startswith("w_") else v) for k, v in w.items()}

    def _release(self, w):
        if self.stream:
            for k, v in w.items():
                if k in BIG or k.startswith("w_"):
                    ttnn.deallocate(v)

    # -- ops
    def _up(self, t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh, memory_config=DRAM, mesh_mapper=self.rep)

    def _linear(self, x, w, b=None, dtype=ttnn.bfloat16):
        S = int(x.shape[-2])
        c = next((c for c in (MM_CHUNK, MM_CHUNK // 2, MM_CHUNK // 4) if S > c and S % c == 0), None)
        if c:
            x = ttnn.reshape(x, (1, S // c, c, int(x.shape[-1])))
        y = ttnn.linear(x, w, bias=b, dtype=dtype, memory_config=DRAM, compute_kernel_config=self.ck_mm)
        if c:
            y = ttnn.reshape(y, (1, 1, S, int(y.shape[-1])))
        return y

    def _layernorm(self, x, wb, dtype=ttnn.bfloat16):
        y = ttnn.layer_norm(x, epsilon=self.eps, weight=wb[0], bias=wb[1], memory_config=DRAM,
                            compute_kernel_config=self.ck_ln)
        if y.dtype == dtype:
            return y
        z = ttnn.typecast(y, dtype)
        ttnn.deallocate(y)
        return z

    def _sdpa_cfg(self, S):
        q = k = 256 if S % 256 == 0 and S >= 2048 else 128  # (k 512: the CBs outgrow L1)
        return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
                                      q_chunk_size=q, k_chunk_size=k, exp_approx_mode=False)

    def _rope(self, x, cos, sin):
        """x cos + (x rot) sin in fp32 (the pair swap is exact on the bf16 x; cos / sin fp32), one bf16 rounding."""
        y = ttnn.multiply(x, cos, dtype=ttnn.float32, memory_config=DRAM)
        xt = ttnn.matmul(x, self.rot, dtype=ttnn.float32, compute_kernel_config=self.ck_mm)
        ttnn.multiply(xt, sin, output_tensor=xt)
        ttnn.add(y, xt, output_tensor=y)
        ttnn.deallocate(xt)
        out = ttnn.typecast(y, ttnn.bfloat16)
        ttnn.deallocate(y)
        return out

    def _mlp(self, x, w):
        """x += l2 * fc2(gelu(fc1(LN(x)))) (fp32, in place)."""
        h = self._layernorm(x, w["ln2"])
        f = self._linear(h, w["w1"], w["b1"])
        ttnn.deallocate(h)
        ttnn.gelu(f, fast_and_approximate_mode=False, output_tensor=f)
        m = self._linear(f, w["w2"], w["b2"], dtype=ttnn.float32)
        ttnn.deallocate(f)
        ttnn.multiply(m, w["l2"], output_tensor=m)
        ttnn.add(x, m, output_tensor=x)
        ttnn.deallocate(m)
        return x

    def _layer(self, x, w, cos, sin, win, S):
        """One encoder layer on the fp32 residual x (updated in place; the MLP in row blocks past 2 x MLP_ROWS)."""
        h = self._layernorm(x, w["ln1"])
        qkv = self._linear(h, w["wqkv"], w["bqkv"])
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=self.nh, num_kv_heads=self.nh,
                                                         transpose_k_heads=False, memory_config=DRAM)
        ttnn.deallocate(qkv)
        qr = self._rope(q, cos, sin)
        ttnn.deallocate(q)
        kr = self._rope(k, cos, sin)
        ttnn.deallocate(k)
        o = ttnn.transformer.scaled_dot_product_attention(
            qr, kr, v, is_causal=False, scale=self.hd**-0.5, cu_window_seqlens=win,
            program_config=self._sdpa_cfg(S), compute_kernel_config=self.ck_sdpa)
        for t in (qr, kr, v):
            ttnn.deallocate(t)
        o2 = ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM)
        ttnn.deallocate(o)
        a = self._linear(o2, w["wo"], w["bo"], dtype=ttnn.float32)
        ttnn.deallocate(o2)
        ttnn.multiply(a, w["l1"], output_tensor=a)
        ttnn.add(x, a, output_tensor=x)
        ttnn.deallocate(a)
        if S <= 2 * MLP_ROWS:
            return self._mlp(x, w)
        parts = []
        for r0 in range(0, S, MLP_ROWS):
            parts.append(self._mlp(ttnn.slice(x, [0, 0, r0, 0], [1, 1, r0 + MLP_ROWS, self.D], memory_config=DRAM), w))
        ttnn.deallocate(x)
        y = ttnn.concat(parts, dim=2, memory_config=DRAM)
        for t in parts:
            ttnn.deallocate(t)
        return y

    def forward_device(self, hi: HostInputs, capture=None):
        """-> [1, 1, S_pad / 4, Dout] bf16 (row i = token i for i < N; the rest is padding)."""
        S, Sp = hi.S, hi.S_pad
        D = self.D
        pt = self._up(hi.patches.reshape(1, 1, Sp, -1), ttnn.float32)
        x = self._linear(pt, self.w_pe, dtype=ttnn.float32)
        ttnn.deallocate(pt)
        add = self._up(hi.add.reshape(1, 1, Sp, D), ttnn.float32)
        x0 = ttnn.add(x, add, memory_config=DRAM)
        ttnn.deallocate(x)
        ttnn.deallocate(add)
        x = self._layernorm(x0, self.ln_pre, dtype=ttnn.float32)
        ttnn.deallocate(x0)
        if capture is not None:
            capture("embed", x)
        cos = self._up(hi.cos.reshape(1, 1, Sp, self.hd), ttnn.float32)
        sin = self._up(hi.sin.reshape(1, 1, Sp, self.hd), ttnn.float32)
        win = ttnn.from_torch(torch.tensor([0, S, Sp], dtype=torch.int32), dtype=ttnn.int32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, mesh_mapper=self.rep)
        for i, w in enumerate(self.layers):
            wd = self._on_device(w)
            x = self._layer(x, wd, cos, sin, win, Sp)
            self._release(wd)
            if capture is not None:
                capture(i, x)
        for t in (cos, sin, win):
            ttnn.deallocate(t)
        # merger: token i = the 4 patch rows 4i .. 4i + 3 (row-major reinterpretation of the padded sequence)
        xb = ttnn.typecast(x, ttnn.bfloat16)
        ttnn.deallocate(x)
        rm = ttnn.to_layout(xb, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(xb)
        m = ttnn.to_layout(ttnn.reshape(rm, (1, 1, Sp // 4, 4 * D)), ttnn.TILE_LAYOUT)
        ttnn.deallocate(rm)
        mw = self._on_device(self.merger)
        h = self._linear(m, mw["w_in"])
        ttnn.deallocate(m)
        h2 = ttnn.gelu(h, fast_and_approximate_mode=False, memory_config=DRAM)
        ttnn.deallocate(h)
        out = self._linear(h2, mw["w_out"])
        ttnn.deallocate(h2)
        self._release(mw)
        return out

    def __call__(self, pixel_values, grid):
        """One image -> its token embeddings [N, Dout] bf16 (host)."""
        hi = HostInputs(self, pixel_values, grid)
        out = self.forward_device(hi)
        t = ttnn.to_torch(ttnn.get_device_tensors(out)[0]) if self.mesh.get_num_devices() > 1 else ttnn.to_torch(out)
        ttnn.deallocate(out)
        return t.reshape(-1, self.Dout)[: hi.N]

    def warm(self, max_tok=None):
        """Every bucket's programs up to images of max_tok tokens (default max_tokens()), before any trace capture."""
        n_max = max_tokens() if max_tok is None else max_tok
        done = []
        for b in BUCKETS:
            n = min((b - 2) // 4, n_max)  # the largest image of this bucket
            if n < 1 or bucket(4 * n + 1) != b:
                break
            self(torch.zeros(n, PATCH, PATCH, 3), torch.tensor([[1, 1, n]]))
            done.append(b)
            if n == n_max:
                break
        return done


def vision_config(snap):
    import json

    with open(os.path.join(snap, "config.json")) as f:
        return json.load(f)["vision_config"]
