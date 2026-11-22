# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode sampling from the device's top-32 candidates (the vLLM plugin's sample_on_device_mode, decode steps).

On device, inside the decode trace: the logits [1, 1, W, Vp] fp32 -> bf16 -> qwen36 decode_sampling.RowsTopK (one fused
scan + merge program, <= 8 live rows per call, UINT32 indices above 65536 columns) twice, the second on the rows shifted
by the first pass's max (see DeviceTopK.device). Only [W, 32] indices + values cross PCIe (vs [W, Vp] fp32 logits), and vLLM's CPU sampler over the
full vocab is skipped. On the host: vLLM's semantics over the candidates (greedy at temperature 0; temperature, top-k,
top-p; per-request seeds; repetition / frequency / presence penalties applied to the candidates). Truncating the
distribution to the top 32 is what models/common/sampling (max_top_k 32) does for the other ports.
"""

from __future__ import annotations

import torch

import ttnn

K = 32
GROUP = 8  # RowsTopK's live-row limit


class DeviceTopK:
    def __init__(self, mesh, Vp: int, V: int):
        from models.demos.blackhole.qwen36.tt.decode_sampling.ops import RowsTopK

        self.mesh, self.Vp, self.V = mesh, Vp, V
        self.N = mesh.get_num_devices()
        self.rtk = RowsTopK(mesh, Vp, k=K)

    def device(self, lg, W: int):
        """Device half (traceable): lg [1, 1, W, Vp] fp32 TILE -> [(indices [1, 1, rows, K], logits - row max
        [1, 1, rows, K] bf16, rows)] per group of <= 8 rows. Two fused top-k passes: the first's best bf16 value (within
        one bf16 ulp of the row max) shifts the fp32 row so the candidates sit near 0, where bf16 resolves them to
        ~1e-3 (softmax is shift-invariant; the arg max stays exact); a gather of the fp32 values would cost 3.5 ms."""
        lb = ttnn.typecast(lg, ttnn.bfloat16)
        parts = []
        for g0 in range(0, W, GROUP):
            rows = min(GROUP, W - g0)
            part_b = lb if W <= GROUP else ttnn.slice(lb, [0, 0, g0, 0], [1, 1, g0 + rows, self.Vp])
            part_f = lg if W <= GROUP else ttnn.slice(lg, [0, 0, g0, 0], [1, 1, g0 + rows, self.Vp])
            v1, _ = self.rtk(part_b, rows)
            top = ttnn.typecast(ttnn.slice(v1, [0, 0, 0, 0], [1, 1, rows, 1]), ttnn.float32)
            if int(part_f.shape[2]) != rows:
                part_f = ttnn.slice(part_f, [0, 0, 0, 0], [1, 1, rows, self.Vp])
            shifted = ttnn.typecast(ttnn.subtract(part_f, top), ttnn.bfloat16)
            v2, i2 = self.rtk(shifted, rows)
            parts.append((ttnn.slice(i2, [0, 0, 0, 0], [1, 1, rows, K]), ttnn.slice(v2, [0, 0, 0, 0], [1, 1, rows, K]),
                          rows))
        return parts

    def host(self, parts):
        """-> host (indices [W, K] int64, logits [W, K] fp32) by row, sorted descending."""
        d0 = (lambda t: ttnn.get_device_tensors(t)[0]) if self.N > 1 else (lambda t: t)
        idx = torch.cat([ttnn.to_torch(d0(i)).reshape(r, K).to(torch.int64) for i, _, r in parts])
        val = torch.cat([ttnn.to_torch(d0(v)).float().reshape(r, K) for _, v, r in parts])
        return idx, val

    def __call__(self, lg, W: int):
        return self.host(self.device(lg, W))


def _param(sp, name, r, default):
    v = getattr(sp, name, None)
    if v is None:
        return default
    if isinstance(v, torch.Tensor):
        return v.reshape(-1)[r].item() if v.numel() > r else default
    if isinstance(v, (list, tuple)):
        return v[r] if len(v) > r else default
    return v


class HostSampler:
    """vLLM-semantics sampling over each row's top-K candidates."""

    def __init__(self, V: int):
        self.V = V
        self.gens: dict[int, tuple[int, torch.Generator]] = {}

    def forget(self, row: int):
        self.gens.pop(row, None)

    def _gen(self, row, seed):
        if seed is None or int(seed) < 0:
            return None
        g = self.gens.get(row)
        if g is None or g[0] != int(seed):
            gen = torch.Generator()
            gen.manual_seed(int(seed))
            self.gens[row] = g = (int(seed), gen)
        return g[1]

    def sample(self, idx, vals, rows, sp, output_tokens=None, prompt_tokens=None):
        """idx / vals [W, K] by slot; rows: list of (vLLM row r, slot s). Returns {r: token}."""
        out = {}
        for r, s in rows:
            ids, lg = idx[s].clone(), vals[s].double().clone()
            ok = ids < self.V  # the LM head's zero-padded vocab columns are never sampled
            ids, lg = ids[ok], lg[ok]
            rep = float(_param(sp, "repetition_penalty", r, 1.0))
            fre = float(_param(sp, "frequency_penalty", r, 0.0))
            pre = float(_param(sp, "presence_penalty", r, 0.0))
            if (rep != 1.0 or fre != 0.0 or pre != 0.0) and output_tokens is not None:
                outs = output_tokens[r][output_tokens[r] >= 0].to(torch.int64)
                cnt = (ids[:, None] == outs[None, :]).sum(-1).double()
                if rep != 1.0:
                    seen = cnt > 0
                    if prompt_tokens is not None:
                        pr = prompt_tokens[r][prompt_tokens[r] >= 0].to(torch.int64)
                        seen = seen | (ids[:, None] == pr[None, :]).any(-1)
                    lg = torch.where(seen, torch.where(lg > 0, lg / rep, lg * rep), lg)
                lg = lg - fre * cnt - pre * (cnt > 0).double()
            temp = float(_param(sp, "temperature", r, 0.0))
            if temp <= 0.0:
                out[r] = int(ids[int(torch.argmax(lg))])  # ties: the lower index (candidates sorted descending)
                continue
            lg = lg / temp
            order = torch.argsort(lg, descending=True)
            ids, lg = ids[order], lg[order]
            tk = int(_param(sp, "top_k", r, -1))
            if 0 < tk < lg.numel():
                ids, lg = ids[:tk], lg[:tk]
            p = torch.softmax(lg, -1)
            tp = float(_param(sp, "top_p", r, 1.0))
            if 0.0 < tp < 1.0:
                keep = torch.cumsum(p, -1) - p < tp
                keep[0] = True
                ids, p = ids[keep], p[keep] / p[keep].sum()
            gen = self._gen(r, _param(sp, "seed", r, None))
            out[r] = int(ids[int(torch.multinomial(p, 1, generator=gen))])
        return out
