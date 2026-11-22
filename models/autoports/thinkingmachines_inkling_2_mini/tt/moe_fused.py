# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2 decode MoE through the shared qwen36 fused MoE decode kernels (models/demos/blackhole/qwen36/tt/moe_decode:
router top-k + gate|up + activation + down in ~3 programs, expert weights streamed per DRAM bank).

  * router "sigb": sigmoid scores, top-k of score + selection bias, weights normalized to 1 (route_scale 8 folded
    into the down weights); the logits come from this port's fp32 router matmul (tt/moe.py MoE.router_w).
  * activation: the kernels' ACT_INKLING path (QWEN35_MOE_ACT=inkling, tt/__init__.py): act = tanh(g/4) s(g)
    tanh(u'), u' = u / 25 folded into the up weights, the factor 100 folded into the down weights.
  * the kernels take a die's expert slice of 128..768 columns (moe_decode/fused.py supported()): the die's 1152 / N
    columns run as sub-shards (N = 1: 3 x 384; N = 2: 384 + 192 padded to 256; N = 4: 288 padded to 384; zero padding
    is exact, act(0, 0) = 0), each with its own weights, the partial outputs summed.
  * HiFi2 matmuls (QWEN35_MOE_DECODE_FIDELITY=hifi2): the full bfp8 weight, no 5-bit rounding.
Output: this die's partial sum [1, 1, M, H] bf16 (the caller all-reduces), times 1/sqrt(5) (the post-MLP conv scale,
as tt/moe.py).
"""

from __future__ import annotations

import math
import os

import torch

import ttnn

from .dmm import dmm
from models.autoports.thinkingmachines_inkling_2_mini.tt.wload import cached

ROW1 = os.environ.get("I2M_MOE_ROW1", "0") == "1"
# I2M_MOE_DUPROW (default 1): one decode row duplicated into its tiles' padding row in place (tt/duprow.py), not concat
DUPROW = os.environ.get("I2M_MOE_DUPROW", "1") == "1"
_SHARED = __import__("types").SimpleNamespace()  # the fused ops / buffers cache, shared by every layer (one set)
SUBS = {1: [(384, 384)] * 3, 2: [(384, 384), (192, 256)], 4: [(288, 384)]}  # (real, stored) widths per sub-shard
# I2M_MOE_SUBS1 (one die): "768,384" -> two sub-shards (m = 6 and 3 column tiles per bank: both supported) instead of 3
if os.environ.get("I2M_MOE_SUBS1"):
    SUBS[1] = [(int(w), int(w)) for w in os.environ["I2M_MOE_SUBS1"].split(",")]


class FusedExperts:
    def __init__(self, moe, ck, li, dtype=ttnn.bfloat8_b, cache_dir=None, rows_max=None):
        from models.demos.blackhole.qwen36.tt.moe_decode.fused import FusedMoEDecode

        self.moe, self.mesh = moe, moe.mesh
        N, E, H, I, Il = moe.N, moe.E, moe.H, moe.I, moe.Il
        self.N, self.H = N, H
        subs = SUBS[N]
        assert sum(r for r, _ in subs) == Il, (subs, Il)
        mm = (lambda d: ttnn.ShardTensorToMesh(self.mesh, dim=d)) if N > 1 else (lambda d: None)
        dt = {ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}[dtype]
        cname = (lambda n: os.path.join(cache_dir, f"L{li}.fused.{n}.{dt}.tp{N}")) if cache_dir else (lambda n: None)
        w13 = w2 = None
        self.subs = []
        self.widths = subs
        off = 0
        for si, (real, W) in enumerate(subs):
            def make_ug(off=off, real=real, W=W):
                nonlocal w13
                if w13 is None:
                    w13 = ck.experts(li, "w13")  # [E, 2I, H] rows [gate | up]
                cols = []
                for d in range(N):
                    a = d * Il + off
                    g = torch.zeros(E, H, W)
                    u = torch.zeros(E, H, W)
                    g[:, :, :real] = w13[:, a : a + real].transpose(1, 2)
                    u[:, :, :real] = w13[:, I + a : I + a + real].transpose(1, 2) / 25.0
                    cols.append(torch.cat([g, u], dim=-1))  # [gate | up] per die (gate_first)
                return torch.cat(cols, -1).unsqueeze(0).to(torch.bfloat16)  # [1, E, H, N * 2W]

            def make_dn(off=off, real=real, W=W):
                nonlocal w2
                if w2 is None:
                    w2 = ck.experts(li, "w2")  # [E, H, I]
                rows = []
                for d in range(N):
                    a = d * Il + off
                    t = torch.zeros(E, W, H)
                    t[:, :real] = w2[:, :, a : a + real].transpose(1, 2) * (100.0 * moe.route_scale / math.sqrt(5.0))
                    rows.append(t)
                return torch.cat(rows, 1).unsqueeze(0).to(torch.bfloat16)  # [1, E, N * W, H]

            tag = f"{si}" if subs == [(384, 384)] * 3 or N > 1 else f"{si}o{off}w{W}"  # (old names for the old split)
            ug = cached(make_ug, dtype, ttnn.TILE_LAYOUT, self.mesh, cname(f"ug{tag}"), mesh_mapper=mm(-1))
            dn = cached(make_dn, dtype, ttnn.TILE_LAYOUT, self.mesh, cname(f"dn{tag}"), mesh_mapper=mm(-2))
            ckc = moe.rckc
            op = FusedMoEDecode(
                self.mesh, _SHARED, hidden=H, inter_local=W, num_experts=E, k=moe.k, tp=N, dtype=dtype,
                router_weight=None, shared_gate_weight=None, shared_gate_proj=None, shared_up_proj=None,
                shared_down_proj=None, cache_name=lambda n: None, router_ckc=ckc, router="sigb",
                selection_bias=ck.L(li, "mlp.gate.bias").float(), router_logits=self._logits, gate_first=True,
            )
            self.subs.append((op, ug, dn))
            off += real
        del w13, w2
        # I2M_MOE_ROWS_MAX: the widest decode batch through one fused call (32: every expert of the union read once;
        # 16: two calls above 16 rows)
        self.rows_max = int(os.environ.get("I2M_MOE_ROWS_MAX", "16")) if rows_max is None else rows_max

    def _logits(self, x):
        xf = ttnn.typecast(x, ttnn.float32)
        return dmm(xf, self.moe.router_w, self.moe.rckc, dtype=ttnn.float32)

    def build_rows(self, widths):
        """Build the rows ops of every decode width before any trace capture (host uploads)."""
        for op, _, _ in self.subs:
            for w in widths:
                w = w if ROW1 else max(w, 2)
                if w <= self.rows_max:
                    op.rows_ops(w)

    def forward(self, x, logits=None, dup=False, raw=False):
        """x [1, 1, M, H] bf16 (normed), M <= rows_max (3 down readers per bank: 16 rows; wider batches stay on
        tt/moe.py) -> this die's partial [1, 1, M, H] bf16 (summed sub-shards)."""
        assert int(x.shape[-2]) <= self.rows_max
        M = int(x.shape[-2])
        # one row: the 2-row path with the row twice (the single-row kernels' sigb routing is wrong here: pcc 0.5);
        # I2M_MOE_ROW1=1: the rows kernels at one row instead (no concat / slice / clone)
        if M == 1 and not ROW1:
            if dup:  # (row 1 already = row 0: written by the producer)
                xr, m = x, 2
            elif DUPROW:  # row 1 of x's tiles (padding) <- row 0, in place: the 2-row path reads both
                if getattr(_SHARED, "duprow", None) is None:
                    from .duprow import DupRow
                    _SHARED.duprow = DupRow(self.mesh, self.H)
                xr, m = _SHARED.duprow(x), 2
            else:
                xr, m = ttnn.concat([x, x], dim=2), 2
        else:
            xr, m = x, M
        from models.demos.blackhole.qwen36.tt.moe_decode.union import act_buffer

        op0 = self.subs[0][0]
        route, _, _, b0 = op0.rows_ops(m)
        own = logits is None
        logits = self._logits(xr) if own else logits
        route(logits, b0.table, bias=op0.bias)  # ONE routing per layer: every sub-shard reads this table
        ttnn.deallocate(logits)
        acc = None
        for (op, ug, dn), (_, W) in zip(self.subs, self.widths):
            _, gu, dnop, b = op.rows_ops(m)
            act = act_buffer(self.mesh, W, self.moe.E, self.moe.k, m)
            gu(xr, b0.table, b0.identity, ug, ug, act, ep_offset=None)
            dnop(act, b0.table, dn, dn, b.out, ep_offset=None)
            ttnn.deallocate(act)
            if acc is None:  # (one sub-shard: the shared out buffer itself, consumed before the next layer's MoE)
                acc = b.out if len(self.subs) == 1 and (ROW1 or raw) else ttnn.clone(b.out)
            else:
                acc = ttnn.add(acc, b.out)
        if raw and len(self.subs) == 1:  # (the shared out buffer, m rows: the caller reads its first M right away)
            return acc
        if M == 1 and not ROW1:
            acc = ttnn.slice(acc, [0, 0, 0, 0], [1, 1, 1, self.H])
        return acc
