# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling's short convolution ("sconv") on TT: y = x + causal_depthwise_conv1d(x, w[C, 4]), computed in fp32
(transformers `InklingShortConvolution`: the input is upcast, the conv and the residual add run in fp32, the result is
cast back to the input dtype).

Every layer has four: on K and V right after their projections (before k_norm and the KV cache), on o_proj's output
and on the MLP/MoE output (each before the residual add). All four are depthwise, so they run on whatever channel
shard a die holds: K / V on the die's kv heads (column-parallel QKV), the two hidden-size ones on the die's slice of
the fractured residual (after the reduce-scatter). No collective is added.

Output at token t:  y_t = x_t + w0 x_{t-3} + w1 x_{t-2} + w2 x_{t-1} + w3 x_t   (w = the checkpoint's [C, 4])

State: the last 3 inputs (x_{t-3}, x_{t-2}, x_{t-1}), zero at the start of a sequence, kept as ONE row-major fp32
[1, 1, 3, C] tensor: rows() concatenates it with the new rows directly and a commit is one slice + one copy (the
per-row tile states cost 9 ops per commit and 3 conversions per rows() call, x 4 convs x 42 layers per verify pass).

  decode(x)          one row: shift register in place (trace-safe), returns y
  rows(x)            R rows at consecutive positions (spec verify / prefill segment) against the state; returns y and
                     keeps the R inputs so commit(n) can set the state to the inputs ending at accepted row n-1
  commit_host(n)     host-side commit (eager paths); the traced commit uses a device index (see commit())
"""

from __future__ import annotations

import torch

import ttnn

K = 4  # sconv_kernel_size


def host_reference(x: torch.Tensor, w: torch.Tensor, state: torch.Tensor | None = None):
    """fp32 reference: x [T, C], w [C, K], state [K-1, C] or None -> (y [T, C], new state [K-1, C])."""
    x = x.float()
    s = torch.zeros(K - 1, x.shape[1]) if state is None else state.float()
    seq = torch.cat([s, x], 0)  # [K-1+T, C]
    T = x.shape[0]
    y = x.clone()
    for j in range(K):
        y += w[:, j].float() * seq[j : j + T]
    return y, seq[-(K - 1) :].clone()


class ShortConv:
    """One sconv on a mesh. `w` [C, K] is the full checkpoint weight; `shard(w)` -> the [C_local, K] block each die
    holds is the caller's mapping (a mesh mapper), so the same class serves the K/V and the hidden-size convs."""

    def __init__(self, mesh, w: torch.Tensor, mesh_mapper, c_local: int, max_rows: int = 32):
        self.mesh = mesh
        self.C = c_local
        self.max_rows = max_rows
        # taps as [1, 1, 1, C] fp32 row vectors, sharded like the activations (broadcast over rows)
        self.taps = [
            ttnn.from_torch(
                w[:, j].float().reshape(1, 1, 1, -1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=mesh_mapper,
            )
            for j in range(K)
        ]
        z = lambda: ttnn.from_torch(
            torch.zeros(1, 1, K - 1, w.shape[0]),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=mesh_mapper,
        )
        self.st = z()  # rows x_{t-3}, x_{t-2}, x_{t-1}
        self.state = [self.st]  # (the persistent state tensors, for callers that save / restore them)
        self._zero = z()

    def reset(self):
        ttnn.copy(self._zero, self.st)

    def decode(self, x):
        """x [1, 1, 1, C] (bf16 or fp32) -> y same dtype; advances the state by one token (same arithmetic as before
        the single-tensor state: y = x + w3 x + w0 s0 + w1 s1 + w2 s2)."""
        dt = x.dtype
        xf = ttnn.typecast(x, ttnn.float32) if dt != ttnn.float32 else x
        y = ttnn.mac(xf, self.taps[K - 1], xf)  # x + w3 x
        for j in range(K - 1):
            sj = ttnn.to_layout(ttnn.slice(self.st, [0, 0, j, 0], [1, 1, j + 1, self.C]), ttnn.TILE_LAYOUT)
            y = ttnn.mac(sj, self.taps[j], y)
        seq = ttnn.concat([self.st, ttnn.to_layout(xf, ttnn.ROW_MAJOR_LAYOUT)], dim=2)
        ttnn.copy(ttnn.slice(seq, [0, 0, 1, 0], [1, 1, K, self.C]), self.st)
        out = ttnn.typecast(y, dt) if dt != ttnn.float32 else y
        return out

    def rows(self, x, commit_all: bool = True):
        """x [1, 1, R, C] at R consecutive positions -> y [1, 1, R, C]. With commit_all the state advances past all R
        rows (prefill segments); otherwise the caller commits the accepted prefix with commit_rows()."""
        dt = x.dtype
        R = int(x.shape[2])
        xt = ttnn.typecast(x, ttnn.float32) if dt != ttnn.float32 else x  # tile
        seq = ttnn.concat([self.st, ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT)], dim=2)  # [1, 1, K-1+R, C]
        y = None
        for j in range(K - 1):
            sl = ttnn.to_layout(ttnn.slice(seq, [0, 0, j, 0], [1, 1, j + R, self.C]), ttnn.TILE_LAYOUT)
            y = ttnn.multiply(sl, self.taps[j]) if y is None else ttnn.mac(sl, self.taps[j], y)
        y = ttnn.mac(xt, self.taps[K - 1], y)  # rows K-1.. of seq are x itself: the same values, no slice
        y = ttnn.add(y, xt)
        self._last_seq = seq
        if commit_all:
            self.commit_rows(R)
        return ttnn.typecast(y, dt) if dt != ttnn.float32 else y

    def commit_rows(self, n: int):
        """State := the 3 inputs ending at row n-1 of the last rows() call (host-known n: eager paths)."""
        seq = self._last_seq  # rows: [s0, s1, s2, x_0 .. x_{R-1}]
        ttnn.copy(ttnn.slice(seq, [0, 0, n, 0], [1, 1, n + K - 1, self.C]), self.st)
