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
        # the same taps row-major: the tap arithmetic runs on row-major rows (see _conv)
        self.taps_rm = [
            ttnn.from_torch(
                w[:, j].float().reshape(1, 1, 1, -1),
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
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

    def _conv(self, x, out_dtype=ttnn.float32):
        """x [1, 1, R, C] (tile, any float dtype) -> (y tile [1, 1, R, C] in out_dtype, seq = [s0, s1, s2, x_0 .. x_{R-1}]
        row-major fp32). The tap arithmetic runs on row-major rows, in the host reference's order
        (y = x + w0 seq[0:R] + w1 seq[1:R+1] + w2 seq[2:R+2] + w3 x), and only y is tilized: a tilize of a one-row fp32
        tile runs on one core (~65 us at C = 1024 per die); slicing each shifted tap and tilizing it cost three per
        conv, ~0.6 ms of a ~1.25 ms decode layer. Row-major slices / mul / add are a few us each and exact."""
        R = int(x.shape[2])
        xf = ttnn.typecast(x, ttnn.float32) if x.dtype != ttnn.float32 else x
        xr = ttnn.to_layout(xf, ttnn.ROW_MAJOR_LAYOUT)
        seq = ttnn.concat([self.st, xr], dim=2)  # [1, 1, K-1+R, C]
        y = xr
        for j in range(K - 1):
            y = ttnn.add(y, ttnn.multiply(ttnn.slice(seq, [0, 0, j, 0], [1, 1, j + R, self.C]), self.taps_rm[j]))
        y = ttnn.add(y, ttnn.multiply(xr, self.taps_rm[K - 1]))  # rows K-1.. of seq are x itself
        if out_dtype != ttnn.float32:  # cast before the tilize: same values, half the bytes (~80 -> ~42 us at 1024)
            y = ttnn.typecast(y, out_dtype)
        return ttnn.to_layout(y, ttnn.TILE_LAYOUT), seq

    def decode(self, x):
        """x [1, 1, 1, C] (bf16 or fp32) -> y same dtype; advances the state by one token (in place: trace-safe)."""
        y, seq = self._conv(x, x.dtype)
        ttnn.copy(ttnn.slice(seq, [0, 0, 1, 0], [1, 1, K, self.C]), self.st)
        return y

    def rows(self, x, commit_all: bool = True):
        """x [1, 1, R, C] at R consecutive positions -> y [1, 1, R, C]. With commit_all the state advances past all R
        rows (prefill segments); otherwise the caller commits the accepted prefix with commit_rows()."""
        y, seq = self._conv(x, x.dtype)
        self._last_seq = seq
        if commit_all:
            self.commit_rows(int(x.shape[2]))
        return y

    def commit_rows(self, n: int):
        """State := the 3 inputs ending at row n-1 of the last rows() call (host-known n: eager paths)."""
        seq = self._last_seq  # rows: [s0, s1, s2, x_0 .. x_{R-1}]
        ttnn.copy(ttnn.slice(seq, [0, 0, n, 0], [1, 1, n + K - 1, self.C]), self.st)
