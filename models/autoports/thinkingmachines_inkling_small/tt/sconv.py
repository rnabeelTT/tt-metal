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

import os
from pathlib import Path

import torch

import ttnn

K = 4  # sconv_kernel_size
KERNELS = Path(__file__).parent / "sconv_kernels"
# INKLING_SCONV_KERNEL (default 1): rows of up to FUSED_MAX_ROWS (verify / decode) through one fused program per conv
# (sconv_kernels/: the shifted rows built in L1, the taps in FP32 DST; ~15 ttnn ops -> 1) and the commit through one
# more; longer row sets (prefill chunks) keep the ttnn ops
FUSED = os.environ.get("INKLING_SCONV_KERNEL", "1") == "1"
FUSED_MAX_ROWS = 29  # 3 carried rows + R rows in one tile


def _range_set(cores):
    rs = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
    for c in cores[1:]:
        rs = rs.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
    return rs


def _cores(mesh, n):
    grid = mesh.compute_with_storage_grid_size()
    assert n <= grid.x * grid.y
    return [ttnn.CoreCoord(i // grid.y, i % grid.y) for i in range(n)]


def _cb(cores, index, dtype, page, pages):
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
    return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores, format_descriptors=[fmt])


def _rt(cores, args):
    rt = ttnn.RuntimeArgs()
    for c, a in zip(cores, args):
        rt[c.x][c.y] = a
    return rt


def _dm(source, cores, ct, rt, riscv, noc):
    return ttnn.KernelDescriptor(kernel_source=str(KERNELS / source), core_ranges=_range_set(cores),
                                 compile_time_args=ct, runtime_args=_rt(cores, rt),
                                 config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))


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
        self.fused = FUSED and c_local % 32 == 0
        if self.fused:  # the taps as full bf16 tiles per die, [4, 1, 32, C]: every row of tap j's tile holds tap row j
            tf = w.float().T.reshape(K, 1, 1, -1).expand(K, 1, 32, -1).contiguous()
            self.taps_tiles = ttnn.from_torch(tf.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                              device=mesh, mesh_mapper=mesh_mapper)
            self.cols = c_local // 32
            self.cores = _cores(mesh, self.cols)
        self._seq_tile = None

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

    def _fused_ok(self, x):
        return self.fused and x.dtype == ttnn.bfloat16 and int(x.shape[2]) <= FUSED_MAX_ROWS

    def _rows_kernel(self, x, commit_all):
        """One program: y [1, 1, R, C] bf16, and (unless commit_all) the seq tile [s0, s1, s2, x...] for commit_rows;
        commit_all advances the state past all R rows inside the program."""
        R = int(x.shape[2])
        mesh = self.mesh
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        # DRAM, not L1: a trace-owned L1 output would lower every core's L1 ceiling for the programs after the capture
        y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, R, self.C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh,
                                           ttnn.DRAM_MEMORY_CONFIG)
        seq = None if commit_all else ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 32, self.C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        cores, cr, cols = self.cores, _range_set(self.cores), self.cols
        cb_x, cb_s, cb_t, cb_scr, cb_o = 0, 1, 2, 3, 16
        cbs = [_cb(cr, cb_x, ttnn.bfloat16, 2048, 2), _cb(cr, cb_s, ttnn.bfloat16, 2048, 6),
               _cb(cr, cb_t, ttnn.bfloat16, 2048, 8), _cb(cr, cb_scr, ttnn.float32, 768, 1),
               _cb(cr, cb_o, ttnn.bfloat16, 2048, 2)]
        seq_t = seq if seq is not None else y  # the accessor needs a tensor; seq_addr 0 disables the write
        rd_ct = [cb_x, cb_s, cb_t, cb_scr, cols, R, int(commit_all)]
        for t in (x, self.st, self.taps_tiles, seq_t):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        seq_addr = seq.buffer_address() if seq is not None else 0
        rd_rt = [[x.buffer_address(), self.st.buffer_address(), self.taps_tiles.buffer_address(), seq_addr, i, 1]
                 for i in range(cols)]
        wr_ct = [cb_o] + ttnn.TensorAccessorArgs(y).get_compile_time_args()
        wr_rt = [[y.buffer_address(), i, 1] for i in range(cols)]
        kernels = [
            _dm("reader.cpp", cores, rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            _dm("writer.cpp", cores, wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "compute.cpp"), core_ranges=cr, compile_time_args=[cb_x, cb_s, cb_t, cb_o],
                runtime_args=_rt(cores, [[1]] * cols),
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                    math_approx_mode=False),
            ),
        ]
        io = [x, self.st, self.taps_tiles] + ([seq] if seq is not None else []) + [y]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return y, seq

    def _commit_kernel(self, seq, n):
        cores, cr, cols = self.cores, _range_set(self.cores), self.cols
        cb_scr = 0
        ct = [cb_scr, cols, int(n)]
        for t in (seq, self.st):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = [[seq.buffer_address(), self.st.buffer_address(), i, 1] for i in range(cols)]
        k = _dm("commit.cpp", cores, ct, rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)
        ttnn.generic_op([seq, self.st], ttnn.ProgramDescriptor(kernels=[k], semaphores=[],
                                                               cbs=[_cb(cr, cb_scr, ttnn.float32, 2048 + 384, 1)]))

    def decode(self, x):
        """x [1, 1, 1, C] (bf16 or fp32) -> y same dtype; advances the state by one token (in place: trace-safe)."""
        if self._fused_ok(x):
            return self._rows_kernel(x, commit_all=True)[0]
        y, seq = self._conv(x, x.dtype)
        ttnn.copy(ttnn.slice(seq, [0, 0, 1, 0], [1, 1, K, self.C]), self.st)
        return y

    def rows(self, x, commit_all: bool = True):
        """x [1, 1, R, C] at R consecutive positions -> y [1, 1, R, C]. With commit_all the state advances past all R
        rows (prefill segments); otherwise the caller commits the accepted prefix with commit_rows()."""
        if self._fused_ok(x):
            y, seq = self._rows_kernel(x, commit_all)
            if seq is not None:
                self._last_seq, self._seq_tile = seq, True
            return y
        y, seq = self._conv(x, x.dtype)
        self._last_seq, self._seq_tile = seq, False
        if commit_all:
            self.commit_rows(int(x.shape[2]))
        return y

    def commit_rows(self, n: int):
        """State := the 3 inputs ending at row n-1 of the last rows() call (host-known n: eager paths)."""
        seq = self._last_seq  # rows: [s0, s1, s2, x_0 .. x_{R-1}]
        if self._seq_tile:
            self._commit_kernel(seq, n)
            return
        ttnn.copy(ttnn.slice(seq, [0, 0, n, 0], [1, 1, n + K - 1, self.C]), self.st)
