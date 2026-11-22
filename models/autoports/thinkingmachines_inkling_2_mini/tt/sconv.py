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
PF_MAXSEG = 8  # sequences per packed prefill chunk (sconv_kernels/pfseg_reader.cpp MAXSEG)
# INKLING_SCONV_PF_KERNEL (default 1): a prefill chunk (R % 32 == 0 rows, bf16) through one program over the row tiles
# (sconv_kernels/pf_*.cpp: the same arithmetic as the fused rows kernel, the carried rows passed between tiles) plus a
# state copy, instead of ~15 ttnn ops per conv
PF_KERNEL = os.environ.get("INKLING_SCONV_PF_KERNEL", "1") == "1"
# INKLING_SCONV_COMMIT_RM (default 1): a prefill chunk's commit through sconv_kernels/commit_rm.cpp (offset at run time)
COMMIT_RM = os.environ.get("INKLING_SCONV_COMMIT_RM", "1") == "1"


def _stable_hash(*key):
    """A program-cache key that is the same in every process (Python's hash() is salted per process and signed)."""
    import hashlib

    return int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)


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
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / source),
        core_ranges=_range_set(cores),
        compile_time_args=ct,
        runtime_args=_rt(cores, rt),
        config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc),
    )


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
        self._pf_stn = None  # the prefill kernel's new-state rows (prefill_commit), allocated with its first program
        self.fused = FUSED and c_local % 32 == 0
        if self.fused:  # the taps as full bf16 tiles per die, [4, 1, 32, C]: every row of tap j's tile holds tap row j
            tf = w.float().T.reshape(K, 1, 1, -1).expand(K, 1, 32, -1).contiguous()
            self.taps_tiles = ttnn.from_torch(
                tf.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=mesh_mapper,
            )
            self.cols = c_local // 32
            # one core per 32-column tile, or (more columns than cores: the full-width replicated convs) cpc tiles per
            # core; the kernels take (first column, count) per core
            grid = mesh.compute_with_storage_grid_size()
            cap = grid.x * grid.y
            ncores = self.cols if self.cols <= cap else max(d for d in range(1, cap + 1) if self.cols % d == 0)
            self.cpc = self.cols // ncores
            self.cores = _cores(mesh, ncores)
        self._seq_tile = None

    snap_row = None  # prefix cache (tt/prefix.py): set for the prefill chunk holding a snapshot position
    snap_out = None

    def reset(self):
        ttnn.copy(self._zero, self.st)

    def _state_at(self, x, r):
        """The state after row r - 1 of chunk x (r = 0: the current state): x's rows r-3 .. r-1 as fp32 row-major."""
        if r == 0:
            return ttnn.clone(self.st)
        assert r % 32 == 0 and r >= 32, r
        t = ttnn.slice(x, [0, 0, r - 32, 0], [1, 1, r, self.C])
        t = ttnn.to_layout(ttnn.typecast(t, ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.slice(t, [0, 0, 32 - (K - 1), 0], [1, 1, 32, self.C])

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
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, R, self.C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        seq = (
            None
            if commit_all
            else ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 32, self.C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
        )
        cores, cr, cols = self.cores, _range_set(self.cores), self.cols
        cb_x, cb_s, cb_t, cb_scr, cb_o = 0, 1, 2, 3, 16
        cbs = [
            _cb(cr, cb_x, ttnn.bfloat16, 2048, 2),
            _cb(cr, cb_s, ttnn.bfloat16, 2048, 6),
            _cb(cr, cb_t, ttnn.bfloat16, 2048, 8),
            _cb(cr, cb_scr, ttnn.float32, 768, 1),
            _cb(cr, cb_o, ttnn.bfloat16, 2048, 2),
        ]
        seq_t = seq if seq is not None else y  # the accessor needs a tensor; seq_addr 0 disables the write
        rd_ct = [cb_x, cb_s, cb_t, cb_scr, cols, R, int(commit_all)]
        for t in (x, self.st, self.taps_tiles, seq_t):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        seq_addr = seq.buffer_address() if seq is not None else 0
        cpc = self.cpc
        rd_rt = [
            [x.buffer_address(), self.st.buffer_address(), self.taps_tiles.buffer_address(), seq_addr, i * cpc, cpc]
            for i in range(len(cores))
        ]
        wr_ct = [cb_o] + ttnn.TensorAccessorArgs(y).get_compile_time_args()
        wr_rt = [[y.buffer_address(), i * cpc, cpc] for i in range(len(cores))]
        kernels = [
            _dm("reader.cpp", cores, rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            _dm("writer.cpp", cores, wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "compute.cpp"),
                core_ranges=cr,
                compile_time_args=[cb_x, cb_s, cb_t, cb_o],
                runtime_args=_rt(cores, [[cpc]] * len(cores)),
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        io = [x, self.st, self.taps_tiles] + ([seq] if seq is not None else []) + [y]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return y, seq

    def _pf_ok(self, x):
        R = int(x.shape[2])
        return (
            self.fused
            and PF_KERNEL
            and x.dtype == ttnn.bfloat16
            and x.layout == ttnn.TILE_LAYOUT
            and R > FUSED_MAX_ROWS
            and R % 32 == 0
            and not x.is_sharded()
        )

    def _pf_program(self, R):
        """The prefill conv program for R rows (built once per R; the per-call addresses and the commit row count are
        common runtime arguments, so a call only rewrites those)."""
        P = getattr(self, "_pf_progs", None)
        if P is None:
            P = self._pf_progs = {}
        if R in P:
            return P[R]
        grid = self.mesh.compute_with_storage_grid_size()
        gx, gy = int(grid.x), int(grid.y)
        cols, Rt = self.cols, R // 32
        G = max(1, min(Rt, (gx * gy) // cols))
        n_items = cols * G
        cores = [ttnn.CoreCoord(i // gy, i % gy) for i in range(n_items)]
        cr = _range_set(cores)
        if self._pf_stn is None:
            self._pf_stn = ttnn.from_torch(
                torch.zeros(1, 1, K - 1, self.C * self.mesh.get_num_devices()),
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=3),
            )
        cb_x, cb_s, cb_t, cb_scr, cb_o = 0, 1, 2, 3, 16
        cbs = [
            _cb(cr, cb_x, ttnn.bfloat16, 2048, 2),
            _cb(cr, cb_s, ttnn.bfloat16, 2048, 6),
            _cb(cr, cb_t, ttnn.bfloat16, 2048, 4),
            _cb(cr, cb_scr, ttnn.float32, 768 + 2048, 1),
            _cb(cr, cb_o, ttnn.bfloat16, 2048, 2),
        ]
        P[R] = dict(cr=cr, cbs=cbs, ct=(cols, Rt, G, gy), hash=_stable_hash("inkling_sconv_pf", self.C, R, G, gy))
        return P[R]

    def prefill_commit(self, x, n=None):
        """A prefill chunk: x [1, 1, R, C] bf16 tile (R % 32 == 0) -> y [1, 1, R, C] bf16; the state advances past the
        first n rows (default R: a padded last chunk passes its real row count). One program + a state copy."""
        R = int(x.shape[2])
        n = R if n is None else int(n)
        if self.snap_row is not None:  # prefix cache: the state at row snap_row of this chunk, before it advances
            self.snap_out = self._state_at(x, int(self.snap_row))
        if not self._pf_ok(x) or n < 1:
            y = self.rows(x, commit_all=(n == R))
            if n != R:
                self.commit_final(n)
            return y
        pr = self._pf_program(R)
        cols, Rt, G, gy = pr["ct"]
        cr, stn = pr["cr"], self._pf_stn
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, R, self.C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        cb_x, cb_s, cb_t, cb_scr, cb_o = 0, 1, 2, 3, 16
        rd_ct = [cb_x, cb_s, cb_t, cb_scr, cols, Rt, G, gy]
        for t in (x, self.st, self.taps_tiles, stn):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = [cb_o, cols, Rt, G, gy] + ttnn.TensorAccessorArgs(y).get_compile_time_args()

        def kd(source, ct, common, cfg):
            k = ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / source), core_ranges=cr, compile_time_args=ct, config=cfg
            )
            k.common_runtime_args = common
            return k

        kernels = [
            kd(
                "pf_reader.cpp",
                rd_ct,
                [
                    x.buffer_address(),
                    self.st.buffer_address(),
                    self.taps_tiles.buffer_address(),
                    stn.buffer_address(),
                    n,
                ],
                ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            kd(
                "pf_writer.cpp",
                wr_ct,
                [y.buffer_address()],
                ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            kd(
                "pf_compute.cpp",
                [cb_x, cb_s, cb_t, cb_o, cols, Rt, G, gy],
                [],
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=pr["cbs"])
        prog.custom_program_hash = pr["hash"]
        ttnn.generic_op([x, self.st, self.taps_tiles, stn, y], prog)
        ttnn.copy(stn, self.st)
        self._last_seq = None
        return y

    def prefill_packed(self, x, segs):
        """A packed prefill chunk (several sequences): x [1, 1, T, C] bf16 tile -> y [1, 1, T, C] bf16 in one program
        (sconv_kernels/pfseg_reader.cpp; the program depends on T only). segs: [(row0, R, n, lane_conv)] (64-row
        aligned, covering all T rows): segment rows run against lane_conv's state, which then advances past the
        segment's first n rows; a lane conv with snap_row set also gets the state at that row in its snap_buf."""
        T = int(x.shape[2])
        assert self._pf_ok(x) and len(segs) <= PF_MAXSEG, (T, len(segs))
        pr = self._pf_program(T)
        cols, Rt, G, gy = pr["ct"]
        cr = pr["cr"]
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, T, self.C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        NONE = 0xFFFFFFFF
        args = [x.buffer_address(), self.taps_tiles.buffer_address(), len(segs)]
        for row0, R, n, lc in segs:
            assert row0 % 32 == 0 and R % 32 == 0
            snap = lc.snap_row
            args += [row0 // 32, (row0 + R) // 32, lc.st.buffer_address(), lc._pf_stn.buffer_address(), int(n),
                     lc.snap_buf.buffer_address() if snap is not None else 0, int(snap) if snap is not None else NONE]
        args += [0] * (3 + 7 * PF_MAXSEG - len(args))
        cb_x, cb_s, cb_t, cb_scr, cb_o = 0, 1, 2, 3, 16
        rd_ct = [cb_x, cb_s, cb_t, cb_scr, cols, Rt, G, gy]
        for t in (x, self.st, self.taps_tiles):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = [cb_o, cols, Rt, G, gy] + ttnn.TensorAccessorArgs(y).get_compile_time_args()

        def kd(source, ct, common, cfg):
            k = ttnn.KernelDescriptor(kernel_source=str(KERNELS / source), core_ranges=cr, compile_time_args=ct,
                                      config=cfg)
            k.common_runtime_args = common
            return k

        kernels = [
            kd("pfseg_reader.cpp", rd_ct, args,
               ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)),
            kd("pf_writer.cpp", wr_ct, [y.buffer_address()],
               ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)),
            kd("pf_compute.cpp", [cb_x, cb_s, cb_t, cb_o, cols, Rt, G, gy], [],
               ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                            math_approx_mode=False)),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=pr["cbs"])
        prog.custom_program_hash = _stable_hash("inkling_sconv_pfseg", self.C, T, G, gy)
        ttnn.generic_op([x, self.st, self.taps_tiles, y], prog)
        for row0, R, n, lc in segs:
            ttnn.copy(lc._pf_stn, lc.st)
            if lc.snap_row is not None:
                lc.snap_out = lc.snap_buf
        return y

    def _commit_kernel(self, seq, n):
        cores, cr, cols = self.cores, _range_set(self.cores), self.cols
        cb_scr = 0
        ct = [cb_scr, cols, int(n)]
        for t in (seq, self.st):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = [[seq.buffer_address(), self.st.buffer_address(), i * self.cpc, self.cpc] for i in range(len(cores))]
        k = _dm("commit.cpp", cores, ct, rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)
        ttnn.generic_op(
            [seq, self.st],
            ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(cr, cb_scr, ttnn.float32, 2048 + 384, 1)]),
        )

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
            self._last_seq = None  # committed: do not keep a prefill chunk's sequence (~4 MB per conv) alive
        return y

    def commit_final(self, n: int):
        """commit_rows(n), then release the sequence (a prefill chunk's: nothing commits from it again)."""
        self.commit_rows(n)
        self._last_seq = None

    def commit_rows(self, n: int):
        """State := the 3 inputs ending at row n-1 of the last rows() call (host-known n: eager paths)."""
        seq = self._last_seq  # rows: [s0, s1, s2, x_0 .. x_{R-1}]
        if self._seq_tile:
            self._commit_kernel(seq, n)
            return
        if COMMIT_RM and seq.layout == ttnn.ROW_MAJOR_LAYOUT and seq.dtype == ttnn.float32:
            self._commit_rm(seq, n)
            return
        ttnn.copy(ttnn.slice(seq, [0, 0, n, 0], [1, 1, n + K - 1, self.C]), self.st)

    def _commit_rm(self, seq, n):
        """commit_rows for a prefill chunk's row-major fp32 seq: one single-core program, n a runtime argument
        (sconv_kernels/commit_rm.cpp)."""
        core = ttnn.CoreCoord(0, 0)
        rowb = self.C * 4
        ct = [0, rowb]
        for t in (seq, self.st):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        k = _dm(
            "commit_rm.cpp",
            [core],
            ct,
            [[seq.buffer_address(), self.st.buffer_address(), int(n)]],
            ttnn.DataMovementProcessor.RISCV_1,
            ttnn.NOC.NOC_0,
        )
        ttnn.generic_op(
            [seq, self.st],
            ttnn.ProgramDescriptor(
                kernels=[k], semaphores=[], cbs=[_cb(_range_set([core]), 0, ttnn.float32, rowb, K - 1)]
            ),
        )


class DynShortConv:
    """Inkling-2's dynamic short convolution on K / V of the global-attention layers (transformers
    `Inkling2DynamicShortConvolution`), one sequence, fp32 like the reference:

        y_t = (x_t + sum_{L=0..K-1} (static[L, c] + dyn_t[head(c), L]) * x_{t-L}) / 3

    with dyn_t = weight_proj(h_t) [heads * K] (h = the projection's input). The 1/3 is folded into the projection
    weights by the caller (the conv is linear in x for fixed taps), so this computes y_t * 3. ``static`` [K, C_local],
    ``wproj`` [heads_local * K, H] (this die's heads). State: the last K-1 inputs, row-major fp32 [1, 1, K-1, C] like
    ShortConv (zero at a sequence start)."""

    def __init__(self, mesh, static: torch.Tensor, wproj: torch.Tensor, mesh_mapper, c_local: int, heads_local: int,
                 ckc=None):
        self.mesh = mesh
        self.C, self.nh = c_local, heads_local
        hd = c_local // heads_local
        rep = ttnn.ReplicateTensorToMesh(mesh) if mesh_mapper is None else mesh_mapper
        # static taps as [1, 1, 1, C] fp32 row-major rows (sharded like the activations)
        self.static = [
            ttnn.from_torch(static[L].float().reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=mesh, mesh_mapper=mesh_mapper)
            for L in range(K)
        ]
        # weight_proj [nh_total * K, H] -> per die [H, nh_local * K] (column shard by head); expansion matrices X_L
        # [nh_local * K, C_local] with X_L[h * K + L, h * hd + c] = 1, so dyn @ X_L broadcasts head h's lag-L weight
        # over its hd channels (exact: one nonzero per column)
        self.wproj = ttnn.from_torch(wproj.to(torch.bfloat16).T.contiguous().reshape(1, 1, wproj.shape[1], -1),
                                     dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                     mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1) if mesh.get_num_devices() > 1
                                     else None)
        X = torch.zeros(K, heads_local * K, c_local)
        for L in range(K):
            for h in range(heads_local):
                X[L, h * K + L, h * hd : (h + 1) * hd] = 1.0
        self.expand = [ttnn.from_torch(X[L].reshape(1, 1, heads_local * K, c_local), dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=mesh,
                                       mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)) for L in range(K)]
        self.ckc = ckc or ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=False)
        z = torch.zeros(1, 1, K - 1, c_local * mesh.get_num_devices())
        mk = lambda: ttnn.from_torch(z, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,  # noqa: E731
                                     mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3))
        self.st, self._zero = mk(), mk()
        self.state = [self.st]
        self._last_seq = None

    snap_row = None  # prefix cache (see ShortConv)
    snap_out = None

    def reset(self):
        ttnn.copy(self._zero, self.st)

    def rows(self, x, h, commit_n=None):
        """x [1, 1, R, C] (the projection output, any float dtype), h [1, 1, R, H] (the projection input, bf16 tile) ->
        y [1, 1, R, C] bf16 (times 3, see the class doc). The state advances past the first commit_n rows (default R;
        0: no advance, commit_rows(n) later)."""
        R = int(x.shape[2])
        # transformers: weight_proj is a bf16 Linear (its output bf16-rounded, then .float()): bf16 in, bf16 out
        hb = ttnn.typecast(h, ttnn.bfloat16) if h.dtype != ttnn.bfloat16 else h
        dyn_b = ttnn.matmul(hb, self.wproj, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)  # [1, 1, R, nh*K]
        xf = ttnn.typecast(x, ttnn.float32) if x.dtype != ttnn.float32 else x
        xr = ttnn.to_layout(xf, ttnn.ROW_MAJOR_LAYOUT)
        seq = ttnn.concat([self.st, xr], dim=2)  # [1, 1, K-1+R, C]: rows t-3, t-2, t-1, then x
        if self.snap_row is not None:  # prefix cache: the state after row snap_row - 1
            r = int(self.snap_row)
            self.snap_out = ttnn.slice(seq, [0, 0, r, 0], [1, 1, r + K - 1, self.C])
        y = xr
        for L in range(K):
            # per-token taps for lag L: static[L] + dyn expanded over the head's channels (exact), in fp32
            wl = ttnn.matmul(dyn_b, self.expand[L], compute_kernel_config=self.ckc, dtype=ttnn.float32)
            wl = ttnn.add(ttnn.to_layout(wl, ttnn.ROW_MAJOR_LAYOUT), self.static[L])
            xs = ttnn.slice(seq, [0, 0, K - 1 - L, 0], [1, 1, K - 1 - L + R, self.C])  # x_{t-L}
            y = ttnn.add(y, ttnn.multiply(xs, wl))
        ttnn.deallocate(dyn_b)
        n = R if commit_n is None else int(commit_n)
        if n:
            ttnn.copy(ttnn.slice(seq, [0, 0, n, 0], [1, 1, n + K - 1, self.C]), self.st)
            self._last_seq = None
        else:
            self._last_seq = seq
        y = ttnn.typecast(y, ttnn.bfloat16)
        return ttnn.to_layout(y, ttnn.TILE_LAYOUT)

    def commit_rows(self, n: int):
        ttnn.copy(ttnn.slice(self._last_seq, [0, 0, n, 0], [1, 1, n + K - 1, self.C]), self.st)

    def rows_packed(self, x, h, segs, bc):
        """A packed prefill chunk (several sequences; see ShortConv.prefill_packed): x [1, 1, T, C], h [1, 1, T, H] ->
        y [1, 1, T, C] bf16 (times 3). The lagged rows x_{t-L} come from one block-copy program per chunk (each
        segment's rows shifted by L, its lane state in front), so every op's shape depends on T only."""
        T = int(x.shape[2])
        hb = ttnn.typecast(h, ttnn.bfloat16) if h.dtype != ttnn.bfloat16 else h
        dyn_b = ttnn.matmul(hb, self.wproj, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)  # [1, 1, T, nh*K]
        xf = ttnn.typecast(x, ttnn.float32) if x.dtype != ttnn.float32 else x
        xr = ttnn.to_layout(xf, ttnn.ROW_MAJOR_LAYOUT)
        xs = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, self.C]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT,
                                             self.mesh, ttnn.DRAM_MEMORY_CONFIG) for _ in range(K - 1)]
        jobs, after = [], []
        for row0, R, n, lc in segs:
            for L in range(1, K):
                jobs.append((xr, row0, xs[L - 1], row0 + L, R - L))
                jobs.append((lc.st, K - 1 - L, xs[L - 1], row0, L))
            for P, dst in ((int(n), lc._pf_stn), (lc.snap_row, lc.snap_buf)):
                if P is None:
                    continue
                P = int(P)
                for r in range(K - 1):  # state row r = seq[P + r] = the old state row P + r, or segment row P + r - 3
                    if P + r < K - 1:
                        jobs.append((lc.st, P + r, dst, r, 1))
                    else:
                        jobs.append((xr, row0 + P + r - (K - 1), dst, r, 1))
        bc(jobs)
        y = xr
        for L in range(K):
            wl = ttnn.matmul(dyn_b, self.expand[L], compute_kernel_config=self.ckc, dtype=ttnn.float32)
            wl = ttnn.add(ttnn.to_layout(wl, ttnn.ROW_MAJOR_LAYOUT), self.static[L])
            y = ttnn.add(y, ttnn.multiply(xr if L == 0 else xs[L - 1], wl))
        ttnn.deallocate(dyn_b)
        for t in xs:
            ttnn.deallocate(t)
        for row0, R, n, lc in segs:
            ttnn.copy(lc._pf_stn, lc.st)
            if lc.snap_row is not None:
                lc.snap_out = lc.snap_buf
        y = ttnn.typecast(y, ttnn.bfloat16)
        return ttnn.to_layout(y, ttnn.TILE_LAYOUT)


def dyn_host_reference(x, h, static, wproj, state=None):
    """fp32 reference of DynShortConv.rows (times 3): x [T, C], h [T, H], static [K, C], wproj [nh*K, H]."""
    T, C = x.shape
    nh = wproj.shape[0] // K
    hd = C // nh
    s = torch.zeros(K - 1, C) if state is None else state.float()
    seq = torch.cat([s, x.float()], 0)
    dyn = (h.bfloat16() @ wproj.bfloat16().T).float().view(T, nh, K)  # [T, nh, K] (bf16 Linear, as transformers)
    y = x.float().clone()
    for L in range(K):
        w = static[L].float()[None, :] + dyn[:, :, L].repeat_interleave(hd, dim=1)
        y += w * seq[K - 1 - L : K - 1 - L + T]
    return y, seq[-(K - 1):].clone()
