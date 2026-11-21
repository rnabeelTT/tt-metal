# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode projection with the weight streamed per DRAM bank (after the gpt-oss fused decode's LinearStream,
models/demos/gpt_oss/tt/fused_decode/stream.py), at the model's matmul fidelity.

ttnn's decode matmuls read the interleaved weight one tile per request from every bank and top out near 350 GB/s on
the (1, 4) mesh's chips (fusion/tools/mbench_mm.py). Here a decode-only copy of the weight is stored per bank,
column-major (bank b holds output columns [b C, (b + 1) C), each column's K tiles contiguous), so each reader core
streams its columns as one contiguous range in 16 KB packets. `readers` cores per bank, placed along the bank's
optimal reader row (moe_decode.experts.place_along_rows); each computes its columns with matmul_tiles (x rows 0..B-1 of
the [32, K] activation tiles; the other rows are don't-care) and writes them as output tile pages of a normal
[1, 1, B, N] TILE tensor, so the consumers are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

from ..moe_decode.experts import _range_set, bank_workers, place_banks_offrow

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
MAX_PACKET = 16384


def _packet(kt, tile_bytes):
    return max(d for d in range(1, kt + 1) if kt % d == 0 and d * tile_bytes <= MAX_PACKET) * tile_bytes


class DecodeLinear:
    def __init__(self, mesh_device, k, n, weight_dtype, readers=6, fidelity=ttnn.MathFidelity.HiFi2, buffers=3,
                 exclude=()):
        """exclude: (x, y) cores to leave free (a fused program's other kernels, e.g. FabricBoundary.reserved_cores())."""
        assert k % TILE == 0
        self.mesh_device = mesh_device
        self.banks = len(bank_workers(mesh_device))
        self.kt, self.n = k // TILE, n
        self.n_tiles = -(-n // TILE)
        self.C = -(-self.n_tiles // self.banks)
        self.readers = min(readers, self.C)
        self.cpc = -(-self.C // self.readers)
        self.weight_dtype, self.fidelity, self.buffers = weight_dtype, fidelity, buffers
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.page_bytes = _packet(self.kt, self.tile_bytes)
        placed, self.noc1_banks = place_banks_offrow(mesh_device, {b: self.readers for b in range(self.banks)},
                                                     used=set(exclude))
        self.assign = []  # (core, bank, col0 within bank, ncols)
        for b in range(self.banks):
            for r, c in enumerate(placed[b]):
                c0 = r * self.cpc  # padding columns (past n_tiles) are not streamed
                self.assign.append((c, b, c0, max(0, min(self.cpc, self.C - c0, self.n_tiles - b * self.C - c0))))
        self.cores = _range_set([c for c, _, _, _ in self.assign])
        self.noc1_cores = _range_set([c for c, b, _, _ in self.assign if b in self.noc1_banks]) if self.noc1_banks else None
        self.noc0_cores = _range_set([c for c, b, _, _ in self.assign if b not in self.noc1_banks])

    # ---- weights ------------------------------------------------------------------------------------------------------
    def layout(self, w):
        """[K, N] torch weight -> [C * K, banks * 32]: bank b's 32-wide column block holds output columns
        [b C, (b + 1) C) (zero padded), each column's K rows contiguous."""
        K, N = w.shape
        pad = self.C * self.banks * TILE - N
        w = torch.nn.functional.pad(w, (0, pad))
        cols = w.reshape(K, self.C * self.banks, TILE).permute(1, 0, 2)  # [column, K, 32]
        return torch.cat([cols[b * self.C:(b + 1) * self.C].reshape(-1, TILE) for b in range(self.banks)], dim=-1)

    def memory_config(self):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
            ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.banks - 1, 0))}),
                           (self.C * self.kt * TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR),
        )

    def weight(self, per_device, cache_file_name=None):
        """per_device: one [K, N] torch weight per device of the mesh (TP shards, mesh order) -> the decode copy."""
        t = torch.stack([self.layout(w) for w in per_device])  # [ndev, C K, banks 32] (the weights' torch dtype)
        nd = len(per_device)
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0) if nd > 1 else None
        return ttnn.as_tensor(t if nd > 1 else t[0], dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT,
                              device=self.mesh_device, memory_config=self.memory_config(), mesh_mapper=mapper,
                              cache_file_name=cache_file_name)

    # ---- op ------------------------------------------------------------------------------------------------------------
    def _parts(self, x, w, rows, out_memory_config, flat=None, out_dtype=ttnn.bfloat16, out_rows=None, splits=None,
               heads=None, pf=False):
        """flat = (noc_xy, sem_id): x is a flat BF16 vector on that core, ready once program semaphore sem_id is set.
        out_dtype: bfloat16 (default) or float32 (the FP32 DST packed unrounded, e.g. logits).
        splits: [(c0, c1, rows_k), ...] (tile-aligned column ranges covering [0, n) in order, at most 5): the output as
        that many tensors [1, 1, rows_k, c1 - c0] (each output tile page written straight to its part: no slices)."""
        xb = None
        if isinstance(x, (tuple, list)):  # the activation as two K parts [xa | xb] (no concat): tile-aligned split
            x, xb = x
            assert int(x.shape[-1]) % TILE == 0 and int(x.shape[-1]) + int(xb.shape[-1]) == self.kt * TILE
        if heads is not None:
            # per-head (batched) projection: N = Hn heads x NH column tiles, head h's columns read the activation tiles
            # [h XS, h XS + kt) of x [1, Hn, 32, *] (XS: x's tiles per head row); out [1, Hn, rows, N / Hn]
            Hn, XS = heads
            NH = self.n_tiles // Hn
            assert NH * Hn == self.n_tiles and XS >= self.kt and splits is None and xb is None
            for c, b, c0, nc in self.assign:
                if nc:
                    assert (b * self.C + c0) // NH == (b * self.C + c0 + nc - 1) // NH, "a core's columns span 2 heads"
        outs = None
        if splits is not None:
            assert 1 <= len(splits) <= 5 and splits[0][0] == 0 and -(-splits[-1][1] // TILE) == self.n_tiles
            assert all(a % TILE == 0 for a, _, _ in splits) and all(splits[i][1] == splits[i + 1][0]
                                                                     for i in range(len(splits) - 1))
            outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rk or out_rows or rows, b - a]), out_dtype,
                                                   ttnn.TILE_LAYOUT, self.mesh_device, out_memory_config)
                    for a, b, rk in splits]
            out = outs[0]
        elif heads is not None:
            out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, heads[0], out_rows or rows, self.n // heads[0]]),
                                                 out_dtype, ttnn.TILE_LAYOUT, self.mesh_device, out_memory_config)
        else:
            out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, out_rows or rows, self.n]), out_dtype,
                                                 ttnn.TILE_LAYOUT, self.mesh_device, out_memory_config)
        out_page = 4096 if out_dtype == ttnn.float32 else 2048
        cb_x, cb_w, cb_out = 0, 1, 16
        fmt = lambda i, dt, page, pages: ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])
        cbs = [fmt(cb_x, ttnn.bfloat16, 2048, self.kt), fmt(cb_w, self.weight_dtype, self.tile_bytes, self.buffers * self.kt),
               fmt(cb_out, out_dtype, out_page, 2)]
        if pf:  # the weight CB first: at the CB base, where prefetch_parts' kernel left the first columns
            cbs = [cbs[1], cbs[0], cbs[2]]
        rd_ct = [cb_w, self.kt, self.tile_bytes, self.page_bytes, self.buffers]
        wr_ct = [cb_x, cb_out, self.kt, rows, self.n_tiles]
        wr_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args() + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        ext = []
        if outs is not None:  # parts 1..4 (unused: part 0 again, never addressed)
            more = outs[1:] + [outs[0]] * (5 - len(outs))
            for t in more:
                wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
            ext = [t.buffer_address() for t in more]
        xb_cta = len(wr_ct)
        if xb is not None:
            wr_ct += ttnn.TensorAccessorArgs(xb).get_compile_time_args()
        cp_ct = [cb_x, cb_w, cb_out, self.kt]
        cp_rt = ttnn.RuntimeArgs()
        rd_args, wr_args = {}, {}
        for c, b, c0, nc in self.assign:
            rd_args[(c.x, c.y)] = [w.buffer_address(), b, c0, nc, (b + c0) & 0x3] + (
                [self.pf_cols(nc), self.pf_addr()] if pf else [])
            wr_args[(c.x, c.y)] = ([x.buffer_address(), out.buffer_address(), b * self.C + c0, nc] + ext
                                   + ([xb.buffer_address()] if xb is not None else []))
            cp_rt[c.x][c.y] = [nc]

        def rt_for(args, cores):
            rt = ttnn.RuntimeArgs()
            for c in ttnn.corerange_to_cores(cores, row_wise=True):
                rt[c.x][c.y] = args[(c.x, c.y)]
            return rt
        defines = [("OUT_PAGE_BYTES", str(out_page))] if out_page != 2048 else []
        if out_rows is not None and out_rows > rows:  # rows read < the output's rows: zero the padded rows
            defines.append(("ZERO_PAD", "1"))
        if heads is not None:
            defines += [("X_HEADS", "1"), ("X_NH", str(self.n_tiles // heads[0])), ("X_XS", str(heads[1]))]
        if xb is not None:
            defines += [("X_SPLIT", "1"), ("X_KT1", str(int(x.shape[-1]) // TILE)), ("X_B_ARG", str(4 + len(ext))),
                        ("XB_CTA_OFFSET", str(xb_cta))]
        if outs is not None:
            bounds = [a // TILE for a, _, _ in splits[1:]] + [1 << 20] * (5 - len(splits))
            defines.append(("OUT_SPLIT", "1"))
            defines += [(f"OB{i + 1}", str(v)) for i, v in enumerate(bounds)]
        if flat is not None:
            (fx, fy), sem = flat
            cb_xf = 2
            cbs.append(fmt(cb_xf, ttnn.bfloat16, self.kt * 64, 1))
            defines += [("X_FROM_FLAT", "1"), ("X_NOC_X", str(fx)), ("X_NOC_Y", str(fy)), ("X_SEM", str(sem)),
                       ("CB_XF", str(cb_xf))]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = []
        rd_def = [("PF", "1")] if pf else []
        # readers of a bank sharing its row with another bank's: on the next row, NOC1 (place_banks_offrow)
        for cores, rnoc, wnoc in ((self.noc0_cores, ttnn.NOC.NOC_0, ttnn.NOC.NOC_1),
                                  (self.noc1_cores, ttnn.NOC.NOC_1, ttnn.NOC.NOC_0)):
            if cores is None:
                continue
            kernels += [
                ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "dl_reader.cpp"), core_ranges=cores,
                                      compile_time_args=rd_ct, runtime_args=rt_for(rd_args, cores), defines=rd_def,
                                      config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=rnoc)),
                ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "dl_writer.cpp"), core_ranges=cores,
                                      compile_time_args=wr_ct, runtime_args=rt_for(wr_args, cores), defines=defines,
                                      config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=wnoc)),
            ]
        kernels += [
            ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "dl_compute.cpp"), core_ranges=self.cores,
                                  compile_time_args=cp_ct, runtime_args=cp_rt,
                                  config=ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True,
                                                                      math_approx_mode=False)),
        ]
        xs = [x] + ([xb] if xb is not None else [])
        if outs is not None:
            return kernels, cbs, xs + [w] + outs, outs
        return kernels, cbs, xs + [w, out], out

    def __call__(self, x, w, rows, out_memory_config=ttnn.L1_MEMORY_CONFIG, out_dtype=ttnn.bfloat16, out_rows=None,
                 splits=None, heads=None, pf=False):
        """x: [.., rows, K] BF16 TILE (rows <= 16, or 32: whole tiles, every row defined); w: weight();
        -> [1, 1, rows, N] BF16 TILE (splits: the list of column parts, see _parts). pf: the previous program ran
        prefetch_parts(w) (its first columns are in L1 already)."""
        assert 1 <= rows <= 16 or rows == 32
        kernels, cbs, io, out = self._parts(x, w, rows, out_memory_config, out_dtype=out_dtype, out_rows=out_rows,
                                            splits=splits, heads=heads, pf=pf)
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out

    # ---- cross-op prefetch ----------------------------------------------------------------------------------------
    def pf_cols(self, nc):
        """Columns of a core's nc an earlier program prefetches: as many as the weight CB holds."""
        return min(nc, self.buffers)

    def pf_addr(self):
        """The CB base on the reader cores (where the first CB of a program lands)."""
        if not hasattr(self, "_pf_addr"):
            self._pf_addr = int(ttnn.device.get_allocator_base_address(self.mesh_device, ttnn.BufferType.L1))
        return self._pf_addr

    def prefetch_parts(self, w, cb_index):
        """(kernels, cbs) to add to an earlier program that leaves this linear's reader cores idle: each core reads its
        first pf_cols columns into L1 at the CB base (a guard CB `cb_index` there keeps L1 buffers off the range); the
        linear then runs with pf=True. Nothing between the two programs may use those cores' CB range."""
        ct = [self.kt, self.tile_bytes, self.page_bytes, cb_index]
        args = {(c.x, c.y): [w.buffer_address(), b, c0, self.pf_cols(nc), (b + c0) & 0x3, self.pf_addr()]
                for c, b, c0, nc in self.assign}
        dm = ttnn.DataMovementConfigDescriptor
        kernels = []
        for cores, noc in ((self.noc0_cores, ttnn.NOC.NOC_0), (self.noc1_cores, ttnn.NOC.NOC_1)):
            if cores is None:
                continue
            rt = ttnn.RuntimeArgs()
            for c in ttnn.corerange_to_cores(cores, row_wise=True):
                rt[c.x][c.y] = args[(c.x, c.y)]
            kernels.append(ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "dl_prefetch.cpp"), core_ranges=cores,
                                                 compile_time_args=ct, runtime_args=rt,
                                                 config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=noc)))
        guard = ttnn.CBDescriptor(
            total_size=self.tile_bytes * self.buffers * self.kt, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_index, data_format=self.weight_dtype,
                                                        page_size=self.tile_bytes)])
        return kernels, [guard]

    def consumer(self, w, out_memory_config=ttnn.L1_MEMORY_CONFIG):
        """This projection of one row as the consumer of a fabric boundary (FabricBoundary(..., consumer=)): its weight
        streams from the program start, its writers wait for the boundary's normed row."""
        op = self
        noc = [self.mesh_device.worker_core_from_logical_core(c) for c, _, _, _ in self.assign]

        def parts(x_flat, noc_xy, sem_id):
            kernels, cbs, io, out = op._parts(x_flat, w, 1, out_memory_config, flat=(noc_xy, sem_id))
            return kernels, cbs, [], io, noc, out

        parts.cores = self.cores
        return parts


def decode_linear_for(args, mesh_device, weight, fb=None, readers=6, replicated=False, fidelity=ttnn.MathFidelity.HiFi2):
    """(DecodeLinear, decode copy) for a [K, N]-per-device column-parallel ttnn weight (TP shards along N; replicated:
    the same [K, N] on every device), one op per shape (cached on args), its cores clear of the fabric boundary's
    (fb.reserved_cores())."""
    nd = mesh_device.get_num_devices()
    # bf16 on the host: exact for bfp8 / bfp4 weights, half the memory of FP32 (the LM head: 4 x 2048 x 62080 values)
    full = ttnn.to_torch(weight, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).to(torch.bfloat16)
    full = full.reshape(full.shape[-2], full.shape[-1])
    K, N = full.shape[0], full.shape[1] // nd
    cache = args.__dict__.setdefault("_qwen36_decode_linear", {})
    key = (id(mesh_device), K, N, weight.dtype, fidelity)
    if key not in cache:
        cache[key] = DecodeLinear(mesh_device, K, N, weight.dtype, readers=readers,
                                  exclude=fb.reserved_cores() if fb is not None else (), fidelity=fidelity)
    op = cache[key]
    parts = [full[:, :N]] * nd if replicated else [full[:, d * N:(d + 1) * N] for d in range(nd)]
    return op, op.weight(parts)
