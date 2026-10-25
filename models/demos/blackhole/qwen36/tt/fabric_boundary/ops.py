# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-row decode layer boundary as one op per site: fabric all-reduce of the packed partial sums + residual add +
RMSNorm, after the gpt-oss fused decode boundary (models/demos/gpt_oss/tt/fused_decode/boundary.py, whose sender and
compute kernels these are).

Contract (one decode row, TP over the (1, N) mesh, `QWEN35_FABRIC_BOUNDARY=1`):
  * the residual is a flat BF16 vector (value h at byte 2 h, `tiles` 2 KB pages) on BOUNDARY_CORE, replicated on
    every device: each boundary produces the next one's residual; no other op reads it;
  * the mixer / MoE return their per-device partial sum as an L1 BF16 [1, 1, 32, H] TILE tensor (row 0 used); the
    boundary's sender packs row 0 of its tiles flat (2 x 32 B per tile) and multicasts the packed 4 KB over the fabric
    into slot `ring index` of the site's receive buffer on every device (the all-reduce payload packed: 9.0 -> 4.0 us
    for ttnn's all_reduce_async on these shapes, fusion/tools/mbench_ccl.py);
  * the boundary core sums the slots in slot order (bit-identical residual on every device), adds the residual and
    applies the RMSNorm (FP32 statistics); the to-rows cores then write the normed row as row 0 of the [1, 1, 32, H]
    BF16 TILE output (rows 1..31 zero) in whatever layout the consumer reads (L1 interleaved: the decode matmuls and
    the fused MoE take it without a copy; DRAM for the LM head).
Replaces, per site, the bfp8 reshard + all_reduce_async + add + sharded RMSNorm (+ the consumers' sharded-to-
interleaved copies).

Receive-buffer reuse without a completion handshake: the two sites ("mixer", "mlp") strictly alternate and every
boundary waits for every device's partial, so no device can send into a site's slots before every device consumed
the previous use (the gpt-oss argument).
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import torch
from loguru import logger

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
PAGE = TILE * TILE * 2
# Off the DRAM banks' reader rows (0, 1, 3, 4, 6, 7, 9 on these chips; decode_linear / moe_decode place their streamers
# there), so a consumer op can share the program (consumer=)
BOUNDARY_CORE = (9, 2)
ROWS_CORE_ROWS = (2, 5)  # to-rows / packer cores: x 0..7 on these rows


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class FabricBoundary:
    def __init__(self, mesh_device, hidden, eps, ring_topology=None):
        assert hidden % 1024 == 0, "whole flat pages"
        self.mesh_device = mesh_device
        self.hidden, self.eps = hidden, eps
        self.tiles = hidden // 1024
        self.ntiles = hidden // TILE  # 32x32 tiles of a [1, 1, 32, H] tensor
        self.ring = mesh_device.shape[1]
        assert mesh_device.shape[0] == 1, "(1, N) mesh"
        if ring_topology is None:
            ring_topology = os.environ.get("QWEN35_FB_TOPOLOGY", "ring") == "ring"
        self.ring_topology = ring_topology
        self.core = ttnn.CoreCoord(*BOUNDARY_CORE)
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.core, self.core)])
        self.rows_cores = [ttnn.CoreCoord(x, y) for y in ROWS_CORE_ROWS for x in range(8)]
        assert self.ntiles % len(self.rows_cores) == 0
        self.rows_per_core = self.ntiles // len(self.rows_cores)
        self.rows_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(7, y)) for y in ROWS_CORE_ROWS]
        )
        assert not self.rows_set.contains(self.core)
        phys = mesh_device.worker_core_from_logical_core(self.core)
        self.noc_xy = (phys.x, phys.y)
        self.rows_noc = [mesh_device.worker_core_from_logical_core(c) for c in self.rows_cores]
        self.payload = self.tiles * PAGE
        max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        self.chunk = min(self.payload, max_payload // PAGE * PAGE)
        self.coords = [ttnn.MeshCoordinate(0, c) for c in range(self.ring)]
        self._warned = set()
        grid = mesh_device.compute_with_storage_grid_size()
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.sites = {
            name: (
                ttnn.empty(self._flat_shape(self.ring * self.tiles), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device,
                           self._flat_config(self.ring * self.tiles)),
                ttnn.create_global_semaphore(mesh_device, all_cores, 0),
            )
            for name in ("mixer", "mlp")
        }

    # ---- layouts ----------------------------------------------------------------------------------------------------
    def _flat_config(self, pages):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(self.cores, (TILE, pages * TILE), ttnn.ShardOrientation.ROW_MAJOR),
        )

    @staticmethod
    def _flat_shape(pages):
        return [1, 1, TILE, pages * TILE]

    def _empty_flat(self):
        return ttnn.empty(self._flat_shape(self.tiles), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh_device,
                          self._flat_config(self.tiles))

    def gamma(self, weight):
        """An RMSNorm weight (the norm's device tensor, +1 offset applied, any shape of H values) -> the flat BF16
        [1, 1, 1, tiles * 1024] row-major DRAM tensor (one page) the boundary reads."""
        w = ttnn.to_torch(weight, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0].reshape(-1).float()
        flat = torch.zeros(1, 1, 1, self.tiles * 1024)
        flat[..., : self.hidden] = w[: self.hidden]
        return ttnn.from_torch(
            flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    # ---- program ----------------------------------------------------------------------------------------------------
    def _hops(self, idx):
        if self.ring_topology:
            fwd = self.ring // 2
            return fwd, self.ring - 1 - fwd
        return self.ring - 1 - idx, idx

    def _neighbor(self, idx, step):
        return ttnn.MeshCoordinate(0, (idx + step) % self.ring)

    def _cb(self, index, size, dtype=ttnn.bfloat16, tensor=None, cores=None):
        fmt = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                                       page_size=PAGE if dtype == ttnn.bfloat16 else 4096)]
        if tensor is not None:
            cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
            cb.total_size = size
            cb.format_descriptors = fmt
            return cb
        return ttnn.CBDescriptor(total_size=size, core_ranges=cores or self.cores, format_descriptors=fmt)

    def reserved_cores(self):
        """(x, y) of the cores this op uses (a consumer sharing the program must avoid them)."""
        return {(self.core.x, self.core.y)} | {(c.x, c.y) for c in self.rows_cores}

    def __call__(self, residual, partial, gamma, site=None, out_memory_config=ttnn.L1_MEMORY_CONFIG, consumer=None,
                 keep_x=False):
        """residual: the flat residual, or (entry, partial None) a [1, 1, 32, H] BF16 TILE tensor whose row 0 is the
        residual (the gathered embedding rows); partial: this device's [1, 1, 32, H] BF16 TILE L1 partial sum (row 0),
        or None; gamma: gamma(norm weight). Returns (residual' flat, x [1, 1, 1, H] BF16 TILE in out_memory_config,
        row 0 the normed residual', padding rows 1..31 zero).
        consumer: the op that consumes x, run in this program (the gpt-oss consumer fusion: its weights stream while the
        boundary computes): consumer(x_flat, noc_xy, sem_id) -> (kernels, cbs, semaphores, io, notify_noc, out) with
        notify_noc the NOC coordinates of its cores waiting for program semaphore sem_id (incremented once x_flat holds
        the normed row); no to-rows output then: returns (residual' flat, out); keep_x: the to-rows output too (another
        op also reads x): returns (residual' flat, x, out)."""
        entry = partial is None
        converted = None
        if not entry and (partial.dtype != ttnn.bfloat16 or partial.memory_config().buffer_type != ttnn.BufferType.L1):
            if (site, str(partial.dtype)) not in self._warned:  # a producer off the fast path: one copy per boundary
                self._warned.add((site, str(partial.dtype)))
                logger.warning(f"fabric boundary {site}: {partial.dtype} {partial.memory_config().buffer_type} partial "
                               f"copied to L1 BF16")
            partial = converted = ttnn.to_memory_config(partial, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        T = self.tiles
        out_res = self._empty_flat()
        out_xf = self._empty_flat()
        fused = consumer is not None
        if fused:
            assert not entry, "a consumer-fused boundary all-reduces"
            c_kernels, c_cbs, c_sems, c_io, notify_noc, c_out = consumer(out_xf, self.noc_xy, 0)
            x = c_out
            if keep_x:
                notify_noc = list(notify_noc) + list(self.rows_noc)
                x = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.hidden]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                                   self.mesh_device, out_memory_config)
        else:
            notify_noc = self.rows_noc
            x = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, self.hidden]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                               self.mesh_device, out_memory_config)
        slots = 0 if entry else self.ring
        recv, sem = (None, None) if entry else self.sites[site]
        sem_addr = 0 if entry else ttnn.get_global_semaphore_address(sem)
        cb_res, cb_recv, cb_gamma, cb_scaler, cb_res_out, cb_sq, cb_rs, cb_x, cb_y = 8, 9, 10, 11, 24, 25, 26, 27, 28
        cb_pk, cb_scr = 12, 13
        cbs = [
            self._cb(cb_res, T * PAGE, tensor=None if entry else residual),
            self._cb(cb_res_out, T * PAGE, tensor=out_res),
            self._cb(cb_gamma, T * PAGE),
            self._cb(cb_scaler, PAGE),
            self._cb(cb_sq, 4096, ttnn.float32),
            self._cb(cb_rs, 4096, ttnn.float32),
            self._cb(cb_x, T * PAGE, tensor=out_xf),
            self._cb(cb_y, T * 4096, ttnn.float32),
            self._cb(cb_scr, self.rows_per_core * PAGE, cores=self.rows_set),
        ]
        if not entry:
            cbs += [self._cb(cb_recv, slots * T * PAGE, tensor=recv), self._cb(cb_pk, PAGE, cores=self.rows_set)]
        res_src = residual
        reader_ct = [cb_res, cb_recv, cb_gamma, cb_scaler, T, slots, self.ring, 0 if not entry else 2, cb_x, self.ntiles]
        reader_ct += ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(res_src).get_compile_time_args()
        reader_rt = ttnn.RuntimeArgs()
        reader_rt[self.core.x][self.core.y] = [gamma.buffer_address(), sem_addr, res_src.buffer_address(),
                                               len(notify_noc)] + [v for p in notify_noc for v in (p.x, p.y)]
        compute_ct = [cb_res, cb_recv, cb_gamma, cb_scaler, cb_res_out, cb_sq, cb_rs, cb_x, T, slots]
        compute_ct += [_f32_bits(1.0 / self.hidden), _f32_bits(self.eps), 1, cb_y]
        rows_ct = [cb_scr, self.rows_per_core, self.noc_xy[0], self.noc_xy[1]]
        rows_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
        if not entry:
            rows_ct += ttnn.TensorAccessorArgs(partial).get_compile_time_args() + [cb_pk]
        rows_rt = {(c.x, c.y): [out_xf.buffer_address(), x.buffer_address(), i * self.rows_per_core]
                   for i, c in enumerate(self.rows_cores)}
        rows_rt_dev = None if entry else [  # per device: + (partial, this device's slot)
            (partial.buffer_address(), recv.buffer_address() + idx * self.payload) for idx in range(self.ring)
        ]
        dm = ttnn.DataMovementConfigDescriptor
        base = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "fb_reader.cpp"), core_ranges=self.cores, compile_time_args=reader_ct,
                runtime_args=reader_rt, config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "fb_compute.cpp"), core_ranges=self.cores, compile_time_args=compute_ct,
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True,
                                                    math_approx_mode=False),
            ),
        ]

        def rows_kernel(idx):
            rt = ttnn.RuntimeArgs()
            for c in self.rows_cores:
                rt[c.x][c.y] = rows_rt[(c.x, c.y)] + ([] if entry else list(rows_rt_dev[idx]))
            return ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "fb_rows.cpp"), core_ranges=self.rows_set, compile_time_args=rows_ct,
                runtime_args=rt, config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=([] if entry else [("FB_PACK_ROWS", "1")]) + ([("FB_PACK_ONLY", "1")] if fused and not keep_x else []),
            )

        both = self.rows_set.merge(self.cores)
        notified = both.merge(consumer.cores) if fused else both  # consumer: a callable with .cores (its CoreRangeSet)
        semaphores = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, notified, 0),
                      ttnn.SemaphoreDescriptor(1, ttnn.CoreType.WORKER, both, 0)]
        if fused:
            semaphores += list(c_sems)
            cbs = cbs + list(c_cbs)
        mesh_program = ttnn.MeshProgramDescriptor()
        for idx, coord in enumerate(self.coords):
            program = ttnn.ProgramDescriptor(kernels=list(base) + [rows_kernel(idx)] + (list(c_kernels) if fused else []),
                                             semaphores=list(semaphores), cbs=list(cbs))
            if not entry:
                self._add_sender(idx, coord, program, recv, sem_addr)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        io = [residual, out_res, out_xf, x, gamma] + ([] if entry else [partial, recv]) + (list(c_io) if fused else [])
        ttnn.generic_op(io, mesh_program)
        ttnn.deallocate(out_xf)
        if converted is not None:
            ttnn.deallocate(converted)
        if fused and keep_x:
            return out_res, x, c_out
        return out_res, x

    def _add_sender(self, idx, coord, program, recv, sem_addr):
        fwd, bwd = self._hops(idx)
        rt = ttnn.RuntimeArgs()
        rt[self.core.x][self.core.y] = [0, recv.buffer_address() + idx * self.payload, self.noc_xy[0], self.noc_xy[1],
                                        sem_addr]
        ct = [self.payload, self.chunk, fwd, bwd, len(self.rows_cores)]
        sender = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "fb_sender.cpp"), core_ranges=self.cores, compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        program.kernels = list(program.kernels) + [sender]
        src = self.mesh_device.get_fabric_node_id(coord)
        fabric_args = []
        for step, hops in ((1, fwd), (-1, bwd)):
            fabric_args.append(1 if hops > 0 else 0)
            if hops > 0:
                fabric_args += ttnn.setup_fabric_connection(
                    src_fabric_node_id=src,
                    dst_fabric_node_id=self.mesh_device.get_fabric_node_id(self._neighbor(idx, step)),
                    link_idx=0, program_descriptor=program, worker_core=self.core,
                )
        program.kernels[len(program.kernels) - 1].runtime_args[self.core.x][self.core.y].extend(fabric_args)


def fabric_boundary(args, mesh_device):
    """The model-wide FabricBoundary (one per args object), built before any trace capture."""
    b = args.__dict__.get("_fabric_boundary")
    if b is None:
        b = FabricBoundary(mesh_device, args.dim, args.norm_eps)
        args._fabric_boundary = b
    return b
