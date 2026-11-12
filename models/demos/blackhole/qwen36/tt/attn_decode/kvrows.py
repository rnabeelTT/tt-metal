# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paged K / V cache update of n rows of ONE sequence at consecutive positions (a speculative verify's or a drafter's
rows) as one ``ttnn.generic_op`` program, instead of n masked ``paged_fused_update_cache`` calls (rows sharing a cache
tile race when the stock op writes them from different cores in one call).

One core per (K or V, KV head): it reads the at most 2 cache tile rows the positions span (bfp8 tiles), untilizes each,
copies every row's head row from the input tiles into its place (32-byte face rows, L1 -> L1), tilizes it back and
writes it, the stock op's untilize / tilize steps, so the cache bytes match the per-row calls exactly.

Inputs: k / v [1, n, NKV, HD] bf16 TILE in L1 (heads as the rows of each row's tile row, the prep op's output); idx
[n, n] int32 ROW_MAJOR (row u: -1 except column u = row u's position, or -1 to skip row u: the masks the per-row calls
take); page_table [rows, max_blocks] int32 ROW_MAJOR DRAM (the first written row's page-table row is used); the paged
caches [blocks, NKV, block, HD] TILE (bfp8 / bf16)."""

from __future__ import annotations

from pathlib import Path

import ttnn
from models.demos.blackhole.qwen36.tt.gdn_decode.ops import TILE, _cb, _cores, _range_set, _rt

KERNEL_DIR = Path(__file__).parent / "kernels"
CB = dict(cache=0, meta=1, scr=2, out=16, ucache=24, ucache2=25)


def _dm(source, cores, ct, rt, riscv, noc):
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / source),
        core_ranges=_range_set(cores),
        compile_time_args=ct,
        runtime_args=_rt(cores, rt),
        config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc),
    )


def _tile_bytes(dtype):
    return {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048, ttnn.bfloat4_b: 576}[dtype]


def kv_rows_update(k_cache, k, v_cache, v, idx, page_table):
    """Writes row u of k / v ([1, n, NKV, HD]) at position idx[u, u] of the paged caches (in place)."""
    n, hd = k.shape[1], k.shape[3]
    blocks_shape = k_cache.shape  # [blocks, NKV, block, HD]
    nkv = blocks_shape[1]  # k / v may carry more (padding) head rows than the cache: only the cache's are written
    if k.shape[2] < nkv or blocks_shape[3] != hd or hd % TILE or nkv > TILE or tuple(v.shape) != tuple(k.shape):
        raise ValueError(f"kv_rows_update: cache {tuple(blocks_shape)} vs rows {tuple(k.shape)}")
    if tuple(idx.shape) != (n, n):
        raise ValueError(f"kv_rows_update: idx {tuple(idx.shape)}, expected ({n}, {n})")
    block = blocks_shape[2]
    Wt = hd // TILE
    mesh = k.device()
    cores = _cores(mesh, 2 * nkv)  # (K, head 0..), (V, head 0..)
    cr = _range_set(cores)
    ct_bytes = _tile_bytes(k_cache.dtype)
    ub = 2048  # untilized bf16 block (one tile's worth of bytes per page)
    pt_row = page_table.shape[-1] * 4
    idx_row = n * 4
    cbs = [
        _cb(cr, CB["cache"], k_cache.dtype, ct_bytes, Wt),
        _cb(cr, CB["out"], k_cache.dtype, ct_bytes, Wt),
        _cb(cr, CB["meta"], ttnn.uint32, 256, 1),
        _cb(cr, CB["scr"], ttnn.uint32, n * 64 + ((pt_row + 63) // 64) * 64, 1),
        ttnn.CBDescriptor(  # the untilized block, patched in place: one buffer, two CB ids (stock op layout)
            total_size=ub * Wt,
            core_ranges=cr,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB["ucache"], data_format=ttnn.bfloat16, page_size=ub),
                ttnn.CBFormatDescriptor(buffer_index=CB["ucache2"], data_format=ttnn.bfloat16, page_size=ub),
            ],
        ),
    ]
    rd_ct = [n, Wt, nkv, block, idx_row, pt_row, CB["cache"], CB["meta"], CB["scr"]]
    for t in (k_cache, idx, page_table):
        rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    caches, ins = (k_cache, v_cache), (k, v)
    rd_rt, wr_rt = [], []
    for i in range(2 * nkv):
        t, h = divmod(i, nkv)
        rd_rt.append([caches[t].buffer_address(), idx.buffer_address(), page_table.buffer_address(), h])
        wr_rt.append([caches[t].buffer_address(), ins[t].buffer_address(), h])
    if k_cache.dtype != v_cache.dtype or k_cache.memory_config() != v_cache.memory_config():
        raise ValueError(f"kv_rows_update: K / V caches differ ({k_cache.dtype} / {v_cache.dtype})")
    writers = []  # one writer kernel per side: k / v may sit in different shard layouts (compile-time accessor args)
    for t in range(2):
        wr_ct = [n, Wt, CB["meta"], CB["ucache"], CB["ucache2"], CB["out"], Wt * TILE * 2, 2]
        wr_ct += ttnn.TensorAccessorArgs(caches[t]).get_compile_time_args()
        wr_ct += ttnn.TensorAccessorArgs(ins[t]).get_compile_time_args()
        side = cores[t * nkv : (t + 1) * nkv]
        writers.append(
            _dm(
                "kvrows_writer.cpp",
                side,
                wr_ct,
                wr_rt[t * nkv : (t + 1) * nkv],
                ttnn.DataMovementProcessor.RISCV_0,
                ttnn.NOC.NOC_1,
            )
        )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
    )
    kernels = [
        _dm("kvrows_reader.cpp", cores, rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
        *writers,
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "kvrows_compute.cpp"),
            core_ranges=cr,
            compile_time_args=[CB["cache"], CB["ucache"], CB["ucache2"], CB["out"], Wt],
            runtime_args=_rt(cores, [[] for _ in cores]),
            config=cfg,
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([k_cache, k, v_cache, v, idx, page_table], program)
