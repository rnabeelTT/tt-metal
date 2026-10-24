# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet decode mixer for 1-32 users as two ``ttnn.generic_op`` programs between the in- and out-projection
matmuls (Qwen3.5 / Qwen3-Next linear-attention layers, per tensor-parallel die):

* ``conv``: the K = 4 short conv over the qkv columns of the in-proj output, one tile column per work item (all users
  of a column share its tiles): silu(sum_j tap_j h_j) as FP32 into ``conv_out``, and the carried rows
  ``conv_states[1..3]`` shifted in place (``conv_states[0]`` is not read or written; the unfused path, the MTP pass
  and the prefill hand-off only use rows 1..3 as the carry).
* ``head``: one work item per (user, value head): gating, q / k L2 norms, the delta-rule step on the head's bf16
  [Dk, Dv] state (in place, FP32 in DST), the gated RMSNorm and the silu(z) gate, written as the user's row of the
  out-proj input. Users >= B are not touched.

The in-proj output must be the fused [q | k | v | z | a (tile padded) | b (tile padded)] projection in L1 (TILE,
<= 32 rows); conv_out and the outputs are L1 too. Constants per layer (``prepare``): the 4 conv taps as full tiles
([4, 1, 32, qkv] bf16, every row the tap row) and a [1, 1, 32, 128] FP32 params tile per die (row 0: -exp(A_log)
per head, row 1: dt_bias, row 2: the gated-norm weight).
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
NUM_CBS = 64

# head program CB ids (kernels/head_compute.cpp)
CB = dict(q=0, k=1, v=2, gin=3, beta=4, S=5, o=6, state=7, sdh=8, sdl=9, vread=10, uh=11, kcol=12, dcol=13, qnh=14,
          knh=15, nrm=16, scr=17, xh=18, xl=19, qnl=20, knl=21, kcoll=22, ul=23, sdx=24, qsd=25, qk=26, kn1h=27,
          kn1l=28, z=29, w=30, y=31)
UNPACK_FP32 = ("q", "k", "v", "gin", "vread", "dcol", "nrm", "qsd", "qk", "w")


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _cores(mesh_device, n):
    """The first n worker cores, column-major (y fastest)."""
    grid = mesh_device.compute_with_storage_grid_size()
    assert n <= grid.x * grid.y
    return [ttnn.CoreCoord(i // grid.y, i % grid.y) for i in range(n)]


def _split(n_items, n_cores):
    q, r = divmod(n_items, n_cores)
    out, s = [], 0
    for i in range(n_cores):
        c = q + (1 if i < r else 0)
        out.append((s, c))
        s += c
    return out


def _range_set(cores):
    rs = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
    for c in cores[1:]:
        rs = rs.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
    return rs


def _cb(cores, index, dtype, tile_bytes, tiles):
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)
    return ttnn.CBDescriptor(total_size=tile_bytes * tiles, core_ranges=cores, format_descriptors=[fmt])


def _rt(cores, args):
    rt = ttnn.RuntimeArgs()
    for c, a in zip(cores, args):
        rt[c.x][c.y] = a
    return rt


def _dm(source, cores, ct, rt, riscv, noc):
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / source), core_ranges=_range_set(cores), compile_time_args=ct,
        runtime_args=_rt(cores, rt),
        config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc),
    )


def _ndev(device):
    return getattr(device, "get_num_devices", lambda: 1)()


def _per_device(t):
    """torch copies of a (mesh) tensor's per-device shards."""
    if _ndev(t.device()) == 1:
        return [ttnn.to_torch(t)]
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


class GdnDecode:
    """One instance per model (shape set)."""

    def __init__(self, mesh_device, num_k_heads, num_v_heads, head_k_dim, head_v_dim, proj_dim, scale, max_batch=32,
                 eps=1e-6, norm_eps=1e-6):
        self.mesh = mesh_device
        self.Nk, self.Nv, self.Dk, self.Dv = num_k_heads, num_v_heads, head_k_dim, head_v_dim
        assert head_k_dim % TILE == 0 and head_v_dim == 128 and num_v_heads % num_k_heads == 0 and num_v_heads <= 128
        assert max_batch <= TILE, "one tile row of users"
        self.Kt, self.Vt = head_k_dim // TILE, head_v_dim // TILE
        self.qkv = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        self.vdim = num_v_heads * head_v_dim
        self.proj_dim = proj_dim
        self.z_tile = self.qkv // TILE
        self.a_tile = (self.qkv + self.vdim) // TILE
        self.b_tile = self.a_tile + 1  # a and b each padded to one tile column (gdn_ab_tile_pad)
        assert self.a_tile * TILE + 2 * TILE <= proj_dim and num_v_heads <= TILE
        self.scale, self.eps, self.norm_eps = scale, eps, norm_eps
        self.max_batch = max_batch
        self._rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if _ndev(mesh_device) > 1 else {}
        self.keep_conv = False  # op test: keep the last call's conv_out (self.conv_out) instead of freeing it
        self.conv_out = None

    # ---- constants ---------------------------------------------------------------------------------------------
    def prepare(self, conv_taps, neg_exp_a, dt_bias, norm_w):
        """Per-layer device constants from the layer's ttnn tensors (per-die TP shards or replicated): conv_taps = 4
        [.., qkv] tap rows, neg_exp_a / dt_bias = [.., Nv], norm_w = [.., Dv]. -> (taps, params)."""
        taps = [_per_device(t) for t in conv_taps]
        na, dt, nw = _per_device(neg_exp_a), _per_device(dt_bias), _per_device(norm_w)
        n = len(na)
        tf = torch.zeros(n * 4, 1, TILE, self.qkv)
        pr = torch.zeros(n, 1, TILE, 128)
        for d in range(n):
            for j in range(4):
                tf[d * 4 + j, 0, :, :] = taps[j][d].reshape(-1)[: self.qkv].float()
            pr[d, 0, 0, : self.Nv] = na[d].reshape(-1)[: self.Nv].float()
            pr[d, 0, 1, : self.Nv] = dt[d].reshape(-1)[: self.Nv].float()
            pr[d, 0, 2, : self.Dv] = nw[d].reshape(-1)[: self.Dv].float()
        shard = {"mesh_mapper": ttnn.ShardTensorToMesh(self.mesh, dim=0)} if n > 1 else {}
        taps_t = ttnn.from_torch(tf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, **shard)
        params_t = ttnn.from_torch(pr, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, **shard)
        return taps_t, params_t

    def _l1(self, shape, dtype):
        """Per-call L1 buffers (conv_out, y): persistent L1 buffers lower every core's L1 ceiling, and the 2-chip prefill
        in-proj's circular buffers then clashed with them at warmup (Ornith p300)."""
        return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)

    def output(self, B):
        """The [1, B, Nv * Dv] bf16 out-proj input for width B (allocated per call; the caller frees it)."""
        return self._l1([1, B, self.vdim], ttnn.bfloat16)

    # ---- programs -------------------------------------------------------------------------------------------------
    def conv(self, proj, st1, st2, st3, taps):
        conv_out = self._l1([1, 1, TILE, self.qkv], ttnn.float32)
        cols = self.qkv // TILE
        ncores = min(cols, self.mesh.compute_with_storage_grid_size().x * self.mesh.compute_with_storage_grid_size().y)
        cores = _cores(self.mesh, ncores)
        work = _split(cols, ncores)
        cr = _range_set(cores)
        cb_x, cb_h, cb_t, cb_o = 0, 1, 2, 16
        cbs = [_cb(cr, cb_x, ttnn.bfloat16, 2048, 2), _cb(cr, cb_h, ttnn.bfloat16, 2048, 6),
               _cb(cr, cb_t, ttnn.bfloat16, 2048, 8), _cb(cr, cb_o, ttnn.float32, 4096, 2)]
        rd_ct = [cb_x, cb_h, cb_t, cols]
        for t in (proj, st1, taps):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = [cb_o] + ttnn.TensorAccessorArgs(conv_out).get_compile_time_args()
        rd_rt = [[proj.buffer_address(), st1.buffer_address(), st2.buffer_address(), st3.buffer_address(),
                  taps.buffer_address(), s, c] for s, c in work]
        wr_rt = [[conv_out.buffer_address(), s, c] for s, c in work]
        kernels = [
            _dm("conv_reader.cpp", cores, rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            _dm("conv_writer.cpp", cores, wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "conv_compute.cpp"), core_ranges=cr,
                compile_time_args=[cb_x, cb_h, cb_t, cb_o], runtime_args=_rt(cores, [[c] for _, c in work]),
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                    math_approx_mode=False),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([proj, st1, st2, st3, taps, conv_out], program)

    def head(self, conv_out, proj, params, state, B):
        assert state.dtype == ttnn.bfloat16, "bf16 recurrent state"
        y = self.output(B)
        units = B * self.Nv
        grid = self.mesh.compute_with_storage_grid_size()
        ncores = min(units, grid.x * grid.y)
        cores = _cores(self.mesh, ncores)
        work = _split(units, ncores)
        cr = _range_set(cores)
        Kt, Vt = self.Kt, self.Vt
        kv = Kt * Vt
        f32, bf = (ttnn.float32, 4096), (ttnn.bfloat16, 2048)
        sizes = dict(q=(f32, 2 * Kt), k=(f32, 2 * Kt), v=(f32, 2 * Vt), gin=(f32, 8), S=(bf, kv), o=(f32, Vt),
                     state=(bf, kv), vread=(f32, Vt), uh=(f32, Vt), kcol=(f32, Kt), dcol=(f32, 2), qnh=(f32, Kt),
                     knh=(f32, Kt), nrm=(f32, 2), scr=(f32, 5), xh=(f32, 2 * Kt), xl=(f32, 2 * Kt), qnl=(f32, Kt),
                     knl=(f32, Kt), kcoll=(f32, Kt), ul=(f32, Vt), qsd=(f32, Vt), qk=(f32, 1), kn1h=(f32, Kt),
                     kn1l=(f32, Kt), z=(bf, 2 * Vt), w=(f32, Vt), y=(bf, 2 * Vt))
        cbs = [_cb(cr, CB[k], dt, tb, n) for k, ((dt, tb), n) in sizes.items()]
        rd_ct = [Kt, Vt, self.Nv, self.Nk, self.qkv // TILE, self.proj_dim // TILE, self.z_tile, self.a_tile,
                 self.b_tile, CB["q"], CB["k"], CB["v"], CB["gin"], CB["S"], CB["z"], CB["w"], CB["scr"]]
        for t in (conv_out, proj, params, state):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = [Kt, Vt, self.Nv, CB["state"], CB["y"]]
        for t in (state, y):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rd_rt = [[conv_out.buffer_address(), proj.buffer_address(), params.buffer_address(),
                  state.buffer_address(), s, c] for s, c in work]
        wr_rt = [[state.buffer_address(), y.buffer_address(), s, c] for s, c in work]
        cp_ct = [Kt, Vt, _bits(self.eps), _bits(self.scale), _bits(self.norm_eps), _bits(1.0 / self.Dv),
                 _bits(1.0), _bits(1.0), _bits(20.0)]
        utd = [ttnn.UnpackToDestMode.Default] * NUM_CBS
        for k in UNPACK_FP32:
            utd[CB[k]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           math_approx_mode=False)
        cfg.unpack_to_dest_mode = utd
        kernels = [
            _dm("head_reader.cpp", cores, rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            _dm("head_writer.cpp", cores, wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "head_compute.cpp"), core_ranges=cr, compile_time_args=cp_ct,
                runtime_args=_rt(cores, [[c] for _, c in work]), config=cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([conv_out, proj, params, state, y], program)

    def __call__(self, proj, conv_states, taps, params, state, B):
        """proj: the [1, B, P] in-proj output (L1); conv_states: the layer's 4 carried-row tensors; -> the [1, B, Nv *
        Dv] gated output (the out-proj input; a fresh L1 tensor the caller frees)."""
        conv_out = self.conv(proj, conv_states[1], conv_states[2], conv_states[3], taps)
        y = self.head(conv_out, proj, params, state, B)
        if self.keep_conv:
            self.conv_out = conv_out
        else:
            ttnn.deallocate(conv_out)
        return y
