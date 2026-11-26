# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The packed decode all-reduce inside its producer's program (I2M_AR_PUSH): the producer's output cores write their
packed rows straight into the exchange, and the all-reduce's shard / add / sender kernels (tt/ccl.py _Packed.push) run in
the same program: no pack, no separate all-reduce program; the sums are arfuse_add's (bit-identical). Producers: the
MoE down ("mlp": the qwen36 ExpertDownRows program, built the same way, with this port's copy of its writer,
kernels/dnp_writer.cpp) and the prefetched o_proj ("mixer": tt/prefetch.py GcbO, kernels/qpf_writer.cpp). -> PackedAR,
as the all-reduce's."""

import os

import ttnn

from models.demos.blackhole.qwen36.tt.moe_decode.experts import _cb, _ep_defines, _kernels, _range_set
from models.demos.blackhole.qwen36.tt.moe_decode.union import ACT_PAGE

from .bconv import KDIR, _ks


_WARNED = []


def free_rows(mesh):
    """the grid rows whose first 9 cores neither producer uses (the MoE down's 3 readers per bank, the o_proj's
    receivers: the qwen36 / tt/prefetch.py placements), for the all-reduce's shard cores"""
    from models.demos.blackhole.qwen36.tt.moe_decode.experts import place_banks_offrow

    busy = set()
    for n in (int(os.environ.get("QWEN35_MOE_DN_READERS", "3")), 9):
        placed, _ = place_banks_offrow(mesh, {b: n for b in range(8)})
        busy |= {(c.x, c.y) for v in placed.values() for c in v}
    g = mesh.compute_with_storage_grid_size()
    return [y for y in range(g.y) if not any((x, y) in busy for x in range(9))]
L1_CHECK = os.environ.get("I2M_L1_CHECK") == "1"


def _tiles_of(op):
    return {(c.x, c.y): [b + op.banks * (r * op.cols + j) for j in range(op.cols)] for c, b, r in op.assign}


def run_push(pk, site, T, tiles_of, make, io):
    """One producer program with the packed all-reduce in it, per die: make(out, push args, push defines) -> (kernels,
    cbs) of the producer (its writers take the args / defines); pk.push adds the exchange's kernels. tiles_of: the
    producer cores' output tiles. -> PackedAR, or None (the exchange's cores collide with the producer's)."""
    from .ccl import PackedAR

    got = pk.push(site, T, tiles_of, set(tiles_of))
    if got is None:
        if site not in _WARNED:
            _WARNED.append(site)
            print(f"arpush {site}: the exchange's cores collide with the producer's ({sorted(tiles_of)}): old path",
                  flush=True)
        return None
    out, push_args, per_die, push_defs, io_x = got
    if (site, T) not in _WARNED:
        _WARNED.append((site, T))
        print(f"arpush {site}: {T} rows in the producer's program", flush=True)
    mesh_prog = ttnn.MeshProgramDescriptor()
    for idx in range(pk.mesh.get_num_devices()):
        kernels, cbs = make(out, push_args(idx), push_defs)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=list(cbs))
        per_die(idx, prog)
        if idx == 0 and L1_CHECK:  # (the mesh program is not checked by tt/__init__.py's hook: its die 0 program here)
            from . import l1_check

            l1_check(list(io) + [out] + io_x, prog)
        coord = ttnn.MeshCoordinate(0, idx)
        mesh_prog[ttnn.MeshCoordinateRange(coord, coord)] = prog
    ttnn.generic_op(list(io) + [out] + io_x, mesh_prog)
    return PackedAR(out, T, 32 * sum(len(ts) for ts in tiles_of.values()), pk)


def down_push(op, act, table, down, pk, site, T):
    """op: the ExpertDownRows of this width (one row group, routed only, no EP); pk: the packed all-reduce (ccl._packed).
    -> PackedAR, or None (the caller keeps the old path)."""
    if op.groups != 1 or op.shared or op.ep_experts or T > op.trows:
        return None
    cb_in0, cb_w, cb_scr, cb_tab, cb_out = 0, 1, 3, 5, 16
    tb = 64 * op.trows
    cbs = [
        _cb(op.cores, cb_in0, ttnn.bfloat16, tb, 2 * op.groups * op.chunk * op.kt, tiny=True, rows=op.trows),
        _cb(op.cores, cb_w, op.weight_dtype, op.tile_bytes, 2 * op.chunk * op.kt * op.cols),
        _cb(op.cores, cb_scr, ttnn.uint16, 64 * op.rows + 128, 1),
        _cb(op.cores, cb_tab, ttnn.uint16, 64 * op.rows + 128, 1),
        _cb(op.cores, cb_out, ttnn.bfloat16, tb, op.groups * op.cols, tiny=True, rows=op.trows),
    ]
    if op.split:
        cbs.append(_cb(op.cores, 6, ttnn.bfloat16, tb, 2 * op.groups * op.cols, tiny=True, rows=op.trows))
    rd_ct = [cb_w, cb_scr, op.kt, op.m, op.cols, op.k, op.tile_bytes, op.rows, op.chunk]
    rd_ct += ttnn.TensorAccessorArgs(table).get_compile_time_args()
    wr_ct = [cb_in0, cb_out, cb_tab, op.kt, op.cols, op.banks, op.k, op.rows, op.trows, op.groups, op.chunk, ACT_PAGE]
    for t in (act, down, table):  # (the out accessor is unused on the push path: any DRAM tensor)
        wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    cp_ct = [cb_in0, cb_w, cb_out, op.kt, op.cols, op.chunk, op.groups]
    defines = [("ROWS", str(op.trows))] + ([("DST_FULL", "1")] if op.dst_full else [])
    defines += [("MOE_SUBS", str(op.subs))] if op.subs > 1 else []
    defines += [("DN_SPLIT", str(op.split)), ("DN_TMP", "6")] if op.split else []
    defines += [("DN_PAD", str(op.pad))] if op.pad else []
    defines += [("DN_SLOT", "1")] if os.environ.get("QWEN35_MOE_DN_SLOT") == "1" and op.groups == 1 else []
    defines += [("MOE_ZONES", "1")] if os.environ.get("QWEN35_MOE_ZONES") == "1" else []
    defines.append(("NO_SHARED", "1"))
    defines += _ep_defines(op.ep_experts)
    wsrc = _ks(KDIR, "dnp_writer.cpp")

    def make(out, pa, pdefs):
        rd_rt, wr_rt = {}, {}
        for c, b, r in op.assign:
            c0 = r * op.cols
            rd_rt[(c.x, c.y)] = [down.buffer_address(), down.buffer_address(), table.buffer_address(), b, c0,
                                 (b + r) & 0x3]
            wr_rt[(c.x, c.y)] = [act.buffer_address(), out.buffer_address(), table.buffer_address(), b, c0] + pa
        kernels = []
        for noc1 in (False, True):
            sel = [c for c, b, _ in op.assign if (b in op.noc1_banks) == noc1]
            if not sel:
                continue
            keys = {(c.x, c.y) for c in sel}
            ks = _kernels(
                _range_set(sel), "dnr",
                (rd_ct, {k: v for k, v in rd_rt.items() if k in keys}),
                (wr_ct, {k: v for k, v in wr_rt.items() if k in keys}),
                (cp_ct, None), defines, dst_full=op.dst_full, reader_noc=ttnn.NOC.NOC_1 if noc1 else None)
            ks[1].kernel_source = wsrc
            ks[1].defines = list(defines) + list(pdefs)
            kernels += ks
        return kernels, cbs

    return run_push(pk, site, T, _tiles_of(op), make, [down, act, table])
