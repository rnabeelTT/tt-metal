# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode sampling from the device's top-32 candidates (the vLLM plugin's sample_on_device_mode, decode steps).

On device, inside the decode trace: the logits [1, 1, W, Vp] fp32 -> bf16 -> qwen36 decode_sampling.RowsTopK (one fused
scan + merge program, <= 8 live rows per call, UINT32 indices above 65536 columns) twice, the second on the rows shifted
by the first pass's max (see DeviceTopK.device). Only [W, 32] indices + values cross PCIe (vs [W, Vp] fp32 logits), and vLLM's CPU sampler over the
full vocab is skipped. On the host: vLLM's semantics over the candidates (greedy at temperature 0; temperature, top-k,
top-p; per-request seeds; repetition / frequency / presence penalties applied to the candidates). Truncating the
distribution to the top 32 is what models/common/sampling (max_top_k 32) does for the other ports.
"""

from __future__ import annotations

import torch

import ttnn

K = 32
GROUP = 8  # RowsTopK's live-row limit
# I2M_TOPK_GATHER (default 1): a sharded head's per-die candidates all-gathered on device (one die read back)
GATHER = __import__("os").environ.get("I2M_TOPK_GATHER", "1") == "1"
# I2M_TOPK_DRAM (default 1): the candidates the decode traces keep live in DRAM: as one-page L1 interleaved tensors each
# reserved its 2 KB on every core, and with every width's trace their region reached the prefill SDPA's static CBs at
# 2048-row chunks (p300x2: "CBs clash with L1 buffers")
TOPK_DRAM = __import__("os").environ.get("I2M_TOPK_DRAM", "1") == "1"
# I2M_TOPK_MAX (default 1): the shift of the second top-k pass, bf16(row max), from one program reading only the live
# rows of the fp32 logits (tt/kernels/rowmax.cpp) instead of a whole-tile typecast + a first top-k pass (bf16 rounding is
# monotone: the same value; candidates and tokens bitwise)
TOPK_MAX = __import__("os").environ.get("I2M_TOPK_MAX", "1") == "1"


class RowMax:
    """fp32 TILE [1, 1, rows, Vp] (rows <= 8 live) -> fp32 TILE [1, 1, rows, 1]: bf16(row max) (tt/kernels/rowmax.cpp)."""

    NC, BATCH = 80, 16

    def __init__(self, mesh, Vp: int):
        self.mesh, self.T = mesh, Vp // 32
        nc = min(self.NC, self.T)
        self.cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(nc)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        per = -(-self.T // nc)
        self.work = [(c, i * per, max(0, min(per, self.T - i * per))) for i, c in enumerate(self.cores)]
        self.mxy = mesh.worker_core_from_logical_core(self.cores[0])

    def __call__(self, lg, rows, fill=False):
        """fill (rows 1): the value in all 32 rows ([1, 1, 32, 1])."""
        import hashlib
        from pathlib import Path

        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32 if fill else rows, 1]), ttnn.float32, ttnn.TILE_LAYOUT,
                                             self.mesh, ttnn.L1_MEMORY_CONFIG)
        nc = len(self.cores)
        ct = [0, rows, nc, self.mxy.x, self.mxy.y, 0, self.BATCH, int(fill)] + ttnn.TensorAccessorArgs(lg).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        for i, (c, t0, n) in enumerate(self.work):
            rt[c.x][c.y] = [lg.buffer_address(), out.buffer_address(), t0, n, i]
        k = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).parent / "kernels" / "rowmax.cpp"),
                                  core_ranges=self.crs, compile_time_args=ct, runtime_args=rt,
                                  config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                           noc=ttnn.NOC.NOC_0))
        nb = (nc * -(-rows // 4) * 16 + 63) // 64 * 64 + max(4096, self.BATCH * 2 * rows * 64)
        fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=nb)
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, self.crs, 0)],
                                      cbs=[ttnn.CBDescriptor(total_size=nb, core_ranges=self.crs, format_descriptors=[fmt])])
        key = ("i2m_rowmax", tuple(ct), repr(lg.memory_config()))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([lg, out], prog)
        return out


# I2M_TOPK_DENSE (default 1, with TOPK_MAX): a group's shift + bf16 cast on a dense copy of its live rows (row r of
# tile t -> row t % 32 of dense tile r TD + t / 32, tt/kernels/rowpack.cpp), scattered back into those rows of the top-k input: the
# same subtract / typecast ops on 1 / 32 of the tiles (the padding rows are never read by the top-k); the dense copy and
# the shift in one program. With TOPK_MAX, one die W = 1: the trace's sampling tail 523 -> 235 us; served p150 plain TPOT
# T = 1 17.78 -> 17.44 ms, T = 0 17.83 -> 17.50 (texts identical)
TOPK_DENSE = __import__("os").environ.get("I2M_TOPK_DENSE", "1") == "1"
# I2M_TOPK_DENSE_W: the dense tail for groups of up to this many rows (row r -> dense block r: [1, rows, 32, 32 TD], the
# shift per block); default by the dies sharing the vocab: 8 (one), 4 (two), 1 (four: its smaller rows lose; TP4 W = 8
# 2319 -> 2364 us)
DENSE_W = int(__import__("os").environ.get("I2M_TOPK_DENSE_W", "0"))
# I2M_TOPK_DENSE_L1: the dense copies (fp32, its shifted fp32, bf16) in L1 up to this many tiles per copy (one die W = 1:
# 197), else in DRAM (transient L1 like W = 1's at most: one die W = 8 would need 8 x that)
DENSE_L1 = int(__import__("os").environ.get("I2M_TOPK_DENSE_L1", "200"))


class RowPack:
    """[1, 1, R, 32 T] TILE rows 0 .. R - 1 <-> dense [1, R, 32, 32 ceil(T / 32)] TILE (tt/kernels/rowpack.cpp)."""

    BATCH = 16
    # I2M_ROWPACK_WIDE (default 4): groups of at least this many rows on both data-movement RISCs of every core of the
    # grid (the max scan is RISC-bound: one die W = 4 tail 518 -> 434 us, W = 8 906 -> 734; W = 1 230 -> 264, so
    # fewer rows keep 80 cores x 1 RISC)
    WIDE = int(__import__("os").environ.get("I2M_ROWPACK_WIDE", "4"))

    def __init__(self, mesh, Vp: int):
        self.mesh, self.T = mesh, Vp // 32
        self.TD = -(-self.T // 32)
        self.layouts = {}

    def _layout(self, rows):
        wide = rows >= self.WIDE
        if wide not in self.layouts:
            g = self.mesh.compute_with_storage_grid_size()
            ns = 2 if wide else 1
            nc = min(g.x * g.y if wide else 80, -(-self.T // ns))
            gx = g.x if wide else 8
            cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(nc)]
            nw = nc * ns  # worker w: core w % nc, RISC w // nc
            per = -(-self.T // nw)
            work = [(cores[w % nc], w // nc, w * per, max(0, min(per, self.T - w * per))) for w in range(nw)]
            self.layouts[wide] = (ns, cores, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]), work)
        return self.layouts[wide]

    def run(self, src, dst, pack, rows, top=None):
        import hashlib
        from pathlib import Path

        NS, cores, crs, work = self._layout(rows)
        ct = [0, self.BATCH, self.TD] + ttnn.TensorAccessorArgs(src).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(dst).get_compile_time_args()
        ct += [NS]
        if top is not None:
            m = self.mesh.worker_core_from_logical_core(cores[0])
            ct += [len(work), m.x, m.y] + ttnn.TensorAccessorArgs(top).get_compile_time_args()
        rts = [ttnn.RuntimeArgs() for _ in range(NS)]
        for i, (c, k, t0, n) in enumerate(work):
            rts[k][c.x][c.y] = [src.buffer_address(), dst.buffer_address(), t0, n, k] + (
                [i, top.buffer_address()] if top is not None else [])
        risc = ((ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0), (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1))
        ks = [ttnn.KernelDescriptor(kernel_source=str(Path(__file__).parent / "kernels" / "rowpack.cpp"),
                                    core_ranges=crs, compile_time_args=ct, runtime_args=rts[k],
                                    defines=[("RP_PACK", str(int(pack))), ("RP_MAX", str(int(top is not None))),
                                             ("RP_ROWS", str(rows))],
                                    config=ttnn.DataMovementConfigDescriptor(processor=risc[k][0], noc=risc[k][1]))
              for k in range(NS)]
        nb = NS * self.BATCH * rows * 2 * 64 + (len(work) * 32 + 4096 * rows if top is not None else 0)
        fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=nb)
        prog = ttnn.ProgramDescriptor(kernels=ks, semaphores=[ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, crs, 0)], cbs=[
            ttnn.CBDescriptor(total_size=nb, core_ranges=crs, format_descriptors=[fmt])])
        key = ("i2m_rowpack", tuple(ct), pack, rows, top is not None, repr(src.memory_config()), repr(dst.memory_config()))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([src, dst] + ([top] if top is not None else []), prog)
        return dst

    def pack(self, lg, rows=1, with_max=False, mc=ttnn.L1_MEMORY_CONFIG):
        """-> dense fp32 copy (with_max: and the fp32 [1, rows, 32, 1] tiles, tile r: row r's bf16(max) in all 32 rows,
        as RowMax(fill=True) for one row)"""
        d = ttnn.allocate_tensor_on_device(ttnn.Shape([1, rows, 32, 32 * self.TD]), ttnn.float32, ttnn.TILE_LAYOUT,
                                           self.mesh, mc)
        if not with_max:
            return self.run(lg, d, True, rows)
        top = ttnn.allocate_tensor_on_device(ttnn.Shape([1, rows, 32, 1]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh,
                                             ttnn.L1_MEMORY_CONFIG)
        return self.run(lg, d, True, rows, top), top

    def unpack(self, db, rows=1):
        s = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, 32 * self.T]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                           self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        return self.run(db, s, False, rows)


# I2M_TOPK_X (default 1): on a vocab-sharded head the tail's two die exchanges (the shift max; the candidates) as one
# fabric program each (TailX, tt/kernels/tkx.cpp) instead of three ttnn.all_gather (+ ttnn.max, slices, DRAM copies);
# the same values in the same die order (candidates and tokens bitwise)
TOPK_X = __import__("os").environ.get("I2M_TOPK_X", "1") == "1"


class TailX:
    """One core per die multicasts its payload over the fabric into a persistent DRAM scratch slot on every other die
    (a global semaphore per exchange counts the arrivals), then builds its output from the N slots (tt/kernels/tkx.cpp).
    A die's next write to a slot follows an exchange that needs that die's reader past its read (strict alternation of
    the max and candidate exchanges), so one scratch + semaphore per exchange suffices."""

    def __init__(self, mesh, ib):
        self.mesh, self.N, self.ib = mesh, mesh.get_num_devices(), ib
        g = mesh.compute_with_storage_grid_size()
        self.core = ttnn.CoreCoord(g.x - 1, 0)
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(self.core, self.core)])
        self.xy = mesh.worker_core_from_logical_core(self.core)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        mk = lambda b: ttnn.from_torch(torch.zeros(self.N, b // 4, dtype=torch.int32), dtype=ttnn.uint32,  # noqa: E731
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                       mesh_mapper=rep)
        self.scr = {"max": mk(64), "cand": mk(1024 * ib + 2048)}
        self.sem = {k: ttnn.create_global_semaphore(mesh, self.crs, 0) for k in self.scr}

    def _run(self, kind, ins, outs, L, dense):
        from pathlib import Path

        mx = kind == "max"
        scr, sem = self.scr[kind], self.sem[kind]
        in1 = ins[1] if len(ins) > 1 else ins[0]
        out1 = outs[1] if len(outs) > 1 else outs[0]
        acc = []
        for t in (ins[0], in1, scr, outs[0], out1):
            acc += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        slot = 64 if mx else 1024 * L + 2048
        nb = self.N * slot + (4096 * (L if dense else 1) if mx else 0)
        sem_addr = ttnn.get_global_semaphore_address(sem)
        rt0 = [ins[0].buffer_address(), in1.buffer_address(), scr.buffer_address(), outs[0].buffer_address(),
               out1.buffer_address(), sem_addr]
        mesh_prog = ttnn.MeshProgramDescriptor()
        c = self.core
        for me in range(self.N):
            rt = ttnn.RuntimeArgs()
            rt[c.x][c.y] = list(rt0)
            k = ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).parent / "kernels" / "tkx.cpp"), core_ranges=self.crs,
                compile_time_args=[0, self.N, me, L, int(dense), self.xy.x, self.xy.y] + acc, runtime_args=rt,
                defines=[("TKX_MAX", str(int(mx)))],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                         noc=ttnn.NOC.NOC_0))
            fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=nb)
            prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[
                ttnn.CBDescriptor(total_size=nb, core_ranges=self.crs, format_descriptors=[fmt])])
            coord = ttnn.MeshCoordinate(0, me)
            src = self.mesh.get_fabric_node_id(coord)
            fab = []
            for step, hops in ((1, self.N - 1 - me), (-1, me)):
                fab.append(1 if hops > 0 else 0)
                if hops > 0:
                    fab += ttnn.setup_fabric_connection(
                        src_fabric_node_id=src,
                        dst_fabric_node_id=self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, me + step)),
                        link_idx=0, program_descriptor=prog, worker_core=c)
            prog.kernels[0].runtime_args[c.x][c.y].extend(fab)
            mesh_prog[ttnn.MeshCoordinateRange(coord, coord)] = prog
        ttnn.generic_op(list(ins) + [scr] + list(outs), mesh_prog)

    def max(self, top, rows):
        """fp32 TILE shifts [1, 1, rows, 1], or [1, rows, 32, 1] (RowPack: tile r holds row r's) -> the max over the
        dies, same shape."""
        out = ttnn.allocate_tensor_on_device(top.shape, ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG)
        self._run("max", [top], [out], rows, int(top.shape[2]) == 32)
        return out

    def cand(self, i2, v2, rows, mc, out=None):
        """the top-k's [1, 1, 32, K] index / value tiles -> [1, 1, rows, N K] (die d's at tile d) in mc (out: a previous
        call's (index, value) outputs, written in place)."""
        mk = lambda dt: ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, self.N * K]), dt,  # noqa: E731
                                                       ttnn.TILE_LAYOUT, self.mesh, mc)
        oi, ov = out if out is not None else (mk(i2.dtype), mk(v2.dtype))
        self._run("cand", [i2, v2], [oi, ov], 4 if i2.dtype == ttnn.uint32 else 2, False)
        return oi, ov


class DeviceTopK:
    def __init__(self, mesh, Vp: int, V: int, sharded: bool = False):
        """Vp: one die's logits columns. sharded: the LM head is split over the vocab (die d holds columns
        [d * Vp, (d + 1) * Vp)): each die's candidates are its slice's top K, shifted by the max over ALL dies (one
        small all-gather) so the host can merge them by value."""
        from models.demos.blackhole.qwen36.tt.decode_sampling.ops import RowsTopK

        self.mesh, self.Vp, self.V = mesh, Vp, V
        self.N = mesh.get_num_devices()
        self.sharded = sharded and self.N > 1
        self.rtk = RowsTopK(mesh, Vp, k=K)
        self.rmax = RowMax(mesh, Vp) if TOPK_MAX else None
        self.rpack = RowPack(mesh, Vp) if TOPK_MAX and TOPK_DENSE else None
        self.dense_w = DENSE_W or {1: 8, 2: 4}.get(self.N if self.sharded else 1, 1)
        self.x = TailX(mesh, 4 if self.rtk.wide else 2) if self.sharded and TOPK_X and mesh.get_num_devices() in (2, 4) else None

    def device(self, lg, W: int, outs=None):
        """Device half (traceable): lg [1, 1, W, Vp] fp32 TILE -> [(indices [1, 1, rows, K], logits - row max
        [1, 1, rows, K] bf16, rows)] per group of <= 8 rows (outs: a previous call's result at this W, written in
        place: a trace's live outputs allocated before any capture). Two fused top-k passes: the first's best bf16 value (within
        one bf16 ulp of the row max) shifts the fp32 row so the candidates sit near 0, where bf16 resolves them to
        ~1e-3 (softmax is shift-invariant; the arg max stays exact); a gather of the fp32 values would cost 3.5 ms."""
        lb = ttnn.typecast(lg, ttnn.bfloat16) if self.rmax is None else None
        parts = []
        for g0 in range(0, W, GROUP):
            rows = min(GROUP, W - g0)
            part_f = lg if W <= GROUP else ttnn.slice(lg, [0, 0, g0, 0], [1, 1, g0 + rows, self.Vp])
            dense = self.rpack is not None and rows <= self.dense_w
            if dense:  # (the dense copy and the shift in one program)
                dmc = ttnn.L1_MEMORY_CONFIG if rows * self.rpack.TD <= DENSE_L1 else ttnn.DRAM_MEMORY_CONFIG
                d, top = self.rpack.pack(part_f, rows, with_max=True, mc=dmc)
            elif self.rmax is not None:
                top = self.rmax(part_f, rows)
            else:
                part_b = lb if W <= GROUP else ttnn.slice(lb, [0, 0, g0, 0], [1, 1, g0 + rows, self.Vp])
                v1, _ = self.rtk(part_b, rows)
                top = ttnn.typecast(ttnn.slice(v1, [0, 0, 0, 0], [1, 1, rows, 1]), ttnn.float32)
            if self.sharded and self.x is not None:  # every die shifts by the same value: the max over the dies' maxima
                top = self.x.max(top, rows)
            elif self.sharded:
                tops = ttnn.all_gather(top, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)  # [1, 1, rows, N]
                top = ttnn.max(tops, dim=-1, keepdim=True)
            if int(part_f.shape[2]) != rows:
                part_f = ttnn.slice(part_f, [0, 0, 0, 0], [1, 1, rows, self.Vp])
            if dense:
                db = ttnn.typecast(ttnn.subtract(d, top, memory_config=dmc), ttnn.bfloat16, memory_config=dmc)
                ttnn.deallocate(d)
                shifted = self.rpack.unpack(db, rows)
                ttnn.deallocate(db)
            else:
                shifted = ttnn.typecast(ttnn.subtract(part_f, top), ttnn.bfloat16)
            v2, i2 = self.rtk(shifted, rows)
            if self.sharded and GATHER and self.x is not None:
                gi, gv = self.x.cand(i2, v2, rows, ttnn.DRAM_MEMORY_CONFIG if TOPK_DRAM else ttnn.L1_MEMORY_CONFIG,
                                     out=outs[len(parts)][:2] if outs is not None else None)
                ttnn.deallocate(i2)
                ttnn.deallocate(v2)
                parts.append((gi, gv, rows))
                continue
            i2, v2 = ttnn.slice(i2, [0, 0, 0, 0], [1, 1, rows, K]), ttnn.slice(v2, [0, 0, 0, 0], [1, 1, rows, K])
            if self.sharded and GATHER:  # every die's candidates on every die: the host reads one die (2 reads, not 2 N)
                i2 = ttnn.all_gather(i2, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)  # [1, 1, rows, N K]
                v2 = ttnn.all_gather(v2, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)
            if outs is not None:
                i2d, v2d, _ = outs[len(parts)]
                ttnn.copy(i2, i2d)
                ttnn.copy(v2, v2d)
                ttnn.deallocate(i2)
                ttnn.deallocate(v2)
                i2, v2 = i2d, v2d
            elif TOPK_DRAM:
                i2d, v2d = ttnn.to_memory_config(i2, ttnn.DRAM_MEMORY_CONFIG), ttnn.to_memory_config(v2, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(i2)
                ttnn.deallocate(v2)
                i2, v2 = i2d, v2d
            parts.append((i2, v2, rows))
        return parts

    def host(self, parts):
        """-> host (indices [W, K] int64, logits [W, K] fp32) by row, sorted descending (sharded head: the dies'
        candidates merged by value, ties to the lower index)."""
        devs = (lambda t: ttnn.get_device_tensors(t)) if self.N > 1 else (lambda t: [t])
        nd = self.N if self.sharded else 1
        idx, val = [], []
        for i, v, r in parts:
            if self.sharded and GATHER:  # die 0 holds every die's K candidates, die by die
                ii = ttnn.to_torch(devs(i)[0]).reshape(r, nd, K).to(torch.int64)
                ii = (ii + self.Vp * torch.arange(nd).reshape(1, nd, 1)).reshape(r, nd * K)
                vv = ttnn.to_torch(devs(v)[0]).float().reshape(r, nd * K)
            else:
                ii = [ttnn.to_torch(t).reshape(r, K).to(torch.int64) + d * self.Vp for d, t in enumerate(devs(i)[:nd])]
                vv = [ttnn.to_torch(t).float().reshape(r, K) for t in devs(v)[:nd]]
                ii, vv = torch.cat(ii, -1), torch.cat(vv, -1)
            if nd > 1:
                # by value descending, then by index ascending (stable sort on the index order first)
                o = torch.argsort(ii, dim=-1, stable=True)
                ii, vv = ii.gather(-1, o), vv.gather(-1, o)
                o = torch.argsort(vv, dim=-1, descending=True, stable=True)[:, :K]
                ii, vv = ii.gather(-1, o), vv.gather(-1, o)
            idx.append(ii)
            val.append(vv)
        return torch.cat(idx), torch.cat(val)

    def __call__(self, lg, W: int):
        return self.host(self.device(lg, W))


class DeviceSampler:
    """The token on device from DeviceTopK.device's candidates (tt/kernels/tsample.cpp, one core per row group): argmax_j
    (logit_j + tg_j) over the candidates with id < V, tg = T * Gumbel noise from the host (zeros: greedy, the host
    sampler's first-max tie rule). Exactly the host sampler's softmax(logits / T) over the same candidates when the
    request has no penalties / top-k < 32 / top-p (eligible). Written straight into the decode's token input."""

    def __init__(self, topk: DeviceTopK):
        self.tk = topk
        self.nt = topk.N if (topk.sharded and GATHER) else 1
        mesh = topk.mesh
        rep = ttnn.ReplicateTensorToMesh(mesh) if topk.N > 1 else None
        self.rep = rep
        self.tg = ttnn.from_torch(torch.zeros(32, K), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
        core = ttnn.CoreCoord(0, 0)
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    def device(self, parts, tokens, W: int):
        """parts: DeviceTopK.device(lg, W); tokens: uint32 [1, W] ROW_MAJOR (traceable)."""
        import hashlib
        from pathlib import Path

        g0 = 0
        for i2, v2, rows in parts:
            io = [i2, v2, self.tg, tokens]
            ct = [0, rows, g0, W, self.nt, self.tk.Vp if self.nt > 1 else 0, self.tk.V,
                  4 if i2.dtype == ttnn.uint32 else 2]
            for t in io:
                ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
            rt = ttnn.RuntimeArgs()
            rt[0][0] = [t.buffer_address() for t in io]
            k = ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).parent / "kernels" / "tsample.cpp"), core_ranges=self.crs,
                compile_time_args=ct, runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                         noc=ttnn.NOC.NOC_0))
            nb = self.nt * 6144 + 8 * 128 + 128 + self.nt * K * 8
            fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=nb)
            prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[
                ttnn.CBDescriptor(total_size=nb, core_ranges=self.crs, format_descriptors=[fmt])])
            key = ("i2m_tsample", tuple(ct))
            prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & (
                (1 << 63) - 1)
            ttnn.generic_op(io, prog)
            g0 += rows
        return tokens

    def set_noise(self, tg):
        """tg: torch [W <= 32, K] (T * Gumbel; zeros = greedy), by physical slot."""
        x = torch.zeros(32, K)
        x[: tg.shape[0]] = tg.float()
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                        mesh_mapper=self.rep), self.tg)


def device_ok(sp, r) -> bool:
    """Row r's sampling is exactly DeviceSampler's: no penalties, no top-k truncation below the candidates (top-k 1 =
    greedy), no top-p."""
    if float(_param(sp, "repetition_penalty", r, 1.0)) != 1.0 or float(_param(sp, "frequency_penalty", r, 0.0)) != 0.0 \
            or float(_param(sp, "presence_penalty", r, 0.0)) != 0.0:
        return False
    tk = int(_param(sp, "top_k", r, -1))
    tp = float(_param(sp, "top_p", r, 1.0))
    if tk == 1:
        return True
    return (tk <= 0 or tk >= K) and not (0.0 < tp < 1.0)


def _param(sp, name, r, default):
    v = getattr(sp, name, None)
    if v is None:
        return default
    if isinstance(v, torch.Tensor):
        return v.reshape(-1)[r].item() if v.numel() > r else default
    if isinstance(v, (list, tuple)):
        return v[r] if len(v) > r else default
    return v


class HostSampler:
    """vLLM-semantics sampling over each row's top-K candidates."""

    def __init__(self, V: int):
        self.V = V
        self.gens: dict[int, tuple[int, torch.Generator]] = {}
        # sampled drafts: the drafter's temperature = qt x the request's (any q keeps the acceptance exact; qt tunes
        # how well q matches the target's p: I2M_SPEC_QT)
        self.qt = float(__import__("os").environ.get("I2M_SPEC_QT", "1.0"))

    def forget(self, row: int):
        self.gens.pop(row, None)

    def _gen(self, row, seed):
        if seed is None or int(seed) < 0:
            return None
        g = self.gens.get(row)
        if g is None or g[0] != int(seed):
            gen = torch.Generator()
            gen.manual_seed(int(seed))
            self.gens[row] = g = (int(seed), gen)
        return g[1]

    def dist(self, ids, vals, r, sp, outs=None, prompt=None):
        """One row's candidates (ids, logits [K]) under vLLM's semantics -> (ids, probs) after penalties, temperature,
        top-k, top-p; probs None at temperature 0 (ids then holds the greedy token alone). outs / prompt: the
        request's output / prompt tokens (penalties)."""
        ids, lg = ids.clone(), vals.double().clone()
        ok = ids < self.V  # the LM head's zero-padded vocab columns are never sampled
        ids, lg = ids[ok], lg[ok]
        rep = float(_param(sp, "repetition_penalty", r, 1.0))
        fre = float(_param(sp, "frequency_penalty", r, 0.0))
        pre = float(_param(sp, "presence_penalty", r, 0.0))
        if (rep != 1.0 or fre != 0.0 or pre != 0.0) and outs is not None:
            outs = outs[outs >= 0].to(torch.int64)
            cnt = (ids[:, None] == outs[None, :]).sum(-1).double()
            if rep != 1.0:
                seen = cnt > 0
                if prompt is not None:
                    pr = prompt[prompt >= 0].to(torch.int64)
                    seen = seen | (ids[:, None] == pr[None, :]).any(-1)
                lg = torch.where(seen, torch.where(lg > 0, lg / rep, lg * rep), lg)
            lg = lg - fre * cnt - pre * (cnt > 0).double()
        temp = float(_param(sp, "temperature", r, 0.0))
        if temp <= 0.0:
            i = int(torch.argmax(lg))  # ties: the lower index (candidates sorted descending)
            return ids[i : i + 1], None
        lg = lg / temp
        order = torch.argsort(lg, descending=True)
        ids, lg = ids[order], lg[order]
        tk = int(_param(sp, "top_k", r, -1))
        if 0 < tk < lg.numel():
            ids, lg = ids[:tk], lg[:tk]
        p = torch.softmax(lg, -1)
        tp = float(_param(sp, "top_p", r, 1.0))
        if 0.0 < tp < 1.0:
            keep = torch.cumsum(p, -1) - p < tp
            keep[0] = True
            ids, p = ids[keep], p[keep] / p[keep].sum()
        return ids, p

    def sample(self, idx, vals, rows, sp, output_tokens=None, prompt_tokens=None):
        """idx / vals [W, K] by slot; rows: list of (vLLM row r, slot s). Returns {r: token}."""
        out = {}
        for r, s in rows:
            ids, p = self.dist(idx[s], vals[s], r, sp, None if output_tokens is None else output_tokens[r],
                               None if prompt_tokens is None else prompt_tokens[r])
            if p is None:
                out[r] = int(ids[0])
                continue
            gen = self._gen(r, _param(sp, "seed", r, None))
            out[r] = int(ids[int(torch.multinomial(p, 1, generator=gen))])
        return out

    def gumbel(self, r, sp, d, k):
        """[d, k] T * Gumbel noise for a sampled draft (zeros at temperature 0: the drafter's argmax)."""
        temp = float(_param(sp, "temperature", r, 0.0))
        if temp <= 0.0:
            return torch.zeros(d, k)
        u = torch.rand(d, k, generator=self._gen(r, _param(sp, "seed", r, None)), dtype=torch.float64)
        return (self.qt * temp * -torch.log(-torch.log(u.clamp(1e-12, 1 - 1e-12)))).float()

    def spec(self, idx, vals, r, drafts, sp, outs=None, prompt=None, q=None):
        """Speculative acceptance. idx / vals [d + 1, K]: verify row j scores the token after drafts[:j]. q: per draft,
        the drafter's (ids, logits) candidates it was sampled from (q_j = softmax(logits / T)); None: a point-mass
        (greedy) proposal. Draft j is kept with probability min(1, p_j / q_j) (greedy target: iff it is the argmax); the
        first rejection draws from max(p_j - q_j, 0) renormalised; all kept: a draw from row d. The emitted tokens follow
        the target's distribution exactly. -> (tokens emitted, drafts kept)."""
        gen = self._gen(r, _param(sp, "seed", r, None))
        temp = float(_param(sp, "temperature", r, 0.0))
        hist = outs[outs >= 0].to(torch.int64) if outs is not None else torch.zeros(0, dtype=torch.int64)
        emitted = []
        for j in range(len(drafts) + 1):
            ids, p = self.dist(idx[j], vals[j], r, sp, torch.cat([hist, torch.tensor(emitted, dtype=torch.int64)]),
                               prompt)
            last = j == len(drafts)
            if p is None:  # greedy
                if not last and int(ids[0]) == drafts[j]:
                    emitted.append(drafts[j])
                    continue
                return emitted + [int(ids[0])], j
            if not last:
                pq = torch.zeros_like(p)  # q_j on p's support
                qd = 1.0
                if q is not None:
                    qi, ql = q[j]
                    qp = torch.softmax(ql.double() / (self.qt * temp), -1)
                    qd = float(qp[qi == drafts[j]].sum())
                    m = ids[:, None] == qi[None, :]
                    pq = (m.double() * qp[None, :]).sum(-1)
                else:
                    pq = (ids == drafts[j]).double()
                hit = ids == drafts[j]
                pd = float(p[hit].sum())
                if pd > 0.0 and qd > 0.0 and float(torch.rand(1, generator=gen, dtype=torch.float64)) < min(1.0, pd / qd):
                    emitted.append(drafts[j])
                    continue
                res = (p - pq).clamp(min=0.0)
                if float(res.sum()) > 0.0:
                    p = res / res.sum()
            return emitted + [int(ids[int(torch.multinomial(p, 1, generator=gen))])], j
        raise AssertionError("unreachable")
