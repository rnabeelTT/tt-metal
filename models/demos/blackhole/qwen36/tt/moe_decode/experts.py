# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE decode for one token (Qwen3.5-MoE family: softmax top-k router + routed SwiGLU experts + gated shared
expert), as two ``ttnn.generic_op`` programs after the router matmul:

* ``ExpertGateUp``: every core takes the top-k + softmax and the shared-gate sigmoid from the router logits itself
  (kernels/router_topk.hpp), then streams its expert columns and writes ``w_s * silu(x Wg_s) * (x Wu_s)`` into the
  activation buffer, one row per slot (slot 0 = the shared expert, 1..k = the routed experts).
* ``ExpertDown``: streams the down columns of the k + 1 slots and writes ``sum_s act_s Wd_s`` (routed + shared) into
  row 0 of the [32, H] partial-sum tiles the layer's reduce-scatter takes.

The routed weights are read IN PLACE from the ttnn expert tensors ([1, E, K, N] TILE, DRAM interleaved: page p in
bank p % banks), no decode copy: with N tiles a multiple of the bank count, each expert's columns in a bank are one
contiguous [K, N / banks] run (see gu_layout, kernels/gu_reader.cpp, kernels/dn_reader.cpp). The shared expert must be stored the same way
([1, 1, H, 2 I] packed [up | gate] per TP shard, [1, 1, I, H] down). The router logits carry the shared-expert gate
logit as column E (router weight [H, E + 32]).

Matmuls: custom_mm (1x32 activation tiles, LoFi, FP32 accumulation), worker cores around each bank's optimal reader.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
MAX_PACKET_BYTES = 16384


def _page_bytes(tiles, tile_bytes):
    """Largest NOC packet holding whole tiles that divides a `tiles`-tile block."""
    return max(d for d in range(1, tiles + 1) if tiles % d == 0 and d * tile_bytes <= MAX_PACKET_BYTES) * tile_bytes


RING_BYTES = 768 * 1024  # gate|up weight ring per core (the 2-die bfp8 shapes' 5 x [64, 2] blocks: 696 KB)


def gu_layout(n_tiles, banks):
    """Gate|up column groups (kernels/gu_layout.hpp): m = n_tiles / banks columns per bank and expert; a group reads
    `nb` whole [kt, m] bank blocks per slot (bank q; banks q and q + banks / 2 when m is odd, where up and gate columns
    sit in opposite banks) and owns the up|gate pairs among them. -> (m, nb, groups, [[pair index] per group])."""
    assert n_tiles % banks == 0, f"{n_tiles} up|gate tiles do not split over {banks} banks"
    m = n_tiles // banks
    nb = 2 if m % 2 else 1
    groups = banks // nb
    out = []
    for q in range(groups):
        cols = [q + h * (banks // 2) + banks * c for h in range(nb) for c in range(m)]
        out.append([c for c in cols if c < n_tiles // 2])
    assert sorted(p for g in out for p in g) == list(range(n_tiles // 2))
    return m, nb, groups, out


def gu_ring(kt, m, nb, tile_bytes, want, even=False):
    """Blocks in the gate|up weight ring: `want`, fewer (>= 3) when a [kt, m] block makes it exceed RING_BYTES; even
    when the compute indexes a slot's nb = 2 blocks together (custom_mm's tile index does not wrap)."""
    n = max(3, min(want, RING_BYTES // (kt * m * tile_bytes)))
    if even and nb == 2:
        n += n % 2
    return n


ROUTERS = ("softmax", "sigb")  # Qwen3.5-MoE softmax top-k + gated shared expert; GLM-4.5-Air sigmoid + selection bias


def ep_offsets(mesh_device, experts_per_die):
    """Expert parallelism: [1, 1, 1, 32] UINT32 per die holding its first expert id (die d: d * experts_per_die), one
    DRAM page in bank 0 at the same address on every die (the kernels read it; runtime args stay identical)."""
    n = mesh_device.get_num_devices()
    t = torch.zeros(n, 1, 1, 32, dtype=torch.int32)
    t[:, 0, 0, 0] = torch.arange(n, dtype=torch.int32) * experts_per_die
    return ttnn.from_torch(
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def _ep_defines(ep_experts, gate_first=False):
    return ([("EP_EXPERTS", str(ep_experts))] if ep_experts else []) + ([("GATE_FIRST", "1")] if gate_first else [])


def logits_row_bytes(num_experts, router):
    """L1 bytes of one gathered router-logits row (kernels/router_topk.hpp): BF16 with the shared-gate column (softmax),
    FP32 without it (sigb), + the sigb selection bias (Q23 int32 per expert)."""
    if router == "sigb":
        return 128 * (num_experts // TILE) + 4 * num_experts + 64  # + 64: the bias lands 64-byte aligned
    return 64 * (num_experts // TILE + 1)


def bank_workers(mesh_device):
    return list(ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, ttnn.NOC.NOC_0))


_NOC_COORDS = {}


def noc_coords(mesh_device):
    """{(logical x, y): (NOC0 x, y)} for every worker core, read back from the cores themselves (kernels/noc_coords.cpp)
    on every device; asserts the devices agree (the ops' runtime args are the same on every device)."""
    key = id(mesh_device)
    if key in _NOC_COORDS and _NOC_COORDS[key][0] is mesh_device:
        return _NOC_COORDS[key][1]
    grid = mesh_device.compute_with_storage_grid_size()
    n = grid.x * grid.y
    multi = getattr(mesh_device, "get_num_devices", lambda: 1)() > 1
    rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if multi else {}
    out = ttnn.from_torch(
        torch.zeros(n, 16, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, **rep,
    )
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    rt = ttnn.RuntimeArgs()
    for y in range(grid.y):
        for x in range(grid.x):
            rt[x][y] = [out.buffer_address(), y * grid.x + x]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "noc_coords.cpp"),
        core_ranges=cores,
        compile_time_args=[0] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
    )
    cb = ttnn.CBDescriptor(total_size=64, core_ranges=cores,
                           format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=64)])
    dummy = ttnn.from_torch(torch.zeros(1, 16, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=mesh_device, **rep)  # generic_op takes at least one input
    ttnn.generic_op([dummy, out], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))
    ttnn.deallocate(dummy)
    tabs = [ttnn.to_torch(t).reshape(n, 16) for t in (ttnn.get_device_tensors(out) if multi else [out])]
    assert all(torch.equal(t[:, :3], tabs[0][:, :3]) for t in tabs), "devices disagree on worker NOC coordinates"
    assert bool((tabs[0][:, 2] == 1).all()), "noc_coords: some cores did not report"
    table = {(i % grid.x, i // grid.x): (int(tabs[0][i, 0]), int(tabs[0][i, 1])) for i in range(n)}
    ttnn.deallocate(out)
    _NOC_COORDS[key] = (mesh_device, table)
    return table


def place_along_rows(mesh_device, per_bank, used=None):
    """{bank: count} -> {bank: [CoreCoord]}: each bank's cores on its optimal reader's row, nearest first (+dx before
    -dx), then the rows above / below (off-row placement measured slower). `used`: (x, y) cores already taken
    (updated)."""
    workers = bank_workers(mesh_device)
    grid = mesh_device.compute_with_storage_grid_size()
    used = set() if used is None else used
    out = {}
    for b, n in per_bank.items():
        w = workers[b]
        cand = [(w.x + dx, w.y + dy) for dy in (0, -1, 1, -2, 2) for d in range(grid.x) for dx in ((d, -d) if d else (0,))]
        cores = []
        for x, y in cand:
            if len(cores) == n:
                break
            if 0 <= x < grid.x and 0 <= y < grid.y and (x, y) not in used:
                used.add((x, y))
                cores.append(ttnn.CoreCoord(x, y))
        assert len(cores) == n, f"no room for {n} cores near bank {b}"
        out[b] = cores
    return out


def _range_set(cores):
    rs = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
    for c in cores[1:]:
        rs = rs.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
    return rs


def _cb(cores, index, dtype, page, pages, tiny=False, rows=1):
    """tiny: [rows, 32] activation tiles."""
    kw = dict(buffer_index=index, data_format=dtype, page_size=page)
    if tiny:
        kw["tile"] = ttnn.TileDescriptor(ttnn.Tile([rows, TILE]))
    return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(**kw)])


def _rt(args, cores=None):
    """{(x, y): [args]} -> ttnn.RuntimeArgs (only the cores in `cores`, a list of (x, y), when given)."""
    rt = ttnn.RuntimeArgs()
    for (x, y), a in args.items():
        if cores is None or (x, y) in cores:
            rt[x][y] = a
    return rt


def _kernels(cores, name, reader, writer, compute, defines=(), dst_full=False):
    """reader / writer / compute: (compile args, {(x, y): runtime args})."""
    dm = ttnn.DataMovementConfigDescriptor
    rd_ct, rd_args = reader
    wr_ct, wr_args = writer
    cp_ct, cp_args = compute
    cp_rt = _rt(cp_args) if cp_args is not None else None
    groups = [(list(rd_args), ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)]
    dms = []
    for grp, rd_noc, wr_noc in groups:
        cr = _range_set([ttnn.CoreCoord(x, y) for x, y in grp])
        rd_rt, wr_rt = _rt(rd_args, grp), _rt(wr_args, grp)
        dms += [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"{name}_reader.cpp"),
                core_ranges=cr,
                compile_time_args=rd_ct,
                runtime_args=rd_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=rd_noc),
                defines=list(defines),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"{name}_writer.cpp"),
                core_ranges=cr,
                compile_time_args=wr_ct,
                runtime_args=wr_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=wr_noc),
                defines=list(defines),
            ),
        ]
    return dms + [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / f"{name}_compute.cpp"),
            core_ranges=cores,
            compile_time_args=cp_ct,
            runtime_args=cp_rt if cp_rt is not None else [],
            defines=list(defines),
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False,
                dst_full_sync_en=dst_full,
            ),
        ),
    ]


class MoEDecodeBuffers:
    """The ops' L1 buffers, one set per model (each layer consumes them before the next layer writes them): the
    route record [1, 32] UINT16, the activations [k + 1, I] BF16 row-major and the output [32, H] BF16 TILE (zeros;
    only row 0 is ever written). Host uploads: build before any trace capture."""

    def __init__(self, mesh_device, hidden, inter, k):
        rep = (
            {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)}
            if getattr(mesh_device, "get_num_devices", lambda: 1)() > 1
            else {}
        )
        kw = dict(device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG, **rep)
        self.route = ttnn.from_torch(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, **kw
        )
        self.act = ttnn.from_torch(
            torch.zeros(1, 1, k + 1, inter, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, **kw
        )
        self.out = ttnn.from_torch(
            torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **kw
        )


class ExpertGateUp:
    """(x, logits, up_gate, shared_up_gate, act, route) -> act. readers: cores per column group (gu_layout), placed
    around the group's first bank reader; the routed slots go round-robin over them and core 0 also takes the shared
    expert (slot 0), which it streams while the router tail (top-k + softmax) runs."""

    def __init__(self, mesh_device, hidden, inter, num_experts, k, weight_dtype, readers=4, buffers=5, used=None,
                 router="softmax", shared=True, ep_experts=None, gate_first=False):
        """shared False: routed experts only (no slot 0; the port adds its shared expert itself). ep_experts: expert
        parallelism, the experts each die holds (num_experts stays the global count the router ranks; the weight
        tensors hold the die's window). gate_first: [gate | up] packed weights instead of [up | gate]."""
        assert router in ROUTERS, router
        assert not ep_experts or not shared, "expert parallelism: routed experts only"
        self.router, self.shared, self.ep_experts, self.gate_first = router, shared, ep_experts, gate_first
        self.mesh_device = mesh_device
        self.banks = len(bank_workers(mesh_device))
        self.kt = hidden // TILE
        self.n_tiles = 2 * inter // TILE
        self.pairs = self.n_tiles // 2
        self.m, self.nb, self.groups, self.group_pairs = gu_layout(self.n_tiles, self.banks)
        assert num_experts % 64 == 0 and k <= 15, (num_experts, k)
        self.E, self.k, self.inter = num_experts, k, inter
        self.readers = min(readers, k)
        self.slots = [([0] if r == 0 and shared else []) + list(range(1 + r, k + 1, self.readers)) for r in range(self.readers)]
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.weight_dtype = weight_dtype
        self.buffers = gu_ring(self.kt, self.m, self.nb, self.tile_bytes, buffers)  # bank blocks in the cb_w ring (buffers - 2 read ahead)
        self.page_bytes = _page_bytes(self.kt * self.m, self.tile_bytes)
        self.dst_full = self.m * self.nb > 4  # FP32 DST: 4 tiles half-sync, 8 full-sync
        assert self.m * self.nb <= 8, f"{self.m} up|gate columns per bank: more than 8 DST tiles"
        # each group's cores sit around its (up column's) bank reader (around the gate column's: 28 vs 20 us for the
        # m = 1 pair on banks 0 / 4, whose readers share logical row 9 on QuietBox 2)
        placed = place_along_rows(mesh_device, {q: self.readers for q in range(self.groups)}, used)
        self.assign = [(c, q, r) for q in range(self.groups) for r, c in enumerate(placed[q])]
        self.cores = _range_set([c for c, _, _ in self.assign])

    def parts(self, x, logits, up_gate, shared_up_gate, act, route, notify=None, bias=None, ep_offset=None):
        """(kernels, cbs). notify = (semaphore id, [(NOC x, y)] of the down cores) for the one-program MoE. bias: the
        sigb router's selection bias (Q23 int32 [1, 1, 1, E] row-major, one DRAM page)."""
        assert (bias is not None) == (self.router == "sigb") and (notify is None or self.router == "softmax")
        assert (ep_offset is not None) == bool(self.ep_experts) and (notify is None or not self.ep_experts)
        cb_x, cb_w, cb_route, cb_scr, cb_part, cb_lg, cb_act = 0, 1, 2, 3, 4, 5, 16
        lg_bytes = logits_row_bytes(self.E, self.router)
        cbs = [
            _cb(self.cores, cb_x, ttnn.bfloat16, 64, self.kt, tiny=True),
            _cb(self.cores, cb_w, self.weight_dtype, self.tile_bytes, self.buffers * self.kt * self.m),
            _cb(self.cores, cb_route, ttnn.uint16, 128, 1),
            _cb(self.cores, cb_scr, ttnn.bfloat16, lg_bytes + (128 if self.ep_experts else 0), 1),
            _cb(self.cores, cb_part, ttnn.uint32, 8 * self.k, 1),
            _cb(self.cores, cb_lg, ttnn.bfloat16, lg_bytes, 1),
            _cb(self.cores, cb_act, ttnn.bfloat16, 64, 2 * len(self.group_pairs[0]), tiny=True),
        ]
        tail = [] if notify is None else [len(notify[1])] + [v for xy in notify[1] for v in xy]
        bias_arg = [] if bias is None else [bias.buffer_address()]
        rd_rt, wr_rt, cp_rt = {}, {}, {}
        for i, (c, j, r) in enumerate(self.assign):
            sl = self.slots[r]
            vc = (j + r) & 0x3
            ep_arg = [ep_offset.buffer_address()] if ep_offset is not None else []
            rd_rt[(c.x, c.y)] = [logits.buffer_address(), up_gate.buffer_address(), shared_up_gate.buffer_address(), j, vc, len(sl)] + sl + bias_arg + ep_arg
            wr_rt[(c.x, c.y)] = [x.buffer_address(), act.buffer_address(), route.buffer_address(), logits.buffer_address(), j, int(i == 0 and notify is None), len(sl)] + sl + bias_arg + tail
            cp_rt[(c.x, c.y)] = [len(sl)]
        rd_ct = [cb_w, cb_route, cb_scr, self.kt, self.n_tiles, self.E, self.k, cb_part, self.tile_bytes]
        rd_ct += [self.page_bytes, self.banks, self.buffers] + ttnn.TensorAccessorArgs(logits).get_compile_time_args()
        wr_ct = [cb_x, cb_act, cb_route, self.kt, self.k, self.E, 2 * self.inter, cb_part, cb_lg, self.m, self.banks]
        for t in (x, act, route, logits):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        cp_ct = [cb_x, cb_w, cb_act, self.kt, self.m, self.banks]
        defines = [] if notify is None else [("MOE_FUSED", "1"), ("MOE_SEM", str(notify[0]))]
        if self.router == "sigb":
            defines.append(("ROUTER_SIGB", "1"))
        defines += _ep_defines(self.ep_experts, self.gate_first)
        return _kernels(self.cores, "gu", (rd_ct, rd_rt), (wr_ct, wr_rt), (cp_ct, cp_rt), defines, dst_full=self.dst_full), cbs

    def __call__(self, x, logits, up_gate, shared_up_gate, act, route, bias=None, ep_offset=None):
        kernels, cbs = self.parts(x, logits, up_gate, shared_up_gate, act, route, bias=bias, ep_offset=ep_offset)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        io = [up_gate, shared_up_gate, x, logits, route, act] + [t for t in (bias, ep_offset) if t is not None]
        return ttnn.generic_op(io, program)


class ExpertDown:
    """(act, route, down, shared_down, out) -> out. readers: cores per DRAM bank, each with m / readers of the bank's
    m output columns (None: enough for <= 4 columns per core, the half-sync FP32 DST; at least 2)."""

    def __init__(self, mesh_device, hidden, inter, num_experts, k, weight_dtype, readers=None, used=None, shared=True,
                 ep_experts=None):
        """shared False: routed experts only (shared_down is not read). ep_experts: expert parallelism (the route
        record holds this die's live count and local ids)."""
        assert not ep_experts or not shared
        self.shared, self.ep_experts = shared, ep_experts
        self.mesh_device = mesh_device
        self.banks = len(bank_workers(mesh_device))
        self.kt = inter // TILE
        self.h_tiles = hidden // TILE
        assert self.h_tiles % self.banks == 0, f"{self.h_tiles} hidden tiles do not split over {self.banks} banks"
        self.m = self.h_tiles // self.banks
        if readers is None:
            readers = max(2, -(-self.m // 4))
        assert self.m % readers == 0, f"{self.m} columns per bank do not split over {readers} readers"
        self.cols = self.m // readers
        self.E, self.k, self.inter = num_experts, k, inter
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.weight_dtype = weight_dtype
        placed = place_along_rows(mesh_device, {b: readers for b in range(self.banks)}, used)
        self.assign = [(c, b, r) for b in range(self.banks) for r, c in enumerate(placed[b])]
        self.cores = _range_set([c for c, _, _ in self.assign])

    def parts(self, act, src, down, shared_down, out, wait=None):
        """(kernels, cbs). src: the route buffer, or (one-program MoE, wait = (semaphore id, notifier count)) the
        router logits, whose top-k these cores take themselves."""
        assert wait is None or self.shared
        cb_in0, cb_w, cb_scr, cb_part, cb_lg, cb_out = 0, 1, 3, 4, 5, 16
        slots = self.k + 1 if self.shared else self.k
        kin = slots * self.kt
        lg_bytes = 64 * (self.E // TILE + 1)
        cbs = [
            _cb(self.cores, cb_in0, ttnn.bfloat16, 64, kin, tiny=True),
            _cb(self.cores, cb_w, self.weight_dtype, self.tile_bytes, kin * self.cols),
            _cb(self.cores, cb_scr, ttnn.uint16, lg_bytes, 1),
            _cb(self.cores, cb_part, ttnn.uint32, 8 * self.k, 1),
            _cb(self.cores, cb_lg, ttnn.bfloat16, lg_bytes, 1),
            _cb(self.cores, cb_out, ttnn.bfloat16, 64, self.cols, tiny=True),
        ]
        rd_rt, wr_rt = {}, {}
        for c, b, r in self.assign:
            c0 = r * self.cols
            rd_rt[(c.x, c.y)] = [down.buffer_address(), shared_down.buffer_address(), src.buffer_address(), b, c0, (b + r) & 0x3]
            wr_rt[(c.x, c.y)] = [act.buffer_address(), out.buffer_address(), b, c0] + ([src.buffer_address()] if wait else [])
        if self.ep_experts:
            assert wait is None
            for c, b, r in self.assign:
                wr_rt[(c.x, c.y)] = wr_rt[(c.x, c.y)] + [src.buffer_address()]  # the route record (live count)
        rd_ct = [cb_w, cb_scr, self.kt, self.m, self.cols, self.k, self.tile_bytes, self.E, cb_part]
        rd_ct += ttnn.TensorAccessorArgs(src).get_compile_time_args()
        wr_ct = [cb_in0, cb_out, slots, 2 * self.inter, self.cols, self.banks, cb_part, cb_lg, self.E]
        for t in (act, out) + ((src,) if wait or self.ep_experts else ()):
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        cp_ct = [cb_in0, cb_w, cb_out, kin, self.cols] + ([slots] if self.ep_experts else [])
        defines = [] if wait is None else [("MOE_FUSED", "1"), ("MOE_SEM", str(wait[0])), ("MOE_NOTIFIERS", str(wait[1]))]
        if not self.shared:
            defines.append(("NO_SHARED", "1"))
        defines += _ep_defines(self.ep_experts)
        return _kernels(self.cores, "dn", (rd_ct, rd_rt), (wr_ct, wr_rt), (cp_ct, None), defines), cbs

    def __call__(self, act, route, down, shared_down, out):
        kernels, cbs = self.parts(act, route, down, shared_down, out)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([down, shared_down, act, route, out], program)


class ExpertMoE:
    """ExpertGateUp + ExpertDown as ONE program on disjoint cores: the down cores take the router top-k themselves and
    stream their down weights while the gate|up cores run, then wait on a program semaphore that every gate|up core
    increments once its activations are written. (x, logits, up_gate, shared_up_gate, down, shared_down, act, out)."""

    SEM = 0

    def __init__(self, mesh_device, hidden, inter, num_experts, k, weight_dtype, gu_readers=4, dn_readers=2):
        used = set()
        self.gu = ExpertGateUp(mesh_device, hidden, inter, num_experts, k, weight_dtype, readers=gu_readers, used=used)
        self.dn = ExpertDown(mesh_device, hidden, inter, num_experts, k, weight_dtype, readers=dn_readers, used=used)
        coords = noc_coords(mesh_device)
        self.dn_noc = [coords[(c.x, c.y)] for c, _, _ in self.dn.assign]

    def __call__(self, x, logits, up_gate, shared_up_gate, down, shared_down, act, out):
        gk, gc = self.gu.parts(x, logits, up_gate, shared_up_gate, act, act, notify=(self.SEM, self.dn_noc))
        dk, dc = self.dn.parts(act, logits, down, shared_down, out, wait=(self.SEM, len(self.gu.assign)))
        sem = ttnn.SemaphoreDescriptor(id=self.SEM, core_type=ttnn.CoreType.WORKER, core_ranges=self.dn.cores, initial_value=0)
        program = ttnn.ProgramDescriptor(kernels=gk + dk, semaphores=[sem], cbs=gc + dc)
        return ttnn.generic_op([up_gate, shared_up_gate, down, shared_down, x, logits, act, out], program)
