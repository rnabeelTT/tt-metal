# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""I2M_PREFETCH (default 1; meshes of up to I2M_PREFETCH_DIES dies, default 2): the decode q|k|v with the q | r | s | g weight streamed ahead by the DRISC
tensor prefetcher into a global CB (ttnn.experimental.start_tensor_prefetcher), during the glue before the MoE.

* No copy of the weight: the model's interleaved [K, N] q weight (N / 32 % 8 == 0) is byte-identical to a legacy
  WIDTH_SHARDED DRAM tensor with shard [K, N / 8] (bank b: columns b, b + 8, ..., K-row-major), the prefetcher's
  K-row-major layout. krow_view() makes that view at the weight's own address for the prefetch request (the request
  keeps only the address).
* Receivers: R per bank (R | the bank's q columns: 13 on one die and TP2), receiver r of bank b holding bank-local
  columns [r npr, (r + 1) npr) (global n = b + 8 c); the GCB holds the whole q (one page per kbq k rows).
* k and v stream from DRAM (kernels/ilstream_reader.cpp's loop) on each bank's nearest receivers, one column each.
* Compute: kernels/ms_compute.cpp's per-column K order (bitwise with tt/mstream.py).
* Order: each q|k|v call queues its request right before it runs (captured into the trace: replayed at every
  execute_trace); the prefetcher fills the ring as the previous call frees it, i.e. during that layer's attention
  glue, o_proj, post and gates. Every consumer pops exactly one request (the FIFO guard below keeps them one to one).

Kernels: kernels/qpf_reader.cpp, qpf_compute.cpp, qpf_writer.cpp, qpf_rcb.hpp."""

from __future__ import annotations

import contextlib
import os

import ttnn
from models.demos.blackhole.qwen36.tt.moe_decode.experts import _range_set, place_banks_offrow

from . import mstream as MS
from .head import KDIR

# one die, traced (I2M_PREFETCH 0 -> 1, logits bitwise): B=1 17.28 -> 16.12 ms, B=4 24.91 -> 23.98, verify 28.63 -> 27.38
# (q|k|v 51.3 -> 20.5 us per call with q prefetched); TP2 traced (dies 2+3, logits bitwise): B=1 11.89 -> 11.58 ms,
# B=4 16.08 -> 15.86, verify 18.21 -> 17.88
ON = os.environ.get("I2M_PREFETCH", "1") == "1"
DIES = int(os.environ.get("I2M_PREFETCH_DIES", "2"))
KBQ = int(os.environ.get("I2M_PREFETCH_KBQ", "24"))
RING = int(os.environ.get("I2M_PREFETCH_RING", "0"))  # (GCB pages; 0: the whole q)


RCB = 31


def dram_banks():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])


def krow_mc(k, n):
    """the legacy WIDTH_SHARDED K-row-major memory config whose bank b holds an interleaved [K, n] weight's columns
    b, b + 8, ... (bank-local tile (k, c) at page k n / 8 + c: byte-identical to the interleaved buffer)"""
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                             ttnn.ShardSpec(dram_banks(), [k, n // 8], ttnn.ShardOrientation.ROW_MAJOR))


def _untracked(dev):
    """allocations outside the trace allocation tracker's accounting (a no-op without TT_METAL_TRACE_ALLOC_TRACKING)"""
    scope = getattr(ttnn, "corruptible_allocation_scope", None)
    return scope(dev) if scope is not None else contextlib.nullcontext()


def krow_view(w, fn):
    """The no-copy view: run fn(view) with `view` a live WIDTH_SHARDED (K-row-major) tensor over w's own DRAM bytes ->
    (the interleaved tensor that replaces w: same address, same bytes; w itself is deallocated, ok: fn ran). Python has
    no API to alias a buffer with another spec, so the allocator is used: w's range is freed, the view allocated (DRAM
    is first fit, bottom up: it lands there unless a lower free hole fits; if not, fn is skipped and ok is False), fn
    queues its prefetch requests (the request keeps only the address), the view is freed and w's range re-taken.
    Allocation never writes DRAM.
    Trace safety: the range is the weight's from its load on (between the free and the re-take only the view, at the
    same range, is allocated), so no trace's intermediates ever sit there and the re-taken weight is as safe as the
    original. The allocations are outside the trace allocation tracker's accounting (they are new buffers made during
    a later capture, which it would flag at every earlier trace's replay)."""
    dev, shape, dt = w.device(), w.shape, w.dtype
    a = w.buffer_address()
    with _untracked(dev):
        ttnn.deallocate(w)
        v = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, dev,
                                           krow_mc(int(shape[-2]), int(shape[-1])))
    ok = v.buffer_address() == a
    try:
        if ok:
            fn(v)
    finally:
        with _untracked(dev):
            ttnn.deallocate(v)
            w2 = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, dev,
                                                ttnn.DRAM_MEMORY_CONFIG)
        assert w2.buffer_address() == a, f"weight re-taken at {w2.buffer_address():#x}, not {a:#x}"
    return w2, ok


class GcbQKV:
    def __init__(self, mesh, k, nq, nkv, R=13, kvr=4, kbq=8, kb=8, nblk=7, ring_blocks=None, h=4, kv_split=False, fidelity=ttnn.MathFidelity.HiFi4):
        """nq / nkv: q and k (= v) output columns of one die; R receivers per bank (divides nq / 256); kvr of them per
        bank also stream k and v (divides nkv / 256); ring_blocks: GCB pages (default: the whole q, kt / kbq)."""
        self.mesh, self.k, self.kt = mesh, k, k // 32
        self.nq, self.nkv, self.R, self.kvr, self.kbq, self.kb, self.nblk, self.h = nq, nkv, R, kvr, kbq, kb, nblk, h
        cq, ckv = nq // 256, nkv // 256  # columns per bank
        assert cq % R == 0 and ckv % kvr == 0 and self.kt % kbq == 0 and self.kt % kb == 0
        self.npr, self.kvc = cq // R, ckv // kvr
        placed, noc1 = place_banks_offrow(mesh, {b: R for b in range(8)})
        # an off-row bank's cores read DRAM on NOC1 (place_banks_offrow; on NOC0 its k / v reader straggled at ~31 us
        # vs ~9); their GCB credits go back from the writer on NOC0 (a NOC1 credit atomic never acks: the GCB's sender
        # coordinates are NOC0's)
        self.noc1_banks = set(noc1)
        self.b2r = []
        self.assign = {}  # (x, y) -> (bank, receiver index r)
        for b in range(8):
            crs = _range_set(placed[b])
            self.b2r.append((b, crs))
            for r, c in enumerate(ttnn.corerange_to_cores(crs, row_wise=True)):  # (the GCB's receiver order)
                self.assign[(c.x, c.y)] = (b, r)
        self.cores = _range_set([ttnn.CoreCoord(x, y) for x, y in self.assign])
        # kv_split: k and v columns on different receivers (2 kvr per bank, one column of one tensor each);
        # else each of kvr receivers streams its k and v columns
        self.kv_split = kv_split
        nkr = 2 * kvr if kv_split else kvr
        assert nkr <= R
        # the receivers of each bank that also stream k / v: its nearest ones (placement order)
        self.kv_rb = {}
        for b in range(8):
            order = [(c.x, c.y) for c in ttnn.corerange_to_cores(_range_set(placed[b]), row_wise=True)]
            self.kv_rb[b] = [order.index((c.x, c.y)) for c in placed[b][:nkr]]
        self.fidelity = fidelity
        self.qblk = kbq * self.npr
        self.page = self.qblk * 1088
        self.nqb = self.kt // kbq
        self.ring_blocks = ring_blocks or self.nqb
        self.gcb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(
            mesh, self.b2r, self.ring_blocks * self.page)

    def request(self, view):
        """queue the q prefetch (view: krow_view's view of the q weight); captured when the CQ is capturing a trace"""
        ttnn.experimental.queue_tensor_prefetcher_request(self.mesh, [(view, self.nqb)], global_cb=self.gcb,
                                                          capture_into_trace=True)

    def __call__(self, xs, ws, rows, drain=False, out_mc=ttnn.DRAM_MEMORY_CONFIG, kv=True, q=True):
        """xs: [xq, xk, xv] [1, 1, rows, K] bf16; ws: [w_q, w_k, w_v] interleaved (w_q's bytes are not read: its pages
        come through the GCB) -> [q fp32, k bf16, v bf16]. drain: only pop the GCB (the stream-rate test)."""
        hmin = MS.knobs(self.mesh.get_num_devices(), self.kt)[0]  # (tt/mstream.py's activation tile rows: bitwise)
        h = next(v for v in (hmin, 8, 16, 32) if v >= rows) if MS.TINY else 32
        kt = self.kt
        outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, n]), dt, ttnn.TILE_LAYOUT, self.mesh, out_mc)
                for n, dt in ((self.nq, ttnn.float32), (self.nkv, ttnn.bfloat16), (self.nkv, ttnn.bfloat16))]
        cb_x, cb_w, cb_wq, cb_oq, cb_o = 0, 1, 2, 16, 17
        fmt = lambda i, d, page, pages, th=32: ttnn.CBDescriptor(  # noqa: E731
            total_size=page * pages, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=d, page_size=page,
                                                        tile=ttnn.TileDescriptor(th, 32) if th < 32 else None)])
        gq = ttnn.CBDescriptor()
        gq.total_size = self.ring_blocks * self.page
        gq.core_ranges = self.cores
        gq.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=cb_wq, data_format=ttnn.bfloat8_b, page_size=1088)]
        gq.remote_format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=RCB, data_format=ttnn.bfloat8_b,
                                                                page_size=self.page)]
        gq.set_global_circular_buffer(self.gcb)
        cbs = [fmt(cb_x, ttnn.bfloat16, 64 * h, kt * 3, h), fmt(cb_w, ttnn.bfloat8_b, 1088, self.nblk * self.kb),
               gq, fmt(cb_oq, ttnn.float32, 4096 * h // 32, 2 * self.npr, h), fmt(cb_o, ttnn.bfloat16, 2048 * h // 32, 2, h)]
        rd_ct = [cb_wq, cb_w, kt, 1088, self.nkv // 256, self.nblk, self.kb, 1, self.qblk, RCB]
        wr_ct = [cb_x, cb_oq, cb_o, kt, rows, h]
        wr_ct += ttnn.TensorAccessorArgs(xs[0]).get_compile_time_args()
        wr_ct += ttnn.TensorAccessorArgs(outs[0]).get_compile_time_args()
        cp_ct = [cb_x, cb_wq, cb_w, cb_oq, cb_o, kt, self.kbq, self.npr, self.kb, 1]
        rd, wr, cp = {}, {}, ttnn.RuntimeArgs()
        for (x, y), (b, r) in self.assign.items():
            col0 = b + 8 * r * self.npr
            if drain:
                rd[(x, y)] = [self.nqb, b, 0, 0]
                wr[(x, y)] = [0]
                cp[x][y] = [0]
                continue
            segs_w = [(xs[0], outs[0], col0, self.npr, 0)] if q else []
            segs_c = [0, self.npr] if q else []
            nqb = self.nqb if q else 0
            rd[(x, y)] = [nqb, b, 0, 0]
            if kv and r in self.kv_rb[b]:
                ri = self.kv_rb[b].index(r)
                if self.kv_split:
                    mem = [1 + ri // self.kvr]
                    j = (ri % self.kvr) * self.kvc
                else:
                    mem, j = [1, 2], ri * self.kvc
                rd[(x, y)] = [nqb, b, (b + j) & 0x3, len(mem)]
                for i in mem:
                    rd[(x, y)] += [ws[i].buffer_address(), j, self.kvc]
                    segs_w.append((xs[i], outs[i], b + 8 * j, self.kvc, 1))
                    segs_c += [1, self.kvc]
            wr[(x, y)] = [len(segs_w)] + [v for xx, o, c0, n, kd in segs_w
                                          for v in (xx.buffer_address(), o.buffer_address(), c0, n, kd)]
            cp[x][y] = [len(segs_w)] + segs_c
        # (fixed lengths: the program cache keys on the descriptor, not on runtime-arg counts)
        for d, n in ((rd, 10), (wr, 16)):
            for key in d:
                d[key] = d[key] + [0] * (n - len(d[key]))
        for (x, y), (b, _) in self.assign.items():
            wr[(x, y)].append(self.nqb if (q or drain) else 0)
        for (x, y) in self.assign:
            cp[x][y] = list(cp[x][y]) + [0] * (7 - len(cp[x][y]))
        if drain:
            rd_ct.append(1)
        else:
            rd_ct.append(0)

        def rt_for(args, cores):
            rt = ttnn.RuntimeArgs()
            for c in ttnn.corerange_to_cores(cores, row_wise=True):
                rt[c.x][c.y] = args[(c.x, c.y)]
            return rt

        n1 = [ttnn.CoreCoord(x, y) for (x, y), (b, _) in self.assign.items() if b in self.noc1_banks]
        n0 = [ttnn.CoreCoord(x, y) for (x, y), (b, _) in self.assign.items() if b not in self.noc1_banks]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = []
        for cl, rnoc, wnoc in ((n0, ttnn.NOC.NOC_0, ttnn.NOC.NOC_1), (n1, ttnn.NOC.NOC_1, ttnn.NOC.NOC_0)):
            if not cl:
                continue
            cores = _range_set(cl)
            ack = 1 if rnoc == ttnn.NOC.NOC_1 else 0
            assert not ack or self.ring_blocks == self.nqb, "credits from the writer need the whole q in the ring"
            kernels += [
                ttnn.KernelDescriptor(kernel_source=str(KDIR / "qpf_reader.cpp"), core_ranges=cores, compile_time_args=rd_ct + [ack],
                                      runtime_args=rt_for(rd, cores),
                                      config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=rnoc)),
                ttnn.KernelDescriptor(kernel_source=str(KDIR / "qpf_writer.cpp"), core_ranges=cores, compile_time_args=wr_ct + [ack, RCB, 1 if drain else 0],
                                      runtime_args=rt_for(wr, cores),
                                      config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=wnoc)),
            ]
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "qpf_compute.cpp"), core_ranges=self.cores, compile_time_args=cp_ct, runtime_args=cp,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True,
                                                math_approx_mode=False)))
        io = [xs[0], xs[1], xs[2], ws[1], ws[2]] + outs
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return outs


_STATE = {}  # id(mesh) -> {"mesh", "objs": {key: GcbQKV}, "cap": capturing a trace}

# FIFO balance: the prefetcher pushes requests into a consumer's GCB in queue order and each consumer call pops one
# whole request, so requests and consumer calls must stay one to one. qkv() queues the request right before its
# consumer; it counts each one queued without its consumer:
#   * eagerly (the request was sent at once): an orphan, drained (a pop-only consumer call) before that GCB's next
#     consumer and at stop(), so a failed eager call never shifts the later layers' weights;
#   * during a trace capture (the request is captured): the capture must end balanced; end_trace_capture releases an
#     unbalanced trace and raises (its every replay would add an orphan). A capture abandoned before its end sends
#     nothing (release_trace drops the captured requests).
# A trace replay re-sends its requests at execute_trace and the device runs the whole trace, so replays stay balanced;
# a device hang / reset needs a new mesh (a new state, the prefetcher restarted).
# When the view or the request fails, the call takes today's DRAM path (tt/mstream.py) instead.


def _hook_capture():
    if getattr(ttnn, "_i2m_pf_hooked", False):
        return
    begin, end = ttnn.begin_trace_capture, ttnn.end_trace_capture

    def _begin(mesh, *a, **k):
        st = _STATE.get(id(mesh))
        if st is not None:
            for pf in st["objs"].values():  # (an orphan left in the FIFO would shift every replay's weights)
                resync(pf)
                pf.cap_bal = 0
            st["cap"] = True
        return begin(mesh, *a, **k)

    def _end(mesh, tid, *a, **k):
        r = end(mesh, tid, *a, **k)
        st = _STATE.get(id(mesh))
        if st is not None:
            st["cap"] = False
            bad = [pf.cap_bal for pf in st["objs"].values() if pf.cap_bal]
            if bad:
                ttnn.release_trace(mesh, tid)
                raise RuntimeError(f"I2M_PREFETCH: trace captured with {bad} unconsumed q prefetch requests "
                                   "(released: its replays would desync the prefetcher FIFO)")
        return r

    ttnn.begin_trace_capture, ttnn.end_trace_capture = _begin, _end
    ttnn._i2m_pf_hooked = True


def qkv_prefetch(mesh, k, nq, nkv, fidelity):
    """The shared prefetching q|k|v for these shapes (starts the mesh's prefetcher once), or None (unsupported)."""
    nd = mesh.get_num_devices()
    cq, ckv = nq // 256, nkv // 256
    if nd > min(DIES, 2) or nq % 256 or nkv % 256 or not ttnn.experimental.is_tensor_prefetcher_supported(mesh):
        return None
    R = max(d for d in range(1, 14) if cq % d == 0)
    if 2 * ckv > R:
        return None
    _hook_capture()
    st = _STATE.get(id(mesh))
    if st is None:
        ttnn.experimental.start_tensor_prefetcher(mesh)
        st = _STATE[id(mesh)] = {"mesh": mesh, "objs": {}, "cap": False}
    key = (k, nq, nkv, fidelity)
    if key not in st["objs"]:
        kt = k // 32
        kbq = KBQ if kt % KBQ == 0 else kt
        pf = GcbQKV(mesh, k, nq, nkv, R=R, kvr=ckv, kbq=kbq, ring_blocks=RING or None, kv_split=True,
                    fidelity=fidelity)
        pf.orphans, pf.cap_bal, pf.last = 0, 0, None
        st["objs"][key] = pf
    return st["objs"][key]


def resync(pf):
    """Drain pf's orphan requests (eagerly queued without their consumer): one pop-only call each (the drain reads no
    activation or k / v weight: its io tensors are persistent dummies and the last call's k / v weights)."""
    if not pf.orphans or pf.last is None:
        return
    if getattr(pf, "dummy", None) is None:
        pf.dummy = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, pf.k]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                                   pf.mesh, ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    while pf.orphans:
        pf(pf.dummy, pf.last, 1, drain=True)
        pf.orphans -= 1


def stop(mesh):
    """Stop the mesh's prefetcher (teardown, after the last consumer; orphan requests drained first)."""
    st = _STATE.pop(id(mesh), None)
    if st is not None:
        for pf in st["objs"].values():
            resync(pf)
        ttnn.synchronize_device(mesh)
        ttnn.experimental.stop_tensor_prefetcher(mesh)


def qkv(pf, w, xs, rows):
    """One q|k|v application: queue its q prefetch (through the no-copy view; w.w_q is re-taken at the same address),
    then run the consumer -> [q fp32, k bf16, v bf16]; today's DRAM path if the request cannot be queued."""
    st = _STATE.get(id(pf.mesh))
    cap = st is not None and st["cap"]
    if not cap:
        resync(pf)
    queued = []

    def req(v):
        pf.request(v)
        queued.append(1)
        pf.last = list(w.ms_w)  # (the k / v weights a drain call can name)
        if cap:
            pf.cap_bal += 1
        else:
            pf.orphans += 1

    try:
        w.w_q, _ = krow_view(w.w_q, req)
    finally:
        w.ms_w[0] = w.w_q
    if not queued:  # (the view did not land on the weight: no request, today's path)
        return w.ms(xs, w.ms_w, rows)
    out = pf(xs, w.ms_w, rows)
    if cap:
        pf.cap_bal -= 1
    else:
        pf.orphans -= 1
    return out
