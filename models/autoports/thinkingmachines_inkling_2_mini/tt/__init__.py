# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Knobs of the shared qwen36 fused MoE decode kernels this port uses (read when those modules are first imported):
# Inkling-2's softcapped expert activation, and HiFi2 expert matmuls (the full bfp8 weight; no 5-bit rounding).
import os as _os

_os.environ.setdefault("QWEN35_MOE_ACT", "inkling")
_os.environ.setdefault("QWEN35_MOE_DECODE_FIDELITY", "hifi2")
# H = 2304 is 9 tile columns per DRAM bank: the rows down op needs a reader count dividing 9 (3 serve up to 16 rows)
_os.environ.setdefault("QWEN35_MOE_DN_READERS", "3")
# 17 .. 32 rows (4 row groups: one column per core in the DST budget): 9 readers per bank, one call for the whole batch
_os.environ.setdefault("QWEN35_MOE_DN_READERS_WIDE", "9")
# reader placement of the fused MoE rows ops (traced sweep 2026-11-22, P150, layer 7, bfp8: per MoE layer at 1 / 4 rows
# 364 / 1162 us default -> 294 / 703 us with these three)
_os.environ.setdefault("QWEN35_MOE_GU_READERS", "3")  # (3 with GU_OFFROW=2: 1 / 4 rows 217 / 641 -> 202 / 555 us)
_os.environ.setdefault("QWEN35_MOE_DN_OFFROW", "1")
_os.environ.setdefault("QWEN35_MOE_DN_CHUNK", "2")
# the gate|up readers of the two banks sharing a NOC row spread over the free rows (traced, P150, layer 0, bfp8, the
# merged sub-shards: 1 / 4 / 8 rows 223 / 691 / 1215 -> 218 / 641 / 1110 us per MoE layer)
_os.environ.setdefault("QWEN35_MOE_GU_OFFROW", "2")
# the sigb router's top-k split over 8 cores per row (4 at 8 rows; <= 32 cores): the same selection and weights;
# TP4 traced B = 1 16.6 -> 16.1 ms, B = 8 29.2 -> 28.8
_os.environ.setdefault("QWEN35_ROUTE_SPLIT", "8")
# the sigb routing weights from the merged selection keys (no second sigmoid pass on the leader; bitwise)
_os.environ.setdefault("QWEN35_ROUTE_KEYS", "1")
# I2M_ROUTE_FAST: the route program's sigmoid on 32-bit operands, the leader's merges as register-resident threshold
# inserts, the weights' 2^59 / sum in 32-bit divisions (bitwise; route 6.9 / 7.3 -> 5.6 / 6.0 us at 2 / 4 rows, one die)
_os.environ.setdefault("QWEN35_ROUTE_FAST", _os.environ.get("I2M_ROUTE_FAST", "1"))
# the prefill MoE dispatch over many cores (qwen36 dispatch_mc; bit-identical to moe_sorted_dispatch)
_os.environ.setdefault("QWEN35_DISPATCH_MC", "1")
# decode MoE fixed costs (one-die bench of the per-die shards, eager us gate|up / down, all bitwise):
# I2M_MOE_GU_KCH: gate|up blocks pushed as K-row chunks, the compute starts on a block's first rows (TP2 B=1 63 -> 54)
_os.environ.setdefault("QWEN35_MOE_GU_KCH", _os.environ.get("I2M_MOE_GU_KCH", "4"))
_os.environ.setdefault("QWEN35_MOE_REM_KCH", _os.environ.get("I2M_MOE_REM_KCH", "2"))
# I2M_MOE_GU_OFFROW1: one-bank gate|up groups (TP2 / TP4) whose bank shares its reader row with another bank's: readers
# on the next row inward reading on NOC1 (TP2 4-row verify 128 -> 120 with KCH)
_os.environ.setdefault("QWEN35_MOE_GU_OFFROW1", _os.environ.get("I2M_MOE_GU_OFFROW1", "1"))
# I2M_MOE_DN_SKIPPAD: TP4's zero down K pad row not read from DRAM (TP4 down 17.7 / 36.4 -> 16.8 / 34.0 at 1 / 4 rows)
_os.environ.setdefault("QWEN35_MOE_DN_SKIPPAD", _os.environ.get("I2M_MOE_DN_SKIPPAD", "1"))
# I2M_MOE_GU_EAST: one die's gate|up group of banks 0 / 4 (the OFFROW=2 group) read from the east bank's columns (its
# readers beside the other groups' west readers ended 6-10 us late; one-die gate|up 103.3 / 236.9 -> 96.6 / 232.1 us)
_os.environ.setdefault("QWEN35_MOE_GU_OFFROW_X", "east" if _os.environ.get("I2M_MOE_GU_EAST", "1") == "1" else "")
# I2M_MOE_DN_SLOT: down weight slots pushed / multiplied one by one as they land (down -0.2 / -0.35 us at TP4 / TP2)
_os.environ.setdefault("QWEN35_MOE_DN_SLOT", _os.environ.get("I2M_MOE_DN_SLOT", "1"))

# I2M_L1_CHECK=1 (debug): before every generic_op, compare the program's circular buffers per core with the lowest
# allocated L1 address, CBs over an L1 tensor excluded (fast runtime mode does not check CB / L1-buffer clashes: an
# overlap hangs or corrupts)
if _os.environ.get("I2M_L1_CHECK") == "1":
    import traceback as _tb

    import ttnn as _ttnn

    _gen = _ttnn.generic_op
    _seen = set()

    def _checked(io, prog, *a, **k):
        dev = next((t.device() for t in io if t.device() is not None), None)
        if dev is not None and hasattr(prog, "cbs"):  # (a MeshProgramDescriptor, e.g. tt/ccl.py's: not checked)
            per = {}
            for cb in prog.cbs:
                if cb.has_buffer():  # (backed by an L1 tensor: its memory is that buffer's, not the CB region)
                    continue
                for r in cb.core_ranges.ranges():
                    for x in range(r.start.x, r.end.x + 1):
                        for y in range(r.start.y, r.end.y + 1):
                            per[(x, y)] = per.get((x, y), 0) + cb.total_size
            cbs = max(per.values()) if per else 0
            base = _ttnn.get_allocator_base_address(dev, _ttnn.BufferType.L1)
            mv = _ttnn.get_memory_view(dev, _ttnn.BufferType.L1)
            # (block_table addresses are offsets from the allocator base)
            low = min((base + int(b["address"]) for b in mv.block_table
                       if str(b.get("allocated", "")).lower() in ("yes", "true", "1")), default=None)
            site = next((f"{f.filename.split('/')[-1]}:{f.lineno}" for f in reversed(_tb.extract_stack()[:-1])
                         if "thinkingmachines_inkling_2_mini/tt/" in f.filename), "?")
            key = (site, cbs, low)
            if key not in _seen:
                _seen.add(key)
                over = low is not None and base + cbs > low
                print(f"L1CHECK {'OVERLAP ' if over else ''}{site}: CBs {cbs / 1024:.0f} KB from {base:#x} -> "
                      f"{base + cbs:#x}, lowest L1 buffer {low if low is None else hex(low)}", flush=True)
        return _gen(io, prog, *a, **k)

    _ttnn.generic_op = _checked

# I2M_RING_KB (default 32): the Tensix kernel-config ring (kernel binaries, runtime args, CB configs of the programs in
# flight; 69 KB by default) grown by this many KB, the CB space below the L1 buffers shrunk by as much: a decode program
# whose kernels do not fit beside the running ones waits for them, then loads with every core idle (aprep -> SDPA, the
# SDPA's ~57 KB; pre -> attention gates; agate -> o_proj). Traced one die, position 300: B=1 / B=4 / 4-row verify
# 17.68 / 28.71 / 31.24 -> 17.18 / 28.21 / 30.76 ms (+16 / +64 KB: B=1 17.32 / 17.07); logits bitwise (B=1, B=4,
# verify). Every mesh this process opens without its own worker_l1_size gets it; the served profiles pass the same size
# (tt-model.yaml additional_config tt.worker_l1_size, the &i2m_worker_l1 anchor: keep the two in step).
RING_KB = int(_os.environ.get("I2M_RING_KB", "32"))
WORKER_L1_DEFAULT = 1461248  # Blackhole: 1536 KB - the unreserved base (MEM_MAP_END + the 69 KB ring)
WORKER_L1_SIZE = WORKER_L1_DEFAULT - RING_KB * 1024
if RING_KB:
    import ttnn as _ttnn_r

    _open = _ttnn_r.open_mesh_device

    def _open_ring(*a, **k):
        if not k.get("worker_l1_size"):
            k["worker_l1_size"] = WORKER_L1_SIZE
        return _open(*a, **k)

    _ttnn_r.open_mesh_device = _open_ring
