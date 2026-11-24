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
# the prefill MoE dispatch over many cores (qwen36 dispatch_mc; bit-identical to moe_sorted_dispatch)
_os.environ.setdefault("QWEN35_DISPATCH_MC", "1")

# I2M_L1_CHECK=1 (debug): before every generic_op, compare the program's circular buffers per core with the lowest
# allocated L1 address (fast runtime mode does not check CB / L1-buffer clashes: an overlap hangs or corrupts)
if _os.environ.get("I2M_L1_CHECK") == "1":
    import traceback as _tb

    import ttnn as _ttnn

    _gen = _ttnn.generic_op
    _seen = set()

    def _checked(io, prog, *a, **k):
        dev = next((t.device() for t in io if t.device() is not None), None)
        if dev is not None:
            per = {}
            for cb in prog.cbs:
                for r in cb.core_ranges.ranges():
                    for x in range(r.start.x, r.end.x + 1):
                        for y in range(r.start.y, r.end.y + 1):
                            per[(x, y)] = per.get((x, y), 0) + cb.total_size
            cbs = max(per.values()) if per else 0
            base = _ttnn.get_allocator_base_address(dev, _ttnn.BufferType.L1)
            mv = _ttnn.get_memory_view(dev, _ttnn.BufferType.L1)
            low = min((int(b["address"]) for b in mv.block_table
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
