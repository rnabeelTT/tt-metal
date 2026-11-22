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
_os.environ.setdefault("QWEN35_MOE_GU_READERS", "2")
_os.environ.setdefault("QWEN35_MOE_DN_OFFROW", "1")
_os.environ.setdefault("QWEN35_MOE_DN_CHUNK", "2")
