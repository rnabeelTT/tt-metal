# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Knobs of the shared qwen36 fused MoE decode kernels this port uses (read when those modules are first imported):
# Inkling-2's softcapped expert activation, and HiFi2 expert matmuls (the full bfp8 weight; no 5-bit rounding).
import os as _os

_os.environ.setdefault("QWEN35_MOE_ACT", "inkling")
_os.environ.setdefault("QWEN35_MOE_DECODE_FIDELITY", "hifi2")
