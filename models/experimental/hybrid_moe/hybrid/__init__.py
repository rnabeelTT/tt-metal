# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hybrid MoE prototype overlay for the Qwen3.5 autoport inside the tt-model image.

Usage (before importing anything from models.autoports.*):

    import hybrid
    hybrid.install()            # env + patches; reads HYB_* env vars (see HANDOFF.md)

The device holds C expert slots per MoE layer; the router still scores all E experts. Every MoE call reads the routing
back, maps expert ids -> slots on the host (misses get slot 0 with weight 0) and adds the host-computed contribution of
the missed experts before the TP reduce.
"""

import os

# The hybrid path hooks the plain (unfused) MoE block. These must be set before the autoport modules are imported.
_ENV_OFF = {
    "QWEN35_MOE_FUSED_DECODE": "0",  # turned back on below when HYB_FUSED=1
    "QWEN35_ROUTER_TOPK_FUSED": "0",
    "QWEN35_ROUTER_DRAM_SHARDED": "0",  # router weight stays interleaved: its width is E while the experts hold C slots
    "QWEN35_ROUTER_GATE_OP": "0",
    "QWEN35_FB_ROUTER": "0",
    "QWEN35_FABRIC_BOUNDARY": "0",
    "QWEN35_DECODE_REPLICATED": "0",
    "QWEN35_MTP": "0",
    "QWEN35_VISION": "0",
    "QWEN35_PROGRAM_GUARD": "0",
    "QWEN35_BATCHED_PREFILL": "0",
    "QWEN35_PREFILL_TRACE": "0",
}

# Production decode fusions that do not touch the MoE block (serve-default.sh sets them); defaults, env may override.
_ENV_ON = {
    "QWEN35_ATTN_DECODE_OPS": "1",  # attention decode head prep / output gate ops
    "QWEN35_GDN_DECODE_OPS": "1",  # GDN decode conv + head step ops
    "QWEN35_ROWS_TOPK": "1",  # sampler top-k over the tile rows
}


def install():
    for k, v in _ENV_OFF.items():
        os.environ[k] = v
    if os.environ.get("HYB_FUSED", "1") == "1":
        # fused MoE decode kernels on crafted logits (patch._forward_fused); one decode row (no multi-row ops)
        os.environ["QWEN35_MOE_FUSED_DECODE"] = "1"
        os.environ["QWEN35_MOE_FUSED_ROWS_MAX"] = "1"
    for k, v in _ENV_ON.items():
        os.environ.setdefault(k, v)
    from . import patch, seg

    seg.install()  # segmented decode traces (host step between per-layer segments)
    patch.install()
    return patch.HYB
