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
    "QWEN35_MTP": "0",  # HYB_MTP sets it below (fused path)
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
    if os.environ.get("HYB_SWITCH_US"):  # GIL switch interval (default 5000 us): how long a Python thread may hold it
        import sys

        sys.setswitchinterval(float(os.environ["HYB_SWITCH_US"]) * 1e-6)
    for k, v in _ENV_OFF.items():
        os.environ[k] = v
    if os.environ.get("HYB_FUSED", "1") == "1":
        # fused MoE decode kernels on crafted logits (patch._forward_fused); one decode row (no multi-row ops)
        os.environ["QWEN35_MOE_FUSED_DECODE"] = "1"
        mtp = int(os.environ.get("HYB_MTP", "0"))  # MTP speculative decode: k drafts, verify width k + 1 rows
        os.environ["QWEN35_MTP"] = str(mtp)
        # rows ops up to the MTP verify width; >= 2 for the one-row decode through the 2-row kernels (HYB_SINGLE_ROWS)
        os.environ["QWEN35_MOE_FUSED_ROWS_MAX"] = str(max(2, 1 << mtp.bit_length()) if mtp else 2)
        if os.environ.get("HYB_ROWS"):  # multi-row verify width without the MTP head (e.g. DFlash: 4 rows)
            os.environ["QWEN35_MOE_FUSED_ROWS_MAX"] = os.environ["HYB_ROWS"]
        if os.environ.get("HYB_BOUNDARY", "0") == "1":  # 397B full model: slower and degraded text (2026-11-04)
            # production decode boundary: replicated residual, fabric all-reduce + add + norm in one program, the
            # router matmul and the next layer's in-projection inside it (patch._forward_fused handles "fabric")
            for k in ("QWEN35_DECODE_REPLICATED", "QWEN35_FABRIC_BOUNDARY", "QWEN35_FB_ROUTER"):
                os.environ.setdefault(k, "1")
            # FB_CONSUMER (next layer's in-projection inside the boundary) holds L1 that, at hidden 4096, clashes with
            # the prefill RMSNorm's static circular buffers (397B 4-layer: TT_THROW in the prefill warm-up)
            os.environ.setdefault("QWEN35_FB_CONSUMER", "0")
    for k, v in _ENV_ON.items():
        os.environ.setdefault(k, v)
    if os.environ.get("HYB_FETCH_ASYNC") == "1":  # the async fetch writes on CQ1: every mesh opens with two queues
        import ttnn

        _open = ttnn.open_mesh_device

        def open_mesh_device(*a, **kw):
            kw.setdefault("num_command_queues", 2)
            return _open(*a, **kw)

        ttnn.open_mesh_device = open_mesh_device
    from . import patch, seg

    seg.install()  # segmented decode traces (host step between per-layer segments)
    patch.install()
    return patch.HYB
