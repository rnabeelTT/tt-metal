# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-token fused MoE decode for the Qwen3.5-MoE ports (Qwen35MoE blocks): router matmul -> ExpertGateUp ->
ExpertDown -> the layer's reduce-scatter, in 3 ops + the collective instead of ~21 (see experts.py).

What a port provides: its routed expert tensors as ttnn already holds them for prefill (``up_gate`` [1, E, H, 2 I]
packed [up_d | gate_d] per TP shard, ``down`` [1, E, I, H] per shard; DRAM interleaved, bfp8 or bfp4), with the
weights rounded by :func:`round_sig` to the LoFi operand width (:data:`WEIGHT_SIG_BITS`) before upload, and the HF
router / shared-expert weights. Built here (decode-only, small): the router weight with the shared-expert gate as
column E ([H, E + 32] BF16), the shared expert in the routed layout, the ops and the model-wide L1 buffers.

Numerics: custom_mm multiplies the weight operand (srcA) at 5 significant bits and the activation (srcB) at 7, so the
weights are stored rounded to 5 significant bits (unbiased; a bfp8 element keeps up to 7) and the gate|up outputs are
rounded to 7 before the down projection; FP32 accumulation. The router stays a ttnn matmul (BF16 weight, the port's
fidelity).
"""

from __future__ import annotations

import torch

import ttnn

from .experts import TILE, ExpertDown, ExpertGateUp, MoEDecodeBuffers

WEIGHT_SIG_BITS = 5  # significant bits custom_mm's LoFi multiply reads from the weight operand (srcA)
CACHE_SUFFIX = f".r{WEIGHT_SIG_BITS}"


def round_sig(t: torch.Tensor, bits: int = WEIGHT_SIG_BITS) -> torch.Tensor:
    """Round to nearest (half away from zero) with `bits` significant bits, on the FP32 bit pattern."""
    drop = 24 - bits  # FP32 has 24 significant bits
    i = t.float().contiguous().view(torch.int32)
    i = (i + (1 << (drop - 1))) & ~((1 << drop) - 1)
    return i.view(torch.float32)


def supported(num_experts, hidden, inter_local, dtype, banks=8):
    return (
        dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b)
        and (2 * inter_local // TILE) % banks == 0
        and (hidden // TILE) % banks == 0
        and num_experts % TILE == 0
    )


class FusedMoEDecode:
    """Per-layer decode state + the model-wide ops/buffers (cached on `args`)."""

    def __init__(self, mesh_device, args, *, hidden, inter_local, num_experts, k, tp, dtype, router_weight,
                 shared_gate_weight, shared_gate_proj, shared_up_proj, shared_down_proj, cache_name, router_ckc):
        """router_weight [E, H], shared_gate_weight [1, H], shared_{gate,up}_proj [S, H], shared_down_proj [H, S]
        (HF layouts, S = I * tp); cache_name(name) -> tensor cache path or None."""
        self.mesh_device = mesh_device
        self.tp = tp
        self.router_ckc = router_ckc
        E, H, Il = num_experts, hidden, inter_local
        rep = ttnn.ReplicateTensorToMesh(mesh_device) if tp > 1 else None
        shard = (lambda dim: ttnn.ShardTensorToMesh(mesh_device, dim=dim)) if tp > 1 else (lambda dim: None)

        def upload(t, dtype_, name, mapper):
            kw = {"mesh_mapper": mapper} if mapper is not None else {}
            return ttnn.as_tensor(
                t.contiguous(),
                dtype=dtype_,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(name),
                **kw,
            )

        # router [H, E + 32]: column E = the shared-expert gate logit (read by ExpertGateUp)
        rw = torch.zeros(H, E + TILE)
        rw[:, :E] = router_weight.T.float()
        rw[:, E] = shared_gate_weight.reshape(-1).float()
        self.router_w = upload(rw, ttnn.bfloat16, "fused_decode.router", rep)
        grid_w = mesh_device.compute_with_storage_grid_size().x
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self.router_pc = tpc.create_matmul_1d_decode_progcfg(
            TILE, H, E + TILE, (E + TILE) // TILE, fp32_acc=True, grid_w=grid_w,
            in0_block_w=16 if (H // TILE) % 16 == 0 else None,
        )
        # shared expert in the routed layout: [1, 1, H, tp * 2 I] packed [up_d | gate_d] per shard, [1, 1, tp * I, H]
        g, u = shared_gate_proj.T.float(), shared_up_proj.T.float()  # [H, S]
        packed = torch.cat(
            [torch.cat([u[:, d * Il : (d + 1) * Il], g[:, d * Il : (d + 1) * Il]], dim=-1) for d in range(tp)], dim=-1
        )
        self.shared_up_gate = upload(round_sig(packed).reshape(1, 1, H, -1), dtype, "fused_decode.shared_up_gate" + CACHE_SUFFIX, shard(-1))
        self.shared_down = upload(
            round_sig(shared_down_proj.T.float()).reshape(1, 1, -1, H), dtype, "fused_decode.shared_down" + CACHE_SUFFIX, shard(-2)
        )
        cache = args.__dict__.setdefault("_qwen35_fused_moe_decode", {})
        key = (H, Il, E, k, dtype)
        if key not in cache:
            cache[key] = (
                ExpertGateUp(mesh_device, H, Il, E, k, dtype),
                ExpertDown(mesh_device, H, Il, E, k, dtype),
                MoEDecodeBuffers(mesh_device, H, Il, k),
            )
        self.gate_up, self.down_op, self.buf = cache[key]

    def __call__(self, x, up_gate, down):
        """x: the normed hidden [1, 1, 1, H] BF16 TILE (interleaved). Returns this device's partial sum (routed +
        shared) in the model-wide [1, 1, 1, H] buffer: consume it (the layer reduce) before the next layer's MoE."""
        logits = ttnn.linear(
            x, self.router_w, compute_kernel_config=self.router_ckc, program_config=self.router_pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        b = self.buf
        self.gate_up(x, logits, up_gate, self.shared_up_gate, b.act, b.route)
        ttnn.deallocate(logits)
        self.down_op(b.act, b.route, down, self.shared_down, b.out)
        return b.out
