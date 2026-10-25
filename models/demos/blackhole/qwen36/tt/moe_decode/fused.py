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

from .experts import TILE, ExpertDown, ExpertGateUp, MoEDecodeBuffers, bank_workers, ep_offsets
from .union import ExpertDownRows, ExpertGateUpRows, RouteRows, RowsBuffers, act_buffer, upload_selection_bias

WEIGHT_SIG_BITS = 5  # significant bits custom_mm's LoFi multiply reads from the weight operand (srcA)
CACHE_SUFFIX = f".r{WEIGHT_SIG_BITS}"


def round_sig(t: torch.Tensor, bits: int = WEIGHT_SIG_BITS) -> torch.Tensor:
    """Round to nearest (half away from zero) with `bits` significant bits, on the FP32 bit pattern."""
    drop = 24 - bits  # FP32 has 24 significant bits
    i = t.float().contiguous().view(torch.int32)
    i = (i + (1 << (drop - 1))) & ~((1 << drop) - 1)
    return i.view(torch.float32)


def supported(num_experts, hidden, inter_local, dtype, banks=8):
    """The gate|up kernels read m = 2 I / 32 / banks up|gate columns per bank and expert, with the matmul result in DST
    (m even: m tiles; m odd: 2 m; + 1 at multi-row widths; <= 8 FP32 tiles): m in 1, 2, 3, 4, 6 (Ornith 35B TP4 / TP2 /
    one die = 1 / 2 / 4, 122B TP4 = 2). 7-bank parts (p100) keep the unfused path."""
    m = (2 * inter_local // TILE) // banks
    return (
        dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b)
        and (2 * inter_local // TILE) % banks == 0
        and m * (2 if m % 2 else 1) + 1 <= 8
        and (hidden // TILE) % banks == 0
        and num_experts % TILE == 0
        and num_experts <= 512  # router_topk.hpp route_union's expert mask
    )


class FusedMoEDecode:
    """Per-layer decode state + the model-wide ops/buffers (cached on `args`)."""

    def __init__(self, mesh_device, args, *, hidden, inter_local, num_experts, k, tp, dtype, router_weight,
                 shared_gate_weight, shared_gate_proj, shared_up_proj, shared_down_proj, cache_name, router_ckc,
                 router="softmax", selection_bias=None, down_dtype=None, router_logits=None, ep_experts=None,
                 gate_first=False):
        """router_weight [E, H], shared_gate_weight [1, H], shared_{gate,up}_proj [S, H], shared_down_proj [H, S]
        (HF layouts, S = I * tp); cache_name(name) -> tensor cache path or None.

        router "sigb" (GLM-4.5-Air): sigmoid + selection_bias ([E] e_score_correction_bias) top-k, normalized weights;
        router_logits(x) -> the port's FP32 [rows, E] router logits (its own router matmul); no shared-gate column.
        shared_* None: routed experts only (the port adds its shared expert). down_dtype: the routed down weights' dtype
        when it differs from the gate|up dtype. ep_experts (expert parallelism, Qwen3-Coder-30B): each die holds the
        whole experts [d * ep_experts, (d + 1) * ep_experts) as its up_gate / down tensors (inter_local = the full
        expert width), num_experts is the global count the router ranks, tp the die count; shared_gate_weight None
        means no shared-gate column logit (a zero column). gate_first: [gate | up] packed weights."""
        self.mesh_device = mesh_device
        self.tp = tp
        self.router_ckc = router_ckc
        self.args_ref = args
        self.router, self.router_logits = router, router_logits
        self.shared = shared_up_proj is not None
        self.ep_experts = ep_experts
        down_dtype = down_dtype or dtype
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

        if router == "sigb":
            assert router_logits is not None and selection_bias is not None
            self.bias = upload_selection_bias(mesh_device, selection_bias)
        else:
            # router [H, E + 32]: column E = the shared-expert gate logit (read by ExpertGateUp)
            self.bias = None
            rw = torch.zeros(H, E + TILE)
            rw[:, :E] = router_weight.T.float()
            if shared_gate_weight is not None:
                rw[:, E] = shared_gate_weight.reshape(-1).float()
            self.router_w = upload(rw, ttnn.bfloat16, "fused_decode.router", rep)
            grid_w = mesh_device.compute_with_storage_grid_size().x
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            self.router_pc = tpc.create_matmul_1d_decode_progcfg(
                TILE, H, E + TILE, (E + TILE) // TILE, fp32_acc=True, grid_w=grid_w,
                in0_block_w=16 if (H // TILE) % 16 == 0 else None,
            )
        if self.shared:
            # shared expert in the routed layout: [1, 1, H, tp * 2 I] packed [up_d | gate_d] per shard, [1, 1, tp * I, H]
            g, u = shared_gate_proj.T.float(), shared_up_proj.T.float()  # [H, S]
            packed = torch.cat(
                [torch.cat([u[:, d * Il : (d + 1) * Il], g[:, d * Il : (d + 1) * Il]], dim=-1) for d in range(tp)], dim=-1
            )
            self.shared_up_gate = upload(round_sig(packed).reshape(1, 1, H, -1), dtype, "fused_decode.shared_up_gate" + CACHE_SUFFIX, shard(-1))
            self.shared_down = upload(
                round_sig(shared_down_proj.T.float()).reshape(1, 1, -1, H), down_dtype, "fused_decode.shared_down" + CACHE_SUFFIX, shard(-2)
            )
        cache = args.__dict__.setdefault("_qwen35_fused_moe_decode", {})
        key = (H, Il, E, k, dtype, down_dtype, router, self.shared, ep_experts, gate_first)
        if key not in cache:
            ep = dict(ep_experts=ep_experts)
            cache[key] = (
                ExpertGateUp(mesh_device, H, Il, E, k, dtype, router=router, shared=self.shared, gate_first=gate_first, **ep),
                ExpertDown(mesh_device, H, Il, E, k, down_dtype, shared=self.shared, **ep),
                MoEDecodeBuffers(mesh_device, H, Il, k),
            )
            if ep_experts:
                cache[key + ("ep_offset",)] = ep_offsets(mesh_device, ep_experts)
        self.gate_up, self.down_op, self.buf = cache[key]
        self.ep_offset = cache.get(key + ("ep_offset",))
        self._cache, self._key, self._mesh = cache, key, mesh_device

    def rows_ops(self, rows):
        """(RouteRows, ExpertGateUpRows, ExpertDownRows, RowsBuffers) for `rows` decode rows, one set per model; build
        every width the server traces before capturing (host uploads)."""
        key = self._key + ("rows", rows)
        if key not in self._cache:
            H, Il, E, k, dtype, down_dtype, router, shared, ep_experts, gate_first = self._key
            ep = dict(ep_experts=ep_experts)
            self._cache[key] = (
                RouteRows(self._mesh, E, k, rows, router=router),
                ExpertGateUpRows(self._mesh, H, Il, E, k, dtype, rows, shared=shared, gate_first=gate_first, **ep),
                ExpertDownRows(self._mesh, H, Il, E, k, down_dtype, rows, shared=shared, **ep),
                RowsBuffers(self._mesh, H, Il, E, k, rows, dtype),
            )
        return self._cache[key]

    def _logits(self, x):
        if self.router_logits is not None:
            return self.router_logits(x)
        return ttnn.linear(
            x, self.router_w, compute_kernel_config=self.router_ckc, program_config=self.router_pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _shared_tensors(self, up_gate, down):
        """Routed-only: the routed tensors stand in for the shared ones (their addresses are passed, never read)."""
        return (self.shared_up_gate, self.shared_down) if self.shared else (up_gate, down)

    def rows(self, x, up_gate, down, out=None):
        """x: the normed hidden [1, 1, M, H] (2 <= M <= 32). Returns the model-wide [1, 1, M, H] partial-sum buffer, or
        `out` (a [1, 1, M, H] BF16 TILE tensor, e.g. allocated per call in L1 for the fabric boundary: rows >= Mp of its
        tiles are left as they were)."""
        m = int(x.shape[-2])
        route, gu, dn, b = self.rows_ops(m)
        logits = self._logits(x)
        route(logits, b.table, bias=self.bias)
        ttnn.deallocate(logits)
        H, Il, E, k = self._key[:4]
        sug, sdn = self._shared_tensors(up_gate, down)
        act = act_buffer(self._mesh, Il, E, k, m)
        gu(x, b.table, b.identity, up_gate, sug, act, ep_offset=self.ep_offset)
        dst = b.out if out is None else out
        dn(act, b.table, down, sdn, dst, ep_offset=self.ep_offset)
        ttnn.deallocate(act)
        return dst

    def router_consumer(self, fb):
        """The router matmul of one row as the consumer of a fabric boundary (fabric_boundary/ops.py consumer=; a
        DecodeLinear stream copy of the replicated BF16 router weight at its fidelity, built on first use)."""
        if getattr(self, "_dl_router", None) is None:
            from models.demos.blackhole.qwen36.tt.decode_linear.ops import decode_linear_for

            fid = getattr(self.router_ckc, "math_fidelity", ttnn.MathFidelity.HiFi4)
            self._dl_router = decode_linear_for(self.args_ref, self._mesh, self.router_w, fb, replicated=True,
                                                fidelity=fid)
        op, w = self._dl_router
        return op.consumer(w, ttnn.L1_MEMORY_CONFIG)

    def __call__(self, x, up_gate, down, logits=None):
        """x: the normed hidden [1, 1, 1, H] BF16 TILE (interleaved). Returns this device's partial sum (routed +
        shared) in the model-wide [1, 1, 1, H] buffer: consume it (the layer reduce) before the next layer's MoE.
        logits: the router output when the boundary computed it (router_consumer)."""
        if logits is None:
            logits = self._logits(x)
        b = self.buf
        sug, sdn = self._shared_tensors(up_gate, down)
        self.gate_up(x, logits, up_gate, sug, b.act, b.route, bias=self.bias, ep_offset=self.ep_offset)
        ttnn.deallocate(logits)
        self.down_op(b.act, b.route, down, sdn, b.out)
        return b.out
