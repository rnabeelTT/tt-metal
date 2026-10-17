# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash Kimi Delta Attention (KDA) layer: prefill on the ttnn.experimental.kda kernels, decode on ttnn ops.

HF semantics (transformers glm5_next `Glm5NextTextLinearAttention`), 64 heads x 128, per token:
    qkv = silu(causal depthwise conv4([q_proj; k_proj; v_proj](x)))       (conv in fp32, tap 3 = current token)
    q, k = l2norm(q), l2norm(k) (eps 1e-6 inside the sqrt);  q *= 128**-0.5
    g = -5 * sigmoid(exp(A_log)[h] * (f_b(f_a(x)) + dt_bias))             (per key channel log-decay)
    beta = sigmoid(b_proj(x))
    S <- S * exp(g)[:, None];  delta = beta * (v - k^T S);  S <- S + k (x) delta;  o = q^T S
    out = o_proj( rmsnorm_128(o) * o_norm * sigmoid(g_b(g_a(x))) )

Prefill: `ttKDA` (models/demos/deepseek_v3_d_p, the Kimi-K3 KDA layer, bounded gate) for one user, one constructed
physical length per bucket, runtime `actual_start` / `actual_end` (non-aligned logical lengths), carried KdaState
between chunks of a long prompt. Decode: the recurrence step below on fixed-address per-slot state buffers
(trace-safe, batch of up to `max_batch` users), weights shared with the prefill layer (fused input projection,
folded decay projection, conv taps, o_norm, o_proj).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA

from .weight_cache import load_kda_weights_cached

DRAM = ttnn.DRAM_MEMORY_CONFIG
KDA_CHUNK = 32
GATE_LOWER_BOUND = -5.0


def _EXACT_DECAY_CKC(mesh_device):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def kda_config(tc: dict) -> KDAConfig:
    la = tc["linear_attn_config"]
    return KDAConfig(
        hidden_size=tc["hidden_size"],
        num_heads=la["num_heads"],
        head_k_dim=la["head_dim"],
        head_v_dim=la["head_dim"],
        conv_kernel_size=la["short_conv_kernel_size"],
        norm_eps=tc["rms_norm_eps"],
        use_full_rank_gate=False,
        gate_lower_bound=la["gate_lower_bound"],
    )


def kda_state_dict(layer_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint suffixes of one layer -> the layer-local names ttKDA loads (self_attn.* without the prefix)."""
    out = {}
    for name, t in layer_sd.items():
        if name.startswith("self_attn."):
            out[name[len("self_attn.") :]] = t
    return out


def uint32_scalar(mesh_device, value: int) -> ttnn.Tensor:
    """Caller-owned replicated UINT32 row-major scalar (ttKDA runtime bound); allocate before any trace capture."""
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


class _GlmPrefillKDA(ttKDA):
    """ttKDA whose decay projection reads a column slice of the fused projection.

    The upstream layer passes `in0_column_offset` to ttnn.linear (a matmul feature of a newer tt-metal than this
    branch carries); slicing first gives the same product.
    """

    def _compute_decay(self, projected: ttnn.Tensor, decay_rank_offset: int) -> ttnn.Tensor:
        rank = self.config.head_k_dim
        shape = list(projected.shape)
        start = [0] * len(shape)
        start[-1] = decay_rank_offset
        end = list(shape)
        end[-1] = decay_rank_offset + rank
        rank_in = ttnn.slice(projected, start, end, memory_config=DRAM)
        gate = ttnn.linear(
            rank_in,
            self.weights.decay_output_projection,
            bias=self.weights.decay_bias_flat,
            program_config=self.decay_program_config,
            memory_config=self.decay_memory_config,
            # exact (math_approx_mode=False) fused sigmoid: ttKDA's compute_config leaves approx mode at its default
            compute_kernel_config=_EXACT_DECAY_CKC(self.device),
        )
        ttnn.deallocate(rank_in)
        return gate


@dataclass
class KdaDecodeState:
    """Fixed-address per-slot decode state (trace inputs): recurrent [B,H,K,V] fp32, conv history 3 x [1,1,B,W]."""

    recurrent: ttnn.Tensor
    conv: list  # three bf16 TILE [1,1,B,W] tensors, oldest first (taps 0, 1, 2; tap 3 multiplies the current row)


class GlmKDA(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tc: dict,
        layer_sd: dict[str, torch.Tensor],
        layer_idx: int,
        *,
        prefill_lengths=(128, 512, 2048),
        max_batch: int = 32,
        tt_ccl=None,
        weight_cache=None,
        dtypes=None,
    ):
        self.mesh = mesh_device
        self.layer_idx = layer_idx
        self.cfg = kda_config(tc)
        tp = tuple(mesh_device.shape)[1]
        self.tp = tp
        self.H = self.cfg.num_heads // tp  # local heads
        self.K = self.cfg.head_k_dim
        self.V = self.cfg.head_v_dim
        self.W = self.H * (2 * self.K + self.V)  # local conv width (q|k|v)
        self.max_batch = max_batch
        # weight_cache: the layer's tt/weight_cache.WeightCache; dtypes: (input, output) projection dtypes (policy)
        self.weights = load_kda_weights_cached(mesh_device, self.cfg, layer_sd, weight_cache, *(dtypes or (None, None)))
        self.prefill_lengths = tuple(sorted(prefill_lengths))
        self._prefill = {
            n: _GlmPrefillKDA(
                mesh_device,
                self.cfg,
                weights=self.weights,
                layer_idx=layer_idx,
                tt_ccl=tt_ccl,
                sp_axis=0,
                tp_axis=1,
                program_config=KDAProgramConfig(
                    # the chunk-recurrence matmuls at HiFi4 (default HiFi2 truncates: attn output ~0.9 % low, review #2)
                    recurrence=KDARecurrenceProgramConfig(
                        affine_prefix_math_fidelity=ttnn.MathFidelity.HiFi4,
                        scan_math_fidelity=ttnn.MathFidelity.HiFi4,
                    ),
                ),
                active_seq_len=n,
            )
            for n in self.prefill_lengths
        }
        # column layout of the fused input projection per TP rank: [q | k | v | f_a (rank K) | g_direct (V*H) | b (H)]
        self.off_decay = self.W
        self.off_gate = self.W + self.K
        self.off_beta = self.off_gate + self.H * self.V
        self.proj_width = self.off_beta + self.H
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        # o_norm weight as a [1,1,1,V] fp32 row; l2norm-as-rmsnorm weights: rms_norm(x) * K**-0.5 = x / ||x||
        self.o_norm_w = ttnn.from_torch(
            layer_sd["self_attn.o_norm.weight"].float().reshape(1, 1, 1, self.V),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        self.q_norm_w = ttnn.from_torch(
            torch.full((1, 1, 1, self.K), self.K**-0.5 * self.K**-0.5),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        self.k_norm_w = ttnn.from_torch(
            torch.full((1, 1, 1, self.K), self.K**-0.5),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )

    # --------------------------------------------------------------------------------------------------- prefill
    def prefill_layer(self, n: int) -> _GlmPrefillKDA:
        return self._prefill[n]

    def prefill_bucket(self, length: int) -> int:
        for n in self.prefill_lengths:
            if length <= n:
                return n
        return self.prefill_lengths[-1]

    def new_prefill_state(self) -> KdaState:
        return self._prefill[self.prefill_lengths[0]].allocate_state(1)

    def prefill_inputs(self, start: int, length: int):
        """Runtime bounds of one chunk: (actual_start, actual_end) device scalars (input construction, outside the pass)."""
        return uint32_scalar(self.mesh, start), uint32_scalar(self.mesh, start + length)

    def prefill_forward(self, x: ttnn.Tensor, state: KdaState, bounds):
        """One user's chunk: x [1,1,n,hidden] (n = a constructed bucket), state carried from the previous chunk (or
        new_prefill_state()), bounds = prefill_inputs(start, length). Returns (y [1,1,n,hidden], new KdaState);
        rows past the chunk's length are padding (unspecified output)."""
        n = x.shape[-2]
        layer = self._prefill[n]
        a0, a1 = bounds
        h3 = ttnn.reshape(x, (1, n, x.shape[-1]))
        y, new_state = layer.forward(h3, state, actual_start=a0, actual_end=a1)
        return ttnn.reshape(y, (1, 1, n, y.shape[-1])), new_state

    # ---------------------------------------------------------------------------------------------------- decode
    def allocate_decode_state(self, batch: int | None = None) -> KdaDecodeState:
        B = batch or self.max_batch
        rec = ttnn.zeros(
            (B, self.H, self.K, self.V),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=DRAM,
        )
        conv = [
            ttnn.zeros(
                (1, 1, B, self.W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM
            )
            for _ in range(3)
        ]
        return KdaDecodeState(rec, conv)

    def write_slot(self, dstate: KdaDecodeState, slot: int, pstate: KdaState):
        """Copy one user's prefill carry (KdaState, batch 1) into decode slot `slot` (outside traces)."""
        rec = pstate.recurrent
        if rec.dtype != ttnn.float32:
            rec = ttnn.typecast(rec, ttnn.float32)
        B = dstate.recurrent.shape[0]
        _write_rows(dstate.recurrent, rec, slot, dim=0, n=B)
        # conv carry [1, 3, W] row-major (oldest row first) -> the three history rows of slot `slot`
        conv = ttnn.to_layout(pstate.convolution, ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1, 3, W]
        for j in range(3):
            row = ttnn.slice(conv, (0, j, 0), (1, j + 1, self.W), memory_config=DRAM)
            row = ttnn.reshape(row, (1, 1, 1, self.W))
            _write_rows(dstate.conv[j], row, slot, dim=2, n=B)
            ttnn.deallocate(row)
        ttnn.deallocate(conv)

    def _decode_host_inputs(self, positions):
        """Per-step decode inputs: positions[s] = position of slot s's token, -1 = inactive (state left unchanged)."""
        active = torch.tensor([p >= 0 for p in positions]).float().reshape(1, 1, len(positions), 1)
        return {
            "active": (active, ttnn.float32, ttnn.TILE_LAYOUT),
            "active_bf": (active, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        }

    def decode_inputs(self, positions):
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        return {
            k: ttnn.from_torch(t, dtype=dt, layout=lay, device=self.mesh, mesh_mapper=rep)
            for k, (t, dt, lay) in self._decode_host_inputs(positions).items()
        }

    def update_decode_inputs(self, bufs, positions):
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for k, (t, dt, lay) in self._decode_host_inputs(positions).items():
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=rep), bufs[k])

    @staticmethod
    def shift_conv_history(dstate: KdaDecodeState, cur: ttnn.Tensor, active_bf: ttnn.Tensor):
        """h0 <- h1, h1 <- h2, h2 <- current for active slots; inactive slots keep their history (fixed addresses)."""
        h0, h1, h2 = dstate.conv
        new = [
            ttnn.where(active_bf, h1, h0, memory_config=DRAM),
            ttnn.where(active_bf, h2, h1, memory_config=DRAM),
            ttnn.where(active_bf, cur, h2, memory_config=DRAM),
        ]
        for n, buf in zip(new, dstate.conv):
            ttnn.copy(n, buf)
            ttnn.deallocate(n)

    def decode_forward(self, x: ttnn.Tensor, dstate: KdaDecodeState, *, inputs) -> ttnn.Tensor:
        """x [1,1,B,hidden] (B = the state's batch) -> y [1,1,B,hidden]; updates dstate in place (trace-safe)."""
        B = x.shape[-2]
        H, K, V, W = self.H, self.K, self.V, self.W
        w = self.weights
        cc = self.ckc
        proj = ttnn.linear(x, w.input_projection, memory_config=DRAM, compute_kernel_config=cc)  # [1,1,B,P]

        # ---- short causal conv over (h0, h1, h2, current), fp32 accumulation, SiLU
        cur = ttnn.slice(proj, (0, 0, 0, 0), (1, 1, B, W), memory_config=DRAM)
        taps = w.convolution_taps  # four [1,1,W] bf16
        acc = ttnn.multiply(cur, _tap4(taps[3]), dtype=ttnn.float32, memory_config=DRAM)
        for j in range(3):
            acc = ttnn.add(
                acc,
                ttnn.multiply(dstate.conv[j], _tap4(taps[j]), dtype=ttnn.float32, memory_config=DRAM),
                memory_config=DRAM,
            )
        qkv = ttnn.silu(acc, memory_config=DRAM)
        # inactive slots: zero q / k / v (a select, so non-finite rows of an idle slot cannot reach its state)
        qkv_all = qkv
        qkv = ttnn.where(inputs["active"], qkv_all, 0.0, memory_config=DRAM)
        ttnn.deallocate(qkv_all)
        ttnn.deallocate(acc)
        self.shift_conv_history(dstate, cur, inputs["active_bf"])
        ttnn.deallocate(cur)

        q = ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, B, H * K), memory_config=DRAM)
        k = ttnn.slice(qkv, (0, 0, 0, H * K), (1, 1, B, 2 * H * K), memory_config=DRAM)
        v = ttnn.slice(qkv, (0, 0, 0, 2 * H * K), (1, 1, B, W), memory_config=DRAM)
        ttnn.deallocate(qkv)
        # per-head l2 norm as rms_norm on [1,1,B*H,K] rows: x/||x|| = rms_norm(x, eps/K) * K**-0.5 (q: * K**-0.5 more)
        q = ttnn.rms_norm(
            ttnn.reshape(q, (1, 1, B * H, K)),
            epsilon=1e-6 / K,
            weight=self.q_norm_w,
            memory_config=DRAM,
            compute_kernel_config=self.ckc,
        )
        k = ttnn.rms_norm(
            ttnn.reshape(k, (1, 1, B * H, K)),
            epsilon=1e-6 / K,
            weight=self.k_norm_w,
            memory_config=DRAM,
            compute_kernel_config=self.ckc,
        )

        # ---- gates
        rank_in = ttnn.slice(proj, (0, 0, 0, self.off_decay), (1, 1, B, self.off_decay + K), memory_config=DRAM)
        sg = ttnn.linear(
            rank_in,
            w.decay_output_projection,
            bias=w.decay_bias_flat,
            activation="sigmoid",
            dtype=ttnn.float32,
            memory_config=DRAM,
            compute_kernel_config=cc,
        )  # sigmoid(scale * (f + dt_bias)) [1,1,B,H*K]
        ttnn.deallocate(rank_in)
        eg = ttnn.exp(ttnn.multiply(sg, GATE_LOWER_BOUND, memory_config=DRAM), memory_config=DRAM)
        ttnn.deallocate(sg)
        beta_logit = ttnn.slice(proj, (0, 0, 0, self.off_beta), (1, 1, B, self.off_beta + H), memory_config=DRAM)
        beta_all = ttnn.sigmoid(ttnn.typecast(beta_logit, ttnn.float32), memory_config=DRAM)  # [1,1,B,H]
        # inactive slots: beta = 0 and decay = 1 (selects, not multiplies) leave the recurrent state unchanged
        beta = ttnn.where(inputs["active"], beta_all, 0.0, memory_config=DRAM)  # select: a NaN idle row stays out
        eg_all = eg
        eg = ttnn.where(inputs["active"], eg_all, 1.0, memory_config=DRAM)
        ttnn.deallocate(beta_all)
        ttnn.deallocate(eg_all)
        ogate = ttnn.slice(proj, (0, 0, 0, self.off_gate), (1, 1, B, self.off_gate + H * V), memory_config=DRAM)
        ttnn.deallocate(proj)

        # ---- recurrence on head rows [B,H,1,*] (fp32)
        q_r = ttnn.reshape(ttnn.typecast(q, ttnn.float32), (B, H, 1, K))
        k_r = ttnn.reshape(ttnn.typecast(k, ttnn.float32), (B, H, 1, K))
        v_r = ttnn.reshape(ttnn.typecast(v, ttnn.float32), (B, H, 1, V))
        eg_r = ttnn.reshape(eg, (B, H, 1, K))
        eg_c = ttnn.reshape(eg, (B, H, K, 1))
        k_c = ttnn.reshape(k_r, (B, H, K, 1))
        b_r = ttnn.reshape(beta, (B, H, 1, 1))
        S = dstate.recurrent
        v_read = ttnn.matmul(
            ttnn.multiply(k_r, eg_r, memory_config=DRAM), S, memory_config=DRAM, compute_kernel_config=cc
        )
        delta = ttnn.multiply(ttnn.subtract(v_r, v_read, memory_config=DRAM), b_r, memory_config=DRAM)
        outer = ttnn.multiply(k_c, delta, memory_config=DRAM)  # [B,H,K,V]
        S_new = ttnn.add(ttnn.multiply(S, eg_c, memory_config=DRAM), outer, memory_config=DRAM)
        ttnn.copy(S_new, S)
        ttnn.deallocate(S_new)
        ttnn.deallocate(outer)
        o = ttnn.matmul(q_r, S, memory_config=DRAM, compute_kernel_config=cc)  # [B,H,1,V]
        for t in (v_read, delta, q_r, k_r, v_r, eg_r, eg_c, k_c, b_r, eg, beta, q, k, v):
            ttnn.deallocate(t)

        # ---- gated RMSNorm over V per head (fp32) and output projection
        o = ttnn.reshape(o, (1, 1, B * H, V))
        o = ttnn.rms_norm(
            o, epsilon=self.cfg.norm_eps, weight=self.o_norm_w, memory_config=DRAM, compute_kernel_config=self.ckc
        )
        og = ttnn.sigmoid(ttnn.typecast(ttnn.reshape(ogate, (1, 1, B * H, V)), ttnn.float32), memory_config=DRAM)
        o = ttnn.multiply(o, og, memory_config=DRAM)
        ttnn.deallocate(og)
        ttnn.deallocate(ogate)
        o = ttnn.typecast(ttnn.reshape(o, (1, 1, B, H * V)), ttnn.bfloat16)
        y = ttnn.linear(o, w.output_projection, memory_config=DRAM, compute_kernel_config=cc)
        ttnn.deallocate(o)
        return y  # TP > 1: partial sums, reduced by the caller


def _tap4(tap: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.reshape(tap, (1, 1, 1, tap.shape[-1]))


def _write_rows(buf: ttnn.Tensor, src: ttnn.Tensor, idx: int, dim: int, n: int):
    """buf[..idx..] = src along `dim` (rebuild + copy into the fixed buffer; outside traces)."""
    shape = list(buf.shape)
    parts = []

    def sl(a, b):
        s = [0] * len(shape)
        e = list(shape)
        s[dim], e[dim] = a, b
        return ttnn.slice(buf, s, e, memory_config=DRAM)

    if idx > 0:
        parts.append(sl(0, idx))
    parts.append(src if src.dtype == buf.dtype else ttnn.typecast(src, buf.dtype))
    if idx < n - 1:
        parts.append(sl(idx + 1, n))
    new = ttnn.concat(parts, dim=dim, memory_config=DRAM) if len(parts) > 1 else parts[0]
    ttnn.copy(new, buf)
    if new is not src:
        ttnn.deallocate(new)
