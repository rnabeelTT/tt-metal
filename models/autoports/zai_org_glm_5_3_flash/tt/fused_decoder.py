# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decoder layer of GLM-5.3-Flash (stage 2): the functional decoder with its op graph rewritten into fewer,
larger, dedicated ops. Semantics, contracts and state are those of tt/functional_decoder.py.

Rewrites (doc/fused_decoder/README.md has the before / after op tables and measurements):
  * mHC pre-collapse h = sum_i pre_i * X_i: one ttnn.experimental.deepseek_prefill.mhc_pre call (dedicated op, written
    for this port) instead of n stream slices + n coefficient-column slices + n multiply/addcmul; its bf16 output feeds
    the RMSNorm directly (the fp32 -> bf16 typecast folds into the op).
  * mHC projection scale rsqrt(mean(X^2) + eps): ttnn.rms_norm_pre_all_gather (one pass over X giving sum(X^2)) instead
    of multiply + mean over the 16384-wide stream.
  * KDA decode recurrence: one ttnn.experimental.kda.recurrent_decode call (dedicated op, written for this port from
    ttnn.transformer.fused_recurrent_gated_delta_rule with a per-key-channel decay) instead of the q / k / v slices,
    two reshape + L2-norm pairs and the 12-op fp32 recurrence over [B, H, K, V]; it reads the head rows straight out of
    the conv output and updates the state in place.
"""

from __future__ import annotations

import torch

import ttnn

from .functional_decoder import FunctionalDecoder
from .kda import GATE_LOWER_BOUND, GlmKDA, KdaDecodeState, _tap4
from .mhc import DRAM, HyperConnection


class FusedFlags:
    """Rewrite switches (all on in the fused decoder; the before / after measurements toggle them)."""

    def __init__(
        self,
        mhc_pre_op: bool = True,
        mhc_sumsq_op: bool = False,
        kda_recurrent_op: bool = True,
        mhc_mix_op: bool = True,
    ):
        self.mhc_pre_op = mhc_pre_op
        # mhc_sumsq_op: rms_norm_pre_all_gather on the 16384-wide fp32 stream sizes its CBs by the row width (8.5 MB
        # > 1.5 MB L1, 2026-10-14): rejected; superseded by mhc_mix_op
        self.mhc_sumsq_op = mhc_sumsq_op
        self.kda_recurrent_op = kda_recurrent_op
        # mhc_mix_op: split-K mix projection + row sum of squares (ttnn.experimental.deepseek_prefill.mhc_mix) over
        # many cores instead of a single-core matmul and a single-core mean
        self.mhc_mix_op = mhc_mix_op


def mix_chunks(T: int, k_tiles: int) -> int:
    """K chunks of mhc_mix: about one unit per core (110 cores), at least 16 (a chunk <= 32 tiles of L1), <= 64."""
    c = 16
    while c < 64 and (T // 32) * c * 2 <= 110 and k_tiles % (c * 2) == 0:
        c *= 2
    return c


class FusedHyperConnection(HyperConnection):
    def __init__(self, base: HyperConnection, flags: FusedFlags):
        self.__dict__.update(base.__dict__)
        self.flags = flags
        self._masks = {}  # T -> [1,1,T,32] fp32: 1 at (r, r % 32), picks the Gram diagonals (row sums of squares)

    def prepare(self, row_counts):
        """Create the diagonal masks of every row count a pass will use (setup; no host transfer inside a pass)."""
        for T in row_counts:
            self._mask(int(T))

    def _mask(self, T: int):
        m = self._masks.get(T)
        if m is None:
            dev = self.fn_T.device()
            r = torch.arange(T)
            mask = torch.zeros(1, 1, T, 32)
            mask[0, 0, r, r % 32] = 1.0
            m = ttnn.from_torch(
                mask,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
            )
            self._masks[T] = m
        return m

    def _mixes(self, x):
        T = x.shape[-2]
        if self.flags.mhc_mix_op:
            return self._mixes_split_k(x)
        mixes = ttnn.matmul(x, self.fn_T, compute_kernel_config=self.ckc, memory_config=DRAM)
        if self.flags.mhc_sumsq_op:
            stats = ttnn.rms_norm_pre_all_gather(x, dtype=ttnn.float32, compute_kernel_config=self.ckc)  # [.., T, 32]
            ss = ttnn.slice(stats, (0, 0, 0, 0), (1, 1, T, 1))
            ttnn.deallocate(stats)
            scale = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / x.shape[-1]), self.cfg.norm_eps))
            ttnn.deallocate(ss)
        else:
            ms = ttnn.mean(ttnn.multiply(x, x, memory_config=DRAM), dim=-1, keepdim=True)
            scale = ttnn.rsqrt(ttnn.add(ms, self.cfg.norm_eps))
            ttnn.deallocate(ms)
        out = ttnn.multiply(mixes, scale, memory_config=DRAM)
        ttnn.deallocate(mixes)
        ttnn.deallocate(scale)
        return out

    def _mixes_split_k(self, x):
        """mixes * rsqrt(mean(x^2) + eps) from the mhc_mix split-K partials: sum over the chunks, the row sums of squares
        are the diagonals of the 32-row Gram tiles (masked by one identity tile)."""
        T, K = x.shape[-2], x.shape[-1]
        part = ttnn.experimental.deepseek_prefill.mhc_mix(
            x, self.fn_T, mix_chunks(-(-T // 32) * 32, K // 32)
        )  # [1,C,T,64]
        tot = ttnn.sum(part, dim=1, keepdim=True, memory_config=DRAM)  # [1,1,T,64]
        ttnn.deallocate(part)
        mixes = ttnn.slice(tot, (0, 0, 0, 0), (1, 1, T, 32))
        gram = ttnn.slice(tot, (0, 0, 0, 32), (1, 1, T, 64))
        ttnn.deallocate(tot)
        # select (not multiply): the Gram tile also mixes the activation's padding rows, which may hold non-finite
        # leftovers; 0 * inf would poison the row sum
        ss = ttnn.sum(ttnn.where(self._mask(T), gram, 0.0, memory_config=DRAM), dim=-1, keepdim=True)  # [1,1,T,1]
        ttnn.deallocate(gram)
        scale = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / K), self.cfg.norm_eps), memory_config=DRAM)
        ttnn.deallocate(ss)
        out = ttnn.multiply(mixes, scale, memory_config=DRAM)
        ttnn.deallocate(mixes)
        ttnn.deallocate(scale)
        return out

    def pre(self, x: ttnn.Tensor, out_dtype=ttnn.bfloat16):
        """x [1,1,T,n*C] fp32 -> (h [1,1,T,C] `out_dtype`, post [1,1,T,n], comb [1,1,T,n*n])."""
        T = x.shape[-2]
        mixes = self._mixes(x)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, self.n, self.cfg.sinkhorn_iters, self.cfg.eps
        )
        ttnn.deallocate(mixes)
        r4 = lambda t: ttnn.reshape(t, (1, 1, T, t.shape[-1]))
        if self.flags.mhc_pre_op:
            h = ttnn.experimental.deepseek_prefill.mhc_pre(x, r4(pre), self.bcast, self.n, output_dtype=out_dtype)
        else:
            h, _, _ = super().pre(x)  # composite reference path (functional decoder)
        ttnn.deallocate(pre)
        return h, r4(post), r4(comb)


def decode_linear(mod, x, w, role, ckc):
    """x @ w for a decode step: the module's decode_mm hook (optimized decoder: explicit decode program configs) when
    it takes the shape, else ttnn's default."""
    hook = getattr(mod, "decode_mm", None)
    out = hook(x, w, role, ckc, DRAM) if hook is not None else None
    return out if out is not None else ttnn.linear(x, w, memory_config=DRAM, compute_kernel_config=ckc)


class FusedGlmKDA(GlmKDA):
    """GlmKDA with the fused decode step (prefill unchanged)."""

    def __init__(self, base: GlmKDA, flags: FusedFlags):
        self.__dict__.update(base.__dict__)
        self.flags = flags

    def decode_forward(self, x: ttnn.Tensor, dstate: KdaDecodeState, *, inputs) -> ttnn.Tensor:
        if not self.flags.kda_recurrent_op:
            return super().decode_forward(x, dstate, inputs=inputs)
        B = x.shape[-2]
        H, K, V, W = self.H, self.K, self.V, self.W
        w = self.weights
        cc = self.ckc
        proj = ttnn.linear(x, w.input_projection, memory_config=DRAM, compute_kernel_config=cc)  # [1,1,B,P]

        # ---- short causal conv over (h0, h1, h2, current), fp32 accumulation, SiLU
        cur = ttnn.slice(proj, (0, 0, 0, 0), (1, 1, B, W), memory_config=DRAM)
        taps = w.convolution_taps
        acc = ttnn.multiply(cur, _tap4(taps[3]), dtype=ttnn.float32, memory_config=DRAM)
        for j in range(3):
            acc = ttnn.add(
                acc,
                ttnn.multiply(dstate.conv[j], _tap4(taps[j]), dtype=ttnn.float32, memory_config=DRAM),
                memory_config=DRAM,
            )
        qkv = ttnn.silu(acc, memory_config=DRAM)  # [1,1,B,W] fp32
        # inactive slots: zero q / k / v (a select, so non-finite rows of an idle slot cannot reach its state)
        qkv_all = qkv
        qkv = ttnn.where(inputs["active"], qkv_all, 0.0, memory_config=DRAM)
        ttnn.deallocate(qkv_all)
        ttnn.deallocate(acc)
        self.shift_conv_history(dstate, cur, inputs["active_bf"])
        ttnn.deallocate(cur)

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
        )
        ttnn.deallocate(rank_in)
        eg = ttnn.exp(ttnn.multiply(sg, GATE_LOWER_BOUND, memory_config=DRAM), memory_config=DRAM)  # [1,1,B,H*K]
        ttnn.deallocate(sg)
        beta_logit = ttnn.slice(proj, (0, 0, 0, self.off_beta), (1, 1, B, self.off_beta + H), memory_config=DRAM)
        beta_all = ttnn.sigmoid(ttnn.typecast(beta_logit, ttnn.float32), memory_config=DRAM)  # [1,1,B,H]
        # inactive slots: beta = 0 and decay = 1 leave the recurrent state unchanged
        beta = ttnn.where(inputs["active"], beta_all, 0.0, memory_config=DRAM)  # select: a NaN idle row stays out
        eg_all = eg
        eg = ttnn.where(inputs["active"], eg_all, 1.0, memory_config=DRAM)
        ttnn.deallocate(beta_all)
        ttnn.deallocate(eg_all)
        ogate = ttnn.slice(proj, (0, 0, 0, self.off_gate), (1, 1, B, self.off_gate + H * V), memory_config=DRAM)
        ttnn.deallocate(proj)

        # ---- L2 norms + recurrence, state in place: o [1,1,B*H,V] fp32 (row b*H + h)
        o = ttnn.experimental.kda.recurrent_decode(dstate.recurrent, qkv, eg, beta, K**-0.5)
        for t in (qkv, eg, beta):
            ttnn.deallocate(t)

        # ---- gated RMSNorm over V per head (fp32) and output projection
        o = ttnn.rms_norm(
            o, epsilon=self.cfg.norm_eps, weight=self.o_norm_w, memory_config=DRAM, compute_kernel_config=self.ckc
        )
        og = ttnn.sigmoid(ttnn.typecast(ttnn.reshape(ogate, (1, 1, B * H, V)), ttnn.float32), memory_config=DRAM)
        o = ttnn.multiply(o, og, memory_config=DRAM)
        ttnn.deallocate(og)
        ttnn.deallocate(ogate)
        o = ttnn.typecast(ttnn.reshape(o, (1, 1, B, H * V)), ttnn.bfloat16)
        y = decode_linear(self, o, w.output_projection, "o", cc)
        ttnn.deallocate(o)
        return y


class FusedDecoder(FunctionalDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, flags: FusedFlags | None = None, **kw):
        dec = cls(mesh_device, hf_config, state_dict, layer_idx, **kw)
        dec.flags = flags or FusedFlags()
        dec.attn_hc = FusedHyperConnection(dec.attn_hc, dec.flags)
        dec.ffn_hc = FusedHyperConnection(dec.ffn_hc, dec.flags)
        if dec.attn_kind == "kda":
            dec.attn = FusedGlmKDA(dec.attn, dec.flags)
        return dec

    def allocate_decode_state(self, batch: int | None = None):
        state = super().allocate_decode_state(batch)
        rows = list(self.prefill_lengths) + [batch or self.max_batch]
        self.attn_hc.prepare(rows)
        self.ffn_hc.prepare(rows)
        return state

    def _norm(self, h: ttnn.Tensor, w) -> ttnn.Tensor:
        if h.dtype != ttnn.bfloat16:
            return super()._norm(h, w)
        out = ttnn.rms_norm(h, epsilon=self.eps, weight=w, memory_config=DRAM, compute_kernel_config=self.norm_ckc)
        ttnn.deallocate(h)
        return out
