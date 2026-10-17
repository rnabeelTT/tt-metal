# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized decoder layer of GLM-5.3-Flash (stage 3): the fused decoder (tt/fused_decoder.py) with a per-tensor-group
precision / fidelity policy and decode program configs for one die. Semantics, contracts, paging and state are those
of tt/functional_decoder.py.

Decode is weight-read bound after fusing (doc/fused_decoder/README.md): the routed experts (all 192 read at 32 users),
the KDA input projection (4096 x 32960, bf16 in the functional / fused decoders) and the dense projections. The
policy candidates are named in CANDIDATES; doc/optimized_decoder/README.md has the measurements that select SELECTED.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import ttnn

from .functional_decoder import FUNCTIONAL_MOE
from .fused_decoder import FusedDecoder, FusedFlags
from .moe_opt import DenseDecodeMatmul, OptimizationConfig


@dataclass(frozen=True)
class OptConfig:
    """Precision policy of one optimized-decoder candidate."""

    name: str
    kda_in_dtype: object = ttnn.bfloat16  # KDA fused input projection (q|k|v|decay rank|gate|beta)
    kda_out_dtype: object = ttnn.bfloat16  # KDA output projection
    moe: OptimizationConfig = field(default_factory=lambda: FUNCTIONAL_MOE)
    dsa_bmm_dtype: object = ttnn.bfloat16  # DSA absorbed per-head W_UK / W_UV
    dsa_bmm_reuse: bool = False  # DSA per-head matmuls on the full-grid reuse config (dsa.GlmDSA._bmm)
    out_proj_1d: bool = False  # KDA / DSA output projections: 1D decode program config (moe_opt.DenseDecodeMatmul)

    def describe(self) -> dict:
        m = self.moe
        return {
            "name": self.name,
            "kda_in": str(self.kda_in_dtype),
            "kda_out": str(self.kda_out_dtype),
            "dsa_bmm": f"{self.dsa_bmm_dtype}{' reuse' if self.dsa_bmm_reuse else ''}",
            "out_proj": "1d" if self.out_proj_1d else "default",
            "routed": f"{m.routed_gate_up_dtype}/{m.routed_down_dtype} {m.routed_fidelity}",
            "shared": f"{m.shared_dtype} {m.shared_fidelity}",
        }


_B4, _B8 = ttnn.bfloat4_b, ttnn.bfloat8_b
_LOFI, _HIFI2, _HIFI4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4


def _moe(**kw) -> OptimizationConfig:
    return dataclasses.replace(FUNCTIONAL_MOE, **kw)


CANDIDATES = {
    # the fused decoder's policy (stage-2 baseline)
    "fused": OptConfig("fused"),
    # KDA projections bfp8
    "kda8": OptConfig("kda8", _B8, _B8),
    # + routed experts bfp4 / LoFi (the GLM-4.5-Air / Qwen3.5 bundles' expert policy), shared expert bfp8 / HiFi2
    "kda8_exp4_lofi": OptConfig(
        "kda8_exp4_lofi",
        _B8,
        _B8,
        _moe(
            routed_gate_up_dtype=_B4,
            routed_down_dtype=_B4,
            routed_fidelity=_LOFI,
            shared_dtype=_B8,
            shared_fidelity=_HIFI2,
        ),
    ),
    # bfp4 experts at HiFi2 (does fidelity buy accuracy at the same weight bytes?)
    "kda8_exp4_hifi2": OptConfig(
        "kda8_exp4_hifi2",
        _B8,
        _B8,
        _moe(routed_gate_up_dtype=_B4, routed_down_dtype=_B4, routed_fidelity=_HIFI2, shared_fidelity=_HIFI2),
    ),
    # bfp4 gate / up, bfp8 down (the down projection feeds the residual directly)
    "kda8_exp4gu8dn": OptConfig(
        "kda8_exp4gu8dn",
        _B8,
        _B8,
        _moe(routed_gate_up_dtype=_B4, routed_down_dtype=_B8, routed_fidelity=_LOFI, shared_fidelity=_HIFI2),
    ),
}
# + decode program configs: DSA per-head matmuls bfp8 on the full-grid reuse config, KDA / DSA output projections 1D
for _base in ("kda8_exp4_lofi", "kda8_exp4_hifi2"):
    CANDIDATES[f"{_base}_pc"] = dataclasses.replace(
        CANDIDATES[_base], name=f"{_base}_pc", dsa_bmm_dtype=_B8, dsa_bmm_reuse=True, out_proj_1d=True
    )
# selected: bfp4 routed experts at LoFi (bit-identical to HiFi2 on bfp4 weights; tools/moe_scale_probe.py --policy:
# TT vs the bfp4-rounded torch reference PCC 0.99997, rms 0.995), shared expert bfp8 at HiFi4 (HiFi2 left it 0.8 % low,
# the activation-mantissa truncation of review #2), KDA projections bfp8, decode program configs
CANDIDATES["opt"] = dataclasses.replace(
    CANDIDATES["kda8_exp4_lofi_pc"],
    name="opt",
    moe=dataclasses.replace(CANDIDATES["kda8_exp4_lofi_pc"].moe, shared_fidelity=_HIFI4),
)
SELECTED = CANDIDATES["opt"]


class OptimizedDecoder(FusedDecoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        opt: OptConfig | None = None,
        flags: FusedFlags | None = None,
        **kw,
    ):
        opt = opt or SELECTED
        kw.setdefault("moe_opt", opt.moe)
        # policy dtypes applied at load (same on-device typecasts as _retype_kda / _set_dsa, before the weight cache
        # stores the tensors: a cached build loads the final dtype and the retypes below find nothing to do)
        kw.setdefault("kda_dtypes", (opt.kda_in_dtype, opt.kda_out_dtype))
        kw.setdefault("dsa_bmm_dtype", opt.dsa_bmm_dtype)
        dec = super().from_state_dict(
            state_dict, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device, flags=flags, **kw
        )
        dec.opt = opt
        if dec.attn_kind == "kda":
            dec._retype_kda(opt)
        else:
            dec._set_dsa(opt)
        if opt.out_proj_1d:
            grid_w = mesh_device.compute_with_storage_grid_size().x
            dec.attn.decode_mm = DenseDecodeMatmul(opt.moe, grid_w, mode="1d")  # single-tile decode rows only
        return dec

    def _set_dsa(self, opt: OptConfig):
        a = self.attn
        a.bmm_reuse = opt.dsa_bmm_reuse
        for name in ("w_uk", "w_uv_t"):
            t = getattr(a, name)
            if t.dtype != opt.dsa_bmm_dtype:
                setattr(a, name, ttnn.typecast(t, opt.dsa_bmm_dtype))
                ttnn.deallocate(t)

    def _retype_kda(self, opt: OptConfig):
        """Retype the KDA projections on device (one copy, shared by the prefill kernels and the decode step)."""
        w = self.attn.weights
        new = {}
        for name, dt in (("input_projection", opt.kda_in_dtype), ("output_projection", opt.kda_out_dtype)):
            t = getattr(w, name)
            if t.dtype != dt:
                new[name] = ttnn.typecast(t, dt)
                ttnn.deallocate(t)
        if not new:
            return
        nw = dataclasses.replace(w, **new)
        self.attn.weights = nw
        for p in self.attn._prefill.values():
            p.weights = nw
