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
import os
from dataclasses import dataclass, field

import ttnn

from . import fused_decoder as _fd
from .functional_decoder import FUNCTIONAL_MOE
from .fused_decoder import FusedDecoder, FusedFlags, FusedHyperConnection
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
    # precision trials (review #1 P1-3; None = the loaded dtype): DSA q_a / q_b / kv_a / o projections, and the dense MLP
    # (layers 0-2) gate|up and down with their compute fidelity. Reloaded from the host state dict with that dtype
    # (one host block-float rounding, as the routed experts), single die only.
    dsa_proj_dtype: object = None
    dsa_proj_names: tuple = ("w_q_a", "w_q_b", "w_kv_a", "w_o")
    mlp_dtype: object = None
    mlp_fidelity: object = None
    # the stage-5 decode options that need no collective, on one die (review #1 P1-5; tt/multichip_decoder.DEFAULT_MC,
    # doc/optimized_multichip_decoder/README.md): width-sharded decode RMSNorms (layer norms, DSA q_a / kv_a norms), the
    # five DSA x-projections as two packed matmuls on 1D decode configs whose packed weights also serve prefill, DSA
    # q_b / wq_b and the MoE router on 1D decode configs, DSA prefill matmuls on an explicit 2D config
    single_die_mc: bool = False
    # mhc_mix K chunks at decode rows (0 = fused_decoder.mix_chunks: 32 on one die; stage 5 took 16 at TP4)
    mix_chunks_decode: int = 0
    # prefill program configs (review #1 P2-3, tools/s3r_probe_dsa.py / s3r_probe_dense.py at 2048 rows): DSA
    # flash_mla_prefill q / k chunks 128 (22852 -> 6654 us), DSA o_proj and dense-MLP down on an explicit 2D config
    # (7707 -> 3708, 5792 -> 2778 us). The per-head reuse config on prefill rows (1900 -> 765 us in the probe) stays off:
    # it broke DSA continuation chunks (GLM53_PF_OFF=bmm_on turns it on for that check)
    prefill_cfg: bool = False

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
            "dsa_proj": f"{self.dsa_proj_dtype} {','.join(self.dsa_proj_names)}" if self.dsa_proj_dtype else "loaded",
            "mlp": f"{self.mlp_dtype} {self.mlp_fidelity}" if self.mlp_dtype or self.mlp_fidelity else "loaded",
            "single_die_mc": self.single_die_mc,
            "mix_chunks_decode": self.mix_chunks_decode,
            "prefill_cfg": self.prefill_cfg,
        }

    def quant_spec(self) -> dict:
        """{checkpoint-name pattern: "bfp4" | "bfp8"} of the expert weights this policy stores host-rounded below bf16:
        the quantization-aware HF reference of the MoE layers (tests/hf_ref/hf_quant_ref.py). Only the routed and
        shared experts: the precision trials of other tensor groups (KDA / DSA projections, dense MLP) are measured
        against this same reference, so their own loss shows as a PCC / scale change against `opt`."""
        fmt = {ttnn.bfloat4_b: "bfp4", ttnn.bfloat8_b: "bfp8"}
        m = self.moe
        spec = {}
        for proj, dt in (
            ("gate_proj", m.routed_gate_up_dtype),
            ("up_proj", m.routed_gate_up_dtype),
            ("down_proj", m.routed_down_dtype),
        ):
            if dt in fmt:
                spec[f"mlp.experts.*.{proj}.weight"] = fmt[dt]
        if m.shared_dtype in fmt:
            spec["mlp.shared_experts.*.weight"] = fmt[m.shared_dtype]
        return spec


_B4, _B8 = ttnn.bfloat4_b, ttnn.bfloat8_b
_LOFI, _HIFI2, _HIFI3, _HIFI4 = (
    ttnn.MathFidelity.LoFi,
    ttnn.MathFidelity.HiFi2,
    ttnn.MathFidelity.HiFi3,
    ttnn.MathFidelity.HiFi4,
)


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
# review #1 trials on top of `opt` (doc/optimized_decoder/README.md "Precision trials"):
_opt = CANDIDATES["opt"]
_r = lambda o, name, **kw: dataclasses.replace(o, name=name, **kw)
CANDIDATES.update(
    {
        # routed experts at HiFi3 / HiFi4: the activation mantissa bits that LoFi / HiFi2 drop (P1-2: -0.5 % scale)
        "opt_rhifi3": _r(_opt, "opt_rhifi3", moe=dataclasses.replace(_opt.moe, routed_fidelity=_HIFI3)),
        "opt_rhifi4": _r(_opt, "opt_rhifi4", moe=dataclasses.replace(_opt.moe, routed_fidelity=_HIFI4)),
        # bfp4 trials of the largest non-expert decode matmuls (P1-3)
        "t_kda_in4": _r(_opt, "t_kda_in4", kda_in_dtype=_B4),
        "t_kda_out4": _r(_opt, "t_kda_out4", kda_out_dtype=_B4),
        "t_dsa4": _r(_opt, "t_dsa4", dsa_proj_dtype=_B4),
        "t_dsa_o4": _r(_opt, "t_dsa_o4", dsa_proj_dtype=_B4, dsa_proj_names=("w_o",)),
        "t_mlp4_lofi": _r(_opt, "t_mlp4_lofi", mlp_dtype=_B4, mlp_fidelity=_LOFI),
        "t_mlp4_hifi4": _r(_opt, "t_mlp4_hifi4", mlp_dtype=_B4, mlp_fidelity=_HIFI4),
        "t_mlp8_lofi": _r(_opt, "t_mlp8_lofi", mlp_fidelity=_LOFI),
        # stage-5 single-die options (P1-5)
        "opt_s5": _r(_opt, "opt_s5", single_die_mc=True),
        "opt_s5_mix16": _r(_opt, "opt_s5_mix16", single_die_mc=True, mix_chunks_decode=16),
        "opt_s5_pf": _r(_opt, "opt_s5_pf", single_die_mc=True, prefill_cfg=True),
        "opt_s5_rhifi3": _r(
            _opt, "opt_s5_rhifi3", single_die_mc=True, moe=dataclasses.replace(_opt.moe, routed_fidelity=_HIFI3)
        ),
    }
)
# selected (review #1): `opt` + the stage-5 single-die decode options + the prefill program configs; routed experts
# stay LoFi (HiFi3 removes the -0.5 % routed scale bias but costs +1.3 % / +3 % / +5 % at B=1 / B=32 / prefill on
# the MoE layers; doc/optimized_decoder/README.md "Accuracy")
SELECTED = CANDIDATES["opt_s5_pf"]


class _DecodeMixChunks(FusedHyperConnection):
    """FusedHyperConnection with `decode_chunks` mhc_mix K chunks at decode rows (one tile of users); prefill unchanged.
    The chunk count only changes how many fp32 partials the exact ttnn.sum adds."""

    def _mixes_split_k(self, x):
        n = self.decode_chunks
        if int(x.padded_shape[-2]) != 32 or (int(x.shape[-1]) // 32) % n:
            return super()._mixes_split_k(x)
        orig = _fd.mix_chunks
        _fd.mix_chunks = lambda T, k_tiles: n
        try:
            return super()._mixes_split_k(x)
        finally:
            _fd.mix_chunks = orig


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
        # bfp4 KDA projections (precision trial): loaded bf16, then host-rounded in _retype_kda (the device typecast to
        # bfp4 is not round-to-nearest: it left the KDA layer update 8-10 % low, logs/s3r_pcc_t_kda_out4.log)
        kw.setdefault(
            "kda_dtypes", tuple(None if d == ttnn.bfloat4_b else d for d in (opt.kda_in_dtype, opt.kda_out_dtype))
        )
        kw.setdefault("dsa_bmm_dtype", opt.dsa_bmm_dtype)
        dec = super().from_state_dict(
            state_dict, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device, flags=flags, **kw
        )
        dec.opt = opt
        if dec.attn_kind == "kda":
            dec._retype_kda(opt)
        else:
            dec._set_dsa(opt)
        if opt.dsa_proj_dtype is not None and dec.attn_kind == "dsa":
            dec._reload_dsa_proj(state_dict, opt)
        if dec.mlp_kind == "dense" and (opt.mlp_dtype is not None or opt.mlp_fidelity is not None):
            dec._reload_mlp(state_dict, opt)
        if opt.out_proj_1d:
            grid_w = mesh_device.compute_with_storage_grid_size().x
            dec.attn.decode_mm = DenseDecodeMatmul(opt.moe, grid_w, mode="1d")  # single-tile decode rows only
        if opt.prefill_cfg and dec.tp == 1:
            off = set(os.environ.get("GLM53_PF_OFF", "").split(","))  # bisection: flash, bmm, o2d, down
            grid = mesh_device.compute_with_storage_grid_size()
            if dec.attn_kind == "dsa":
                if "flash" not in off:
                    dec.attn.flash_prefill_pc = ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=(grid.x, grid.y),
                        q_chunk_size=128,
                        k_chunk_size=128,
                        exp_approx_mode=False,
                    )
                # the per-head reuse config on prefill rows is off: with it, DSA continuation chunks (prompts > 2048)
                # read 11-22 % low (W_UK) or non-finite (W_UV) (logs/s3r_bisect_pf.log, logs/s3r_final_suite.log)
                dec.attn.bmm_prefill_pm = 8 if (dec.attn.bmm_reuse and "bmm_on" in off) else 0
                dec.attn.prefill_o_2d = "o2d" not in off
            if dec.mlp_kind == "dense":
                dec.mlp.down_prefill_2d = "down" not in off
        dec._norm_sharded = False
        if opt.mix_chunks_decode and dec.tp == 1:
            for hc in (dec.attn_hc, dec.ffn_hc):
                if type(hc) is FusedHyperConnection:
                    hc.__class__ = _DecodeMixChunks
                    hc.decode_chunks = opt.mix_chunks_decode
        if opt.single_die_mc and dec.tp == 1:
            from .multichip_decoder import MCOpts, MultichipDecoder, MultichipDSA, _install_router_ds

            if not issubclass(cls, MultichipDecoder):  # MultichipDecoder applies its own MCOpts
                # the stage-5 options measured on one die (review #1), pinned here: tt/multichip_decoder.DEFAULT_MC keeps
                # changing with the stage-5 rework (its later options are measured on one die as MultichipDecoder at
                # TP = 1, README). No collectives on one die; the DSA prefill keeps its own configs (stage-5 prefill_2d
                # was sized for the TP4 K split: on one die its circular buffers grow to 3.5 MB at 2048 rows,
                # logs/s3r_suite_opt_s5.prefill2d.log)
                mc = MCOpts(
                    norm_sharded=True,
                    dsa_pack=True,
                    pack_a_1d=True,
                    pack_b_1d=True,
                    dsa_1d=True,
                    router_1d=True,
                    ds_shared=True,
                )
                dec.mc = mc
                if dec.attn_kind == "dsa":
                    dec.attn.__class__ = MultichipDSA
                    dec.attn.mc_init(mc)
                    if mc.ds_shared and mc.dsa_pack:
                        dec.attn.free_separate()
                if dec.mlp_kind == "moe":
                    _install_router_ds(dec.mlp, mc)
                dec._norm_sharded = mc.norm_sharded
        return dec

    def _norm(self, h, w):
        if self._norm_sharded and int(h.padded_shape[-2]) == 32 and h.dtype == ttnn.bfloat16:
            from .multichip_decoder import sharded_rms_norm

            out = sharded_rms_norm(h, w, self.eps, self.norm_ckc)  # decode rows: width-sharded on 32 cores
            ttnn.deallocate(h)
            return out
        return super()._norm(h, w)

    def decode_forward(self, X, *, inputs):
        out = super().decode_forward(X, inputs=inputs)
        if hasattr(self.attn, "pack_flush"):
            self.attn.pack_flush()
        return out

    def prefill_forward(self, X, *, inputs, attn_state=None):
        out = super().prefill_forward(X, inputs=inputs, attn_state=attn_state)
        if hasattr(self.attn, "pack_flush"):
            self.attn.pack_flush()
        return out

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
                if dt == ttnn.bfloat4_b:  # host block-float packing (round to nearest), as the routed experts
                    assert self.tp == 1, "bfp4 KDA trial: single die"
                    new[name] = ttnn.from_torch(
                        ttnn.to_torch(t),
                        dtype=dt,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                else:
                    new[name] = ttnn.typecast(t, dt)
                ttnn.deallocate(t)
        if not new:
            return
        nw = dataclasses.replace(w, **new)
        self.attn.weights = nw
        for p in self.attn._prefill.values():
            p.weights = nw

    def _reload_dsa_proj(self, sd, opt: OptConfig):
        """Precision trial: the DSA projections rebuilt from the host weights in opt.dsa_proj_dtype (one host rounding)."""
        assert self.tp == 1, "DSA projection trial: single die"
        hf = {"w_q_a": "q_a_proj", "w_q_b": "q_b_proj", "w_kv_a": "kv_a_proj_with_mqa", "w_o": "o_proj"}
        a = self.attn
        for n in opt.dsa_proj_names:
            old = getattr(a, n)
            setattr(
                a,
                n,
                ttnn.from_torch(
                    sd[f"self_attn.{hf[n]}.weight"].T.contiguous(),
                    dtype=opt.dsa_proj_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
            ttnn.deallocate(old)

    def _reload_mlp(self, sd, opt: OptConfig):
        """Precision trial: the dense MLP rebuilt from the host weights in opt.mlp_dtype at opt.mlp_fidelity."""
        from .mlp import DenseMLP

        old = self.mlp
        packed = old.wgu is not None
        for t in (old.wg, old.wu, old.wd, old.wgu):
            if t is not None:
                ttnn.deallocate(t)
        mlp = DenseMLP(
            self.mesh,
            lambda: sd["mlp.gate_proj.weight"],
            lambda: sd["mlp.up_proj.weight"],
            lambda: sd["mlp.down_proj.weight"],
            self.tc["swiglu_limit"],
            dtype=opt.mlp_dtype or ttnn.bfloat8_b,
        )
        if opt.mlp_fidelity is not None:
            mlp.ckc = ttnn.init_device_compute_kernel_config(
                self.mesh.arch(),
                math_fidelity=opt.mlp_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        if packed:
            mlp.pack()
        self.mlp = mlp
