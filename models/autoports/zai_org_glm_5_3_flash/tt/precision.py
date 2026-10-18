# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Datatype-sweep precision overrides (stage 8) on top of the shipped policy.

The shipped policy is tt/optimized_decoder.SELECTED (per-layer weight dtypes and MoE fidelities), tt/multichip_decoder
.DEFAULT_MC (decode programs, CCL) and the model-level constants of tt/model.py (LM head bfp8 at HiFi2, bf16 residual,
bf16 DSA latent pool). doc/datatype_sweep/selected_precision_config.json records the complete selected policy.

GLM53_PRECISION=<path to a JSON candidate> applies a candidate at construction (Glm53Model.__init__, so the generator,
the readiness runners and the vLLM adapter all consume it); unset or empty = the shipped policy, no override. Fields:

  id                       candidate id (logged and recorded)
  requant                  {group: {"dtype": "bfp4" | "bfp8", "layers": "all" | "inner" | [i, ...]}}: host round-to-
                           nearest re-quantization of the loaded weights of a group (groups: kda_in, kda_out, dsa_proj
                           = q_a | kv_a pack + q_b + o_proj, shared_expert, lm_head). The loaded weights come from the
                           shipped-policy weight cache (bfp8); bfp8 -> bfp4 RTN equals bf16 -> bfp4 RTN except at
                           rounding ties of the bfp8 grid (2^-7 of the block maximum).
  fidelity                 {group: "LoFi" | "HiFi2" | "HiFi3" | "HiFi4"}: compute fidelity of the decode-row matmuls of
                           a group (kda_in, kda_out, dsa_proj, dense_mlp, lm_head; prefill rows keep the shipped
                           fidelity), and of every row for shared_expert / routed (the MoE OptimizationConfig fields)
  fidelity_layers          {group: "all" | "inner" | [i, ...]} (default "all")
  moe_layer_exceptions     {layer: {OptimizationConfig field: value}}: per-layer MoE policy (e.g. routed_down_dtype
                           "bfp8"); a layer with a weight-dtype exception is built from the checkpoint (no weight
                           cache), every other layer still loads from the shipped-policy cache
  kv_latent_dtype          "bf16" (shipped) | "fp8_e4m3": the DSA latent pool (KV cache) in fp8 e4m3 rows (half the
                           bytes; tt/dsa.py GlmDSA.latent_dtype); the indexer pool keys stay bf16
  ccl_h_dtype              "bf16" (shipped) | "bfp8": the decode all-gathers of the hidden-sharded residual before the
                           attention / FFN norms run in bfp8 (cast before, cast back after)

"inner" = every layer of the group except the model's first and last layer (0 and 44).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

import ttnn

ENV = "GLM53_PRECISION"
_DT = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
_FID = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
_MOE_DTYPE_FIELDS = ("routed_gate_up_dtype", "routed_down_dtype", "shared_dtype", "router_dtype")
GROUPS = ("kda_in", "kda_out", "dsa_proj", "dense_mlp", "shared_expert", "routed", "lm_head")


def _decode_rows(t) -> bool:
    return int(t.padded_shape[-2]) == 32


def _ckc(mesh, fidelity, fp32_acc=True, l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=l1_acc
    )


def requant(t: ttnn.Tensor, dtype) -> ttnn.Tensor:
    """Host round-to-nearest re-quantization of a device weight to `dtype`, keeping its per-device shards and memory
    config (each die's shard is read back, packed on the host and written to the same die). Frees `t`."""
    if t.dtype == dtype:
        return t
    mesh = t.device()
    n = mesh.get_num_devices()
    host = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)) if n > 1 else ttnn.to_torch(t)
    out = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=t.memory_config(),
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if n > 1 else None,
    )
    ttnn.deallocate(t)
    return out


_INSTANCES: dict = {}


@dataclass
class PrecisionOverrides:
    id: str = "shipped"
    requant: dict = field(default_factory=dict)
    fidelity: dict = field(default_factory=dict)
    fidelity_layers: dict = field(default_factory=dict)
    moe_layer_exceptions: dict = field(default_factory=dict)
    ccl_h_dtype: str = "bf16"
    kv_latent_dtype: str = "bf16"
    source: str | None = None
    applied: list = field(default_factory=list)  # (layer or "model", group, what) records of the runtime changes

    @classmethod
    def from_env(cls) -> "PrecisionOverrides":
        """The process-wide candidate of GLM53_PRECISION (one instance per value: its `applied` log collects the
        changes of every layer and of the model)."""
        p = os.environ.get(ENV, "").strip()
        if p not in _INSTANCES:
            _INSTANCES[p] = cls._load(p)
        return _INSTANCES[p]

    @classmethod
    def _load(cls, p: str) -> "PrecisionOverrides":
        if not p:
            return cls()
        d = json.loads(Path(p).read_text())
        known = {f.name for f in dataclasses.fields(cls)} - {"source", "applied"}
        unknown = set(d) - known - {"description", "notes", "policy"}
        if unknown:
            raise ValueError(f"{p}: unknown precision override fields {sorted(unknown)}")
        o = cls(**{k: v for k, v in d.items() if k in known}, source=str(p))
        for g in list(o.requant) + list(o.fidelity):
            if g not in GROUPS:
                raise ValueError(f"{p}: unknown group {g!r} (groups {GROUPS})")
        logger.warning(f"{ENV}={p}: precision candidate {o.id!r} overrides the shipped policy")
        return o

    @property
    def is_shipped(self) -> bool:
        return not (
            self.requant
            or self.fidelity
            or self.moe_layer_exceptions
            or self.ccl_h_dtype != "bf16"
            or self.kv_latent_dtype != "bf16"
        )

    # ------------------------------------------------------------------------------------------- layer selection
    @staticmethod
    def _in(layers, i: int, first: int, last: int) -> bool:
        if layers in (None, "all"):
            return True
        if layers == "inner":
            return i not in (first, last)
        return int(i) in {int(x) for x in layers}

    def requant_dtype(self, group: str, i: int, first: int, last: int):
        r = self.requant.get(group)
        if r is None or not self._in(r.get("layers", "all"), i, first, last):
            return None
        return _DT[r["dtype"]]

    def fid(self, group: str, i: int, first: int, last: int):
        f = self.fidelity.get(group)
        if f is None or not self._in(self.fidelity_layers.get(group, "all"), i, first, last):
            return None
        return _FID[f]

    # ------------------------------------------------------------------------------------------- construction
    def layer_opt(self, opt, i: int, first: int, last: int):
        """The OptConfig of layer i: MoE runtime fidelities (shared_expert / routed groups) and the per-layer MoE
        exceptions. Weight dtypes of cached layers never change here (layer_has_weight_exception)."""
        if opt is None:
            from .optimized_decoder import SELECTED

            opt = SELECTED
        moe_kw = {}
        f = self.fid("shared_expert", i, first, last)
        if f is not None:
            moe_kw["shared_fidelity"] = f
        f = self.fid("routed", i, first, last)
        if f is not None:
            moe_kw["routed_fidelity"] = f
        for k, v in (self.moe_layer_exceptions.get(str(i)) or self.moe_layer_exceptions.get(i) or {}).items():
            moe_kw[k] = _DT[v] if k in _MOE_DTYPE_FIELDS else (_FID[v] if k.endswith("fidelity") else v)
        if not moe_kw:
            return opt
        return dataclasses.replace(opt, name=f"{opt.name}+{self.id}", moe=dataclasses.replace(opt.moe, **moe_kw))

    def layer_has_weight_exception(self, i: int) -> bool:
        exc = self.moe_layer_exceptions.get(str(i)) or self.moe_layer_exceptions.get(i) or {}
        return any(k in _MOE_DTYPE_FIELDS for k in exc)

    def layer_cache(self, cache, i: int):
        return None if (cache is not None and self.layer_has_weight_exception(i)) else cache

    # ------------------------------------------------------------------------------------------- post-load
    def apply_layer(self, dec, i: int, first: int, last: int):
        """Re-quantize / re-fidelity the groups of one built decoder layer (before warm-up and trace capture)."""
        mesh = dec.mesh if hasattr(dec, "mesh") else dec.attn.mesh
        attn = dec.attn._inner if hasattr(dec.attn, "_inner") else dec.attn
        mlp = dec.mlp._inner if hasattr(dec.mlp, "_inner") else dec.mlp
        if dec.attn_kind == "kda":
            self._apply_kda(attn, mesh, i, first, last)
        else:
            self._apply_dsa(attn, mesh, i, first, last)
        if dec.mlp_kind == "moe":
            dt = self.requant_dtype("shared_expert", i, first, last)
            if dt is not None:
                mlp.shared_packed = requant(mlp.shared_packed, dt)
                mlp.shared_w2 = requant(mlp.shared_w2, dt)
                self.applied.append((i, "shared_expert", f"requant {dt}"))
            self.applied.append((i, "moe", f"routed {mlp.prec.routed_fidelity} shared {mlp.prec.shared_fidelity}"))
        else:
            f = self.fid("dense_mlp", i, first, last)
            if f is not None:
                _swap_ckc_on_decode(mlp, "forward", _ckc(mesh, f))
                self.applied.append((i, "dense_mlp", f"decode fidelity {f}"))
        if self.ccl_h_dtype != "bf16" and getattr(dec, "ccl", None) is not None:
            _install_ccl_h(dec.ccl, _DT[self.ccl_h_dtype])
            self.applied.append((i, "ccl_h", self.ccl_h_dtype))

    def _apply_kda(self, attn, mesh, i, first, last):
        w = attn.weights
        new = {}
        for g, name in (("kda_in", "input_projection"), ("kda_out", "output_projection")):
            dt = self.requant_dtype(g, i, first, last)
            if dt is not None:
                new[name] = requant(getattr(w, name), dt)
                self.applied.append((i, g, f"requant {dt}"))
        if new:
            nw = dataclasses.replace(w, **new)
            attn.weights = nw
            for p in attn._prefill.values():
                p.weights = nw
        f_in, f_out = self.fid("kda_in", i, first, last), self.fid("kda_out", i, first, last)
        if f_in is not None:
            ck = _ckc(mesh, f_in)
            orig = attn._in_proj
            attn._in_proj = lambda x, w_, cc, _o=orig, _c=ck: _o(x, w_, _c if _decode_rows(x) else cc)
            self.applied.append((i, "kda_in", f"decode fidelity {f_in}"))
        if f_out is not None:
            _wrap_decode_mm(attn, _ckc(mesh, f_out))
            self.applied.append((i, "kda_out", f"decode fidelity {f_out}"))

    def _apply_dsa(self, attn, mesh, i, first, last):
        if self.kv_latent_dtype != "bf16":
            assert self.kv_latent_dtype == "fp8_e4m3", self.kv_latent_dtype
            assert attn.latent is None, "the latent pool dtype must be set before allocate_decode_state"
            attn.latent_dtype = ttnn.fp8_e4m3  # tt/dsa.py: ROW_MAJOR fp8 pool, sparse_sdpa kv_format FP8_E4M3
            self.applied.append((i, "kv_latent", "fp8_e4m3"))
        dt = self.requant_dtype("dsa_proj", i, first, last)
        if dt is not None:
            A, B = attn._pack  # q_a | kv_a pack (serves prefill and decode: ds_shared), indexer pack (bf16, kept)
            A.w = requant(A.w, dt)
            attn.w_q_b = requant(attn.w_q_b, dt)
            attn.w_o = requant(attn.w_o, dt)
            self.applied.append((i, "dsa_proj", f"requant {dt} (q_a|kv_a pack, q_b, o_proj)"))
        f = self.fid("dsa_proj", i, first, last)
        if f is not None:
            ck = _ckc(mesh, f)
            ids = set(attn._pack_ids) if getattr(attn, "_pack_ids", None) else set()
            ids.add(id(attn.w_q_b))
            orig = attn._mm

            def _mm(x, w, _o=orig, _ids=ids, _c=ck, _a=attn, **kw):
                if (id(w) in _ids or w is _a.w_q_b) and _decode_rows(x) and id(w) not in _a.__dict__.get("_ix", ()):
                    saved = _a.ckc
                    _a.ckc = _c
                    try:
                        return _o(x, w, **kw)
                    finally:
                        _a.ckc = saved
                return _o(x, w, **kw)

            # the indexer pack (wk | gate | weights, bf16, fp32 output) keeps HiFi4: index scores, not a projection
            attn._ix = {id(attn.w_ik), id(attn.w_ig), id(attn.w_iw)}
            attn._mm = _mm
            _wrap_decode_mm(attn, ck)
            self.applied.append((i, "dsa_proj", f"decode fidelity {f} (q_a|kv_a, q_b, o_proj)"))

    def apply_model(self, model, first: int, last: int):
        dt = self.requant_dtype("lm_head", -1, first, last)
        if dt is not None:
            model.lm_head = requant(model.lm_head, dt)
            self.applied.append(("model", "lm_head", f"requant {dt}"))
        f = self.fid("lm_head", -1, first, last)
        if f is not None:
            model.ckc = _ckc(model.mesh, f)
            self.applied.append(("model", "lm_head", f"fidelity {f}"))

    def record(self) -> dict:
        d = dataclasses.asdict(self)
        d["applied"] = [list(map(str, a)) for a in self.applied]
        return d


def _swap_ckc_on_decode(obj, method: str, ck):
    """obj.<method>(x, ...) runs with obj.ckc = ck on decode rows (the method reads self.ckc per call)."""
    orig = getattr(obj, method)

    def wrapped(x, *a, _o=orig, **kw):
        if not _decode_rows(x):
            return _o(x, *a, **kw)
        saved = obj.ckc
        obj.ckc = ck
        try:
            return _o(x, *a, **kw)
        finally:
            obj.ckc = saved

    setattr(obj, method, wrapped)


def _wrap_decode_mm(attn, ck):
    """The module's decode output-projection hook (RoleDecodeMM, role "o") with compute config ck."""
    inner = attn.decode_mm
    attn.decode_mm = lambda x, w, role, ckc, mc, _i=inner, _c=ck: _i(x, w, role, _c if role == "o" else ckc, mc)


def _install_ccl_h(ccl, dtype):
    """MeshCCL.all_gather of the residual (sites h_attn / h_ffn) in `dtype` on decode rows; bf16 in and out."""
    if getattr(ccl, "_h_dtype", None) is not None:
        return
    ccl._h_dtype = dtype
    orig = ccl.all_gather

    def all_gather(x, dim, dtype, decode, site=None, _o=orig, _dt=dtype):
        if decode and site in ("h_attn", "h_ffn") and x.dtype == ttnn.bfloat16:
            xc = ttnn.typecast(x, _dt)
            g = _o(xc, dim=dim, dtype=_dt, decode=decode, site=None)
            ttnn.deallocate(xc)
            out = ttnn.typecast(g, ttnn.bfloat16, memory_config=g.memory_config())
            ttnn.deallocate(g)
            return out
        return _o(x, dim=dim, dtype=dtype, decode=decode, site=site)

    ccl.all_gather = all_gather


def runtime_policy(model) -> dict:
    """The precision policy read back from the live model objects (weight dtypes and compute fidelities per group,
    first layer of each kind): the propagation check of the sweep."""
    lm = model.lm_head
    out = {"lm_head": {"dtype": str(lm.dtype) if lm is not None else "lm_ds", "fidelity": str(model.ckc.math_fidelity)}}
    seen = set()
    for d in model.layers:
        kind = (d.attn_kind, d.mlp_kind)
        attn = d.attn._inner if hasattr(d.attn, "_inner") else d.attn
        mlp = d.mlp._inner if hasattr(d.mlp, "_inner") else d.mlp
        rec = {}
        if d.attn_kind == "kda":
            rec["kda_in"] = str(attn.weights.input_projection.dtype)
            rec["kda_out"] = str(attn.weights.output_projection.dtype)
            rec["kda_ckc"] = str(attn.ckc.math_fidelity)
            rec["kda_in_decode_wrapped"] = "_in_proj" in attn.__dict__
        else:
            A = attn._pack[0] if getattr(attn, "_pack", None) else None
            rec["dsa_qa_kva_pack"] = str(A.w.dtype) if A is not None else None
            rec["dsa_q_b"] = str(attn.w_q_b.dtype)
            rec["dsa_o"] = str(attn.w_o.dtype)
            rec["dsa_w_uk"] = str(attn.w_uk.dtype)
            rec["dsa_ckc"] = str(attn.ckc.math_fidelity)
            rec["dsa_latent"] = f"{attn.latent.dtype} {attn.latent.layout}" if attn.latent is not None else None
            rec["dsa_pooled_keys"] = str(attn.pooled.dtype) if getattr(attn, "pooled", None) is not None else None
        if d.mlp_kind == "moe":
            p = mlp.prec
            rec["routed"] = f"{p.routed_gate_up}/{p.routed_down} {p.routed_fidelity}"
            rec["shared"] = f"{mlp.shared_packed.dtype}/{mlp.shared_w2.dtype} {mlp.shared_ckc.math_fidelity}"
            rec["router"] = f"{p.router} {p.router_fidelity}"
        else:
            gu = mlp.wgu if mlp.wgu is not None else mlp.wg
            rec["dense_mlp"] = f"{gu.dtype}/{mlp.wd.dtype} {mlp.ckc.math_fidelity}"
            rec["dense_mlp_decode_wrapped"] = "forward" in mlp.__dict__
        key = (kind, json.dumps(rec, sort_keys=True))
        if key not in seen:
            seen.add(key)
            out.setdefault("layers", []).append({"layer": getattr(d, "layer_idx", None), "kind": list(kind), **rec})
    return out
