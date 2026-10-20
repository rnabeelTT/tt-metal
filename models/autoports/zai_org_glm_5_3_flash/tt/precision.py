# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Datatype-sweep precision overrides (stage 8) on top of the shipped policy.

The shipped policy is tt/optimized_decoder.SELECTED (per-layer weight dtypes and MoE fidelities), tt/multichip_decoder
.DEFAULT_MC (decode programs, CCL) and the model-level constants of tt/model.py (LM head bfp8 at HiFi2, bf16 residual,
bf16 DSA latent pool). doc/datatype_sweep/selected_precision_config.json records the complete selected policy.

GLM53_PRECISION selects the policy at construction (Glm53Model.__init__ / MultichipDecoder.from_state_dict, so the
generator, the readiness runners and the vLLM adapter all consume it): unset or empty = the stage-8 selected policy
doc/datatype_sweep/selected_precision_config.json (the model default); "none" / "shipped" = the stage-7 policy (no
override); <path> = another candidate JSON. Fields:

  id                       candidate id (logged and recorded)
  requant                  {group: {"dtype": "bfp4" | "bfp8", "layers": "all" | "inner" | [i, ...]}}: host round-to-
                           nearest re-quantization of the loaded weights of a group (groups: kda_in, kda_out, dsa_proj
                           = q_a | kv_a pack + q_b + o_proj, dense_mlp = gate|up + down, shared_expert, lm_head). The
                           loaded weights come from the
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
SELECTED_CONFIG = Path(__file__).resolve().parents[1] / "doc" / "datatype_sweep" / "selected_precision_config.json"
_NONE = ("none", "shipped", "off", "0")
_DT = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
_FID = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
# layer kind (attention or MLP) whose module a decode-program field drives
_SPEC_KIND = {
    "kda_in": "kda",
    "kda_out": "kda",
    "dsa_o": "dsa",
    "dsa_qb": "dsa",
    "dsa_wqb": "dsa",
    "pack_a_spec": "dsa",
    "pack_b_spec": "dsa",
    "shared_gu": "moe",
    "shared_dn": "moe",
    "mlp_gu": "dense",
    "mlp_dn": "dense",
}
_MOE_DTYPE_FIELDS = ("routed_gate_up_dtype", "routed_down_dtype", "shared_dtype", "router_dtype")
# decode-row matmuls run through the fidelity wrappers, per (group, fidelity): the per-step propagation check
# (tools/dtype_sweep.py --do fidcount resets it and runs one eager decode step)
FIDELITY_SEEN: dict = {}


def _seen(group: str, ck):
    k = f"{group}:{str(ck.math_fidelity).split('.')[-1]}"
    FIDELITY_SEEN[k] = FIDELITY_SEEN.get(k, 0) + 1


GROUPS = ("kda_in", "kda_out", "dsa_proj", "dense_mlp", "shared_expert", "routed", "lm_head")


def _decode_rows(t) -> bool:
    return int(t.padded_shape[-2]) == 32


def _ckc(mesh, fidelity, fp32_acc=True, l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=l1_acc
    )


def _ckc_like(orig, mesh, fidelity):
    """The module's own compute config with only the math fidelity changed (fp32 dest accumulation, packer L1
    accumulation and approx mode are kept: e.g. the LM head's packer_l1_acc=True)."""
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=fidelity,
        math_approx_mode=bool(getattr(orig, "math_approx_mode", False)),
        fp32_dest_acc_en=bool(getattr(orig, "fp32_dest_acc_en", True)),
        packer_l1_acc=bool(getattr(orig, "packer_l1_acc", False)),
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


def _rq_version() -> str:
    import hashlib
    import inspect

    return hashlib.sha256(inspect.getsource(requant).encode()).hexdigest()[:8]


def _short(obj) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def rq_name(key: str, src, dtype, component: str) -> str:
    """File name of a re-quantized tensor in the weight cache: the group key, the target dtype, and a digest of what
    determines its bytes: the owning component's build-source digest (tt/weight_cache.component_digest), the requant
    source digest (scope), and the SOURCE tensor's memory config, padded shape and dtype."""
    sig = {
        "component": component,
        "memory_config": str(src.memory_config()),
        "padded_shape": [int(x) for x in src.padded_shape],
        "src_dtype": str(src.dtype),
    }
    return f"{key}-{str(dtype).split('.')[-1].lower()}-{_short(sig)}"


def _component(kind: str) -> str:
    """Owning component's build-source digest (tt/weight_cache.component_digest). "dsa_pack": the q_a | kv_a pack is
    built after the cache load by MultichipDSA._build_pack (concat + 1D / DRAM-sharded matmul object), so its digest
    also covers that code."""
    try:
        import hashlib
        import inspect

        from .weight_cache import component_digest

        if kind == "dsa_pack":
            from .multichip_decoder import DSMatmul, IL1DMatmul, MultichipDSA, make_role_mm

            src = "".join(inspect.getsource(o) for o in (MultichipDSA._build_pack, DSMatmul, IL1DMatmul, make_role_mm))
            return f"dsa-{component_digest('dsa')}-pack-{hashlib.sha256(src.encode()).hexdigest()[:8]}"
        return f"{kind}-{component_digest(kind)}"
    except ImportError as e:  # host tests without the device package: logged, and the name says so
        logger.warning(f"precision cache: no component digest for {kind} ({e})")
        return f"{kind}-nodigest-ImportError"


@dataclass
class PrecisionOverrides:
    id: str = "shipped"
    requant: dict = field(default_factory=dict)
    fidelity: dict = field(default_factory=dict)
    fidelity_layers: dict = field(default_factory=dict)
    moe_layer_exceptions: dict = field(default_factory=dict)
    ccl_h_dtype: str = "bf16"
    kv_latent_dtype: str = "bf16"
    # {MCOpts role-spec field: {"spec": "1d:c..:ib.." | "ds:..", "layers": "all" | "inner" | [i, ...]}}: decode
    # program of a group on the layers where it applies (e.g. a geometry measured for bfp4 weights); DEFAULT_MC unchanged
    decode_specs: dict = field(default_factory=dict)
    source: str | None = None
    applied: list = field(default_factory=list)  # (layer or "model", group, what) records of the runtime changes
    _cache: object = field(default=None, repr=False)  # WeightCache scope of the re-quantized tensors (current layer)

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
        if p.lower() in _NONE:
            return cls()
        if not p:  # the model default: the stage-8 selected policy
            p = str(SELECTED_CONFIG)
        d = json.loads(Path(p).read_text())
        known = {f.name for f in dataclasses.fields(cls)} - {"source", "applied", "_cache"}
        unknown = set(d) - known - {"description", "notes", "policy"}
        if unknown:
            raise ValueError(f"{p}: unknown precision override fields {sorted(unknown)}")
        o = cls(**{k: v for k, v in d.items() if k in known}, source=str(p))
        for g in list(o.requant) + list(o.fidelity):
            if g not in GROUPS:
                raise ValueError(f"{p}: unknown group {g!r} (groups {GROUPS})")
        if p == str(SELECTED_CONFIG):
            logger.info(f"precision policy: stage-8 selected {o.id!r} ({p}; {ENV}=none for the stage-7 policy)")
        else:
            logger.warning(f"{ENV}={p}: precision candidate {o.id!r} overrides the selected policy")
        return o

    @property
    def is_shipped(self) -> bool:
        return not (
            self.requant
            or self.fidelity
            or self.moe_layer_exceptions
            or self.ccl_h_dtype != "bf16"
            or self.kv_latent_dtype != "bf16"
            or self.decode_specs
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
    def _rq(self, t, dtype, key: str, kind: str):
        """requant(t, dtype) through the weight cache: the re-quantized device tensor is stored under
        <layer or top>/prec-<requant source digest>/<rq_name>.tensorbin (rq_name: group, dtype and a digest of the
        owning component's source digest + the source tensor's memory config / padded shape / dtype), so a warm load
        reads the final tensor (the shipped tensor it replaces is still loaded first and freed). A loaded file whose
        memory config, padded shape or dtype does not match is rebuilt. No cache: re-quantize on every load."""
        c = self._cache
        if c is None:
            return requant(t, dtype)
        name = rq_name(key, t, dtype, _component(kind))
        if c.has(name):
            try:
                out = c.load(name)
                if (
                    str(out.memory_config()) == str(t.memory_config())
                    and [int(x) for x in out.padded_shape] == [int(x) for x in t.padded_shape]
                    and out.dtype == dtype
                ):
                    ttnn.deallocate(t)
                    return out
                logger.warning(f"precision cache: {c.file(name)} does not match its source tensor; re-quantizing")
                ttnn.deallocate(out)
            except RuntimeError as e:  # corrupt file: rebuild it
                logger.warning(f"precision cache: {c.file(name)} unreadable ({e}); re-quantizing")
        return c.store(name, requant(t, dtype))

    def _scope(self, cache, scope: str):
        self._cache = cache.sub(f"{scope}/prec-{_rq_version()}") if cache is not None else None

    def apply_layer(self, dec, i: int, first: int, last: int, cache=None):
        """Re-quantize / re-fidelity the groups of one built decoder layer (before warm-up and trace capture).
        cache: the model's WeightCache (root scope) or None. A layer with a weight-dtype exception is built from the
        checkpoint (layer_cache), so its re-quantized tensors are not cached either (their source differs)."""
        self._scope(None if self.layer_has_weight_exception(i) else cache, f"layer_{i}")
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
                mlp.shared_packed = self._rq(mlp.shared_packed, dt, "shared_packed", "moe")
                mlp.shared_w2 = self._rq(mlp.shared_w2, dt, "shared_w2", "moe")
                self.applied.append((i, "shared_expert", f"requant {dt}"))
            self.applied.append((i, "moe", f"routed {mlp.prec.routed_fidelity} shared {mlp.prec.shared_fidelity}"))
        else:
            dt = self.requant_dtype("dense_mlp", i, first, last)
            if dt is not None:
                for name in ("wgu", "wg", "wu", "wd"):
                    t = getattr(mlp, name, None)
                    if t is not None:
                        setattr(mlp, name, self._rq(t, dt, f"mlp_{name}", "mlp"))
                self.applied.append((i, "dense_mlp", f"requant {dt}"))
            f = self.fid("dense_mlp", i, first, last)
            if f is not None:
                _swap_ckc_on_decode(mlp, "forward", _ckc_like(mlp.ckc, mesh, f), "dense_mlp")
                self.applied.append((i, "dense_mlp", f"decode fidelity {f}"))
        for fld, d in self.decode_specs.items():
            kind = _SPEC_KIND.get(fld)
            if kind is not None and kind not in (dec.attn_kind, dec.mlp_kind):
                continue  # e.g. kda_in on a DSA layer: inert, not applied or logged
            if self._in(d.get("layers", "all"), i, first, last) and hasattr(getattr(dec, "mc", None), fld):
                setattr(dec.mc, fld, d["spec"])  # dec.mc is this layer's own MCOpts copy (MCOpts.from_env)
                attn = dec.attn._inner if hasattr(dec.attn, "_inner") else dec.attn
                if fld == "pack_a_spec" and getattr(attn, "_pack", None) is not None:
                    # the q_a | kv_a pack object is built at construction (ds_shared): rebuild it on the same weight
                    from .multichip_decoder import make_role_mm

                    A, B = attn._pack
                    attn._pack = (make_role_mm(d["spec"], A.w, mesh.compute_with_storage_grid_size().x), B)
                    attn._pack_specs = (d["spec"], attn._pack_specs[1]) if hasattr(attn, "_pack_specs") else None
                self.applied.append((i, f"spec:{fld}", d["spec"]))
        if self.ccl_h_dtype != "bf16" and getattr(dec, "ccl", None) is not None:
            _install_ccl_h(dec.ccl, _DT[self.ccl_h_dtype])
            self.applied.append((i, "ccl_h", self.ccl_h_dtype))

    def _apply_kda(self, attn, mesh, i, first, last):
        w = attn.weights
        new = {}
        for g, name in (("kda_in", "input_projection"), ("kda_out", "output_projection")):
            dt = self.requant_dtype(g, i, first, last)
            if dt is not None:
                new[name] = self._rq(getattr(w, name), dt, f"kda_{name}", "kda")
                self.applied.append((i, g, f"requant {dt}"))
        if new:
            nw = dataclasses.replace(w, **new)
            attn.weights = nw
            for p in attn._prefill.values():
                p.weights = nw
        f_in, f_out = self.fid("kda_in", i, first, last), self.fid("kda_out", i, first, last)
        if f_in is not None:
            ck = _ckc_like(attn.ckc, mesh, f_in)
            orig = attn._in_proj

            def _in_proj(x, w_, cc, _o=orig, _c=ck):
                if _decode_rows(x):
                    _seen("kda_in", _c)
                    return _o(x, w_, _c)
                return _o(x, w_, cc)

            attn._in_proj = _in_proj
            self.applied.append((i, "kda_in", f"decode fidelity {f_in}"))
        if f_out is not None:
            _wrap_decode_mm(attn, _ckc_like(attn.ckc, mesh, f_out), "kda_out")
            self.applied.append((i, "kda_out", f"decode fidelity {f_out}"))

    def _apply_dsa(self, attn, mesh, i, first, last):
        if self.kv_latent_dtype != "bf16":
            assert self.kv_latent_dtype == "fp8_e4m3", self.kv_latent_dtype
            assert attn.latent is None, "the latent pool dtype must be set before allocate_decode_state"
            # tt/dsa.py: ROW_MAJOR fp8 pool, sparse_sdpa kv_format FP8_E4M3; the block-sharded cache (tt/dsa_shard.py)
            # takes the same dtype in its 576-wide fp8 row format
            attn.latent_dtype = ttnn.fp8_e4m3
            self.applied.append(
                (i, "kv_latent", "fp8_e4m3 (dsa_shard)" if getattr(attn, "dsa_shard", False) else "fp8_e4m3")
            )
        dt = self.requant_dtype("dsa_proj", i, first, last)
        if dt is not None:
            A, B = attn._pack  # q_a | kv_a pack (serves prefill and decode: ds_shared), indexer pack (bf16, kept)
            A.w = self._rq(A.w, dt, "dsa_pack_a", "dsa_pack")
            attn.w_q_b = self._rq(attn.w_q_b, dt, "dsa_w_q_b", "dsa")
            attn.w_o = self._rq(attn.w_o, dt, "dsa_w_o", "dsa")
            self.applied.append((i, "dsa_proj", f"requant {dt} (q_a|kv_a pack, q_b, o_proj)"))
        f = self.fid("dsa_proj", i, first, last)
        if f is not None:
            ck = _ckc_like(attn.ckc, mesh, f)
            ids = set(attn._pack_ids) if getattr(attn, "_pack_ids", None) else set()
            ids.add(id(attn.w_q_b))
            orig = attn._mm

            def _mm(x, w, _o=orig, _ids=ids, _c=ck, _a=attn, **kw):
                if (id(w) in _ids or w is _a.w_q_b) and _decode_rows(x) and id(w) not in _a.__dict__.get("_ix", ()):
                    _seen("dsa_proj", _c)
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
            _wrap_decode_mm(attn, ck, "dsa_proj")
            self.applied.append((i, "dsa_proj", f"decode fidelity {f} (q_a|kv_a, q_b, o_proj)"))

    def apply_model(self, model, first: int, last: int):
        self._scope(getattr(model, "weight_cache", None), "top")
        dt = self.requant_dtype("lm_head", -1, first, last)
        if dt is not None and getattr(model, "lm_head", None) is None:  # GLM53_HEAD=lm_ds: column-split DS weights
            logger.warning("precision: lm_head requant skipped (GLM53_HEAD=lm_ds splits are not re-quantized)")
            self.applied.append(("model", "lm_head", "requant skipped (lm_ds)"))
        elif dt is not None:
            model.lm_head = self._rq(model.lm_head, dt, "lm_head", "top")
            self.applied.append(("model", "lm_head", f"requant {dt}"))
        f = self.fid("lm_head", -1, first, last)
        if f is not None:
            model.ckc = _ckc_like(model.ckc, model.mesh, f)
            self.applied.append(("model", "lm_head", f"fidelity {f}"))
        spec = (self.decode_specs.get("lm_head") or {}).get("spec")
        if spec and getattr(model, "lm_head", None) is not None:
            # decode rows of the LM head on an explicit role program (tt/multichip_decoder.make_role_mm) on the same
            # interleaved weight; prefill rows (all_logits) keep ttnn's default
            from .multichip_decoder import make_role_mm

            mm = make_role_mm(spec, model.lm_head, model.mesh.compute_with_storage_grid_size().x)
            orig = model._lm_head

            def _lm_head(hn, _o=orig, _m=model, _mm=mm):
                if _decode_rows(hn) and _m.lm_head is _mm.w:
                    return _mm(hn, _m.ckc)
                return _o(hn)

            model._lm_head = _lm_head
            self.applied.append(("model", "lm_head", f"decode spec {spec}"))

    def record(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "_cache"}
        d["applied"] = [list(map(str, a)) for a in self.applied]
        return d


def _swap_ckc_on_decode(obj, method: str, ck, group: str):
    """obj.<method>(x, ...) runs with obj.ckc = ck on decode rows (the method reads self.ckc per call)."""
    orig = getattr(obj, method)

    def wrapped(x, *a, _o=orig, **kw):
        if not _decode_rows(x):
            return _o(x, *a, **kw)
        _seen(group, ck)
        saved = obj.ckc
        obj.ckc = ck
        try:
            return _o(x, *a, **kw)
        finally:
            obj.ckc = saved

    setattr(obj, method, wrapped)


def _wrap_decode_mm(attn, ck, group: str):
    """The module's decode output-projection hook (RoleDecodeMM, role "o") with compute config ck."""
    inner = attn.decode_mm

    def decode_mm(x, w, role, ckc, mc, _i=inner, _c=ck):
        if role == "o":
            if _decode_rows(x):  # the hook declines prefill rows (RoleDecodeMM: decode rows only); count decode only
                _seen(f"{group}_o", _c)
            return _i(x, w, role, _c, mc)
        return _i(x, w, role, ckc, mc)

    attn.decode_mm = decode_mm


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
