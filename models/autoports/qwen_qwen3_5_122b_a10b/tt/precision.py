# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision policy of the Qwen3.5-122B-A10B autoport (datatype-sweep stage).

A policy is a JSON document (``doc/datatype_sweep/selected_precision_config.json`` is the shipped one) naming the weight
dtype of every tensor group, per-layer exceptions, the compute fidelity of every matmul group and the activation /
residual / CCL / KV-cache / recurrent-state / logits / sampling dtypes. :func:`apply_policy` turns it into the
``MultichipConfig`` (and its ``OptimizationConfig`` base) the runtime already consumes, so every field maps onto a real
knob; values the runtime cannot construct raise. :func:`effective_policy` reads the policy BACK from a built model (tensor
dtypes, kernel configs, args) so the sweep can prove the measured runtime used the requested policy
(:func:`check_effective`).

``QWEN35_PRECISION_CONFIG`` selects the file (``none`` = code defaults, i.e. the stage-7 policy) — see ``model.create_model``.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import ttnn

MODEL_DIR = Path(__file__).resolve().parents[1]
SELECTED_PRECISION_CONFIG = MODEL_DIR / "doc" / "datatype_sweep" / "selected_precision_config.json"

_DTYPES = {
    "bf16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfp4": ttnn.bfloat4_b,
    "fp32": ttnn.float32,
    "uint32": ttnn.uint32,
}
_DTYPE_NAMES = {v: k for k, v in _DTYPES.items()}
_FIDELITIES = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}
_FIDELITY_NAMES = {v: k for k, v in _FIDELITIES.items()}

# The stage-7 shipped runtime, spelled out as a policy (== MULTICHIP_SELECTED with no precision override).
BASELINE_POLICY = {
    "config_id": "baseline_s7",
    "description": "Stage-7 optimized full model: bfp4 routed experts (LoFi), bfp8 shared expert / token mixers / LM head "
    "(HiFi2), bf16 router (HiFi4), bfp8 paged KV cache, bf16 GDN state, bf16 activations / residual / CCL payload.",
    "weight_groups": {
        "embedding": {"dtype": "bf16"},
        "norms": {"dtype": "bf16"},
        "attention": {"dtype": "bfp8"},
        "linear_attention": {"dtype": "bfp8"},
        "router": {"dtype": "bf16"},
        "shared_moe": {"dtype": "bfp8"},
        "routed_moe": {"gate_up": "bfp4", "down": "bfp4"},
        "lm_head": {"dtype": "bfp8"},
    },
    # {"<layer_idx>": {"routed_moe": {"gate_up": "bfp8", "down": "bfp8"}, "routed_moe_fidelity": "hifi2"}}
    "layer_exceptions": {},
    "compute_fidelities": {
        "attention": "hifi2",
        "linear_attention": "hifi2",
        "router": "hifi4",
        "shared_moe": "hifi2",
        "routed_moe": "lofi",
        "lm_head": "ttnn_default",
    },
    "activation_dtype": "bf16",
    "residual_dtype": "bf16",
    "ccl_dtype": "bf16",
    "kv_cache_dtype": "bfp8",
    "linear_state_dtype": "bf16",
    "logits_dtype": "bf16",
    "sampling_dtype": "uint32",
    "notes": {
        "attention_linear_attention_fidelity": "one knob (MultichipConfig.mixer_fidelity -> args.mixer_compute_cfg) drives both "
        "token-mixer kinds AND everything the TP mixers run with that compute config: the projection matmuls (decode and prefill, "
        "incl. the fused all-gather+matmul) and the chunked PREFILL SDPA (TPAttention.compute_cfg); decode SDPA takes no compute "
        "config. hifi2 = tp_common.COMPUTE_HIFI2 (HiFi2, approx, fp32 dest acc, packer L1 acc); lofi = the same flags with LoFi "
        "math, i.e. the only delta is the fidelity",
        "lm_head_fidelity_ttnn_default": "ttnn.linear with no compute config resolves, for a bf16 activation x bfp8/bfp4 weight, "
        "to HiFi2 with fp32_dest_acc_en=False and packer_l1_acc=True (matmul_device_operation.cpp: increase_fidelity when the "
        "activation is not a block float); the shipped bfp4 LM head therefore runs bfp4 + HiFi2. 'hifi2_default' pins exactly "
        "that config explicitly; 'lofi' | 'hifi2' | 'hifi4' build the mixers' style config with fp32 dest accumulation ON",
        "routed_moe_fidelity": "routed expert matmuls run with fp32_dest_acc_en=False (tt-metal #49068) at any fidelity",
        "ccl_dtype": "bf16 = decode out-projection partials in bf16; bfp8 = the matmul emits bfp8 partials straight into "
        "the reduce-scatter (MultichipConfig.out_proj_decode_dtype); prefill collectives stay bf16",
        "lm_head_fidelity": "ttnn_default = ttnn.linear's default compute config; lofi | hifi2 | hifi4 build a "
        "WormholeComputeKernelConfig (fp32 dest + packer L1 accumulation) for every LM-head matmul (args.lm_head_compute_cfg)",
        "fixed": "embedding (ROW_MAJOR bf16 gather tables), norms (bf16 replicated weights), residual bf16, logits bf16 "
        "(LM head output read by the on-device sampler) and sampling uint32 tokens are fixed by the runtime: any other "
        "value raises in apply_policy",
    },
}

_FIXED = {
    ("weight_groups", "embedding", "dtype"): "bf16",
    ("weight_groups", "norms", "dtype"): "bf16",
    ("residual_dtype",): "bf16",
    ("logits_dtype",): "bf16",
    ("sampling_dtype",): "uint32",
}


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def _dtype(name, what):
    try:
        return _DTYPES[str(name).lower()]
    except KeyError:
        raise ValueError(f"{what}: unsupported dtype {name!r} (one of {sorted(_DTYPES)})")


def _fidelity(name, what):
    try:
        return _FIDELITIES[str(name).lower()]
    except KeyError:
        raise ValueError(f"{what}: unsupported compute fidelity {name!r} (one of {sorted(_FIDELITIES)})")


def load_policy(path=None):
    """The policy dict: ``path`` / ``QWEN35_PRECISION_CONFIG`` / the shipped file if present / :data:`BASELINE_POLICY`.
    ``QWEN35_PRECISION_CONFIG=none`` forces the code defaults."""
    env = os.environ.get("QWEN35_PRECISION_CONFIG")
    if path is None and env:
        if env.lower() == "none":
            return copy.deepcopy(BASELINE_POLICY), None
        path = env
    if path is None and SELECTED_PRECISION_CONFIG.is_file():
        path = SELECTED_PRECISION_CONFIG
    if path is None:
        return copy.deepcopy(BASELINE_POLICY), None
    path = Path(path)
    policy = json.loads(path.read_text())
    return policy, path


def policy_from_overrides(config_id, description="", **overrides):
    """A candidate = the baseline policy with nested overrides (``weight_groups={"routed_moe": {...}}`` merges)."""

    def merge(base, over):
        out = copy.deepcopy(base)
        for k, v in over.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
        return out

    pol = merge(BASELINE_POLICY, overrides)
    pol["config_id"] = config_id
    if description:
        pol["description"] = description
    pol.pop("notes", None)
    return pol


def apply_policy(mc, policy):
    """``MultichipConfig`` -> the same config constructing ``policy``. Raises on values the runtime cannot construct."""
    for path, fixed in _FIXED.items():
        v = _get(policy, path)
        if str(v).lower() != fixed:
            raise ValueError(
                f"{'.'.join(path)}={v!r} is fixed at {fixed!r} by this runtime (see BASELINE_POLICY notes)"
            )
    wg, cf = policy["weight_groups"], policy["compute_fidelities"]
    if str(cf["attention"]).lower() != str(cf["linear_attention"]).lower():
        raise ValueError("attention and linear_attention fidelity must agree (one mixer_fidelity knob)")
    mixer_fid = str(cf["attention"]).lower()
    if mixer_fid not in ("hifi2", "lofi"):
        raise ValueError(f"token-mixer fidelity {mixer_fid!r}: the runtime constructs hifi2 | lofi")
    act = _dtype(policy["activation_dtype"], "activation_dtype")
    if act != ttnn.bfloat16:
        raise ValueError("activation_dtype: the runtime constructs bf16 activations")
    ccl = str(policy["ccl_dtype"]).lower()
    if ccl not in ("bf16", "bfp8"):
        raise ValueError("ccl_dtype: bf16 | bfp8")
    state = str(policy["linear_state_dtype"]).lower()
    if state not in ("bf16", "fp32"):
        raise ValueError("linear_state_dtype: bf16 | fp32")
    base = mc.base.with_(
        routed_gate_up_dtype=_dtype(wg["routed_moe"]["gate_up"], "routed_moe.gate_up"),
        routed_down_dtype=_dtype(wg["routed_moe"]["down"], "routed_moe.down"),
        routed_fidelity=_fidelity(cf["routed_moe"], "routed_moe"),
        shared_dtype=_dtype(wg["shared_moe"]["dtype"], "shared_moe"),
        shared_fidelity=_fidelity(cf["shared_moe"], "shared_moe"),
        router_dtype=_dtype(wg["router"]["dtype"], "router"),
        router_fidelity=_fidelity(cf["router"], "router"),
        attn_weight_dtype=_dtype(wg["attention"]["dtype"], "attention"),
        gdn_weight_dtype=_dtype(wg["linear_attention"]["dtype"], "linear_attention"),
        act_dtype=act,
    )
    exceptions = []
    for layer, over in sorted(policy.get("layer_exceptions", {}).items(), key=lambda kv: int(kv[0])):
        fields = {}
        if "routed_moe" in over:
            if "gate_up" in over["routed_moe"]:
                fields["routed_gate_up"] = _dtype(over["routed_moe"]["gate_up"], f"layer {layer} routed_moe.gate_up")
            if "down" in over["routed_moe"]:
                fields["routed_down"] = _dtype(over["routed_moe"]["down"], f"layer {layer} routed_moe.down")
        if "routed_moe_fidelity" in over:
            fields["routed_fidelity"] = _fidelity(over["routed_moe_fidelity"], f"layer {layer} routed_moe_fidelity")
        if "shared_moe" in over:
            fields["shared"] = _dtype(over["shared_moe"]["dtype"], f"layer {layer} shared_moe")
        if "shared_moe_fidelity" in over:
            fields["shared_fidelity"] = _fidelity(over["shared_moe_fidelity"], f"layer {layer} shared_moe_fidelity")
        unknown = set(over) - {"routed_moe", "routed_moe_fidelity", "shared_moe", "shared_moe_fidelity"}
        if unknown:
            raise ValueError(
                f"layer_exceptions[{layer}]: only MoE-block exceptions are constructible, got {sorted(unknown)}"
            )
        exceptions.append((int(layer), fields))
    lm_fid = str(cf["lm_head"]).lower()
    if lm_fid not in ("ttnn_default", "hifi2_default", "lofi", "hifi2", "hifi4"):
        raise ValueError("compute_fidelities.lm_head: ttnn_default | hifi2_default | lofi | hifi2 | hifi4")
    return mc.with_(
        base=base,
        attn_weight_dtype=base.attn_weight_dtype,
        gdn_weight_dtype=base.gdn_weight_dtype,
        lm_head_dtype=_dtype(wg["lm_head"]["dtype"], "lm_head"),
        lm_head_fidelity=None if lm_fid == "ttnn_default" else lm_fid,
        precision_layer_exceptions=tuple(exceptions),
        kv_dtype=_dtype(policy["kv_cache_dtype"], "kv_cache_dtype"),
        gdn_state_dtype=ttnn.float32 if state == "fp32" else ttnn.bfloat16,
        gdn_decode_high_precision=state == "fp32",
        out_proj_decode_dtype="bfp8" if ccl == "bfp8" else None,
        mixer_fidelity=mixer_fid,
    )


def _name(dtype):
    return _DTYPE_NAMES.get(dtype, str(dtype))


def _fid_name(ckc):
    return _FIDELITY_NAMES.get(getattr(ckc, "math_fidelity", ckc), str(ckc))


def _lm_head_fid_name(ckc):
    """Name of the LM-head compute config: None = "ttnn_default" (NOT read back from the kernel: ttnn.linear resolves it at
    dispatch — HiFi2 / fp32 dest off / packer L1 acc on for bf16 x block-float in this build, matmul_device_operation.cpp);
    an explicit config is named by its fidelity, "hifi2_default" when it spells the resolved default out."""
    if ckc is None:
        return "ttnn_default"
    fid = _fid_name(ckc)
    if fid == "hifi2" and not bool(ckc.fp32_dest_acc_en) and bool(ckc.packer_l1_acc):
        return "hifi2_default"
    return fid


def effective_policy(model, args, kv_dtype=None):
    """Read the policy back from the built model: tensor dtypes of every weight group per layer, the kernel configs and
    the args the runtime consumed. ``kv_dtype`` = the generator's allocated KV cache dtype (args.kv_dtype if None)."""
    layers = {}
    for i, layer in enumerate(model.layers):
        li = int(getattr(args, "layer_indices", None)[i]) if getattr(args, "layer_indices", None) else i
        rec = {}
        attn = layer.attention
        tw = getattr(attn, "tw", {})
        if layer.is_full_attention:
            rec["attention"] = _name((tw.get("wqkv_fused") or tw.get("wqkv")).dtype)
            rec["attention_out"] = _name(tw["wo"].dtype)
        else:
            rec["linear_attention"] = _name(tw["qkvz"].dtype)
            rec["linear_attention_out"] = _name(tw["out"].dtype)
        ff = layer.feed_forward
        if hasattr(ff, "up_gate"):
            rec["routed_gate_up"] = _name(ff.up_gate.dtype)
            rec["routed_down"] = _name(ff.down.dtype)
            rec["routed_fidelity"] = _fid_name(ff.routed_ckc)
            shared = (
                getattr(ff, "shared_packed", None) or getattr(ff, "shared_w1", None) or ff.shared_w2
            )  # packed [w3|w1|gate]
            rec["shared"] = _name(shared.dtype)
            rec["shared_down"] = _name(ff.shared_w2.dtype)
            rec["shared_fidelity"] = _fid_name(ff.shared_ckc)
            rec["router"] = _name(ff.router_w.dtype)
            rec["router_fidelity"] = _fid_name(ff.router_ckc)
        layers[li] = rec
    mixer = getattr(args, "mixer_compute_cfg", None)
    return {
        "lm_head": _name(model.lm_head_weight.dtype),
        "lm_head_fidelity": _lm_head_fid_name(getattr(args, "lm_head_compute_cfg", None)),
        "mixer_fidelity": _fid_name(mixer) if mixer is not None else "hifi2",
        "kv_cache_dtype": _name(kv_dtype if kv_dtype is not None else args.kv_dtype),
        "linear_state_dtype": _name(getattr(args, "gdn_state_dtype", ttnn.bfloat16)),
        "linear_state_high_precision": bool(getattr(args, "gdn_decode_high_precision", False)),
        "mixer_compute_cfg": (
            {
                "fidelity": _fid_name(mixer),
                "fp32_dest_acc_en": bool(mixer.fp32_dest_acc_en),
                "packer_l1_acc": bool(mixer.packer_l1_acc),
                "math_approx_mode": bool(mixer.math_approx_mode),
            }
            if mixer is not None
            else "tp_common.COMPUTE_HIFI2"
        ),
        "ccl_dtype": "bfp8" if getattr(args, "out_proj_decode_dtype", None) == ttnn.bfloat8_b else "bf16",
        "layers": layers,
    }


def check_effective(policy, eff):
    """Every requested field == the built model's. Returns the list of mismatches (empty = proven)."""
    wg, cf = policy["weight_groups"], policy["compute_fidelities"]
    exc = {int(k): v for k, v in policy.get("layer_exceptions", {}).items()}
    bad = []

    def want(li, group, key, default):
        return (
            exc.get(li, {}).get(group, {}).get(key, default)
            if isinstance(exc.get(li, {}).get(group), dict)
            else default
        )

    def eq(what, got, exp):
        if str(got).lower() != str(exp).lower():
            bad.append(f"{what}: built {got} != policy {exp}")

    eq("lm_head", eff["lm_head"], wg["lm_head"]["dtype"])
    eq("lm_head_fidelity", eff.get("lm_head_fidelity", "ttnn_default"), cf["lm_head"])
    eq("mixer_fidelity", eff["mixer_fidelity"], cf["attention"])
    eq("kv_cache_dtype", eff["kv_cache_dtype"], policy["kv_cache_dtype"])
    eq("linear_state_dtype", eff["linear_state_dtype"], policy["linear_state_dtype"])
    eq(
        "linear_state_high_precision",
        eff.get("linear_state_high_precision"),
        str(policy["linear_state_dtype"]).lower() == "fp32",
    )
    eq("ccl_dtype", eff["ccl_dtype"], policy["ccl_dtype"])
    for li, rec in eff["layers"].items():
        li = int(li)
        if "attention" in rec:
            eq(f"layer {li} attention", rec["attention"], wg["attention"]["dtype"])
            eq(f"layer {li} attention_out", rec["attention_out"], wg["attention"]["dtype"])
        else:
            eq(f"layer {li} linear_attention", rec["linear_attention"], wg["linear_attention"]["dtype"])
            eq(f"layer {li} linear_attention_out", rec["linear_attention_out"], wg["linear_attention"]["dtype"])
        if "routed_gate_up" in rec:
            eq(
                f"layer {li} routed_gate_up",
                rec["routed_gate_up"],
                want(li, "routed_moe", "gate_up", wg["routed_moe"]["gate_up"]),
            )
            eq(f"layer {li} routed_down", rec["routed_down"], want(li, "routed_moe", "down", wg["routed_moe"]["down"]))
            eq(
                f"layer {li} routed_fidelity",
                rec["routed_fidelity"],
                exc.get(li, {}).get("routed_moe_fidelity", cf["routed_moe"]),
            )
            eq(f"layer {li} shared", rec["shared"], want(li, "shared_moe", "dtype", wg["shared_moe"]["dtype"]))
            if "shared_down" in rec:
                eq(
                    f"layer {li} shared_down",
                    rec["shared_down"],
                    want(li, "shared_moe", "dtype", wg["shared_moe"]["dtype"]),
                )
            eq(
                f"layer {li} shared_fidelity",
                rec["shared_fidelity"],
                exc.get(li, {}).get("shared_moe_fidelity", cf["shared_moe"]),
            )
            eq(f"layer {li} router", rec["router"], wg["router"]["dtype"])
            eq(f"layer {li} router_fidelity", rec["router_fidelity"], cf["router"])
    return bad


def describe_config(mc):
    """The precision-relevant MultichipConfig fields as names (for logs / sweep_results)."""
    b = mc.base
    return {
        "routed_gate_up": _name(b.routed_gate_up_dtype),
        "routed_down": _name(b.routed_down_dtype),
        "routed_fidelity": _FIDELITY_NAMES.get(b.routed_fidelity, str(b.routed_fidelity)),
        "shared": _name(b.shared_dtype),
        "shared_fidelity": _FIDELITY_NAMES.get(b.shared_fidelity, str(b.shared_fidelity)),
        "router": _name(b.router_dtype),
        "router_fidelity": _FIDELITY_NAMES.get(b.router_fidelity, str(b.router_fidelity)),
        "attention": _name(mc.attn_weight_dtype or ttnn.bfloat8_b),
        "linear_attention": _name(mc.gdn_weight_dtype or ttnn.bfloat8_b),
        "lm_head": _name(mc.lm_head_dtype or ttnn.bfloat8_b),
        "lm_head_fidelity": mc.lm_head_fidelity or "ttnn_default",
        "mixer_fidelity": mc.mixer_fidelity,
        "kv_cache_dtype": _name(mc.kv_dtype),
        "linear_state_dtype": _name(mc.gdn_state_dtype),
        "ccl_dtype": mc.out_proj_decode_dtype or "bf16",
        "layer_exceptions": {
            str(k): {f: _name(v) if v in _DTYPE_NAMES else _FIDELITY_NAMES.get(v, str(v)) for f, v in d.items()}
            for k, d in mc.precision_layer_exceptions
        },
    }
