# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-Small checkpoint access and config, without transformers' Inkling classes (the serving image's transformers
predates them). Key mapping = transformers' conversion_mapping for `inkling_mm_model` (and ref_inkling.py, which the
host tests check against the HF modules):

  model.llm.layers.N.attn.{wq_du,wk_dv,wv_dv,wr_du,wo_ud}.weight -> self_attn.{q,k,v,r,o}_proj.weight
  model.llm.layers.N.attn.{q_norm,k_norm}.weight, .rel_logits_proj.proj, .{k,v}_sconv.weight ([C, 1, 4])
  model.llm.layers.N.{attn_norm,mlp_norm}.weight -> input_layernorm / post_attention_layernorm
  model.llm.layers.N.{attn_sconv,mlp_sconv}.weight ([H, 1, 4])
  dense (N < 2):  mlp.w13_dn.weight [2*I, H] row-interleaved (gate even rows), mlp.w2_md.weight, mlp.global_scale
  sparse:         mlp.gate.{weight [258, H], bias [256], global_scale}, mlp.experts.w13_weight [E, 2*I, H] (interleaved on
                  dim 1), mlp.experts.w2_weight [E, H, I], mlp.shared_experts.shared_w13_weight [2, 2*I, H] (interleaved),
                  mlp.shared_experts.shared_w2_weight [2, H, I]
  model.llm.{embed,embed_norm,norm,unembed}.weight
"""

from __future__ import annotations

import glob
import json
import os
from types import SimpleNamespace

import torch
from safetensors import safe_open

REPO = "thinkingmachines/Inkling-Small"


def snapshot_dir(repo: str = REPO) -> str:
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    snaps = sorted(glob.glob(f"{hub}/models--{repo.replace('/', '--')}/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"{repo} not in {hub}")
    return snaps[0]


def load_config(snap: str | None = None) -> SimpleNamespace:
    """The text config with the derived fields transformers' InklingTextConfig adds (layer_types, mlp_layer_types)."""
    snap = snap or snapshot_dir()
    raw = json.load(open(os.path.join(snap, "config.json")))
    t = dict(raw["text_config"])
    L = t["num_hidden_layers"]
    local = set(t["local_layer_ids"])
    t["layer_types"] = ["hybrid_sliding" if i in local else "hybrid" for i in range(L)]
    t["mlp_layer_types"] = ["dense" if i < t["dense_mlp_idx"] else "sparse" for i in range(L)]
    # the checkpoint's intermediate_size is the expert width (2048); dense_intermediate_size the dense MLPs' (16384).
    # (transformers 5.17's InklingTextConfig mis-resolves this: moe_intermediate_size stays at its default 3072.)
    t["moe_intermediate_size"] = t["intermediate_size"]
    t["intermediate_size"] = t["dense_intermediate_size"]
    t["conv_kernel_size"] = t.get("sconv_kernel_size", 4)
    t["mtp"] = raw.get("mtp_config")
    t["eos_token_id"] = raw.get("eos_token_id")
    return SimpleNamespace(**t)


def deinterleave(t: torch.Tensor, dim: int):
    """Row-interleaved fused gate/up (gate0, up0, gate1, ...) along dim -> (gate, up)."""
    pairs = t.unflatten(dim, (t.shape[dim] // 2, 2))
    return pairs.select(dim + 1, 0).contiguous(), pairs.select(dim + 1, 1).contiguous()


class InklingCheckpoint:
    def __init__(self, snap: str | None = None):
        self.snap = snap or snapshot_dir()
        self.wmap = json.load(open(os.path.join(self.snap, "model.safetensors.index.json")))["weight_map"]
        self._files = {}
        self.cfg = load_config(self.snap)

    def get(self, key: str) -> torch.Tensor:
        shard = self.wmap[key]
        f = self._files.get(shard) or self._files.setdefault(shard, safe_open(os.path.join(self.snap, shard), "pt"))
        return f.get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self.wmap

    def attn(self, li: int, prefix: str = "model.llm.layers") -> dict:
        a = f"{prefix}.{li}.attn."
        g = self.get
        return {
            "q_proj.weight": g(a + "wq_du.weight"),
            "k_proj.weight": g(a + "wk_dv.weight"),
            "v_proj.weight": g(a + "wv_dv.weight"),
            "r_proj.weight": g(a + "wr_du.weight"),
            "o_proj.weight": g(a + "wo_ud.weight"),
            "q_norm.weight": g(a + "q_norm.weight"),
            "k_norm.weight": g(a + "k_norm.weight"),
            "k_sconv.conv1d.weight": g(a + "k_sconv.weight"),
            "v_sconv.conv1d.weight": g(a + "v_sconv.weight"),
            "rel_logits_proj.proj": g(a + "rel_logits_proj.proj"),
        }

    def layer_misc(self, li: int, prefix: str = "model.llm.layers") -> dict:
        p = f"{prefix}.{li}."
        return {
            "input_layernorm.weight": self.get(p + "attn_norm.weight"),
            "post_attention_layernorm.weight": self.get(p + "mlp_norm.weight"),
            "attn_sconv": self.get(p + "attn_sconv.weight"),
            "mlp_sconv": self.get(p + "mlp_sconv.weight"),
        }

    def dense_mlp(self, li: int, prefix: str = "model.llm.layers") -> dict:
        m = f"{prefix}.{li}.mlp."
        g, u = deinterleave(self.get(m + "w13_dn.weight"), 0)
        return {"gate_proj.weight": g, "up_proj.weight": u, "down_proj.weight": self.get(m + "w2_md.weight"),
                "global_scale": self.get(m + "global_scale")}

    def router(self, li: int) -> dict:
        m = f"model.llm.layers.{li}.mlp."
        return {"weight": self.get(m + "gate.weight"), "bias": self.get(m + "gate.bias"),
                "global_scale": self.get(m + "gate.global_scale")}

    def shared_experts(self, li: int) -> dict:
        m = f"model.llm.layers.{li}.mlp.shared_experts."
        g, u = deinterleave(self.get(m + "shared_w13_weight"), 1)  # [2, I, H] each
        return {"gate_proj": g, "up_proj": u, "down_proj": self.get(m + "shared_w2_weight")}  # down [2, H, I]

    def experts(self, li: int):
        """(gate_up [E, 2I, H] with gate rows first, down [E, H, I]) -- the layout the qwen35 port's fused experts use."""
        m = f"model.llm.layers.{li}.mlp.experts."
        g, u = deinterleave(self.get(m + "w13_weight"), 1)
        return torch.cat([g, u], 1), self.get(m + "w2_weight")


# ---------------------------------------------------------------------------------------------------------------------
# Streaming state dict in the qwen35 autoport's internal key scheme (what the shared layer / MoE code and the hybrid
# overlay read). The 2 shared experts are appended to the routed ones as experts 256 and 257: Inkling's router weights
# them in the same normalisation as the 6 routed picks (tt/moe.py), so the MoE is 8 weighted experts of 258.
#
#   tok_embeddings.weight, embed_norm.weight, norm.weight, output.weight
#   layers.N.{input,post_attention}_layernorm.weight, layers.N.{attn,mlp}_sconv.weight [H, 1, 4]
#   layers.N.self_attn.{q,k,v,r,o}_proj.weight, .{q,k}_norm.weight, .{k,v}_sconv.conv1d.weight, .rel_logits_proj.proj
#   dense:  layers.N.mlp.{gate,up,down}_proj.weight, layers.N.mlp.global_scale
#   sparse: layers.N.mlp.gate.weight [258, H], .gate.e_score_correction_bias [256], .gate.global_scale [1]
#           layers.N.mlp.experts.gate_up_proj [258, 2I, H] (gate rows first), .experts.down_proj [258, H, I]
# ---------------------------------------------------------------------------------------------------------------------
from collections.abc import Mapping  # noqa: E402

N_SHARED = 2
_ATTN = {
    "wq_du.weight": "q_proj.weight",
    "wk_dv.weight": "k_proj.weight",
    "wv_dv.weight": "v_proj.weight",
    "wr_du.weight": "r_proj.weight",
    "wo_ud.weight": "o_proj.weight",
    "q_norm.weight": "q_norm.weight",
    "k_norm.weight": "k_norm.weight",
    "k_sconv.weight": "k_sconv.conv1d.weight",
    "v_sconv.weight": "v_sconv.conv1d.weight",
    "rel_logits_proj.proj": "rel_logits_proj.proj",
}


class StreamingStateDict(Mapping):
    """Lazy read-only mapping internal key -> tensor (bf16 as stored; fp32 scalars stay fp32). Also the hybrid overlay's
    accessors: ``one_expert(li, e)`` (one expert, slice-read from the fused checkpoint tensor) and
    ``_fused_experts(li, which, ids=None)`` (a stack of experts in a given order, -1 = zero expert)."""

    def __init__(self, checkpoint_dir=None, *, num_layers=None, **_unused):
        self.ck = InklingCheckpoint(str(checkpoint_dir) if checkpoint_dir else None)
        self.checkpoint_dir = self.ck.snap
        cfg = self.ck.cfg
        self.n_main_layers = num_layers if num_layers is not None else cfg.num_hidden_layers
        self.E = cfg.n_routed_experts
        self._keys: dict[int, list[str]] = {}
        for li in range(self.n_main_layers):
            ks = [f"layers.{li}.{n}" for n in ("input_layernorm.weight", "post_attention_layernorm.weight",
                                                "attn_sconv.weight", "mlp_sconv.weight")]
            ks += [f"layers.{li}.self_attn.{v}" for v in _ATTN.values()]
            if cfg.mlp_layer_types[li] == "dense":
                ks += [f"layers.{li}.mlp.{n}" for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight",
                                                        "global_scale")]
            else:
                ks += [f"layers.{li}.mlp.{n}" for n in ("gate.weight", "gate.e_score_correction_bias",
                                                        "gate.global_scale", "experts.gate_up_proj",
                                                        "experts.down_proj")]
            self._keys[li] = ks
        self._top = {"tok_embeddings.weight": "model.llm.embed.weight", "embed_norm.weight": "model.llm.embed_norm.weight",
                     "norm.weight": "model.llm.norm.weight", "output.weight": "model.llm.unembed.weight"}
        self._slices = {}
        self._scales = {}

    # --- raw access
    def _hf(self, hf_key):
        return self.ck.get(hf_key)

    def _slice(self, hf_key):
        sl = self._slices.get(hf_key)
        if sl is None:
            shard = self.ck.wmap[hf_key]
            f = self.ck._files.get(shard) or self.ck._files.setdefault(
                shard, safe_open(os.path.join(self.ck.snap, shard), "pt"))
            sl = self._slices[hf_key] = f.get_slice(hf_key)
        return sl

    def expert_scale(self, li: int) -> float:
        """route_scale x the router's global_scale of layer li: folded into every expert's down projection (routed and
        shared), so the router's weights are a plain normalisation (sum 1) on every path (device, host int4, streamed
        device images, the hybrid's crafted rows)."""
        sc = self._scales.get(li)
        if sc is None:
            gs = float(self.ck.get(f"model.llm.layers.{li}.mlp.gate.global_scale").float().reshape(-1)[0])
            sc = self._scales[li] = float(self.ck.cfg.route_scale) * gs
        return sc

    def one_expert(self, li: int, e: int):
        """(gate_up [2I, H] gate rows first, down [H, I] x expert_scale(li)) bf16 of expert e (e >= 256: shared expert
        e - 256)."""
        m = f"model.llm.layers.{li}.mlp."
        if e < self.E:
            gu, dn = self._slice(m + "experts.w13_weight")[e], self._slice(m + "experts.w2_weight")[e]
        else:
            s = e - self.E
            gu = self._slice(m + "shared_experts.shared_w13_weight")[s]
            dn = self._slice(m + "shared_experts.shared_w2_weight")[s]
        g, u = deinterleave(gu, 0)
        return torch.cat([g, u], 0).to(torch.bfloat16), (dn.float() * self.expert_scale(li)).to(torch.bfloat16)

    def _fused_experts(self, li: int, which: str, ids=None):
        """Stack of experts in `ids` order (default all 258; -1 = an all-zero expert): which = gate_up_proj | down_proj."""
        m = f"model.llm.layers.{li}.mlp."
        if ids is None:
            if which == "gate_up_proj":
                g, u = deinterleave(self.ck.get(m + "experts.w13_weight"), 1)
                sg, su = deinterleave(self.ck.get(m + "shared_experts.shared_w13_weight"), 1)
                return torch.cat([torch.cat([g, u], 1), torch.cat([sg, su], 1)], 0)
            dn = torch.cat([self.ck.get(m + "experts.w2_weight"), self.ck.get(m + "shared_experts.shared_w2_weight")], 0)
            sc = self.expert_scale(li)
            for s0 in range(0, dn.shape[0], 32):  # in place, in blocks (the stack is 4 GB in bf16)
                dn[s0 : s0 + 32] = (dn[s0 : s0 + 32].float() * sc).to(dn.dtype)
            return dn
        k = 0 if which == "gate_up_proj" else 1
        real = [e for e in ids if e >= 0]
        first = self.one_expert(li, real[0])[k]
        out = torch.zeros((len(ids), *first.shape), dtype=torch.bfloat16)
        for slot, e in enumerate(ids):
            if e >= 0:
                out[slot] = first if e == real[0] else self.one_expert(li, e)[k]
        return out

    # --- model-view transforms (exact or bit-identical folds, applied once at load; the raw checkpoint stays available
    # through self.ck): the embedding table carries embed_norm (it depends on the token only: HF's op order, fp32 norm
    # then the bf16 weight multiply), the final norm carries the 1/logits_mup_width_multiplier (16, a power of two), the
    # dense MLPs' down projection carries their global_scale.
    def normed_embedding(self) -> torch.Tensor:
        e = self.ck.get("model.llm.embed.weight")
        w = self.ck.get("model.llm.embed_norm.weight").to(torch.bfloat16)
        eps = float(self.ck.cfg.rms_norm_eps)
        out = torch.empty_like(e, dtype=torch.bfloat16)
        for s in range(0, e.shape[0], 16384):
            x = e[s : s + 16384].float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            out[s : s + 16384] = w * x.to(torch.bfloat16)
        return out

    # --- Mapping
    def __getitem__(self, key: str):
        if key == "tok_embeddings.weight":
            return self.normed_embedding()
        if key == "norm.weight":
            return self.ck.get("model.llm.norm.weight") / float(self.ck.cfg.logits_mup_width_multiplier)
        if key in self._top:
            return self.ck.get(self._top[key])
        if not key.startswith("layers."):
            raise KeyError(key)
        _, li, sub = key.split(".", 2)
        li = int(li)
        p = f"model.llm.layers.{li}."
        simple = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "mlp_norm.weight",
                  "attn_sconv.weight": "attn_sconv.weight", "mlp_sconv.weight": "mlp_sconv.weight",
                  "mlp.gate_proj.weight": None, "mlp.up_proj.weight": None, "mlp.down_proj.weight": "mlp.w2_md.weight",
                  "mlp.global_scale": "mlp.global_scale", "mlp.gate.weight": "mlp.gate.weight",
                  "mlp.gate.e_score_correction_bias": "mlp.gate.bias", "mlp.gate.global_scale": "mlp.gate.global_scale"}
        if sub.startswith("self_attn."):
            inv = {v: k for k, v in _ATTN.items()}
            return self.ck.get(p + "attn." + inv[sub[len("self_attn."):]])
        if sub in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            g, u = deinterleave(self.ck.get(p + "mlp.w13_dn.weight"), 0)
            return g if sub == "mlp.gate_proj.weight" else u
        if sub == "mlp.down_proj.weight":  # dense MLP: x global_scale folded in
            gs = float(self.ck.get(p + "mlp.global_scale").float().reshape(-1)[0])
            return (self.ck.get(p + "mlp.w2_md.weight").float() * gs).to(torch.bfloat16)
        if sub in simple:
            return self.ck.get(p + simple[sub])
        if sub == "mlp.experts.gate_up_proj":
            return self._fused_experts(li, "gate_up_proj")
        if sub == "mlp.experts.down_proj":
            return self._fused_experts(li, "down_proj")
        raise KeyError(key)

    def __iter__(self):
        for ks in self._keys.values():
            yield from ks
        yield from self._top

    def __len__(self):
        return sum(len(v) for v in self._keys.values()) + len(self._top)

    def __contains__(self, key):
        return key in self._top or any(key in v for v in self._keys.values())

    @property
    def num_layers(self):
        return self.n_main_layers

    def layer_keys(self, li):
        return list(self._keys[li])

    def layer(self, li):
        from models.demos.blackhole.qwen36.utils.substate import LazyState

        return LazyState(self.__getitem__, self._keys[li])

    def has_mtp(self):
        return False  # the MTP heads live in mtp.safetensors (tt/mtp.py loads them)

    def visual_keys(self):
        return []

    def close(self):
        self.ck._files.clear()
        self._slices.clear()
