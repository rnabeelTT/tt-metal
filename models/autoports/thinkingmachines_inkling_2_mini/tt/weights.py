# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini checkpoint access (original tensor names, no transformers dependency).

The official checkpoint stores the routed experts quantized: NVFP4 in layers 0-22 (two E2M1 codes per byte, an E4M3
scale per 16 inputs and a per-expert factor amax / outer_scale_denominator), MXFP8 in layer 23 (E4M3 values, an E8M0
scale per 32 inputs). ``experts()`` dequantizes them exactly as transformers' ``Inkling2LoopDequantizeExperts`` does
(to fp32, cast by the caller); w13's rows are gate/up interleaved (row 2i = gate_i, 2i + 1 = up_i).

Every other tensor is BF16/F32 and read as stored.
"""

from __future__ import annotations

import glob
import json
import os
from types import SimpleNamespace

import torch
from safetensors import safe_open

REPO = "thinkingmachines/Inkling-2-Mini"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def snapshot_dir(repo: str = REPO) -> str:
    env = os.environ.get("INKLING2_CKPT")
    if env:
        return env
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", "/opt/tenstorrent/models"), "hub")
    snaps = sorted(glob.glob(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(f"no snapshot of {repo} under {hub}")
    return snaps[-1]


def load_config(snap: str | None = None) -> SimpleNamespace:
    snap = snap or snapshot_dir()
    with open(os.path.join(snap, "config.json")) as f:
        cfg = json.load(f)
    t = dict(cfg["text_config"])
    loop = t.pop("loop_config")
    c = SimpleNamespace(**t)
    c.loop_start, c.loop_end, c.loop_passes, c.loop_gate_rank = (
        loop["core_start"],
        loop["core_end"],
        loop["num_passes"],
        loop["gate_rank"],
    )
    c.global_layers = [i for i in range(c.num_hidden_layers) if i not in set(c.local_layer_ids)]
    c.mtp = cfg.get("mtp_config")
    c.eos_token_id = cfg.get("eos_token_id", 200006)
    c.snap = snap
    return c


def schedule(c) -> list[tuple[int, int]]:
    """The 48 layer applications in order: (layer index, pass). Pass 0 for the pre/post-loop layers."""
    s = [(li, 0) for li in range(c.loop_start)]
    for p in range(c.loop_passes):
        s += [(li, p) for li in range(c.loop_start, c.loop_end)]
    s += [(li, 0) for li in range(c.loop_end, c.num_hidden_layers)]
    return s


class Checkpoint:
    def __init__(self, snap: str | None = None):
        self.snap = snap or snapshot_dir()
        self.f = safe_open(os.path.join(self.snap, "model-00001-of-00001.safetensors"), "pt")
        self.keys = set(self.f.keys())

    def get(self, key: str) -> torch.Tensor:
        return self.f.get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self.keys

    def L(self, li: int, name: str) -> torch.Tensor:
        return self.get(f"model.llm.layers.{li}.{name}")

    def experts(self, li: int, which: str, experts: slice | None = None) -> torch.Tensor:
        """which: 'w13' -> [E, 2I, H] with rows [gate (I) | up (I)] (de-interleaved); 'w2' -> [E, H, I]. fp32."""
        p = f"model.llm.layers.{li}.mlp.experts.{which}_weight"
        sl = experts or slice(None)
        w = self.f.get_slice(p)[sl]
        sc = self.f.get_slice(p + "_scale")[sl].float()
        if self.has(p + "_amax"):
            # transformers casts every source to the param dtype (bf16) before dequantizing; the golden reference is
            # built that way, so the per-expert factor is formed from bf16-rounded amax / denominator here too
            amax = self.get(p + "_amax")[sl].to(torch.bfloat16).float()
            den = self.get(p + "_outer_scale_denominator").to(torch.bfloat16).float()
            codes = w.to(torch.uint8).long()
            byte = torch.arange(256)
            lut = torch.stack((E2M1[byte & 0xF], E2M1[byte >> 4]), dim=-1)  # [256, 2]
            vals = lut[codes].view(*sc.shape, 16) * sc[..., None]
            out = (vals * (amax / den)[:, None, None, None]).reshape(sc.shape[0], sc.shape[1], -1)
        else:  # MXFP8
            vals = w.float().view(*sc.shape, 32) * torch.exp2(sc - 127)[..., None]
            out = vals.reshape(sc.shape[0], sc.shape[1], -1)
        if which == "w13":  # rows interleaved g0 u0 g1 u1 ... -> [g | u]
            E, R2, H = out.shape
            out = out.reshape(E, R2 // 2, 2, H).transpose(1, 2).reshape(E, R2, H)
        return out

    def message_queries(self) -> torch.Tensor:
        """[n_loop_layers, 4, H] fp32: the (query, key, value, mlp) message queries of each loop layer."""
        q = self.get("model.llm.middle_cycle_cross_pass_queries").float().unflatten(0, (-1, 2))
        kv = self.get("model.llm.middle_cycle_cross_pass_kv_queries").float().unflatten(0, (-1, 2))
        return torch.cat([q[:, :1], kv, q[:, 1:]], dim=1)

    def attn_query(self, li: int) -> torch.Tensor | None:
        """The block-attention-residual queries of layer li's attention input: [nq, H] fp32 (nq = 3 on global layers,
        1 on sliding), None for layer 0. The checkpoint's attn_queries.i belongs to layer i + 1."""
        if li == 0:
            return None
        q = self.get(f"model.llm.attn_res.attn_queries.{li - 1}").float()
        return q.reshape(-1, q.shape[-1])

    def mlp_query(self, li: int) -> torch.Tensor:
        return self.get(f"model.llm.attn_res.mlp_queries.{li}").float().reshape(1, -1)

    def final_query(self) -> torch.Tensor:
        return self.get("model.llm.attn_res.output_query").float().reshape(1, -1)
