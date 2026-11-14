# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Streaming state dict for the hybrid prototype.

Extends the autoport's ``StreamingStateDict`` with:
* FP8 block dequantization: any ``X.weight`` with a ``X.weight_scale_inv`` companion is returned as bf16
  (``w * scale_inv`` over 128x128 blocks, the Qwen3.5 FP8 checkpoint format).
* per-expert checkpoints (``mlp.experts.N.{gate,up,down}_proj.weight``, the 397B FP8 layout) and fused ones
  (``mlp.experts.gate_up_proj`` / ``down_proj``, the 122B bf16 layout) behind one interface.
* ``layer(i)`` returns the routed experts fused for the RESIDENT experts of that layer only (slot order), so the device
  upload is ``[C, ...]``; ``expert(i, e)`` returns one expert's (gate_up [2N, H], down [H, N]) bf16 for the host.
"""

from __future__ import annotations

import re

import torch

from models.autoports.qwen_qwen3_5_122b_a10b.tt.weights import StreamingStateDict

_EXPERT_RE = re.compile(r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
BLOCK = 128


def dequant_fp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """FP8 [O, I] with block scale_inv [ceil(O/128), ceil(I/128)] -> bf16."""
    O, I = w.shape
    sf = s.float().repeat_interleave(BLOCK, 0)[:O].repeat_interleave(BLOCK, 1)[:, :I]
    return (w.float() * sf).to(torch.bfloat16)


class HybridStateDict(StreamingStateDict):
    def __init__(self, checkpoint_dir, *, linear_q_dim, linear_k_dim, resident_fn):
        super().__init__(checkpoint_dir, linear_q_dim=linear_q_dim, linear_k_dim=linear_k_dim)
        self.resident_fn = resident_fn  # layer -> list of expert ids in slot order
        self._scale = {}  # weight hf_key -> (shard, scale hf_key)
        self._experts = {}  # (layer, e, proj) -> hf_key
        self._fused = {}  # layer -> {"gate_up": key, "down": key} (122B layout)
        for key in list(self._entries):
            e = self._entries[key]
            if e.hf_key.endswith(".weight_scale_inv"):
                del self._entries[key]
                self._drop_layer_key(key)
                continue
            m = _EXPERT_RE.match(key)
            if m:
                del self._entries[key]
                self._drop_layer_key(key)
                self._experts[(int(m.group(1)), int(m.group(2)), m.group(3))] = key
                self._expert_entry = getattr(self, "_expert_entry", {})
                self._expert_entry[key] = e
                continue
            if key.endswith("mlp.experts.gate_up_proj") or key.endswith("mlp.experts.down_proj"):
                li = int(key.split(".")[1])
                self._fused.setdefault(li, {})["gate_up" if key.endswith("gate_up_proj") else "down"] = key
        # scale companions straight from the index
        import json

        with open(self.checkpoint_dir / "model.safetensors.index.json") as f:
            wm = json.load(f)["weight_map"]
        for hk, shard in wm.items():
            if hk.endswith(".weight_scale_inv"):
                self._scale[hk[: -len("_scale_inv")]] = (shard, hk)
        self.fp8 = bool(self._scale)
        self.per_expert = bool(self._experts)
        expert_layers = {li for (li, _e, _p) in self._experts}
        for li in list(self._layer_keys):
            if li in expert_layers:
                self._layer_keys[li] += [f"layers.{li}.mlp.experts.gate_up_proj", f"layers.{li}.mlp.experts.down_proj"]
        self._down_stash = {}

    def mtp_state(self, fuse_experts: bool = True):
        """StreamingStateDict.mtp_state with the FP8 checkpoint dequantized (weight_scale_inv blocks) and the scale
        tensors dropped; per-expert MTP experts fused as [E, 2I, H] (gate rows first) / [E, H, I]."""
        if not self._mtp:
            raise KeyError("checkpoint has no mtp.* tensors")
        out, per_expert = {}, {}
        for hf_key, shard in self._mtp.items():
            if hf_key.endswith(".weight_scale_inv"):
                continue
            p = hf_key.split(".")
            if len(p) >= 7 and p[1] == "layers" and p[3] == "mlp" and p[4] == "experts" and p[5].isdigit():
                per_expert[(int(p[2]), int(p[5]), p[6])] = (shard, hf_key)
                if fuse_experts:
                    continue
            out[hf_key] = self._read(shard, hf_key)
        if fuse_experts and per_expert:
            for li in sorted({li for li, _, _ in per_expert}):
                experts = sorted({e for l2, e, _ in per_expert if l2 == li})
                if experts != list(range(len(experts))):
                    raise KeyError(f"mtp layer {li}: expert ids not contiguous from 0 ({experts[:4]}..{experts[-4:]})")
                # filled in place: a per-expert list then torch.stack peaked at twice the fused stack (~26 GB for 512)
                gu = dn = None
                for e in experts:
                    g, u, d = (self._read(*per_expert[(li, e, q)]) for q in ("gate_proj", "up_proj", "down_proj"))
                    if gu is None:
                        gu = torch.empty((len(experts), 2 * g.shape[0], g.shape[1]), dtype=g.dtype)
                        dn = torch.empty((len(experts), *d.shape), dtype=d.dtype)
                    gu[e, : g.shape[0]] = g
                    gu[e, g.shape[0] :] = u
                    dn[e] = d
                out[f"mtp.layers.{li}.mlp.experts.gate_up_proj"] = gu
                out[f"mtp.layers.{li}.mlp.experts.down_proj"] = dn
        return out

    def _drop_layer_key(self, key):
        parts = key.split(".")
        if parts[0] == "layers":
            lk = self._layer_keys.get(int(parts[1]))
            if lk and key in lk:
                lk.remove(key)

    def _read(self, shard, hf_key):
        t = self._shard(shard).get_tensor(hf_key)
        sc = self._scale.get(hf_key)
        if sc is not None:
            t = dequant_fp8(t, self._shard(sc[0]).get_tensor(sc[1]))
        return t

    def __getitem__(self, key):
        if key.endswith("mlp.experts.gate_up_proj") or key.endswith("mlp.experts.down_proj"):
            li = int(key.split(".")[1])
            return self._resident_fused(li, "gate_up" if key.endswith("gate_up_proj") else "down")
        entry = self._entries[key]
        t = self._read(entry.shard, entry.hf_key)
        return entry.transform(t) if entry.transform is not None else t

    # -- experts -------------------------------------------------------------------------------------------------
    def expert(self, li: int, e: int):
        """(gate_up [2N, H] = [gate; up], down [H, N]) bf16 for one expert."""
        if self.per_expert:

            def get(p):
                key = self._experts[(li, e, p)]
                ent = self._expert_entry[key]
                return self._read(ent.shard, ent.hf_key).to(torch.bfloat16)

            return torch.cat([get("gate_proj"), get("up_proj")], 0), get("down_proj")
        fk = self._fused[li]
        g = self._entries[fk["gate_up"]]
        d = self._entries[fk["down"]]
        gu = self._shard(g.shard).get_slice(g.hf_key)[e].to(torch.bfloat16)
        dn = self._shard(d.shard).get_slice(d.hf_key)[e].to(torch.bfloat16)
        return gu, dn

    def _resident_fused(self, li, which):
        """Resident experts stacked in slot order; id -1 is an all-zero expert (the fused hybrid's sink slot)."""
        ids = list(self.resident_fn(li))
        if not self.per_expert:
            ent = self._entries[self._fused[li]["gate_up" if which == "gate_up" else "down"]]
            sl = self._shard(ent.shard).get_slice(ent.hf_key)
            real = [sl[e] for e in ids if e >= 0]
            return torch.stack([sl[e] if e >= 0 else torch.zeros_like(real[0]) for e in ids], 0).to(torch.bfloat16)
        if which == "down" and li in self._down_stash:
            return self._down_stash.pop(li)
        gus, dns = [], []
        for e in ids:
            if e < 0:
                gus.append(None)
                dns.append(None)
                continue
            gu, dn = self.expert(li, e)
            gus.append(gu)
            dns.append(dn)
        z_gu = next(g for g in gus if g is not None)
        z_dn = next(d for d in dns if d is not None)
        gus = [torch.zeros_like(z_gu) if g is None else g for g in gus]
        dns = [torch.zeros_like(z_dn) if d is None else d for d in dns]
        gu_t, dn_t = torch.stack(gus, 0), torch.stack(dns, 0)
        if which == "gate_up":
            self._down_stash[li] = dn_t
            return gu_t
        return dn_t


def hybrid_fused_state_dict(port, checkpoint_dir, *, resident_fn):
    """The port's own ``StreamingStateDict`` (it already fuses per-expert checkpoints and dequantizes FP8), with the
    fused routed experts of each layer restricted to that layer's RESIDENT experts in slot order (id -1: an all-zero
    expert, the fused hybrid's sink slot), and ``expert(li, e)`` for host compute without an int4 library."""
    import importlib

    base = importlib.import_module(f"models.autoports.{port}.tt.weights").StreamingStateDict

    class _Hybrid(base):
        def _fused_experts(self, layer_idx, which):
            ids = list(resident_fn(layer_idx))
            if hasattr(base, "one_expert"):  # the port reads single experts itself (fused / interleaved checkpoints)
                return base._fused_experts(self, layer_idx, which, ids=ids)
            per = self._experts[layer_idx]
            real = [e for e in ids if e >= 0]
            if which == "gate_up_proj":
                one = lambda e: torch.cat(
                    [self._hf(per[(e, "gate_proj")]), self._hf(per[(e, "up_proj")])], 0
                )  # noqa: E731
            else:
                one = lambda e: self._hf(per[(e, "down_proj")])  # noqa: E731
            first = one(real[0])
            out = torch.zeros((len(ids), *first.shape), dtype=first.dtype)
            for slot, e in enumerate(ids):
                if e >= 0:
                    out[slot] = first if e == real[0] else one(e)
            return out

        def expert(self, li, e):
            if hasattr(base, "one_expert"):
                return base.one_expert(self, li, e)
            per = self._experts[li]
            gu = torch.cat([self._hf(per[(e, "gate_proj")]), self._hf(per[(e, "up_proj")])], 0).to(torch.bfloat16)
            return gu, self._hf(per[(e, "down_proj")]).to(torch.bfloat16)

    return _Hybrid(checkpoint_dir)
