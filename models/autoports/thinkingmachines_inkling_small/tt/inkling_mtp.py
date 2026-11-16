# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-Small's native MTP drafter (mtp.safetensors: 8 chained depths) on TT.

Depth d (vLLM's InklingMTPDepthLayer; models/experimental/hybrid_moe/mtp_inkling_ref.py is the CPU reference):
    in    = [hidden_norm_d(prev) | embed_norm_d(normed_embedding(token))]        (2H, replicated)
    x     = in @ input_proj_d^T                                                 (H, fractured: input_proj column-sharded)
    out   = block_d(x)          an Inkling decoder layer: attention (sliding / global per the MTP's local_layer_ids, no
                                tau), the 4 short convs, dense MLP (I 16384); its own KV and conv states
    draft = argmax(all_gather(out) / 16 @ unembed^T)                            (no final norm: chain_hidden_post_norm off)
    prev_0 = the target's final_norm(h) (the model's normed hidden before the LM head), prev_{d+1} = all_gather(out_d)

Here: the depths as InklingDecoderLayers built from an MTP view of the state (mtp_state), one-row decode, the draft
token, and the hidden for the next depth. The spec-decode loop that drives them (draft k, verify k+1 rows on the target,
accept, commit conv states) is tools/spec_bench.py.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch
import ttnn
from safetensors import safe_open

from models.autoports.thinkingmachines_inkling_small.tt.layer import InklingDecoderLayer
from models.autoports.thinkingmachines_inkling_small.tt.weights import deinterleave
from models.demos.blackhole.qwen36.utils.substate import LazyState


class _MtpArgs:
    """The model args with the MTP's layer layout: depth d is 'layer' d, sliding iff d in mtp local_layer_ids, dense MLP."""

    def __init__(self, args, mtp_cfg):
        self.__dict__["_a"] = args
        r = dict(args.inkling)
        r["local_layer_ids"] = list(mtp_cfg["local_layer_ids"])
        self.__dict__["inkling"] = r
        self.__dict__["attention_type_list"] = ["full_attention"] * int(mtp_cfg["num_nextn_predict_layers"])

    def __getattr__(self, k):
        return getattr(self.__dict__["_a"], k)

    def __setattr__(self, k, v):
        setattr(self.__dict__["_a"], k, v)

    def is_moe_layer(self, layer_idx):
        return False

    def is_sliding_layer(self, layer_idx):
        return layer_idx in self.inkling["local_layer_ids"]


def mtp_state(snap: str, n_depth: int):
    """A LazyState over mtp.safetensors in the port's internal layer keys (layers.<d>.*), plus the input parts."""
    f = safe_open(os.path.join(snap, "mtp.safetensors"), "pt")
    keys = set(f.keys())

    def get(k):
        _, d, sub = k.split(".", 2)
        p = f"model.mtp.layers.{d}.transformer_block."
        attn = {
            "q_proj.weight": "wq_du.weight",
            "k_proj.weight": "wk_dv.weight",
            "v_proj.weight": "wv_dv.weight",
            "r_proj.weight": "wr_du.weight",
            "o_proj.weight": "wo_ud.weight",
            "q_norm.weight": "q_norm.weight",
            "k_norm.weight": "k_norm.weight",
            "k_sconv.conv1d.weight": "k_sconv.weight",
            "v_sconv.conv1d.weight": "v_sconv.weight",
            "rel_logits_proj.proj": "rel_logits_proj.proj",
        }
        if sub.startswith("self_attn."):
            return f.get_tensor(p + "attn." + attn[sub[len("self_attn.") :]])
        simple = {
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "mlp_norm.weight",
            "attn_sconv.weight": "attn_sconv.weight",
            "mlp_sconv.weight": "mlp_sconv.weight",
        }
        if sub in simple:
            return f.get_tensor(p + simple[sub])
        if sub in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            g, u = deinterleave(f.get_tensor(p + "mlp.w13_dn.weight"), 0)
            return g if sub == "mlp.gate_proj.weight" else u
        if sub == "mlp.down_proj.weight":  # x global_scale (as the backbone's dense layers)
            gs = float(f.get_tensor(p + "mlp.global_scale").float().reshape(-1)[0])
            return (f.get_tensor(p + "mlp.w2_md.weight").float() * gs).to(torch.bfloat16)
        raise KeyError(k)

    names = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "attn_sconv.weight",
        "mlp_sconv.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ] + [
        f"self_attn.{n}"
        for n in (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "r_proj.weight",
            "o_proj.weight",
            "q_norm.weight",
            "k_norm.weight",
            "k_sconv.conv1d.weight",
            "v_sconv.conv1d.weight",
            "rel_logits_proj.proj",
        )
    ]
    state = LazyState(get, [f"layers.{d}.{n}" for d in range(n_depth) for n in names])
    parts = [
        {
            "hidden_norm": f.get_tensor(f"model.mtp.layers.{d}.hidden_norm.weight"),
            "embed_norm": f.get_tensor(f"model.mtp.layers.{d}.embed_norm.weight"),
            "input_proj": f.get_tensor(f"model.mtp.layers.{d}.input_proj.weight"),
        }
        for d in range(n_depth)
    ]
    assert all(f"model.mtp.layers.{d}.input_proj.weight" in keys for d in range(n_depth))
    return state, parts


class InklingMtp:
    """n_depth MTP depths on the mesh, sharing the target model's embedding table (normed) and LM head."""

    def __init__(self, model, n_depth: int):
        args = model.args
        self.model, self.mesh = model, model.mesh_device
        self.N = self.mesh.get_num_devices()
        raw = json.load(open(os.path.join(args.CKPT_DIR, "config.json")))["mtp_config"]
        self.n_depth = min(n_depth, int(raw["num_nextn_predict_layers"]))
        self.args = _MtpArgs(args, raw)
        state, parts = mtp_state(args.CKPT_DIR, self.n_depth)
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        # the drafter's own CCL semaphores: a traced target pass bakes in the target's double-buffered global
        # semaphores; eager drafter collectives cycling those same semaphores between replays hung the next replay
        from models.tt_transformers.tt.ccl import TT_CCL

        self.ccl = TT_CCL(self.mesh) if self.N > 1 else model.tt_ccl
        self.depths = []
        for d in range(self.n_depth):
            blk = InklingDecoderLayer(self.mesh, self.args, state, d, None, tt_ccl=self.ccl)
            p = parts[d]
            nw = lambda w: ttnn.as_tensor(
                w.reshape(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,  # noqa
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            w_in = ttnn.as_tensor(
                p["input_proj"].to(torch.bfloat16).T.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=-1),
            )  # [2H, H] col-parallel
            # the input projection as two halves (hidden part, embedding part): x = hn(prev) @ W_a + en(e) @ W_b instead of
            # a concat to 2H (the concat alone was ~0.7 ms of a ~5.6 ms depth on the device profile, tools/draft_prof.py)
            wt = p["input_proj"].to(torch.bfloat16).T.contiguous()  # [2H, H]
            Hh = wt.shape[0] // 2
            half = lambda t: ttnn.as_tensor(
                t.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,  # noqa
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=-1),
            )
            self.depths.append(
                SimpleNamespace(
                    block=blk,
                    hn=nw(p["hidden_norm"]),
                    en=nw(p["embed_norm"]),
                    w_in=w_in,
                    w_a=half(wt[:Hh]),
                    w_b=half(wt[Hh:]),
                )
            )
        self.eps = float(args.inkling["rms_norm_eps"])
        self.inv_mup = 1.0 / float(args.logits_mup_width_multiplier)

    def reset(self):
        for dp in self.depths:
            dp.block.reset()

    def _gather(self, x):
        from models.tt_transformers.tt.ccl import tt_all_gather

        return tt_all_gather(x, self.mesh, self.ccl, cluster_axis=None, dim=3, topology=self.model.args.ccl_topology())

    def own_ccl(self):
        """Context: the target model's shared helpers (embedding, LM head) run on the drafter's CCL semaphores."""
        import contextlib

        @contextlib.contextmanager
        def ctx():
            prev = self.model.tt_ccl
            self.model.tt_ccl = self.ccl
            try:
                yield
            finally:
                self.model.tt_ccl = prev

        return ctx()

    def _norm(self, x, w):
        return ttnn.multiply(ttnn.rms_norm(x, epsilon=self.eps), w)

    def inproj(self, dp, h, e):
        """The depth's input projection of [hidden_norm(h) | embed_norm(e)] -> [1, 1, R, H/N] fractured.
        INKLING_MTP_SPLIT=0: the concat + one matmul (the original form)."""
        if os.environ.get("INKLING_MTP_SPLIT", "1") != "1":
            return ttnn.linear(ttnn.concat([self._norm(h, dp.hn), self._norm(e, dp.en)], dim=-1), dp.w_in)
        grid = ttnn.CoreGrid(y=8, x=8)
        xa = ttnn.linear(self._norm(h, dp.hn), dp.w_a, core_grid=grid)
        xb = ttnn.linear(self._norm(e, dp.en), dp.w_b, core_grid=grid)
        x = ttnn.add(xa, xb)
        ttnn.deallocate(xa)
        ttnn.deallocate(xb)
        return x

    def step(self, d, prev_hidden, token_tt, cur_pos, page_table):
        """Depth d on one row: prev_hidden [1, 1, 1, H] replicated, token_tt [1, 1] uint32 (the token whose embedding
        feeds this depth), cur_pos [1] int32 (the position this depth writes). -> (logits [1, 1, 1, V] replicated,
        out_hidden [1, 1, 1, H] replicated: prev for depth d + 1)."""
        dp = self.depths[d]
        e = self.model.embd(token_tt)  # the normed embedding (the state dict folds the backbone embed_norm in)
        e = ttnn.reshape(e, (1, 1, 1, e.shape[-1]))
        if self.N > 1:  # the framework embedding is hidden-sharded: gather to the full row for the depth's norm
            e = self._gather(e)
        x = self.inproj(dp, prev_hidden, e)  # [1, 1, 1, H/N] fractured
        out = dp.block.forward(x, mode="decode", position_tensor=cur_pos, page_table=page_table)
        full = self._gather(out) if self.N > 1 else out
        logits = self.model._lm_head(ttnn.multiply(full, self.inv_mup))
        return logits, full
