# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-flash text-backbone configuration (tencent/Hy4-flash `text_config`), as the device port reads it.

Every field is taken from the checkpoint's config.json; nothing model-specific is hard coded here except the names of
the two attention kinds. See models/experimental/hybrid_moe/HANDOFF_HY4.md for the architecture notes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

HF_MODEL_ID = "tencent/Hy4-flash"


HF_REVISION = os.environ.get("HY4_CKPT_REVISION", "b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f")  # tt-model.yaml weights.revision


def checkpoint_dir(repo: str = HF_MODEL_ID) -> str:
    """The local snapshot directory of the checkpoint (HF cache, offline)."""
    env = os.environ.get("HY4_CKPT_DIR")
    if env:
        return env
    from huggingface_hub import snapshot_download

    # the pinned revision, never refs/main: a newer upstream revision in the HF cache (refs/main moved by a later
    # download or `tt-model serve --refresh`) must not change which snapshot the server loads
    return snapshot_download(repo, revision=HF_REVISION, local_files_only=True)


@dataclass
class Hy4Config:
    hidden: int
    vocab: int
    n_layers: int
    rms_eps: float
    # residual hyper-connections
    hc_mult: int
    hc_eps: float
    hc_magnitude: float
    # attention (per kind: "sparse" = indexed / DSA layers, "swa" = sliding)
    layer_kind: list  # per layer: "sparse" | "swa"
    heads: dict  # kind -> query heads
    q_lora: int
    kv_lora: int
    qk_nope: int
    qk_rope: int
    v_head: int
    rope_theta: dict  # kind -> theta
    window: int  # SWA window incl. the current token (512)
    # compressed indexer
    idx_heads: int
    idx_dim: int
    idx_ratio: int  # tokens per group
    idx_group_topk: int  # groups kept (incl. the query's own)
    idx_candidate_topk: int  # dense below this many keys
    # MLP / MoE
    dense_inter: int
    first_dense: int
    n_experts: int
    top_k: int
    moe_inter: int
    latent: int  # routed_expert_hidden_size
    n_shared: int
    routed_scale: float
    swiglu_limit: float
    # engram
    engram_layers: list  # 0-based decoder layer indices that add the engram residual before their attention HC
    engram_cfg: dict = field(default_factory=dict)
    # tokens
    bos: int = 120000
    eos: int = 120025
    pad: int = 120002
    raw: dict = field(default_factory=dict)

    @property
    def qk_head(self) -> int:
        return self.qk_nope + self.qk_rope

    @property
    def latent_width(self) -> int:
        """Width of one cached MLA latent row: normed c_kv then the roped shared key."""
        return self.kv_lora + self.qk_rope

    def is_moe(self, li: int) -> bool:
        return li >= self.first_dense

    @classmethod
    def load(cls, ckpt_dir: str | None = None) -> "Hy4Config":
        ckpt_dir = ckpt_dir or checkpoint_dir()
        full = json.load(open(os.path.join(ckpt_dir, "config.json")))
        r = full["text_config"]
        kinds = []
        for t in r["layer_types"]:
            if t in ("deepseek_sparse_attention", "indexed_attention"):
                kinds.append("sparse")
            elif t == "sliding_attention":
                kinds.append("swa")
            else:
                raise ValueError(f"unknown layer type {t}")
        rp = r["rope_parameters"]
        theta = {
            "sparse": float(rp.get("deepseek_sparse_attention", rp.get("indexed_attention"))["rope_theta"]),
            "swa": float(rp["sliding_attention"]["rope_theta"]),
        }
        assert r["qk_nope_head_dim"] == r["swa_qk_nope_head_dim"] and r["qk_rope_head_dim"] == r["swa_qk_rope_head_dim"]
        assert r["v_head_dim"] == r["swa_v_head_dim"]
        assert all(t in ("dense", "sparse") for t in r["mlp_layer_types"])
        first_dense = r["mlp_layer_types"].index("sparse")
        assert all(t == "sparse" for t in r["mlp_layer_types"][first_dense:])
        assert r.get("n_group", 1) == 1 and r.get("topk_group", 1) == 1
        assert r["moe_router_use_sigmoid"] and r["moe_router_enable_expert_bias"] and r["norm_topk_prob"]
        engram_cfg = {
            k: r[k]
            for k in (
                "engram_layer_head_vocab_sizes",
                "engram_max_ngram_size",
                "engram_n_embed_per_ngram",
                "engram_n_head_per_ngram",
                "engram_pad_id",
                "engram_seed",
                "engram_fp8_embedding",
            )
        }
        return cls(
            hidden=int(r["hidden_size"]),
            vocab=int(r["vocab_size"]),
            n_layers=int(r["num_hidden_layers"]),
            rms_eps=float(r["rms_norm_eps"]),
            hc_mult=int(r["hc_mult"]),
            hc_eps=float(r["hc_eps"]),
            hc_magnitude=float(r["hc_magnitude"]),
            layer_kind=kinds,
            heads={"sparse": int(r["num_attention_heads"]), "swa": int(r["swa_num_attention_heads"])},
            q_lora=int(r["q_lora_rank"]),
            kv_lora=int(r["kv_lora_rank"]),
            qk_nope=int(r["qk_nope_head_dim"]),
            qk_rope=int(r["qk_rope_head_dim"]),
            v_head=int(r["v_head_dim"]),
            rope_theta=theta,
            window=int(r["sliding_window"]),
            idx_heads=int(r["compressed_indexer_num_attention_heads"]),
            idx_dim=int(r["compressed_indexer_head_dim"]),
            idx_ratio=int(r["compressed_indexer_compress_ratio"]),
            idx_group_topk=int(r["compressed_indexer_group_topk"]),
            idx_candidate_topk=int(r["compressed_indexer_candidate_topk"]),
            dense_inter=int(r["intermediate_size"]),
            first_dense=first_dense,
            n_experts=int(r["n_routed_experts"]),
            top_k=int(r["num_experts_per_tok"]),
            moe_inter=int(r["moe_intermediate_size"]),
            latent=int(r["routed_expert_hidden_size"]),
            n_shared=int(r["n_shared_experts"]),
            routed_scale=float(r["routed_scaling_factor"]),
            swiglu_limit=float(r["swiglu_limit"]),
            engram_layers=[int(i) - 1 for i in r["engram_layer_ids"]],
            engram_cfg=engram_cfg,
            bos=int(r["bos_token_id"]),
            eos=int(r["eos_token_id"]),
            pad=int(r["pad_token_id"]),
            raw=r,
        )
