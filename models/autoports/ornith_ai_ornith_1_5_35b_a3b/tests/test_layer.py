# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-decoder-layer PCC against the HF reference layer.

Covers BOTH hybrid layer types deliberately. `layer_types` alternates 3 Gated DeltaNet
layers then 1 full-attention layer, so layer 0 is DeltaNet and layer 3 is the first full
attention. A DeltaNet-only pass says nothing about the 10 full-attention layers, and vice
versa -- they share only the norm/residual pattern and the MoE block.

This is the first test where attention, the recurrent state, RoPE and the MoE all run
together, so it is where an interface mistake between them shows up.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.layer import OrnithDecoderLayer
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import OrnithModelArgs
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.weight_mapping import remap_ornith_state_dict
from models.common.utility_functions import comp_pcc

# Looser than the isolated MoE block (0.999): a whole layer stacks attention or the GDN
# recurrence, two norms and two residual adds on top of the expert block, each in bf16.
PCC_BY_DTYPE = {"bfloat16": 0.99, "bfloat8_b": 0.98, "bfloat4_bfloat8": 0.97, "bfloat4_b": 0.95}


def _weights_dir() -> Path:
    d = os.environ.get("ORNITH_WEIGHTS") or os.environ.get("HF_MODEL")
    if not d or not Path(d, "config.json").is_file():
        pytest.skip("ORNITH_WEIGHTS is not pointing at a local checkpoint")
    return Path(d)


def _load_layer(weights: Path, layer: int) -> dict[str, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer}."
    index = json.loads((weights / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k in [k for k in index if k.startswith(prefix)]:
        by_shard.setdefault(index[k], []).append(k)
    out: dict[str, torch.Tensor] = {}
    for shard, ks in by_shard.items():
        with safe_open(weights / shard, framework="pt") as f:
            for k in ks:
                # The layer module wants keys rooted at "layers.<n>." -- that is what
                # substate() slices on inside the decoder layer.
                out[k[len("model.language_model.") :]] = f.get_tensor(k)
    return out


@pytest.mark.parametrize("layer_num", [0, 3], ids=["layer0_deltanet", "layer3_fullattn"])
# 128 and 256, not 32: the GDN chunk-prefill kernel supports exactly chunk_size=128
# (see Qwen36DecoderLayer.forward's default), and a shorter sequence takes a path that
# slices the fused conv state against a dim it does not have. 256 additionally checks
# that a multi-chunk prefill carries the recurrent state across chunk boundaries -- the
# thing most likely to be wrong once one chunk works.
@pytest.mark.parametrize("seq_len", [128, 256], ids=["prefill128", "prefill256"])
def test_layer_pcc(mesh_device, layer_num, seq_len):
    weights = _weights_dir()
    args = OrnithModelArgs(mesh_device, max_batch_size=1, max_seq_len=max(seq_len, 128))
    cfg = json.loads((weights / "config.json").read_text())["text_config"]

    expected_type = cfg["layer_types"][layer_num]
    is_full = args.is_full_attention_layer(layer_num)
    assert is_full == (expected_type == "full_attention"), (
        f"layer {layer_num}: checkpoint says {expected_type} but the port thinks "
        f"is_full_attention={is_full}; the hybrid dispatch is misaligned"
    )

    raw = _load_layer(weights, layer_num)
    # The device layer wants the port's DeltaNet names; the HF reference wants the
    # checkpoint's. Keep both, and build each from the form it expects.
    if is_full and seq_len > 128:
        pytest.skip(
            "full-attention prefill past one page needs a page_table, which the model "
            "builds and a bare layer does not; covered by test_model.py"
        )
    state = remap_ornith_state_dict(raw, cfg)
    layer = OrnithDecoderLayer(mesh_device, args, state, layer_num)

    torch.manual_seed(0)
    # [B, T, dim] -- the 3D convention the decoder layer and GDN use (see qwen36's own
    # unit tests); a 4D [1, 1, T, dim] input makes the GDN slice its channel dim out of
    # the sequence dim.
    x = torch.randn(1, seq_len, args.dim, dtype=torch.float32)
    # The two hybrid layer types name their state reset differently: DeltaNet carries a
    # recurrent + conv state (reset_state), full attention a paged KV cache (reset_cache).
    if hasattr(layer.attention, "reset_state"):
        layer.attention.reset_state(1)
    else:
        layer.attention.reset_cache()

    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextConfig

    hf_cfg = Qwen3_5MoeTextConfig(**cfg)
    ref = Qwen3_5MoeDecoderLayer(hf_cfg, layer_idx=layer_num).to(torch.float32).eval()
    ref_sd = {k[len(f"layers.{layer_num}.") :]: v.to(torch.float32) for k, v in raw.items()}
    missing, unexpected = ref.load_state_dict(ref_sd, strict=False)
    real_missing = [m for m in missing if "rotary" not in m]
    assert not real_missing, f"reference layer did not receive: {real_missing}"

    # The reference layer takes rope cos/sin as an argument rather than owning a rotary
    # module. Build them from the same config the device path reads. The DeltaNet layers
    # ignore them, but the signature is positional so they are required either way.
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

    rotary = Qwen3_5MoeTextRotaryEmbedding(hf_cfg)
    pos = torch.arange(seq_len).unsqueeze(0)
    if getattr(hf_cfg, "rope_parameters", {}).get("mrope_section"):
        # m-RoPE wants a (T, H, W) position triple. Text-only input has no spatial
        # extent, so all three sections carry the same linear position -- which is what
        # collapses m-RoPE back to ordinary RoPE for text.
        pos = pos.unsqueeze(0).expand(3, -1, -1)
    with torch.no_grad():
        cos, sin = rotary(x, pos)
        out = ref(x, position_embeddings=(cos, sin))
    expected = out[0] if isinstance(out, tuple) else out
    expected = expected.squeeze(0) if expected.dim() == 3 else expected

    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    # Full-attention layers need the RoPE tables; DeltaNet layers ignore them but the
    # arguments are harmless, so build them unconditionally rather than branching.
    from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup

    rope = Qwen36RoPESetup(mesh_device, args)
    tt_cos, tt_sin = rope.get_rot_mats(torch.arange(seq_len).unsqueeze(0))

    # 128 is the only chunk size the GDN chunk-seq prefill kernel supports.
    tt_out = layer.forward(tt_x, cos=tt_cos, sin=tt_sin, mode="prefill", chunk_size=128)
    got = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    got = got.reshape(-1, seq_len, args.dim)[0].to(torch.float32)

    threshold = PCC_BY_DTYPE[args.routed_expert_dtype]
    passing, pcc = comp_pcc(expected, got, threshold)
    assert passing, (
        f"layer {layer_num} ({expected_type}) PCC {pcc} < {threshold} on " f"{args.device_name}, seq_len={seq_len}"
    )
