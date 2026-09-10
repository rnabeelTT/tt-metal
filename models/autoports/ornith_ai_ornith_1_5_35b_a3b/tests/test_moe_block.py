# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC gate for the one component Ornith adds over the qwen36 Blackhole port.

Everything downstream — full model, multi-chip, long context, serving — assumes this
block is right, so it is checked here against the *actual* HF module
(`Qwen3_5MoeSparseMoeBlock`) rather than a reimplementation. A hand-written reference
would only prove the device code agrees with my reading of the architecture; using the
real module proves it agrees with the checkpoint's semantics.

Three specific ways this block can be wrong while still producing fluent-looking text,
each with its own assertion:

1. **gate/up swapped.** The checkpoint fuses them into one `gate_up_proj` tensor.
   Swapping the halves computes `silu(up) * gate` — different, but plausible.
2. **shared-expert gate dropped.** The block scales its shared expert by a learned
   per-token `sigmoid(x @ shared_expert_gate)`. Omitting it (or substituting the MoE
   config's fixed scalar scale) shifts every token slightly.
3. **expert-to-device map wrong.** On >1 device the routed experts are sharded; a
   wrong map routes tokens to the wrong weights, which still returns *something*.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import OrnithModelArgs
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.moe import OrnithMoE
from models.common.utility_functions import comp_pcc

LAYER = 0
PREFIX = f"model.language_model.layers.{LAYER}.mlp."

# Thresholds are set against the MEASURED ceiling, not a round number. Recomputing the
# reference with bf16 activations AND bf16 weights scores 0.99978 vs fp32, so ~0.9998 is
# the best any correct bf16 implementation can do here; 0.999 leaves room for arithmetic
# order without admitting real bugs. A loose bf16 bound is what let a genuine defect
# (scaling the expert INPUT by the routing weight rather than its output -- wrong,
# because SwiGLU is nonlinear) sit at 0.979 looking like precision noise.
# The quantized meshes get looser bounds because there the loss is the point.
PCC_BY_DTYPE = {"bfloat16": 0.999, "bfloat8_b": 0.99, "bfloat4_bfloat8": 0.975, "bfloat4_b": 0.96}


def _weights_dir() -> Path:
    d = os.environ.get("ORNITH_WEIGHTS") or os.environ.get("HF_MODEL")
    if not d or not Path(d, "config.json").is_file():
        pytest.skip("ORNITH_WEIGHTS is not pointing at a local checkpoint")
    return Path(d)


def _load_moe_state(weights: Path) -> dict[str, torch.Tensor]:
    """Pull just this layer's MoE tensors out of the shards — 72 GB is not loadable."""
    index = json.loads((weights / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [k for k in index if k.startswith(PREFIX)]
    assert keys, f"no MoE tensors under {PREFIX}"
    out: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for k in keys:
        by_shard.setdefault(index[k], []).append(k)
    for shard, ks in by_shard.items():
        with safe_open(weights / shard, framework="pt") as f:
            for k in ks:
                out[k[len(PREFIX) :]] = f.get_tensor(k)
    return out


def _reference_block(state: dict[str, torch.Tensor], weights: Path):
    """The real HF block, loaded with the real weights, in fp32 for a clean target."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock, Qwen3_5MoeTextConfig

    cfg = Qwen3_5MoeTextConfig(**json.loads((weights / "config.json").read_text())["text_config"])
    block = Qwen3_5MoeSparseMoeBlock(cfg).to(torch.float32).eval()
    sd = {
        "gate.weight": state["gate.weight"],
        "experts.gate_up_proj": state["experts.gate_up_proj"],
        "experts.down_proj": state["experts.down_proj"],
        "shared_expert.gate_proj.weight": state["shared_expert.gate_proj.weight"],
        "shared_expert.up_proj.weight": state["shared_expert.up_proj.weight"],
        "shared_expert.down_proj.weight": state["shared_expert.down_proj.weight"],
        "shared_expert_gate.weight": state["shared_expert_gate.weight"],
    }
    missing, unexpected = block.load_state_dict({k: v.to(torch.float32) for k, v in sd.items()}, strict=False)
    # strict=False only to tolerate buffer-shaped entries; a missing *parameter* means
    # the key names moved and the test would silently grade against random weights.
    real_missing = [m for m in missing if not m.endswith("_buffer")]
    assert not real_missing, f"reference block did not receive: {real_missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    return block


@pytest.fixture(scope="module")
def moe_fixture():
    weights = _weights_dir()
    state = _load_moe_state(weights)
    return weights, state, _reference_block(state, weights)


@pytest.mark.parametrize("seq_len", [1, 32, 128], ids=["decode", "prefill32", "prefill128"])
def test_moe_block_pcc(mesh_device, moe_fixture, seq_len):
    """Device MoE block vs the HF block, on the mesh selected by MESH_DEVICE."""
    weights, state, ref = moe_fixture
    args = OrnithModelArgs(mesh_device, max_batch_size=1, max_seq_len=max(seq_len, 128))

    torch.manual_seed(0)
    x = torch.randn(1, seq_len, args.dim, dtype=torch.float32)
    expected = ref(x).squeeze(0)

    moe = OrnithMoE(mesh_device, args, state, layer_id=LAYER)
    tt_x = ttnn.from_torch(
        x.reshape(1, 1, seq_len, args.dim).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    mode = "decode" if seq_len == 1 else "prefill"
    tt_out = moe.forward(tt_x, mode=mode)
    got = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    # Assert the width BEFORE reshaping. A reduce that scatters instead of gathering
    # returns hidden/num_devices, and `reshape(seq_len, dim)` would silently reinterpret
    # that narrow output as full width and still produce a plausible PCC.
    assert got.shape[-1] == args.dim, (
        f"MoE output is {got.shape[-1]} wide, expected {args.dim} -- the block is not "
        f"returning a full-width hidden state"
    )
    got = got.reshape(seq_len, args.dim).to(torch.float32)

    threshold = PCC_BY_DTYPE[args.routed_expert_dtype]
    passing, pcc = comp_pcc(expected, got, threshold)
    assert passing, (
        f"MoE PCC {pcc} < {threshold} on {args.device_name} "
        f"(routed={args.routed_expert_dtype}, mode={mode}, seq_len={seq_len})"
    )


def test_gate_up_halves_are_not_swapped(moe_fixture):
    """Host-side: prove the fused gate_up_proj split matches the reference's chunk().

    Cheap, no device, and it isolates the single most likely silent error in loading
    this checkpoint. If the halves were swapped, `test_moe_block_pcc` would also fail,
    but with a PCC number that looks like a precision problem rather than a layout one.
    """
    from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.moe import _split_fused_experts

    weights, state, _ = moe_fixture
    cfg = json.loads((weights / "config.json").read_text())["text_config"]
    inter, hidden = cfg["moe_intermediate_size"], cfg["hidden_size"]
    w_gate, w_up, _ = _split_fused_experts(state, cfg["num_experts"], hidden, inter)

    # The reference computes linear(x, gate_up_proj[e]).chunk(2, -1): gate is the first
    # `inter` output rows. Our split must reproduce exactly that, transposed.
    e = 7
    ref_gate, ref_up = state["experts.gate_up_proj"][e].chunk(2, dim=0)
    torch.testing.assert_close(w_gate[e], ref_gate.to(torch.bfloat16).T.contiguous())
    torch.testing.assert_close(w_up[e], ref_up.to(torch.bfloat16).T.contiguous())


def test_shared_expert_gate_is_applied(moe_fixture):
    """The learned per-token sigmoid gate must actually change the answer.

    Guards against a regression where the shared expert gets folded back into
    TTMoEDecode's fixed `shared_expert_scale`. If dropping the gate made no difference
    the assertion below would fail, telling us the test itself has stopped discriminating.
    """
    weights, state, ref = moe_fixture
    torch.manual_seed(0)
    x = torch.randn(1, 16, ref.shared_expert_gate.in_features, dtype=torch.float32)

    with torch.no_grad():
        with_gate = ref(x)
        saved = ref.shared_expert_gate.weight.detach().clone()
        try:
            # sigmoid(0) = 0.5 everywhere: a *constant* scale, i.e. exactly the
            # fixed-scalar approximation we are refusing to use.
            ref.shared_expert_gate.weight.zero_()
            without_gate = ref(x)
        finally:
            ref.shared_expert_gate.weight.copy_(saved)

    _, pcc = comp_pcc(with_gate, without_gate, 0.0)
    assert pcc < 0.9999, (
        "replacing the learned shared-expert gate with a constant changed nothing; "
        "this test can no longer detect a dropped gate"
    )
