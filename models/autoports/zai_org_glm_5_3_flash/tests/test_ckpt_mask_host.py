# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The bundle reads the 288-expert PARENT snapshot through tt/e192_mask.json (CheckpointReader.mask). Dense layers
(0-2) and the MTP layer have no mask entry; listing their state must not index the mask (stage-11 cold verify of the
bundle: KeyError 0 at the first layer_state). Host only: a tiny index, no tensors are read."""

import json

from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import LAYER_PREFIX, MTP_LAYER, CheckpointReader


def _parent(tmp_path, mask):
    names = [LAYER_PREFIX.format(0) + "mlp.gate_proj.weight", LAYER_PREFIX.format(0) + "self_attn.q.weight"]
    for e in range(288):  # parent: every routed expert of MoE layer 3 + MTP experts
        for p in ("gate_proj", "up_proj", "down_proj"):
            names.append(LAYER_PREFIX.format(3) + f"mlp.experts.{e}.{p}.weight")
            names.append(LAYER_PREFIX.format(MTP_LAYER) + f"mlp.experts.{e}.{p}.weight")
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"n_routed_experts": 288}}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "x.safetensors" for n in names}})
    )
    (tmp_path / "e192_mask.json").write_text(json.dumps({"mask": {"3": mask}}))
    return CheckpointReader(tmp_path)


def test_dense_and_mtp_layers_do_not_touch_the_mask(tmp_path):
    mask = list(range(5, 197))  # 192 kept parent experts
    rd = _parent(tmp_path, mask)
    dense = rd.layer_state(0, lazy=True)
    assert not [k for k in dense if k.startswith("mlp.experts.")]
    moe = rd.layer_state(3, lazy=True)
    assert len([k for k in moe if k.startswith("mlp.experts.")]) == 192 * 3
    assert rd._expert_name(3, 0, "gate_proj.weight").endswith("mlp.experts.5.gate_proj.weight")
    assert rd._expert_name(MTP_LAYER, 7, "gate_proj.weight").endswith("mlp.experts.7.gate_proj.weight")
