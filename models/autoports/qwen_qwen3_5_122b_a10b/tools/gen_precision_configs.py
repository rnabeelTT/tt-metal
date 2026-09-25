# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Write the datatype-sweep candidate policies to doc/datatype_sweep/configs/<config_id>.json (stage 8).

Every candidate is the stage-7 baseline with one policy change (the coarse search of the datatype-sweep skill), so a
top-1 / top-5 change is attributable. All candidates fit the per-device DRAM budget (routed experts bfp4 15.2 GiB; the
bfp8-down candidate adds ~4.5 GiB and shrinks the KV pool)."""
from __future__ import annotations

import json

from models.autoports.qwen_qwen3_5_122b_a10b.tt import precision as P

OUT = P.MODEL_DIR / "doc" / "datatype_sweep" / "configs"

CANDIDATES = {
    "baseline_s7": dict(description="stage-7 shipped policy (refresh)"),
    # BFP4 fidelity comparison for the selected routed group (skill: every material BFP4 group needs BFP4+LoFi AND the
    # HiFi2 comparison)
    "routed_bfp4_hifi2": dict(compute_fidelities={"routed_moe": "hifi2"}, description="routed experts bfp4 with HiFi2"),
    # BFP8+LoFi vs BFP8+HiFi2 on the dominant decode projection groups (token mixers)
    "mixer_lofi": dict(
        compute_fidelities={"attention": "lofi", "linear_attention": "lofi"},
        description="GDN / attention bfp8 projections with LoFi (fp32 dest + packer L1 acc)",
    ),
    # dense (shared expert) FF policy
    "shared_bfp8_lofi": dict(compute_fidelities={"shared_moe": "lofi"}, description="shared expert bfp8 LoFi"),
    "shared_bfp4_lofi": dict(
        weight_groups={"shared_moe": {"dtype": "bfp4"}},
        compute_fidelities={"shared_moe": "lofi"},
        description="shared expert bfp4 LoFi",
    ),
    "shared_bfp4_hifi2": dict(weight_groups={"shared_moe": {"dtype": "bfp4"}}, description="shared expert bfp4 HiFi2"),
    # CCL payload
    "ccl_bfp8": dict(ccl_dtype="bfp8", description="decode out-projection partials in bfp8 into the reduce-scatter"),
    # canonical safe KV (context contract recompute) and accuracy-restore candidates
    "kv_bf16": dict(kv_cache_dtype="bf16", description="bf16 paged KV cache (canonical safe policy)"),
    "gdn_state_fp32": dict(linear_state_dtype="fp32", description="fp32 GDN recurrent state + fp32 decode step"),
    "routed_down_bfp8": dict(
        weight_groups={"routed_moe": {"gate_up": "bfp4", "down": "bfp8"}},
        description="routed expert down projections bfp8 (+4.5 GiB/device)",
    ),
    # lower-precision wins outside the MoE block
    "lm_head_bfp4": dict(
        weight_groups={"lm_head": {"dtype": "bfp4"}},
        description="LM head bfp4 (halves the 0.76B-parameter read per token)",
    ),
    "mixer_bfp4_lofi": dict(
        weight_groups={"attention": {"dtype": "bfp4"}, "linear_attention": {"dtype": "bfp4"}},
        compute_fidelities={"attention": "lofi", "linear_attention": "lofi"},
        description="GDN / attention projections bfp4 LoFi",
    ),
    # round 2 (after the single-change round): combinations of the surviving lower-precision wins
    "combo_mixer_lmhead_bfp4": dict(
        weight_groups={
            "attention": {"dtype": "bfp4"},
            "linear_attention": {"dtype": "bfp4"},
            "lm_head": {"dtype": "bfp4"},
        },
        compute_fidelities={"attention": "lofi", "linear_attention": "lofi"},
        description="round 2: GDN / attention projections bfp4 LoFi + LM head bfp4",
    ),
    "combo_mixer_lmhead_bfp4_ccl_bfp8": dict(
        weight_groups={
            "attention": {"dtype": "bfp4"},
            "linear_attention": {"dtype": "bfp4"},
            "lm_head": {"dtype": "bfp4"},
        },
        compute_fidelities={"attention": "lofi", "linear_attention": "lofi"},
        ccl_dtype="bfp8",
        description="round 2: mixers bfp4 LoFi + LM head bfp4 + bfp8 decode CCL payload",
    ),
    # round 3: the BFP4 fidelity pairs the skill asks for on every material bfp4 group that is selected
    "mixer_bfp4_hifi2": dict(
        weight_groups={"attention": {"dtype": "bfp4"}, "linear_attention": {"dtype": "bfp4"}},
        description="GDN / attention projections bfp4 HiFi2 (fidelity pair of mixer_bfp4_lofi)",
    ),
    "lm_head_bfp4_lofi": dict(
        weight_groups={"lm_head": {"dtype": "bfp4"}},
        compute_fidelities={"lm_head": "lofi"},
        description="LM head bfp4 with LoFi (fidelity pair of lm_head_bfp4 = ttnn default)",
    ),
    "combo_all_bfp4_lofi": dict(
        weight_groups={
            "attention": {"dtype": "bfp4"},
            "linear_attention": {"dtype": "bfp4"},
            "lm_head": {"dtype": "bfp4"},
        },
        compute_fidelities={"attention": "lofi", "linear_attention": "lofi", "lm_head": "lofi"},
        description="round 3: mixers bfp4 LoFi + LM head bfp4 LoFi",
    ),
    # canonical first/last-layer exception on the selected bfp4 routed group
    "routed_first_last_bfp8": dict(
        layer_exceptions={
            "0": {"routed_moe": {"gate_up": "bfp8", "down": "bfp8"}, "routed_moe_fidelity": "hifi2"},
            "47": {"routed_moe": {"gate_up": "bfp8", "down": "bfp8"}, "routed_moe_fidelity": "hifi2"},
        },
        description="routed experts bfp8 HiFi2 in layers 0 and 47, bfp4 LoFi elsewhere",
    ),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for cid, over in CANDIDATES.items():
        over = dict(over)
        desc = over.pop("description", "")
        pol = P.policy_from_overrides(cid, desc, **over)
        (OUT / f"{cid}.json").write_text(json.dumps(pol, indent=2) + "\n")
        print(cid, "->", OUT / f"{cid}.json")


if __name__ == "__main__":
    main()
