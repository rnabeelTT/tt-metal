# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-die DRAM budget of the full model on a (1, tp) mesh: weights by tensor group under a dtype policy, KDA state per
user slot and DSA latent / pool-key cache per token, and the context that fits in what remains.

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.memory_budget --tp 4 [--json out.json]
       python -m models.autoports.zai_org_glm_5_3_flash.tools.memory_budget --measured REDUCED.json CAPACITY.json --json out.json
         (stage 6: the serving budget from the device probes of tools/memory_probe.py)"""

import argparse
import json
from pathlib import Path

DIE_BYTES = 8 * 4278190080  # Blackhole: 8 DRAM channels x 4 GiB (minus firmware carve-out)
BYTES = {
    "bf16": 2.0,
    "bfp8": 1088 / 1024,
    "bfp4": 576 / 1024,
    "fp32": 4.0,
}  # block-float: 16 B exponents per 1024 B tile


def budget(tc: dict, tp: int, policy: dict, reserve_gib: float) -> dict:
    H, L, V = tc["hidden_size"], tc["num_hidden_layers"], tc["vocab_size"]
    kinds = tc["layer_types"]
    n_kda = sum(k == "linear_attention" for k in kinds)
    n_dsa = L - n_kda
    mlp = tc["mlp_layer_types"]
    n_dense = sum(m == "dense" for m in mlp)
    n_moe = L - n_dense
    E, I, Ish = (
        tc["n_routed_experts"],
        tc["moe_intermediate_size"],
        tc["moe_intermediate_size"] * tc["n_shared_experts"],
    )
    NH, D = tc["linear_num_heads"], tc["linear_head_dim"]
    kda_in = H * (3 * NH * D + 128 + NH * D + NH)  # q|k|v|decay rank|gate|beta
    kda_out = NH * D * H
    QA, R, DN, DV, AH = (
        tc["q_lora_rank"],
        tc["kv_lora_rank"],
        tc["qk_nope_head_dim"],
        tc["v_head_dim"],
        tc["num_attention_heads"],
    )
    IH, ID = tc["index_n_heads"], tc["index_head_dim"]
    g = {}  # group -> (params, dtype, sharded over tp)
    g["routed experts"] = (n_moe * E * 3 * H * I, policy["experts"], True)
    g["shared experts"] = (n_moe * 3 * H * Ish, policy["shared"], True)
    g["dense MLP (layers 0-2)"] = (n_dense * 3 * H * tc["intermediate_size"], policy["dense"], True)
    g["KDA in/out projections"] = (n_kda * (kda_in + kda_out), policy["kda"], True)
    g["DSA q_b / W_UK / W_UV / o_proj"] = (
        n_dsa * (QA * AH * DN + AH * DN * R + AH * R * DV + AH * DV * H),
        policy["dsa"],
        True,
    )
    g["DSA q_a / kv_a (replicated)"] = (n_dsa * (H * QA + H * R), policy["dsa"], False)
    g["DSA indexer (replicated)"] = (n_dsa * (QA * IH * ID + 2 * H * ID + H * IH), "bf16", False)
    g["router gates (replicated)"] = (n_moe * H * E, "bf16", False)
    g["embedding (host lookup / device)"] = (V * H, policy["embed"], True)
    g["LM head"] = (V * H, policy["lm_head"], True)
    rows = {k: p * BYTES[dt] / (tp if sh else 1) for k, (p, dt, sh) in g.items()}
    w = sum(rows.values())
    free = DIE_BYTES - w - reserve_gib * 2**30
    kda_state_slot = n_kda * (NH // tp) * D * D * 4  # fp32 [H/tp, K, V] per layer per slot
    conv_slot = n_kda * 3 * (3 * NH * D // tp) * 2
    # DSA per token per layer: latent R bf16 (replicated) + pool key ID bf16 per 4 tokens
    dsa_tok = n_dsa * (R * 2 + ID * 2 / 4)
    out = {
        "tp": tp,
        "policy": policy,
        "die_bytes": DIE_BYTES,
        "weights_bytes_per_die": {k: round(v) for k, v in rows.items()},
        "weights_total_bytes_per_die": round(w),
        "reserve_bytes": round(reserve_gib * 2**30),
        "free_bytes_per_die": round(free),
        "kda_state_bytes_per_slot_per_die": kda_state_slot + conv_slot,
        "dsa_cache_bytes_per_token_per_die": dsa_tok,
    }
    for slots in (1, 4, 8, 16, 32):
        out[f"max_tokens_total_{slots}_slots"] = int((free - slots * (kda_state_slot + conv_slot)) // dsa_tok)
    return out


SLOT_PROFILES = (1, 4, 8, 16, 32)


def measured_budget(red: dict, cap: dict, trace_region: int, reserve: int) -> dict:
    """Serving DRAM budget per die from the device probes (tools/memory_probe.py): the 1-slot full-model capacity run
    (weights + model-level tensors + 1 slot measured after load; the pool and workspace measured after bind / warm /
    prefill) and the reduced run (per-slot KDA state, trace bytes per layer kind, prefill peaks).

    pool_tokens(S) = (A - trace_region - base(1) - (S - 1) * per_slot - workspace - reserve) / dsa_bytes_per_token,
    floored to whole 128-token blocks, minus vLLM's null block and the model's dummy block."""
    MiB = 1 << 20
    A = cap["die_allocatable_bytes"] + cap["trace_region_bytes"]  # DRAM per die without a trace region
    base1 = cap["after_load"]["allocated"]  # weights + model-level + 1 KDA slot + layer constants (defer mode)
    lc = {v["kind"]: v for v in red["layers_construct"].values()}
    S_red = red["S"]
    kda_layers = 34
    rec_slot_layer = 128 * 128 * 4 * 16  # fp32 [16 heads, 128, 128] per slot per KDA layer per die
    conv_layer = lc["kda+moe"]["attn_state_bytes"] - S_red * rec_slot_layer  # 3 conv rows [32-row tile, 6144] bf16
    per_slot = kda_layers * rec_slot_layer
    per_block = cap["dsa_bytes_per_block_all_layers"]
    per_token = per_block / 128
    # workspace: everything allocated after the pool (DSA prefill constants at bind, decode inputs, warm-up
    # persistents) + the measured prefill peak over the post-warm state + the decode-step workspace
    post_bind_persist = cap["after_capture"]["allocated"] - cap["after_bind"]["allocated"]
    # bind_kv_cache allocates the pools (num_blocks + the dummy block) and each DSA layer's prefill constants / decode
    # inputs: the non-pool part is workspace
    bind_extra = cap["dsa_pool_bytes_measured"] - (cap["num_blocks"] + 1) * cap["dsa_bytes_per_block_all_layers"]
    pf_peak = max(
        [r["peak_bytes"] for r in cap.get("prefill_tracked_rows", [])] + [cap.get("warm_and_capture_peak_bytes", 0)]
    )  # long-prompt chunks (first two, last two) and the warm-up prefills (128 / 2048 / 2050 / 4224 rows)
    pf_over = max(0, pf_peak - cap["after_capture"]["allocated"])
    dec_ws = red["decode_step_eager"]["peak_over_entry_bytes"]
    workspace = bind_extra + post_bind_persist + max(pf_over, dec_ws)
    rows = {}
    for S in SLOT_PROFILES:
        free = A - trace_region - base1 - (S - 1) * per_slot - workspace - reserve
        blocks = int(free // per_block)
        tokens = (blocks - 2) * 128 if S == 1 else (blocks - 1 - S) * 128  # vLLM null block (+1 per user), dummy block
        rows[S] = {
            "pool_tokens": max(0, tokens),
            "max_model_len": max(0, min(tokens, 1048576) // 2048 * 2048),
            "kda_state_bytes": S * per_slot + kda_layers * conv_layer,
        }
    return {
        "source": {"reduced": red.get("_path"), "capacity": cap.get("_path")},
        "die_dram_bytes": A,
        "trace_region_bytes": trace_region,
        "reserve_bytes": reserve,
        "weights_and_model_level_1slot_bytes": base1,
        "kda_state_per_slot_bytes": per_slot,
        "kda_conv_state_fixed_bytes": kda_layers * conv_layer,
        "dsa_bytes_per_token": per_token,
        "workspace_bytes": workspace,
        "workspace_parts": {
            "dsa_constants_and_decode_inputs_at_bind": bind_extra,
            "warm_and_capture_persistent": post_bind_persist,
            "prefill_peak_over_steady_state": pf_over,
            "decode_step_workspace": dec_ws,
        },
        "profiles": rows,
        "MiB": {
            k: round(v / MiB, 1)
            for k, v in (("trace_region", trace_region), ("workspace", workspace), ("per_slot", per_slot))
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--config", default=str(Path.home() / "glm53-flash-bringup/artifacts/e192_config.json"))
    ap.add_argument("--experts", default="bfp4")
    ap.add_argument("--reserve-gib", type=float, default=1.0, help="trace region, activations, CCL buffers, firmware")
    ap.add_argument("--json", default=None)
    ap.add_argument("--measured", nargs=2, metavar=("REDUCED", "CAPACITY"), default=None)
    ap.add_argument("--trace-region-mib", type=float, default=96.0)
    ap.add_argument("--reserve-mib", type=float, default=128.0)
    a = ap.parse_args()
    if a.measured:
        red, cap = (json.loads(Path(p).read_text()) for p in a.measured)
        red["_path"], cap["_path"] = a.measured
        r = measured_budget(red, cap, int(a.trace_region_mib * 2**20), int(a.reserve_mib * 2**20))
        for S, row in r["profiles"].items():
            print(f"{S:2d} slots: pool {row['pool_tokens']:,} tokens, max_model_len {row['max_model_len']:,}")
        if a.json:
            Path(a.json).write_text(json.dumps(r, indent=1))
        return
    c = json.loads(Path(a.config).read_text())
    tc = c.get("text_config", c)
    pol = {
        "experts": a.experts,
        "shared": "bfp8",
        "dense": "bfp8",
        "kda": "bfp8",
        "dsa": "bfp8",
        "embed": "bf16",
        "lm_head": "bfp8",
    }
    r = budget(tc, a.tp, pol, a.reserve_gib)
    gib = lambda b: f"{b / 2**30:7.3f} GiB"
    for k, v in r["weights_bytes_per_die"].items():
        print(f"{k:36s} {gib(v)}")
    print(
        f"{'weights / die':36s} {gib(r['weights_total_bytes_per_die'])} of {gib(DIE_BYTES)}; free after reserve {gib(r['free_bytes_per_die'])}"
    )
    print(
        f"KDA state / slot / die {r['kda_state_bytes_per_slot_per_die'] / 2**20:.1f} MiB; DSA cache / token / die {r['dsa_cache_bytes_per_token_per_die']:.0f} B"
    )
    for s in (1, 4, 8, 16, 32):
        print(f"{s:2d} slots: {r[f'max_tokens_total_{s}_slots']:,} tokens in total")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
