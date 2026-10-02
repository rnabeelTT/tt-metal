# Qwen3-Coder-Next on Blackhole (4 dies: p300x2 / p150x4, 2 dies: p300 / p150x2)

TTNN port of `Qwen/Qwen3-Coder-Next` (80B total / 3B active MoE: 512 experts, top-10 + a shared expert, 36 Gated
DeltaNet + 12 gated attention layers, text only) served through vLLM and the Tenstorrent plugin: chat, tool calls
(`qwen3_coder` parser), 262144-token context, up to 32 concurrent sequences, chunked prefill always on.

Ported from the Qwen3.5-122B-A10B autoport (`models/autoports/qwen_qwen3_5_122b_a10b`, same architecture family) as a
full copy; the changes are listed below. The `QWEN35_*` environment names are kept from that port.

## Serve it

```bash
tt-model serve raahemnabeel/qwen3-coder-next-blackhole --profile p300x2   # 4 dies (default); p150x4 / p300 / p150x2
tt-model curl "Write a Python function that reverses a linked list."
```

| profile | hardware | precision policy | single-user decode (token-out, 48 layers) |
|---|---|---|---|
| `p300x2` (default), `p150x4` | 4 dies, TP4 | bfp8 routed experts, bfp8 token mixers, bfp4 LM head | 22.54 ms/step (44.4 tok/s) |
| `p300`, `p150x2` | 2 dies, TP2 | bfp4 routed experts, bfp8 token mixers, bfp4 LM head | 24.74 ms/step (40.4 tok/s) |

The p150 profiles are untested on hardware (built for the identical meshes). The 2-die profiles pin
`TT_VISIBLE_DEVICES=0,1` (a no-op on a 2-chip host; on a bigger host they use the first board).

For reference, the shipped Qwen3.5-122B-A10B autoport measures 30.50 ms/step with the same tool on the same 4-die box.

## Accuracy (teacher forcing vs the HF bf16 reference, 100 tokens)

| mesh | coding chat prompt top1 / top5 | book p2048 top1 / top5 |
|---|---|---|
| 4 dies | 0.99 / 1.00 | 0.97 / 0.99 |
| 2 dies | 0.98 / 1.00 | 0.92 / 0.99 |

Precision selection: `doc/datatype_sweep/selected_precision_config.json` (4 dies) and
`selected_precision_config_tp2.json` (2 dies), each with the candidates it was chosen over. bfp8 experts do not fit two
32 GB dies (~41 GB each), so the 2-die profile keeps bfp4 experts.

## What changed vs the 122B port

- `tt/weights.py`: checkpoint prefix `model.`; Qwen3-Next fused GDN projections (`in_proj_qkvz`, `in_proj_ba`, grouped
  per key head, b before a) split into the Qwen3.5 layout; per-expert tensors fused (bit-exact vs HF).
- `tt/model_config.py`: shared-expert width overrides the flat config's `intermediate_size`; mesh helpers; 2-device
  meshes run Linear CCLs.
- `tt/multichip_decoder.py`, `tt/model.py`: the mesh follows the opened device ((1,4) or (1,2)); fused matmul +
  reduce-scatter out-projections are 4-device-only (that op always builds a ring); TP2 on-device sampling.
- `tt/generator_vllm.py`: text-only `Qwen3NextForCausalLM` (no MTP head or vision tower in this checkpoint), chunked
  prefill on by default, per-mesh KV budget (`doc/context_contract.json` `serving_tp4` / `serving_tp2`).
- Decode perf pass (27.39 -> 22.54 ms/step on 4 dies): router top-k through Blackhole's large-k route
  (`QWEN35_ROUTER_TOPK_ROUTE_K`, default 80), one-token decode on the gather path (`QWEN35_MOE_PAIRS_MIN_TOKENS`,
  default 2), down projection on 64 cores.

## Layout

`tt/` runtime, `tests/` device tests (`test_multichip_decoder.py` runs on either mesh: `MESH_DEVICE=P300x2` or
`MESH_DEVICE=P300 TT_VISIBLE_DEVICES=0,1`), `tools/` readiness / benchmarks, `doc/` precision evidence and the context
contract, `tt-model.yaml` the bundle manifest.
