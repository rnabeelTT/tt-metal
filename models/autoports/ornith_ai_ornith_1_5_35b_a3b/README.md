# Ornith-1.5-35B-A3B v2 on Blackhole

`ornith-ai/Ornith-1.5-35B-A3B` served through vLLM on 4, 2 or 1 Blackhole dies. The checkpoint is a
`qwen3_5_moe` (`Qwen3_5MoeForConditionalGeneration`) model, the same architecture and checkpoint format as
Qwen3.5-122B-A10B, with Qwen3-Coder-Next's dimensions (hidden 2048, 16 q / 2 kv heads x 256, Gated DeltaNet 16 key /
32 value heads, expert and shared-expert width 512), 256 experts top-8, 40 layers (30 GDN + 10 full attention),
vocab 248320, an MTP head and a vision tower.

v2 replaces the v1 port (branch `ornith-1.5-35b-a3b`: one die replicated per vLLM lane, batch 1, host sampling,
a 64-token union-of-experts MoE prefill at ~400 tok/s) with the Qwen3.5-122B-A10B / Qwen3-Coder-Next autoport code:

- tensor-parallel GDN and gated attention, routed experts sharded on the intermediate dimension (all 256 experts on
  every die), on-device sampling with the lockstep multi-die sampler, per-width decode traces 1..32;
- expert-sorted MoE prefill (`ttnn.moe_sorted_dispatch`, group-parallel `ttnn.sparse_matmul` with resident weights,
  `ttnn.moe_gather_rows` / `ttnn.moe_gather_add_rows`), traced prefill buckets;
- MTP speculative decoding from the checkpoint's own `mtp.*` head when one request is live;
- the vision tower (`QWEN35_VISION=1`).

What differs from the Coder-Next copy: `tt/weights.py` is the 122B loader (qwen3_5_moe checkpoint keys, MTP, vision);
`tt/generator_vllm.py` is the 122B adapter (multimodal processor, M-RoPE, MTP) with Coder-Next's per-mesh KV budget;
`_KV_LAYERS = 10`; a one-die mesh (`MESH_DEVICE=P150`) runs the TP modules at degree 1 (`tp_modules_single_device`,
no fabric, 512-row prefill matmul slices, 1024-token prefill segments).

## Meshes and precision

| profile | mesh | routed experts | policy file |
| --- | --- | --- | --- |
| p300x2 / p150x4 | (1,4) | bfp8 | `doc/datatype_sweep/selected_precision_config.json` |
| p300 / p150x2 | (1,2) | bfp8 | `doc/datatype_sweep/selected_precision_config_tp2.json` |
| p150 | (1,1) | bfp4 | `doc/datatype_sweep/selected_precision_config_tp1.json` |

Teacher forcing against HF bf16 (100 tokens, top-1 / top-5 / top-100) and the single-user token-out step:

| mesh | code chat | book p2048 | step |
| --- | --- | --- | --- |
| 4 dies | 0.97 / 1.00 / 1.00 | 0.89 / 0.98 / 1.00 | 18.7 ms |
| 2 dies | 0.96 / 1.00 / 1.00 | 0.91 / 0.97 / 1.00 | 21.3 ms |
| 1 die | 0.96 / 1.00 / 1.00 | 0.84 / 0.94 / 0.99 | 28.6 ms |

bfp4 experts on 4 / 2 dies: book p2048 0.85 top-1. Per-mesh KV budgets: `doc/context_contract.json` (`serving_tp{4,2,1}`).

## Run

```bash
export HF_MODEL=ornith-ai/Ornith-1.5-35B-A3B MESH_DEVICE=P300x2   # or P300 (TT_VISIBLE_DEVICES=0,1) / P150
python -m models.autoports.ornith_ai_ornith_1_5_35b_a3b.tools.readiness teacher-forcing --reference <ref.refpt>
pytest models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_multichip_decoder.py   # follows MESH_DEVICE
```

Serving: `tt-model.yaml` (one image, profiles p300x2 / p150x4 / p300 / p150x2 / p150).
