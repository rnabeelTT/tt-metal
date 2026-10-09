# GLM-4.5-Air on Blackhole

`zai-org/GLM-4.5-Air` (HF `glm4_moe`, `Glm4MoeForCausalLM`; 106B total / 12B active) served through vLLM on 4 Blackhole
dies (p300x2 / p150x4) or 2 dies (p300 / p150x2).

The port reuses the Qwen3.5-122B-A10B / Qwen3-Coder-Next / Ornith v2 autoport code (this directory started as a copy of
`models/autoports/ornith_ai_ornith_1_5_35b_a3b`): tensor-parallel attention, routed experts sharded on the intermediate
dimension (all 128 experts on every die), expert-sorted MoE prefill, traced prefill buckets, per-width decode traces,
on-device sampling with the lockstep multi-die sampler, MTP speculative decoding, the vLLM adapter and the plugin shims.

## What is GLM-specific

| GLM-4.5-Air | where |
| --- | --- |
| every one of the 46 layers is full attention (no Gated DeltaNet) | `tt/model_config.py` (`attention_type_list`); the GDN hooks of the shared generator run over an empty layer list |
| q/k/v projections carry biases; no attention output gate; no q/k norm | qwen36 `attention/tp.py` flags `attn_gated` / `attn_qk_norm` / `attn_qkv_bias` (Qwen defaults unchanged); the bias is one fused `[q_d|k_d|v_d]` add after the QKV matmul |
| plain RMSNorm weights (Qwen3.5 stores zero-centred `1 + w`) | `norm_unit_offset = False` (qwen36 `layer.py`, `model.py`, `tt/mtp.py`) |
| partial RoPE 64 of 128 dims, theta 1e6 | the qwen36 partial-RoPE path (rotate-half on the first `rope_head_dim` dims) |
| 2-die decode: 48 local q heads | `_use_nlp_decode_heads = NH <= 32`, slice/reshape head split above 32 |
| router: `sigmoid(logits) + e_score_correction_bias` selects the top 8 of 128 experts; weights = the selected sigmoid scores normalized | `tt/moe.py` `_route_sigmoid_bias`: fp32 logits / sigmoid / top-k; the bias (~16-20 with a spread of 0.02-0.05, unusable in bf16) is mean-centred per layer (selection is shift-invariant); decode rows select the weights by a one-hot compare, prefill rows by `ttnn.gather` |
| one shared expert (1408) without a gate | `shared_expert_gated = False` (`tt/moe.py`, `tt/fused_decoder.py`, `tt/optimized_decoder.py`) |
| layer 0 is a dense SwiGLU of 10944 | zero-padded to 11008 (exact) so it splits into tiles on 4 and 2 dies (`tt/weights.py` `pad_dense_mlp`) |
| per-expert checkpoint tensors | fused on load into the `[E, 2I, H]` / `[E, H, I]` layout (`tt/weights.py`) |
| MTP layer = checkpoint layer 46 (`eh_proj`, `enorm`, `hnorm`, `shared_head`) | mapped onto the Qwen3.5 MTP head scheme (`StreamingStateDict.mtp_state`); its embedding and head are bit-identical to the main ones |
| routed-expert width 352 per die (TP4) | sparse-matmul geometry: gate/up one output tile per core (22 / 44 cores), down projection K in one block |

## Meshes and precision

| profile | mesh | routed experts | attention / shared / LM head | serving | policy file |
| --- | --- | --- | --- | --- | --- |
| p300x2 / p150x4 | (1,4) | gate/up bfp4, down bfp8 | bfp8 / bfp8 / bfp8 | 32 seqs, 131072 context, KV pool 348,672 tokens, adaptive MTP | `doc/datatype_sweep/selected_precision_config.json` |
| p300 / p150x2 | (1,2) | bfp4 | bfp4 / bfp4 / bfp4 | 8 seqs, 32768 context, KV pool 36,672 tokens, no MTP | `doc/datatype_sweep/selected_precision_config_tp2.json` |

The 2-die build holds 28.7 GiB of weights per die (31.3 GiB DRAM), so its profile drops the MTP head, uses a 512 MiB trace
region and 1024-token prefill segments (`tt-model.yaml` profile env; budget in `doc/context_contract.json` `serving_tp2`).

Teacher forcing against HF bf16 (100 greedy tokens unless noted, top-1 / top-5 / top-100) and the single-user traced
decode step (token-out bench, plain decode, no MTP), final configuration:

| mesh | code chat | book p2048 | book p32768 (32 tokens) | decode step |
| --- | --- | --- | --- | --- |
| 4 dies | 0.99 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 0.875 / 1.00 / 1.00 | 26.9 ms (37 tok/s) |
| 2 dies | 0.99 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 | - | 31.4 ms (32 tok/s) |

4-die datatype sweep (before the decode tuning; the token-out step is identical for all three, decode is op-overhead
bound):

| routed experts | code chat | book p2048 | DRAM / die (B = 1, 131K KV) | token-out |
| --- | --- | --- | --- | --- |
| bfp4 | 0.98 / 1.00 / 1.00 | 0.96 / 1.00 / 1.00 | 18.3 GiB | 31.4 ms |
| gate/up bfp4 + down bfp8 (selected) | 0.99 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 | 22.2 GiB | 31.4 ms |
| bfp8 | 1.00 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 | 29.9 GiB (no KV room) | 31.4 ms |

MTP (checkpoint layer 46, k = 3, exact vs greedy): 3.28 tokens per pass on code (56 tok/s single user), 2.21 on prose
(38 tok/s), traced offline; served single-user rows use it automatically (adaptive: plain decode when several requests
are live).

## Tuning that changed defaults

- Decode 31.4 -> 26.9 ms per step (4 dies): the router chain in fp32 with a plain top-8 (no k = 80 route), a one-hot
  weight select for decode rows instead of a single-core `ttnn.gather`, and expert matmul geometry for the 352-wide
  per-die expert shard (gate/up one output tile per core, down K = 11 tiles in one block: the Qwen value collapsed it
  to 1 tile and the down projection ran 98 us instead of 28 us).
- Long prompts: prefill SDPA q / k chunks 256 and the streaming kernel (`QWEN35_SDPA_PREFILL_*`, defaults set in
  `tt/model_config.py`): served 130K-token TTFT 108 -> 84.5 s, 32K teacher forcing unchanged.

## Defects found and fixed

- Batched decode wrote garbage V rows for some users (7 of 16 at width 16): `TPAttention.forward_decode` freed the
  unpadded K / V before resharding the padded copies, and `ttnn.pad` returns its input when the rows fit the tile, so
  the K shard allocation reused V rows. Latent with Qwen's one local KV head; fixed in the shared qwen36 code.
- A lone request admitted after a concurrent batch could enter the MTP phase one position late (stale device row of the
  slot's previous occupant): fixed in `generator_vllm._authoritative`. The same code is in the Qwen3.5-122B, Ornith v2
  and Qwen3-Coder-Next adapters.

## Run

```bash
export HF_MODEL=zai-org/GLM-4.5-Air MESH_DEVICE=P300x2   # or P300 with TT_VISIBLE_DEVICES=0,1
python -m models.autoports.zai_org_glm_4_5_air.tools.readiness teacher-forcing --reference <ref.refpt>
python -m models.autoports.zai_org_glm_4_5_air.tools.tokenout_bench --layers all
pytest models/autoports/zai_org_glm_4_5_air/tests/test_multichip_decoder.py -k real_weight   # follows MESH_DEVICE
```

Serving: `tt-model.yaml` (one image, profiles p300x2 / p150x4 / p300 / p150x2); tool calls and reasoning use vLLM's
`glm45` parsers.
