# GLM-5.3-Flash (E192) optimized decoder

Stage 3 of the tt-model-bringup flow. `tt/optimized_decoder.py` (`OptimizedDecoder`) is the fused decoder
(`tt/fused_decoder.py`) with a precision policy and decode program configurations for one die. Contract, state,
paging and semantics are those of `tt/functional_decoder.py`. The policies are `OptConfig` entries in `CANDIDATES`;
`SELECTED` (`opt`) is the default.

## Selected policy (`opt`)

| Tensor group | Fused decoder (stage 2) | Optimized (`opt`) | Evidence |
|---|---|---|---|
| Routed experts gate / up / down | bfp8, HiFi4 | **bfp4, LoFi** | E192 at bfp8 does not fit 4 dies (`tools/memory_budget.py`: 26.6 GiB per die at bfp4). LoFi and HiFi2 give bit-identical results on bfp4 weights. |
| Shared expert | bfp8, HiFi4 | bfp8, HiFi4 | HiFi2 left the shared output 0.8 % low (activation-mantissa truncation, the review #2 finding). |
| KDA input / output projections | bf16 | **bfp8** | weight-read bound (414 µs -> about 400 µs at bfp8 is the DRAM limit; see Rejected) |
| DSA W_UK / W_UV (absorbed per-head) | bf16, ttnn default config | **bfp8, full-grid reuse config (decode rows)** | `tools/dsa_bmm_probe.py`: 222 -> 48 µs and 381 -> 45 µs |
| KDA / DSA output projections (decode) | ttnn default config | **1D config** (`moe_opt.DenseDecodeMatmul`) | `tools/dense_decode_probe.py`: 147 -> 100 µs (KDA 8192 x 4096) |
| mHC, norms, KDA recurrence, router | unchanged | unchanged | fp32 streams and HiFi4 norms are stage-1 accuracy fixes |

## Accuracy of the bfp4 experts

The routed experts must be bfp4 (memory). `tools/moe_scale_probe.py --layer 4 --rows 128 --policy <candidate>` separates
the quantization from the port:

| Comparison (layer 4 routed output, 128 real-stream rows) | PCC | RMS ratio |
|---|---|---|
| bfp4-rounded weights (torch) vs checkpoint weights (torch) | 0.98258 | 1.0275 |
| TT bfp4 LoFi vs bfp4-rounded weights (torch) | **0.99997** | 0.9951 |
| TT bfp4 HiFi2 vs bfp4-rounded weights (torch) | 0.99997 | 0.9951 (bit-identical to LoFi) |
| TT bfp4 vs checkpoint weights (torch) | 0.98255 | 1.0225 |

The port reproduces the quantized model; the layer-level loss is the bfp4 quantization itself. The optimized suite
(`tests/test_optimized_decoder.py`) holds the MoE layers (3, 4) to that floor: update PCC >= 0.98, sequence scale
1 +- 0.03, token scale 1 +- 0.06. Layer 0 (dense MLP, no bfp4) keeps the functional bars. Model-level quality of the bfp4
policy is measured in the full-model stage (readiness top-k, qualitative check) and the datatype sweep.

## Performance (one die, warmed; Tracy + tt-perf-report)

Device time of one pass (decode: one traced replay), µs. Candidate runs: `doc/optimized_decoder/candidates/<name>/`.

| Case | fused (stage 2) | kda8 | kda8_exp4_lofi | kda8_exp4_hifi2 | kda8_exp4gu8dn | kda8_exp4_lofi_pc |
|---|---|---|---|---|---|---|
| decode kda_dense, 1 user | 2403 | 2088 | 2089 | 2089 | 2087 | 2047 |
| decode kda_moe, 1 user | 2937 | 2627 | 2292 | 2309 | 2395 | 2254 |
| decode kda_moe, 32 users | 16588 | 16274 | 12390 | 12941 | 13424 | 12212 |
| decode dsa_moe, 1 user | 3328 | 3322 | 2998 | 3004 | 3095 | 2389 |
| prefill kda_moe, 2048 rows | 79477 | 80013 | 61464 | 62496 | 66441 | 61744 |

Final default (`opt`, `doc/optimized_decoder/tracy/`, run 2026-10-17 on the final fp32-exact kernels and the final
stage-2 rewrites). A decode report holds 3 replays, so the table gives the report sum divided by 3:

| Case | 1 user | 32 users |
|---|---|---|
| decode kda_dense | 2082 | 2901 |
| decode kda_moe | 2287 | 11530 |
| decode dsa_moe | 2421 | 11779 |
| prefill 2048 rows: kda_dense / kda_moe / dsa_moe | 29966 / 66865 / 97938 | |

The prefill MoE rows include the stage-6 memory split of the routed experts (`GLM53_MOE_SORTED_ROWS=1024`, about
+5 ms per MoE layer at 2048 rows; stage 7 reviews it for TTFT). Against the fused decoder (stage 2, same kernels):
decode kda_moe at 32 users is 28 % faster (15946 -> 11530 µs) and prefill kda_dense is equal (29911 -> 29966 µs; the
dense layer has no bfp4 experts). The candidate table above predates the fp32-exact kernels; its ranking holds
because the kernel change adds the same cost to every candidate.

## Topology audit (one die)

| Op sequence (decode, per layer) | Material repeated work | Candidate | Constraint | Action |
|---|---|---|---|---|
| KDA: in-proj (q,k,v,decay rank,gate,beta in one matmul) -> conv -> recurrent_decode -> o_norm, gate -> out-proj | none: one packed projection | bfp4 in-proj (265 µs vs 400 µs in the probe) | accuracy of 34 layers' projections untested | datatype sweep (stage 8) |
| KDA out-proj 8192 x 4096 | default config at 243 GB/s | 1D (358 GB/s), DRAM-sharded (LoFi 75 µs) | DRAM-sharded weights need an interleaved copy for prefill (2x footprint) | 1D taken; DRAM-sharded rejected |
| DSA: q_a -> norm -> q_b -> per-head W_UK -> sparse attention -> per-head W_UV -> o_proj | per-head matmuls on 8-16 cores | full-grid reuse config, bfp8 | prefill at 2048 rows hung on the reuse config | decode rows only |
| MoE: router -> sorted dispatch -> sparse gate/up -> SwiGLU -> sparse down -> combine; shared expert | sparse matmuls read every routed expert at 32 users | bfp4 experts | accuracy floor above | taken |
| mHC pre/post per sub-block | sinkhorn on 1 core (82 µs per call) | multi-core sinkhorn | rows = users; 20 sequential iterations | not taken (stage 5 / 7 candidate) |

## Rejected

| Candidate | Result | Reason |
|---|---|---|
| KDA in-proj 1D / DRAM-sharded configs | 1D no faster (400 µs at 360 GB/s, already DRAM bound); DRAM-sharded fails (L1 / CB size at N = 32960, 129 tiles per bank) | program-config limit |
| bfp4 gate/up with bfp8 down (`kda8_exp4gu8dn`) | 13424 µs at 32 users vs 12390 | slower; bfp8 down is not needed for accuracy (TT matches the bfp4 reference) |
| HiFi2 routed experts (`kda8_exp4_hifi2`) | bit-identical output, 4 % slower at 32 users | no accuracy gain on bfp4 weights |
| DSA reuse config in prefill | device hang at 2048 rows | decode rows only |
| DRAM-sharded output projections | 75 µs vs 100 µs (1D) | prefill would need a second interleaved weight copy |

## Correctness

`tests/test_optimized_decoder.py` (the functional suite on `OptimizedDecoder`, policy `opt`), final kernels and fused
flags: **43 passed** (`s3_opt_suite3.log`, 2026-10-17, final stage-1 test harness with the y_swap near-tie check). The lowest update PCC is on the bfp4 MoE layers: synthetic
kda_moe prefill 0.97331 and real-weight kda_moe decode 0.99049. Both are above the bfp4 bars (update PCC bar
0.98 - 0.015 = 0.965 on synthetic, 0.98 on real; see "Accuracy of the bfp4 experts").

## Commands

```
GLM53_DECODER=optimized [GLM53_OPT=<candidate>] TRACY_OUT=<dir> bash ~/glm53-flash-bringup/scripts/01-tracy.sh
pytest models/autoports/zai_org_glm_5_3_flash/tests/test_optimized_decoder.py
python -m models.autoports.zai_org_glm_5_3_flash.tools.moe_scale_probe --layer 4 --rows 128 --policy opt
```
