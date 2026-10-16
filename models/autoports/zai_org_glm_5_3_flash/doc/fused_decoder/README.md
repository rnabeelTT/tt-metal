# GLM-5.3-Flash (E192) fused decoder

Stage 2 of the tt-model-bringup flow. `tt/fused_decoder.py` (`FusedDecoder`) is the functional decoder
(`tt/functional_decoder.py`) with its op graph rewritten into fewer, dedicated ops. Contract, state, paging and
semantics are unchanged (same `from_state_dict` / `prefill_inputs` / `prefill_forward` / `commit_prefill` /
`set_page_table` / `decode_inputs` / `decode_forward`); `FusedFlags` turns each rewrite on or off for before / after
measurements. One die (mesh 1x1), real E192 weights.

## Rewrites

| # | Subgraph (functional decoder) | Replacement | Kind |
|---|---|---|---|
| 1 | KDA decode: q / k / v slices, two reshape + L2-norm pairs, 12-op fp32 recurrence over `[B, 64, 128, 128]` | `ttnn.experimental.kda.recurrent_decode` (new op, written for this port from `ttnn.transformer.fused_recurrent_gated_delta_rule` of origin/main): per-key-channel decay, in-kernel L2 norms, head rows read straight from the conv output, state updated in place (trace-safe) | dedicated op |
| 2 | mHC pre-collapse: n stream slices + n coefficient slices + n multiply / addcmul | `ttnn.experimental.deepseek_prefill.mhc_pre` (new op), bf16 output feeds the RMSNorm (typecast folded) | dedicated op |
| 3 | mHC mixes: single-core `[T, 16384] x [16384, 32]` matmul + single-core mean of squares | `ttnn.experimental.deepseek_prefill.mhc_mix` (new op): split-K partials of the projection and of the per-row-tile Gram matrix (diagonal = row sum of squares), one (row tile, K chunk) unit per core (16-64 chunks), then sum / diagonal / rsqrt | dedicated op |

Op unit tests (`tests/test_kda_recurrent_decode.py`, `tests/test_mhc_pre.py`, `tests/test_mhc_mix.py`): output /
state PCC >= 0.9999993 (recurrent_decode, B = 1 / 8 / 32, chained steps, traced replay), >= 0.9999985 (mhc_pre),
mixes PCC 0.9999998 and row sum of squares within 1.6e-3 relative (mhc_mix: the Gram matmul reads fp32 at TF32
precision; the resulting ~8e-4 on the rsqrt scale is below the HF stream's own bf16 rounding).

## Performance (one die, warmed; Tracy + tt-perf-report)

`scripts/s2_tracy.sh`: `scripts/01-tracy.sh` over the 12 perf cases with `GLM53_DECODER=functional` and `=fused`
(tables / CSVs: `doc/functional_decoder/tracy/<case>/`, `doc/fused_decoder/tracy/<case>/`). Device time = sum of the
op device times of one pass (decode: one traced replay). Host wall under the profiler is not used for decisions.

| case | functional ops / device us | fused ops / device us | speed-up |
|---|---|---|---|
| traced decode kda_dense, 1 user | 130 / 3510 | 82 / 2403 | 1.46x |
| traced decode kda_dense, 32 users | 128 / 16492 | 78 / 3384 | 4.87x |
| traced decode kda_moe, 1 user | 166 / 4041 | 118 / 2937 | 1.38x |
| traced decode kda_moe, 32 users | 172 / 29760 | 122 / 16588 | 1.79x |
| traced decode dsa_moe, 1 user | 181 / 3900 | 159 / 3328 | 1.17x |
| traced decode dsa_moe, 32 users | 187 / 17143 | 163 / 16560 | 1.04x |
| prefill kda_dense 128 / 2048 rows | 80 / 4043, 43349 | 56 / 3212, 38650 | 1.26x / 1.12x |
| prefill kda_moe 128 / 2048 rows | 118 / 21294, 84292 | 94 / 20482, 79477 | 1.04x / 1.06x |
| prefill dsa_moe 128 / 2048 rows | 148 / 21987, 130998 | 124 / 21259, 126370 | 1.03x / 1.04x |

What remains (fused, top op groups): at 32 users the routed-expert sparse matmuls (7.4 + 3.7 ms per MoE layer) read
every expert (bfp8, ~3.4 GB per step at ~460 GB/s: DRAM bound); at 1 user the bf16 KDA input projection (710 us,
DRAM bound), the MoE sparse matmuls (0.8 ms) and the DSA absorbed q / out projections (0.6 ms). These are weight-read
bound: tensor parallelism (stage 4) and bfp4 / bfp8 weights (stage 8) address them, not graph fusion.

## Fusing-pattern assessment (graph-fusing skill)

| Pattern / candidate | Status |
|---|---|
| Dedicated ops: KDA decode recurrence, mHC pre-collapse, mHC mix projection | applied (rewrites 1-3) |
| RMSNorm (`ttnn.rms_norm`), SiLU / sigmoid activations, softmax / SDPA (`flash_mla_prefill`, `sparse_sdpa`), topk (`ttnn.topk`) | already dedicated ops in the functional decoder |
| Distributed RMSNorm (`rms_norm_pre_all_gather`) for the mHC sum of squares | rejected: its CBs scale with the 16384-wide row (8.5 MB > 1.5 MB L1); superseded by `mhc_mix` |
| KDA gated output norm -> `kda.sigmoid_gated_rms_norm` | not applicable at decode: needs T % 32 == 0 and a `[B, T, W]` gate; decode rows are users (any count), the `[1, 1, B, P]` projection would need a relayout costing what the op saves |
| Packed-decode `kda.qkv_causal_conv1d_silu` (main's qwen38 trick) | rejected: bf16 SiLU input (the port's conv accumulates in fp32), B % 8 == 0 only, and the history update stays with the caller; the fp32 conv is 8 small ops |
| Shared-LHS matmul packing (QKV / gate-up) | already packed: KDA input projection holds q / k / v / decay rank / gate / beta in one matmul; MLP / experts pack gate + up |
| Matmul + activation / bias folding | applied where it exists (decay `linear(bias, activation="sigmoid")`) |
| Paged indexer for decode | dedicated op (`dsa_decode_scores`, stage-1 paging work) |
| Remaining eltwise chains (KDA gates, DSA open-pool update, MoE routing math) | 0.2-1 ms per step in total; candidates for the optimized-decoder stage (sharding / L1 placement), not single dedicated ops |

## Correctness, determinism, watcher

`tests/test_fused_decoder.py` runs the full stage-1 suite (`tests/test_functional_decoder.py`, same references and
bars) against `FusedDecoder`, plus a check that the suite really runs the fused class. Log `s2_fused_suite4.log`
(2026-10-15, final code): 43 tests, 43 passed.

| case | min update PCC kda_dense / dsa_moe / kda_moe | functional decoder (stage 1) |
|---|---|---|
| real weights (prefill 100 at slot 5, 3 decode steps, eager then traced) | 0.99997 / 0.99983 / 0.99888 | 0.99997 / 0.99979 / 0.99915 |
| lengths (KDA 1 .. 4000, DSA 1 .. 4000 incl. 2051 / 2177, kda_moe 33 / 2049) | >= 0.99996 / >= 0.99961 / >= 0.99870 | >= 0.99996 / 0.98969* / >= 0.99870 |
| 4 users, one-by-one prefill, batched decode | 0.99996 / 0.99877 / 0.99917 | 0.99997 / 0.99877 / 0.99911 |
| 32 users, all slots active, eager then traced | 0.99992 / 0.99539 / 0.99435* | 0.99993 / 0.98292* / 0.99437* |
| paged caches, shuffled block table | 0.99910 (dsa_moe) | 0.99189* |
| inactive slots keep their state (NaN row in the idle slot) | 0.99997 / 0.99989 / 0.99939 | 0.99997 / 0.99989 / 0.99941 |
| determinism (prefill and traced decode bit-identical over two runs) | 0.99996 / 0.99982 / 0.99923 | 0.99997 / 0.99983 / 0.99914 |
| decode from an empty state | 0.99997 / 0.99993 / 0.99969 | 0.99997 / 0.99993 / 0.99974 |
| synthetic weights (prefill; MoE bar 0.98, see stage-1 README) | 0.99957 / 0.99818 / 0.98974 | 0.99957 / 0.99797 / 0.99206 |
| no host fallback during decode | pass x 3 | pass x 3 |

\* Router near-tie tokens (HF 8th-vs-9th expert margin < 5e-3, bar 0.98; stage-1 README). The fused decoder's mHC
mixes round differently (split-K fp32 partials), so a different set of tokens lands on the near-tie side; the
non-near-tie tokens agree to the functional decoder within the table.

Watcher (`TT_METAL_WATCHER=10`, fused decoder, `scripts/s2_watcher.sh`, log `watcher/console.log`): `test_perf_prefill[kda_moe-2048]`, `test_perf_traced_decode[kda_moe-32]` and `[dsa_moe-32]` (worker_l1_size 1,444,864) pass, 1807 kernel-status lines, 0 error / assert / sanitize lines (2026-10-15, final code).

## Limitations

- One die; weight-bound matmuls dominate (see Performance): stages 4 and 8.
- The mHC row sum of squares carries the Gram matmul's TF32 rounding (~1.5e-3 relative).
