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
| 4 | DSA head split (q_b output; prefill indexer queries) and head concat (W_UV output): reshape + permute | `ttnn.experimental.nlp_create_qkv_heads` (`num_kv_heads=0`) and `nlp_concat_heads` (groups of 16 heads, then slice + concat) at prefill rows (> 32); decode rows keep reshape + permute | dedicated op (`dsa_heads_op`) |
| 5 | Dense MLP: two gate / up linears + minimum, silu, clamp, multiply. Routed experts: minimum + tensor maximum + swiglu (slice, slice, silu, multiply) | dense: one packed `[gate \| up]` linear (on-device concat of the bfp8 tiles), two slices and one multiply; experts: two slices and one multiply. The clamps and SiLU are input activations of the multiply (`HARDTANH(-L, L)` on up, `MINIMUM(L)` + `SILU` on gate) | shared-LHS merge + activation fold (`mlp_act_fold`) |
| 6 | KDA decode: conv 4 multiply + 3 add; history 3 where + 3 copy; output gate reshape + typecast + sigmoid + multiply + reshape | conv: multiply + 3 `addcmul` on fp32 tap constants; history: 3 `where` written straight into the history buffers; gate: applied in the `[B, H*V]` order, sigmoid as an input activation of the multiply | op merge + fold (`kda_decode_fold`) |

Rewrites 4-6 come from stage review #1. Each one is bit-identical to the graph it replaces:
- op level: `tools/fusing_probe.py` (`torch.equal` against the current graph), logs `~/glm53-flash-bringup/logs/s2r_fusing_probe{,2,3}.log`;
- layer level: `tools/fused_flag_check.py` (real weights, L0 / L3 / L4, a 2048 + 4 row prefill and a 32-slot decode step, flags off against flags on), log `~/glm53-flash-bringup/logs/s2r_flag_check.log`.

Op unit tests (`tests/test_kda_recurrent_decode.py`, `tests/test_mhc_pre.py`, `tests/test_mhc_mix.py`): output /
state PCC >= 0.9999993 (recurrent_decode, B = 1 / 8 / 32, chained steps, traced replay), >= 0.9999985 (mhc_pre),
mixes PCC 0.9999998 and row sum of squares within 1.6e-3 relative (mhc_mix: the Gram matmul reads fp32 at TF32
precision; the resulting ~8e-4 on the rsqrt scale is below the HF stream's own bf16 rounding).

## Performance (one die, warmed; Tracy + tt-perf-report)

`scripts/01-tracy.sh` runs each perf case of `tests/test_perf_decoder.py` under Tracy (tables / CSVs:
`doc/functional_decoder/tracy/<case>/`, `doc/fused_decoder/tracy/<case>/`). Device time = sum of the op device times of
one pass (decode: one traced replay, the 3-replay sum / 3). Host wall under the profiler is not used for decisions.

Captures (stage review #1, 2026-10-16, final code, all flags on):
- fused, all 12 cases: `scripts/s2r_tracy_dsa.sh` (log `logs/s2r_tracy_dsa.log`); the six prefill cases again with
  `scripts/s2r_tracy_prefill_redo.sh` (log `logs/s2r_tracy_prefill_redo.log`), because the raw `ops.csv` of the first
  run was lost. The table uses the second run (within 1 % of the first).
- functional, the four DSA cases, same session (`scripts/s2r_tracy_dsa.sh`). The functional KDA cases are from
  2026-10-15 20:10-20:18 (`scripts/s2_tracy.sh`); this work does not change the functional decoder.
- "fused, before" is the stage-2 capture (2026-10-15; the DSA cases there ran before the `mhc_mix` NaN fix).

| case | functional ops / device us | fused, before ops / device us | fused ops / device us | speed-up vs functional |
|---|---|---|---|---|
| traced decode kda_dense, 1 user | 130 / 3511 | 82 / 2403 | 75 / 2413 | 1.46x |
| traced decode kda_dense, 32 users | 128 / 16464 | 78 / 3384 | 71 / 3308 | 4.98x |
| traced decode kda_moe, 1 user | 166 / 4044 | 118 / 2937 | 113 / 2944 | 1.37x |
| traced decode kda_moe, 32 users | 172 / 29707 | 122 / 16588 | 114 / 15946 | 1.86x |
| traced decode dsa_moe, 1 user | 181 / 3949 | 159 / 3328 | 159 / 3308 | 1.19x |
| traced decode dsa_moe, 32 users | 187 / 17188 | 163 / 16560 | 160 / 15749 | 1.09x |
| prefill kda_dense 128 / 2048 rows | 80 / 4043, 43349 | 56 / 3212, 38650 | 54 / 3225, 29911 | 1.25x / 1.45x |
| prefill kda_moe 128 / 2048 rows | 118 / 21294, 84292 | 94 / 20482, 79477 | 91 / 19730, 77823 | 1.08x / 1.08x |
| prefill dsa_moe 128 / 2048 rows | 148 / 22251, 131283 | 124 / 21259, 126370 | 124 / 19919, 108594 | 1.12x / 1.21x |

Review #1 rewrites in the captures:
- DSA prefill 2048: the head split / concat drop from 17.4 ms (ReshapeView 6122 + Transpose 3946 + Transpose 1376 +
  ReshapeView 5984 us) to 1.05 ms (NlpCreateHeads 384 + NLPConcatHeads 327 + Concat 341 us).
- Dense MLP prefill 2048: two 7.9 ms linears and four eltwise ops (1.24 ms) become one packed linear, two slices and one
  multiply (`kda_dense-2048`: 38.7 -> 29.9 ms).
- MoE: the expert activation chain halves (probe: 1307 -> 665 us at 32 users).
- KDA decode at 32 users (`kda_dense-32`): conv 248 -> 149 us, history shift 80 -> 54 us, output gate 237 -> 150 us
  (212 us less). The total drops only 76 us (3384 -> 3308 us): `kda.recurrent_decode` went 692 -> 838 us in the same
  period (the parallel fp32-state precision work on that kernel, review #1 P1), not part of these rewrites.
- Decode rows keep the DSA reshape + permute, so the DSA decode gain comes from the MoE fold only.

### Traced DSA decode: the 20 ms replay gaps

The stage-2 capture of fused `dsa_moe-32` (2026-10-15 18:19) showed 20.06 and 19.27 ms op-to-op gaps at the replay
boundaries and 56.5 ms/step host wall. The re-capture on the final code does not show them:
- replay-boundary gaps 0.28 ms (rows 160 and 320 of `perf_report.csv`), the same as the functional case (0.28 ms) and
  the 1-user case (0.28 ms);
- host wall 16.6 ms/step against 15.7 ms device time (functional: 18.1 against 17.2 ms).

The replay-boundary gap holds the host work between two replays: `DecodeRunner.traced_step` builds the next step's
inputs on the host (`GlmDSA.update_decode_inputs`: the torch index tensors and 11 host-to-device copies) while the
previous replay runs. When that work takes longer than one replay, the device waits. On the final code it takes less
than the 15.7 ms replay, so the gap is the normal 0.28 ms dispatch gap. The stale capture ran before the paged-DSA commit
(339700f8901, 18:56) and the `mhc_mix` NaN fix, and in parallel with other host jobs. Its exact cause was not
reproduced. The review's "host-only measurement before stage 6" of the DSA input cost stays open.

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
| DSA split heads (`nlp_create_qkv_heads`) | applied at prefill rows (rewrite 4). Bit-identical. H 64: 9966 -> 392 us at 2048 rows, 532 -> 215 us at 128 rows. H 16 (TP4 per die): 2232 -> 84 us at 2048 rows. Indexer queries (H 32, D 128): 2176 -> 84 us at 2048 rows. Rejected at decode rows: the op splits work by 32-row blocks, so one tile of users runs on one core (H 64: 209 us against 187 us; H 16: 53 against 50 us). |
| DSA concat heads (`nlp_concat_heads`) | applied at prefill rows (rewrite 4). The op holds every head of a 32-row block in L1, double-buffered: 64 heads x 256 need 2.2 MB > 1.5 MB (TT_THROW, probe log 1). Adapted retry: a free `[4, 16, rows, 256]` view, the op per group, then slice + concat. H 64: 7410 -> 1043 us at 2048 rows (2 groups: 1101 us), 548 -> 115 us at 128 rows. H 16 needs no groups: 2043 -> 85 us. Decode rows keep permute + reshape (same one-core limit). |
| `split_query_key_value_and_split_heads`, `nlp_create_qkv_heads_decode` / `nlp_concat_heads_decode`, head-major W_UK / W_UV packing | not applicable: the first needs K / V; the decode variants emit `[1, B, H, D]` (users major), but the per-head W_UK / W_UV bmm needs `[1, H, B, D]`; the projections already write head-major columns, so the split is the only relayout left |
| Shared-LHS packing: KDA input projection | already packed (q / k / v / decay rank / gate / beta in one matmul) |
| Shared-LHS packing: dense MLP gate + up | applied (rewrite 5). The stage-2 README claimed this was packed; `tt/mlp.py` ran two linears. Bit-identical. 2048 rows: 17194 -> 8182 us (two linears 15843 us, packed linear 7190 us); 32 decode rows: 334 -> 338 us (+4 us, within noise of the 307 / 314 us linears). One weight set serves both. |
| Matmul + activation / bias folding | applied where it exists (decay `linear(bias, activation="sigmoid")`). swiglu_limit clamps + SiLU: folded into the multiply (rewrite 5) |
| MoE routed-expert activation chain | applied (rewrite 5). Bit-identical. Static dispatch buffer `[1, G, 32, 4096]`: 32 decode users (G 195): 1307 -> 665 us per MoE layer; 2048 prefill rows (G 699): 4635 -> 2352 us. The rest of the cost scales with the static worst-case dispatch rows (6240 rows for 256 real pairs at 32 users). Folding SwiGLU into the gate / up sparse matmul would remove it, but the swiglu pairs columns of two N halves that the grouped factory computes on different blocks: a kernel change, not a graph rewrite. |
| MoE down projection `untilize_out` (no separate untilize) | not applicable (code): `moe_opt._grouped_configs` enables `untilize_out` only with a resident expert block, `k_tiles * n_tiles * tile_bytes + ... <= 1.3 MB`. A GLM-5.3 down block is 64 x 128 tiles: 8.9 MB at bfp8, 4.7 MB at bfp4. So the 259 us (decode 32) / 910 us (prefill 2048) UntilizeCodegen stays. |
| KDA gated output norm -> `kda.sigmoid_gated_rms_norm` | rejected, measured: the op needs `[B*H, T, V]` head-first input and T % 32 == 0. At decode the users are T, so the `[B*H, V]` recurrence output needs a permute: 274 us against 249 us at B 32; B 1 is rejected by the op (T % 32). |
| KDA gated output norm, adapted (gate in `[B, H*V]` order, sigmoid as input activation) | applied (rewrite 6). Bit-identical. B 32: 249 -> 163 us; B 1: 71 -> 51 us. The rejected earlier try (sigmoid activation on the bf16 gate, fp32 output) gave PCC 0.91 against the current graph; with the gate typecast to fp32 first it is exact. |
| KDA decode conv: `kda.qkv_causal_conv1d_silu` (main's packed-decode trick, `models/demos/qwen38_27b_qb2/tt/decode_conv.py`) | rejected, measured: op + main's adapters at B 32 take 663 us against 270 us for the current fp32 conv, before the extra layout changes this port needs (row-major `[B, 3, C]` history, fp32 packed qkv for `recurrent_decode`). B % 8 != 0 is not supported (B 1). |
| KDA decode conv: `addcmul` merges, in-place history shift | applied (rewrite 6). Bit-identical. Conv with fp32 tap constants: 270 -> 174 us (B 32), 183 -> 172 us (B 1). Shift: 81 -> 55 us (B 32 and B 1). Padding B to 8 only serves the packed conv (rejected). A rotating history index needs the tap order to change every step: three captured traces or a per-step tap select, for at most the 55 us that remain. Not taken. |
| Paged indexer for decode | dedicated op (`dsa_decode_scores`, stage-1 paging work) |
| Remaining eltwise chains (KDA gates, DSA open-pool update, MoE routing math) | small at decode (each op 1-25 us at B 32); candidates for the optimized-decoder stage (sharding / L1 placement), not single dedicated ops |

## Correctness, determinism, watcher

`tests/test_fused_decoder.py` runs the full stage-1 suite (`tests/test_functional_decoder.py`, same references and
bars) against `FusedDecoder`, plus a check that the suite really runs the fused class. Log `s2_fused_suite4.log`
(2026-10-15, final code): 43 tests, 43 passed. This log predates rewrites 4-6. Rewrites 4-6 are bit-identical at
layer level (`tools/fused_flag_check.py`, L0 / L3 / L4, prefill and decode, `max|d| = 0`), so the suite results carry
over. A suite rerun on the final code is still open.

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
