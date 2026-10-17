# GLM-5.3-Flash (E192) fused decoder

Stage 2 of the tt-model-bringup flow. `tt/fused_decoder.py` (`FusedDecoder`) is the functional decoder
(`tt/functional_decoder.py`) with its op graph rewritten into fewer, dedicated ops. Contract, state, paging and
semantics are unchanged (same `from_state_dict` / `prefill_inputs` / `prefill_forward` / `commit_prefill` /
`set_page_table` / `decode_inputs` / `decode_forward`); `FusedFlags` turns each rewrite on or off for before / after
measurements. One die (mesh 1x1), real E192 weights.

Evidence: logs in `evidence/` (copies of `~/glm53-flash-bringup/logs/`), the suite logs `s2_fused_suite{4,5,6}.log`,
Tracy tables in `tracy/` and `tracy_nosplit/`, watcher runs in `watcher_correctness/`. The run scripts are copied to
`scripts/` (host paths: `~/glm53-flash-bringup`; every device job goes through `scripts/run-locked.sh`).

## Rewrites

| # | Subgraph (functional decoder) | Replacement | Kind |
|---|---|---|---|
| 1 | KDA decode: q / k / v slices, two reshape + L2-norm pairs, 12-op fp32 recurrence over `[B, 64, 128, 128]` | `ttnn.experimental.kda.recurrent_decode` (new op, written for this port from `ttnn.transformer.fused_recurrent_gated_delta_rule` of origin/main): per-key-channel decay, in-kernel L2 norms, head rows read straight from the conv output, state updated in place (trace-safe) | dedicated op |
| 2 | mHC pre-collapse: n stream slices + n coefficient slices + n multiply / addcmul | `ttnn.experimental.deepseek_prefill.mhc_pre` (new op), bf16 output feeds the RMSNorm (typecast folded) | dedicated op |
| 3 | mHC mixes: single-core `[T, 16384] x [16384, 32]` matmul + single-core mean of squares | `ttnn.experimental.deepseek_prefill.mhc_mix` (new op): split-K partials of the projection and of the per-row-tile Gram matrix (diagonal = row sum of squares), one (row tile, K chunk) unit per core (16-64 chunks), then sum / diagonal / rsqrt | dedicated op |
| 4 | DSA head split (q_b output; prefill indexer queries) and head concat (W_UV output): reshape + permute | `ttnn.experimental.nlp_create_qkv_heads` (`num_kv_heads=0`) and `nlp_concat_heads` (groups of 16 heads, then slice + concat) at prefill rows (> 32); 32 decode rows take rewrite 7; 1 row keeps reshape + permute | dedicated op (`dsa_heads_op`) |
| 5 | Dense MLP: two gate / up linears + minimum, silu, clamp, multiply. Routed experts: minimum + tensor maximum + swiglu (slice, slice, silu, multiply) | dense: one packed `[gate \| up]` linear (on-device concat of the bfp8 tiles), two slices and one multiply; experts: two slices and one multiply. The clamps and SiLU are input activations of the multiply (`HARDTANH(-L, L)` on up, `MINIMUM(L)` + `SILU` on gate) | shared-LHS merge + activation fold (`mlp_act_fold`) |
| 6 | KDA decode: conv 4 multiply + 3 add; history 3 where + 3 copy; output gate reshape + typecast + sigmoid + multiply + reshape | conv: multiply + 3 `addcmul` on fp32 tap constants; history: 3 `where` written straight into the history buffers; gate: applied in the `[B, H*V]` order, sigmoid as an input activation of the multiply | op merge + fold (`kda_decode_fold`) |
| 7 | 32-user decode: KDA `o` `[B*H, V] -> [B, H*V]` tiled reshape; DSA head split / concat tiled reshape + permute | untilize, row-major view / permute, tilize | layout change (`decode_rm_relayout`) |
| 8 | `mhc_mix` finalize: multiply by 1/K, add eps, rsqrt, multiply | one multiply with `MUL_UNARY_SFPU(1/K)`, `ADD_UNARY_SFPU(eps)`, `RSQRT` as input activations | activation fold (`mhc_finalize_fold`) |

Rewrites 4-6 come from stage review #1, rewrites 7-8 from stage review #2. Each one is bit-identical to the graph it
replaces:
- op level: `tools/fusing_probe.py` (`torch.equal` against the current graph), logs `evidence/s2r_fusing_probe{,2,3}.log`
  (review #1) and `evidence/s2r2_probe.log`, `evidence/s2r2_probe2.log` (review #2);
- layer level: `tools/fused_flag_check.py` (real weights, L0 / L3 / L4; a 2048 + 4 row prefill, then one eager and 4
  traced decode steps, at 32 slots and at 1 slot; all flags off against all flags on): `evidence/s2r2_flag_check.log`,
  36 / 36 outputs bit-identical. (`evidence/s2r_flag_check.log`: the review #1 run, one eager step at 32 slots.)

Op unit tests on the final (fp32-exact) kernels, `fp32k_after3.log` / `fp32k_after4.log` (stage-1 review #4, logs in
`doc/functional_decoder/kernel_precision/`) and `evidence/s2r2_mhc_mix.log`:
- `kda.recurrent_decode`: o / state PCC >= 0.9999999 (B 1 / 8 / 32, 3 chained steps, traced replay); 1000 chained steps
  with decays >= 0.99 / 0.999: state PCC 0.999999993 / 0.999999983, state rms ratio 0.99997 (no drift).
- `mhc_mix` (split-K projection + Gram partials, x split hi + lo, no TF32 read): mixes PCC >= 0.9999998, row sum of
  squares within 2.3-3.1e-4 relative. The rest is the FPU's own in-face accumulation, one-signed (`evidence/fp32k_diag4.log`:
  mean +8.5e-5 on positive exact operands).
- `FusedHyperConnection._mixes_split_k` (the decoder's finalize, where the stage-2 NaN bug lived), T = 1 and 8, NaN / inf
  in the tile's padding rows: finite, bit-identical to the zero-padded run, mixes PCC >= 0.99999999. Signed row sum of
  squares error (implied by the rsqrt scale): fused mean -2.3e-4 (T 1) / -1.7e-4 (T 8), max 2.3e-4; the functional
  composite (multiply + mean, fp32 SFPU) on the same rows: -5e-8 / -3e-8. The fused scale is therefore ~1e-4 low, one
  sign; this is 40x below the bf16 rounding (3.9e-3) of the mHC output `h` that feeds the RMSNorm.
- `mhc_pre`, `mhc_post`, `mhc_split_sinkhorn`: pass (`fp32k_after3.log`).

## Performance (one die, warmed; Tracy + tt-perf-report)

`scripts/01-tracy.sh` runs each perf case of `tests/test_perf_decoder.py` under Tracy (tables / CSVs:
`doc/functional_decoder/tracy/<case>/`, `doc/fused_decoder/tracy/<case>/`). Device time = sum of the op device times of
one pass (decode: one traced replay, the 3-replay sum / 3). Host wall under the profiler is not used for decisions.

Captures (stage review #2, 2026-10-17 10:56-12:04, final code and final fp32-exact kernels, every rewrite on;
`scripts/s2r2_chain.sh`, log `evidence/s2r2_tracy.log`):
- fused: all 12 cases;
- functional: the 8 KDA cases, same session. The 4 functional DSA cases are from 2026-10-16 21:37-21:41
  (`evidence/s2r_tracy_dsa.log`), after the final kernels were swapped in (21:28).
- "fused, before" is the stage-2 capture (f46a06ae9ad, 2026-10-15, the TF32 kernels).
- The worktree carries an uncommitted stage-6 change, `GLM53_MOE_SORTED_ROWS=1024` (`tt/moe_opt.py`, bit-identical,
  memory): a 2048-row MoE prefill chunk runs its routed experts in two 1024-row blocks. The 2048-row MoE rows in the
  table ran with it, except functional `dsa_moe-2048` (2026-10-16, before it). Fused without it (`tracy_nosplit/`,
  `evidence/s2r2_tracy_moe2048.log`, the committed stage-2 MoE path): `kda_moe-2048` 85 ops / 77621 us, `dsa_moe-2048`
  118 ops / 109227 us.

| case | functional ops / device us | fused, before ops / device us | fused ops / device us | speed-up vs functional |
|---|---|---|---|---|
| traced decode kda_dense, 1 user | 130 / 3542 | 82 / 2404 | 69 / 2443 | 1.45x |
| traced decode kda_dense, 32 users | 128 / 16527 | 78 / 3370 | 67 / 3262 | 5.07x |
| traced decode kda_moe, 1 user | 166 / 4075 | 118 / 2942 | 107 / 2979 | 1.37x |
| traced decode kda_moe, 32 users | 172 / 29727 | 122 / 16581 | 110 / 15898 | 1.87x |
| traced decode dsa_moe, 1 user | 181 / 3949 | 159 / 3328 | 153 / 3347 | 1.18x |
| traced decode dsa_moe, 32 users | 187 / 17188 | 163 / 16540 | 158 / 15590 | 1.10x |
| prefill kda_dense 128 / 2048 rows | 80 / 4097, 43557 | 56 / 3212, 38650 | 48 / 3200, 30000 | 1.28x / 1.45x |
| prefill kda_moe 128 / 2048 rows (2048: both with the MoE split) | 118 / 21300, 96832 | 94 / 20482, 79477 | 85 / 19679, 89115 | 1.08x / 1.09x |
| prefill dsa_moe 128 / 2048 rows (2048: both without the split) | 148 / 22251, 131283 | 124 / 21259, 126370 | 118 / 19824, 109227 | 1.12x / 1.20x |

Fused against "fused, before": every 32-user case and every prefill case is faster (2048-row MoE prefill: compared
without the stage-6 split). The 1-user decode cases are 39 /
37 / 19 us slower. The fp32-exact kernels cost more than the review #1 / #2 rewrites save at one user (kernel table
below). Their precision gain is the reason (below).

The rewrites in the captures:
- DSA prefill 2048: head split / concat 17.4 ms (ReshapeView 6122 + Transpose 3946 + Transpose 1376 + ReshapeView
  5984 us) -> 1.05 ms (NlpCreateHeads 384 + NLPConcatHeads 327 + Concat 341 us).
- DSA decode, 32 users: head split 186 -> 45 us (untilize 6.6 + view 21.8 + permute 9.0 + tilize 7.3 us), concat
  186 -> ~45 us (row-major permute path).
- Dense MLP prefill 2048: two 7.9 ms linears and four eltwise ops (1.24 ms) -> one packed linear, two slices and one
  multiply (`kda_dense-2048`: 38.7 -> 30.0 ms).
- MoE: expert activation chain 1298 -> 668 us per MoE layer at 32 users.
- KDA decode, 32 users (`kda_dense-32`): conv 248 -> 149 us, history shift 80 -> 54 us, output gate 237 -> 150 us,
  `o` relayout 127 -> 55 us (untilize 10 + view 28 + tilize 17 us).
- `mhc_mix` finalize: 3 ops fewer per mHC (rsqrt chain folded).

### fp32-exact kernels: cost and reason

The five custom kernels were made fp32-exact in stage-1 review #4 (b57f7da0b59; `doc/functional_decoder/README.md` and
`work_log.md`). They no longer read fp32 operands through the FPU's TF32 source registers: fp32 through DST, hi / lo
operand splits, exact reciprocals. Per call, Tracy device time, before = stage-2 TF32 kernels (f46a06ae9ad captures),
after = final (this capture):

| op | 1 user | 32 users | prefill 2048 rows |
|---|---|---|---|
| `kda.recurrent_decode` | 36.3 -> 67.0 us | 686.6 -> 837.9 us | - |
| `mhc_post` | 47.7 -> 58.4 us | 48.3 -> 58.7 us | 761.6 -> 1003.0 us |
| `mhc_split_sinkhorn` | 81.9 -> 89.5 us | 81.9 -> 89.4 us | 89.6 -> 96.9 us |
| `mhc_mix` | 21.3 -> 24.5 us | 21.2 -> 24.4 us | 727.9 -> 845.9 us |
| `mhc_pre` | 24.0 -> 24.8 us | 24.1 -> 24.5 us | 392.6 -> 399.6 us |
| per KDA layer step (1 recurrent_decode + 2 x the four mHC ops) | 386 -> 461 us | 1038 -> 1232 us | 3944 -> 4691 us |

Reason (precision against fp32):
- KDA state over 1000 chained decode steps (`test_kda_recurrent_decode_long_chain`): decays >= 0.999, state rms ratio
  0.847 -> 0.99997, state PCC 0.98697 -> 0.999999983 (`kernel_precision/s2_kda_chain.log` -> `fp32k_after3.log` in
  `doc/functional_decoder/`). The ratio is the same at step 100 and step 1000: nothing compounds. 34 of 45 layers are
  KDA.
- mHC post / comb against exact fp32: +0.30 % -> -0.02 % mean (`doc/functional_decoder/decode_stage_l4c.log`).
- A cheaper `recurrent_decode` variant (826 us at 32 users, `evidence/fp32k_after.log`) kept rms ratio 0.99983 and failed the
  1e-4 rms bound of the chain test. The final variant costs 22 us more at 32 users (about 0.75 ms per step over 34
  layers) for that 1.5e-4. `recurrent_decode` at one user (+31 us) is a stage-3 target.

### Traced DSA decode: the 20 ms replay gaps

The stage-2 capture of fused `dsa_moe-32` (2026-10-15 18:19) showed 20.06 and 19.27 ms op-to-op gaps at the replay
boundaries and 56.5 ms/step host wall. The re-captures do not show them (2026-10-16 21:17 and 2026-10-17 11:11):
- replay-boundary gaps 0.28 ms, the same as the functional case (0.28 ms) and the 1-user case (0.28 ms);
- host wall 16.5 ms/step against 15.6 ms device time (functional: 18.1 against 17.2 ms).

The replay-boundary gap holds the host work between two replays: `DecodeRunner.traced_step` builds the next step's
inputs on the host (`GlmDSA.update_decode_inputs`: the torch index tensors and 11 host-to-device copies) while the
previous replay runs. When that work takes longer than one replay, the device waits. On the final code it takes less
than the 15.6 ms replay, so the gap is the normal 0.28 ms dispatch gap. The stale capture ran before the paged-DSA commit
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
| DSA concat heads (`nlp_concat_heads`) | applied at prefill rows (rewrite 4). The op holds every head of a 32-row block in L1, double-buffered: 64 heads x 256 need 2.2 MB > 1.5 MB (TT_THROW, probe log 1). Adapted retry: a free `[4, 16, rows, 256]` view, the op per group, then slice + concat. H 64: 7410 -> 1043 us at 2048 rows (2 groups: 1101 us), 548 -> 115 us at 128 rows. H 16 needs no groups: 2043 -> 85 us. Decode rows: grouped op measured in review #2 (below); 32 rows take the row-major path (rewrite 7). |
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
| `mhc_mix` finalize (review #2 P1-a): 11-13 small ops per mHC | partly applied (rewrite 8): the scale chain folds into the final multiply, bit-identical (`evidence/s2r2_probe.log`: 168 -> 114 us host-timed at T 1, 157 -> 109 at T 32, 266 -> 255 at T 2048). Rejected, measured: `moreh_sum` over the chunks (fp32 input unsupported, TT_FATAL: bf16 / int32 only); the 64-wide diagonal mask on the unsliced sum (no gain: 173 vs 168 us). Rejected, cost: the chunk sum inside `mhc_mix` (the C partials of one row tile live on C different cores; an in-op sum needs a second reduction phase across cores). The chunk sum costs FillPad + Transpose + Reduce + Transpose = 14 us per mHC at B 1 (Tracy), so the in-op reduction can save at most ~28 us per layer step. |
| KDA decode `o` relayout after `recurrent_decode` (review #2 P1-b) | applied for 32 users (rewrite 7): untilize + row-major view + tilize, 129 -> 56 us, bit-identical. B 1 keeps the tiled reshape (9 vs 45 us). Not taken: `o` in `[B, H*V]` order from the op. `ttnn.rms_norm` normalises whole rows, so the per-head RMSNorm over V would then need its own grouped-norm kernel. A gated-norm epilogue inside `recurrent_decode` adds a cross-tile reduction, the norm weight and the gate input to the fp32-exact kernel: not a small change. Both save at most the remaining 56 us. |
| Single-core RMSNorm after `mhc_pre` (review #2 P1-c) | handed to stage 3/4 (sharding): the interleaved `ttnn.rms_norm` splits work by tile rows, so one tile of decode rows runs on one core (66 us). Width-sharded `rms_norm` (`multichip_decoder.sharded_rms_norm`, already used by the multichip decoder): 33 us (16 cores) / 36 us (32 cores) at B 1 and B 32, PCC 0.9999987, not bit-identical (max 1 bf16 ulp). This is an L1 sharding rewrite, not a fusion. |
| KDA prefill: slice (511 us) + untilize (479 us) of the qkv columns after the input projection (review #2 P1-d) | rejected, measured. (1) `ttnn.untilize_with_unpadding` (untilize + crop in one op) on `[1, 2048, 32960]` hung the device: the probe hit its 1800 s timeout (`evidence/s2r2_probe.log`). (2) Adapted retry, a separate qkv matmul with `untilize_out` plus a second matmul for the other columns (`evidence/s2r2_probe2.log`): `MatmulMultiCoreReuseMultiCast1DProgramConfig` (in0 multicast, `per_core_M` = all 64 row tiles, `per_core_N` 8) needs 3.8 MB of L1 per core (TT_THROW, limit 1.5 MB). `MatmulMultiCoreReuseProgramConfig` rejects the rank-2 weight, and subblocks wider than 4 tiles are rejected. The two ops stay (3.3 % of `kda_dense-2048`). |
| DSA decode head concat, grouped `nlp_concat_heads` at decode rows (review #2 P1-e) | rejected, measured: 4 x 16 groups at 32 rows 124 us, at 1 row 115 us (permute + reshape: 186 / 15 us). Row-major permute is faster at 32 rows (63 us; split 52 us) and is applied (rewrite 7); 1 row keeps the tiled path (15 / 25 us). |
| `sparse_sdpa` untilize / tilize pairs in DSA decode (q 11.7 us, indices 10.7 us, output 13.2 us at B 32) | kept: the op contract needs ROW_MAJOR q, indices and output |
| Packed dense MLP at decode rows | kept: probe 338 us packed against 320 us for two linears + folded multiply at 32 rows (+18 us per dense layer, 3 layers). Picking by row count needs both weight layouts in DRAM (+100 MB per dense layer at bfp8). |
| MoE down-projection untilize via a split `per_core_N` | stage 3 item: the resident-block condition might hold with `per_core_N` 16 (about 1.1 MB). |
| Paged indexer for decode | dedicated op (`dsa_decode_scores`, stage-1 paging work) |
| Remaining eltwise chains (KDA gates, DSA open-pool update, MoE routing math) | small at decode (each op 1-25 us at B 32); candidates for the optimized-decoder stage (sharding / L1 placement), not single dedicated ops |

## Correctness, determinism, watcher

`tests/test_fused_decoder.py` runs the full stage-1 suite (`tests/test_functional_decoder.py`, same references and
bars) against `FusedDecoder` with every `FusedFlags` rewrite on, plus a check that the suite really runs the fused class.
Log `s2_fused_suite6.log` (2026-10-17, final code and kernels): 43 tests, 43 passed. Every PCC equals the one of
`s2_fused_suite5.log` (2026-10-17 00:38, before rewrites 7-8), as the layer-level bit-exactness predicts. The functional
column is `doc/functional_decoder/s1_functional6.log` (same kernels).

| case | min update PCC kda_dense / dsa_moe / kda_moe | functional decoder |
|---|---|---|
| real weights (prefill 100 at slot 5, 3 decode steps, eager then traced) | 0.99997 / 0.99981 / 0.99945 | 0.99997 / 0.99984 / 0.99945 |
| lengths (KDA 1 .. 4000, DSA 1 .. 4000 incl. 2051 / 2177, kda_moe 33 / 2049) | >= 0.99996 / >= 0.99962 / >= 0.99874 | >= 0.99996 / 0.98972* / >= 0.99874 |
| 4 users, one-by-one prefill, batched decode | 0.99997 / 0.99878 / 0.99944 | 0.99997 / 0.99879 / 0.99951 |
| 32 users, all slots active, eager then traced | 0.99994 / 0.98296* / 0.99274* | 0.99992 / 0.99568** / 0.99428* |
| paged caches, shuffled block table | 0.99909 (dsa_moe) | 0.99191* |
| inactive slots keep their state (NaN row in the idle slot) | 0.99997 / 0.99990 / 0.99957 | 0.99997 / 0.99989 / 0.99957 |
| determinism (prefill and traced decode bit-identical over two runs) | 0.99997 / 0.99942 / 0.99915 | 0.99997 / 0.99942 / 0.99929 |
| decode from an empty state | 0.99998 / 0.99993 / 0.99953 | 0.99998 / 0.99993 / 0.99952 |
| synthetic weights (prefill; MoE bar 0.98, see stage-1 README) | 0.99984 / 0.99769 / 0.99093 | 0.99984 / 0.99785 / 0.98976 |
| no host fallback during decode | pass x 3 | pass x 3 |

\* Router near-tie tokens (HF 8th-vs-9th expert margin < 5e-3, bar 0.98; stage-1 README). The fused decoder's mHC
mixes round differently (split-K partials, sum of squares ~2e-4 low), so a different set of tokens lands on the
near-tie side. New in the fused path: L3 slot 2 pos 38 at 32 users (delta PCC 0.98296, update ratio within bounds). HF's
margin there is 7.97e-4 (`doc/functional_decoder/route_margin_l3_s1302.log`). `tools/route_flip_check.py --fused`
(`evidence/flip_tt_s1302_fused.log`) shows the TT top-8 with HF's 9th expert 111 in place of 55: a router flip. The
functional path routes this token as HF does (`doc/functional_decoder/flip_tt_s1302.log`); both paths flip pos 39
(margin 1.9e-4).

\*\* `s1_functional6.log` stops this case at the confirmed pos 39 flip (stage-1 README); the min covers the tokens
before it.

The decode runs at most 5 consecutive steps per slot (suite: 4; `fused_flag_check`: 1 eager + 4 traced). Long decode is
covered at op level only (1000 chained `recurrent_decode` steps, above).

Watcher correctness run (final code and kernels, every `FusedFlags` rewrite on; `TT_METAL_WATCHER=10`,
`TT_METAL_WATCHER_NOINLINE=1`, worker_l1_size 1,444,864), archived in `watcher_correctness/`:
- `suite/`: `tests/test_fused_decoder.py -k "test_real_weights or test_batched_users or test_batch32_all_active or
  test_inactive_slot_state or (test_lengths and (kda_dense-2049 or dsa_moe-2051 or kda_moe-2049 or dsa_moe-1))"`
  (real weights, HF-compared; B 32 traced decode with `recurrent_decode` on 110 cores; NaN idle rows): 17 passed,
  `watcher.log.gz` 65,193 lines, 0 error / assert / sanitize / overflow lines (2026-10-17 08:30,
  `scripts/s2r2_chain.sh`).
- `ops/`: `test_kda_recurrent_decode.py` (incl. traced and the 1000-step chain), `test_mhc_mix.py` (incl. the T 1 / 8
  padded-row cases), `test_mhc_pre.py`, `test_mhc_post.py`, `test_mhc_split_sinkhorn.py`: 37 passed, 0 suspicious lines
  (2026-10-17, `scripts/s2_watcher_correct.sh`).

The earlier `watcher/` run (2026-10-15) covers three perf cases on the stage-2 kernels only (no assertions).

## Limitations

- One die; weight-bound matmuls dominate (see Performance): stages 4 and 8.
- The mHC row sum of squares carries the FPU's in-face accumulation error (~2-3e-4 relative, one-signed; the
  functional composite is exact to ~1e-7). See the op unit tests above.
- `kda.recurrent_decode` requires `(B * H) % 32 == 0`: B 1 at TP4 (16 heads per die) is rejected (stage 4 item).
