# GLM-5.3-Flash (E192) optimized multichip decoder

Stage 5 of the tt-model-bringup flow. It optimizes `tt/multichip_decoder.py` (`MultichipDecoder`) in place. The decoder
runs TP4 on the (1, 4) mesh of a p300x2: 4 Blackhole dies, FABRIC_1D. Contract, state, paging and semantics stay those
of `tt/functional_decoder.py`. Decode comes first (tokens/s/user, then 32-user throughput), then prefill.

The stage-5 changes are options of one dataclass, `MCOpts` (`tt/multichip_decoder.py`). `DEFAULT_MC` holds the
selected values. `MCOpts()` (all options off, linear topology) is the stage-4 path. `GLM53_MC="key=value,..."`
overrides fields at construction for A/B runs and logs a warning.

Device ops that belong to the stage:

- `ttnn.experimental.deepseek_prefill.kda_decode_conv` (new): the KDA decode conv step in one op instead of 12. It does
  the 4-tap causal conv in fp32, SiLU, the idle-slot select and the in-place history shift (`kda_conv_op`).
- `mhc_split_sinkhorn`, `perm_fidelity` attribute (`MHC_SINKHORN_PERM_FIDELITY`, default 4): the Sinkhorn row and
  column sums at HiFi3 are bit-identical to HiFi4 (`sinkhorn_perm_fidelity = 3`). HiFi2 is not exact; the op refuses
  it unless `MHC_SINKHORN_ALLOW_INEXACT=1` (probes only).
- `mhc_split_sinkhorn`, `faces` attribute (`MHC_SINKHORN_FACES=1`, default off): the SFPU work of the 40
  normalizations runs only on the DST faces that hold the 16 used columns (`sinkhorn_faces`). See "Sinkhorn".

Review history (`~/glm53-flash-bringup/artifacts/stage5_review_{1,2,3}.md`):

- Review #1 asked for the deferred items. Round 6 answers them (sections marked "review #1").
- Review #2 found the headline numbers measured with `ag_persistent` on, which the final default does not have. The
  2026-10-18 rework re-measured everything on the final default. It added the face-restricted Sinkhorn, the DSA
  data-movement audit and the stacked (full-model) gate.
- Review #3 found the sparse-regime paths unchecked and some labels wrong. The second 2026-10-18 rework (marked
  "review #3") adds the sparse-regime checks, fixes a buffer-lifetime hazard in `dsa_tail_shared`, and corrects the
  labels.

## Result

`DEFAULT_MC` (review #3) holds these options:

- Collectives: ring when the fabric routes it (`topology="auto"`), 2 links, decode outputs in L1, non-persistent.
- Norms: width-sharded decode RMSNorms.
- DSA: the five replicated projections packed into two matmuls, and q_b / wq_b / o_proj on 1D programs. The 1-slot
  q_abs is one matmul (`dsa_q_mm`, in0 blocks of 64). The sparse-index tail is computed once per step
  (`dsa_tail_shared`).
- KDA: o_proj on a 1D program; the fused decode conv up to 16 slots (`kda_conv_max_batch = 16`).
- mHC: HiFi3 and face-restricted Sinkhorn; 16 mix chunks at decode.
- MoE: router, shared expert and dense MLP on 1D programs; grouped sparse matmuls at in0 blocks of 8.
- Prefill: the packed weights shared with prefill; explicit 2D programs for the DSA projections, the packs, the dense
  MLP, the shared expert and the KDA output projection.

Every number below comes from that default, with no `GLM53_MC`, unless a row says otherwise. Code comes from a snapshot
outside the live tree (see "Method", Provenance). The review-2 numbers come from job `records/s5r2_c2.log`.

Decode, device time of one traced replay (Tracy, device 0, µs). `tracy/before` holds the stage-4 options (run
2026-10-17 05:17). `tracy/after` holds the review-2 default: decode cases 2026-10-18 05:26-05:32, prefill cases
05:33-05:36. The 1-slot dsa_moe rows come from the review-3 default (`tracy/after_r3`, `tracy/after_r3_pos130000`, ticket 6,
15:47-15:49); the review-3 options change no other row.

| Layer kind | Users | Before | After | Change |
|---|---|---|---|---|
| kda_dense (L0) | 1 | 1134 | 844 | -25.6 % |
| kda_moe (L4) | 1 | 1338 | 1053 | -21.3 % |
| dsa_moe (L3) | 1 | 1745 | 1278 (review #3) | -26.8 % |
| kda_moe | 32 | 4599 | 4368 | -5.0 % |
| dsa_moe | 32 | 4975 | 4611 | -7.3 % |
| dsa_moe, position 130,000 (`tracy/after_r3_pos130000`) | 1 | - | 1838 (review #3) | - |

The 32-user rows move by up to 5 % between Tracy runs of the same build. The cause is the DRAM-bound sparse expert
matmuls and row gather: 2499 / 2718 / 2684 µs in three runs. The replay table (200 replays) is the stable figure.

The long-position step costs +563 µs over position 128 (review-2 Tracy pair: 1287 -> 1848), all in the sparse
attention path:

- sparse_sdpa 33 -> 356 µs (2,048 selected tokens gathered from a 131K-token cache instead of 129);
- the top-512 pool selection over ~32.5K pooled keys (`TopkLargeIndices` 7 -> 142 µs);
- the indexer scores 5 -> 38 µs, and their masking ops.

Decode, traced replay wall on the shipped build, µs. The tool is `tools/mc_decode_ab.py --base default --determinism
200`: the `DEFAULT_MC` construction including `ds_shared`, 30 back-to-back replays. Records: `records/stress_r2.jsonl`,
`longpos_r2.jsonl`; the 1-slot dsa_moe rows `records/stress_r3b.jsonl`, `longpos_r3b.jsonl` (review-3 default, ticket 6). The stage-4 column is variant `s4` of `records/ab6.jsonl`, on a stage-4 build.

| Layer kind | Users | Stage 4 | Final | Change |
|---|---|---|---|---|
| kda_dense | 1 | 1167 | 872 | -25.3 % |
| kda_moe | 1 | 1371 | 1079 | -21.3 % |
| dsa_moe, first DSA layer (writes the step tail) | 1 | 1805 | 1360 | -24.6 % |
| dsa_moe, the other 10 (read the tail) | 1 | 1805 | ~1324 (-36 µs) | -26.6 % |
| kda_moe | 32 | 4608 | 4362 | -5.3 % |
| dsa_moe | 32 | 5119 | 4717 | -7.9 % |
| dsa_moe, position 130,000 of 131,072 | 1 | 2361 | 1912 | -19.0 % |

The 45-layer stack has 3 kda_dense, 31 kda_moe and 11 dsa_moe layers. Its layer replays sum to 50.7 ms per token at 1
user, against 65.9 ms with the stage-4 options (review #2: 50.8 ms). All layer numbers use the stage-5 reference
precision (`GLM53_PRECISION=none`, stage 3's policy); stage 8 changes the shipping precision. This counts decoder layers only (no embedding, head or sampling). The
review-1 figure of 52.3 ms was measured with `ag_persistent` on.

Prefill, 2048 rows, device time of the signposted pass (Tracy, device 0, µs):

| Layer kind | Before | After | Change | CCL before -> after |
|---|---|---|---|---|
| kda_dense | 12179 | 10025 | -17.7 % | 1889 -> 1451 |
| kda_moe | 25601 | 24410 | -4.7 % | 2150 -> 1458 |
| dsa_moe | 33523 | 30647 | -8.6 % | 1617 -> 1069 |

Review-3 code changes after these runs: the shared tail buffer is allocated at build time, one head mask serves all
DSA layers, and a double free is gone. None adds device work to the step. The new 1-slot in0 block for `dsa_q_mm`
(`dsa_q_mm_ib = 64`) saves another ~7 µs per DSA layer; "Option equivalence" has the review-3 A/Bs.

## Correctness on the shipped default

Full model (the stage gate, see "Stacked gate"):

- Review #3, the stage-5 gate (ticket 5, `records/s5r3_tf_dflt5.log`): final code on a snapshot of HEAD b8198430147
  plus the review-3 edits, no `GLM53_MC`, the shipping precision (stage 8's combo_a_kv, `GLM53_PRECISION` unset).
  Teacher forcing 91 / 100 / 100, prefill check 91 / 100 / 100. With `dsa_tail_shared=0`: the same
  (`records/s5r3_tf_notail5.log`).
- Review #3 at the stage-7 precision (ticket 1, `records/s5r3_tf_final.log`): 92 / 100 / 100 and 91 / 100 / 100, with
  `dsa_idx_rep` in the default at the time (rejected later, see the audit).
- Review #2 (`records/s5r2_tf_cand3.log`): teacher forcing 92 / 100 / 100, prefill check 91 / 100 / 100. That run set
  the four review-2 options through `GLM53_MC` on the file before they became the default; the values are the same.
- Sparse regime (review #3): the full model decodes positions 2100-2107 and 3000-3007 (`tools/s5_sparse_probe.py`).
  With and without `dsa_tail_shared`, the prefill logits, all 16 decode steps' logits and the tokens are equal (max
  |diff| 0; tickets 1 and 4, `records/s5r3_sparse_full_compare.txt`). `dsa_q_mm_ib` 64 vs 8 on the shipping default:
  equal as well (ticket 5).
- Repeatability, 8 slots (review #2): decode is always traced; each prompt ran twice with each prefill path. Tokens
  were equal and max |diff| 0, on layers 0, 3-7 (`records/s5r2_det_reduced_cand.log`) and on the full model
  (`records/s5r2_det_full_final.log`). The full-model tokens equal stage 7's final run (`records/s7_final_s8.log`).

Layer suite and fault injection:

- `tests/test_multichip_decoder.py` (TP4, `records/s5r2_suite.log`): 45 passed.
- The single-die baseline compare passes with the stage-4 rework's capped near-tie allowance. kda_moe decode row 2
  (PCC 0.99645, router margin 2.9e-4, cap 1) is set aside. The other rows reach PCC 0.999998, rms ratio 1.00011.
- That row is the one that failed the round-8 compare against the 'opt_s5' baseline. The single-die 'opt_s5' run
  puts that token on the other side of the tie (`records/baseline8_rows.txt`).
- Fault injection, routed-expert output x 0.9722 (`GLM53_FAULT_ROUTED_SCALE`, `records/s5r2_suite_fault.log`): 17
  failed and 2 errors, 28 passed.
- The failing checks are the update-rms ratio (1 +- 0.01) and the baseline compare's rms (+- 0.005). Tests that
  catch the fault: real weights (both MoE kinds), batched users, determinism and paged blocks. Also: lengths (kda_moe
  33 / 2049; dsa_moe 33 to 4000), the baseline compare, `test_synthetic_weights[kda_moe]`,
  `test_batch32_all_active[dsa_moe]` and `test_inactive_slot_state[dsa_moe]`.
- MoE tests that do not catch it on their own: `test_decode_only[kda_moe]` and `[dsa_moe]`,
  `test_batch32_all_active[kda_moe]`, `test_inactive_slot_state[kda_moe]`, `test_synthetic_weights[dsa_moe]`,
  `test_lengths[dsa_moe-1]`, and `test_no_host_fallback` (no numeric check). The kda_dense tests have no routed experts.

Kernels, replays and option equivalence:

- Kernel tests (review #3, `records/s5r3_kernel.log`): 18 passed: `tests/test_mhc_split_sinkhorn.py` 12 (including
  the 5 new faces cases), the 4 upstream sinkhorn tests and `tests/test_kda_decode_conv.py` 2. The kernel ran with
  the `0.0f` stores (review #3 P3-4).
- Replay determinism: 200 replays from a fresh state are bit-identical for kda_dense / kda_moe / dsa_moe at 1 user and
  kda_moe / dsa_moe at 32 users (`records/stress_r2.jsonl`). At position 130,000, 50 replays (`longpos_r2.jsonl`).
- Option equivalence uses `tools/mc_decode_ab.py --compare 8`: 1 eager and 8 traced decode steps from a fresh state,
  compared bit by bit with the variant that has the review-2 options off. Results are in "Option equivalence".

Watcher (`records/s5r2_watcher.log`, `watcher/<case>/console.log`), 7 cases on the final default:

- All 7 tests PASSED. No watcher.log line flags a model kernel or a collective.
- Every process ends abnormally at teardown, in one of two forms seen on this board before. The 122B port's control
  (`models/autoports/qwen_qwen3_5_122b_a10b/doc/multichip_decoder/watcher/TALLY.md`) records both.
- Form (i), 4 cases: the host times out waiting for eth core 29-25 at teardown (exit 134).
- Form (ii), 3 cases (real weights dsa_moe and kda_moe, 32 users dsa_moe): the watcher trips an assert on eth core
  (0, 9) / virtual (29, 25). The message is "subordinate_erisc detected invalid NOC command buffer state", in
  `fabric_erisc_router.cpp`. It appears after the test PASSED, when a watcher dump falls into the teardown window.
- `scripts/s5_watcher.sh` now counts "tripped assert" in console.log as well as watcher.log lines.
- With the eth cores' watcher off (`TT_METAL_WATCHER_DISABLE_ETH=1`, `records/s5r3_watcher_noeth.log`): both dsa_moe cases that showed form (ii) pass with
  exit 0, 0 suspicious lines and 0 tripped asserts. The assert belongs to the eth fabric-router teardown, not to the
  decoder's kernels.

## Option equivalence (review #3 P1-1, P2-1)

Each variant runs 1 eager and 8 traced decode steps from a fresh state on the same inputs. `fin` = the review-2
options off (`dsa_tail_shared=0`, `dsa_q_mm=0`, `MHC_SINKHORN_FACES=0`). Each other variant turns one option on.
Positions above 2050 decode in the sparse regime, where the sparse index list (with its tail) feeds sparse_sdpa.
Records: `records/sparse_ab.jsonl` (positions 3000 and 130,000), `records/dense_ab.jsonl` (position 128).

Replay µs; every listed variant is bit-identical to `fin` over the 9 steps unless the row says otherwise.

Limit of this harness at the sparse positions: the A/B layer's latent cache holds only 128 written tokens, so most
of the 2048 selected keys are equal zero rows. A change that only reorders the keys can pass here and still change
the full model (`dsa_idx_rep`). The tail, the q path and the Sinkhorn do not depend on cache contents; the tail is also
checked on the full model in the sparse regime ("Stacked gate").

| Variant | dsa_moe 1 user, pos 128 | 1 user, pos 3000 | 1 user, pos 130,000 | 32 users, pos 3000 | 32 users, pos 128 |
|---|---|---|---|---|---|
| `fin` (review-2 options off) | 1413.4 | 1736.6 | 1970.1 | 5084.6 | 4719.1 |
| `dsa_tail_shared`, producer | 1413.6 | 1736.6 | 1968.4 | 5086.8 | 4721.8 |
| `dsa_tail_shared`, consumer (host tail) | 1377.8 | 1700.3 | 1941.5 | 5033.9 | 4665.9 |
| `dsa_q_mm` (in0 block 8) | 1396.3 | 1719.7 | 1949.8 | - | - |
| `dsa_q_mm`, in0 block 16 / 32 / 64 | 1390.9 / 1389.3 / 1389.0 | 1713.8 / 1712.1 / 1711.8 | 1944.6 / 1943.5 / 1942.7 | - | - |
| `sinkhorn_faces` | 1383.0 | 1705.0 | 1940.8 | 5052.5 | 4684.2 |
| `dsa_idx_rep` | 1388.5 | 1711.6 | 1942.7 | 5241.9, not identical | - |
| all four review-2 options (the default before `dsa_q_mm_ib` / `dsa_idx_rep`) | 1368.3 | 1691.0 | 1919.9 | 5068.1 | 4701.6 |

- `dsa_idx_rep` at 32 users is not bit-identical (PCC 0.99919, max rel 4.9e-2) and is +157 µs slower. Its 1-slot rows
  look bit-identical only because the A/B layer's cache is mostly empty; the option is rejected (see the audit).
- `sinkhorn_faces` on the KDA layers (pos 128): kda_moe 1 user 1109.0 -> 1078.2 µs, 32 users 4380.1 -> 4353.6 µs, both
  bit-identical. With all review-2 options: 1078.7 / 4355.1 µs.
- The `fin2` repeats (last variant) match `fin` within 15 µs, which is the run noise.
- `sinkhorn_faces` replaces the review-2 citation of `records/skf_ab.jsonl`, which timed the option without a
  cross-variant compare.

## Performance accounting (review #1 P2-5)

Roofline = bytes one decode step must read per die / 512 GB/s per die. The bytes are the weights at their stored
dtypes, the KDA state (read and write) and the DSA latent rows of 2048 selected tokens. Device = Tracy above. Replay =
the shipped-build replay wall. 1 user; the 1-slot KDA layers carry 2 state slots.

| Case | MB per die | Roofline µs | Device µs | Replay µs | Device / roofline |
|---|---|---|---|---|---|
| kda_dense | KDA in 36.3, out 8.9, decay 0.3, state 4.2, dense MLP 40.1 = 89.8 | 175 | 844 | 872 | 4.8x |
| kda_moe | KDA 49.7, router 1.6, 8 experts 28.3, shared 6.7 = 86.3 | 169 | 1053 | 1079 | 6.2x |
| dsa_moe | packs 11.3, q_b 6.7, W_UK / W_UV 4.5, wq_b 12.6, o 17.8, latent 2.1, MoE 36.6 = 91.6 | 179 | 1278 | 1360 | 7.1x |

Named gap terms of the 1-user kda_moe layer (1053 µs device, final Tracy):

- element-wise / data movement 323 µs (router 85, mHC glue 120, KDA gates 70, the rest below 10 µs each);
- matmuls 177 µs, at 41 to 64 % of DRAM bandwidth (decode M = 32 on 22 to 87 cores);
- Sinkhorn 142 µs (2 calls of 71 µs on one core; the permutation matmuls and full-tile packs remain);
- sparse matmuls 123, mHC kernels 94, CCL 80, KDA recurrence 60, MoE dispatch 35, norms 17 µs;
- dispatch: 26 µs between device time and replay wall, over 121 ops.

## Method

- Decode A/B: `tools/mc_decode_ab.py` builds one real layer per (kind, batch). Every variant sets options at run time
  (`MCOpts` fields, `env.NAME=value`, `moe.FIELD=value`, `fid.GROUP=HiFi2`). It runs one eager step, captures the
  decode trace and times 30 back-to-back replays. `base` first and `base2` last give the run noise: 0 to 60 µs at 32
  users, under 3 µs at 1 user.
- A/B flags: `--units` compares options with the stage-4 path on one input and with an fp64 product. `--determinism
  N` replays 2 x N steps from a fresh state. `--compare N` (review #2) compares each variant with the first, bit by
  bit. `--base default` builds `DEFAULT_MC` (the shipped build).
- Op-level sweep (review #1 P1-2): `tools/mc_matmul_sweep.py` on one die with per-die TP4 shapes and random data
  (`records/mm6.jsonl`). It covers ttnn's default, 1D multicast and DRAM-sharded programs, each family's best at
  HiFi2, and prefill on the auto and explicit 2D configs. Only differences inside a role count: the replay wall
  includes a fixed per-trace overhead.
- Prefill A/B: `tools/mc_prefill_ab.py`, a warm 2048-row pass, best of 3, host wall. It reports output, update PCC and
  update rms ratio against the first variant.
- Tracy: `~/glm53-flash-bringup/scripts/s5_tracy.sh before|after`, device 0. Decode = one traced replay; prefill = the
  signposted pass. Times are sums of `DEVICE KERNEL DURATION` (`~/glm53-flash-bringup/scripts/s5_ops.py`).
- Provenance (review #2 / #3): final numbers come from a code snapshot outside the live tree. Review #2 used
  `~/glm53-flash-bringup/s5r2_snapshot`: the stage-8 snapshot of d48b7fd06ad (92 / 100 / 100) plus the stage-5 files.
  Review #3 used `~/glm53-flash-bringup/s5r3_snapshot` (HEAD cdab692 plus the review-3 edits) for tickets 1-4 and
  `s5r3h_snapshot` (HEAD b8198430147 plus the same edits) for the gate and ticket 6. md5s are in
  `records/snapshot_md5.txt`.
- Isolation: `PYTHONSAFEPATH=1` and pytest `--import-mode=importlib` keep the live tree off `sys.path`. Other stages'
  uncommitted live edits do not enter.
- Provenance (review #1 P2-4): round 6 ran on the worktree of 2026-10-17 with the stage-3 rework's `tt/moe_opt.py` row
  split (`SORTED_ROWS_DEFAULT = 1024`) for both the before and after prefill windows.

## Operation-topology audit (decode, per layer, per device)

Stage-4 path (`doc/multichip_decoder/tracy`) unless noted. Element-wise rows come from the final Tracy.

| Op sequence | Cost | Candidate | Constraint | Result |
|---|---|---|---|---|
| all-gather h -> RMSNorm [T, 4096] | 17 + 65 µs (1 core) | width-sharded L1 norm | fp32 accumulation | taken: 6 µs |
| DSA: 5 projections of the same x | 250 µs | 2 packed matmuls | bfp8 pack + bf16 pack | taken: ~60 µs |
| DSA q_a / kv_a norms | 25 + 11 µs (1 core) | sharded norm | - | taken |
| MoE router [T, 4096] x [4096, 192] | 49 µs (6 cores) | DS / 1D | fp32 routing | taken: 1D, 18 µs |
| row-parallel matmul -> reduce-scatter | 17 to 32 µs per RS | ring, L1 out, fused, bfp8 | - | ring + L1 taken |
| mHC mix partials -> sum -> all-gather -> sum | 42 + 13 µs per sub-block | `fast_reduce_nc`; fewer chunks | fp32-exact sums | 16 chunks taken |
| `mhc_split_sinkhorn` | 2 x 89 µs, 1 core | HiFi3; face-restricted SFPU | fp32-exact | taken: 71 µs per call |
| KDA decode conv (12 ops on [B, 6144]) | ~105 µs | one fused op | exact history | taken: 38 µs |
| KDA o-norm + gate | 16 µs | `sigmoid_gated_rms_norm` | T must be 32 rows | rejected (below) |
| MoE router glue (TopK 1 core 34 µs) | 85 µs | padded top-k; fused gate op | fp32 selection | rejected |
| FillPad (11 per layer) | 61 µs | wider tensors | exact reduce | mix FillPads shrink |
| MoE 32 users (gather, untilize, SwiGLU) | 1.2 ms glue | pairs; union; geometry | 32-row groups | in0 8 taken |
| DSA q path, 1 slot | 56 µs | one masked matmul | bit-identical | taken (`dsa_q_mm`) |
| DSA o path, 1 slot | 46 µs | one masked matmul | +2 MB per layer | not adopted |
| DSA sparse-index tail | 34 µs | once per step | positions only | taken (`dsa_tail_shared`) |
| DSA index expansion | 31 µs | order-free repeat | list order | rejected (`dsa_idx_rep` not exact) |
| DSA open-pool softmax | ~80 µs | fused fp32 kernel | exact state | open (new kernel) |
| DSA latent write untilize | 10 µs, 1 core | multi-core untilize | one tile row | not taken |
| KDA conv at 32 slots | 158 µs | unfused chain | bit-identical | fused up to 16 slots |
| inter-layer boundary | none | - | - | contract below |

Details of the DSA and KDA rows (review #2 / #3 P2-2):

- DSA q path, 1 slot, before (µs): `ReshapeView` of the q_b output to 16 heads 10.6, `Transpose` 3.3, per-head bmm
  W_UK 17.5, `FillPad` + `Pad` to 32 heads 5.8 + 9.6, `UntilizeWithUnpadding` 9.1.
- `dsa_q_mm` replaces this with one matmul. Its in0 is the q row replicated to 32 rows and masked per head. Its in1 is
  W_UK viewed as [16 x 256, 512]. q_abs comes out with heads as rows, already padded to 32 heads. The other heads' K
  blocks add exact zeros, so the result is bit-identical. Gain: -17 µs per DSA layer (`records/dsa_dm_ab.jsonl`).
- `dsa_q_mm` in0 block (review #3 P3-5): blocks of 16 / 32 / 64 tiles save 5 to 7 µs more. They stay
  bit-identical, because a block spans whole heads and the off-head K tiles add exact zeros. 64 is the default.
- DSA o path, 1 slot. Before: sparse_sdpa rows [16, 1, 512] -> `Slice` (1.8) -> `TilizeWithValPadding` (23.0) ->
  per-head bmm W_UV (14.7) -> `Transpose` + `ReshapeView` (2.6 + 3.9).
- `dsa_o_mm` 1 is one [16, 512] x [512, 16 x 256] matmul on a relaid W_UV, off-head blocks masked, rows summed. It
  saves 12 µs but needs +2 MB of relaid W_UV per DSA layer per die. `dsa_o_mm` 2 uses a W_UV view and costs +33 µs.
- Neither is bit-identical to the bmm (different bf16 rounding). Against fp64 both are equally exact: rel L2 1.70e-3
  vs 1.72e-3 (`tools/dsa_out_probe.py`). Not adopted: 12 µs per layer does not pay for 22 MB of DRAM per die.
- DSA sparse-index tail: start = 4 floor((t + 1) / 4); tail = start + j if <= t, else the sentinel. That is 8
  one-tile element-wise ops on 110 cores, 2 to 5 µs each. It depends only on the step's positions.
- `dsa_tail_shared` computes it once per step in the model's first DSA layer. The other DSA layers read it: -36 µs each at
  position 128 (consumer emulated with a host-written tail; `records/dense_ab.jsonl`). The buffer handling is in "Stacked gate".
- DSA index expansion. Before: `repeat_interleave` of the top-512 pool ids x 4 (`Typecast`, `Transpose`,
  `UntilizeWithUnpadding` 10.4, `Concat` 8.7 of 4 scalars, `TilizeWithValPadding` 5.4, `Transpose`).
- `dsa_idx_rep` (review #3 P2-5) builds the 2048 indices as repeat(ids, 4) x 4 + offsets laid out offset-major. That
  is a tile repeat along the width, with no untilize, tilize or transpose. The list order changes.
- In the 1-slot layer A/B it looked bit-identical at positions 128, 3000 and 130,000 and saved 25 to 27 µs. That
  check was weak: the A/B layer's cache holds only 128 written tokens, so most selected keys are equal zero rows.
- In the full model the index order changes sparse_sdpa's result. Sparse prefill logits changed (max |diff| 9.9),
  and with one decode row the decode logits changed (max |diff| 8.7, tokens diverge at position 3004;
  `records/s5r3.log`, tickets 2 and 4). At 32 users the layer A/B already showed PCC 0.99919 and +157 µs.
- Rejected: the option is not exact. A full-model teacher-forcing run with it in the default gave 14 / 32 / 90
  (ticket 4). That is also the signature of the open prefill-to-decode race the stage-6/7 work is chasing, so that
  run is unclassified.
- DSA open-pool online softmax (the pooled index key): ~14 `BinaryNg` / `Unary` / `Typecast` ops on [1, 128] fp32, plus
  3 `Ternary` + `Copy` state writes, ~80 µs. There is no RoPE in the indexer; review #2's "indexer RoPE chain" is
  this update. It needs a new fused fp32 kernel: open.
- KDA o-norm + gate: `rms_norm` 4.5, typecast 2.0, reshape 3.0, gate multiply 4.7, typecast 1.9 µs.
  `sigmoid_gated_rms_norm` needs whole 32-row tiles of T, so each (slot, head) row becomes its own padded tile. It ran
  +28 µs (kda_moe 1 user), +52 (kda_dense) and +228 (32 users), and gave wrong output on kda_moe. Rejected.
- KDA decode conv: the fused op is bit-identical to the unfused chain at 1, 8, 16 and 32 slots. It is faster up to 16
  slots (-78 / -62 / -28 µs) and slower at 32 (+53 µs, `records/kda_conv_cmp.jsonl`).

## Candidates and decisions

Replay wall in µs (`tools/mc_decode_ab.py`) against `fin` (the previous default) or `base` of the same build.
Records: `records/ab{1,2,3,5,6}.jsonl`, `records/pf{2,5,6}.jsonl`, review #2 / #3 records named in each row.

| Option (`MCOpts`) | Measurement | Precision | Decision |
|---|---|---|---|
| `norm_sharded` | dsa_moe 1 user 1806 -> 1684 | PCC 0.999999 | taken |
| `dsa_pack` | 1806 -> 1686 | = separate matmuls | taken |
| `pack_a_1d` | DS 1487, 1D 1436 | 1D = separate | 1D |
| `pack_b_1d` | DS 1548, 1D 1519 | 1D = separate | 1D |
| `router_1d` | DS 1222, 1D 1224 | 1D exact | 1D |
| `topology` ring | -23 µs; prefill -0.5 to -1 ms | AG identical; RS order | `auto` |
| `num_links` | 1821 (1) vs 1806 (2) | identical | 2 |
| `ccl_l1` | -9 µs (1 user), -86 (32) | identical | taken |
| `ag_persistent` | -9 to -71 µs per layer | 1-layer gates pass | rejected (below) |
| `mix_nc` | -28 µs | 4.3e-4 vs 7e-8 | rejected |
| `ds_shared` | prefill dsa_moe 33.8 -> 32.8 ms | bit-identical | taken |
| `dsa_1d` | 1436 -> 1432 | = default | taken |
| `prefill_2d` | prefill dsa_moe 33.35 -> 31.39 ms | PCC 1.0 | taken |
| 1-slot KDA padded inputs | 1398 -> 1371 | identical | taken |
| `sinkhorn_perm_fidelity = 3` | 89.4 -> 85.8 µs per call | bit-identical | taken |
| `kda_conv_op` | kda_moe 1214 -> 1129 | qkv 1.2e-7; history exact | taken |
| `kda_conv_mixed` | - | `TT_FATAL` (addcmul dtypes) | blocked |
| `mix_chunks_decode` 16 | 1214 -> 1204 | exact sum | 16 |
| `router_topk_pad` | +38 µs | identical | rejected |
| role programs | -4 to -34 µs per role | same rounding | taken |
| KDA in-projection 1D | in layer 1214 -> 1216 | - | rejected |
| `moe_grouped_in0 = 8` | 32 users -3 to -36 µs; prefill -0.4 ms | PCC 1.0 | taken |
| `prefill_2d_ffn`, `packs_prefill_2d` | kda_dense 12.22 -> 10.49 ms | PCC 0.999999+ | taken |
| HiFi2 | see "Fidelity" | systematic scale loss | rejected |
| `sinkhorn_faces` | -30 µs (1 user), -26 to -43 (32) | bit-identical | taken |
| `dsa_q_mm` | -17 µs per DSA layer | bit-identical | taken |
| `dsa_tail_shared` | -36 µs per later DSA layer | bit-identical (sparse regime too) | taken |
| `dsa_o_mm` 1 / 2 | -12 / +33 µs | other rounding | not adopted |
| `kda_gnorm_op` | +28 to +228 µs | wrong on kda_moe | rejected |
| `kda_conv_max_batch = 16` | -78 / -62 / -28 / +53 µs at 1 / 8 / 16 / 32 | bit-identical | 16 |
| `dsa_idx_rep` (review #3) | -25 to -27 µs (1 user); +157 (32) | not exact in the full model | rejected |
| `dsa_q_mm_ib` (review #3) | -5 to -7 µs vs block 8 | bit-identical | 64 |

`ag_persistent` (review #1 P2-1) was taken in round 6 and reverted on 2026-10-18:

- One-layer gates all passed: 2 x 120 replays bit-identical, the 200-replay stress, the suite and the watcher.
- The stacked model did not. The reduced 0 / 3 / 4 model ran with 1 of 8 slots active
  (`records/s7_det_reduced_s8.log`). Repeated runs gave different decode tokens.
- s7_probe's "eager" / "traced" labels select the prefill path only. Decode is always traced. Both prefill paths
  diverged on repeat.
- 45-layer 1-slot teacher forcing fell to 14 / 32 / 90.
- With a persistent output, ttnn skips the all-gather's start barrier (`all_gather_async_default_program_factory.cpp:752`:
  "use synchronize barrier semaphore" = `barrier_semaphore && !using_persistent_buffers`). Nothing then orders a remote
  write into a buffer after its last reader on the receiving die.
- The one-layer tests rewrite each buffer at the same distance as the stack, so reuse distance does not explain the
  difference. The root cause is not established. The final default keeps the barrier (non-persistent outputs).

## Matmul geometry per role (review #1 P1-2)

`tools/mc_matmul_sweep.py`, decode M = 32, µs per call (replay wall; compare within a row only). Best DS = best
DRAM-sharded geometry over activation cores x in0 block x readers.

- "c / ib" in a spec is the core budget and in0 block passed to `tp_common.create_matmul_1d_decode_progcfg`. That
  function shapes the grid 11 cores wide; the real programs are in the second table.
- The "default" column is the harness's own `ttnn.linear` program on one die, not the in-model program. For the KDA
  in-projection, the layer runs 99.6 µs on 87 cores (Tracy) against 162 µs in the harness.
- Every kept role was decided by the in-layer A/B (Candidates), not by this column.

| Role [K x N], dtype | Default | Best 1D | Best DS | HiFi2 | Final |
|---|---|---|---|---|---|
| KDA in [4096 x 8336] bfp8 | 162.1 | 109.9 (29, 4) | 140.1 (32, 2, 3 r.) | 110.3 | default |
| KDA out [2048 x 4096] bfp8 | 48.2 | 33.7 (16, 4) | 45.4 (8, 4, 2 r.) | 33.3 | 1D 16 / 4 |
| DSA o [4096 x 4096] bfp8 | 84.0 | 57.5 (16, 4) | 76.1 (16, 4, 2 r.) | 56.8 | 1D 16 / 4 |
| DSA q_b [1536 x 4096] bfp8 | 39.4 | 27.9 (16, 4) | 38.2 (8, 3, 2 r.) | 27.5 | 1D 16 / 4 |
| DSA wq_b [1536 x 4096] bf16 | 43.6 | 41.2 (16, 4) | 55.2 (8, 6, 1 r.) | 40.6 | 1D 16 / 4 |
| pack A [4096 x 2048] bfp8 | 82.4 | 34.8 (32, 8) | 45.5 (4, 8, 2 r.) | 34.2 | 1D 32 / 8 |
| shared gate/up [4096 x 1024] bfp8 | 66.8 | 23.8 (32, 16) | 29.5 (4, 8, 2 r.) | 23.3 | 1D 32 / 16 |
| shared down [512 x 4096] bfp8 | 21.3 | 16.1 (16, 4) | 22.4 (4, 4, 2 r.) | 15.6 | 1D 16 / 4 |
| dense gate/up [4096 x 6144] bfp8 | 104.4 | 84.0 (16, 4) | 105.8 (32, 2, 3 r.) | 83.0 | 1D 16 / 4 |
| dense down [3072 x 4096] bfp8 | 66.1 | 45.6 (16, 4) | 61.1 (8, 6, 2 r.) | 45.2 | 1D 16 / 4 |

Final 1D programs (OPT-004: real grid, `per_core_N`, output subblock, in0 block). Tracy agrees: tt-perf-report shows
"in0_block_w=4 and output subblock 1x3" on the 22-core rows.

| Role (spec) | Grid (cores used) | per_core_N | Subblock | in0 block |
|---|---|---|---|---|
| KDA o, DSA o / q_b / wq_b, shared down, dense down (`1d:c16:ib4`) | 11 x 2 (22) | 6 | 1 x 3 | 4 |
| dense gate/up (`1d:c16:ib4`) | 11 x 2 (22) | 9 | 1 x 3 | 4 |
| pack A q_a / kv_a (`1d:c32:ib8`) | 11 x 3 (32) | 2 | 1 x 2 | 8 |
| shared gate/up (`1d:c32:ib16`) | 11 x 3 (32) | 1 | 1 x 1 | 16 |
| DSA q_abs (`dsa_q_mm`, [4096 x 512]) | 11 x 2 (16) | 1 | 1 x 1 | 64 |

Core budgets that round to the same `per_core_N` give the same program (e.g. c16 and c22 for N = 128 tiles). Some
sweep candidates are therefore duplicates; the table lists each distinct program once.

DRAM-sharded never beat the best 1D program at these shapes, with any activation geometry or reader count. in0 blocks
above 8 helped only the shared gate/up (16). Every DS candidate carried 2.5x to 4x the output error of the 1D one: its
K-block partials are re-read through the FPU source registers. Refused geometries (L1 / CB) are in `records/mm6.jsonl`.

Prefill (2048 rows), µs per call: ttnn auto -> explicit 2D (best in0 block) at HiFi4 / HiFi2:

| Role | Auto | 2D HiFi4 | 2D HiFi2 | Final |
|---|---|---|---|---|
| KDA out (fp32 in) | 1098 | 598 (in0 8) | 431 | 2D HiFi4 (`prefill_2d_ffn`) |
| DSA o | 1982 | 990 (8) | 661 | 2D HiFi4 (`prefill_2d`) |
| DSA q_b | 794 | 399 (8) | 275 | 2D HiFi4 (`prefill_2d`) |
| pack A | 1057 | 515 (8) | 343 | 2D HiFi4 (`packs_prefill_2d`) |
| shared gate/up | 536 | 269 (8) | 188 | 2D HiFi4 (`prefill_2d_ffn`) |
| shared down | 306 | 161 (8) | 130 | 2D HiFi4 |
| dense gate/up | 3075 | 1848 (4) | 1320 | 2D HiFi4 |
| dense down | 1499 | 764 (8) | 508 | 2D HiFi4 |
| KDA in | 3904 | upstream tuned config | - | unchanged |

## Sparse-matmul geometry at TP4 (review #1 P1-2)

The grouped sparse matmuls came from single-die tuning: `in0_block_w = 4`, subblock 1 x 8 capped at 4 by fp32
accumulation. They serve the pairs path at 1 user and the sorted path at 32 users and in prefill. TP4 sweep
(`moe.grouped_*`), replay µs / prefill ms:

| Geometry | kda_moe 1 user | kda_moe 32 users | dsa_moe 32 users | prefill kda_moe 2048 |
|---|---|---|---|---|
| in0 4 (stage 3) | 1214 / 1213 | 4422 / 4364 | 4751 / 4752 | 25.48 / 25.42 |
| in0 2 | 1213 | 4484 | - | 26.03 |
| in0 8 | 1213 | 4361 | 4715 | 25.09 |
| in0 16 | - | 4380 | - | 25.23 |
| subblock w 4 | - | - | - | 25.31 |

in0 8 is selected (`moe_grouped_in0 = 8`).

## Fidelity (review #1 P1-1)

- Decode matmuls are DRAM-bound. HiFi2 saved 0.4 to 1.1 µs per call and adds the truncation bias stage 3 measured.
- Prefill matmuls are compute-bound; HiFi2 is 25 to 35 % faster per call. On real weights (`tools/mc_prefill_ab.py`,
  update rms vs HiFi4) it costs scale: dense MLP 0.99735 for -0.7 ms; shared expert 0.99964 for -0.3 ms; the DSA
  module 0.959 (PCC 0.9965).
- The explicit 2D config at HiFi4 gives most of the gain without the bias (dense layer 12.2 -> 10.5 ms).
- HiFi2 is rejected for every group. The datatype sweep (stage 8) can revisit it against model accuracy.

## Collectives (OPT-009 fields)

Six per layer, `ttnn.experimental.*_async` (`MeshCCL`): `chunks_per_sync` 10, `num_workers_per_link` 2,
`num_buffers_per_channel` 2, 2 links, ring.

| Collective | Payload per device | Dtype | Output | Persistent | kda_moe 1 user µs |
|---|---|---|---|---|---|
| all-gather of the mix partials, dim 1 | [T, 64] -> [4, T, 64] | fp32 | L1 (decode) | no | 12.0 -> 11.0 |
| all-gather of h, dim 3 | [T, 1024] -> [T, 4096] | bf16 | L1 (decode) | no | 17.8 -> 12.6 |
| reduce-scatter, dim 3 | [T, 4096] -> [T, 1024] | bf16 | L1 (decode) | no | 16.2 -> 16.7 |

- Times: stage 4 -> final, the mean of the 2 calls per layer in `tracy/after/test_perf_traced_decode_kda_moe-1_`.
- Prefill outputs stay in DRAM. The reduce-scatter's intermediate is in DRAM.
- `ag_persistent` (rejected, see Candidates) measured 7.1 / 10.8 / 15.0 µs in round 7.
- A persistent reduce-scatter output gave no gain in the 122B port (`~/qwen35-122b-bringup/logs/05-r2-ccl-probe.log`).
- Single calls vary: one 32-user reduce-scatter took 56.6 µs in the final kda_moe-32 Tracy (the other 19.4).

Ring on a FABRIC_1D fabric:

- `ttnn.get_usable_topology` returns Ring only when the fabric graph has the wraparound link (both p300 board-to-board
  links up). Otherwise it returns Linear.
- FABRIC_1D_RING adds dateline deadlock avoidance for traffic that crosses the wrap. The decode collectives are
  latency-bound single-round transfers (one chunk per direction per link).
- Decode A/B on the round-7 build (`records/ring_fabric7.jsonl`, `ag_persistent` on; 1 user, replay µs, ring fabric vs
  1D fabric): kda_moe 1104.8 vs 1103.1, dsa_moe 1407.6 vs 1407.0. The difference is noise (< 0.2 %).
- Prefill on the review-2 default (`records/pf_ring_r2.jsonl`, host wall, best of 3): kda_dense 11.83 -> 11.52 ms,
  kda_moe 27.70 -> 28.60 ms, dsa_moe 31.95 -> 32.46 ms on FABRIC_1D_RING. Outputs are bit-identical. The +-3 % is
  the run-to-run range of these walls.
- The decoder does not pick the fabric; the model opens the mesh (stage 6 / 7). The layer tests use `GLM53_FABRIC`
  (`tests/harness.py`). Recommendation for the model owner: FABRIC_1D_RING, which settles the deadlock question for
  multi-chunk prefill at no measured cost. Nothing has hung on FABRIC_1D in this stage.

Rejected collective families, measured on this board by the 122B port (same ops and sizes):

- fused all-gather + matmul: 0.14 to 0.60 ms vs 0.07 ms separate (`~/qwen35-122b-bringup/logs/05-agmm-probe*.log`);
  here an RMSNorm also sits between the all-gather and every consumer;
- fused matmul + reduce-scatter: 2D configs only, <= 8 compute cores for one tile of rows (`05-mmrs-probe*.log`);
- bfp8 payloads: -1.4 / -2.6 µs plus a typecast (`05-r2-ccl-probe.log`);
- all-reduce instead of reduce-scatter: 4x the bytes and the mHC work on every die.

## Stacked gate (review #2 P2-1, review #3 P1-1)

Single-layer gates (suite, replay stress, determinism, watcher) do not cover hazards that only appear with many layers
or several captured traces. Two such hazards appeared in this stage:

- `ag_persistent`: every one-layer gate passed; the stacked model was not repeatable (see Candidates).
- `dsa_tail_shared`, first version: the step tail lived in the step's input dict as a fresh tensor, and the producer
  freed the previous one. The generator captures several decode traces, so a later capture freed a buffer that an
  earlier trace still read. Every single-layer A/B was bit-identical (one trace); teacher forcing fell to 10 / 48 / 100.
- `dsa_tail_shared`, review-2 version: one buffer per input dict, created with `ttnn.clone` on the first call. A
  generator that hands the layers a new dict during a later warm-up or capture re-creates it, and dropping the old dict
  frees a buffer that earlier traces read. The stage-6/7 generator's first run on cdab692 showed the same signature
  (10 / 48 / 100); this path is the likely cause.
- `dsa_tail_shared`, review-3 version: the buffer is allocated once, when the model's first DSA layer is built, before
  any warm-up or capture. It lives in a module registry keyed by mesh and slot count and is never freed or replaced.
  The producer writes it in place and marks the step dict with a reference to it; consumers read it through that mark.

Every `DEFAULT_MC` change, collective option or buffer-sharing option must pass, before it becomes the default. The
live tree's `DEFAULT_MC` changes only after the gate has passed on a snapshot: in review #3, `dsa_idx_rep` sat in the
live default for about 4 hours before its full-model check failed.

1. Full-model teacher forcing, 1 slot (`tools/s7_probe.py --slots 1 --do ready`, the stage-7 reference), at the stage-7
   level (92 / 100 / 100). Code comes from a snapshot outside the live tree.
2. Stacked repeatability, 8 slots (`tools/s7_probe.py --slots 8 --do ptcheck`). Decode is always traced; each prompt
   runs twice with each prefill path, and the tokens must repeat. The run covers a reduced stack with a producer and a
   consumer DSA layer (layers 0, 3-7) and the full model.
3. The sparse regime (review #3): full-model decode at positions above 2050
   (`tools/s5_sparse_probe.py --lengths 2100,3000`). Prefill logits, every decode step's logits and the tokens must
   be equal with and without the option.

Results:

| Run | Options | Teacher forcing top-1 / 5 / 100 | Prefill check | Other |
|---|---|---|---|---|
| review #2 control | round-6 default, review-2 files | 92 / 100 / 100 | 91 / 100 / 100 | - |
| review #2 final | + the 4 review-2 options (`GLM53_MC`) | 92 / 100 / 100 | 91 / 100 / 100 | 8 slots repeat; tokens = stage 7 |
| first `dsa_tail_shared` | same | 10 / 48 / 100 | 91 / 100 / 100 | - |
| review #3 final (ticket 5) | `DEFAULT_MC`, no override, combo_a_kv precision | 91 / 100 / 100 | 91 / 100 / 100 | sparse regime equal with / without the tail and with in0 block 8 |
| review #3, `dsa_idx_rep` in the default (ticket 4) | stage-7 precision | 14 / 32 / 90 | 91 / 100 / 100 | unclassified (see the anomaly ledger) |
| stage 7, `ag_persistent` | round-6 default | 14 / 32 / 90 | - | 8 slots do not repeat |

## Inter-layer residual contract (for the full model)

- Input and output of every layer: the 4-stream mHC state X [1, 1, T, 4 x 4096] fp32, stream-interleaved and
  hidden-sharded. Device d holds columns [1024 d, 1024 (d + 1)) of every stream, DRAM interleaved, TILE.
- No collective sits between layers.
- Decode rows: one tile of users (B <= 32). With B = 1 the KDA layers pad to 2 slots internally. Every prefill chunk
  is a multiple of 128 rows, so 32 padded rows only ever mean decode (`_decode_rows`).

## 1-slot KDA decode

`kda.recurrent_decode` needs batch x local heads % 32 == 0. TP4 leaves 16 heads, so a 1-slot decoder failed in stage 4.
`MultichipKDA` allocates 2 state slots and per-slot inputs for 2 slots (the second inactive) and pads x in place.
`test_single_slot_kda_decode`: 1-slot vs 2-slot bit-identical.

## Sinkhorn (review #1 P1-1, review #2 P1-2)

`mhc_split_sinkhorn` splits token tiles over cores (prefill: 64 cores at 2048 rows). Decode is one tile on one core:
40 sequential normalizations, each of 8 permutation matmuls, an fp32 reciprocal, 2 multiplies and 3 packs. Splitting
users over cores cannot help, because the tile is the SFPU / FPU work unit.

HiFi3 permutation sums (`sinkhorn_perm_fidelity = 3`):

- The permutation operand is 0 / 1. The 4th fidelity phase (low srcA bits x low srcB bits) therefore only adds zeros,
  and HiFi3 is bit-identical to HiFi4 for any input. Verified at T = 1 to 2048, sharded and interleaved.
- HiFi2 is not exact (0.5 to 1.2 % error). Gain: 89.4 -> 85.8 µs per call (Tracy).

Face-restricted normalizations (`sinkhorn_faces`, review #2):

- The n x n = 16 used columns of m are columns 0-15. The SFPU ops of every normalization (eps add, fp32 reciprocal,
  normalizing multiply, the hi / lo split) run on face 0 only (T <= 16: `VectorMode::None`). Above 16 tokens they run
  on faces 0 and 2 (`VectorMode::C`).
- The padding of m is zeroed once, after the exp. The split writes exact zeros to the unused faces of m_hi / m_lo, so
  no stale DST value reaches a permutation matmul. Every padding product is 0 x finite = 0.
- The permutation tiles map the 16 used columns among themselves and each padding column only to itself. The padding
  of m therefore stays exactly 0; comb's padding comes out 0 (it was ~1).
- The used entries are bit-identical. Probe: `tools/sinkhorn_fidelity_probe.py` at T = 1 / 8 / 16 / 17 / 32 / 96 /
  2048, repeat calls identical (`records/s5r2_sk_probe.log`).
- Test: `tests/test_mhc_split_sinkhorn.py::test_mhc_split_sinkhorn_faces` (review #3 P2-2) asserts `torch.equal` on the
  used entries, faces on vs off, at T = 1 / 8 / 16 / 17 / 32, and a repeat call.
- In the layer, the option is bit-identical over 9 steps (`records/dense_ab.jsonl`, `sparse_ab.jsonl`). Tracy per call: 85.8 -> 70.8 µs at 1 user (face
  0), 85.9 -> 76.6 µs at 32 users (faces 0 / 2). Replay: -30 µs per layer at 1 user, -35 to -43 µs at 32 users.

What stays: the 8 permutation matmuls and the 3 packs per normalization still move whole 32 x 32 fp32 tiles.
A 16-row form was not attempted in this stage. It needs 16 x 32 tiles for the m buffers, the permutation matmul on a
16-row in0 and the packer on the smaller tile. The matmul factories already support in0 tiles with fewer than 32
rows; this is the next kernel step.

## 32-user MoE (review #1 P2-6)

Lower-padding forms measured at 32 users (`records/ab6.jsonl`):

- the pairs path (one 32-row tile per (token, expert) pair, 256 groups): 6788 / 6883 µs vs 4364 / 4751, because it
  re-reads one expert block per pair;
- the masked union path: 4383 µs (no gain);
- grouped in0 blocks of 8: taken.

The remaining glue (gather 620, untilize 260, SwiGLU 362 µs) comes from the 32-row tile of an expert group with ~1.8
real rows. A smaller group needs a sub-tile (16- or 8-row) sparse-matmul and gather path, which the grouped factory
does not have. The down projection cannot write row-major output at TP4: its resident block (1.7 MB) exceeds the
factory's 1.3 MB L1 budget.

## KDA prefill chunk terms (review #1 P2-7)

Every TP4 KDA prefill of 2048 rows logs "KDA chunk terms need 595 KiB per core ... keeping them in DRAM".
1024-row chunks fit L1 (298 KiB). Two of them are slower than one 2048-row chunk: kda_dense 2 x 6.85 = 13.7 ms vs 12.2 ms, kda_moe 2 x 13.27 =
26.5 vs 25.5 ms (`records/pf6_1024.jsonl`). The DRAM fallback at 2048 rows is kept.

## Memory and context contract (review #1 P2-3)

`DEFAULT_MC` adds per die:

- the 1-slot KDA padding slot when `max_batch = 1`: +1 MiB of recurrent state per KDA layer, 34 MiB in all. That is
  ~2.9K tokens of DSA cache at 11,968 B per token;
- one `dsa_q_mm` head mask ([32, 4096] bf16, 0.25 MB), shared by all DSA layers (review #3); W_UK is a view;
- one shared sparse-index tail tile ([1, 1, S, 32] fp32) per model;
- nothing else: the packs replace the separate weights, and there are no persistent collective buffers.

The DSA latent cache cannot drop to bfp8 here. `sparse_sdpa` takes a ROW_MAJOR bf16 or FP8_E4M3 KV
(`sparse_sdpa_device_operation.cpp:71,117`), and bfp8 is a tile format. FP8_E4M3 (3 mantissa bits) needs both
`scatter_rows` write paths cast to FP8 and a model-accuracy check: a stage-8 item. The sequence-sharded latent that
reaches the advertised context is the stage-4 rework's `MCOpts.dsa_shard`. `doc/context_contract.json` records this.

## Rejected and not adopted

| Candidate | Evidence | Reason |
|---|---|---|
| multi-core Sinkhorn | one tile holds every user | see Sinkhorn |
| `fast_reduce_nc` mix sums | 4.3e-4 vs 7e-8 relative | fp32-exact mHC |
| DRAM-sharded decode matmuls | geometry sweep | slower, less exact |
| HiFi2 | Fidelity | systematic scale loss |
| KDA in-projection 1D | in layer 1214 -> 1216 | no gain |
| padded router top-k | +38 µs | slower |
| fused MoE gate op | - | bf16 scores |
| `sigmoid_gated_rms_norm` (KDA) | +28 to +228 µs | slower; wrong on kda_moe |
| `dsa_o_mm` | -12 µs / +33 µs | +2 MB per layer per die |
| `ag_persistent` | -9 to -71 µs per layer | breaks the stacked model |
| pairs / masked MoE at 32 users | +2.4 ms / +19 µs | slower |
| 1024-row KDA prefill chunks | +1.0 to 1.5 ms per 2048 tokens | slower |
| `dsa_idx_rep` | full-model logits change (max 8.7 decode, 9.9 prefill) | not exact |
| tt-perf-report "Output subblock 1x1" | one output tile per core is the widest 1D grid | - |
| tt-perf-report "place input 0 in L1" | collective outputs moved (`ccl_l1`) | the rest are 64-256 KB |

## Anomaly ledger

| Anomaly | Evidence | Status |
|---|---|---|
| `ag_persistent` breaks the stacked model; every 1-layer gate passes | `records/s7_det_reduced_s8.log`; TF 14 / 32 / 90 | controlled: off; cause not established |
| first and review-2 `dsa_tail_shared`: TF 10 / 48 / 100 | `records/s5r2_tf_cand.log`; the 6/7 agent's run | fixed: build-time buffer (Stacked gate) |
| baseline compare: one kda_moe decode row on a router near-tie | `records/baseline8_rows.txt` | controlled: capped allowance |
| KDA in-projection: -52 µs in the harness, 0 in the layer | `records/mm6.jsonl`, `ab6.jsonl` | explained: different baseline program |
| idle slot's NaN conv history comes back Inf | `records/s5r_kernel_tests.log` | benign: non-finite either way, masked |
| 32-user Tracy varies up to 5 % | three Tracy runs | DRAM-bound sparse matmuls; replay is stable |
| watcher: teardown abort, forms (i) / (ii) on eth core 29-25 | `records/s5r2_watcher.log`; 122B TALLY.md | known board behaviour after PASSED |
| `kda_gnorm_op`: PCC 0.08 vs the unfused path on kda_moe (0.9999999 on kda_dense) | `records/kda_gn_ab.jsonl` | rejected candidate; not diagnosed |
| KDA prefill host wall +10-13 % in the ring-fabric runs vs round 6, device time unchanged (10025 vs 10084 µs) | `records/pf_ring_r2.jsonl`, Tracy | host contention on the shared machine |
| `dsa_idx_rep` passed the 1-slot layer A/B but changes full-model logits | `records/s5r3.log` tickets 2 / 4 | rejected; the A/B cache was mostly empty |
| teacher forcing 14 / 32 / 90 with `dsa_idx_rep` in the default (ticket 4) | `records/s5r3_tf_final2.log` | same signature as the open prefill-to-decode race; unclassified |

## Open items

These are measured costs that stay. None is a deferred option of this stage.

- Sinkhorn after `sinkhorn_faces`: 71 µs per call at 1 user, 2 calls per layer. The 16-row matmul / pack form was not
  attempted (see Sinkhorn).
- Element-wise glue that needs new fp32 kernels: MoE router glue (TopK on 1 core 34 µs, FillPads), mHC glue (~120 µs
  per layer), the DSA open-pool online softmax (~80 µs per DSA layer).
- 32-user MoE glue: sub-tile expert groups (kernel work in the grouped sparse-matmul factory).
- Long-context decode: +561 µs per DSA layer at position 130,000 (sparse_sdpa over 2,048 gathered tokens, top-512
  selection over ~32.5K pooled keys); profiled, not optimized here.
- `ag_persistent`: the root cause of the stacked-model failure is not established.

## Artifacts

- Code: `tt/multichip_decoder.py` (`MCOpts`, `DEFAULT_MC`, `MeshCCL`, `DSMatmul`, `IL1DMatmul`, `RoleMatmuls`,
  `sharded_rms_norm`, `MultichipDSA`, `MultichipKDA`, `MultichipMLP`, shared-expert and router hooks).
- Hooks: `tt/fused_decoder.py` (`_in_proj`, `_conv_step`, `_gated_norm`), `tt/moe.py` (collective hooks, top-k pad).
- Tests: `tests/harness.py` (`GLM53_FABRIC`), `tests/test_multichip_decoder.py::test_single_slot_kda_decode`,
  `tests/test_kda_decode_conv.py`, `tests/test_mhc_split_sinkhorn.py::test_mhc_split_sinkhorn_faces`.
- Tools: `tools/mc_decode_ab.py` (`--compare`), `tools/mc_prefill_ab.py`, `tools/mc_matmul_sweep.py`,
  `tools/sinkhorn_fidelity_probe.py`, `tools/dsa_out_probe.py`, `tools/s5_sparse_probe.py`.
- ttnn: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/kda_decode_conv/` (new op),
  `deepseek_prefill/mhc_split_sinkhorn/` (`perm_fidelity`, `faces`), `deepseek_prefill/CMakeLists.txt`,
  `experimental/experimental_nanobind.cpp`.
- Tracy + tt-perf-report (with advice, `--active-experts 8`): `tracy/before/<case>/`, `tracy/after/<case>/`,
  `tracy/after_pos130000/`; the raw op list as `ops.csv.xz`.
- Watcher: `watcher/<case>/`, `watcher_noeth/<case>/`.
- Records: `records/` (A/B jsonl, sweep, stress, round logs, suites, watcher logs, gate logs).
- Commands:
  ```
  GLM53_TP=4 pytest models/autoports/zai_org_glm_5_3_flash/tests/test_multichip_decoder.py
  pytest models/autoports/zai_org_glm_5_3_flash/tests/test_mhc_split_sinkhorn.py models/autoports/zai_org_glm_5_3_flash/tests/test_kda_decode_conv.py
  GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.mc_decode_ab --kind dsa_moe --batch 1 --base default \
      --compare 8 --pos 3000 --variants "fin:dsa_tail_shared=0;tail:dsa_tail_shared=1" --out ab.jsonl
  ~/glm53-flash-bringup/scripts/s5_tracy.sh after
  # full-model gate, from a snapshot directory (PYTHONSAFEPATH=1 keeps the live tree off sys.path)
  GLM53_TP=4 GLM53_CACHE_DIR=~/.cache/glm53_flash python -m models.autoports.zai_org_glm_5_3_flash.tools.s7_probe \
      --slots 1 --seq-len 4096 --do ready --reference <readiness_aime24_chat.refpt>
  GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.s5_sparse_probe --lengths 2100,3000 --save a.pt
  ```
