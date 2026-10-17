# Fused decoder work log (stage 2)

- 2026-10-14 Op survey (`~/glm53-flash-bringup/artifacts/stage2_op_survey.md`, Explore subagent): no per-key-decay fused
  recurrence anywhere; main has the GDN (per-head decay) `fused_recurrent_gated_delta_rule` -> forked.
- 2026-10-14 `mhc_pre` op (tests pass). `rms_norm_pre_all_gather` for the mHC sum of squares rejected (CB size 8.5 MB).
- 2026-10-14 `kda.recurrent_decode` op (port of main's GDN kernel + per-key decay + in-kernel L2 norm + no-relayout
  row gathers + in-place state): 5/5 unit tests on the first build. Fused real-weight / decode-only cases pass.
- 2026-10-15 `mhc_mix` split-K op (projection + Gram partials). Bug found and fixed: non-tile-aligned row counts
  (decode batch 1) -> validation on the padded shape, per-row-count diagonal masks created at setup.
- 2026-10-15 Tracy before / after (`scripts/s2_tracy.sh`), table in README. Decode is now weight-bound.
- 2026-10-15 Bug found by the fused suite: mhc_mix finalize multiplied the Gram tile by the diagonal mask; the Gram tile also mixes the activation's padding rows (non-finite leftovers after mhc_post in the DSA layer) -> 0 * inf = NaN in the row sum (dsa_moe decode PCC 0.44). Bisected with tools/fused_bisect.py (mhc_mix off: pass). Fix: ttnn.where(mask, gram, 0) select; dsa_moe decode >= 0.99992.
- 2026-10-16 Stage review #1 (`artifacts/stage2_review_1.md`): more-work-needed. Fusing items a-e and the stale DSA
  Tracy taken up here; the `recurrent_decode` TF32 state item and the mHC kernels went to the stage-1 review #4 work.
- 2026-10-16 `tools/fusing_probe.py` (`s2r_fusing_probe{,2,3}.log`): DSA head split / concat ops win at prefill rows
  (9966 -> 392 us split, 7410 -> 1043 us concat at 2048 rows), lose at one tile of decode rows (single core). Bug on the
  way: `nlp_concat_heads` at 64 heads x 256 overflows L1 (2.2 MB) -> groups of 16 heads. Packed dense MLP + folded
  activations 17.2 -> 8.2 ms at 2048 rows; MoE activation fold 1307 -> 665 us at 32 users; KDA decode conv / shift /
  gate folds. Rejected by measurement: `sigmoid_gated_rms_norm` at decode (permute costs more), main's packed
  `qkv_causal_conv1d_silu` (663 vs 270 us). Anomaly: a multiply with a sigmoid input activation on a bf16 operand and
  an fp32 operand gave PCC 0.91; with the gate typecast to fp32 first it is exact (used).
- 2026-10-16 Flags `dsa_heads_op`, `mlp_act_fold`, `kda_decode_fold` (default on). `tools/fused_flag_check.py`: L0 / L3
  / L4 prefill + decode bit-identical (`s2r_flag_check.log`).
- 2026-10-16 Fused Tracy re-capture (`s2r_tracy_dsa.log`, `s2r_tracy_prefill_redo.log`): the 20 ms DSA replay gaps do
  not reproduce (0.28 ms). Mistake: gzipping the new `ops.csv` files lost the raw prefill CSVs (tracked as plain
  files); the six prefill cases were captured again.
- 2026-10-16 21:28 fp32-exact kernels swapped in (stage-1 review #4, b57f7da0b59): `recurrent_decode`, `mhc_post`,
  `mhc_pre`, `mhc_split_sinkhorn`, `mhc_mix`. The six fused decode captures of 21:15-21:27 ran on an intermediate
  Sinkhorn kernel (60 us instead of 89.5 us); recaptured in review #2.
- 2026-10-17 00:38 Fused suite on the final kernels (`s2_fused_suite5.log`): 43 / 43. New near-tie token L3 slot 2
  pos 38 (delta PCC 0.98296, HF margin 7.97e-4).
- 2026-10-17 Stage review #2 (`artifacts/stage2_review_2.md`): more-work-needed (five fusing items, fresh decode Tracy,
  watcher correctness run, docs, mhc_mix padded-row tests).
- 2026-10-17 `fusing_probe.py finalize / relayout` (`s2r2_probe.log`): row-major relayout at 32 decode rows (KDA o
  129 -> 56 us, DSA split / concat 187 / 186 -> 52 / 63 us), finalize scale-chain fold (3 ops fewer), sharded RMSNorm
  66 -> 33 us (handed to stage 3/4: sharding, not bit-identical). Rejected: `moreh_sum` (no fp32), grouped
  `nlp_concat_heads` at decode rows (124 us), `untilize_with_unpadding` for the KDA prefill qkv (hung the device; the
  probe hit its 1800 s timeout and the board was reset).
- 2026-10-17 `tests/test_mhc_mix.py::test_mixes_split_k_padded_rows` (T 1 / 8, NaN / inf padding rows through
  `_mixes_split_k`): pass, bit-identical to zero padding (`s2r2_mhc_mix.log`). Signed sum-of-squares error fused
  -2.3e-4 vs functional composite -5e-8.
- 2026-10-17 `tools/route_flip_check.py --fused` (`flip_tt_s1302_fused.log`): L3 slot 2 pos 38 picks HF's 9th expert
  111 instead of 55 (margin 7.97e-4): a router flip under the near-tie policy. The functional path routes it as HF.
- 2026-10-17 Flags `decode_rm_relayout`, `mhc_finalize_fold` (default on). `fused_flag_check.py` extended to 1 eager + 4
  traced decode steps at B 32 and B 1.
- 2026-10-17 Layer-level check (`evidence/s2r2_flag_check.log`): every rewrite off vs on, L0 / L3 / L4, B 32 and B 1,
  prefill + 1 eager + 4 traced decode steps: 36 / 36 bit-identical. Fused suite on the final code
  (`s2_fused_suite6.log`): 43 / 43, PCCs equal to suite5.
- 2026-10-17 Watcher correctness run on the final code (`watcher_correctness/suite`, reviewer's `-k`): 17 passed, 0
  suspicious lines in 65,193; op tests under the watcher (`watcher_correctness/ops`): 37 passed, 0 suspicious.
- 2026-10-17 Tracy re-capture on the final kernels (`evidence/s2r2_tracy.log`): 12 fused + 8 functional KDA cases.
  Found: the worktree's uncommitted stage-6 MoE row split (`GLM53_MOE_SORTED_ROWS=1024`) adds ~11 ms to the 2048-row
  MoE prefill cases; the fused MoE 2048 cases were captured again without it (`tracy_nosplit/`). Kernel-cost table
  and the fp32-kernel reason added to the README. One-user decode is 19-39 us slower than the stage-2 TF32 capture.
- 2026-10-17 Rejected with measurement: the KDA prefill qkv `untilize_out` matmul (3.8 MB L1 per core,
  `evidence/s2r2_probe2.log`).
- 2026-10-17 Stage review #3 (`artifacts/stage2_review_3.md`): more-work-needed, P2 only.
- 2026-10-17 Commit hygiene: a01bb86cfe5 carries two items that are not stage-2 work: the stage-6
  `GLM53_DSA_INDEX_BLOCK_MB` hunk in `tt/dsa.py` (`index_row_block`, inert below ~128K tokens) and the functional
  `kda_dense` prefill CSVs of the stage-1 review #5 Tracy (12:08 / 12:09), which overwrote the s2r2 captures. The
  functional baseline of the README is now pinned in `baseline_functional/`; the two `kda_dense` cells use the s1r5
  values with that provenance. `doc/context_contract.json` also carries stage-6 / multichip edits: only its
  `fused_decoder` entry is stage 2.
- 2026-10-17 Mistake found by review #3: the op-test watcher log kept only the last device session (the watcher log
  restarts at every device open; the op tests open a device per test). Fix: one pytest per module with
  `TT_METAL_WATCHER_APPEND=1` (`scripts/s2r3_chain.sh`).
- 2026-10-17 README fixes: DSA decode concat at 32 rows is 60.5 us (not ~45); the mHC scale is ~1.15e-4 high (the sum of
  squares is ~2e-4 low); suite6 and the watcher suite ran on the stage-1 rework's uncommitted y_swap harness.
- 2026-10-17 Review #3 P2-2 (`evidence/s2r3_probe2.log`): `fast_reduce_nc(dims=[1])` for the `mhc_mix` chunk sum is
  9x faster (234 -> 26 us at 2048 rows) but reads fp32 at TF32 (signed mean -2.6e-4 vs -1.2e-8): rejected under the
  fp32-exact mHC policy. Narrower KDA prefill matmul (qkv + rest, bf16 x bf16 HiFi4): bit-identical, 6688 -> 5970 us,
  but needs the projection weight stored split (no column-offset matmul here); handed to stage 3. First probe run
  failed on a T=1 shape mismatch (fast_reduce_nc returns the padded row count), fixed and rerun.
- 2026-10-17 Review #3 P2-1: op-test watcher per module with append (37 tests, 37 sessions, 0 suspicious) and two
  B=1 traced fused decode cases under the watcher (0 suspicious).
