# Stage 5 work log (optimized multichip decoder)

Times are UTC, 2026-10-17. Logs are under `~/glm53-flash-bringup/logs/`, probe records under
`~/glm53-flash-bringup/artifacts/s5/`.

- 00:20 Read the stage-4 Tracy windows per device (`scripts/s5_ops.py`). One decode pass holds 6 collectives at 95 to
  100 µs per device in total. The stage-4 README column "CCL time in TP4" (117 to 864 µs) summed the merged
  4-device report. Top decode costs at 1 user (dsa_moe, 1680 µs device): matmuls 482 (five small replicated
  projections of the attention input on 1 to 16 cores: 250), element-wise and data movement 451, five 1-core
  RMSNorm / LayerNorm calls 171, sinkhorn 120, CCL 95. At 32 users the MoE dominates (sparse matmuls 1.8 ms,
  dispatch / gather / untilize 1.0 to 1.2 ms per MoE layer).
- 00:35 Added `MCOpts` (topology, links, L1 CCL outputs, sharded decode norms, packed DSA projections, DRAM-sharded
  router, fast_reduce_nc mix sums) and `tools/mc_decode_ab.py`. Defaults stay at the stage-4 behaviour while the
  queued stage-1 to stage-4 reruns use the file.
- 02:30 Round 1 (`s5_ab1.log`, `artifacts/s5/ab1.jsonl`). dsa_moe, 1 user, replay µs: base 1806, base2 1806,
  ring 1783, links1 1821, l1 1797, norm 1684, mix_nc 1778, pack 1686, norm+pack 1548, router 1779, all 1495.
  kda_moe, 32 users: base 4587 / 4541, all 4347. dsa_moe, 32 users: 5191 -> 4844. Unit checks: sharded norm vs
  1-core norm PCC 0.999999 (max 1 bf16 ulp); fast_reduce_nc sum error 4.3e-4 relative vs 7e-8 for ttnn.sum (TF32
  inputs of the FPU add) -> mix_nc rejected; ring all-gather bit-identical to linear.
- 02:31 kda_moe and kda_dense at 1 user failed in every variant, also the stage-4 base: `kda.recurrent_decode`
  needs batch x heads % 32 == 0, and TP4 leaves 16 heads per die. The stage-4 "incomplete 1-user KDA pass" was this
  error. Fix: `MultichipKDA` pads the decode batch with one inactive slot (state allocated for 2 slots, the padded
  row dropped).
- 04:30 Added `ds_shared` (the packed / DRAM-sharded weights serve prefill with a 2D program config and the
  separate weights are freed: no second copy) and the persistent all-gather option. Round 2 queued.
- 05:05 Round 2 (`s5_ab2.log`, `ab2.jsonl`, `pf2.jsonl`). dsa_moe 1 user: base 1807 / 1807, norm + pack + L1 + ring
  1525, + DRAM-sharded router 1495, + persistent all-gather 1516. fp64 unit checks: the DRAM-sharded matmuls lose
  precision (router fp32 logits 3.9e-3 relative vs 3.0e-4 interleaved; q_a 9.0e-3 vs 2.9e-3): the K-block partials
  reload through the FPU source registers (TF32). The 1-slot KDA runs crashed (segfault in the next host copy): the
  in-tile `ttnn.pad` of the KDA inputs is a view, and the pad path freed it. Fixed (no free). ds_shared builds: 1-slot
  dsa_moe 1484, kda_moe 1222 µs; prefill with the packed DRAM-sharded weights is bit-identical to the separate
  weights (dsa_moe, kda_moe: max abs 0.0). Prefill ring vs linear: kda_moe 26.9 / 25.9 (base / base2) -> 25.5 ms,
  dsa_moe 33.8 -> 33.3 ms, kda_dense 12.8 -> 12.2 ms; ring changes the reduce-scatter summation order (dsa_moe
  prefill PCC vs linear 0.9995, kda layers 0.999997+).
- 05:10 DRAM-sharded matmuls now use packer L1 accumulation (fp32 partials in L1). Added 1D-config alternatives
  for the indexer pack and the router (interleaved weights, no bank padding).
- 07:23 Round 3 (`s5_round3.log`, `ab3.jsonl`). kda_moe 1 user: base 1398 / 1400, norm + L1 + ring 1255, + DS
  router 1222, + 1D router 1224, + persistent all-gather 1249. kda_dense 1 user: 1195 / 1193 -> 1044. dsa_moe 1 user:
  base 1808 / 1809, + DS pack A + DS pack B 1548, + 1D pack B 1519, + DS router 1486, + 1D pack B + 1D router 1489.
  32 users: kda_moe 4610 / 4547 -> 4361; dsa_moe 5127 / 5133 -> 4805. Precision (fp64 references): 1D router and 1D
  indexer pack equal the interleaved path (3.0e-4 / 3.1e-4); DS with L1 accumulation 5.2e-4 to 7.1e-4. Selected:
  1D router, 1D indexer pack, DS pack A, sharded norms, L1 CCL outputs, ring (auto), ds_shared. DEFAULT_MC set.
- 07:31 Tracy after (`tracy/after`), then the suite and the watcher runs on DEFAULT_MC (`s5_suite.log`,
  `s5_watcher.log`). Tracy shows the DS pack A matmul at 62.6 µs on 80 cores (142 GB/s): round 4 tries it on a 1D
  config, the DSA q_b / wq_b decode matmuls on 1D configs, and the KDA inputs allocated with the padded slot count
  (the in-tile pads of the [B, 1] inputs ran on one core, 10 to 11 µs each).
- 08:12 Suite on the round-3 default: 45 passed (`s5_suite.log`, incl. the single-die baseline compare and
  `test_single_slot_kda_decode`, bit-identical). First watcher attempt (`s5_watcher.log`): the watcher-instrumented
  KDA prefill programs exceed the kernel-config ring (75568 > 70656 B, as in stage 1) -> rerun with
  GLM53_WORKER_L1_SIZE=1444864; dsa_moe passed.
- 08:40 Main agent's full-model prefill check on the round-3 default (`s6_prefill_check3.log`): top-1 0.92, top-5 1.00.
- 11:14 Round 5 (`s5_round5.log`, `ab5.jsonl`, `pf5.jsonl`). dsa_moe 1 user: base 1805 / 1804, round-3 default
  1487, + 1D pack A 1436, + 1D q_b / wq_b 1481, both 1432. dsa_moe 32 users: 5122 / 5121 -> 4800 -> 4753. KDA padded
  inputs: kda_moe 1 user base 1371 / 1371, default 1213; kda_dense 1167 / 1166 -> 1035. DSA prefill 2D config:
  33.35 / 33.32 -> 31.39 ms, output PCC 1.0. DEFAULT_MC += pack_a_1d, dsa_1d, prefill_2d.
- 11:21 Final Tracy (`tracy/after`): decode kda_dense-1 979, kda_moe-1 1157, dsa_moe-1 1348, kda_moe-32 4380,
  dsa_moe-32 4536 µs; prefill 11670 / 24815 / 31488 µs. Final suite 45 passed (`s5_suite2.log`); watcher
  (`s5_watcher2.log`): all three kinds passed, no watcher assert, eth-core 29-25 timeout at mesh close (fabric-router
  teardown under the watcher on this board).

## Review #1 rework (2026-10-17 afternoon, `artifacts/stage5_review_1.md`)

- 14:00-15:30 Code: `perm_fidelity` attribute of `mhc_split_sinkhorn` (MHC_SINKHORN_PERM_FIDELITY, HiFi3 permutation
  sums); new op `ttnn.experimental.deepseek_prefill.kda_decode_conv` (fused KDA decode conv step) + test; role decode
  programs (`RoleMatmuls`, `DSMatmul` geometry / readers, `IL1DMatmul` in0 blocks / cores), dense-MLP / shared-expert /
  KDA-out prefill 2D configs, router top-k padding, mHC mix chunk count, MoE grouped geometry, persistent all-gather
  determinism runs, `tools/mc_matmul_sweep.py`, `tools/sinkhorn_fidelity_probe.py`, `GLM53_FABRIC` in the harness.
- 15:34 Round 6 (`records/s5_round6.log`). ttnn rebuilt (`cmake --build build_Release --target ttnn`, 23 s with ccache);
  the Python module `ttnn/ttnn/_ttnn.so` (Oct 15) replaced atomically by the rebuilt one (it carries the new binding;
  backup in the session scratchpad). Sinkhorn: HiFi3 bit-identical to HiFi4 (autoport test 7/7, upstream 4/4, probe
  T = 32 / 2048), HiFi2 0.8 % error. Fused conv test: qkv 1.2e-7 / 1.8e-7, history exact except an idle slot's NaN
  -> Inf (bf16 packer); test updated to accept non-finite -> non-finite.
- 15:41 Op-level matmul sweep (`records/mm6.jsonl`, 505 rows): DRAM-sharded never beats 1D at these shapes (any
  activation geometry, 1-3 readers); best 1D per role recorded; HiFi2 saves <= 1 µs per decode call.
- 15:42 A tool bug (a unit-check block pasted into `set_opts` by a global string replace) failed the first kda_dense
  prefill trial; fixed and rerun at gate 1.
- 15:50 Prefill trials (`records/pf6.jsonl`): `prefill_2d_ffn` kda_dense 12.22 -> 10.49 ms; MoE grouped in0 8
  25.48 -> 25.09 ms; HiFi2 dense MLP rms 0.99735, shared 0.99964, DSA module 0.959: rejected. 1024-row KDA chunks
  (chunk terms in L1) slower than one 2048-row chunk.
- 15:51-16:02 Phase 2 (`records/ab6.jsonl`, `ab6_det.jsonl`): fused conv -85 / -81 µs (kda_moe / kda_dense, 1 user);
  HiFi3 sinkhorn -7 to -18; mix 16 chunks -10; roles -4 to -34; top-k padding +38 (rejected); kda_conv_mixed refused
  by addcmul (dtypes); pairs / masked MoE at 32 users no gain; persistent all-gather 2 x 120 replays bit-identical.
  DEFAULT_MC updated.
- 16:03 Final Tracy (`tracy/after`): decode 865 / 1064 / 1338 / 4433 / 4416 µs, prefill 10084 / 24411 / 30664 µs.
- 16:26 Suite on that default: 5 failed. The persistent all-gather outputs lived in L1 (with `ccl_l1`): their
  permanent allocation at the top of L1 clashed with the static CBs of later prefill programs (`TT_THROW ... clash with
  L1 buffers`), and their lazy `ttnn.from_torch` creation inside the first decode pass tripped
  `test_no_host_fallback`. Fix: the persistent buffers are DRAM tensors created with `ttnn.zeros` on device.
- 16:43-16:57 Round 6 tail (`records/s5_round6.log`): 200-replay stress on the shipped build (`records/stress6.jsonl`,
  all bit-identical); long position 130,000 of 131,072 (`records/longpos6.jsonl`): dsa_moe 1 user 2361 -> 1963 µs;
  watcher on the 1-slot KDA, 32-user MoE and DSA sparse cases (`records/s5_watcher3.log`): 7 cases, 0 suspicious
  lines (teardown abort = the known eth core timeout). The kda_dense stress was lost to a tool bug (bash `read` with
  `IFS=:` dropped the trailing ":" of "fin:"); the probe now accepts an item without the colon.
- 18:52-19:25 Round 7 (`records/s5_round7.log`): kernel tests 9 passed (`records/s5r_kernel_tests.log`); suite 44
  passed, 1 failed (`records/s5_suite4.log`: the baseline compare refuses the stored baseline, policy 'opt' vs the
  stage-3 rework's SELECTED 'opt_s5'); fault suite with `GLM53_FAULT_ROUTED_SCALE=0.9722`
  (`records/s5_suite_fault2.log`): 17 failed + 2 errors, every MoE test caught the fault through the update-rms check,
  the kda_dense tests (no routed experts) passed. kda_dense 200-replay stress: 895 µs, bit-identical. FABRIC_1D_RING vs
  FABRIC_1D (`records/ring_fabric7.jsonl`): kda_moe 1104.8 vs 1103.1, dsa_moe 1407.6 vs 1407.0 µs: no change.
  Decode Tracy refreshed with the DRAM persistent buffers (`tracy/after`, 19:17): 865 / 1070 / 1332 / 4323 / 4655 µs;
  the h all-gather is ~2 µs slower than with the L1 buffers. Tracy ops.csv archives recompressed with xz (all < 200 KB).
- 20:06 Round 8 (`records/s5_baseline8.log`): single-die 'opt_s5' baseline dumped (`refs/baseline_opt_s5_stage5.pt`),
  TP4 compare 8 of 9 rows pass, kda_moe decode PCC 0.998790 / rms 0.99821 fails. Host comparison of the two single-die
  baselines (`records/baseline8_rows.txt`): 'opt_s5' vs 'opt' differ in the same case by the same PCC (0.998795),
  in one token (rel L2 0.0845, others <= 0.0025): a router near-tie on the single die, not a TP4 error (TP4 vs 'opt'
  0.999993).
- 21:06 Round 9 (`records/s5_baseline_fault9.log`): baseline compare with `GLM53_FAULT_ROUTED_SCALE=0.9722`: kda_dense
  rows pass, dsa_moe prefill_a fails on the rms assert (PCC 0.999904 passes the bar; rms ratio 0.97828). The rms
  assert catches the scale fault that the PCC bar misses.

## 2026-10-18: ag_persistent revert, final suite, review #2 rework (`artifacts/stage5_review_2.md`)

- 2026-10-17 21:06 -> 2026-10-18 00:30 (coordinator / stage 7): `ag_persistent` broke the 45-layer model (1 slot teacher
  forcing top-5 32 %; 8 slots: eager and traced runs not repeatable, `records/s7_det_reduced_s8.log`); with it off the
  model repeats exactly (`records/s7_det_reduced_s8_noagp.log`, `records/s7_final_s8.log`) and reaches 92 / 100 / 100.
  `DEFAULT_MC.ag_persistent = False`. Final suite on that default (stage-4 rework, 2026-10-18 00:36):
  `records/s4r_suite_default_final.log`, 45 passed. Committed: code 08bac32f4c1, docs 6272b83d6c9.
- 01:59 ttnn rebuilt (`cmake --build build_Release --target ttnn`, 24 s; `_ttnn.so` replaced atomically): sinkhorn
  `faces` attribute (`MHC_SINKHORN_FACES`), HiFi2 permutation sums refused unless `MHC_SINKHORN_ALLOW_INEXACT=1`.
- 02:01 Job A (`records/s5r2_a.log`): sinkhorn tests 7 / 7 and upstream 4 / 4 with faces off and on; probe
  (`records/s5r2_sk_probe.log`): faces bit-identical to HiFi4 at T = 1 / 8 / 16 / 17 / 32 / 96 / 2048, repeat-identical,
  88.9 -> 73.7 µs (T = 1, face 0) and 88.9 -> 79.5 µs (T = 32, faces 0 / 2) per call (probe wall). In-layer
  (`records/skf_ab.jsonl`, 2 x 60 replays bit-identical each): -30 / -30 / -29 µs at 1 user, -43 / -35 µs at 32 users.
- 02:13 Job B (`records/dsa_dm_ab.jsonl`, `kda_gn_ab.jsonl`; `--compare 8` = 1 eager + 8 traced steps vs `fin`): dsa_q_mm
  bit-identical, -17 µs; host tail bit-identical, -37 µs; dsa_o_mm PCC 0.99924 vs `fin`; KDA fused gated norm slower
  (+28 / +52 / +228 µs) and wrong on kda_moe (PCC 0.08): rejected.
- 02:22 `DEFAULT_MC.sinkhorn_faces = True` was set in the live tree for ~45 min, then set back to False until the
  full-model gate (coordinator: every DEFAULT_MC change needs a stacked teacher-forcing run).
- 02:27 Job C1 (`records/dsa_dm_ab2*.jsonl`, `kda_conv_cmp.jsonl`): the shared tail (producer / consumer) bit-identical;
  the fused KDA conv is bit-identical to the unfused chain at 1 and 32 users, faster at 1 user (-78 µs) and slower at 32
  (+53 µs).
- 02:45 Job C1b (`records/dsa_dm_ab3*.jsonl`, `s5r2_c1b.log`): W_UV relayout exact (max |diff| 0); the stacked-view
  o-path slower (+33 µs); conv crossover: fused -62 µs at 8 users, -28 at 16, +53 at 32 -> `kda_conv_max_batch = 16`.
- 03:04 `tools/dsa_out_probe.py` (`records/s5r2_dsa_out_probe.log`): the one-matmul W_UV product is as exact as the
  per-head bmm against fp64 (rel L2 1.70e-3 vs 1.72e-3); the two differ by bf16 rounding (max 4.5e-3), which the
  layer A/B shows as PCC 0.99924. Gain -12 µs for +2 MB of relaid W_UV per DSA layer per die: not adopted.
- 03:03-04:45 Full-model gate (stacked teacher forcing, 1 slot, `tools/s7_probe.py --do ready`), code snapshot outside the
  live tree (`~/glm53-flash-bringup/s5r2_snapshot`: the stage-8 snapshot that scored 92 / 100 / 100 on the shipped
  default, `s8/shipped.log`, plus the stage-5 files; run from the snapshot directory so `python -m` does not put the
  live tree first). First runs: a live-tree leak (`python -m` from `$TT_METAL_HOME`) and a HEAD snapshot without the
  weight cache (22 min builds) were discarded. Candidates (`sinkhorn_faces`, `dsa_q_mm`, `dsa_tail_shared`,
  `kda_conv_max_batch = 16`): teacher forcing 10 / 48 / 100 (`records/s5r2_tf_cand.log`). Cause: the shared tail was a
  fresh tensor per call, and the producer freed the previous one; the generator captures several decode traces (model
  + sampler variants), so a later capture freed the buffer an earlier trace's consumers still read (single-layer A/Bs
  have one trace and cannot see it). Fix: one fixed buffer per input dict, never freed, written with `ttnn.copy`; the
  first fix failed the trace capture (`ttnn.copy` first used inside the capture, `records/s5r2_tf_cand2.log`): the
  buffer is now created with `ttnn.clone` + `ttnn.copy` in the eager warm-up step.
- 05:02 Gate job D4 (`records/s5r2_d4.log`; weight cache on): control (review-2 files, flags off) 92 / 100 / 100;
  candidates with the fixed tail (`records/s5r2_tf_cand3.log`) 92 / 100 / 100, prefill check 91 / 100 / 100; 8 slots on
  layers 0, 3-7: eager and traced repeat equal (`records/s5r2_det_reduced_cand.log`). The full-model 8-slot run hit a
  fabric mapping failure at open (`topology_mapper.cpp:559`) and was rerun in job E. The optional `ag_persistent` +
  linear-topology bisect was skipped (coordinator). `DEFAULT_MC` set: `sinkhorn_faces`, `dsa_q_mm`, `dsa_tail_shared`,
  `kda_conv_max_batch = 16` (the cand3 options; md5 588398cfc6f9 of `tt/multichip_decoder.py`).
- 05:23-06:14 Job C2 (`records/s5r2_c2.log`, gate snapshot, `PYTHONSAFEPATH=1`, pytest `--import-mode=importlib`; a
  first attempt with `--import-mode=append` broke the conftest import and was discarded): kernel tests 11 + 11 + 2
  passed; Tracy decode 844 / 1053 / 1287 / 4368 / 4611 µs, prefill 10025 / 24410 / 30647 µs, dsa_moe at position
  130,000 1848 µs; replay stress 872 / 1079 / 1368 / 4362 / 4717 µs, 200 replays bit-identical each; position 130,000
  1920 µs (50 replays bit-identical); suite 45 passed; fault suite 17 failed + 2 errors, 28 passed. The watcher and the
  ring-fabric prefill runs were cut by the queue rule (raw flock) and re-queued through `run-locked.sh` (job F).
- 06:19 Job E (`records/s5r2_e.log`, `s5r2_det_full_final.log`): full model, 8 slots, final default: eager and traced
  repeat equal, tokens = stage 7's final run.
- 07:42 Job F (through `run-locked.sh`, `records/s5r2_watcher.log`): watcher on the final default, 7 cases PASSED, 0
  suspicious lines (teardown abort exit 134, known). 08:44-10:0x: FABRIC_1D vs FABRIC_1D_RING prefill on the review-2
  options (`records/pf_ring_r2.jsonl`): bit-identical, walls within +-3 %.
