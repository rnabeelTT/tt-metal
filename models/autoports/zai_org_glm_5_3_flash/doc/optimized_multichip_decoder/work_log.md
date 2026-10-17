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
