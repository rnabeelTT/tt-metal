# GLM-5.3-Flash (E192) optimized multichip decoder

Stage 5 of the tt-model-bringup flow. The pass optimizes `tt/multichip_decoder.py` (`MultichipDecoder`, TP4 on the
(1, 4) mesh of a p300x2: 4 Blackhole dies, FABRIC_1D) in place. Contract, state, paging and semantics stay those of
`tt/functional_decoder.py`. Decode is the priority (tokens/s/user, then 32-user throughput), then prefill.

The stage-5 changes are options of one dataclass, `MCOpts` (`tt/multichip_decoder.py`). `DEFAULT_MC` holds the
selected values; `MCOpts()` (all options off, linear topology) is the stage-4 path. `GLM53_MC="key=value,..."` overrides
fields at construction for A/B runs and logs a warning. Two new device ops belong to the stage:

- `ttnn.experimental.deepseek_prefill.kda_decode_conv` (new): the KDA decode conv step (4-tap causal conv in fp32, SiLU,
  idle-slot select, in-place history shift) in one op instead of 12 (`kda_conv_op`).
- `mhc_split_sinkhorn` gained a `perm_fidelity` attribute (`MHC_SINKHORN_PERM_FIDELITY`, 4 by default): the Sinkhorn
  row / column-sum matmuls at HiFi3 are bit-identical to HiFi4 (`sinkhorn_perm_fidelity = 3`).

Review #1 (`~/glm53-flash-bringup/artifacts/stage5_review_1.md`) asked for the deferred items; round 6 of this README
answers them (sections marked "review #1").

## Result

`DEFAULT_MC` (round 6): ring collectives when the fabric routes them (`topology="auto"`), 2 links, decode collective
outputs in L1, persistent all-gather outputs, width-sharded decode RMSNorms, the five replicated DSA projections packed
into two matmuls, 1D decode programs per role (q_a | kv_a pack, indexer pack, DSA q_b / wq_b / o_proj, KDA o_proj,
shared expert, dense MLP, MoE router), the fused KDA decode conv, HiFi3 Sinkhorn sums, 16 mHC mix chunks at decode, the
MoE grouped sparse matmuls at in0 blocks of 8, the packed weights shared with prefill, and explicit 2D prefill programs
for the DSA projections, the packs, the dense MLP, the shared expert and the KDA output projection.

Decode, device time of one traced replay (Tracy, device 0, µs). `tracy/before` = stage-4 options (run 05:17, with the
1-slot KDA fix of round 2 and without the padded-slot inputs of round 4, which removed ~27 µs of one-core pads from a
1-user KDA layer); `tracy/after` = the final `DEFAULT_MC` with the persistent all-gather buffers in DRAM (run 19:17,
round 7; the 16:03 run had them in L1, see the work log):

| Layer kind | Users | Before | After | Change |
|---|---|---|---|---|
| kda_dense (L0) | 1 | 1134 | 865 | -23.7 % |
| kda_moe (L4) | 1 | 1338 | 1070 | -20.0 % |
| dsa_moe (L3) | 1 | 1745 | 1332 | -23.7 % |
| kda_moe | 32 | 4599 | 4323 | -6.0 % |
| dsa_moe | 32 | 4975 | 4655 | -6.4 % |

The 32-user rows move by up to 5 % between Tracy runs of the same build: the sparse expert matmuls and the row gather
(DRAM-bound, 2.7 ms of the dsa_moe 32-user step) took 2499 µs in the 16:03 run and 2718 µs in the 19:17 run, with
the same op list and the other groups within 2 µs. The replay table below (200 replays) is the stable figure.

Decode, traced replay wall on the shipped build (`tools/mc_decode_ab.py --base default`, the `DEFAULT_MC` construction
incl. `ds_shared`; 30 back-to-back replays; `records/stress6.jsonl`) against the stage-4 options on a stage-4 build
(`records/ab6.jsonl`, variant `s4`), µs:

| Layer kind | Users | Stage-4 options (`s4`) | Shipped `DEFAULT_MC` | Change |
|---|---|---|---|---|
| kda_dense | 1 | 1167 | 895 | -23.3 % |
| kda_moe | 1 | 1371 | 1102 | -19.6 % |
| dsa_moe | 1 | 1805 | 1407 | -22.0 % |
| kda_moe | 32 | 4608 | 4431 | -3.8 % |
| dsa_moe | 32 | 5119 | 4722 | -7.8 % |
| dsa_moe, position 130,000 of a 131,072-token cache (review #1) | 1 | 2361 / 2361 (base / base2) | 1963 (run-time `DEFAULT_MC` options) | -16.9 % |

The long-position row decodes in the sparse regime (indexer over ~32.5K pooled keys per slot; `records/longpos6.jsonl`).
For the 45-layer stack (3 kda_dense, 31 kda_moe, 11 dsa_moe layers) the shipped layer replays sum to 52.3 ms per
token at 1 user against 65.9 ms with the stage-4 options (decoder layers only, without embedding, head and sampling).

Prefill, 2048 rows, device time of the signposted pass (Tracy, device 0, µs):

| Layer kind | Before | After | Change | CCL before -> after |
|---|---|---|---|---|
| kda_dense | 12179 | 10084 | -17.2 % | 1889 -> 1467 |
| kda_moe | 25601 | 24411 | -4.6 % | 2150 -> 1438 |
| dsa_moe | 33523 | 30664 | -8.5 % | 1617 -> 1087 |

## Correctness on the shipped default

- Suite `tests/test_multichip_decoder.py` (TP4, `DEFAULT_MC`, round 7, `records/s5_suite4.log`): 44 passed, 1 failed.
  The failure is `test_matches_single_die_baseline`: the stored single-die baseline (`refs/baseline_opt.pt`) carries
  policy 'opt', and the stage-3 rework changed the single-die `SELECTED` policy to 'opt_s5'. The test refuses a
  baseline of another policy, as intended.
- Baseline compare against a fresh single-die 'opt_s5' baseline (`tools/baseline_dump.py`, round 8,
  `records/s5_baseline8.log`): 1 failed. 8 of 9 rows pass (PCC 0.99991 to
  0.999996, rms ratio 0.99956 to 1.00098). kda_moe decode fails: PCC 0.998790 (bar 0.999), rms 0.99821. Cause: one
  token. The host comparison of the two single-die baselines (`records/baseline8_rows.txt`) shows that single-die
  'opt_s5' and single-die 'opt' differ in exactly this case, with the same PCC (0.998795), all in one row (decode
  step 2, slot 0: rel L2 0.0845; the other 5 rows <= 0.0025). TP4 matched single-die 'opt' on this case
  (PCC 0.999993, rms 1.00007, on the shipped options in `s5_mc_suite3.log`). So the single-die 'opt_s5' run moves one token across a router near-tie
  (8.5 % of a token update = one of 8 experts swapped). This is not a TP4 sharding or collective error. The baseline
  compare has no near-tie allowance (the suite's reference checks have one: `suite_swap_cap`). Open item for the
  test owner (stage-4 rework): apply the near-tie allowance per row in the baseline compare, or compare the router
  choice and accept a swapped-expert row.
- Fault injection, routed-expert output x 0.9722 (`GLM53_FAULT_ROUTED_SCALE`), full suite (round 7,
  `records/s5_suite_fault2.log`): 17 failed + 2 errors, 28 passed. Every test that runs a MoE layer fails on the
  update-rms check (ratios 0.970 to 0.988, tolerance 1 +- 0.01); the kda_dense tests (no routed experts) and the
  kernel / CCL unit tests pass. Baseline compare with the same fault (round 9, `records/s5_baseline_fault9.log`):
  failed on the first MoE row, dsa_moe prefill_a:
  PCC 0.999904 (passes the 0.999 bar), rms ratio 0.97828 (outside 1 +- 0.005). The rms assert catches the scale fault
  that the PCC bar alone does not.
- Kernel tests (`records/s5r_kernel_tests.log`): 9 passed (`tests/test_mhc_split_sinkhorn.py` 7 with HiFi3
  permutation sums, `tests/test_kda_decode_conv.py` 2). The upstream sinkhorn tests passed 4/4 in round 6
  (`records/s5_round6.log`).
- Replay determinism: 200 replays bit-identical for kda_dense / kda_moe / dsa_moe at 1 user and kda_moe / dsa_moe at
  32 users (`records/stress6.jsonl`); persistent all-gather 2 x 120 replays from a fresh state bit-identical
  (`records/ab6_det.jsonl`).
- Watcher (`records/s5_watcher3.log`, `watcher/`): 1-slot KDA, 32-user MoE and DSA sparse cases, 0 suspicious lines.

## Performance accounting (review #1 P2-5)

Roofline = bytes one decode step must read per die (weights at their stored dtypes, KDA state read + write, DSA
latent rows for 2048 selected tokens) / 512 GB/s per die. Device = Tracy above. Replay = the shipped-build replay
wall. 1 user (the 1-slot KDA layers carry 2 state slots):

| Case | MB per die | Roofline µs | Device µs | Replay µs | Device / roofline |
|---|---|---|---|---|---|
| kda_dense | KDA in 36.3, out 8.9, decay 0.3, state 4.2, dense MLP 40.1 = 89.8 | 175 | 865 | 895 | 4.9x |
| kda_moe | KDA 49.7, router 1.6, 8 experts 28.3, shared 6.7 = 86.3 | 169 | 1070 | 1102 | 6.3x |
| dsa_moe | packs 11.3, q_b 6.7, W_UK / W_UV 4.5, wq_b 12.6, o 17.8, latent 2.1, MoE 36.6 = 91.6 | 179 | 1332 | 1407 | 7.4x |

Named gap terms of the 1-user kda_moe layer (1070 µs device): Sinkhorn 172 (2 calls on one core, 40 sequential
normalizations each), element-wise / data movement 323 (router 85, mHC glue 120, KDA gates 70, the rest below 10 µs
each), the matmuls at 41 to 64 % of DRAM bandwidth (decode M = 32 on 6 to 87 cores), sparse matmuls 124, mHC kernels 96,
KDA recurrence 60, CCL 60, MoE dispatch 35; and between device time and replay wall ~50 µs of dispatch over 121 ops
(the traced replay issues one program per op).

## Method

- Decode A/B: `tools/mc_decode_ab.py` builds one real layer per (kind, batch); every variant sets the options at run
  time (`MCOpts` fields, `env.NAME=value`, `moe.FIELD=value`, `fid.GROUP=HiFi2`), runs one eager step, captures the
  decode trace and times 30 back-to-back replays (`replay`). `base` first and `base2` last give the run noise (0 to
  60 µs at 32 users, under 3 µs at 1 user). `--units` compares options with the stage-4 path on one input and with an
  fp64 product of the same device weights. `--determinism N`: 2 x N replays from a fresh state, every output compared.
  `--base default`: the `DEFAULT_MC` construction (the shipped build).
- Op-level sweep (review #1 P1-2): `tools/mc_matmul_sweep.py` (one die, per-die TP4 shapes, random data): ttnn default,
  1D multicast (cores x in0 block 4 to 64), DRAM-sharded (activation cores 2 to 32 x in0 block x 1 / 2 / 3 readers per
  bank), each family's best again at HiFi2; prefill rows on the auto config and the explicit 2D config (in0 block 2 / 4 /
  8, HiFi4 / HiFi2). 100 traced replays per decode candidate (the replay wall includes a fixed per-trace overhead, so
  only differences inside a role count). `records/mm6.jsonl`.
- Prefill A/B: `tools/mc_prefill_ab.py` (warm 2048-row pass, best of 3, host wall; output, update PCC and update rms
  ratio against the first variant).
- Tracy: `~/glm53-flash-bringup/scripts/s5_tracy.sh before|after`; device 0, one traced replay (decode) or the
  signposted pass (prefill), sum of `DEVICE KERNEL DURATION` (`~/glm53-flash-bringup/scripts/s5_ops.py`).
- Provenance (review #1 P2-4): every number comes from the worktree of 2026-10-17, which carries the stage-3 rework's
  uncommitted `tt/moe_opt.py` row split (`SORTED_ROWS_DEFAULT = 1024`: a 2048-row prefill runs two 1024-row sorted
  dispatches) for both the before and the after prefill windows; it commits with the stage-3 rework. The decision
  records and logs are in `records/` (`ab*.jsonl`, `pf*.jsonl`, `mm6.jsonl`, `stress6.jsonl`, the round logs, the
  suites).

## Operation-topology audit (decode, per layer, per device)

Stage-4 path (`doc/multichip_decoder/tracy`) unless noted. Element-wise rows from the final Tracy (review #1 P1-3).

| Op sequence | Material cost | Candidate | Constraint | Action / result |
|---|---|---|---|---|
| all-gather h -> RMSNorm [T, 4096] | 17 + **65** µs (1 core) | width-sharded L1 norm, 32 cores | fp32 accumulation | **taken**: 6 µs |
| DSA: 5 projections of the same x on 1 to 48 cores | **250** µs | 2 packed matmuls | one bfp8 pack, one bf16 pack | **taken**: 1D configs, ~60 µs |
| DSA q_a / kv_a norms | 25 + 11 µs (1 core) | sharded norm | - | **taken** |
| MoE router [T, 4096] x [4096, 192] | 49 µs (6 cores) | DS / 1D | fp32 routing exactness | **taken**: 1D, 18 µs |
| row-parallel matmul -> reduce-scatter | 17 to 32 µs per RS | ring, L1 out, fused / bfp8 | - | ring + L1 taken; fused / bfp8 rejected |
| mHC mix: `mhc_mix` partials -> sum -> all-gather -> sum -> 9 small ops | 42 µs + 13 µs CCL per sub-block | `fast_reduce_nc`; fewer chunks | fp32-exact sums | fast_reduce_nc rejected (4.3e-4 error); 16 chunks taken (-6 to -10 µs per layer) |
| `mhc_split_sinkhorn` | 2 x 89 µs, 1 core | split users over cores; HiFi3 permutation sums | one tile holds all users; fp32-exact | HiFi3 **taken** (bit-identical, 89 -> 86 µs) |
| KDA decode conv: slice, multiply, 3 x (typecast + addcmul), SiLU, select, 3 shift selects on [B, 6144] | **~105** µs (12 ops) | one fused op | bit-exact history, fp32 conv, NaN-safe select | **taken**: `kda_decode_conv` 38 µs |
| KDA gates: decay projection + sigmoid, exp, beta sigmoid, 2 selects, o-norm, gate multiply, typecasts | ~70 µs (B = 1) | `sigmoid_gated_rms_norm` (upstream fused o-norm + gate) | it takes head-first [B*H, T, V] / [B, T, W]; decode has [B*H, V] rows | rejected: the relayout costs what the fusion saves |
| MoE router glue: sigmoid, add, **TopK (1 core, 34 µs)**, one-hot weights, normalize (FillPad 10 µs on [T, 8]) | 85 µs | padded top-k width (256 / 1024: multi-core top-k); fused gate op | fp32 selection (stage-1 route flips) | padded top-k slower (+39 µs, `ab6`); fused gate op computes bf16 scores: rejected |
| FillPad (11 per layer, 61 µs) | from `ttnn.sum` / reduce on widths 8 / 64 / 192 that are not tile multiples | wider tensors / different reduce | mix sums must stay fp32-exact | the mix-sum FillPads shrink with 16 chunks; the router ones stay (exact reduce) |
| MoE 32 users: sorted dispatch -> gather [G x 32, 4096] -> grouped sparse gate/up -> SwiGLU -> grouped down -> untilize -> gather-add | gather 620, untilize 260, SwiGLU 362, sparse 1866 µs | pairs path; masked (union) path; geometry | 32-row tiles per expert group | pairs +2.4 ms, union no gain; in0 blocks of 8 taken (-36 to -60 µs) |
| inter-layer boundary | none | - | - | contract below |

## Candidates and decisions

Replay wall in µs (`tools/mc_decode_ab.py`) against `fin` (the previous default) or `base` of the same build.
Records: `records/ab{1,2,3,5,6}.jsonl`, `records/pf{2,5,6}.jsonl`.

| Option (`MCOpts`) | Measurement | Precision | Decision |
|---|---|---|---|
| `norm_sharded` | dsa_moe 1 user 1806 -> 1684 | PCC 0.999999, max 1 bf16 ulp | taken |
| `dsa_pack` | dsa_moe 1 user 1806 -> 1686 (+ norm 1548) | packs at the separate matmuls' error (fp64 reference) | taken |
| pack A DS vs 1D (`pack_a_1d`) | DS 1487, 1D 1436 (1 user) | 1D = separate (2.90e-3); DS 3.03e-3 | 1D taken |
| pack B DS vs 1D (`pack_b_1d`) | DS 1548, 1D 1519 | gate (fp32 out): DS 7.0e-4, 1D 3.1e-4 = separate | 1D taken |
| router DS vs 1D | DS 1222, 1D 1224 (kda_moe 1 user) | DS 3.9e-3 (5.2e-4 with L1 acc.), 1D 3.0e-4 = interleaved | 1D taken |
| `topology` ring | dsa_moe 1 user -23 µs; prefill -0.5 to -1 ms per layer | all-gather bit-identical; reduce-scatter order changes (prefill dsa_moe PCC 0.9995 vs linear on a random input; suite passes) | taken (`auto`) |
| `num_links` | 1 link 1821 vs 2 links 1806 | identical | 2 |
| `ccl_l1` | -9 µs (1 user), -86 µs (32 users) | identical | taken |
| `ag_persistent` (review #1 P2-1) | kda_moe 1 user 1212 -> 1203, dsa_moe 1432 -> 1423, kda_moe 32 4433 -> 4362, dsa_moe 32 4783 -> 4759 | 2 x 120 replays from a fresh state bit-identical, every kind and batch (`records/ab6_det.jsonl`) | taken in this stage, then **reverted to off** (2026-10-18): on the 45-layer model the shared per-call-site buffers broke decode (1 slot: teacher forcing top-5 32 %, prefill-check unaffected; 8 slots: identical greedy runs differ). With `GLM53_MC=ag_persistent=0` the model is back to top-5 100 % (stage 7, `doc/optimized_full_model/`). A one-layer replay test cannot expose cross-layer reuse of one buffer. The decode gain was 9-71 µs per layer. |
| `mix_nc` | -28 µs | 4.3e-4 vs 7e-8 | rejected |
| `ds_shared` | prefill dsa_moe 33.8 -> 32.8 ms | prefill bit-identical | taken |
| `dsa_1d` | 1436 -> 1432 | = default (3.01e-3) | taken |
| `prefill_2d` (q_b, wq_b, o_proj) | dsa_moe prefill 33.35 -> 31.39 ms | PCC 1.0 | taken |
| 1-slot KDA padded inputs | kda_moe 1 user 1398 -> 1371 | identical | taken |
| `sinkhorn_perm_fidelity = 3` (review #1 P1-1) | 89.4 -> 85.8 µs per call (Tracy); replay -7 to -18 µs per layer | **bit-identical** to HiFi4 (`tests/test_mhc_split_sinkhorn.py` 7/7, upstream 4/4, `tools/sinkhorn_fidelity_probe.py`); HiFi2 0.8 % error: rejected | taken |
| `kda_conv_op` (review #1 P1-3) | kda_moe 1 user 1214 -> 1129, kda_dense 1034 -> 953 (Tracy: 12 ops ~105 µs -> 38 µs) | qkv 1.2e-7 / 1.8e-7 relative (fp64); history bit-exact; an idle slot's NaN history comes back Inf (bf16 packer NaN encoding), still non-finite and kept out of qkv (`tests/test_kda_decode_conv.py`) | taken |
| `kda_conv_mixed` (bf16 history into addcmul) | - | `TT_FATAL`: addcmul needs equal dtypes | blocked |
| `mix_chunks_decode` 16 / 32 | kda_moe 1 user 1214 -> 1204 / 1204; dsa_moe 1432 -> 1431 | ttnn.sum (exact) over fewer partials | 16 taken |
| `router_topk_pad` 256 / 1024 | +38 / +38 µs (kda_moe), +39 (dsa_moe) | identical selection | rejected |
| role programs (review #1 P1-2): KDA o, DSA o / q_b / wq_b, pack A geometry, shared expert, dense MLP | kda_moe 1 user kout -5, shared -6; dsa_moe 1 user DSA roles -6, shared -4; kda_dense dense MLP -34 | outputs bf16 at the same rounding | taken |
| KDA in-projection 1D (`kda_in=1d:c29:ib4`) | op sweep 162 -> 110 (replay-inflated), in layer 1214 -> 1216 | - | rejected (no gain in the layer: ttnn's 87-core program already reads at 350 GB/s) |
| `moe_grouped_in0 = 8` (review #1 P1-2) | 32 users: kda_moe 4422 / 4364 -> 4361, dsa_moe 4751 -> 4715; prefill kda_moe 25.48 -> 25.09 ms; 1 user unchanged | PCC 1.0 | taken |
| `prefill_2d_ffn` + `packs_prefill_2d` (review #1 P1-1) | prefill kda_dense 12.22 -> 10.49 ms, kda_moe 25.48 -> 25.26, dsa_moe 31.46 -> 31.11 | update PCC 0.999999+, rms 1.0 | taken |
| HiFi2 (review #1 P1-1) | see "Fidelity" | systematic -0.27 % (dense MLP) / -0.04 % (shared) scale, -4 % (DSA module) | rejected |

## Matmul geometry per role (review #1 P1-2)

`tools/mc_matmul_sweep.py`, decode M = 32, µs per call (replay wall; compare within a row). In-model program =
the stage-4 / stage-3 choice. Best DS = best DRAM-sharded geometry over activation cores x in0 block x readers.

| Role [K x N], dtype | ttnn default | best 1D | best DS (geometry) | HiFi2 (best 1D) | Final |
|---|---|---|---|---|---|
| KDA in [4096 x 8336] bfp8 | 162.1 | 109.9 (29 cores, in0 4) | 140.1 (32 act. cores, in0 2, 3 readers) | 110.3 | default (no in-layer gain) |
| KDA out [2048 x 4096] bfp8 | 48.2 | 33.7 (16, 4) | 45.4 (8, 4, 2 r.) | 33.3 | 1D 16 / 4 |
| DSA o [4096 x 4096] bfp8 | 84.0 | 57.5 (16, 4) | 76.1 (16, 4, 2 r.) | 56.8 | 1D 16 / 4 |
| DSA q_b [1536 x 4096] bfp8 | 39.4 | 27.9 (16, 4) | 38.2 (8, 3, 2 r.) | 27.5 | 1D 16 / 4 |
| DSA wq_b [1536 x 4096] bf16 | 43.6 | 41.2 (16, 4) | 55.2 (8, 6, 1 r.) | 40.6 | 1D 16 / 4 |
| pack A [4096 x 2048] bfp8 | 82.4 | 34.8 (32, 8) | 45.5 (4, 8, 2 r.) | 34.2 | 1D 32 / 8 |
| shared gate/up [4096 x 1024] bfp8 | 66.8 | 23.8 (32, 16) | 29.5 (4, 8, 2 r.) | 23.3 | 1D 32 / 16 |
| shared down [512 x 4096] bfp8 | 21.3 | 16.1 (16, 4) | 22.4 (4, 4, 2 r.) | 15.6 | 1D 16 / 4 |
| dense gate/up [4096 x 6144] bfp8 | 104.4 | 84.0 (16, 4) | 105.8 (32, 2, 3 r.) | 83.0 | 1D 16 / 4 |
| dense down [3072 x 4096] bfp8 | 66.1 | 45.6 (16, 4) | 61.1 (8, 6, 2 r.) | 45.2 | 1D 16 / 4 |

DRAM-sharded never beat the best 1D program at these shapes, with any activation geometry or reader count; in0 blocks
above 8 helped only the shared gate/up (16). Every DS candidate also carried 2.5x to 4x the output error of the 1D one
(K-block partials re-read through the FPU source registers). Refused geometries (L1 / CB, e.g. 2 activation cores with
in0 blocks of 32) are in `records/mm6.jsonl`.

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
| KDA in | 3904 | (upstream tuned config in the KDA prefill module) | - | unchanged |

## Sparse-matmul geometry at TP4 (review #1 P1-2)

The grouped sparse matmuls (pairs path at 1 user, sorted path at 32 users and in prefill) came from single-die tuning
(`in0_block_w = 4`, subblock 1 x 8 capped at 4 by fp32 accumulation). TP4 sweep (`moe.grouped_*`), replay µs / prefill
ms:

| Geometry | kda_moe 1 user | kda_moe 32 users | dsa_moe 32 users | prefill kda_moe 2048 |
|---|---|---|---|---|
| in0 4 (stage 3) | 1214 / 1213 | 4422 / 4364 | 4751 / 4752 | 25.48 / 25.42 |
| in0 2 | 1213 | 4484 | - | 26.03 |
| in0 8 | 1213 | 4361 | 4715 | 25.09 |
| in0 16 | - | 4380 | - | 25.23 |
| subblock w 4 | - | - | - | 25.31 |

in0 8 is selected (`moe_grouped_in0 = 8`).

## Fidelity (review #1 P1-1)

Decode matmuls are DRAM-bound: HiFi2 saved 0.4 to 1.1 µs per call (table above) and adds the truncation bias the
stage-3 review measured (activation low bit dropped). Prefill matmuls are compute-bound and HiFi2 is 25 to 35 % faster
per call, but on real weights (`tools/mc_prefill_ab.py`, update rms vs HiFi4): dense MLP HiFi2 0.99735 (-0.27 %, a
systematic scale loss) for -0.7 ms; shared expert HiFi2 0.99964 for -0.3 ms; the DSA module config at HiFi2 0.959 /
PCC 0.9965. The explicit 2D config at HiFi4 gives the larger part of the gain without the bias (dense layer 12.2 ->
10.5 ms). HiFi2 is rejected for every group; the datatype sweep (stage 8) can revisit it against model accuracy.

## Collectives (OPT-009 fields)

Six per layer, `ttnn.experimental.*_async` (`MeshCCL`), `chunks_per_sync` 10, `num_workers_per_link` 2,
`num_buffers_per_channel` 2, 2 links, ring.

| Collective | Payload per device | Dtype | Output | Persistent output | kda_moe 1 user µs (stage 4 -> final) |
|---|---|---|---|---|---|
| all-gather of the mix partials, dim 1 | [T, 64] -> [4, T, 64] | fp32 | DRAM (the persistent buffer) | yes (per site, shared over layers) | 12.0 -> 7.1 |
| all-gather of h, dim 3 | [T, 1024] -> [T, 4096] | bf16 | DRAM (the persistent buffer) | yes | 17.8 -> 10.8 |
| reduce-scatter, dim 3 | [T, 4096] -> [T, 1024] | bf16 (intermediate DRAM) | L1 | no (122B: no gain, `~/qwen35-122b-bringup/logs/05-r2-ccl-probe.log`) | 16.2 -> 15.0 |

Final values = the mean of the 2 calls per layer in `tracy/after/test_perf_traced_decode_kda_moe-1_` (19:17). With the
persistent buffers in L1 (16:03 run) the values were 7.1 / 8.8 / 13.9 µs; L1 placement is not usable (it clashes with
the static CBs of later prefill programs), so the DRAM buffers cost about 2 µs per h all-gather.

Ring on a FABRIC_1D fabric: `ttnn.get_usable_topology` returns Ring only when the fabric graph has the wraparound link
(both p300 board-to-board links up; Linear otherwise). FABRIC_1D_RING adds dateline deadlock avoidance for traffic
that crosses the wrap. These collectives are latency-bound single-round transfers (one chunk per direction per link);
the 122B port has served Ring on FABRIC_1D on this board, and every replay here (thousands, incl. 2 x 120-replay
determinism runs and 200-replay stress runs) completed bit-identical. `GLM53_FABRIC=FABRIC_1D_RING` (tests/harness.py) selects the ring fabric for an A/B. Result (`records/ring_fabric7.jsonl`, shipped `DEFAULT_MC`, 1 user, replay µs, FABRIC_1D_RING vs FABRIC_1D): kda_moe 1104.8 vs 1103.1, dsa_moe 1407.6 vs 1407.0. The difference is in the noise (< 0.2 %), so FABRIC_1D stays the default fabric.

Rejected collective families (measured on this board by the 122B port, same ops and sizes): fused all-gather + matmul
(`~/qwen35-122b-bringup/logs/05-agmm-probe*.log`: 0.14 to 0.60 ms vs 0.07 ms separate; here an RMSNorm also sits between
the all-gather and every consumer matmul), fused matmul + reduce-scatter (`05-mmrs-probe*.log`: 2D configs only,
<= 8 compute cores for one tile of rows), bfp8 payloads (`05-r2-ccl-probe.log`: -1.4 / -2.6 µs, plus a typecast),
all-reduce instead of reduce-scatter (4x the bytes and the mHC work on every die).

## Inter-layer residual contract (for the full model)

- Input and output of every layer: the 4-stream mHC state X [1, 1, T, 4 x 4096] fp32, stream-interleaved and
  hidden-sharded; device d holds columns [1024 d, 1024 (d + 1)) of every stream, DRAM interleaved, TILE.
- No collective sits between layers.
- Decode rows: one tile of users (B <= 32); with B = 1 the KDA layers pad to 2 slots internally. Contract of
  `_decode_rows`: every prefill chunk is a multiple of 128 rows, so 32 padded rows only ever mean decode.

## 1-slot KDA decode

`kda.recurrent_decode` needs batch x local heads % 32 == 0; TP4 leaves 16 heads, so a 1-slot decoder failed (stage 4).
`MultichipKDA` allocates 2 state slots and per-slot inputs for 2 slots (the second inactive) and pads x in place.
`test_single_slot_kda_decode`: 1-slot vs 2-slot bit-identical.

## Sinkhorn (review #1 P1-1)

`mhc_split_sinkhorn` splits token tiles over cores (prefill: 64 cores at 2048 rows); decode is one tile on one core, 40
sequential normalizations of 8 permutation matmuls + an fp32 reciprocal + 2 multiplies + 3 packs each. Splitting
users over cores cannot help (the tile is the SFPU / FPU work unit). The permutation operand is 0 / 1, so the 4th
fidelity phase (low srcA bits x low srcB bits) only adds zeros: HiFi3 is bit-identical (verified for T = 32 / 96 / 2048,
sharded and interleaved) and HiFi2 is not (0.8 % error). Gain: 89.4 -> 85.8 µs per call. The rest of the cost is SFPU
and pack work on full 32 x 32 tiles; processing only the 16 used columns (vector mode C) would leave stale DST values in
columns 16 to 31 that the next permutation matmul multiplies by 0, and 0 x Inf = NaN, so it needs the padding faces
zeroed first: not taken.

## 32-user MoE (review #1 P2-6)

Lower-padding forms measured at 32 users (`records/ab6.jsonl`): the pairs path (one 32-row tile per (token, expert)
pair, 256 groups) 6788 / 6883 µs vs 4364 / 4751 (it re-reads one expert block per pair); the masked union path 4383
(no gain); grouped in0 blocks of 8: taken. The remaining glue (gather 620, untilize 260, SwiGLU 362 µs) is set by the
32-row tile of an expert group with ~1.8 real rows: a smaller group needs a sub-tile (16- or 8-row) sparse-matmul and
gather path, which the grouped factory does not have. The down projection cannot write row-major output at TP4 (its
resident block, 1.7 MB, exceeds the factory's 1.3 MB L1 budget).

## KDA prefill chunk terms (review #1 P2-7)

Every TP4 KDA prefill of 2048 rows logs "KDA chunk terms need 595 KiB per core ... keeping them in DRAM". 1024-row
chunks fit L1 (298 KiB): kda_dense 2 x 6.85 = 13.7 ms vs 12.2 ms for one 2048-row chunk, kda_moe 2 x 13.27 = 26.5 vs
25.5 ms (`records/pf6_1024.jsonl`). The DRAM fallback at 2048 rows is faster overall: kept.

## Memory and context contract (review #1 P2-3)

`DEFAULT_MC` adds per die: the persistent all-gather outputs (< 0.6 MB of DRAM, once per TT_CCL), the 1-slot KDA padding
slot when `max_batch = 1` (+1 MiB of recurrent state per KDA layer: 34 MiB, ~2.9K tokens of DSA cache at 11,968 B per
token), nothing else (the packs replace the separate weights). The DSA latent cache cannot drop to bfp8 here:
`sparse_sdpa` takes a ROW_MAJOR bf16 or FP8_E4M3 KV (`sparse_sdpa_device_operation.cpp:71,117`), and bfp8 is a tile
format. FP8_E4M3 (3 mantissa bits, 16x the bf16 element error) needs `scatter_rows` sources cast to FP8 on both write
paths and a model-accuracy check: a datatype-sweep (stage 8) item; the sequence-sharded latent that reaches the
advertised context is the stage-4 rework's `MCOpts.dsa_shard`. `doc/context_contract.json` records this.

## Rejected and not adopted

| Candidate | Evidence | Reason |
|---|---|---|
| Multi-core Sinkhorn | one tile holds every user | see Sinkhorn |
| `fast_reduce_nc` mix sums | 4.3e-4 vs 7e-8 relative | fp32-exact mHC |
| DRAM-sharded decode matmuls (all roles) | geometry sweep above | slower and less exact than 1D |
| HiFi2 | Fidelity | systematic scale loss |
| KDA in-projection 1D | in-layer 1214 -> 1216 | no gain |
| padded router top-k | +38 µs | slower |
| fused MoE gate op | - | bf16 scores |
| `sigmoid_gated_rms_norm` for the decode o-norm + gate | head-first layout | relayout cost |
| pairs / masked MoE at 32 users | +2.4 ms / +19 µs | slower |
| 1024-row KDA prefill chunks | +1.0 to 1.5 ms per 2048 tokens | slower |
| tt-perf-report "Output subblock 1x1" | one output tile per core is the widest 1D grid | - |
| tt-perf-report "place input 0 in L1" | collective outputs moved (`ccl_l1`) | the rest are 64-256 KB activations |

## Open items

- Sinkhorn (172 µs per layer) and the router / mHC glue (~200 µs) are the largest 1-user costs left; both are exact
  fp32 paths that need new kernels (a decode Sinkhorn over the 16 used columns with zeroed padding faces; an fp32
  router-gate op).
- 32-user MoE glue: sub-tile expert groups (kernel work in the grouped sparse-matmul factory).
- Full-model readiness on the final `DEFAULT_MC` (the round-3 default passed prefill-check at top-1 0.92 / top-5 1.00);
  single-user serving needs `GLM53_MAX_BATCH = 1`.
- Baseline compare vs the 'opt_s5' single-die baseline: one kda_moe decode token sits on a router near-tie that
  the single-die 'opt_s5' run resolves differently (see Correctness). The compare needs a per-row near-tie
  allowance (test owner: stage-4 rework); the stored default baseline `refs/baseline_opt.pt` must be regenerated for
  'opt_s5' (`refs/baseline_opt_s5_stage5.pt` from round 8 can serve).
- 32-user decode Tracy varies by up to 5 % between runs (the DRAM-bound sparse expert matmuls and row gather).

## Artifacts

- Code: `tt/multichip_decoder.py` (`MCOpts`, `DEFAULT_MC`, `MeshCCL`, `DSMatmul`, `IL1DMatmul`, `RoleMatmuls`,
  `sharded_rms_norm`, `MultichipDSA`, `MultichipKDA`, `MultichipMLP`, shared-expert and router hooks),
  `tt/fused_decoder.py` (`_in_proj`, `_conv_step` hooks), `tt/moe.py` (collective hooks, top-k padding hook),
  `tests/harness.py` (`GLM53_FABRIC`), `tests/test_multichip_decoder.py::test_single_slot_kda_decode`,
  `tests/test_kda_decode_conv.py`, `tools/mc_decode_ab.py`, `tools/mc_prefill_ab.py`, `tools/mc_matmul_sweep.py`,
  `tools/sinkhorn_fidelity_probe.py`.
- ttnn: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/kda_decode_conv/` (new op),
  `deepseek_prefill/mhc_split_sinkhorn/` (`perm_fidelity`), `deepseek_prefill/CMakeLists.txt`,
  `experimental/experimental_nanobind.cpp`.
- Tracy + tt-perf-report (with advice): `tracy/before/<case>/`, `tracy/after/<case>/` (the raw op list as `ops.csv.xz`).
- Watcher: `watcher/<case>/`.
- Records: `records/` (A/B jsonl, sweep, stress, round logs, suites, watcher log).
- Commands:
  ```
  GLM53_TP=4 pytest models/autoports/zai_org_glm_5_3_flash/tests/test_multichip_decoder.py
  pytest models/autoports/zai_org_glm_5_3_flash/tests/test_mhc_split_sinkhorn.py models/autoports/zai_org_glm_5_3_flash/tests/test_kda_decode_conv.py
  GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.mc_decode_ab --kind kda_moe --batch 1 --base default --determinism 200 --variants "fin:" --out stress.jsonl
  python -m models.autoports.zai_org_glm_5_3_flash.tools.mc_matmul_sweep --out mm.jsonl
  ~/glm53-flash-bringup/scripts/s5_tracy.sh after
  ```
