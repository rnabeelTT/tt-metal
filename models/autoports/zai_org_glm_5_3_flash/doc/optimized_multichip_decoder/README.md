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
  row / column-sum matmuls at HiFi3 are bit-identical to HiFi4 (`sinkhorn_perm_fidelity = 3`). HiFi2 (not exact) is
  refused unless `MHC_SINKHORN_ALLOW_INEXACT=1` (probes).
- `mhc_split_sinkhorn` gained a `faces` attribute (`MHC_SINKHORN_FACES=1`, off by default): the SFPU work of the 40
  normalizations runs only on the DST faces that hold the 16 used columns (face 0 at <= 16 tokens, faces 0 and 2
  otherwise), with the padding of m zeroed once (`sinkhorn_faces`; review #2 P1-2).

Review #1 (`~/glm53-flash-bringup/artifacts/stage5_review_1.md`) asked for the deferred items; round 6 of this README
answers them (sections marked "review #1"). Review #2 (`stage5_review_2.md`) found the headline numbers measured with
`ag_persistent` on, which the final default no longer has; the 2026-10-18 rework (sections marked "review #2")
re-measures everything on the final default and adds the face-restricted Sinkhorn, the DSA data-movement audit and the
stacked (full-model) gate.

## Result

`DEFAULT_MC` (review #2, 2026-10-18): ring collectives when the fabric routes them (`topology="auto"`), 2 links, decode
collective outputs in L1 (non-persistent: `ag_persistent` is rejected), width-sharded decode RMSNorms, the five
replicated DSA projections packed into two matmuls, 1D decode programs per role (q_a | kv_a pack, indexer pack, DSA q_b /
wq_b / o_proj, KDA o_proj, shared expert, dense MLP, MoE router), the fused KDA decode conv up to 16 slots, HiFi3 and
face-restricted Sinkhorn, 16 mHC mix chunks at decode, the 1-slot DSA q_abs as one matmul, the sparse-index tail once per
step, the MoE grouped sparse matmuls at in0 blocks of 8, the packed weights shared with prefill, and explicit 2D prefill
programs for the DSA projections, the packs, the dense MLP, the shared expert and the KDA output projection. Every
number below is from that default (no `GLM53_MC`), code from the gate snapshot (`~/glm53-flash-bringup/s5r2_snapshot`,
the stage-8 snapshot plus the stage-5 files, md5s in `records/snapshot_md5.txt`), job `records/s5r2_c2.log`.

Decode, device time of one traced replay (Tracy, device 0, µs). `tracy/before` = stage-4 options (run 2026-10-17
05:17); `tracy/after` = the final `DEFAULT_MC` (2026-10-18 05:26-05:32):

| Layer kind | Users | Before | After | Change |
|---|---|---|---|---|
| kda_dense (L0) | 1 | 1134 | 844 | -25.6 % |
| kda_moe (L4) | 1 | 1338 | 1053 | -21.3 % |
| dsa_moe (L3) | 1 | 1745 | 1287 | -26.2 % |
| kda_moe | 32 | 4599 | 4368 | -5.0 % |
| dsa_moe | 32 | 4975 | 4611 | -7.3 % |
| dsa_moe at position 130,000 (sparse regime, `tracy/after_pos130000`) | 1 | - | 1848 | - |

The 32-user rows move by up to 5 % between Tracy runs of the same build (the DRAM-bound sparse expert matmuls and row
gather: 2499 / 2718 / 2684 µs in three runs); the replay table (200 replays) is the stable figure. The long-position
step costs +561 µs over position 128, all in the sparse attention path: sparse_sdpa 33 -> 356 µs (2,048 selected
tokens gathered from a 131K-token cache instead of 129), the top-512 pool selection over ~32.5K pooled keys
(`TopkLargeIndices` 7 -> 142 µs), the indexer scores 5 -> 38 µs and their masking ops.

Decode, traced replay wall on the shipped build (`tools/mc_decode_ab.py --base default --determinism 200`, the
`DEFAULT_MC` construction incl. `ds_shared`; 30 back-to-back replays; `records/stress_r2.jsonl`, `longpos_r2.jsonl`)
against the stage-4 options on a stage-4 build (`records/ab6.jsonl`, variant `s4`), µs:

| Layer kind | Users | Stage-4 options (`s4`) | Shipped `DEFAULT_MC` | Change |
|---|---|---|---|---|
| kda_dense | 1 | 1167 | 872 | -25.3 % |
| kda_moe | 1 | 1371 | 1079 | -21.3 % |
| dsa_moe (first DSA layer: computes the step's tail) | 1 | 1805 | 1368 | -24.2 % |
| dsa_moe (the other 10: read the tail) | 1 | 1805 | ~1333 (-35 µs, `records/dsa_dm_ab2.jsonl`) | -26.2 % |
| kda_moe | 32 | 4608 | 4362 | -5.3 % |
| dsa_moe | 32 | 5119 | 4717 | -7.9 % |
| dsa_moe, position 130,000 of a 131,072-token cache | 1 | 2361 | 1920 | -18.7 % |

For the 45-layer stack (3 kda_dense, 31 kda_moe, 11 dsa_moe layers) the shipped layer replays sum to 50.8 ms per token
at 1 user against 65.9 ms with the stage-4 options (decoder layers only, without embedding, head and sampling; the
review-#1 figure of 52.3 ms was measured with `ag_persistent` on).

Prefill, 2048 rows, device time of the signposted pass (Tracy, device 0, µs):

| Layer kind | Before | After | Change | CCL before -> after |
|---|---|---|---|---|
| kda_dense | 12179 | 10025 | -17.7 % | 1889 -> 1451 |
| kda_moe | 25601 | 24410 | -4.7 % | 2150 -> 1458 |
| dsa_moe | 33523 | 30647 | -8.6 % | 1617 -> 1069 |

## Correctness on the shipped default

All on the final `DEFAULT_MC`, gate snapshot code, review #2 (2026-10-18):

- Full model (the stage gate, see "Stacked gate"): 1-slot teacher forcing 92 / 100 / 100, prefill check 91 / 100 / 100
  (`records/s5r2_tf_cand3.log`; the same as the stage-7 / stage-8 reference); 8-slot repeatability on layers 0, 3-7
  (a producer and a consumer DSA layer): eager and traced, 2 runs each, tokens equal, max |diff| 0
  (`records/s5r2_det_reduced_cand.log`); full model, 8 slots (`records/s5r2_det_full_final.log`): eager and traced, 2 runs each, tokens equal,
  max |diff| 0, and the same greedy tokens as stage 7's final run (`records/s7_final_s8.log`: 46653, 31910, 52614, ...
  at L = 100; 271, 1259, 364, 345, ... at L = 300).
- Suite `tests/test_multichip_decoder.py` (TP4, `records/s5r2_suite.log`): 45 passed. The single-die baseline compare
  passes with the stage-4 rework's capped near-tie allowance: kda_moe decode row 2 (PCC 0.99645, router margin 2.9e-4,
  cap 1) is set aside; the other rows PCC 0.999998, rms ratio 1.00011; every other case PCC >= 0.99991, rms within
  1.00098. (The same row is what failed the round-8 compare against the 'opt_s5' baseline: the single-die 'opt_s5'
  run puts that token on the other side of the tie, `records/baseline8_rows.txt`.) The previous final suite, round-6
  options without `ag_persistent`: `records/s4r_suite_default_final.log`, 45 passed.
- Fault injection, routed-expert output x 0.9722 (`GLM53_FAULT_ROUTED_SCALE`, `records/s5r2_suite_fault.log`): 17
  failed + 2 errors, 28 passed. Caught (update-rms ratio outside 1 +- 0.01, or the baseline compare's rms +- 0.005): the
  real-weight, batched-user, determinism, length (kda_moe 33 / 2049; dsa_moe 33 to 4000), paged-block and
  baseline-compare tests, `test_synthetic_weights[kda_moe]`, `test_batch32_all_active[dsa_moe]` and
  `test_inactive_slot_state[dsa_moe]`. Not caught on their own: `test_decode_only[kda_moe]` / `[dsa_moe]`,
  `test_batch32_all_active[kda_moe]`, `test_inactive_slot_state[kda_moe]`, `test_synthetic_weights[dsa_moe]`,
  `test_lengths[dsa_moe-1]`, and the `test_no_host_fallback` cases (no numeric check). The kda_dense tests have no
  routed experts and pass, as they should.
- Kernel tests: `tests/test_mhc_split_sinkhorn.py` 7 + upstream 4 with faces off and on (`records/s5r2_kernel_f0.log`,
  `s5r2_kernel_f1.log`), `tests/test_kda_decode_conv.py` 2 (`records/s5r2_kernel_conv.log`).
- Replay determinism: 200 replays from a fresh state, bit-identical, kda_dense / kda_moe / dsa_moe at 1 user and
  kda_moe / dsa_moe at 32 users (`records/stress_r2.jsonl`); 50 at position 130,000 (`records/longpos_r2.jsonl`).
- Option equivalence (`tools/mc_decode_ab.py --compare 8`: 1 eager + 8 traced steps vs the option off, bit by bit):
  `sinkhorn_faces`, `dsa_q_mm`, `dsa_tail_shared` and the fused KDA conv at 1, 8, 16 and 32 slots are bit-identical
  (`records/skf_ab.jsonl`, `dsa_dm_ab2.jsonl`, `kda_conv_cmp.jsonl`).
- Watcher (`records/s5r2_watcher.log`, `watcher/<case>/console.log`), final default: 7 cases (real weights of each
  kind, the 1-slot KDA decode, 32 users kda_moe / dsa_moe, dsa_moe 4000-token sparse regime), all PASSED, 0 suspicious
  watcher lines; every process aborts at teardown (exit 134, the known eth-core timeout of this board).

## Performance accounting (review #1 P2-5)

Roofline = bytes one decode step must read per die (weights at their stored dtypes, KDA state read + write, DSA
latent rows for 2048 selected tokens) / 512 GB/s per die. Device = Tracy above. Replay = the shipped-build replay
wall. 1 user (the 1-slot KDA layers carry 2 state slots):

| Case | MB per die | Roofline µs | Device µs | Replay µs | Device / roofline |
|---|---|---|---|---|---|
| kda_dense | KDA in 36.3, out 8.9, decay 0.3, state 4.2, dense MLP 40.1 = 89.8 | 175 | 844 | 872 | 4.8x |
| kda_moe | KDA 49.7, router 1.6, 8 experts 28.3, shared 6.7 = 86.3 | 169 | 1053 | 1079 | 6.2x |
| dsa_moe | packs 11.3, q_b 6.7, W_UK / W_UV 4.5, wq_b 12.6, o 17.8, latent 2.1, MoE 36.6 = 91.6 | 179 | 1287 | 1368 | 7.2x |

Named gap terms of the 1-user kda_moe layer (1053 µs device, final Tracy): element-wise / data movement 323 (router
85, mHC glue 120, KDA gates 70, the rest below 10 µs each), matmuls 177 (41 to 64 % of DRAM bandwidth: decode M = 32
on 22 to 87 cores), Sinkhorn 142 (2 calls of 71 µs on one core: 8 permutation matmuls and 3 full-tile packs per
normalization remain), sparse matmuls 123, mHC kernels 94, CCL 80, KDA recurrence 60, MoE dispatch 35, norms 17; and
between device time and replay wall 26 µs of dispatch over 121 ops.

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
- `--compare N` (review #2): every variant runs 1 eager + N traced steps from a fresh state on the same inputs and is
  compared bit by bit (and by PCC / rms ratio) with the first variant.
- Provenance (review #2): the final numbers come from a code snapshot outside the live tree
  (`~/glm53-flash-bringup/s5r2_snapshot` = the stage-8 snapshot of d48b7fd06ad, which scored 92 / 100 / 100, plus the
  stage-5 files; md5s in `records/snapshot_md5.txt`), with `PYTHONSAFEPATH=1` and pytest `--import-mode=importlib` so
  that neither `python -m` nor pytest puts the live tree first. Other stages' uncommitted live edits do not enter.
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
| DSA q path (1 slot): q_b output [1, 4096] -> `ReshapeView` to [16, 256] heads (10.6 µs, 8 cores) -> `Transpose` (3.3) -> per-head bmm W_UK (17.5) -> `FillPad` + `Pad` to 32 heads (5.8 + 9.6) -> `UntilizeWithUnpadding` (9.1) -> sparse_sdpa | **56 µs** (review #2 P2-2) | one matmul of the head-masked, row-replicated q [32 heads, 4096] with W_UK viewed as [16*256, 512]: q_abs comes out with heads as rows, already padded to 32 heads, one row-major view from the sparse_sdpa input | bit-identical (the other heads' K blocks add exact zeros; in0 blocks of 8 tiles = one head) | **taken** (`dsa_q_mm`): -17 µs per DSA layer (`records/dsa_dm_ab.jsonl`), bit-identical over 9 steps, dense and sparse regime |
| DSA o path (1 slot): sparse_sdpa row-major [16, 1, 512] -> `Slice` (1.8) -> `TilizeWithValPadding` (**23.0**, each head's 1 row padded to a tile) -> per-head bmm W_UV (14.7) -> `Transpose` + `ReshapeView` concat (2.6 + 3.9) | **46 µs** | (1) one [16, 512] x [512, 16*256] matmul (W_UV relaid once), off-head blocks masked, rows summed; (2) the same products from W_UV viewed as [16*512, 256] with the 16 rows block-masked | (1) needs +2 MB of relaid W_UV per DSA layer per die; not bit-identical (different bf16 rounding; as exact as the bmm against fp64: rel L2 1.70e-3 vs 1.72e-3, `tools/dsa_out_probe.py`) | (1) -12 µs, (2) +33 µs (`records/dsa_dm_ab3.jsonl`): **not adopted** (12 µs per DSA layer for 22 MB of DRAM per die) |
| DSA sparse-index tail: start = 4 floor((t+1)/4), tail = start + j <= t ? start + j : sentinel (8 one-tile element-wise ops on 110 cores, 2 to 5 µs each) | **34 µs** per DSA layer | compute once per decode step in the first DSA layer, share through the step's input dict (fixed buffer) | the tail depends only on the step's positions | **taken** (`dsa_tail_shared`): -37 µs in each of the other 10 DSA layers (consumer emulated with a host input: `records/dsa_dm_ab2.jsonl`, bit-identical); the first version freed the shared tensor between trace captures and failed full-model teacher forcing (10 / 48 / 100): fixed to one never-freed buffer, gate below |
| DSA index expansion: top-512 pool ids -> `repeat_interleave` x 4 (`Typecast`, `Transpose`, `UntilizeWithUnpadding` 10.4, `Concat` 8.7 of 4 scalars, `TilizeWithValPadding` 5.4, `Transpose`) | 31 µs | 0 / 1 expansion matmul; hi / lo split of the ids | pool ids reach 32,767: the FPU reads fp32 at TF32 (11 bits), a direct matmul is off by up to 508 tokens; an exact split (ids = 256 a + b) needs ~10 element-wise ops of the same 2-5 µs | blocked (no op count below the current 6) |
| DSA open-pool online softmax (pooled index key): ~14 `BinaryNg` / `Unary` / `Typecast` on [1, 128] fp32, 3 `Ternary` + `Copy` state writes | ~80 µs (sum of the 25 ops) | (the review's "indexer RoPE chain" is this update; there is no RoPE in the indexer) a fused fp32 kernel | fp32 online-softmax state per slot, exact | not taken: a new kernel; recorded as open |
| DSA latent write: `UntilizeWithUnpadding` of the [1, 512] latent row on 1 core (9.9 µs) before `scatter_rows` | 10 µs | multi-core untilize | one tile row: the op runs it on 1 core | not taken (< 10 µs; `scatter_rows` takes row-major only) |
| KDA o-norm + gate: `rms_norm` [B*16, 128] fp32 -> relayout -> sigmoid-gate multiply -> typecast | 16 µs (1 user: rms_norm 4.5, typecast 2.0, reshape 3.0, gate multiply 4.7, typecast 1.9) | `ttnn.experimental.kda.sigmoid_gated_rms_norm` (head-first [B*H, T, V], T a whole tile: each (slot, head) row padded to its own 32-row tile, gate [B, 32, H*V]) | the op's T must be a multiple of 32 | measured (`records/kda_gn_ab.jsonl`): +28 µs (kda_moe 1 user), +52 (kda_dense), +228 (32 users), and wrong on kda_moe (PCC 0.08 vs the unfused path; kda_dense 0.9999999): **rejected** |
| KDA decode conv at 32 slots | fused op 158 µs (Tracy) | unfused chain | bit-identical both ways (`records/kda_conv_cmp.jsonl`) | fused faster up to 16 slots (-78 / -62 / -28 µs at 1 / 8 / 16), slower at 32 (+53): **`kda_conv_max_batch = 16`** |
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
| `ag_persistent` (review #1 P2-1) | kda_moe 1 user 1212 -> 1203, dsa_moe 1432 -> 1423, kda_moe 32 4433 -> 4362, dsa_moe 32 4783 -> 4759 (`records/ab6_det.jsonl`) | 1-layer gates all passed: 2 x 120 replays bit-identical, 200-replay stress, suite, watcher | **rejected** (taken in round 6, reverted 2026-10-18): the 45-layer model is not repeatable with it, eager as well as traced (8 slots, `records/s7_det_reduced_s8.log`: `eager_repeat_equal: false`), and 1-slot teacher forcing falls to top-5 32 %. With a persistent output ttnn skips the all-gather's start barrier (`all_gather_async_default_program_factory.cpp:752`: "use synchronize barrier semaphore" = `barrier_semaphore && !using_persistent_buffers`), so nothing orders a remote write into a buffer after its last reader on the receiving die. The one-layer tests rewrite each buffer at the same distance as the stack, so the reuse distance is not the difference; the root cause is not established (a timing-dependent cross-die effect). See "Stacked gate". |
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
| `sinkhorn_faces` (review #2 P1-2) | -30 / -30 / -29 µs (kda_dense / kda_moe / dsa_moe, 1 user), -43 / -35 µs (32 users) (`records/skf_ab.jsonl`) | bit-identical (probe, tests, 2 x 60 replays) | **taken** |
| `dsa_q_mm` (review #2 P2-2) | dsa_moe 1 user -17 µs (`records/dsa_dm_ab.jsonl`) | bit-identical (9 steps, pos 128 and 130,000) | **taken** |
| `dsa_tail_shared` (review #2 P2-2) | -37 µs in each DSA layer after the first (host-tail emulation, `records/dsa_dm_ab2.jsonl`) | bit-identical | **taken** (after the full-model fix, Stacked gate) |
| `dsa_o_mm` 1 / 2 (review #2 P2-2) | -12 / +33 µs (`records/dsa_dm_ab3.jsonl`) | PCC 0.99924 vs the bmm (rounding; equally exact vs fp64) | not adopted (1: +2 MB per DSA layer per die) |
| `kda_gnorm_op` (review #2 P2-2) | +28 / +52 / +228 µs (`records/kda_gn_ab.jsonl`) | wrong on kda_moe (PCC 0.08) | rejected |
| `kda_conv_max_batch = 16` (review #2) | fused conv -78 / -62 / -28 / +53 µs at 1 / 8 / 16 / 32 slots (`records/kda_conv_cmp.jsonl`) | bit-identical to the unfused chain at every batch | **taken** (fused up to 16 slots) |

## Matmul geometry per role (review #1 P1-2)

`tools/mc_matmul_sweep.py`, decode M = 32, µs per call (replay wall; compare within a row). Best DS = best
DRAM-sharded geometry over activation cores x in0 block x readers. "c / ib" in a spec is the core budget and in0 block
of `tp_common.create_matmul_1d_decode_progcfg`, which shapes the grid 11 cores wide: the final programs are below the
table. The "ttnn default" column is the sweep harness's own `ttnn.linear` program on one die, not the in-model program
(for the KDA in-projection Tracy shows 99.6 µs on 87 cores in the layer vs 162 µs in the harness); every kept role was
decided by the in-layer A/B (Candidates table), not by this column.

| Role [K x N], dtype | ttnn default | best 1D | best DS (geometry) | HiFi2 (best 1D) | Final |
|---|---|---|---|---|---|
| KDA in [4096 x 8336] bfp8 | 162.1 (harness; in model 99.6 on 87 cores) | 109.9 (29 cores, in0 4) | 140.1 (32 act. cores, in0 2, 3 readers) | 110.3 | default (in layer 1214 -> 1216: no gain) |
| KDA out [2048 x 4096] bfp8 | 48.2 | 33.7 (16, 4) | 45.4 (8, 4, 2 r.) | 33.3 | 1D 16 / 4 |
| DSA o [4096 x 4096] bfp8 | 84.0 | 57.5 (16, 4) | 76.1 (16, 4, 2 r.) | 56.8 | 1D 16 / 4 |
| DSA q_b [1536 x 4096] bfp8 | 39.4 | 27.9 (16, 4) | 38.2 (8, 3, 2 r.) | 27.5 | 1D 16 / 4 |
| DSA wq_b [1536 x 4096] bf16 | 43.6 | 41.2 (16, 4) | 55.2 (8, 6, 1 r.) | 40.6 | 1D 16 / 4 |
| pack A [4096 x 2048] bfp8 | 82.4 | 34.8 (32, 8) | 45.5 (4, 8, 2 r.) | 34.2 | 1D 32 / 8 |
| shared gate/up [4096 x 1024] bfp8 | 66.8 | 23.8 (32, 16) | 29.5 (4, 8, 2 r.) | 23.3 | 1D 32 / 16 |
| shared down [512 x 4096] bfp8 | 21.3 | 16.1 (16, 4) | 22.4 (4, 4, 2 r.) | 15.6 | 1D 16 / 4 |
| dense gate/up [4096 x 6144] bfp8 | 104.4 | 84.0 (16, 4) | 105.8 (32, 2, 3 r.) | 83.0 | 1D 16 / 4 |
| dense down [3072 x 4096] bfp8 | 66.1 | 45.6 (16, 4) | 61.1 (8, 6, 2 r.) | 45.2 | 1D 16 / 4 |

Final 1D programs (OPT-004: real grid, `per_core_N`, output subblock, in0 block; Tracy agrees, e.g. tt-perf-report
"in0_block_w=4 and output subblock 1x3" on the 22-core rows):

| Role (spec) | Grid (cores used) | per_core_N | Subblock | in0 block |
|---|---|---|---|---|
| KDA o, DSA o, DSA q_b, DSA wq_b, shared down, dense down (`1d:c16:ib4`) | 11 x 2 (22) | 6 | 1 x 3 | 4 |
| dense gate/up (`1d:c16:ib4`) | 11 x 2 (22) | 9 | 1 x 3 | 4 |
| pack A q_a / kv_a (`1d:c32:ib8`) | 11 x 3 (32) | 2 | 1 x 2 | 8 |
| shared gate/up (`1d:c32:ib16`) | 11 x 3 (32) | 1 | 1 x 1 | 16 |
| DSA q_abs (`dsa_q_mm`, [4096 x 512]) | 11 x 2 (16) | 1 | 1 x 1 | 8 (one head's K) |

Core budgets that round to the same `per_core_N` give the same program (e.g. c16 and c22 for N = 128 tiles), so some
sweep candidates are duplicates; the table lists each distinct program once.

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
| all-gather of the mix partials, dim 1 | [T, 64] -> [4, T, 64] | fp32 | L1 at decode (`ccl_l1`), DRAM in prefill | no (`ag_persistent` rejected: breaks the stacked model) | 12.0 -> 11.0 |
| all-gather of h, dim 3 | [T, 1024] -> [T, 4096] | bf16 | L1 at decode, DRAM in prefill | no (same) | 17.8 -> 12.6 |
| reduce-scatter, dim 3 | [T, 4096] -> [T, 1024] | bf16 (intermediate DRAM) | L1 at decode | no (122B: no gain, `~/qwen35-122b-bringup/logs/05-r2-ccl-probe.log`) | 16.2 -> 16.7 |

Final values = the mean of the 2 calls per layer in `tracy/after/test_perf_traced_decode_kda_moe-1_` (review #2 run,
final default, non-persistent outputs in L1). The `ag_persistent` runs of round 7 (DRAM persistent buffers) measured
7.1 / 10.8 / 15.0 µs; it is rejected for correctness, not speed (Candidates table). Single calls vary: one
32-user reduce-scatter took 56.6 µs in the final kda_moe-32 Tracy (the other 19.4).

Ring on a FABRIC_1D fabric: `ttnn.get_usable_topology` returns Ring only when the fabric graph has the wraparound link
(both p300 board-to-board links up; Linear otherwise). FABRIC_1D_RING adds dateline deadlock avoidance for traffic
that crosses the wrap. These collectives are latency-bound single-round transfers (one chunk per direction per link);
the 122B port has served Ring on FABRIC_1D on this board, and every replay here (thousands, incl. 2 x 120-replay
determinism runs and 200-replay stress runs) completed bit-identical. `GLM53_FABRIC=FABRIC_1D_RING` (tests/harness.py) selects the ring fabric for an A/B. Result (`records/ring_fabric7.jsonl`, shipped `DEFAULT_MC`, 1 user, replay µs, FABRIC_1D_RING vs FABRIC_1D): kda_moe 1104.8 vs 1103.1, dsa_moe 1407.6 vs 1407.0. The difference is in the noise (< 0.2 %).
Prefill on the review-2 default (review #2 "Other concerns": the 2048 x 4096 payloads are multi-chunk ring traffic
across the wrap, which FABRIC_1D has no dateline for; `records/pf_ring_r2.jsonl`, host wall, best of 3, separate
processes): kda_dense 11.83 -> 11.52 ms, kda_moe 27.70 -> 28.60 ms, dsa_moe 31.95 -> 32.46 ms on FABRIC_1D_RING, outputs
bit-identical (max |diff| 0). No measurable cost either way (+-3 %, the run-to-run range of these walls). The decoder
does not pick the fabric: the model opens the mesh (stage 6 / 7, `tests/harness.py` `GLM53_FABRIC` for the layer
tests). Recommendation for the model owner: FABRIC_1D_RING, which removes the deadlock question for multi-chunk ring
prefill at no measured cost; nothing has hung on FABRIC_1D in this stage.

Rejected collective families (measured on this board by the 122B port, same ops and sizes): fused all-gather + matmul
(`~/qwen35-122b-bringup/logs/05-agmm-probe*.log`: 0.14 to 0.60 ms vs 0.07 ms separate; here an RMSNorm also sits between
the all-gather and every consumer matmul), fused matmul + reduce-scatter (`05-mmrs-probe*.log`: 2D configs only,
<= 8 compute cores for one tile of rows), bfp8 payloads (`05-r2-ccl-probe.log`: -1.4 / -2.6 µs, plus a typecast),
all-reduce instead of reduce-scatter (4x the bytes and the mHC work on every die).

## Stacked gate (review #2 P2-1)

The `ag_persistent` episode showed that the stage's single-layer gates (suite, replay stress, determinism, watcher) do
not cover hazards that only appear with many layers and several captured traces. A second case appeared in this
rework: the first `dsa_tail_shared` freed a tensor that an earlier captured trace still read; every single-layer A/B
was bit-identical (one trace), and full-model teacher forcing fell to 10 / 48 / 100. Every `DEFAULT_MC` change,
collective option or buffer-sharing option now passes, before it becomes the default:

1. Full-model teacher forcing, 1 slot (`tools/s7_probe.py --slots 1 --do ready`, the stage-7 reference), at the stage-7
   level (92 / 100 / 100), from a code snapshot outside the live tree (`~/glm53-flash-bringup/s5r2_snapshot`, md5s in
   `records/snapshot_md5.txt`).
2. Stacked repeatability, 8 slots (`tools/s7_probe.py --slots 8 --do ptcheck`): eager and traced runs each repeated,
   tokens equal, on a reduced stack that holds a producer and a consumer DSA layer (layers 0, 3-7) and on the full
   model.

Results for the final default (`records/s5r2_d4.log`):

| Run | Options | Teacher forcing top-1 / 5 / 100 | Prefill check | Repeatability |
|---|---|---|---|---|
| control (round-6 default, review-2 files, flags off) | `DEFAULT_MC` before review #2 | 92 / 100 / 100 | 91 / 100 / 100 | - |
| final | + `sinkhorn_faces`, `dsa_q_mm`, `dsa_tail_shared`, `kda_conv_max_batch = 16` | 92 / 100 / 100 | 91 / 100 / 100 | 8 slots, eager + traced x 2: equal on layers 0, 3-7 and on the full model; tokens = stage 7's (`records/s5r2_det_full_final.log`) |
| first `dsa_tail_shared` (bug) | same | 10 / 48 / 100 | 91 / 100 / 100 | - |
| stage 7, `ag_persistent` on | round-6 default | 14 / 32 / 90 (8 slots not repeatable, eager and traced) | - | fail |

`ag_persistent` stays rejected on that 45-layer evidence (`records/s7_det_reduced_s8.log`); its root cause is not
established (see Candidates). The final default's non-persistent all-gathers keep ttnn's start barrier.

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

## Sinkhorn (review #1 P1-1, review #2 P1-2)

`mhc_split_sinkhorn` splits token tiles over cores (prefill: 64 cores at 2048 rows); decode is one tile on one core, 40
sequential normalizations of 8 permutation matmuls + an fp32 reciprocal + 2 multiplies + 3 packs each. Splitting
users over cores cannot help (the tile is the SFPU / FPU work unit).

- HiFi3 permutation sums (`sinkhorn_perm_fidelity = 3`): the permutation operand is 0 / 1, so the 4th fidelity phase
  (low srcA bits x low srcB bits) only adds zeros: bit-identical to HiFi4 for any input (verified at T = 1 to 2048,
  sharded and interleaved); HiFi2 is not exact (0.5 to 1.2 % error) and the op now refuses it outside probes.
  89.4 -> 85.8 µs per call (Tracy).
- Face-restricted normalizations (`sinkhorn_faces`, review #2): the n x n = 16 used columns of m are columns 0-15, so
  the SFPU ops of every normalization (eps add, fp32 reciprocal, normalizing multiply, the hi / lo split) run on face 0
  only (T <= 16 tokens: `VectorMode::None`) or on faces 0 and 2 (`VectorMode::C`). The padding of m is zeroed once
  after the exp (an SFPU store of 0 to the unused faces); the split writes exact zeros to the unused faces of m_hi /
  m_lo, so no stale DST value reaches a permutation matmul, and every padding product is 0 x finite = 0. The
  permutation tiles map the 16 used columns among themselves and the padding columns only to themselves (identity), so
  the padding of m stays exactly 0 and comb's padding comes out 0 (it was ~1). The used entries are bit-identical:
  `tools/sinkhorn_fidelity_probe.py` (`records/s5r2_sk_probe.log`) at T = 1 / 8 / 16 / 17 / 32 / 96 / 2048, repeat
  calls identical; `tests/test_mhc_split_sinkhorn.py` 7 / 7 and the upstream tests 4 / 4 with faces on
  (`records/s5r2_a.log`). Per call (probe wall): 88.9 -> 73.7 µs at T = 1, 88.9 -> 79.5 µs at T = 32; in the layer
  (`records/skf_ab.jsonl`, 2 x 60 replays bit-identical): -30 / -30 / -29 µs at 1 user (kda_dense / kda_moe /
  dsa_moe), -43 / -35 µs at 32 users. Tracy on the final default: 85.8 -> 70.8 µs per call at 1 user (face 0), 85.9 -> 76.6 µs at 32 users (faces 0 / 2).
- What stays: the 8 permutation matmuls and the 3 packs per normalization still move whole 32 x 32 fp32 tiles. Packing
  or multiplying face 0 alone needs 16 x 32 (or 16 x 16) tile formats for the m / m_hi / m_lo circular buffers, a
  matmul of a 16-row in0 tile against the 32 x 32 permutation tile, and a packer reconfigured per CB inside the loop;
  this kernel's API (`matmul_tiles`, `pack_tile` on the CB's tile shape) does not express that, so it is the next
  kernel step, not an option here.

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

`DEFAULT_MC` adds per die: the 1-slot KDA padding slot when `max_batch = 1` (+1 MiB of recurrent state per KDA layer: 34 MiB, ~2.9K tokens of DSA cache at 11,968 B per
token), the `dsa_q_mm` head mask ([32, 4096] bf16, 0.25 MB per DSA layer; W_UK is a view), the shared sparse-index tail
(one [32, 32] fp32 tile per model), nothing else (the packs replace the separate weights; no persistent collective
buffers: `ag_persistent` is rejected). The DSA latent cache cannot drop to bfp8 here:
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
| `sigmoid_gated_rms_norm` for the decode o-norm + gate | +28 to +228 µs, wrong on kda_moe (`records/kda_gn_ab.jsonl`) | slower |
| DSA o path as one matmul (`dsa_o_mm`) | -12 µs (relaid W_UV), +33 µs (view) | +2 MB per DSA layer per die for 12 µs; different bf16 rounding |
| persistent all-gather outputs (`ag_persistent`) | -9 to -71 µs per layer | breaks the stacked model (see Candidates, Stacked gate) |
| pairs / masked MoE at 32 users | +2.4 ms / +19 µs | slower |
| 1024-row KDA prefill chunks | +1.0 to 1.5 ms per 2048 tokens | slower |
| tt-perf-report "Output subblock 1x1" | one output tile per core is the widest 1D grid | - |
| tt-perf-report "place input 0 in L1" | collective outputs moved (`ccl_l1`) | the rest are 64-256 KB activations |

## Open items

These are measured costs that stay; none is a deferred option of this stage.

- Sinkhorn after `sinkhorn_faces`: 71 µs per call at 1 user (2 per layer). What remains is the 8 permutation matmuls
  and 3 full-tile packs per normalization; restricting those to face 0 needs 16-row tile formats for the m buffers and
  a matching matmul / pack path (see Sinkhorn).
- Element-wise glue that needs new fp32 kernels: MoE router glue (TopK on 1 core 34 µs, FillPads), mHC glue (~120 µs per
  layer), the DSA open-pool online softmax (~80 µs per DSA layer), the DSA index expansion (31 µs; exact alternatives
  are not shorter).
- 32-user MoE glue: sub-tile expert groups (kernel work in the grouped sparse-matmul factory).
- Long-context decode: +561 µs per DSA layer at position 130,000 (sparse_sdpa over 2,048 gathered tokens, top-512
  selection over ~32.5K pooled keys); profiled above, not optimized here.
- `ag_persistent`: root cause of the stacked-model failure not established (rejected, Stacked gate).
- 32-user decode Tracy varies by up to 5 % between runs (the DRAM-bound sparse expert matmuls and row gather).

## Artifacts

- Code: `tt/multichip_decoder.py` (`MCOpts`, `DEFAULT_MC`, `MeshCCL`, `DSMatmul`, `IL1DMatmul`, `RoleMatmuls`,
  `sharded_rms_norm`, `MultichipDSA`, `MultichipKDA`, `MultichipMLP`, shared-expert and router hooks),
  `tt/fused_decoder.py` (`_in_proj`, `_conv_step` hooks), `tt/moe.py` (collective hooks, top-k padding hook),
  `tests/harness.py` (`GLM53_FABRIC`), `tests/test_multichip_decoder.py::test_single_slot_kda_decode`,
  `tests/test_kda_decode_conv.py`, `tools/mc_decode_ab.py` (`--compare`), `tools/mc_prefill_ab.py`,
  `tools/mc_matmul_sweep.py`, `tools/sinkhorn_fidelity_probe.py`, `tools/dsa_out_probe.py`; `tt/fused_decoder.py`
  `_gated_norm` hook.
- ttnn: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/kda_decode_conv/` (new op),
  `deepseek_prefill/mhc_split_sinkhorn/` (`perm_fidelity`, `faces`), `deepseek_prefill/CMakeLists.txt`,
  `experimental/experimental_nanobind.cpp`.
- Tracy + tt-perf-report (with advice, `--active-experts 8`): `tracy/before/<case>/`, `tracy/after/<case>/`,
  `tracy/after_pos130000/` (the raw op list as `ops.csv.xz`).
- Watcher: `watcher/<case>/`.
- Records: `records/` (A/B jsonl, sweep, stress, round logs, suites, watcher log).
- Commands:
  ```
  GLM53_TP=4 pytest models/autoports/zai_org_glm_5_3_flash/tests/test_multichip_decoder.py
  pytest models/autoports/zai_org_glm_5_3_flash/tests/test_mhc_split_sinkhorn.py models/autoports/zai_org_glm_5_3_flash/tests/test_kda_decode_conv.py
  GLM53_TP=4 python -m models.autoports.zai_org_glm_5_3_flash.tools.mc_decode_ab --kind kda_moe --batch 1 --base default --determinism 200 --variants "fin:" --out stress.jsonl
  python -m models.autoports.zai_org_glm_5_3_flash.tools.mc_matmul_sweep --out mm.jsonl
  ~/glm53-flash-bringup/scripts/s5_tracy.sh after
  # full-model gate (from the snapshot directory, so that python -m does not put the live tree first)
  cd ~/glm53-flash-bringup/s5r2_snapshot && GLM53_TP=4 GLM53_CACHE_DIR=~/.cache/glm53_flash python -m models.autoports.zai_org_glm_5_3_flash.tools.s7_probe --slots 1 --seq-len 4096 --do ready --reference <readiness_aime24_chat.refpt>
  ```
