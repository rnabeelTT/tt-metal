# GLM-5.3-Flash (E192) optimized decoder

Stage 3 of the tt-model-bringup flow. `tt/optimized_decoder.py` (`OptimizedDecoder`) is the fused decoder
(`tt/fused_decoder.py`) with a precision policy, decode and prefill program configurations, and the stage-5 decode
options that need no collective, for one die. Contract, state, paging and semantics are those of
`tt/functional_decoder.py`. The policies are `OptConfig` entries in `CANDIDATES`; `SELECTED` is the default.

Review history (reports in `~/glm53-flash-bringup/artifacts/stage3_review_<n>.md`; the review #3 report exists only
as a summary in `PLAN.md`):

- Review #1 changed the reference, the default and the evidence.
- Review #2 added more single-die options, the fidelity trials of the bfp8 projections and the P2 items.
- Review #3 added the sinkhorn faces kernel, the stage-5 decode role specs, the watcher diagnosis of the KDA in-proj
  split and the batched-reuse matmul diagnosis.
- Review #4 made `dsa_q_mm` legal on one die (measured, not taken), added the sparse_sdpa head split, two row-major reshapes, the per-kind KDA prefill
  fidelity and the refreshed accounting ("Review #4").

The name `opt_s5_pf` has held four one-die policies: `opt_s5_pf_r1` (review #1), `opt_s5_pf_r2` (review #2),
`opt_s5_pf_r3` (review #3) and the review #4 default (below). The TP4 path and the weight-cache key never changed.
`GLM53_PRECISION` (the stage-8 policy override) applies only to `MultichipDecoder` / `Glm53Model`. The earlier text of
this README is in git history (commits 21dc7532f26, d48b7fd06ad, 656c998e3df, ea27eda6b7e). The work log has the
failures and the order of events. Logs of `~/glm53-flash-bringup/logs/` are copied per review to `logs_r1/` .. `logs_r4/`.
Tracy runs are in `candidates_r1/` .. `candidates_r4/`.

## Selected policy (`SELECTED` = `opt_s5_pf`, review #4 contents)

Device time of the review #4 default (`candidates_r4/final`, one Tracy run per case):

- Decode, 1 user, position 128: kda_dense 1705, kda_moe 1951, dsa_moe 2024 µs.
- Decode, 1 user, position 16384 (sparse attention): dsa_moe 2374 µs (review #3 default, same session: 2587 µs).
- Decode, 32 users: 2705 / 11515 / 11210 µs.
- Prefill, 2048 rows: 20541 / 65011 / 74967 µs. The 128-row prefill was not rerun (review #2 runs: 2739 / 14025 /
  14555 µs).

| Tensor group / op | Fused decoder (stage 2) | Optimized | Evidence |
|---|---|---|---|
| Routed experts gate / up / down | bfp8, HiFi4 | **bfp4, LoFi** | memory (below); accuracy and fidelity: "Accuracy" |
| Shared expert | bfp8, HiFi4 | bfp8, HiFi4 | HiFi2 left the shared output 0.8 % low (activation-mantissa truncation) |
| KDA input / output projections | bf16 | **bfp8** | weight-read bound; bfp4 rejected ("Precision trials") |
| DSA W_UK / W_UV (absorbed per-head) | bf16, default config | **bfp8, full-grid reuse config** (decode rows) | `tools/dsa_bmm_probe.py`, `tools/s3r_probe_dsa.py` |
| KDA / DSA output projections (decode) | default config | **1D config** (`moe_opt.DenseDecodeMatmul`) | 147 -> 100 µs (KDA) |
| Decode RMSNorms (layer norms, DSA q_a / kv_a norms) | 1 core | **width-sharded** (32 / 24 / 16 cores) | 136 -> 18 µs per KDA layer, 170 -> 29 µs per DSA layer |
| DSA x-projections q_a, kv_a, indexer wk, gate, weights | 5 matmuls on 1-48 cores | **2 packed matmuls on 1D decode configs**, packed weights shared with prefill | dsa_moe 1 user -363 µs with the norms |
| DSA q_b / wq_b decode, MoE router decode | default (103 / 6 cores) | **1D configs** | router -31 µs per MoE layer |
| DSA `flash_mla_prefill` | no program config | **q / k chunks 128** | 22.9 -> 6.7 ms at 2048 rows (probe) |
| DSA o_proj and dense-MLP down, prefill rows | default config | **explicit 2D config, full grid** | 7.7 -> 3.7 ms, 5.8 -> 2.8 ms (probe, bit-exact) |
| KDA decode conv (12-op chain) | unchanged | **one `kda_decode_conv` op for decode batches <= 16 slots** (the chain at 32) | -161 µs per KDA layer at 1 user; op 140 / 256 / 484 µs vs chain 265 / 266 / 264 µs at 8 / 16 / 32 slots (review #3) |
| Sinkhorn | 40 normalizations of the whole tile, permutation matmuls HiFi4 | **valid faces only (`sinkhorn_faces`) + permutation matmuls HiFi3** (both bit-identical) | -7 µs (HiFi3), -30 µs per layer at 1 user (faces, review #3) |
| Decode role programs (`tt/multichip_decoder` specs) | ttnn default / 1D | **1D programs on 22 cores** (`mlp_dn`, `dsa_o`, `kda_out` `1d:c16:ib4`; shared expert `shared_gu` `1d:c32:ib16`, `shared_dn` `1d:c16:ib4`) | -73 / -7 / -3.4 .. -4.6 / -2 .. -3 µs |
| DSA o_proj and dense-MLP gate|up / down, prefill rows | HiFi4 | **HiFi2 with packer L1 accumulation** | prefill 2048 kda_dense 26987 -> 20641 µs, dsa_moe 77076 -> 75000 µs (review #2 P1-B) |
| KDA prefill in / out projections, KDA + MoE layers only | HiFi4 | **HiFi2** (packer L1 accumulation) | review #4 P2-2: L4 prefill scale 0.9938 .. 0.9976; L0 (KDA + dense) keeps HiFi4 |
| DSA q_abs, 1 slot | reshape + permute + per-head bmm + untilize | unchanged: `dsa_q_mm` is legal on one die now (program for the 64 query rows) but 9 µs slower | review #4 P1-1 |
| DSA sparse_sdpa, 1 slot | 64 heads in 1 query row (1 core) | **2 query rows of 32 heads (2 cores)** | review #4 P2-1; bit-identical; about -200 µs per DSA layer at position 4000 |
| DSA indexer query reshape, MoE routing-weight reshape (decode) | tiled reshape | **row-major round trip** | review #4 P2-1; bit-identical |
| mHC, KDA recurrence | unchanged | unchanged | fp32-exact stage-1 / 2 kernels; dispositions in the audit |

`OptConfig` fields: `kda_in_dtype`, `kda_out_dtype`, `moe` (expert dtypes, fidelities, sparse geometry),
`dsa_bmm_dtype`, `dsa_bmm_reuse`, `out_proj_1d`, `single_die_mc` (the pinned stage-5 option set),
`prefill_cfg`, `single_die_r2` (review #2 single-die options), `single_die_r3` / `single_die_specs` (review #3),
`kda_prefill_fidelity` / `kda_prefill_kinds`, `dsa_q_mm_ib`, `sdpa_head_split`, `qi_rm`, `route_rm` (review #4),
`kda_in_split_ds`, and the trial fields `dsa_proj_dtype`, `mlp_dtype`, `mlp_fidelity`, `mix_chunks_decode`,
`out_proj_ds`, `prefill_fidelity`. All single-die options are gated on
one die (`tp == 1`, not a `MultichipDecoder`): the TP4 path, its weights and its weight-cache key are unchanged. The stage-5
option set is pinned in `OptimizedDecoder.from_state_dict` (`MCOpts(norm_sharded, dsa_pack, pack_a_1d, pack_b_1d,
dsa_1d, router_1d, ds_shared)`); it does not follow `multichip_decoder.DEFAULT_MC`, which the stage-5 rework still
changes.

**Memory forces bfp4 experts.** `doc/multichip_decoder/memory_budget_tp4.json`: routed experts 28.54 GB per die at
bfp4 against a 34.2 GB die; bfp8 needs about twice that.

## Accuracy: the quantization-aware reference

The bfp4 experts lose 1 to 1.5 % update PCC per MoE layer against the checkpoint. Review #1 rejected the earlier bar
shift (every PCC bar of the MoE layers down by 0.015, scale bounds 3 % / 6 %): it was wider than the measured floor
and no longer caught the stage-1 routed-scale bug. The optimized suite now compares the MoE layers with **the HF
layer on the quantized weights** and holds them to the **functional bars** (update PCC 0.995, sequence scale 1 +- 0.01,
token scale 1 +- 0.05, decode mean 1 +- 0.01, synthetic MoE 0.98). No `BAR_OVERRIDES` remain.

- `tests/hf_ref/bfp.py`: torch emulation of the tt-metal host block-float packing (shared exponent per 16 elements of
  a tile face row, round to nearest even, saturating mantissa; `blockfloat_common.cpp`). `tools/bfp_check.py`: **0
  mismatches** against `ttnn.from_torch` -> `to_torch` for layer-4 experts 0 / 57 / 191, the shared expert, the 4-D
  packed up|gate layout of `tt/moe.py` (bfp4 and bfp8), and the layer-0 dense MLP (`logs/s3r_bfp_check.log`).
- `tests/hf_ref/hf_quant_ref.py`: the unmodified HF layer with the weights of `OptConfig.quant_spec()` rounded
  (routed experts bfp4, shared expert bfp8), on the base reference's input stream; the router near-tie y_swap is
  computed on the quantized layer too. Cache key: its own code digest (`client.quant_reference`).
  `tools/quant_ref_requests.py` records the suite's 97 requests; `tools/compute_quant_refs.py` computes them in
  per-layer batches (`tests/hf_ref/hf_quant_batch.py`; a batch result is bit-identical to the single-request script,
  `logs/s3r_refs.log`).
- Each MoE check also prints its numbers against the checkpoint reference (the quantization delta; summary at module
  teardown).

### Routed scale: fidelity (`tools/moe_scale_probe.py --policy opt --modes prefill,decode,decode1 --fidelities ...`)

Routed output of one layer, 128 rows drawn from the recorded statistics of the real input stream (synthetic draws,
not recorded activations), TT against the torch MoE on the bfp4-rounded weights (`ref_q4`). `prefill` and `decode`
(32-row tiles) take the grouped sorted path, `decode1` (one row per call) the B=1 gather path.
`logs/s3r_q4probe_l4.log`, `logs/s3r_q4probe_l3.log`.

| Routed fidelity | L4 prefill / 32-row: rms vs ref_q4 (PCC) | L4 1-row | L3 prefill / 32-row | L3 1-row |
|---|---|---|---|---|
| LoFi (= HiFi2, bit-identical) | 0.9951 (0.999969) | 0.9921 (0.999975) | 0.9954 (0.999634) | 0.9924 (0.999639) |
| HiFi3 (= HiFi4, bit-identical) | 1.0035 (0.999982) | 1.0005 (0.999987) | 1.0038 (0.999646) | 1.0009 (0.999651) |

ref_q4 against the checkpoint weights: L4 rms 1.0275, PCC 0.982576; L3 1.0245, 0.983650. LoFi / HiFi2 drop the low
activation mantissa bits, so the routed output reads 0.5 % (grouped) to 0.8 % (gather) low. HiFi3 removes the bias on
the gather path (+0.05 %) and leaves +0.35 % on the grouped path (the same +0.36 % the bfp8 HiFi4 path read in stage 1:
a property of the grouped path, not of the fidelity).

**Decision: LoFi kept; the bias is accepted and bounded.** Cost of HiFi3 on the routed experts (Tracy,
`candidates_r1/opt_rhifi3` vs the same-day `candidates_r1/opt`): kda_moe 1 user 2286 -> 2315 µs (+1.3 %; sparse matmuls
445 -> 472), 32 users sparse matmuls 7350 -> 7564 µs (+2.9 %, layer 11755 -> 11944), prefill 2048 rows 66854 -> 70196 µs
(+5.0 %; sparse 38360 -> 41660). HiFi4 reads bit-identical to HiFi3 and costs more (`candidates_r1/opt_rhifi4`: 2331 / 12423 / 72683 µs). With LoFi the MoE layer updates read 0.2 to
0.8 % low against the quantized reference (prefill sequences L3 0.9916 .. 0.9957, L4 0.9957 .. 0.9984), inside the
functional 1 % bound, and the injected stage-1 bug (-2.8 %) still fails it (below). The datatype sweep (stage 8) owns
the model-level trade-off; `opt_rhifi3` is its HiFi3 point.

### Suite results (functional bars, quantization-aware reference)

| Run | Policy | Result | Lowest real-weight update PCC (no y_swap) | Prefill sequence scale | y_swap matches |
|---|---|---|---|---|---|
| `logs/s3r_opt_suite.log` | `opt_s5` (LoFi) | 44 / 44 | L3 0.995625, L4 0.996759, L0 0.999933 | L3 0.9916 .. 0.9957, L4 0.9957 .. 0.9984, L0 0.9948 .. 0.9969 | 3 of 298 (cap 8) |
| `logs/s3r_suite_opt_s5_b.log` | `opt_s5` (repeat) | 44 / 44, every check line identical | same | same | same |
| `logs/s3r_pcc_opt_rhifi3*.log` | `opt` + HiFi3 (subset) | 15 / 15 | L3 0.995567, L4 0.996246 | L3 0.9985 .. 1.0027, L4 1.0000 .. 1.0024 | 1 of 249 |
| `logs/s3r2_f3_suite.log` | **`opt_s5_pf` (review #2 final)** | **44 / 44** | L3 0.998785, L4 0.996759, L0 0.999933 | L3 0.9945 .. 0.9973, L4 0.9957 .. 0.9984, L0 0.9927 .. 0.9940 (prefill HiFi2) | 3 of 298 (cap 8); decode means >= 0.9929 |
| `logs/s3r_final_suite.log` | `opt_s5_pf` (review #1 contents) | **44 / 44** | L3 0.998784, L4 0.996759, L0 0.999933 | L3 0.9929 .. 0.9960, L4 0.9957 .. 0.9984, L0 0.9948 .. 0.9969 | 3 of 298 (cap 8) |

Synthetic MoE layers (`opt_s5`): MoE on the HF FFN input PCC L3 0.998505, L4 0.999954 (bar 0.995); layer update
L4 0.990369 (bar 0.98). Quantization delta (port against the checkpoint reference, 332 checks): min update PCC
0.973314 (synthetic L4), update scale 0.9757 .. 1.0639 (single tokens).

The earlier README minimum was wrong: it gave 0.99049; the review #1 log (checkpoint reference, shifted bars) had
0.984536 (L4 batched decode) and 0.986295 (L3 paged decode). Before / after against the stage-2 fused suite, per test
group: work log, "PCC before / after".

### The stage-1 routed-scale bug is caught again

`GLM53_FAULT_ROUTED_SCALE=0.9722` (routed output x 0.9722) on the MoE tests of the optimized suite
(`logs/s3r_fault_routed.log`): **14 failed, 3 errors, 5 passed**. Every multi-row real-weight prefill of L3 / L4 fails
the 1 % sequence bound (L3 0.9694 .. 0.9767, L4 0.9796 .. 0.9879); the decode means fail at teardown
(`test_batch32_all_active[dsa_moe]` 0.9741, `test_inactive_slot_state` 0.9759 / 0.9898). Passed: `test_lengths[dsa_moe-1]`
(one-row prefill: token bound), the B=32 per-token checks (their mean fails), `test_inactive_slot_state` per-token
checks (mean fails).

## Performance (one die, warmed; Tracy + tt-perf-report with advice)

Device time, µs: decode = one traced replay (report sum / 3), prefill = the signposted 2048-row pass. Runs:
`candidates_r1/<policy>/`, `candidates_r2/<policy>/` (`ops.csv`, `perf_report.csv`, `perf_report.txt` with advice).
Review #2 runs pin the input seed (`GLM53_PERF_SEED=0`, `tests/test_perf_decoder.py`): the 32-user spread of review #1
(+-4 % between sessions) came from data-dependent active-expert counts (`MoeGatherRows` moved with the sparse
matmuls). B=1 repeats within 2 µs (`opt_s5_pf` 1647 / 1649, 1890 / 1892 µs in two sessions).

Review #2 default (`opt_s5_pf_r2`: `opt_s5_pf` at review #2, with `single_die_r2` and `prefill_fidelity=HiFi2`; decode `candidates_r2/final/`,
prefill `candidates_r2/opt_s5_pf_final/` (measured under the `opt_s5_pf_r2b` policy: the review #2 options with the KDA in-proj split, which is decode-only, so the prefill rows are those of the review #2 default), advice on), against the stage-2 fused decoder (`doc/fused_decoder/tracy`,
3-replay means), the pre-review stage-3 default (`opt`, `candidates_r1/opt`) and the review #1 default
(`opt_s5_pf_r1`, same-day control `candidates_r2/opt_s5_pf_r1`, seed pinned; its prefill from `tracy_r1/`):

| Case | Stage 2 fused | `opt` | Review #1 default | **Final** | vs fused | vs review #1 | Host wall, final (ms) |
|---|---|---|---|---|---|---|---|
| decode kda_dense, 1 user | 2443 | 2083 | 1970 | **1727** | -29.3 % | -12.3 % | 2.390 |
| decode kda_moe, 1 user | 2979 | 2286 | 2140 | **1974** | -33.7 % | -7.8 % | 2.860 |
| decode dsa_moe, 1 user | 3347 | 2421 | 2058 | **2047** | -38.8 % | -0.5 % | 3.229 |
| decode kda_dense, 32 users | 3262 | 2901 | 2790 | **2713** | -16.8 % | -2.8 % | 7.413 (one run; device 2.713) |
| decode kda_moe, 32 users | 15898 | 11755 | 11563 | **11535** | -27.4 % | -0.2 % | 12.304 |
| decode dsa_moe, 32 users | 15590 | 11276 | 11246 | **11263** | -27.8 % | +0.2 % | 12.171 |
| prefill 2048 kda_dense | 30000 | 29968 | 26853 | **20641** | -31.2 % | -23.1 % | 22.04 |
| prefill 2048 kda_moe | 89115 | 66854 | 66588 | **66808** | -25.0 % | +0.3 % | 68.26 |
| prefill 2048 dsa_moe | 119821 | 97988 | 77052 | **75000** | -37.4 % | -2.7 % | 75.77 |
| prefill 128 kda_dense / kda_moe / dsa_moe | 3200 / 19679 / 19824 | - | 2959 / 13968 / 14691 | **2739 / 14025 / 14555** | | | 4.63 / 15.99 / 15.36 |

The table is the review #2 default. The review #4 default: "Review #4" and "Performance accounting". Review #3 default (`candidates_r3/t_r3_specs` = the review #3 policy for every case
but 1-user dsa_moe, `candidates_r3/final/` for that one; same-day control `candidates_r3/opt_s5_pf_r2`): see "Review #3". The `opt_s5_pf_r2b` candidate (with the KDA in-proj split) measured
1647 / 1890 / 2045 µs but trips a watcher assert (below). The kda_dense 32-user host wall of 7.4 ms is one run under host
load (device 2.713 ms); the other host walls are within 0.7 .. 1.2 ms of device time.

### Review #2: single-die options (P1-A), per-op deltas against the same-day review #1 default

`candidates_r2/opt_s5_pf_r1` -> `candidates_r2/opt_s5_pf` (= `opt_s5_pf_r2b`, the P1-A options + the KDA in-proj split), device µs per layer:

| Option | kda_dense 1 | kda_moe 1 | dsa_moe 1 | kda_dense 32 | kda_moe 32 | dsa_moe 32 | Decision |
|---|---|---|---|---|---|---|---|
| `kda_decode_conv` (one op for the 12-op chain; ternary -144, binary -45, typecast -39, unary -17, slice -9, op +93) | -161 | -162 | - | (off) | (off) | - | **taken for decode batches <= 8 slots** (review #3: <= 16) (`kda_conv_op_max_rows`); at 32 slots the op costs 477 µs on one die (+221 µs net, `candidates_r1/mc_tp1`), so 32-slot batches keep the chain. Correct at the one-die width: `tests/test_kda_decode_conv_one_die.py` W = 24576, B = 1 / 2 / 8 / 32: 4 / 4 (`logs/s3r2_conv_one_die.log`); suite below |
| sinkhorn permutation matmuls at HiFi3 | -7 | -7 | -7 | -7 | -7 | -7 | **taken** (bit-identical, `tests/test_mhc_split_sinkhorn.py`) |
| dense-MLP down `1d:c16:ib4` (32 x 12288 x 4096) | -73 | - | - | -74 | - | - | **taken** |
| DSA o_proj `1d:c16:ib4` | - | - | -7 | - | - | -8 | **taken** |
| KDA in-proj split DRAM-sharded at HiFi2 (P1-B a): 32960-col interleaved -415, DS 32768 cols +293, 1D 192 cols +18, concat +12, reshard +8 | -84 | -84 | - | -84 | -84 | - | dropped: watcher assert (P1-B table) |
| dense-MLP gate/up `1d:c16:ib4` | +20 (mc_tp1) | | | | | | rejected (slower on one die) |
| `kda_out`, `dsa_qb`, `dsa_wqb` role specs | 0 | 0 | 0 | | | | review #2: not taken (mc_tp1 rows equal at that resolution); review #3: `kda_out` measured -3.4 .. -4.6 µs and taken with the shared-expert specs ("Review #3") |
| `mix_chunks_decode=16` | +36 .. +40 (review #1) | | | | | | rejected on one die |
| `moe_grouped_in0=8` (stage-5 TP4 geometry) | | | | | | | not taken: forced in0 8 overflows L1 at 32 rows on one die (2.11 MB, review #1 sweep); the module's L1 rule otherwise shrinks it back |

The `mc_tp1` kda_dense prefill figure of review #1 (17811 µs) was not a valid measurement: its CSV has no dense-MLP rows
and a 1.5 s op-to-op gap (truncated window), and the dsa_moe prefill of that candidate failed (stage-5 `prefill_2d`
overflows L1 on one die). It is withdrawn.

## Precision trials (real weights; candidates on `opt`; suite subsets on the quantization-aware reference)

Reusable for the stage-8 datatype sweep. Bars: functional. Time: Tracy B=1 decode unless noted, against the same-day `opt` control
(`candidates_r1/<trial>/`). bfp4 weights are host-rounded (round to nearest), as the experts.

| Trial (`CANDIDATES`) | Tensor group | Result (`logs/s3r_pcc_<trial>*.log`) | Lowest update PCC | Scale | Time | Decision |
|---|---|---|---|---|---|---|
| `t_kda_in4` | KDA in-proj bfp4 (HiFi4) | 1 failed, 1 error / 16 | L4 0.951828 (B=32), L0 0.996819 | L0 seq 0.979 .. 0.992 | kda_dense 2083 -> 1969, kda_moe 2286 -> 2175 µs | reject: B=32 routes flip (5 y_swap > cap 2) |
| `t_kda_out4` | KDA out-proj bfp4 | 12 failed, 2 errors / 16 | L4 0.950915, L0 0.994182 | L0 seq 0.985 .. 0.996, decode mean 0.9874 | kda_moe 2286 -> 2252 µs | reject |
| `t_dsa4` | DSA q_a, q_b, kv_a, o_proj bfp4 | 3 failed, 1 error / 10 | L3 0.986869 | token 0.950 .. 1.057 | dsa_moe 2421 -> 2326 µs | reject: tokens match neither y nor y_swap |
| `t_dsa_o4` | DSA o_proj bfp4 | 2 failed, 2 errors / 10 | L3 0.986986 | token 0.965 .. 1.068 | dsa_moe 2421 -> 2356 µs | reject |
| `t_mlp4_lofi` | dense MLP bfp4, LoFi | 1 failed / 11 | L0 0.993051 (B=32 decode) | seq 0.998 | kda_dense 2083 -> 1995 µs; prefill 29968 -> 26799 | reject (update bar) |
| `t_mlp4_hifi4` | dense MLP bfp4, HiFi4 | 1 failed / 11 | L0 0.993027 | seq 1.001 | kda_dense 2083 -> 1993 µs | reject: the loss is the bfp4 rounding, not the fidelity |
| `t_mlp8_lofi` | dense MLP bfp8, LoFi | 9 failed / 11 | - | seq 0.984 .. 0.985, decode mean 0.9856 | kda_dense 2083 -> 2074 µs | reject: LoFi truncation bias |

The first `t_kda_in4` / `t_kda_out4` runs used the device typecast to bfp4, which is not round to nearest: the KDA
layer update read 7-10 % low (`logs/s3r_pcc_t_kda_in4.log`, `s3r_pcc_t_kda_out4.log`). The table uses the host-rounded
reruns (`*_host.log`).

### Review #2: fidelity trials of the bfp8 projections (P1-B)

Candidates on the review #2 single-die options (`_r2`), real-weight suite subsets on the quantization-aware reference
(`logs/s3r2_pcc_<trial>.log`), Tracy against the same-day default (`candidates_r2/<trial>/`). HiFi2 / LoFi compute
configs use fp32 destination accumulation and packer L1 accumulation.

| Trial | Change | Suite subset | Lowest real update PCC / scale | Time | Decision |
|---|---|---|---|---|---|
| `t_kda_in_ds_hifi2` | KDA in-proj decode: DRAM-sharded copy of 32768 columns (64 cores, in0 2, 2 readers) + 1D 192 columns, HiFi2 | 16 / 16; full suite 44 / 44 | L4 0.995799 (B=32), L0 0.999930; decode mean 0.9979 | -84 µs per KDA layer (in0-interleaved 415 -> DS 293 + 1D 18 + concat 12 + reshard 8); kda_dense 1 user 1647 with the P1-A options | **dropped: TTNN kernel blocker (review #3).** Watcher data race in the multi-reader DRAM-sharded reader (details below the table) |
| `t_out_ds_hifi2` | KDA out / DSA o_proj decode DRAM-sharded (`ds:r2`), HiFi2 | 26 / 26 | L3 0.998769, L4 0.996756 | slower: KDA out 102.3 -> 121.2 µs, DSA o_proj 191.9 -> 239.6 µs (DRAM-sharded `ds:r2` at HiFi2 vs the 1D config at HiFi4); layer kda_moe 1 user 1999, dsa_moe 2099 µs | rejected (slower; the probe's DS LoFi 78.6 µs needs LoFi, which fails) |
| `t_out_ds_lofi` | same at LoFi | 25 / 26: L3 32-slot pos 38 (router margin 8.0e-4) matches neither y nor y_swap | L3 0.995564 | not measured (Tracy step dropped when the queue was consolidated; DS LoFi probe: KDA out 78.6, o_proj 150.6 µs) | rejected (correctness) |
| `t_pf_hifi2` | prefill rows of DSA o_proj and dense-MLP gate|up / down at HiFi2 (+ packer L1 accumulation) | 21 / 21; full suite 44 / 44 (`logs/s3r2_final_suite.log`) | L0 prefill scale 0.9927 .. 0.9940 (0.9948 .. 0.9969 at HiFi4), L3 0.9945 .. 0.9973 | prefill 2048 kda_dense 26987 -> 20676 µs (-23 %), dsa_moe 77076 -> 74974 µs (-2.7 %) | **taken** |

Details of the `t_kda_in_ds_hifi2` blocker (review #3; P3-2 wording of review #4):

- The watcher ran with `TT_METAL_LOGGER_LEVEL=Warning`. The `Error` level of env.sh hid the details in review #2.
  Two decoder repeats and the standalone `tools/s3r3_split_watch.py` gave the same message each time.
- Message: `Device 1 worker core(x=0,y=0) virtual(x=1,y=2): BRISC detected an inter-kernel data race due to kernel
  completing with pending NOC transactions (missing NOC reads flushed barrier)`. Kernel:
  `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`.
- Watcher Device 1 is the test's own chip. Review #2 wrongly said it was an unused chip.
- The race is in the multi-reader path of that reader (SPLIT_DRAM_BANK, workers_per_bank > 1). 2 readers trip it at
  (cores, in0) = (64, 2), (64, 1) and (32, 2).
- 1 reader is legal only at in0 1, and it is 2.3x slower: "split DS c32 in0 1 r1 + 1D" 946.3 µs at HiFi4
  (`logs/s3r_p14_dense_kdain.log`). At (64, 2) it overflows L1 (circular buffers). At (32, 4) it overflows L1 too
  (2,601,984 B, `logs/s3r3_split_standalone.log`). At in0 4 on 64 cores and at (64, 4) it fails the TT_FATAL
  `matmul_device_operation.cpp:1365` (shard width % in0_block_w).
- Not tried: an N-halved 1-reader variant (64 tiles per bank, two DRAM-sharded matmuls).
- The review #2 bisection's last two variants ran after the default had dropped the split. So "sink" off and "all
  off" ran without it.

The review #1 statement that the KDA in-proj DRAM-sharded path "wins only with HiFi2 or bfp4, which the precision
trials reject" was wrong: no real-weight HiFi2 trial existed then. It exists now: it passes the suite but trips a watcher
assert, so it is not in the default.

## DRAM-sharded, geometry and prefill configurations (`tools/s3r_probe_dense.py`, `s3r_probe_sparse.py`, `s3r_probe_dsa.py`)

Logs: `logs/s3r_p14_dense_kdain.log`, `s3r_p14_dense_out.log`, `s3r_p14_sparse.log`, `s3r_p14_dense_pf.log`,
`s3r_p14_sdpa.log`, `s3r_p14_bmm_m{2,4,8}.log`. Host-timed with a device sync; dense / DSA on synthetic weights at the
real shapes, the sparse sweep on real layer-4 weights at the `opt` policy.

| Item | Candidates and numbers | Decision |
|---|---|---|
| KDA in-proj 32 x 4096 x 32960 bfp8 HiFi4 (default 401.3 µs; 1D 64 / 110 cores 432 / 409) | DRAM-sharded natural N (129 tiles per bank): 1 reader overflows L1 at in0_block_w 8 .. 1; **3 readers run, 484 .. 502 µs**. N padded to 1056 tiles: 1 reader in0 4 / 2 overflow; 2 or 3 readers 492 .. 528 µs. N split (1024 tiles DRAM-sharded + 6 tiles 1D + concat): best 486.8 µs (concat 31 µs) | rejected at HiFi4 (the DRAM-sharded kernel is compute bound there); at HiFi2 the N split runs 333.5 µs (vs 398.3 interleaved HiFi2): real-weight HiFi2 trial passes the suite but trips a watcher assert: **not taken** (review #2, "Fidelity trials") |
| KDA out 8192 x 4096 / DSA o_proj 16384 x 4096 decode | 1D 101.7 / 198.9 µs; DRAM-sharded HiFi4 130.0 / 234.9; DS LoFi 78.6 / 150.6; 4 readers illegal (16 tiles per bank) | 1D kept at bfp8 HiFi4 |
| Prefill on the DRAM-sharded weight | `create_prefill_matmul_program_config(2048, K, N)` on 8 x 10: PCC 1.000000, same time as interleaved; KDA out 3872 -> 2420 µs, o_proj 7684 -> 4831 µs. The (11, 10) grid on a DRAM-sharded weight returns wrong output (PCC nan) | the "second weight copy" premise was wrong; DS weights not taken because decode DS loses at HiFi4 |
| Routed sparse matmuls, B=1 gather (default 767.4 µs, 64 cores, in0 32) | in0 16 -6.9 (noise), in0 64 / 128 +4 / +18, 32 cores +24 .. +47, 128 cores invalid; down variants +2 .. +26; pairs path +687 / +1421 | default kept (all variants bit-identical) |
| B=32 grouped (default 8161.7 µs, in0 4, subblock 4) | gate/up in0 2 +697, down in0 2 +308, subblock 2 / 1 +105 / +637, in0 8 / 16 overflow L1 (2.11 / 3.32 MB); pairs 16748 | default kept |
| Prefill 2048 sorted / grouped (default 52151 µs with the 1024-row split) | in0 2 +1073 .. +2832, subblock 2 +583, in0 8 overflows; the stage-6 1024-row split (committed in 08bac32f4c1) costs 7.8 ms per MoE layer (44373 µs unsplit, bit-identical) | default kept; the split is a stage-6 / 7 memory decision |
| DSA per-head matmul at 2048 prefill rows (default 1900 / 2090 µs) | reuse config per_core_M 2 / 4 / 8: 1426 / 941 / **765** (q_abs), 1335 / 873 / 665 (o_uv), PCC 0.999999, no hang (the stage-3 hang was per_core_M = Mt = 64) | **not taken**: the probe is fast and accurate on one chunk, but in the decoder the DSA continuation chunks (prompts over 2048) read 11-22 % low with it on W_UK and non-finite on W_UV (`logs/s3r_bisect_pf.log`, `logs/s3r_final_suite.bmm_wuk.log`). Diagnosed in review #3: out-of-bounds batch-stride writes of the batched reuse matmul with per_core_M < Mt ("Review #2: prefill and sparse-decode items", `logs_r3/s3r3_bmm_oob.log`). Knob kept off (`GLM53_PF_OFF=bmm_on` turns it on) |
| `flash_mla_prefill` 2048 rows (default 22852 µs) | q128 / k128 **6654**, q128 / k64 6753, q64 / k64 11625; PCC 0.99996 vs default; k256 / k512 / q256 overflow L1 | **taken** |
| Large prefill matmuls (bfp8) | explicit 2D config on the full grid, bit-exact: DSA o_proj 7707 -> **3708**, MLP down 5792 -> **2778**; KDA in / MLP gate-up overflow L1 at in0 4 on that grid. HiFi2: KDA in 10115 -> 7951, gate/up 7094 -> 5594 (precision change) | 2D configs **taken** for o_proj and MLP down; HiFi2 prefill: review #1 did not take it; review #2 trialled it on real weights and **took** it for DSA o_proj and the dense MLP (`t_pf_hifi2`); review #3 rejected it for the KDA in / out projections (`t_kda_pf_hifi2`); review #4 took it for the KDA + MoE layers |

Prefill configurations end to end: `candidates_r1/opt_s5_pf` (with the W_UK reuse config, later removed) dsa_moe 2048
rows 97302 -> 77407 µs, kda_dense 29941 -> 26759 µs, kda_moe unchanged (66865); the final numbers without it are in the
performance table.

## Operation-topology audit (decode, per layer, one die)

Measured on the CSVs (`candidates_r1/opt`, `opt_s5`, `candidates_r2/opt_s5_pf`); probes `tools/s3r2_glue_probe.py`
(`logs/s3r2_glue{,2}.log`); stage-5 evidence from
`doc/optimized_multichip_decoder/README.md`.

| Op group | Current (µs) | Candidate | Constraint / evidence | Action |
|---|---|---|---|---|
| Decode RMSNorm (input / post; DSA q_a / kv_a) | 65-67 on 1 core x 2 (KDA), 5 norms 170 (DSA) | width-sharded L1 norm | PCC 0.999999, 1 bf16 ulp (stage 5) | **taken** (`single_die_mc`): 136 -> 18 (KDA), 170 -> 29 (DSA) |
| DSA x-projections (q_a 66.5, kv_a 56, wk / wg 99, w_iw 31) | 252 | two packs on 1D configs, weights shared with prefill | mixed dtypes: two packs | **taken**: -237 net with the packs (+46) |
| DSA q_b / wq_b, router | 82 / 35 / 49 | 1D configs | fp32 router logits exact | **taken** (router -31) |
| Router TopK | 34 on 1 core | fused MoE gate op | bf16 scores in that op; GLM needs fp32 top-k (stage-1 route flips) | rejected |
| mHC sinkhorn | 89-90 on 1 core x 2 = 179 | HiFi3 permutation matmuls; face-restricted kernel (`SK_FACES`, stage-5 worktree) | one tile holds all users | HiFi3 **taken** (bit-identical, -7 µs); valid-faces kernel (`sinkhorn_faces`, committed by stage 5) **taken** in review #3: -30 µs per layer at 1 user, -19 µs at 32 (bit-identical, `tests/test_mhc_split_sinkhorn.py`) |
| mHC mix chunks | 32 chunks at decode | 16 chunks (`opt_s5_mix16`) | stage 5: -6 .. -10 µs at TP4 | **rejected on one die**: +36 .. +40 µs at B=1 (`candidates_r1/opt_s5_mix16_f3` vs `opt_s5_f3`) |
| KDA conv chain (6 ternary 144, 3 typecasts 13 each) | about 254 | `kda_decode_conv` (one op) | 97.5 / 140 / 256 / 484 µs at 1 / 8 / 16 / 32 slots vs the chain 264 .. 266 µs on one die (traced microbenchmark) | **taken for <= 16 slots** (-161 µs at 1 user; review #3 microbenchmark `tools/s3r3_conv_rows.py`); the chain at 32 |
| KDA in-proj split (qkv + rest weights) | packed 32960 columns | two weights | `tools/kda_split_probe.py` bfp8: prefill 2048 6550 -> 5908 µs (bit-exact), decode 410 -> 398 µs (B=32 and B=1) | not taken in stage 3: the packed weight layout is shared with the TP4 per-die grouping (stage 5), the weight cache (stage 6) and the deepseek KDA prefill kernels; -642 µs per KDA layer prefill, -12 µs decode; handed to stage 5 / 7 with these numbers |
| KDA recurrent_decode | B=1 67, B=32 838 (268 MB state) | split each head's state by V columns over 2 cores (the delta-rule update is independent per V column): 128 cores at B=1 | op microbenchmark (`tools/s3r2_glue_probe.py`): B 1 / 2 / 4 / 8 / 16 / 32 = 74.6 / 121.8 / 178.9 / 262.3 / 472.8 / 849.3 µs (112 .. 316 GB/s): **latency bound at B=1** (one head per core, 64 of 110 cores, serial over a 128 x 128 fp32 state), state-bandwidth bound only at 32 | kernel change in `ttnn.experimental.kda.recurrent_decode`; owner: KDA kernel (stage 5) |
| MoeSortedDispatch | 74.5 on 1 core (B=32), 1.22 ms at 2048 rows | multi-core histogram + scan + scatter | the op is single-core by design (`const CoreCoord core{0,0}`, a counting sort) | blocker: kernel rewrite; owner: MoE dispatch kernel (stage 5 / TTNN) |
| MoE routing reshape `[1,32,32(8),32(1)]` -> `[1,1,32,32(8)]` fp32 | 35.6 on 1 core (B=32); 2 at B=1 | ROW_MAJOR -> reshape -> TILE | probe 30.1 vs 39.0 µs host-timed, exact (`ttnn.view` illegal here); decoder A/B bit-identical (`logs_r4/s3r4_ab.log`) | **taken in review #4** (`route_rm`, 32-row decode batches only; an instance wrapper of `Qwen35MoE._route_sigmoid_bias`, no `tt/moe.py` edit) |
| DSA indexer qi reshape `[1,1,32,4096]` -> `[1,1,1024,128]` bf16 | 42.1 (B=32), 19.1 (B=1) | ROW_MAJOR round trip; `nlp_create_qkv_heads` | round trip 33.5 vs 49.6 µs host-timed, exact; the heads op gives head-major order at 55 µs; decoder A/B bit-identical | **taken in review #4** (`qi_rm`, `SingleDieDSA._mm`; no `tt/dsa.py` edit) |
| mHC pre / post (`MhcPreDeviceOperation`, `MhcPostDeviceOperation`) | pre 50 (25 x 2), post 118 (59 x 2) per layer, the same at 1 and 32 users | none measured | fixed per-tile work: the ops process the [hc_mult = 4 streams x 4096] residual tile rows and the 4 x 4 mixing per call; the cost does not scale with users, so it is 8-10 % of 1-user decode; at 32 users it is 1.5 % of the MoE layers and 6 % of kda_dense. fp32-exact stage-1 / 2 kernels (the mHC contract) | not changed in stage 3. Owner: the mHC kernels (DSv4 `mhc_pre` / `mhc_post`, stage 5 / TTNN). A fused pre + mix or post + residual kernel is the candidate |
| MoE gather rows / untilize / slices (B=32) | 637 / 260 / 134 (+ others) | down projection with per_core_N 16 / 32 and `untilize_out`; smaller row tiles | the grouped sparse factory requires per_core_N == Nt (TT_FATAL `sparse_matmul_multicore_reuse_grouped.cpp:93`, at 32 and 2048 rows); about 1.8 real rows per 32-row group; the slice exists because `ttnn.swiglu` has no GLM clamp | blocked: N splitting in the grouped sparse factory, or a smaller row tile; owner: TTNN sparse matmul / stage 5 |
| Decode activations in DRAM | every matmul input 0 | L1 / sharded | 64-256 KB activations, one reshard each; the sharded norm feeds the packs directly | rejected except the producer-sharded path |

## Performance accounting (final default, review #4: `candidates_r4/final`)

Device time and host wall come from the same Tracy run of each case (review #4 default, seed pinned, position 128 unless
noted). Bytes per step: weights from the op shapes and dtypes (bfp8 1.0625, bfp4 0.5625, bf16 2 B per element; 8 routed
experts at B=1, about 178 distinct at B=32), KDA state fp32 read + write; DRAM 512 GB/s. The weight dtypes have not
changed since review #2, so the bytes are those of review #2.

| Case | Device µs | Host wall µs / step | Host - device | Bytes / step | DRAM floor µs | Roofline fraction |
|---|---|---|---|---|---|---|
| kda_dense 1 | 1705 | 2396 | 691 | 350 MB | 684 | 40.1 % |
| kda_moe 1 | 1951 | 2681 | 730 | 331 MB | 647 | 33.2 % |
| dsa_moe 1 | 2024 | 2804 | 780 | 281 MB | 549 | 27.1 % |
| dsa_moe 1, position 16384 | 2374 | 3104 | 730 | 283 MB | 553 | 23.3 % |
| kda_dense 32 | 2705 | 3328 | 623 | 610 MB | 1192 | 44.1 % |
| kda_moe 32 | 11515 | 12242 | 727 | 3.00 GB | 5855 | 50.8 % |
| dsa_moe 32 | 11210 | 12103 | 893 | 2.69 GB | 5250 | 46.8 % |

Host wall minus device (0.6 .. 0.9 ms per step) is the input copy between replays plus the replay-boundary gap. The
host wall is measured under Tracy. The stage-7 generator owns the step loop. Prefill is compute bound.

Per op group, µs per layer (`tools/s3r4_accounting.py groups`, final default):

| Group | kda_dense 1 | kda_moe 1 | dsa_moe 1 | dsa_moe 1, pos 16384 | kda_dense 32 | kda_moe 32 | dsa_moe 32 |
|---|---|---|---|---|---|---|---|
| Routed sparse matmuls | - | 443 | 447 | 446 | - | 7283 | 7712 |
| Dense matmuls | 981 | 630 | 553 | 553 | 983 | 629 | 551 |
| Norms (sharded) | 18 | 18 | 30 | 30 | 25 | 25 | 30 |
| Sinkhorn (faces, HiFi3 permutations) | 142 | 141 | 141 | 142 | 153 | 153 | 153 |
| mHC pre | 51 | 51 | 51 | 51 | 51 | 51 | 52 |
| mHC post | 119 | 117 | 119 | 118 | 118 | 118 | 118 |
| mHC mix | 49 | 49 | 49 | 50 | 50 | 49 | 50 |
| KDA conv op (1 slot; the 12-op chain at 32 is in element-wise / data movement) | 93 | 91 | - | - | - | - | - |
| KDA recurrence | 67 | 67 | - | - | 840 | 841 | - |
| Router TopK | - | 34 | 34 | 34 | - | 34 | 34 |
| DSA attention (sparse SDPA, indexer scores and top-k, cache writes) | - | - | 63 | 402 | - | - | 119 |
| MoE glue (dispatch, gather rows) | - | - | - | - | - | 735 | 797 |
| Data movement | 67 | 106 | 249 | 249 | 111 | 727 | 873 |
| Element-wise / other | 119 | 204 | 289 | 299 | 375 | 871 | 720 |
| **Total** | **1705** | **1951** | **2024** | **2374** | **2705** | **11515** | **11210** |

Remaining gaps at 1 user: the fixed per-layer work (sinkhorn 142, mHC pre / post / mix 219, element-wise 119 .. 299,
data movement 67 .. 249 µs), the KDA in-proj at 65 % of DRAM (its DRAM-sharded HiFi2 form trips a watcher race), the
B=1 sparse matmuls at 45 % of DRAM, the latency-bound recurrence (67 µs). In the sparse regime the sparse SDPA is 354 µs
of the 402 µs attention group. At 32 users: the sparse matmuls (69 % of DRAM), MoE glue (0.7 .. 0.8 ms) and the KDA
state (840 µs). Each has an audit row with its owner.

## Review #2: prefill and sparse-decode items (P2-2, P2-4)

Probes by a subagent; logs `logs/s3r2_{bmm_repro,bmm_repro2,final,prefill2d,glue,glue2,tracy_deep_*}.log`.

**DSA per-head reuse config on prefill rows (P2-2a): a TTNN matmul bug - out-of-bounds writes (review #3 diagnosis).**
`MatmulMultiCoreReuseProgramConfig` on a batched `[1,64,M,K] x [1,64,K,N]` with per_core_M < Mt gives each core several
(batch, M) blocks; the reuse kernels advance their batch stride per block (in0 += MtKt, in1 += KtNt, out += MtNt) and the
op validator only checks M % per_core_M, so tail cores read and write past the tensors. The discriminating repro
(`tools/s3r2_bmm_repro3.py --oob`: no prior default call, NaN-prefilled output via `optional_output_tensor`, a canary
tensor of 7.0 allocated right after it; `logs/s3r3_bmm_oob.log`):

| Case | Blocks | Output tiles never written | Canary elements overwritten | Heads wrong |
|---|---|---|---|---|
| W_UK shape, per_core_M 8, M 2048 | 8 x 64 = 512 on 110 cores | 14848 of 65536 | 786432 | 23 |
| W_UV shape, per_core_M 2, M 2048 | 32 x 64 = 2048 | 8336 of 32768 | 3342336 | 42 |
| control: per_core_M = Mt = 4, M 128 | 64 (one per core) | 0 | 0 | 0 |

The review #2 reading ("the first-allocated input is correct") was an artefact: the reuse output took over the freed
default-config output buffer. The decoder keeps the prefill reuse config off; the shipped decode path has per_core_M =
Mt = 1 (one block per head), and `tools/baseline_dump.py`'s frozen policy uses it only where Mt <= per_core_M. TP4 code
has no `MatmulMultiCoreReuseProgramConfig` use with per_core_M < Mt (the grouped sparse MoE config uses the separate
sparse-matmul factory). TTNN blocker (below).

**Adapted 2D prefill configs (P2-2b, `tools/s3r2_prefill2d.py`, synthetic weights, 2048 rows, bfp8, HiFi4):** gate/up
(2048 x 4096 x 24576) 11 x 10, in0 4, out block 7 x 10, subblock 1 x 2: 6020 vs 7059 µs default, bit-exact; overflowing
variants: in0 4 out 7 x 70 3,846,144 B, 7 x 35 2,036,224 B, in0 2 7 x 70 3,484,160 B, 7 x 35 1,826,560 B. KDA in-proj
(2048 x 4096 x 32960) 11 x 10, in0 4, out 1 x 94: 8273 vs 10123 µs. With packer L1 accumulation on, the default configs
run gate/up 3861 µs and KDA in 5488 µs, and the 2D configs add nothing. Gate/up prefill is covered by `t_pf_hifi2`
(HiFi2 + packer L1 accumulation, real weights, above).

Correction (review #4 P2-3). Review #2 handed packer L1 accumulation of the KDA in-projection prefill to the KDA
prefill owner. That hand-off was wrong:

- `models/demos/deepseek_v3_d_p/tt/kda/kda.py:173-178` already sets `packer_l1_acc=True` (since 382e25f6e51,
  2026-10-14).
- The in-model row is 4989 µs at HiFi4 (`candidates_r2/opt_s5_pf_final/test_perf_prefill_kda_dense-2048_`). That is
  the probe's "packer L1 accumulation" value (5488 µs), not its 10123 µs.
- The decoder can set this compute config itself (`OptimizedDecoder._apply_trials`; `t_kda_pf_hifi2`, review #4
  P2-2). So nothing is handed over.

**DSA sparse-attention decode past position 2048 (P2-4).** `GLM53_PERF_POS` (`tests/test_perf_decoder.py`), policy
`opt_s5_pf` (review #2 options for DSA: o_proj spec, sinkhorn HiFi3), `candidates_r2/deep_pos_{4096,16384}/`:

| Position | 1 user (µs) | 32 users (µs) |
|---|---|---|
| 128 (dense regime) | 2045 | 11253 |
| 4096 | 2563 | 11497 |
| 16384 | 2583 | 11589 |

At 16384: `SparseSDPA` 560 µs at 1 user (51 µs at position 128) and 572 µs at 32 users; `DsaDecodeScores` 13.6 / 96.8;
indexer top-k about 29 / 47; router TopK 34. `sparse_sdpa` splits work over query rows only (`base = S / num_cores`), so
one user's 64 heads run on one core. Candidate (`tools/s3r2_sdpa_probe.py`): q `[1,64,1,512]` -> `[1,32,2,512]` with the
index row duplicated and `cache_batch_idx` = slot: 428.9 -> 287.0 µs at 1 user, bit-exact (about -140 µs per DSA layer
per step past position 2048). Review #4 took it at 1 slot as a stage-3 override (`SingleDieDSA._sparse_attend`;
"Review #4"). At B > 1 it needs a page table with duplicated rows (not probed). Larger key chunks (64 / 128) are
illegal (top-k 2080 must be a multiple of the chunk).

## Review #3: single-die options, trials and blockers

Snapshot runs (HEAD 656c998e3df + the stage-3 edits; `~/glm53-flash-bringup/scripts/s3r3_a.sh`, `s3r3_one.sh`), seed
pinned, `logs/s3r3_*.log`, Tracy `candidates_r3/`.

| Option | Evidence | Decision |
|---|---|---|
| `sinkhorn_faces` (committed stage-5 kernel option, bit-identical) | suite subset 29 / 29, identical check lines; sinkhorn -30 µs per layer at 1 user, -19 µs at 32 (kda_dense 1 user 1725 -> 1698, kda_moe 1971 -> 1943) | **taken** |
| `dsa_q_mm` (1-slot DSA q_abs as one masked matmul, in0 blocks 64) | 1-slot dsa_moe decode: TT_FATAL `matmul_device_operation.cpp:761 num_blocks_total <= num_cores` (`candidates_r3/t_r3/test_perf_traced_decode_dsa_moe-1_/console.log`). The cause is the program size, not the op: the program is built for 32 rows, one die has 64 (review #4 P1-1) | rejected in review #3 on the error; review #4 fixed it, measured it (exact, 9 µs slower) and did not take it ("Review #4") |
| role specs `kda_out` `1d:c16:ib4`, `shared_gu` `1d:c32:ib16`, `shared_dn` `1d:c16:ib4` | KDA out 32 x 8192 x 4096 -3.4 .. -4.6 µs, shared expert -2 .. -3 µs. kda_moe 32 users: the layer moved -47 µs, but only about -8 µs is in the spec rows (KDA out -3.5, shared down -3.0, shared gate / up -1.3). The rest (SparseMatmul -27, MoeGatherRows -10 µs) is run-to-run spread (review #4 P2-3; `tools/s3r4_accounting.py diff`). Suite subset 29 / 29 | **taken** |
| `kda_conv_op_max_rows` | `tools/s3r3_conv_rows.py` (traced): op 97.5 / 140 / 256 / 484 µs, chain 265.5 / 264.5 / 266.4 / 264.3 µs at 1 / 8 / 16 / 32 slots | **16** (was 8) |
| KDA prefill in / out projections at HiFi2 (`t_kda_pf_hifi2`, `KDAProgramConfig` fidelity fields set on the prefill layers) | prefill 2048 kda_dense 20579 -> 18595 µs, kda_moe 66686 -> 65034 µs; suite subset 9 failed / 18. All 9 failures are kda_dense (L0 prefill scale 0.9869 .. 0.9897, two fidelity biases stacked with the MLP prefill HiFi2). All 6 kda_moe tests pass | rejected for all KDA layers in review #3; **taken for the KDA + MoE layers in review #4** ("Review #4") |
| KDA in-proj split DS (P1-2) | watcher race in the DRAM-sharded multi-reader kernel (details under the fidelity table); 1 reader is legal only at in0 1 and is 2.3x slower | **blocked** (TTNN) |

Final review #3 default `opt_s5_pf` = review #2 default + `sinkhorn_faces` + the three role specs + conv op <= 16 slots.
Decode device time (µs, seed pinned; control = review #2 default, same session):

| Case | Review #2 default (control) | **Review #3 default** |
|---|---|---|
| kda_dense 1 user | 1725 | **1696** |
| kda_moe 1 user | 1971 | **1939** |
| dsa_moe 1 user | 2043 | **2013** |
| kda_dense 32 users | 2710 | **2689** |
| kda_moe 32 users | 11556 | **11519** |
| dsa_moe 32 users | 11227 | **11193** |

Against `MultichipDecoder` at TP = 1 with the stage-5 options (`candidates_r1/mc_tp1`: 1745 / 1967 / 2040 µs) the
review #3 default is faster on the KDA layers (1696 / 1939) and on dsa_moe (2013 vs 2040).

Correction (review #4 P2-3): review #3 said "prefill is unchanged from review #2". That was wrong. `MHC_SINKHORN_FACES`
is process-wide, and the op takes face mode 2 for T > 16
(`mhc_split_sinkhorn_device_operation.cpp:133-136`). So the faces change the prefill sinkhorn rows too. Review #3 had
no prefill Tracy of its default. The review #4 prefill runs measure the final default ("Review #4").

Correctness of the review #3 default: full suite **47 / 47** (44 + the new `test_single_slot_decode`
for the three kinds; 1-slot dsa_moe decode update PCC >= 0.999957; y_swap 3 of 304, cap 9; `logs/s3r3_suite.log`);
watcher suite subset (incl. B=32 dsa_moe, decode-only, the 1-slot cases) **23 / 23, 0 suspicious lines**, B=1 traced
decode per kind under the watcher **3 / 3, 0 suspicious lines** (`watcher_correctness_r3/{suite,decode1}/`,
`TT_METAL_LOGGER_LEVEL=Warning`); dsa_moe 1-user Tracy host wall 2.941 ms per step (`candidates_r3/final/`).

## Review #4: one-die decode options, KDA prefill per layer kind, refreshed evidence

Runs: one decisive case per run-locked ticket, on pinned snapshots (git archive of HEAD + the stage-3 files of the
live tree; md5s in `~/glm53-flash-bringup/logs/s3r4.status`, copied to `logs_r4/s3r4.status.log`). Script
`~/glm53-flash-bringup/scripts/s3r4_one.sh`. Logs `logs_r4/`, Tracy `candidates_r4/`.

### P1-1: `dsa_q_mm` on one die

- Cause of the review #3 TT_FATAL: `MultichipDSA._hm_consts` builds the q program (`pc_q`) for m = 32 rows. TP4 has
  16 heads, padded to Hq = 32. One die has Hq = 64. With m = 32, `per_core_M` = 1, so the 2 x 16 row and column tiles
  make 32 blocks on the 22-core grid (11 x 2).
- Fix, in `tt/optimized_decoder.py` only: `SingleDieDSA` (a subclass of `MultichipDSA`, set on one die) builds
  `pc_q` with m = Hq. Host check: m = 64 gives `per_core_M` 2, subblock 2 x 1, 16 blocks on 22 cores, for in0 blocks
  8 / 16 / 32 / 64. `tt/multichip_decoder.py` is not changed, so TP4 is not changed. (At TP4, Hq = 32 and the two
  builders give the same program.)
- A second one-die bug showed up in the first A/B run (`logs_r4/s3r4_ab_run1_segv.log`): a segfault in tilize.
  `MultichipDSA._sparse_attend` slices the output to H heads and frees the op output. With Hq = H = 64 the slice is the
  same tensor, so the freed buffer was read next. `SingleDieDSA._sparse_attend` handles the 1-slot row-major query
  without the slice.
- A/B on one build (`tools/s3r4_ab.py`, `logs_r4/s3r4_ab.log`; dsa_moe, 1 slot, 4000-row real prefill, decode at
  position 4000, traced, host wall of 50 replays; every variant bit-identical to the control, eager and traced):

| Variant | µs / step (host wall) | Exact |
|---|---|---|
| control (review #3 default) | 2895.2 (repeat 2889.3) | - |
| `dsa_q_mm` in0 8 / 16 / 32 / 64 | 2915.6 / 2897.5 / 2881.9 / 2880.4 | yes |
| sparse_sdpa head split | 2677.7 | yes |
| indexer qi row-major reshape | 2883.9 | yes |
| `dsa_q_mm` 64 + head split | 2672.9 | yes |
| `dsa_q_mm` 64 + head split + qi row-major | **2670.6** | yes |

- Tracy of that first review #4 default (`candidates_r4/t_r4/test_perf_traced_decode_dsa_moe-1_`, position 128)
  against `candidates_r3/final` (`tools/s3r4_accounting.py diff`): the `dsa_q_mm` q path costs **92.1 µs** (head-mask
  multiply 16.2, matmul [64 x 16384] x [16384 x 512] 72.4, untilize 3.5). The per-head path it replaces costs **82.9
  µs** (tiled reshape 19.5, transpose 4.7, per-head bmm 46.9, untilize 11.8). The matmul has 16 N tiles (R = 512), so
  the 1D program uses 16 cores, and each core reads 512 K tiles of W_UK (about 7.7 GB/s per core). The per-head bmm
  reads the same 8.9 MB on 64 cores. The host-wall A/B (-10 .. -15 µs) is inside its spread (control 2895.2 / 2889.3).
- Decision: **not taken** (+9 µs per DSA layer at 1 user). The fix stays in the code (`dsa_q_mm_ib`, candidate
  `t_r4`). A faster form needs W_UK split over more cores along K (a 2D or DRAM-sharded program); the 1D multicast
  program cannot split K. At TP4 (16 heads, 4.5 MB of W_UK per die) stage 5 measured it faster, so `DEFAULT_MC` keeps it.

### P2-1: the three exact decode wins as stage-3 overrides

| Option | Where (no shared-file edit) | Gate | A/B (`logs_r4/s3r4_ab.log`) | Decision | TP4 |
|---|---|---|---|---|---|
| sparse_sdpa head split (`sdpa_head_split`) | `SingleDieDSA._sparse_attend` | 1 slot, 64 heads per die | 1 slot, position 4000: -218 µs per DSA layer (host wall); bit-identical | **taken** | no: TP4 has 16 heads per die, padded to one 32-head query row. The op takes whole 32-head rows, so there is nothing to split |
| indexer qi row-major reshape (`qi_rm`) | `SingleDieDSA._mm` (w_iq decode rows) | decode rows | 1 slot: -11 µs; 32 slots: -19 µs (host wall); bit-identical | **taken** | yes, likely: TP4 runs the same reshape on the same [1,1,S,4096] indexer query (the indexer is replicated). Not measured at TP4 |
| MoE routing-weight row-major reshape (`route_rm`) | instance wrapper of `Qwen35MoE._route_sigmoid_bias` (moe.py's `ttnn` name points at a proxy whose `reshape` takes the row-major path for this one shape) | 32-row decode batches | 32 slots: -17 .. -27 µs (host wall); both reshapes -40 .. -50 µs; bit-identical | **taken** | yes, likely: TP4 runs the same router per die at 32 rows. Not measured at TP4 |

The 32-slot host-wall deltas are near the run-to-run spread (control 11457.9 / 11447.7 µs). The Tracy rows below give
the per-op numbers. TP4 code is not changed; stage 10 may take the two reshapes.

New hard checks: `test_single_slot_sparse_decode` (1-slot dsa_moe, 4000-row prefill, decode at positions 4000 ..
4002 in the sparse regime, against HF on the quantization-aware reference). `test_policy_reached_weights` checks the
review #4 options on the built layers. `tests/test_kda_decode_conv_one_die.py` adds B = 9 and 16 (the conv-op range
above the old 8-slot threshold): 6 / 6 (`logs_r4/s3r4_conv_one_die.log`).

### P2-2: KDA prefill HiFi2 per layer kind

`kda_prefill_kinds=("moe",)`: the KDA prefill in / out projections run at HiFi2 on the KDA + MoE layers only. L0
(KDA + dense) keeps HiFi4. Per-layer numbers (real-weight prefill sequences; update rms ratio against the
quantization-aware reference; bound 1 +- 0.01):

| Layer | Review #3 default | KDA prefill HiFi2 | Decision |
|---|---|---|---|
| L4 (KDA + MoE), suite `logs_r4/s3r4_suite*.log` | 0.9957 .. 0.9984 | 0.9938 .. 0.9976 (synthetic-weight prefill 0.9930) | **taken**: passes; 0.3 .. 0.4 % of the bound is left |
| L4 decode after the prefill | mean 0.9979, min PCC 0.992610 | mean 0.9978, min PCC 0.992572 | no visible change |
| L0 (KDA + dense), review #3 trial | 0.9927 .. 0.9939 | 0.9869 .. 0.9897 (with the MLP prefill HiFi2) | not applied to L0 |

The headroom is small: the LoFi routed bias already uses about 0.4 %. Only L4 is tested at layer level. About 31 of the
34 KDA layers are KDA + MoE. A stage-8 model-level check does not apply to this single-die decoder: the full model
builds `MultichipDecoder`, which does not read `kda_prefill_kinds`. A TP4 or full-model use would need its own trial.
`t_kda_pf_hifi2_moe` is the candidate name.
Prefill time: below.

### Final default and evidence

Final review #4 default `opt_s5_pf` = review #3 default + sparse_sdpa head split + `qi_rm` + `route_rm` + KDA prefill
HiFi2 on the KDA + MoE layers. `dsa_q_mm` is off.

Tracy, final default (`candidates_r4/final/`), against the review #3 runs (`candidates_r3/t_r3_specs`, `final`; same
seed, another day) and, at position 16384, against the review #3 default in the same session
(`candidates_r4/opt_s5_pf_r3/`):

| Case | Review #3 | **Review #4** | Change from the review #4 options (per-op diff) |
|---|---|---|---|
| decode kda_dense 1 user | 1696 | **1705** | none (no op moved > 1.5 µs: session drift, about +0.5 %) |
| decode kda_moe 1 user | 1939 | **1951** | none at 1 row (`route_rm` acts at 32 rows) |
| decode dsa_moe 1 user, pos 128 | 2013 | **2024** | sparse SDPA 50.7 -> 34.0, indexer reshape 19.1 -> 14.9 (untilize 2.8, row-major view 10.1, tilize 2.0); about -21 µs. The rest of the +11 is drift in small element-wise rows |
| decode dsa_moe 1 user, pos 16384 (same session) | 2587 | **2374** | **-213 µs (-8.2 %)**: sparse SDPA 560.6 -> 354.3, indexer reshape -4 |
| decode kda_dense 32 users | 2689 | **2705** | none |
| decode kda_moe 32 users | 11519 | **11515** | routing reshape 35.5 -> 13.6 (untilize 4.1, view 5.7, tilize 3.8): -22 µs |
| decode dsa_moe 32 users | 11193 | **11210** | routing reshape -22 µs, indexer reshape 42.6 -> 18.4 (-24 µs); SparseMatmul +27 µs drift |
| prefill 2048 kda_dense | 20641 (review #2) | **20541** | sinkhorn 186 -> 168 (faces, review #3); recurrence prep -98 |
| prefill 2048 kda_moe | 66808 (review #2) | **65011** | KDA in-proj 7047 -> 5174, out-proj 1318 -> 1139 (HiFi2); sinkhorn -18; sparse matmuls +312 drift |
| prefill 2048 dsa_moe | 75000 (review #2) | **74967** | sinkhorn -18; the rest is drift |

The faces kernel changes prefill too (sinkhorn 186 -> 168 µs per layer at 2048 rows, bit-identical): the review #3
"prefill unchanged" claim is corrected above.

Correctness and stability of the final default (snapshots with the final `tt/optimized_decoder.py`, md5
9b551ef2ab7292212500dbedf5b72e86). Base commit of the shared files (`tt/dsa.py`, `tt/moe*.py`,
`tt/multichip_decoder.py`, ...): **0bd45f315e1** for the suite and the watcher subset, **7b3ffd7ec65** for the B=1
watcher decode, **044dbeaa3fda** for the prefill Tracy (bases per ticket in `logs_r4/s3r4.status.log`). Stage 10
changed the shared `GlmDSA` later (0a2663d0b6b: pool-key allocation, device page-table width, sparse prefill views).
Review #5 asked for a rerun of the suite and the watcher runs on a pure git archive of the stage-10 fix HEAD. That rerun
(base 0bcf89f7fd6) passed: "Review #5 rerun" below.

- Full suite **48 / 48** (`logs_r4/s3r4_suite.log`, third run in the file), including `test_single_slot_decode` and
  the new `test_single_slot_sparse_decode` (L3 1 slot, prefill 4000: scale 0.9953; decode at 4000 .. 4002: update PCC
  >= 0.999491, scale 0.9870 .. 0.9940). y_swap 3 of 307 (cap 9).
- Watcher suite subset (with the 1-slot and sparse 1-slot cases) **24 / 24**, B=1 traced decode per kind **3 / 3**,
  both with `TT_METAL_LOGGER_LEVEL=Warning` and **0 suspicious lines** (`watcher_correctness_r4/`).
- One-die conv op at B = 1 / 2 / 8 / 9 / 16 / 32: 6 / 6.
- The kda_dense 1-user decode Tracy ran on the pre-final `tt/optimized_decoder.py` (md5 6b7fc11e). It differs from
  the final file only in the default `dsa_q_mm_ib` (64 vs 0), which acts on DSA layers at 1 slot only. Every other
  final Tracy case ran on the final file.
- There is no same-session review #3 control at position 128. The position-128 comparisons use the review #3 runs of
  another day; the only same-session A/B is at position 16384.
- The first default (with `dsa_q_mm`) also passed: suite 48 / 48, watcher 24 / 24 + 3 / 3, 0 suspicious lines
  (`logs_r4/*_qmm.log`, `watcher_correctness_r4_qmm/`).

TP4 is not changed. `tt/multichip_decoder.py`, `tt/dsa.py` and `tt/moe.py` are not edited. The full model builds
`MultichipDecoder`, which skips the single-die block of `OptimizedDecoder.from_state_dict` (`not issubclass(cls,
MultichipDecoder)`). The new `OptConfig` fields are read only in that block. The weight-cache key (policy name and
weight dtypes) does not change.

Known latent hazard in shared code (not edited; owner stage 5): `MultichipDSA._sparse_attend`
(`tt/multichip_decoder.py:943-944`) slices the sparse_sdpa output to H heads and then frees the op output. When the
heads per die are a multiple of 32 (TP1: 64, TP2: 32) the slice is the same tensor, so the next op reads a freed
buffer (the review #4 segfault, `logs_r4/s3r4_ab_run1_segv.log`). TP4 (16 heads, padded to 32) takes a real copy, so
it is not reachable there. The single-die decoder avoids the path (`SingleDieDSA._sparse_attend`).

### Review #5 rerun (shared files at the stage-10 fix HEAD)

One run-locked ticket (`~/glm53-flash-bringup/scripts/s3r5_gate.sh`, copy in `logs_r5/`) on a **pure git archive of
0bcf89f7fd6** (the stage-10 fix HEAD; no live-tree files). md5s of that snapshot: `tt/optimized_decoder.py` 9b551ef2,
`tt/dsa.py` a4c9f4d0, `tt/multichip_decoder.py` 15aae97d, `tt/moe.py` 6e648211, `tt/moe_opt.py` 31fdc53c,
`tests/test_optimized_decoder.py` c11bee17 (`logs_r5/s3r5.status.log`). The first attempt was killed by the host OOM
killer (another job's CPU HF scoring, not this ticket; `logs_r5/s3r5_gate_oomkilled.log`) and was requeued unchanged.

- Full `tests/test_optimized_decoder.py`: **48 / 48**. The 1-slot sparse-regime case (head split with the stage-10
  page-table width) gives the same check lines as in review #4: prefill 4000 scale 0.9953; decode at 4000 .. 4002
  update PCC 0.999958 / 0.999951 / 0.999491, scale 0.9940 / 0.9870 / 0.9915. y_swap 3 of 307 (cap 9).
- Watcher subset (same `-k`, `TT_METAL_LOGGER_LEVEL=Warning`): **24 / 24, 0 suspicious lines**.
- B=1 traced decode per kind under the watcher: **3 / 3, 0 suspicious lines**.
- Logs `logs_r5/s3r5_gate.log`; watcher logs `watcher_correctness_r5/`.

No fix was needed: the head split passes the wider stage-10 page table with `cache_batch_idx=0`, which selects the
slot's row.

### TTNN blockers (not on the TP4 path)

| Blocker | Repro | TP4 |
|---|---|---|
| Batched `MatmulMultiCoreReuseProgramConfig` with per_core_M < Mt (several blocks per core) writes out of bounds | `python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r2_bmm_repro3 --case uk --pm 8 --m 2048 --oob` (file `tools/s3r2_bmm_repro3.py`) | not used: decode bmm has per_core_M = Mt = 1; the prefill knob is off |
| DRAM-sharded matmul multi-reader path (`reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`, SPLIT_DRAM_BANK) completes with pending NOC reads (watcher data race) | `TT_METAL_WATCHER=10 TT_METAL_LOGGER_LEVEL=Warning python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r3_split_watch --cores 64 --in0 2 --readers 2` | not used: TP4 DRAM-sharded matmuls use 1 reader |

Advice on the final CSVs (`candidates_r3/*/perf_report.txt`): the same items as the table below, plus
`KdaDecodeConvDeviceOperation` 88 .. 97 µs at 1 user (110 cores, SLOW flag): the fused op replaces 265 µs of chain; its
row-parallel cost (it scales with slots, 484 µs at 32) is the reason for the 16-slot threshold; a faster op is stage-5
kernel work (owner stage 5).

## tt-perf-report advice (final CSVs, advice on: `perf_report.txt` in each `candidates_r1/<policy>/<case>/` and
`tracy/<case>/perf_report_advice.txt`)

| Advice | Where | Disposition |
|---|---|---|
| "Try a DRAM-sharded program config" | KDA out, DSA o_proj, wq_b | measured: slower at HiFi4 (DRAM-sharded table) |
| "Try a DRAM-sharded program config" | dense-MLP gate / up decode (32 x 4096 x 24576 bfp8, HiFi4, 310 µs, 64 % of DRAM) | not measured at this shape on one die. The nearest one-die measurement is the KDA in-proj (same K = 4096, N 32960): DRAM-sharded at HiFi4 was compute bound and slower (484 .. 502 vs 401 µs); it wins only at HiFi2, where the multi-reader form trips the watcher race and 1 reader is 2.3x slower. Deferred (review #4 P3-5) |
| "Place input 0 in L1" | every matmul | rejected except producer-sharded (audit) |
| "HiFi2 may also work" | BFP8 HiFi4 matmuls | decode rows are DRAM bound (no gain at HiFi4 -> HiFi2 interleaved: KDA in 401 -> 398 µs); prefill: **taken** for DSA o_proj and dense MLP (review #2, real-weight trial), KDA in / out projections: **taken for the KDA + MoE layers** (review #4), not for L0 (KDA + dense: L0 prefill scale 0.987 .. 0.990) |
| "HiFi2 is sufficient for BFP8" | indexer prefill fp32 matmuls (59 µs x 2) | rejected: fp32 indexer scores, 0.1 % |
| "Use HiFi2 / HiFi4 with BF16 activations" | routed sparse matmuls at LoFi | the routed fidelity trial (Accuracy) |
| "Output subblock 1x1 is small" | small-N decode rows; dense gate/up | small N: one tile per core is the widest grid (rejected, as stage 5); gate/up: <= a few % (311 µs at 344 GB/s), not taken |
| "in0_block_w=1 is small" | DSA prefill matmuls, shared expert prefill | o_proj 2D taken; W_UK / W_UV reuse rejected (continuation chunks); others below 2 % of the pass |
| "Increase grid size" | KDA in-proj (103 of 110 cores) | decode DRAM bound; at most 6 % more cores at prefill |
| "High op-to-op gap" | 235 / 280 µs at each replay boundary (host input copies); 537 µs once in KDA prefill | decode: host-side input copy between replays; prefill gap: KDA prefill host sync, stage-7 traced prefill |
| sparse matmul DRAM % missing | B=32, prefill | computed by hand (accounting) |

## Correctness and stability

| Check | Result | Log / artifact |
|---|---|---|
| `opt_s5` (pre-final): suite twice | 44 / 44 twice, identical | `logs/s3r_opt_suite.log`, `logs/s3r_suite_opt_s5_b.log` |
| **Review #4 default: full suite (incl. `test_single_slot_decode`, `test_single_slot_sparse_decode`)** | **48 / 48** | `logs_r4/s3r4_suite.log` |
| **Review #4 default: watcher** suite subset + B=1 traced decode per kind | **24 / 24 + 3 / 3, 0 suspicious lines** | `watcher_correctness_r4/`, `logs_r4/s3r4_watcher_*.log` |
| `kda_decode_conv` at the one-die width, B 1 / 2 / 8 / 9 / 16 / 32 | 6 / 6 | `logs_r4/s3r4_conv_one_die.log` |
| Review #3 default: full suite (incl. `test_single_slot_decode`) | 47 / 47 | `logs/s3r3_suite.log` |
| Review #3 default: watcher suite subset (incl. B=32 dsa_moe, decode-only, 1-slot) + B=1 traced decode per kind | **23 / 23 + 3 / 3, 0 suspicious lines** | `watcher_correctness_r3/`, `logs/s3r3_watcher_*.log` |
| Review #2 default: full suite | **44 / 44** | `logs/s3r2_f3_suite.log` |
| **Review #2 final: watcher** (suite subset incl. B=32 dsa_moe, decode-only, + B=1 traced decode per kind) | **20 / 20 + 3 / 3, 0 suspicious lines** | `watcher_correctness_r2/{suite,decode1}/`, `logs/s3r2_f3_watcher_*.log` |
| **Review #2 final: repeated runs** (`test_batch32_all_active` dsa_moe / kda_moe, `test_lengths[dsa_moe-2051]`, `test_decode_only[kda_moe]`, x 5) | **20 / 20** | `logs/s3r2_f3_stress.log` |
| Review #2 with the KDA in-proj split (`opt_s5_pf_r2b` contents): suite 44 / 44, stress 20 / 20, watcher aborted (assert, no details) | split dropped | `logs/s3r2_suite.log`, `s3r2_stress.log`, `s3r2_watcher_suite.log`, `s3r2_wbisect.log` |
| `kda_decode_conv` at the one-die width (W 24576, B 1 / 2 / 8 / 32) | 4 / 4 | `logs/s3r2_conv_one_die.log` |
| Review #1 default: full suite | 44 / 44 | `logs/s3r_final_suite.log` |
| Review #1: watcher suite subset incl. B=32 dsa_moe + B=1 traced decode per kind | **17 / 17 + 3 / 3, 0 suspicious lines** | `watcher_correctness/{suite,decode1}/`, `logs/s3r_final_watcher_*.log` |
| Review #1: repeated runs `test_batch32_all_active[dsa_moe]` and `test_lengths[dsa_moe-2051]` x 5 | **10 / 10** | `logs/s3r_final_stress.log` |
| First `opt_s5_pf` default (with the W_UK prefill reuse config) | 6 failed: DSA continuation chunks 11-22 % low; config removed | `logs/s3r_final_suite.bmm_wuk.log` |
| `opt_s5` (pre-final): watcher (`TT_METAL_WATCHER=10`, worker L1 1,444,864 B, `NOINLINE`) | 17 / 17 + 3 / 3, 0 suspicious lines | `logs/s3r_watcher_*.log` |
| `opt_s5` (pre-final): `test_batch32_all_active[dsa_moe]` x 5 (the 2026-10-16 stall case) | 5 / 5 | `logs/s3r_stress_b32dsa.log` |
| Stage-1 bug injected (`GLM53_FAULT_ROUTED_SCALE=0.9722`) | caught (14 failed, 3 errors) | `logs/s3r_fault_routed.log` |
| Stage-2 path unchanged after the `tt/fused_decoder.py` refactor fix | fused suite subset 9 / 9, all 312 check lines identical to `s2_fused_suite6.log`; `fused_flag_check kda_decode_fold 0 4` bit-exact | `logs/s3r_fused_subset.log`, `logs/s3r_fused_flag_fold.log` |
| `test_policy_reached_weights` | weight dtypes and routed / shared fidelity of the built layers equal the policy | suite logs |

The 2026-10-16 16:10 stall is classified as infrastructure (work log).

## Limitations

- Decode performance is measured at context 128. The context contract is unchanged (smaller weights only).
- The review #1 evidence was measured on the code committed in 08bac32f4c1 (it includes the `tt/moe_opt.py` 1024-row
  MoE prefill split, `tt/dsa.py`, `tt/moe_dispatch.py`, `tt/weight_cache.py`, `tt/fused_decoder.py`; `tt/multichip_decoder.py`
  is tracked since a0be1c3edd3). The decoder-level prefill numbers include the 1024-row split (+7.8 ms per MoE layer at
  2048 rows against the unsplit run, `s3r_p14_sparse.log`); the stage-7 model takes it only when DRAM requires it.
- The single-die-only options (`single_die_r2`, `prefill_fidelity`, the trial `kda_in_split_ds`) are gated on one die and are not part of the TP4
  path; `tools/baseline_dump.py` builds the stage-4 one-die baseline without them (`GLM53_BASELINE_FROZEN`).
- The KDA bfp4 / DSA bfp4 / dense-MLP bfp4 rejections hold at layer level against HF; stage 8 owns the model-level
  frontier.
- The quantized reference also rounds the shared expert to bfp8 (stages 1-2 held it to the checkpoint); the effect is
  negligible (ref_q8 shared rms 0.9986).
- The LoFi routed bias uses most of the 1 % scale bound on the tested MoE layers (L3 prefill 0.9929, decode mean 0.9929);
  only L3 and L4 are tested at layer level. Stage 8 should track the per-layer scale and keep `opt_rhifi3` as the fallback.
- The KDA in-proj bfp4 rejection rests on router near-tie flips at B=32 (5 y_swap matches over a cap of 2, one token
  matching neither); stage 8 should re-judge it at model level (about -114 µs per KDA layer).

- Process-wide environment (review #4 P3-13). Building an `OptimizedDecoder` with the review #2 / #3 options sets
  `MHC_SINKHORN_PERM_FIDELITY=3` and `MHC_SINKHORN_FACES=1` (with `setdefault`) and never clears them. A later decoder in
  the same process (for example a fused-decoder suite) also uses them. Both are bit-identical and are part of the
  program hash, so the risk is low. `MultichipDecoder` sets the same variables in the same way.
- KDA prefill chunk terms on one die (review #4 P3-7). Every one-die KDA prefill logs "KDA chunk terms need 2383 KiB
  per core, over the 320 KiB L1 budget; keeping them in DRAM". The term buffer scales with the heads per die: 64 heads
  on one die, 16 at TP4 (595 KiB, stage 5). The DRAM fallback is the designed path. At TP4 the 1024-row chunks that fit
  L1 were slower than one 2048-row chunk (`doc/optimized_multichip_decoder/README.md`, "KDA prefill chunk terms"). On
  one die even 512-row chunks need about 600 KiB. Classified: expected, not a fault; no change.
- Sparse-matmul `nnz` (review #4 P3-12). The decode gather path (1 row) uses `indices=` (no `nnz` contract). The
  grouped path uses the expert id list (`indices=`). The sorted / masked paths pass `nnz=None`: the expert count is
  data dependent, and a wrong static `nnz` deadlocks the device (`tt/moe.py` docstring). tt-perf-report has no DRAM %
  for these rows. The accounting computes their bytes by hand (8 experts at 1 user, about 178 at 32).
- Deferred review #4 P3 items (each needs new device sweeps on the shared queue, where stages 9-11 had priority):
  - P3-5: the dense-MLP gate / up one-die geometry table (DRAM-sharded and 1D in0 / core sweep at K = 4096);
  - P3-6: the indexer `wq_b` bfp8 trial (precision change; about -12 µs per DSA layer expected);
  - P3-2: the N-halved 1-reader KDA in-proj split (two DRAM-sharded matmuls of 64 tiles per bank);
  - P3-11: repeated runs of the review #4 default (the options are bit-identical or a prefill fidelity change);
  - 128-row prefill Tracy of the review #4 default (dropped for device time; review #2 runs are cited).
- The review #3 report is not on disk (`~/glm53-flash-bringup/artifacts/` has `stage3_review_{1,2,4}.md`); only the
  `PLAN.md` summary exists. The KDA in-proj split into two weights (qkv + rest; -642 µs per KDA layer at 2048 prefill rows,
  -12 µs decode, bit-exact; audit) is still unowned (P3-8). It changes the packed weight layout that stage 5 (TP4
  grouping), stage 6 (weight cache) and the deepseek KDA prefill kernels share.

## Commands

```
GLM53_DECODER=optimized [GLM53_OPT=<candidate>] TRACY_OUT=<dir> CASES="..." bash ~/glm53-flash-bringup/scripts/s3r_tracy.sh
pytest models/autoports/zai_org_glm_5_3_flash/tests/test_optimized_decoder.py          # quantization-aware reference
GLM53_REF_QUANT=0 pytest .../test_optimized_decoder.py                                  # checkpoint reference, old bar shift
GLM53_FAULT_ROUTED_SCALE=0.9722 pytest .../test_optimized_decoder.py -k "moe and ..."   # gate check
GLM53_REF_DRYRUN=<dir> python -m models.autoports.zai_org_glm_5_3_flash.tools.quant_ref_requests
python -m models.autoports.zai_org_glm_5_3_flash.tools.compute_quant_refs <dir> --split 1 --check
python -m models.autoports.zai_org_glm_5_3_flash.tools.bfp_check --layer 4
python -m models.autoports.zai_org_glm_5_3_flash.tools.moe_scale_probe --layer 4 --rows 128 --policy opt \
  --modes prefill,decode,decode1 --fidelities LoFi,HiFi2,HiFi3,HiFi4
python -m models.autoports.zai_org_glm_5_3_flash.tools.kda_split_probe
python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r3_conv_rows
TT_METAL_WATCHER=10 TT_METAL_LOGGER_LEVEL=Warning python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r3_split_watch --cores 64 --in0 2 --readers 2
python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r2_bmm_repro3 --case uk --pm 8 --m 2048 --oob
python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r4_ab [--only one,b32]
python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r4_accounting groups <case dir> ...
python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r4_accounting diff <case dir A> <case dir B>
```
Job scripts: `~/glm53-flash-bringup/scripts/s3r_{main,trials,s5,follow1..4,bisect,refs2,tracy,p14}.sh`, `s3r3_{a,one}.sh`,
`s3r4_{one,phase1,phase2,phase3}.sh` (copies in `logs_r4/`).
