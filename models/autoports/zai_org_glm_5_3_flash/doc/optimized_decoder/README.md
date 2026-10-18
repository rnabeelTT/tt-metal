# GLM-5.3-Flash (E192) optimized decoder

Stage 3 of the tt-model-bringup flow. `tt/optimized_decoder.py` (`OptimizedDecoder`) is the fused decoder
(`tt/fused_decoder.py`) with a precision policy, decode and prefill program configurations, and the stage-5 decode
options that need no collective, for one die. Contract, state, paging and semantics are those of
`tt/functional_decoder.py`. The policies are `OptConfig` entries in `CANDIDATES`; `SELECTED` is the default.

Review #1 (`~/glm53-flash-bringup/artifacts/stage3_review_1.md`) changed the reference, the default and the evidence;
re-review #2 (`stage3_review_2.md`) added the further single-die options, the fidelity trials of the bfp8
projections and the P2 items ("Review #2" sections). The earlier text of this README is in git history (commits
21dc7532f26, d48b7fd06ad). The work log has the failures and the order of events. Every `logs/<name>.log` cited here
is copied to `doc/optimized_decoder/logs_r1/` (review #1) or `logs_r2/` (review #2) from `~/glm53-flash-bringup/logs/`.

## Selected policy (`SELECTED` = `opt_s5_pf`)

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
| KDA decode conv (12-op chain) | unchanged | **one `kda_decode_conv` op for decode batches <= 8 slots** (the chain at 32) | -161 µs per KDA layer at 1 user; +221 µs at 32 users if used there (review #2) |
| Sinkhorn permutation matmuls | HiFi4 | **HiFi3** (bit-identical) | -7 µs per layer |
| Dense-MLP down / DSA o_proj decode | ttnn default / 1D | **1D 16-core programs** (`mlp_dn`, `dsa_o` role specs) | -73 / -7 µs |
| DSA o_proj and dense-MLP gate|up / down, prefill rows | HiFi4 | **HiFi2 with packer L1 accumulation** | prefill 2048 kda_dense 26987 -> 20641 µs, dsa_moe 77076 -> 75000 µs (review #2 P1-B) |
| mHC, KDA recurrence | unchanged | unchanged | fp32-exact stage-1 / 2 kernels; dispositions in the audit |

`OptConfig` fields: `kda_in_dtype`, `kda_out_dtype`, `moe` (expert dtypes, fidelities, sparse geometry),
`dsa_bmm_dtype`, `dsa_bmm_reuse`, `out_proj_1d`, `single_die_mc` (the pinned stage-5 option set),
`prefill_cfg`, `single_die_r2` (review #2 single-die options), `kda_in_split_ds`, and the trial fields `dsa_proj_dtype`,
`mlp_dtype`, `mlp_fidelity`, `mix_chunks_decode`, `out_proj_ds`, `prefill_fidelity`. All single-die options are gated on
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

Final default (review #2: `opt_s5_pf` with `single_die_r2` and `prefill_fidelity=HiFi2`; decode `candidates_r2/final/`,
prefill `candidates_r2/opt_s5_pf_final/`, advice on), against the stage-2 fused decoder (`doc/fused_decoder/tracy`,
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

At 1 user the final default is faster than or equal to every correct single-die candidate in the tree, including
`MultichipDecoder` at TP = 1 with the stage-5 rework options (`candidates_r1/mc_tp1`: kda_dense 1745 vs 1727, kda_moe
1967 vs 1974, dsa_moe 2040 vs 2047 µs; mc_tp1 includes the stage-5 `moe_grouped_in0` / role specs this default does not
take and fails its dsa_moe prefill on one die). The `opt_s5_pf_r2b` candidate (with the KDA in-proj split) measured
1647 / 1890 / 2045 µs but trips a watcher assert (below). The kda_dense 32-user host wall of 7.4 ms is one run under host
load (device 2.713 ms); the other host walls are within 0.7 .. 1.2 ms of device time.

### Review #2: single-die options (P1-A), per-op deltas against the same-day review #1 default

`candidates_r2/opt_s5_pf_r1` -> `candidates_r2/opt_s5_pf` (= `opt_s5_pf_r2b`, the P1-A options + the KDA in-proj split), device µs per layer:

| Option | kda_dense 1 | kda_moe 1 | dsa_moe 1 | kda_dense 32 | kda_moe 32 | dsa_moe 32 | Decision |
|---|---|---|---|---|---|---|---|
| `kda_decode_conv` (one op for the 12-op chain; ternary -144, binary -45, typecast -39, unary -17, slice -9, op +93) | -161 | -162 | - | (off) | (off) | - | **taken for decode batches <= 8 slots** (`kda_conv_op_max_rows`); at 32 slots the op costs 477 µs on one die (+221 µs net, `candidates_r1/mc_tp1`), so 32-slot batches keep the chain. Correct at the one-die width: `tests/test_kda_decode_conv_one_die.py` W = 24576, B = 1 / 2 / 8 / 32: 4 / 4 (`logs/s3r2_conv_one_die.log`); suite below |
| sinkhorn permutation matmuls at HiFi3 | -7 | -7 | -7 | -7 | -7 | -7 | **taken** (bit-identical, `tests/test_mhc_split_sinkhorn.py`) |
| dense-MLP down `1d:c16:ib4` (32 x 12288 x 4096) | -73 | - | - | -74 | - | - | **taken** |
| DSA o_proj `1d:c16:ib4` | - | - | -7 | - | - | -8 | **taken** |
| KDA in-proj split DRAM-sharded at HiFi2 (P1-B a): 32960-col interleaved -415, DS 32768 cols +293, 1D 192 cols +18, concat +12, reshard +8 | -84 | -84 | - | -84 | -84 | - | dropped: watcher assert (P1-B table) |
| dense-MLP gate/up `1d:c16:ib4` | +20 (mc_tp1) | | | | | | rejected (slower on one die) |
| `kda_out`, `dsa_qb`, `dsa_wqb` role specs | 0 | 0 | 0 | | | | no gain on one die (mc_tp1 rows equal); not taken |
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
| `t_kda_in_ds_hifi2` | KDA in-proj decode: DRAM-sharded copy of 32768 columns (64 cores, in0 2, 2 readers) + 1D 192 columns, HiFi2 | 16 / 16; full suite 44 / 44 | L4 0.995799 (B=32), L0 0.999930; decode mean 0.9979 | -84 µs per KDA layer (in0-interleaved 415 -> DS 293 + 1D 18 + concat 12 + reshard 8); kda_dense 1 user 1647 with the P1-A options | **dropped: watcher assert, no details captured, correlates with the 2-reader DS split (1 sample each).** The watcher run of the default with it aborted at the first decode step (`logs/s3r2_watcher_suite.log`, empty `info:`; the watcher.log dump stops after the idle device-0 cores, so the asserting core is likely on a chip the test does not use). Bisection with the watcher on `test_decode_only[kda_dense]` (`logs/s3r2_wbisect.log`): all options on, conv op off, mlp_dn off: assert; only the split off: pass. Kept as a candidate (+142 MB per KDA layer) |
| `t_out_ds_hifi2` | KDA out / DSA o_proj decode DRAM-sharded (`ds:r2`), HiFi2 | 26 / 26 | L3 0.998769, L4 0.996756 | slower: KDA out 102.3 -> 121.2 µs, DSA o_proj 191.9 -> 239.6 µs (DRAM-sharded `ds:r2` at HiFi2 vs the 1D config at HiFi4); layer kda_moe 1 user 1999, dsa_moe 2099 µs | rejected (slower; the probe's DS LoFi 78.6 µs needs LoFi, which fails) |
| `t_out_ds_lofi` | same at LoFi | 25 / 26: L3 32-slot pos 38 (router margin 8.0e-4) matches neither y nor y_swap | L3 0.995564 | not measured (Tracy step dropped when the queue was consolidated; DS LoFi probe: KDA out 78.6, o_proj 150.6 µs) | rejected (correctness) |
| `t_pf_hifi2` | prefill rows of DSA o_proj and dense-MLP gate|up / down at HiFi2 (+ packer L1 accumulation) | 21 / 21; full suite 44 / 44 (`logs/s3r2_final_suite.log`) | L0 prefill scale 0.9927 .. 0.9940 (0.9948 .. 0.9969 at HiFi4), L3 0.9945 .. 0.9973 | prefill 2048 kda_dense 26987 -> 20676 µs (-23 %), dsa_moe 77076 -> 74974 µs (-2.7 %) | **taken** |

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
| DSA per-head matmul at 2048 prefill rows (default 1900 / 2090 µs) | reuse config per_core_M 2 / 4 / 8: 1426 / 941 / **765** (q_abs), 1335 / 873 / 665 (o_uv), PCC 0.999999, no hang (the stage-3 hang was per_core_M = Mt = 64) | **not taken**: the probe is fast and accurate on one chunk, but in the decoder the DSA continuation chunks (prompts over 2048) read 11-22 % low with it on W_UK and non-finite on W_UV (`logs/s3r_bisect_pf.log`, `logs/s3r_final_suite.bmm_wuk.log`); not diagnosed. Knob kept off (`GLM53_PF_OFF=bmm_on` turns it on) |
| `flash_mla_prefill` 2048 rows (default 22852 µs) | q128 / k128 **6654**, q128 / k64 6753, q64 / k64 11625; PCC 0.99996 vs default; k256 / k512 / q256 overflow L1 | **taken** |
| Large prefill matmuls (bfp8) | explicit 2D config on the full grid, bit-exact: DSA o_proj 7707 -> **3708**, MLP down 5792 -> **2778**; KDA in / MLP gate-up overflow L1 at in0 4 on that grid. HiFi2: KDA in 10115 -> 7951, gate/up 7094 -> 5594 (precision change) | 2D configs **taken** for o_proj and MLP down; HiFi2 prefill not taken (HiFi2 drops activation bits, the 0.8 % shared-expert finding) |

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
| mHC sinkhorn | 89-90 on 1 core x 2 = 179 | HiFi3 permutation matmuls; face-restricted kernel (`SK_FACES`, stage-5 worktree) | one tile holds all users | HiFi3 **taken** (bit-identical, -7 µs); the face-restricted kernel is uncommitted stage-5 kernel work: take it when it lands (owner stage 5) |
| mHC mix chunks | 32 chunks at decode | 16 chunks (`opt_s5_mix16`) | stage 5: -6 .. -10 µs at TP4 | **rejected on one die**: +36 .. +40 µs at B=1 (`candidates_r1/opt_s5_mix16_f3` vs `opt_s5_f3`) |
| KDA conv chain (6 ternary 144, 3 typecasts 13 each) | about 254 | `kda_decode_conv` (one op) | 93 µs at 1 slot, 477 µs at 32 on one die | **taken for <= 8 slots** (-161 µs at 1 user); the chain at 32 |
| KDA in-proj split (qkv + rest weights) | packed 32960 columns | two weights | `tools/kda_split_probe.py` bfp8: prefill 2048 6550 -> 5908 µs (bit-exact), decode 410 -> 398 µs (B=32 and B=1) | not taken in stage 3: the packed weight layout is shared with the TP4 per-die grouping (stage 5), the weight cache (stage 6) and the deepseek KDA prefill kernels; -642 µs per KDA layer prefill, -12 µs decode; handed to stage 5 / 7 with these numbers |
| KDA recurrent_decode | B=1 67, B=32 838 (268 MB state) | split each head's state by V columns over 2 cores (the delta-rule update is independent per V column): 128 cores at B=1 | op microbenchmark (`tools/s3r2_glue_probe.py`): B 1 / 2 / 4 / 8 / 16 / 32 = 74.6 / 121.8 / 178.9 / 262.3 / 472.8 / 849.3 µs (112 .. 316 GB/s): **latency bound at B=1** (one head per core, 64 of 110 cores, serial over a 128 x 128 fp32 state), state-bandwidth bound only at 32 | kernel change in `ttnn.experimental.kda.recurrent_decode`; owner: KDA kernel (stage 5) |
| MoeSortedDispatch | 74.5 on 1 core (B=32), 1.22 ms at 2048 rows | multi-core histogram + scan + scatter | the op is single-core by design (`const CoreCoord core{0,0}`, a counting sort) | blocker: kernel rewrite; owner: MoE dispatch kernel (stage 5 / TTNN) |
| MoE routing reshape `[1,32,32(8),32(1)]` -> `[1,1,32,32(8)]` fp32 | 35.6 on 1 core (B=32) | ROW_MAJOR -> reshape -> TILE | measured 30.1 vs 39.0 µs host-timed, exact (`ttnn.view` illegal here) | about -9 µs per MoE layer; needs a `tt/moe.py` edit (shared, stage-6/7 owned): handed over with the measurement |
| DSA indexer qi reshape `[1,1,32,4096]` -> `[1,1,1024,128]` bf16 | 42.1 (B=32) | ROW_MAJOR round trip; `nlp_create_qkv_heads` | round trip 33.5 vs 49.6 µs host-timed, exact; the heads op gives head-major order at 55 µs | about -10 .. -16 µs per DSA layer; needs a `tt/dsa.py` edit (stage 8 edits `dsa.py` now): handed over with the measurement |
| MoE gather rows / untilize / slices (B=32) | 637 / 260 / 134 (+ others) | down projection with per_core_N 16 / 32 and `untilize_out`; smaller row tiles | the grouped sparse factory requires per_core_N == Nt (TT_FATAL `sparse_matmul_multicore_reuse_grouped.cpp:93`, at 32 and 2048 rows); about 1.8 real rows per 32-row group; the slice exists because `ttnn.swiglu` has no GLM clamp | blocked: N splitting in the grouped sparse factory, or a smaller row tile; owner: TTNN sparse matmul / stage 5 |
| Decode activations in DRAM | every matmul input 0 | L1 / sharded | 64-256 KB activations, one reshard each; the sharded norm feeds the packs directly | rejected except the producer-sharded path |

## Performance accounting (final default, `candidates_r2/final`)

Bytes per step: weights from the op shapes and dtypes (bfp8 1.0625, bfp4 0.5625, bf16 2 B per element; 8 routed experts
at B=1, about 178 distinct at B=32), KDA state fp32 read + write; DRAM 512 GB/s.

| Case | Device µs | Host wall µs / step | Host - device | Bytes / step | DRAM floor µs | Roofline fraction |
|---|---|---|---|---|---|---|
| kda_dense 1 | 1727 | 2390 | 663 | 350 MB | 684 | 39.6 % |
| kda_moe 1 | 1974 | 2860 | 886 | 331 MB | 647 | 32.8 % |
| dsa_moe 1 | 2047 | 3229 | 1182 | 281 MB | 549 | 26.8 % |
| kda_dense 32 | 2713 | (7413, one loaded run) | - | 610 MB | 1192 | 43.9 % |
| kda_moe 32 | 11535 | 12304 | 769 | 3.00 GB | 5855 | 50.8 % |
| dsa_moe 32 | 11263 | 12171 | 908 | 2.69 GB | 5250 | 46.6 % |

Host wall minus device (0.66 .. 1.2 ms per step) is the input copy between replays plus the replay-boundary gap
(235 / 280 µs, KDA / DSA; gaps inside one replay 43 .. 127 µs); the stage-7 generator owns the step loop. Prefill is
compute bound.

Per op group, µs per layer (final default):

| Group | kda_dense 1 | kda_moe 1 | kda_moe 32 | dsa_moe 1 | dsa_moe 32 |
|---|---|---|---|---|---|
| Routed sparse matmuls | - | 446 | 7268 | 446 | 7727 |
| Dense matmuls | 984 | 636 | 635 | 555 | 551 |
| Norms (sharded) | 17 | 18 | 24 | 29 | 29 |
| Sinkhorn | 172 | 172 | 172 | 172 | 172 |
| mHC mix / pre / post | 217 | 217 | 216 | 217 | 216 |
| KDA conv (op at 1 user, 12-op chain at 32) | 93 | 89 | 145 | - | - |
| KDA recurrence | 67 | 67 | 839 | - | - |
| Router TopK | - | 34 | 34 | 34 | 34 |
| Data movement | 65 | 102 | 1487 | 244 | 1714 |
| Element-wise / other (incl. DSA ternary) | 113 | 194 | 714 | 292 | 746 |
| Attention core (sparse SDPA, context 128) | - | - | - | 56 | 74 |

Remaining gaps at 1 user: the fixed per-layer work (sinkhorn 172, mHC 215, element-wise 114 .. 292, data movement
65 .. 244 µs), the KDA in-proj at 65 % of DRAM (its DRAM-sharded HiFi2 form trips a watcher assert), the B=1 sparse
matmuls at 45 % of DRAM, the latency-bound recurrence (67 µs). At 32 users: the sparse
matmuls (69 % of DRAM), MoE glue (1.5 .. 1.7 ms) and the KDA state (839 µs). Each has an audit row with its owner.

## Review #2: prefill and sparse-decode items (P2-2, P2-4)

Probes by a subagent; logs `logs/s3r2_{bmm_repro,bmm_repro2,final,prefill2d,glue,glue2,tracy_deep_*}.log`.

**DSA per-head reuse config on prefill rows (P2-2a): a TTNN matmul bug.** `tools/s3r2_bmm_repro3.py` (standalone):
`[1,64,2048,K]` bf16 x `[1,64,K,N]` bfp8, `MatmulMultiCoreReuseProgramConfig` on the full 11 x 10 grid, per_core_N = Nt,
in0_block_w 8 / 4. The first-allocated input gives PCC 0.999999; a second input of the same shape gives PCC 0.773 with
23 of 64 heads wrong (per_core_M 8; 42 heads at per_core_M 2; the same head sets for the W_UV shape). Clearing the
program cache does not help, swapping the two tensors' values moves the failure with the second buffer, and per_core_M =
Mt is correct. So it depends on the second buffer allocation when per_core_M < Mt - not data, not the program cache.
In the suite chunk 1 read the first buffer (correct) and the continuation chunks read new buffers (wrong). The shipped
decode path uses per_core_M = Mt = 1 and is not affected; the prefill knob stays off. Owner: TTNN batched reuse matmul
factory; repro `python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r2_bmm_repro3 --case uk --pm 8`.

**Adapted 2D prefill configs (P2-2b, `tools/s3r2_prefill2d.py`, synthetic weights, 2048 rows, bfp8, HiFi4):** gate/up
(2048 x 4096 x 24576) 11 x 10, in0 4, out block 7 x 10, subblock 1 x 2: 6020 vs 7059 µs default, bit-exact; overflowing
variants: in0 4 out 7 x 70 3,846,144 B, 7 x 35 2,036,224 B, in0 2 7 x 70 3,484,160 B, 7 x 35 1,826,560 B. KDA in-proj
(2048 x 4096 x 32960) 11 x 10, in0 4, out 1 x 94: 8273 vs 10123 µs. With packer L1 accumulation on, the default configs
run gate/up 3861 µs and KDA in 5488 µs and the 2D configs add nothing. Gate/up prefill is covered by `t_pf_hifi2` (HiFi2
+ packer L1 accumulation, real weights, above). The KDA in-proj prefill matmul runs inside the deepseek KDA prefill
layer, whose compute config the decoder does not own: packer L1 accumulation there (-4.6 ms per KDA layer at 2048 rows
in the probe) is handed to the KDA prefill owner (stage 5 / 7) with the probe.

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
per step past position 2048). It needs an edit of `_sparse_attend` (`tt/dsa.py`, edited by stage 8 now, and
`MultichipDSA`, stage 5); at B > 1 it also needs a page table with duplicated rows (not probed). Handed over with the
probe; larger key chunks (64 / 128) are illegal (top-k 2080 must be a multiple of the chunk).

## tt-perf-report advice (final CSVs, advice on: `perf_report.txt` in each `candidates_r1/<policy>/<case>/` and
`tracy/<case>/perf_report_advice.txt`)

| Advice | Where | Disposition |
|---|---|---|
| "Try a DRAM-sharded program config" | KDA out, DSA o_proj, wq_b | measured: slower at HiFi4 (DRAM-sharded table) |
| "Place input 0 in L1" | every matmul | rejected except producer-sharded (audit) |
| "HiFi2 may also work" | BFP8 HiFi4 matmuls | decode rows are DRAM bound (no gain); HiFi2 drops activation bits (0.8 % shared-expert finding); prefill HiFi2 not taken |
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
| **Review #2 final default: full suite** | **44 / 44** | `logs/s3r2_f3_suite.log` |
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
```
Job scripts: `~/glm53-flash-bringup/scripts/s3r_{main,trials,s5,follow1..4,bisect,refs2,tracy,p14}.sh`.
