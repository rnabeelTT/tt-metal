# GLM-5.3-Flash (E192) optimized decoder

Stage 3 of the tt-model-bringup flow. `tt/optimized_decoder.py` (`OptimizedDecoder`) is the fused decoder
(`tt/fused_decoder.py`) with a precision policy, decode and prefill program configurations, and the stage-5 decode
options that need no collective, for one die. Contract, state, paging and semantics are those of
`tt/functional_decoder.py`. The policies are `OptConfig` entries in `CANDIDATES`; `SELECTED` is the default.

Review #1 (`~/glm53-flash-bringup/artifacts/stage3_review_1.md`) changed the reference, the default and the evidence.
The earlier text of this README is in git history (commit 21dc7532f26). The work log has the failures and the order
of events. Every `logs/<name>.log` cited here is copied to `doc/optimized_decoder/logs_r1/` (from
`~/glm53-flash-bringup/logs/`).

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
| mHC, KDA recurrence, sinkhorn | unchanged | unchanged | fp32-exact stage-1 / 2 kernels; dispositions in the audit |

`OptConfig` fields: `kda_in_dtype`, `kda_out_dtype`, `moe` (expert dtypes, fidelities, sparse geometry),
`dsa_bmm_dtype`, `dsa_bmm_reuse`, `out_proj_1d`, `single_die_mc` (the pinned stage-5 option set),
`prefill_cfg`, and the trial fields `dsa_proj_dtype`, `mlp_dtype`, `mlp_fidelity`, `mix_chunks_decode`. The stage-5
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
| `logs/s3r_final_suite.log` | **`opt_s5_pf` (final)** | **44 / 44** | L3 0.998784, L4 0.996759, L0 0.999933 | L3 0.9929 .. 0.9960, L4 0.9957 .. 0.9984, L0 0.9948 .. 0.9969 | 3 of 298 (cap 8) |

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
`doc/optimized_decoder/candidates_r1/<policy>/` (`ops.csv`, `perf_report.csv`, `perf_report.txt` with advice).
Decode at 32 users carries a run-to-run spread of the sparse matmuls of up to +-4 % between sessions (the same policy:
dsa_moe 11779 / 11276 µs, kda_moe 11530 / 11755 µs; per-op deltas below exclude it). B=1 repeats within 2 µs.

Final default (`opt_s5_pf`, `tracy_r1/`, 2026-10-18, advice on), against the stage-2 fused decoder (`doc/fused_decoder/tracy`,
3-replay means) and the pre-review stage-3 default (`opt`; same-day control `candidates_r1/opt`):

| Case | Stage 2 fused | Stage 3 `opt` (control) | Final `opt_s5_pf` | vs fused | vs `opt` | Host wall, final (ms / step or pass) |
|---|---|---|---|---|---|---|
| decode kda_dense, 1 user | 2443 | 2083 | **1971** | -19.3 % | -5.4 % | 2.630 |
| decode kda_moe, 1 user | 2979 | 2286 | **2145** | -28.0 % | -6.2 % | 2.853 |
| decode dsa_moe, 1 user | 3347 | 2421 | **2062** | -38.4 % | -14.8 % | 2.865 |
| decode kda_dense, 32 users | 3262 | 2901 | **2791** | -14.4 % | -3.8 % | 3.473 |
| decode kda_moe, 32 users | 15898 | 11755 | **11815** | -25.7 % | +0.5 % (sparse spread) | 12.557 |
| decode dsa_moe, 32 users | 15590 | 11276 | **10769** | -30.9 % | -4.5 % | 11.646 |
| prefill 2048 kda_dense | 30000 | 29968 | **26853** | -10.5 % | -10.4 % | 28.44 |
| prefill 2048 kda_moe | 89115 | 66854 | **66588** | -25.3 % | -0.4 % | 68.06 |
| prefill 2048 dsa_moe | 119821 | 97988 | **77052** | -35.7 % | -21.4 % | 77.84 |
| prefill 128 kda_dense / kda_moe / dsa_moe | 3200 / 19679 / 19824 | - | **2959 / 13968 / 14691** | -7.5 / -29.0 / -25.9 % | | 4.93 / 15.90 / 16.09 |

All candidate runs: `candidates_r1/{opt, opt_s5, opt_s5_f3, opt_s5_mix16_f3, opt_s5_pf, opt_rhifi3, opt_rhifi4,
mc_tp1, t_*}`; `opt_s5` B=1 repeats within 2 µs across sessions (2059 / 2057, 1970 / 1970, 2140 / 2142).

Stage-5 options on one die (`opt` -> `opt_s5`, same session control `candidates_r1/opt`), per-op deltas at B=1:
dsa_moe -363 µs (norms -141, indexer wk / wg -98, q_a -66, kv_a -56, w_iw -31, router -31; the two packs +46), kda_moe
-146 µs (norms -119, router -31), kda_dense -113 µs (norms).

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

## DRAM-sharded, geometry and prefill configurations (`tools/s3r_probe_dense.py`, `s3r_probe_sparse.py`, `s3r_probe_dsa.py`)

Logs: `logs/s3r_p14_dense_kdain.log`, `s3r_p14_dense_out.log`, `s3r_p14_sparse.log`, `s3r_p14_dense_pf.log`,
`s3r_p14_sdpa.log`, `s3r_p14_bmm_m{2,4,8}.log`. Host-timed with a device sync; dense / DSA on synthetic weights at the
real shapes, the sparse sweep on real layer-4 weights at the `opt` policy.

| Item | Candidates and numbers | Decision |
|---|---|---|
| KDA in-proj 32 x 4096 x 32960 bfp8 HiFi4 (default 401.3 µs; 1D 64 / 110 cores 432 / 409) | DRAM-sharded natural N (129 tiles per bank): 1 reader overflows L1 at in0_block_w 8 .. 1; **3 readers run, 484 .. 502 µs**. N padded to 1056 tiles: 1 reader in0 4 / 2 overflow; 2 or 3 readers 492 .. 528 µs. N split (1024 tiles DRAM-sharded + 6 tiles 1D + concat): best 486.8 µs (concat 31 µs) | reject at HiFi4: the DRAM-sharded kernel is compute-bound there. It wins only with HiFi2 (333.5 vs 398.3 µs) or bfp4, which the precision trials reject |
| KDA out 8192 x 4096 / DSA o_proj 16384 x 4096 decode | 1D 101.7 / 198.9 µs; DRAM-sharded HiFi4 130.0 / 234.9; DS LoFi 78.6 / 150.6; 4 readers illegal (16 tiles per bank) | 1D kept at bfp8 HiFi4 |
| Prefill on the DRAM-sharded weight | `create_prefill_matmul_program_config(2048, K, N)` on 8 x 10: PCC 1.000000, same time as interleaved; KDA out 3872 -> 2420 µs, o_proj 7684 -> 4831 µs. The (11, 10) grid on a DRAM-sharded weight returns wrong output (PCC nan) | the "second weight copy" premise was wrong; DS weights not taken because decode DS loses at HiFi4 |
| Routed sparse matmuls, B=1 gather (default 767.4 µs, 64 cores, in0 32) | in0 16 -6.9 (noise), in0 64 / 128 +4 / +18, 32 cores +24 .. +47, 128 cores invalid; down variants +2 .. +26; pairs path +687 / +1421 | default kept (all variants bit-identical) |
| B=32 grouped (default 8161.7 µs, in0 4, subblock 4) | gate/up in0 2 +697, down in0 2 +308, subblock 2 / 1 +105 / +637, in0 8 / 16 overflow L1 (2.11 / 3.32 MB); pairs 16748 | default kept |
| Prefill 2048 sorted / grouped (default 52151 µs with the 1024-row split) | in0 2 +1073 .. +2832, subblock 2 +583, in0 8 overflows; the uncommitted stage-6 1024-row split costs 7.8 ms per MoE layer (44373 µs unsplit, bit-identical) | default kept; the split is a stage-6 / 7 memory decision |
| DSA per-head matmul at 2048 prefill rows (default 1900 / 2090 µs) | reuse config per_core_M 2 / 4 / 8: 1426 / 941 / **765** (q_abs), 1335 / 873 / 665 (o_uv), PCC 0.999999, no hang (the stage-3 hang was per_core_M = Mt = 64) | **not taken**: the probe is fast and accurate on one chunk, but in the decoder the DSA continuation chunks (prompts over 2048) read 11-22 % low with it on W_UK and non-finite on W_UV (`logs/s3r_bisect_pf.log`, `logs/s3r_final_suite.bmm_wuk.log`); not diagnosed. Knob kept off (`GLM53_PF_OFF=bmm_on` turns it on) |
| `flash_mla_prefill` 2048 rows (default 22852 µs) | q128 / k128 **6654**, q128 / k64 6753, q64 / k64 11625; PCC 0.99996 vs default; k256 / k512 / q256 overflow L1 | **taken** |
| Large prefill matmuls (bfp8) | explicit 2D config on the full grid, bit-exact: DSA o_proj 7707 -> **3708**, MLP down 5792 -> **2778**; KDA in / MLP gate-up overflow L1 at in0 4 on that grid. HiFi2: KDA in 10115 -> 7951, gate/up 7094 -> 5594 (precision change) | 2D configs **taken** for o_proj and MLP down; HiFi2 prefill not taken (HiFi2 drops activation bits, the 0.8 % shared-expert finding) |

Prefill configurations end to end: `candidates_r1/opt_s5_pf` (with the W_UK reuse config, later removed) dsa_moe 2048
rows 97302 -> 77407 µs, kda_dense 29941 -> 26759 µs, kda_moe unchanged (66865); the final numbers without it are in the
performance table.

## Operation-topology audit (decode, per layer, one die)

Measured on the final CSVs (`candidates_r1/opt`, `opt_s5`); stage-5 evidence from
`doc/optimized_multichip_decoder/README.md`.

| Op group | Current (µs) | Candidate | Constraint / evidence | Action |
|---|---|---|---|---|
| Decode RMSNorm (input / post; DSA q_a / kv_a) | 65-67 on 1 core x 2 (KDA), 5 norms 170 (DSA) | width-sharded L1 norm | PCC 0.999999, 1 bf16 ulp (stage 5) | **taken** (`single_die_mc`): 136 -> 18 (KDA), 170 -> 29 (DSA) |
| DSA x-projections (q_a 66.5, kv_a 56, wk / wg 99, w_iw 31) | 252 | two packs on 1D configs, weights shared with prefill | mixed dtypes: two packs | **taken**: -237 net with the packs (+46) |
| DSA q_b / wq_b, router | 82 / 35 / 49 | 1D configs | fp32 router logits exact | **taken** (router -31) |
| Router TopK | 34 on 1 core | fused MoE gate op | bf16 scores in that op; GLM needs fp32 top-k (stage-1 route flips) | rejected |
| mHC sinkhorn | 89-90 on 1 core x 2 = 179 | multi-core over users; HiFi3 permutation matmuls | one tile holds all users; the stage-5 rework measures a kernel fidelity option (`sinkhorn_perm_fidelity`) on 4 dies | not on one die in stage 3: handed to stage 5 (owner of the kernel option), cost 179 µs per layer |
| mHC mix chunks | 32 chunks at decode | 16 chunks (`opt_s5_mix16`) | stage 5: -6 .. -10 µs at TP4 | **rejected on one die**: +36 .. +40 µs at B=1 (`candidates_r1/opt_s5_mix16_f3` vs `opt_s5_f3`) |
| KDA conv chain (6 ternary 144, 3 typecasts 13 each) | about 250 | one custom op | stage 5 built `kda_decode_conv` (`MCOpts.kda_conv_op`) in its rework; on one die it runs as MultichipDecoder at TP = 1 | handed to stage 5; measured on one die below |
| KDA in-proj split (qkv + rest weights) | packed 32960 columns | two weights | `tools/kda_split_probe.py` bfp8: prefill 2048 6550 -> 5908 µs (bit-exact), decode 410 -> 398 µs (B=32 and B=1) | not taken in stage 3: the packed weight layout is shared with the TP4 per-die grouping (stage 5), the weight cache (stage 6) and the deepseek KDA prefill kernels; -642 µs per KDA layer prefill, -12 µs decode; handed to stage 5 / 7 with these numbers |
| KDA recurrent_decode | B=1 67, B=32 838 (268 MB state) | - | B=32 at 64 % of DRAM | no action: state-bandwidth bound |
| MoeSortedDispatch | 74 on 1 core (B=32) | multi-core dispatch | custom op | not taken: 0.6 % of B=32; open item |
| MoE routing reshape | 35.7 on 1 core (B=32) | - | fp32 [T, 192] view | not taken: 0.3 %; open item |
| DSA indexer qi reshape | 42.1 (B=32) | heads op | - | not taken: 0.4 %; open item |
| MoE gather rows / untilize / slices (B=32) | 611 / 264 / 290 | fewer padded rows per expert group | 32 rows per group, 1.8 real rows | stage-5 open item |
| Decode activations in DRAM | every matmul input 0 | L1 / sharded | 64-256 KB activations, one reshard each; the sharded norm feeds the packs directly | rejected except the producer-sharded path |

Stage-5 rework options measured on one die (`MultichipDecoder` at TP = 1, `candidates_r1/mc_tp1`, same day):
kda_dense 1 user 1745 µs, kda_moe 1967, dsa_moe 2040; 32 users 2941 / 11570 / 11214; prefill kda_dense 17811,
kda_moe 65949; dsa_moe prefill fails (its `prefill_2d` overflows L1 on one die). Against `opt_s5` (1970 / 2140 / 2059)
the rework (KDA conv op, 1D program specs, MoE grouped in0 8) saves 175 .. 225 µs per KDA layer at 1 user. These
options are stage-5 code in active rework; the production path (4 dies) takes them there.

## Performance accounting

Measured on the `opt` CSVs (`tracy/`, 3-replay means; the `opt_s5` decode deltas above lower the B=1 device times by
113 .. 363 µs). Bytes per step: weights from the op shapes and dtypes (bfp8 1.0625, bfp4 0.5625, bf16 2 B per element;
8 routed experts at B=1, about 178 distinct at B=32, all 192 in prefill), KDA state fp32 read + write. DRAM 512 GB/s.

| Case | Device µs | Host wall µs / step | Bytes / step | DRAM floor µs | Roofline fraction | Gaps in one replay | Gap at the replay boundary |
|---|---|---|---|---|---|---|---|
| kda_dense 1 | 2082 | 2782 | 350 MB | 684 | 33 % | 58 | 236 |
| kda_dense 32 | 2901 | 3614 | 610 MB | 1192 | 41 % | 43 | 235 |
| kda_moe 1 | 2287 | 3007 | 331 MB | 647 | 28 % | 89 | 235 |
| kda_moe 32 | 11530 | 12235 | 3.00 GB | 5855 | 51 % | 76 | 236 |
| dsa_moe 1 | 2421 | 3212 | 281 MB | 549 | 23 % | 127 | 279 |
| dsa_moe 32 | 11779 | 12602 | 2.69 GB | 5250 | 45 % | 104 | 280 |

Host wall minus device (0.7 .. 0.8 ms per step) is the input copies between replays plus the replay-boundary gap; the
device cannot remove it (stage 7 owns the generator loop). Sparse matmuls: B=32 gate/up 355 GB/s (69 %), down 349 GB/s
(68 %), B=1 230 GB/s (45 %, 64 cores; the geometry sweep found no faster legal config). Prefill is compute bound
(dense matmuls 24 .. 78 % of FLOPs, sparse about 49 %).

Remaining gap at B=1 (per layer, about 1.6 ms above the DRAM floor): fixed non-matmul work about 900 µs (sinkhorn 179,
norms 136 .. 170 -> 18 .. 29 with `single_die_mc`, mHC mix / pre / post 216, KDA conv chain about 180, element-wise
about 240), the DSA x-projections (252 -> 15 net with the packs and norms), the KDA in-proj at 65 % of DRAM, the B=1
sparse matmuls at 45 %. At B=32: the sparse matmuls at about 69 %, MoE data movement 1.5 .. 1.7 ms (gather rows,
untilize, slices), the KDA state 838 µs. Each item has a row in the audit.

Per op group, µs per layer (`opt`):

| Group | kda_moe 1 | kda_moe 32 | dsa_moe 1 | dsa_moe 32 |
|---|---|---|---|---|
| Routed sparse matmuls | 446 | 7137 | 446 | 7890 |
| Dense matmuls | 664 | 668 | 796 | 795 |
| Norms (1 core) | 136 | 141 | 170 | 172 |
| Sinkhorn | 179 | 179 | 179 | 179 |
| mHC mix / pre / post | 217 | 216 | 215 | 216 |
| KDA conv chain | about 183 | about 146 + typecasts | - | - |
| KDA recurrence | 67 | 838 | - | - |
| Router TopK | 34 | 34 | 34 | 34 |
| Data movement | 104 | 1460 | 229 | 1679 |
| Element-wise / other | 235 | 631 | 246 | 678 |
| Sparse SDPA | - | - | 51 | 61 |

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
| **Final default `opt_s5_pf`: full suite** | **44 / 44** | `logs/s3r_final_suite.log` |
| **Final: watcher** suite subset incl. B=32 dsa_moe + B=1 traced decode per kind | **17 / 17 + 3 / 3, 0 suspicious lines** | `watcher_correctness/{suite,decode1}/`, `logs/s3r_final_watcher_*.log` |
| **Final: repeated runs** `test_batch32_all_active[dsa_moe]` and `test_lengths[dsa_moe-2051]` x 5 | **10 / 10** | `logs/s3r_final_stress.log` |
| First `opt_s5_pf` default (with the W_UK prefill reuse config) | 6 failed: DSA continuation chunks 11-22 % low; config removed | `logs/s3r_final_suite.bmm_wuk.log` |
| `opt_s5` (pre-final): watcher (`TT_METAL_WATCHER=10`, worker L1 1,444,864 B, `NOINLINE`) | 17 / 17 + 3 / 3, 0 suspicious lines | `logs/s3r_watcher_*.log` |
| `opt_s5` (pre-final): `test_batch32_all_active[dsa_moe]` x 5 (the 2026-10-16 stall case) | 5 / 5 | `logs/s3r_stress_b32dsa.log` |
| Stage-1 bug injected (`GLM53_FAULT_ROUTED_SCALE=0.9722`) | caught (14 failed, 3 errors) | `logs/s3r_fault_routed.log` |
| Stage-2 path unchanged after the `tt/fused_decoder.py` refactor fix | fused suite subset 9 / 9, all 312 check lines identical to `s2_fused_suite6.log`; `fused_flag_check kda_decode_fold 0 4` bit-exact | `logs/s3r_fused_subset.log`, `logs/s3r_fused_flag_fold.log` |
| `test_policy_reached_weights` | weight dtypes and routed / shared fidelity of the built layers equal the policy | suite logs |

The 2026-10-16 16:10 stall is classified as infrastructure (work log).

## Limitations

- Decode performance is measured at context 128. The context contract is unchanged (smaller weights only).
- The measured code includes uncommitted later-stage changes in the worktree (`tt/moe_opt.py` 1024-row MoE split,
  `tt/dsa.py` stage-7 traced prefill, `tt/moe_dispatch.py`, `tt/weight_cache.py`, `tt/fused_decoder.py` refactor) and
  the untracked `tt/multichip_decoder.py` that `single_die_mc` imports. A stage-3 commit must include them or be
  re-measured.
- The KDA bfp4 / DSA bfp4 / dense-MLP bfp4 rejections hold at layer level against HF; stage 8 owns the model-level
  frontier.

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
