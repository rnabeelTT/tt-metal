# Functional decoder (stage 1): work log

Target: zai-org/GLM-5.3-Flash, E192 expert set (192 of 288 routed experts per MoE layer, mask of
autotrust/GLM5.3-Flash-E192-DGX-Spark applied to the FP8 parent; `tt/ckpt.py`). Hardware: Blackhole p300x2
(4 dies); stage 1 runs on one die (`ttnn.open_mesh_device(MeshShape(1, 1))`). Branch `glm_5_3_flash`.

## 2026-10-14

- 16:20 Workspace `~/glm53-flash-bringup`; HF reference venv (transformers 5.19, `glm5_next`); the TTNN environment
  keeps transformers 5.12 (shared by other bundles), so the HF layer runs as a subprocess (`tests/hf_ref/`).
- 16:25 E192 mask derived from the router rows of the E192 checkpoint (range reads only): every one of the 42 MoE
  layers keeps 192 parent experts in ascending order, router rows and correction biases bit-identical
  (`~/glm53-flash-bringup/artifacts/e192_mask.json`). The pruned FP8 checkpoint (204 GiB) was built shard by shard
  from the parent (`scripts/build_e192_ckpt.py`); the router rows stay 288-wide in the files and are masked on read.
- 16:30-17:00 Device ops taken onto this branch:
  - `ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn` (origin/main #49039) and the fused `mhc_post`
    (origin/nmilicevic/ds-mhc-dtype);
  - `ttnn.experimental.kda` and `models/demos/deepseek_v3_d_p/tt/kda` from origin/main (fixes #55626 / runtime
    bounds / non-aligned actual_end); main's newer host API calls translated; `prepare_chunk_recurrence` uses the
    base branch's `recip_tile<legacy_compat=false>` (= main's Precise mode at math_approx_mode=false).
  - new `ttnn.scatter_rows` (in-place row scatter into a ROW_MAJOR cache by a device index stick): the DSA latent
    cache is ROW_MAJOR for `sparse_sdpa`, `paged_update_cache` writes TILE caches only. Test: `tests/test_scatter_rows.py`.
- 16:50 Layer 0 (KDA + dense MLP) vs HF on real weights: prefill update-PCC 0.9990, decode 0.9985-0.9996.
- 17:05 Layer 3 (DSA + MoE): first `mhc_post` returns rank 2 (reshaped back); MoE masked path freed its input when the
  slice covered the whole tensor (`ttnn.slice` full extent aliases); `paged_flash_multi_latent_attention_decode` with 64
  heads x 512 needs k_chunk 32 (L1). Prefill 0.9986, decode 0.9982-0.9992.
- 17:20-17:30 DSA long-context path: per-slot ROW_MAJOR latent cache, pooled indexer keys (TILE, paged view with an
  identity table), online-softmax open-pool state for decode, top-512 pools + tail, `sparse_sdpa` for every query row.
  Bug found: with `cache_batch_idx` the sparse_sdpa indices are rows within the slot (prefill used flat rows).
  4000-token prompt (second chunk in the sparse regime) + decode at 4000..4003: prefill 0.9986, decode 0.9984-0.9994.
- 17:35 Layer 4 (KDA + MoE): prefill 0.9974, decode 0.9969-0.9988.
- 17:45 Passes made host-free (prefill_inputs / decode input buffers prepared outside the pass, constants at setup).
- 18:00-18:35 Synthetic-weight tests of the MoE layers failed (0.97-0.98). Root causes, each isolated with a probe:
  1. synthetic input streams were isotropic; the real stream has outlier channels and, from layer 3 on, four nearly
     uncorrelated streams -> per-channel stream statistics recorded (`weight_stats.json` "stream").
  2. RMSNorm with a bf16 accumulator shifted the normalized scale by ~2 % (the collapsed stream has rms ~3e-3, so
     mean(x^2) ~ eps) -> fp32-accumulating compute config on every norm.
  3. Routed experts 12-14 % too large (`tools/moe_route_probe.py`): `ttnn.sparse_matmul` with a bf16 DST accumulator
     is biased upward (+5.4 % at K = 4096, +2.9 % at K = 2048; `tools/sparse_mm_probe.py`), independent of in0_block_w;
     with fp32 DST the gather mode is exact but the masked mode returns wrong results on Blackhole (PCC 0.56-0.78 for
     every core count / subblock). Fix: the optimized MoE of the GLM-4.5-Air autoport (`tt/moe_opt.py`: gather / pairs
     decode, expert-sorted grouped prefill for every chunk) with fp32 DST and subblocks <= 4 tiles, HiFi4.
     MoE output PCC vs the HF MoE 0.997 -> 0.9998 at 1-512 rows. Remaining: routed output -2.5 % with bfp8 expert
     weights (bf16 weights: 0.9998 scale) - the bfp8 conversion appears to truncate; to be revisited with
     round-to-nearest pre-quantization in the datatype stage.
- 18:40 HF reference "long" mode (cache fill segment by segment, HF layer on the tail rows + decode steps), bit-exact
  against the full forward at 4K for a DSA and a KDA layer; 1M references started (CPU).
- 18:45-19:40 Test suite completed (35 tests: real / synthetic weights, lengths around tile / page / chunk / pool /
  dense-sparse boundaries, 4 users in mixed slots, determinism, host-fallback audit, decode from an empty state): all
  pass. Tracy perf of 12 cases (`scripts/01-tracy.sh`, `doc/functional_decoder/tracy/`). Watcher: DSA + MoE and every
  decode path clean; KDA prefill program exceeds the kernel config buffer under the watcher (environment limit).
- 19:50 1M long-context test: kda_dense pass (0.99994); dsa_moe fail (0.9877).
- 20:00-20:30 CPU experiments: bf16 rounding of the selection scores changes 10-32 of 512 pools per row at 1M, but the
  HF layer with bf16 selection stays at update PCC 0.99876 -> selection precision is not the cause.
- 20:35 Root cause (`tools/index_expand_probe.py`): the matmul pool-id expansion in `_sparse_rows` is not exact for pool
  ids >= 2**11 (TF32 operands; max error 508 tokens at HiFi4). Fixed with repeat_interleave + multiply-add; added the
  `max_batch * max_context <= 2**24` assert. 1M dsa_moe: 0.99768 / decode >= 0.99969 (pass); kda_moe 0.99855 / >= 0.99936
  (pass). DSA functional tests rerun after the fix: 13/13 pass (`logs/t1_dsa_regress.log`).
- 2026-10-14 21:40 Stage review #1 (`~/glm53-flash-bringup/artifacts/stage1_review_1.md`): more-work-needed (P1 DSA caches
  not paged; P2 stale DSA perf / watcher evidence, batch-32 decode untested, KDA-prefill watcher limit not hard,
  output-scale deficits unexplained, inactive slots mutated). Remediation below.
- 21:50-22:40 Paged DSA caches. New kernel support: `sparse_sdpa(page_table=, page_block_size=)` (reader remaps logical
  indices in place), `indexer_score_dsa(page_table=)` (K tile rows through the table), new op
  `ttnn.experimental.dsa_decode_scores` (paged decode indexer scores, per-slot limits, -inf past the limit). Unit tests
  bit-exact / PCC >= 0.9999995 vs contiguous caches. Bug found on the way: custom compute kernels must call
  `reconfig_data_format` explicitly (state_configure is a no-op without the sentinel build flag).
  `tt/dsa.py`: block pool latent / pool keys, block table [S, T/128 + 1] with a dummy overrun block, logical index lists.
  Decode inputs now also carry `active`: KDA (beta = 0, decay = 1, masked conv shift) and DSA (open-pool state kept)
  leave inactive slots untouched. Prefill pool softmax shifted per pool (no underflow across a chunk). MoE sorted-
  dispatch constants created at setup and frozen (no constant creation / eviction inside a pass).
  New tests: `test_paged_permuted_blocks` (pass, >= 0.99974), `test_inactive_slot_state` (pass, all kinds),
  `test_batch32_all_active` (kda_dense pass; MoE kinds: 3 single tokens 0.983-0.9957, all HF router near-ties with
  margins 6e-5..8e-4 -> near-tie rule with HF margins in the reference, `scripts/route_margin.py`).
- 22:30 KDA prefill under the FULL watcher with worker_l1_size 1,444,864 (kernel-config ring 87,040 B): clean
  (`scripts/01-watcher-kda-l1.sh`).
- 2026-10-15 Reference cache key now hashes the reference code (AST, no docstrings); references recomputed in parallel
  (`GLM53_REF_DRYRUN` + `tools/compute_refs.py`, systemd user units: session end kills nohup / setsid jobs).
- 2026-10-15 Routed-MoE scale deficit root-caused (`tools/moe_scale_probe.py`): bfp8 weight conversion is NOT the cause
  (emulated bfp8 weights: routed rms 0.9995); TT vs the bfp8-emulated reference 0.9756 at PCC 0.99993, routing exact,
  dense matmuls / SwiGLU exact -> the grouped / 1d sparse_matmul factories reload the fp32 partials through SrcA (TF32
  rounding at every K block). Fix ported from matmul_multicore_reuse_mcast_1d: c_5 UnpackToDestFp32.
- 2026-10-15 19:45 Stage review #2 (`artifacts/stage1_review_2.md`): more-work-needed, 4 x P2. Remediation:
  1. DSA watcher evidence refreshed on the final code: test_real_weights[dsa_moe], test_lengths[dsa_moe-4000] (sparse
     prefill), test_batch32_all_active[dsa_moe] (32-row decode), test_real_weights[kda_moe] (KDA prefill, worker_l1_size
     1,444,864): all pass, 0 suspicious watcher lines (`watcher_final/`).
  2. README perf table regenerated from the refreshed Tracy CSVs; stale 10-14 console.log.gz removed.
  3. Inactive decode rows: an idle slot's index row now keeps index 0 (no all-sentinel row: sparse_sdpa gathered a
     wrapped address before the cache); KDA idle rows of q / k / v zeroed by a select before the recurrence;
     test_inactive_slot_state now feeds the idle slot a NaN row (passes).
  4. Output scale on real weights (`tools/layer_stage_probe.py --weights real`, layers 0 / 4): dense matmuls exact
     (`tools/matmul_scale_probe.py`: 1.00003); dense MLP ran at HiFi2 (activation mantissa truncation): ffn_out on the HF
     input 0.9907 -> 0.9989 at HiFi4; KDA chunk-recurrence matmuls (KDARecurrenceProgramConfig
     affine prefix / scan) HiFi2 -> HiFi4: attn_out 0.9911 -> 0.9954. Residual KDA attn_out -0.47 % at PCC 0.99998 not
     resolved (projections exact at HiFi4, norm eps equal, fused sigmoid exact); bounded by the new test metric.
     Tests now also check the update scale |X' - X| TT / HF in [0.95, 1.05] (observed 0.984 - 1.045).
  Other: HF long mode vs full forward at 4K checked (`scripts/long_mode_check.py`: KDA layer bit-identical, DSA layer
  max |diff| 4.9e-4); dummy-block guard in the DSA page mapping.
  Evidence after the fidelity change (2026-10-15, `scripts/review2_evidence_b.sh`): functional suite 42/42
  (`s1_functional3.log`, results table regenerated; rerun 2026-10-16 as `s1_functional5.log`), fused suite 43/43; 1M kda_dense 76 s / 0.99997 / decode >= 0.99997,
  kda_moe 98 s / 0.99923 / decode >= 0.99936 (`s1_long3_*.log`, contract updated); KDA watcher case rerun; KDA Tracy
  (functional and fused) refreshed.

## 2026-10-16: review #3 P2 (update scale)

- `tools/decode_stage_probe.py`, layer 4, real weights, 8 decode steps (`decode_stage_l4.log`): attention and FFN
  outputs within 0.5 % of HF per token; the residual updates scatter 0.99 .. 1.04. An fp32 host recomputation of HF's mHC
  on HF's own inputs misses HF's update by the same amount, TF32 input rounding changes nothing: the per-token scatter
  is the bf16 HF reference. (Correction, review #4: the systematic +0.3 .. +0.6 % offset on the MoE kinds was the
  port's own mHC precision, not the reference; fixed on 2026-10-16, see below.)
- Gate: prefill sequence 1 +- 0.01, decode token 1 +- 0.05, mean of a test's decode tokens 1 +- 0.01.
  Functional 42/42 (`s1_functional5.log`), fused 43/43 (`s2_fused_suite4.log`).

## 2026-10-16: review #4 P2 (mHC offset, gate evidence)

- Root cause of the offset: the mHC kernels (`mhc_split_sinkhorn`, `mhc_post`, `mhc_pre`) read fp32 operands through
  srcA / srcB at TF32 precision, and the Sinkhorn reciprocal (`recip_tile` legacy mode) was 1.4e-3 low: comb column
  sums about 1e-3 below 1. The same TF32 reload made `kda.recurrent_decode` drift over long decodes (1000 steps: state
  PCC 0.987). Fixed in the kernels (fp32 through DST, hi / lo operand split, exact reciprocal).
- `decode_stage_l4c.log`: TT post / comb vs exact fp32 +0.30 % -> -0.02 %; `mhc_post` +0.10 % -> 0.000 %. L4 prefill
  sequences 1.0035 .. 1.0058 -> 0.9995 .. 1.0023.
- Fault injection (`GLM53_FAULT_ROUTED_SCALE=0.9722`, `s1_fault_routed.log`): 17 tests fail on the scale gate, on both
  MoE kinds (L4 prefill 0.985 .. 0.990, L3 0.979 .. 0.984, decode means 0.982). (Correction, review #5: these are the
  faulted ratios, not the bug's factor. The factor (faulted / clean) is 0.977 .. 0.979 on L3 and 0.984 .. 0.991 on L4.
  Not every real-weight MoE prefill fails: `test_lengths[kda_moe-2049]` passes with the fault.)
- Functional suite on the final kernels (`s1_functional6.log`): 41 / 42. The failure, L3 32-slot slot 2 pos 39 at scale
  1.0507, is a confirmed router flip (HF margin 1.86e-4; TT routes expert 31 for HF's 149, `flip_tt_s1302.log`).
  Near-tie tokens now get a per-token scale bound of 1 +- 0.08 and stay in the decode mean. (Replaced on 2026-10-17 by
  the y_swap reference, below. Correction, review #5: before the kernel fix pos 38 and pos 39 both flipped.)

## 2026-10-17: review #5 P2 (near-tie reference, L3 excess, kernel evidence, docs)

- P2-A, router near-ties. The margin-keyed relaxations (PCC 0.98, scale 1 +- 0.08 for every decode token with HF margin
  < 5e-3, 65 % of the MoE decode tokens) are removed. For those tokens the HF reference also gives `y_swap`, the
  unmodified HF layer rerun with the router's 8th expert replaced by its 9th (`tests/hf_ref/hf_flip_ref.py`, own cache
  key, so the base references stay cached; every other row reproduces the cached `y` bit-exactly). A token passes on
  the normal bars against `y` or `y_swap`; `y_swap` matches are counted and capped (per test max(1, 5 %), per suite
  max(2, 3 %) of the MoE decode tokens). 48 references, 0 .. 11 swap rows each, 0.4 .. 1.8 min per reference on the
  CPU (`scripts/s1r5_host/flip_refs.py`, `s1r5_flip_refs.log`).
- Functional suite (`s1_functional7.log`, 01:15 build): 42 / 42. Four tokens match `y_swap`, all others match `y`:
  pos 2052 (0.99989 / 0.9931), pos 4002 (0.99991 / 0.9998), L3 32-slot slot 2 pos 39 (0.99993 / 1.0004; against `y`
  1.0507), L4 32-slot slot 15 pos 130 (0.99963 / 0.9909). Suite total 4 of 298 MoE decode tokens.
- Fault injection on the final code (`s1_fault_routed2.log`): 15 tests fail and 2 fail at teardown (L3 decode means).
  L4: 4 real-weight prefills + the synthetic prefill; L3: 10 real-weight prefills + 2 decode means. Factors (faulted /
  clean): L3 0.977 .. 0.979, L4 0.984 .. 0.991 (real), synthetic L4 0.975, synthetic L3 0.994 (passes). L4 decode
  means pass with the fault (0.9933, 0.9912).
- P2-B, L3 excess. HF fp32 taps of the same reference (`tests/hf_ref/hf_fp32_taps.py`); `tools/layer_stage_probe.py`
  and `tools/decode_stage_probe.py` take `--hf32`. Against HF fp32, L3 100-row prefill: update 1.0050, attention
  output 0.9942 (on HF's own input 0.9941), FFN output 1.0047, FFN on HF's FFN input 1.0019 (routed 1.0028, shared
  0.9982) (`s1r5_layer_stage_l3.log`); 8 decode steps: update 1.0024, attention 0.9949, FFN on HF input 0.9998
  (`s1r5_decode_stage_l3.log`). `tools/dsa_stage_probe.py` (fp32 torch on the device weights, `s1r5_dsa_stage_l3.log`):
  `ttnn.rms_norm` q_a 0.9991, latent 0.9993; `flash_mla_prefill` on exact inputs 0.9983; W_UK / W_UV / o_proj
  matmuls exact; attention output 0.9950. Host checks (`s1r5_host_checks.log`, `scripts/s1r5_host/run_all.sh`): the
  port's bfp8 attention weights cost 0.9991; the fp32 FFN on the dumped TT FFN input is 1.0034 of the same FFN on HF's
  input (4 of 100 rows reroute; the others move as much); the HF fp32 layer with its attention output x 0.9942 gives
  update 1.0035, FFN 1.0021; random 1 % noise on the FFN input reroutes 8 .. 12 rows and moves the FFN by -0.08 ..
  +0.04 % only. So the attention deficit (shared `rms_norm` / SDPA kernels) gives about +0.35 %, the MoE device compute
  about +0.2 %. Not fixed in stage 1; README Limitations. The "fixed" wording of review #4 now covers mHC / L4 only.
- P2-C, kernels. ttnn rebuilt at HEAD (`cmake --build build_Release --target ttnn`, 01:15 UTC: the sinkhorn and
  mhc_post factories). `mhc_post` takes its SFPU / hi-lo path only for fp32 inputs (define `MHC_POST_FP32`); bf16 /
  bfp8_b inputs keep the FPU path and do not allocate the hi / lo CBs. On the rebuilt ttnn:
  `tests/test_mhc_{split_sinkhorn,post,pre,mix}.py` + `test_kda_recurrent_decode.py` 37 passed (`s1r5_kernels.log`;
  `mhc_post` fp32 max rel err <= 3.1e-7, bf16 path PCC 0.9999986); upstream nightly `test_mhc_split_sinkhorn.py`
  (incl. sharded c20_tpc1 / c4_tpc5), deepseek `test_mhc.py` and `test_mhc_post_op.py` 41 passed, 1 skipped by design
  (`s1r5_upstream_mhc.log`); deepseek `tests/perf/test_mhc_perf.py` + `test_fp32_kernel_perf.py` 13 passed
  (`s1r5_mhc_perf.log`). Watcher (`watcher_r5/`): `test_real_weights[dsa_moe]`, `test_real_weights[kda_moe]`,
  `test_decode_only[kda_moe]` pass, 0 findings.
- P2-C, perf. The 12 functional Tracy cases were recaptured on the final code (12:08 .. 12:24 UTC); the README table is
  regenerated from these CSVs (`s1r5_perf_tables.log`). The mHC kernels cost +35 .. +36 us per decode step and
  +458 .. +483 us per 2048-row prefill per layer (Sinkhorn 164 -> 179 us, `mhc_post` 95 .. 97 -> 115 .. 117 us at
  decode; `mhc_post` 1527 .. 1570 -> 1988 .. 2014 us at 2048 rows). The 2048-row MoE prefills include an uncommitted
  stage-6 change in `tt/moe_opt.py` (routed experts in 1024-row blocks): +31 ops, +12 .. +13 ms; disclosed in the README.
  The stage-2 commit a01bb86cfe5 (12:11 UTC) picked up this capture's first two cases (`kda_dense-128`, `kda_dense-2048`)
  together with its own earlier captures of the other 6 KDA cases; the working tree now holds this capture for all 12.
- After the kernel tests, `mhc_post`'s bf16 path called the deprecated `mul_tiles_init(a, b)`; it now calls
  `mul_init(a, b, true)`, which the deprecated wrapper forwards to (same code). Rechecked: `s1r5_post_recheck.log`
  `test_mhc_post.py` and the deepseek `test_mhc_post_op.py`, 14 passed, no compiler warning.
- P2-D, docs: fault table counts (4 L4, 10 L3) and per-test list; test comment (no longer "every real-weight MoE
  prefill"); bug factor values (faulted / clean); pos 38 and pos 39 both flipped before the kernel fix; full-stream
  minimum 0.99788; the 0.9934 sequence is the 1-row L0 prefill (a token for `check()`), the 42 multi-row sequences are
  0.9955 .. 1.0059; "fixed" now covers mHC / L4 only.
