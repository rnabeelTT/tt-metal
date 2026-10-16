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
  (`s1_functional3.log`, results table regenerated; rerun 2026-10-16 as `s1_functional4.log`), fused suite 43/43; 1M kda_dense 76 s / 0.99997 / decode >= 0.99997,
  kda_moe 98 s / 0.99923 / decode >= 0.99936 (`s1_long3_*.log`, contract updated); KDA watcher case rerun; KDA Tracy
  (functional and fused) refreshed.

## 2026-10-16: review #3 P2 (update scale)

- `tools/decode_stage_probe.py`, layer 4, real weights, 8 decode steps (`decode_stage_l4.log`): attention and FFN
  outputs within 0.5 % of HF per token; the residual updates scatter 0.99 .. 1.04. An fp32 host recomputation of HF's mHC
  on HF's own inputs misses HF's update by the same amount, TF32 input rounding changes nothing: the scatter is the
  bf16 HF reference, not the port.
- Gate: prefill sequence 1 +- 0.01, decode token 1 +- 0.05, mean of a test's decode tokens 1 +- 0.01.
  Functional 42/42 (`s1_functional4.log`), fused 43/43 (`s2_fused_suite4.log`).
