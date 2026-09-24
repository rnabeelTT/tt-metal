# Optimized decoder — work log (stage 3)
Baseline: stage-2 fused decoder (commit 1a4e5f255eb; doc/fused_decoder/tracy). Device: one Blackhole die, real weights (layers 0/3).
Times below are host-clock estimates; authoritative timestamps are inside the referenced logs.

## 2026-09-24
- (host 04:25) Op-topology audit from the stage-2 fused Tracy tables (per traced decode replay; prefill T=128):
  GDN layer decode 118 ops / 1.464 ms: Matmul 550 us (qkvz+a+b in-proj [1x3072]x[3072x20608], out-proj [8192x3072], packed shared
  expert, router), SparseMatmul 307 us (gather gate/up + down, 8 experts), ReshapeView 156, BinaryNg 139, LayerNorm 111 (3 norms on
  ONE core: attention norm, ff norm, GDN out-norm), untilize/tilize ~45 (conv-window FIR), recurrent step small matmuls ~25, gate op ~20.
  Attention layer decode 89 ops / 1.128 ms: Matmul 486 us (fused qkv [3072x17408], o_proj [8192x3072], packed shared, router),
  SparseMatmul 305, LayerNorm 109 (2 norms + q/k norms), SDPA-decode 25, paged_update_cache 2x4, BinaryNg ~50.
  Prefill T=128: SparseMatmul 17.6 / 21.5 ms of 26.7 / 27.7 ms (masked 32-token groups over the ~164-expert union), Unary (swiglu over the
  expanded [E,32,1024]) 2.2 ms, elementwise 1-2 ms.
  Candidates (families in tools/sweep_optimized.py): routed-expert dtype/fidelity (bfp8 HiFi2 -> bfp8 LoFi -> bfp4 LoFi, mixed),
  shared-expert dtype/fidelity, attention/GDN projection dtype (bfp4 trial, OPT-007), KV cache bfp8 (OPT-002), sparse-matmul geometry
  per role (cores x in0_block_w; masked chunk 32/64/128), sharded decode norms (OPT-003), SDPA-decode chunk sizes, gather/masked
  decode threshold (batch>1). Roofline per decode token (one die, bfp8 everything): GDN layer ~184 MB (in-proj 63 MB, out 25 MB,
  8 experts 80 MB, shared 10 MB) -> 0.45 ms at ~410 GB/s; attention layer ~150 MB -> 0.37 ms. Measured 1.46 / 1.13 ms = 31% / 33%.
- (host 04:30) tt/optimized_decoder.py (`OptimizationConfig`, `OptimizedDecoder`), shared opt-in knobs: `AttentionConfig.weight_dtype`,
  `GDNConfig.weight_dtype`, SDPA-decode chunk sizes, bfp8 cache fill cast in the attention kernel, `args.decode_norm_sharded`
  (sharded RMSNorm on a width-sharded L1 copy of the residual). tools/sweep_optimized.py measures PCC vs HF (real weights, T=128 prefill,
  decode at 128), warmed prefill wall and traced 3-replay decode wall per candidate; rows -> doc/optimized_decoder/sweeps/*.jsonl.
- (host 04:19) sweep chain 1 (sweeps/{baseline_stage2,precision,shared}.jsonl; real weights; PCC prefill T=128 / decode; wall ms prefill / traced decode):
  baseline stage-2 policy (routed bfp8 HiFi2, shared bfp8 HiFi2): L0 0.99882/0.99958, 28.66/1.488; L3 0.99960/0.99983, 28.53/1.190.
  routed experts: bfp8 LoFi 28.40/1.482 (L0) — same PCC as HiFi2; bfp4 gate-up + bfp8 down LoFi 23.26/1.414; bfp4 LoFi 21.15/1.370
  (L0 PCC 0.99882/0.99958 — unchanged to 4 decimals); bfp4 HiFi2 21.69/1.378 (bit-identical PCC to LoFi: a bfp4 srcB has no
  mantissa bits left for the second fidelity phase). L3: bfp4 LoFi 19.36/1.076 (PCC 0.99952/0.99983 vs 0.99952/0.99983 bfp8).
  SELECTED routed = bfp4 / LoFi (also the only dtype that fits the 4-die DRAM budget).
  shared expert: bfp8 HiFi2 21.29/1.372 vs bfp8 LoFi 20.89/1.370 (PCC L3 0.99959 -> 0.99952) vs bfp4 LoFi 21.84/1.369 (PCC L0
  0.99879, L3 0.99932): no measurable time difference, so SELECTED shared = bfp8 / HiFi2 (best PCC).
- (host 04:20) mixer projection dtype (sweeps/mixer_weights.jsonl, real weights, materialized dtype recorded per row):
  attention q/k/v/o bfp4: decode 1.074 -> 1.024 ms (-4.7%) but PCC prefill 0.99952 -> 0.99466 (BELOW the 0.995 bar), decode 0.99983 -> 0.99819.
  REJECTED (model-visible real-weight correctness loss; OPT-007 trial recorded). GDN in-proj bfp4: no time gain (1.366 -> 1.371 ms decode,
  20.91 -> 21.09 prefill) and PCC 0.99882 -> 0.99528 / 0.99958 -> 0.99774. REJECTED. Both stay bfp8 (HiFi2 fp32-acc as in qwen36).
- (host 04:21) KV cache dtype (sweeps/kv.jsonl, attention layer, context 128): bf16 PCC 0.99952/0.99983, 19.16/1.074 ms; bfp8 PCC
  0.99954/0.99983, 19.20/1.077 ms (SDPA-decode is ~25 us at this context, so no time signal; the win is 2x KV capacity). Candidate
  SELECTED pending the long-context check (attention 16421-token HF parity + K/V content with bfp8 cache).
- (host 04:23) gather (decode) sparse-matmul geometry (sweeps/gather.jsonl; cores x in0_block_w cap for gate/up [K=3072] and down [K=1024]):
  gu64/in96 + dn64/in32 (stage-2 default) 1.375 / 1.078 ms; gu64/in32 + dn64/in32 1.358 / 1.063 ms (best, -1.2%); gu64/in8 + dn64/in8
  1.373 / 1.080; gu32/in96 + dn32/in32 1.363 / 1.064; gu64/in48 + dn64/in16 1.362 / 1.072; 110-core grid ILLEGAL ("Illegal kernel
  placement ... not on_dispatch_core": the 11x10 grid includes dispatch cores; 64 = 8x8 is the largest clean grid here). PCC identical
  for all (same math). SELECTED gather: gate/up 64 cores in0_block_w<=32, down 64 cores in0_block_w<=32.
- (host 04:25) masked (prefill) sparse-matmul geometry (sweeps/masked.jsonl; group width x gate/up cores/in0 x down cores; prefill T=128 wall):
  c32 gu32/in64 dn32 21.22 / 19.17 ms (stage-2 default); c32 gu64 21.74 / 20.15; c32 gu16/in96 25.08 / 23.46; c64 gu32/in64 dn32
  17.88 / 15.77 (best, -16% / -18%); c64 gu64/in96 dn64 19.23 / 17.41; c128 gu64/in96 dn64 FAILS ("Statically allocated circular
  buffers ... clash": M=128 rows x in0 96 tiles x 2N block does not fit L1). Decode unchanged (masked path is prefill-only at B<=4).
  Follow-up family masked2: c64 with in0 96 / narrower down blocks, c128 with smaller blocks (gu32/in32, gu16/in64).
- (host 04:25) decode norm layout (sweeps/norm.jsonl): interleaved one-core RMSNorm (stage-2) 1.369 / 1.074 ms; sharded RMSNorm program on
  a width-sharded L1 copy of the residual (`args.decode_norm_sharded`, tt_transformers `get_norm_config("attn", DECODE)` grid) 1.306 /
  1.002 ms (-4.6% / -6.7%), PCC unchanged (0.99958 / 0.99983). SELECTED decode_norm_sharded=True. (Residual stays L1-interleaved at the
  token-mixer boundaries: the qwen36 single-device modules consume interleaved inputs; the (1,4) TP path has its own DistributedNorm.)
- (host 04:27) paged SDPA-decode program config (sweeps/sdpa.jsonl, attention layer, context 128): q/k chunk 32/64 (kernel default)
  1.074 ms, 32/128 1.075, 32/256 1.078, 0/0 (op auto) 1.077 — no signal at 128 keys (SDPA row ~25 us). Re-measured at 16384 context
  in the sdpa_long family before deciding.
- (host 04:30) family masked2 (chunk 64 in0 variants, chunk 128 with smaller blocks), real weights:
  c64 gu32/in96 dn32/in32 18.32 / 15.92 ms; c64 gu32/in32 dn32/in16 18.05 / 16.26 ms; c64 gu16/in96 dn16/in32 21.61 / 21.17 ms
  (GDN / attention prefill T=128). None beats the selected c64 gu32/in64 dn32/in32 (17.88 / 15.77 ms). Every chunk-128
  candidate failed before the first matmul ("Operand to matmul must be on device", both kinds): at chunk 128 the T=128 sweep input
  is one chunk and the masked path's mask/nnz builder produced a host tensor; chunk 128 is rejected for this stage (chunk 64 selected),
  the builder bug is noted for the multichip stage where chunk widths are re-derived per device.
- (host 04:31) family sdpa_long (attention layer, decode at 16384 context after a chunked fill; PCC still at T=128):
  default (32/64) 1.100 ms; k_chunk 128 1.099; k_chunk 256 1.111; k_chunk 512 fails (static CB region overflow);
  KV bfp8 + default chunks 1.080 (-1.8%); KV bfp8 + k_chunk 256 1.063 (-3.4%). Decision: SDPA decode chunks stay at the kernel
  default (no signal above noise, 512 is invalid); `kv_dtype` stays bf16 for this stage. KV bfp8 is recorded as the proven,
  PCC-neutral (at 128) context/concurrency lever for the datatype sweep (stage 8), where the full-model teacher-forcing metric
  decides; the stage-3 long-parity test is parametrized by the selected KV dtype so a later switch is covered.
- SELECTED after round 1 (superseded by round 2 below): routed bfp4/LoFi, shared bfp8/HiFi2, attention + GDN weights bfp8, KV bf16, gather 64 cores /
  in0 32 (gate-up and down), masked chunk 64 with gate-up 32 cores / in0 64 and down 32 cores / in0 32, gather up to 4 tokens,
  sharded decode norms, SDPA defaults, all fusion flags on.
- (host 04:40) tests/test_optimized_decoder.py first run (logs/03-opt-suite.log): 18/20 PASS; both `test_opt_batched_decode_against_hf`
  cases failed in the sharded decode norm ("Shard height 32 must match physical height 1024"): the single-device batched decode
  layout is [1,B,1,dim] (one padded tile per user, physical height 32*B) and the 32-row shard spec rejects it. Fix in
  `qwen36/tt/layer.py::_decode_norm`: shard only when the padded activation is exactly one 32-row tile, otherwise the interleaved
  norm (batch-1 is the optimized target; batched decode keeps the stage-2 path). First fix version sliced a ttnn Shape (TypeError,
  seen in the perf chain) -> `list(x.padded_shape)`. Rerun (logs/03-opt-b32.log): 2/2 PASS -> suite 20/20.
- (host 04:49) long HF parity with the SELECTED policy (logs/03-opt-long-{attn,gdn}.log): attention 16421 tokens prefill 0.99998
  (last 2048: 0.99999), decode after it 0.99999, K/V windows pass; GDN 65573 tokens prefill 0.99741 (last 2048: 0.99732), decode
  0.99790 — identical to the stage-1/2 values (routed bfp4 and the other round-1 knobs do not move long-context parity).
- (host 04:45) Round 2 from the stage-2 decode tables (per replay): router gate matmul `32x3072x256` 70 us on 8 cores (SLOW, dram 4%);
  packed shared-expert matmul `32x3072x2080` 56 us on 65 cores (SLOW, dram 23%) + shared down `32x1024x3072` 25 us; dense mixer
  projections: attention qkv `32x3072x17408` 153 us (109 cores, dram 68%, in0_block_w 2), o_proj `32x8192x3072` 182 us (96 cores,
  SLOW, dram 28%, in0_block_w 2); GDN mega in-proj `32x3072x20608` 190 us (108 cores, dram 66%), out-proj 183 us (SLOW). New knobs in
  `OptimizationConfig`: `router_decode_pc` (ttnn | derived 2D-mcast from TTMoEGate | 1D 8 cores; decode rows only, prefill keeps
  ttnn's choice), `shared_decode_cores` (1D mcast_in0 small-grid configs via qwen36 `tp_common.create_matmul_1d_decode_progcfg`),
  `dense_decode_mode` = default | 1d (cores_in/cores_out, optional in0_block_w override) | dram_sharded (DRAM width-sharded weights,
  DRAM-sharded kernel, activation width-sharded in L1 on the common-divisor grid, `num_workers_per_dram_bank` 1..3). Shared-code
  plumbing: `decode_matmul_fn` hook (x, weight, role, ckc, mc) in `ttnn_gated_attention.py` (roles qkv/q/o) and
  `ttnn_gated_deltanet.py` (mega/o), `AttentionConfig`/`GDNConfig` fields `decode_matmul_fn` + `weight_memcfgs`, loaders place the
  projection weights per role (cache tag `.ds`).
- CORRECTION to the 04:20 GDN-dtype entry: the GDN decode in-projection is the kernel's `mega_fused_weight` (qkv|a|b|z in one matmul),
  which `gdn/weights.py` built from device copies with a hardcoded bfloat8_b and no cache file; `gdn_weight_dtype=bfp4` therefore
  never reached the decode matmul, so "no time gain" was not a valid decode measurement. The mega weight is now built from the
  checkpoint tensors with the configured dtype, cached (`linear_attn.mega_fused.weight`) and placed per `weight_memcfgs["mega"]`;
  family `gdn_dtype2` re-measures bfp4 on the real decode matmul (PCC decides: the earlier prefill PCC drop 0.99882 -> 0.99528 came
  from the non-mega prefill weights and stands).
- (host 04:50) family router (attention layer): ttnn auto 0.987 ms; TTMoEGate derived 2D-mcast config (in0_block_w 32, 8 N-cores)
  0.924 ms (-63 us); 1D 8 cores 0.930 ms. PCC identical. SELECTED router_decode_pc = "derived".
- (host 04:55) round-2 families (attention layer unless noted; base = round-1 SELECTED, router ttnn): shared_pc — packed shared +
  down matmuls with 1D mcast_in0 configs: none 0.987 ms, 16 cores 0.942, 32 0.938, 44 0.937, 64 0.937, 88 0.938 (-50 us, PCC
  identical). SELECTED shared_decode_cores = 44. dense_1d — attention qkv/o and GDN mega/out with 1D configs (cores_in/cores_out):
  GDN 1.306 -> in64/out32 1.140, in88/out32 1.143, in44/out32 1.147, in64/out48 1.135, in64/out96 1.137, in88/out88 1.141;
  attention 0.987 -> 0.888 / 0.889 / 0.897 / 0.886 / 0.882 / 0.882; in0_block_w 16 / 32 (in64/out32) 1.147 / 1.149 and 0.892 / 0.897
  — no better than the helper's 8. PCC identical. SELECTED dense_decode_mode = 1d, cores_in 64, cores_out 96 (-170 us GDN, -105 us
  attention vs round 1). dense_ds — DRAM width-sharded weights + DRAM-sharded decode kernel (readers 1/2/3): every candidate failed
  in PREFILL, before any decode measurement: the auto-selected MultiCore fallback matmul asserts "Input B memory layout must be
  INTERLEAVED" (matmul_device_operation.cpp:1291). Adapted attempt: the hook now also serves prefill rows on a sharded weight with
  the explicit 2D-mcast prefill config (`tp_common.create_prefill_matmul_program_config`); re-run queued. gdn_dtype2 (GDN layer,
  mega weight now really bfp4 in decode): bfp8 1.284 ms -> bfp4 1.225 (-59 us) but PCC prefill 0.99882 -> 0.99527 (at the 0.995 bar)
  and decode 0.99958 -> 0.99773. REJECTED for this stage (largest correctness move of any knob; the full-model top-k metric in the
  datatype sweep is the right arbiter). Attention-layer decode with router derived + shared 44 + dense 1d: measured next in the
  final before/after collection.
- (host 05:00) GDN decode structural rewrite (from the op-to-code audit of the stage-2 GDN decode replay: ~426 us of structural
  ops next to 373 us of large matmuls). Changes, all in the shared kernels so the TP path (which imports the same recurrent step)
  inherits them: (1) conv step — the autoport now allocates per-tap [B,1,D] shift-register buffers (`gdn_state["split"]`, refreshed
  from the fused window after every prefill with `restore_split_conv_from_fused`), so decode takes the split path (three `ttnn.copy`
  instead of untilize/slice/tilize/concat rows), and the tap sum uses `ttnn.mac` (K-1 fused multiply-adds instead of 2(K-1) ops);
  (2) recurrent step (`recurrent_gated_delta_rule_decode_ttnn`): head-row tensors stay tile-padded [B,H,1,*] (no padded->dense
  reshapes of v/k/v_read), the decay is applied to the read row instead of the whole state, beta is folded into the [B,H,1,V]
  operand, the outer product is a column x row broadcast multiply (`transpose(k_row)` x `d_row`) instead of an inner-dim-1 matmul,
  and the update is ONE `ttnn.mac(h, exp(g), outer)` written straight into the persistent state buffer (`state_out`) — 2 state-sized
  passes instead of 6 (decay-mul, matmul write, beta-mul, add, 2 staging copies); the L1 budget now uses the real element size;
  (3) GVA q/k head repeat as one exact one-hot expansion matmul (`gva_expand`, HiFi4 fp32-acc) instead of the composite
  repeat_interleave (B==1 decode); (4) the gated output norm runs on the [B,H,1,V] head rows (gate rows via a cheap dense->padded
  reshape) with one dense reshape before the out-projection. Algebra unchanged; rounding equal or better (one rounding of
  h*decay+outer instead of two). Validation queued (03-chain3): autoport GDN tests, 9B unit suite (GDN decode/prefill), 9B decode
  check, then the adapted dense_ds attempt. New knob `gdn_state_fp32` (fp32 external state + fp32 step, as the qwen36 TP path
  does by default) with sweep family `gdn_state` to decide by PCC/time.
- (host 05:05) dense_ds adapted attempt (DRAM width-sharded weights, 2D-mcast prefill config from `tp_common`): r1 fails in PREFILL
  with "Statically allocated circular buffers on core range [0-0 - 7-9] grow to 1701632 B beyond max L1 1572864 B" (the 2D config is
  sized for the TP=4 per-device widths; at the single-chip N=20608 / K=3072 it overflows L1); r2 fails in the DRAM-sharded kernel:
  "weight shard width 81 tiles per bank must be divisible by workers_per_bank 2" (20736/8 banks = 81 tiles, odd); the process then
  segfaulted in `zeros_like` after the device errors. BLOCKED on one chip with these exact errors; the 1D configs (-170/-105 us) are
  kept. Note for stage 4/5: the qwen36 TP modules place exactly these roles DRAM-sharded at N/4 widths (measured by that team on
  this hardware), so the DRAM-sharded question is re-decided there with legal widths.
- (host 05:08) GDN rewrite validation: prefill/decode T=5/129/333 and real T=37/300 PASS (decode PCC 0.99948/0.99959 unchanged);
  B=32 first OOM'd ("Not enough space to allocate 67108864 B L1 buffer"): the state-temporary budget used 4 bytes per element for a
  bf16 state, so 64 MiB was sent to L1 -> budget now counts real bytes and is 24 MiB (B=32 -> DRAM, PASS 0.99911). The traced test
  then segfaulted in `zeros_like` at `reset_gdn_state`: `ttnn.mac(..., output_tensor=state_buffer)` returns a NEW Python tensor
  aliasing the persistent buffer, the in-place wrapper saw `new_h is not state_buffer`, copied and DEALLOCATED it -> dangling state.
  Fixed (the wrapper returns the caller's buffer object). (Review #1 corrected an earlier claim here: the kernel's mega
  in-projection branch already had a split-conv branch; the enabler was the autoport allocating the split buffers. A duplicate
  branch added by mistake was removed.) Re-validation running (03-chain5).
- (host 05:10) MoE PREFILL topology (TTFT): the masked path computes every 64-token group against the union of its experts
  (~221 of 256), ~27x the useful expert work, and the swiglu runs over the dense [E, 64, 2N] intermediate (2.2 ms of 27 ms at
  T=128). Per-token prefill cost is therefore flat (~0.2 ms/token/layer: 27 ms at T=128, 411 ms at T=2048 in stage 1). New
  `prefill_mode="sorted"` (tt/moe_dispatch.py): sort the (token, expert) pairs by expert on device, pad every expert's block to
  32 rows, gather the rows with `ttnn.embedding`, run ONE compact `sparse_matmul` per projection (every 32-row group = one expert,
  `nnz` = groups, static upper-bound shapes), swiglu on the compact intermediate, weight, and un-dispatch with k row gathers + adds.
  Compute per token = k experts + padding (<= 31 rows per expert) instead of the union. Correctness check tool
  (tools/check_sorted_dispatch.py) and the prefill_mode sweep at T=128 / 2048 queued (03-chain6).
- (host 05:15) GDN rewrite validated: all 10 GDN cases PASS (B=32 0.99911, traced/determinism, chunked 2048+2048+333, audit,
  stress). Family gdn_state (GDN layer, all round-2 selections + rewrite): bf16 state 0.884 ms decode (PCC 0.99882 / 0.99960 —
  decode PCC slightly UP from 0.99958 with the single-rounding update); fp32 state + fp32 step 0.941 ms (+57 us), PCC 0.99882 /
  0.99957 (no gain at 128 context). Kept bf16 (`gdn_state_fp32=False`); the fp32 candidate is recorded for the full-model
  long-context checks (stage 6/8) where a state-precision effect would show. GDN layer decode so far: stage 1 1.610 -> stage 2
  1.464 -> round 1 1.306 -> round 2 + rewrite 0.884 ms (-45% vs stage 2).
- (host 05:20) sorted dispatch bring-up (tools/check_sorted_dispatch.py, torch reference of every index tensor): (1) `ttnn.scatter`
  rejects fp32 TILE operands and int32 TILE rows > 256 -> scatters moved to ROW_MAJOR uint32 / bf16; (2) `sparse_matmul` refuses
  `output_tile` together with `optional_output_tensor` -> dropped; (3) the fp32 one-hot selection matmul of the per-expert start /
  offset table was inexact above ~8 mantissa bits (max abs error 31 at values ~24000; `ttnn.sort`, the counts matmul and
  `ttnn.cumsum` were exact) -> the table is split into high/low bytes and recombined (exact). Check: T=5/37/128/333/2048 all PASS
  (destinations distinct, source token / weight / group expert exact, padding rows weight 0, gather+un-dispatch round trip
  relative error ~1e-2 = bf16 products). Index build wall 23-60 ms per call (includes a device sync).
  First timing (T=128, before fix 3; index math correct at that length): sorted 46.8 / 38.5 ms vs masked 18.1 / 15.8 ms (GDN /
  attention): at 128 tokens the static upper bound (Rd = 8992 rows for 1024 real pairs, 281 groups of which 32 carry data) and
  the ~60-op index build dominate -> the sorted path needs a token threshold; T=2048 comparison running.
- (host 05:22) prefill_mode at T=2048 (real weights, PCC vs HF identical to 1e-8): masked 244.6 / 259.0 ms -> sorted 164.5 /
  135.4 ms (GDN / attention layer, whole-layer warmed wall; -33% / -48%). At T=128 sorted loses (41.5 vs 18.1 ms). sorted_geom at
  2048 (attention layer): gu64/in32 136.6, gu64/in96 134.9, gu32/in64 133.6, gu64/in8 135.9 ms — within noise; 96 cores illegal
  (dispatch cores). SELECTED prefill_mode = "sorted" with `sorted_min_tokens` from the 512/1024 crossover runs; geometry kept at
  64 cores / in0 32 (same as the decode gather configs). The sorted layer is still far from the ~20x compute reduction: the index
  build alone is ~40 ms per call at 2048 tokens (measured with a sync in tools/check_sorted_dispatch.py) — profiled in the final
  Tracy prefill_2048 collection; follow-ups noted in the README.
- (host 05:25) crossover (attention layer, warmed wall, masked -> sorted): T=256 30.7 -> 43.7 ms (masked wins), T=512 62.2 -> 54.4,
  T=1024 126.4 -> 80.6, T=2048 259.0 -> 135.4. SELECTED sorted_min_tokens = 512. Tracy of the index build alone at 2048
  (tracy_dispatch/dispatch_2048_perf_report.csv, signposted, warmed): 73 ops, 10.7 ms device — Sort 3.56 ms (16384 fp32 keys),
  Gather 2.21 ms, four 1-core Scatters 0.62-0.89 ms each (3.1 ms), counts matmul 0.22, two cumsums 0.28, rest < 0.1 ms each.
  Tracy of the whole optimized attention layer at 2048 in 4x512 chunks (tracy_dispatch/layer_prefill_2048_perf_report.csv):
  972 ops, 91 ms device, of which sparse gate-up 28.1 ms + down 15.4 ms for 4 x 377 (group, expert) pairs = 18.7 us per
  [32 x 3072] x [3072 x 2048] bfp4 pair — the kernel streams the expert's 3.5 MB to the grid for every 32-row group, so the pair
  count (not the token count) sets the cost, and at 512-token chunks 60% of the 377 groups are static padding. Reshape 11 ms
  (108 views on the dispatched rows), embedding gathers 5.7 ms, scatters 3.5 ms, SDPA 3.5 ms, dense projections 5.4 ms.
  Under the profiler the host gap is 326 ms (profiler-inflated; the unprofiled wall for one 2048 chunk is 135 ms). Follow-up
  knob `sorted_group_rows` (32/64/128: larger expert blocks -> fewer weight re-reads, more padding) sweeping at 2048 / 512.
- (host 05:30) sorted_group_rows at 2048 (attention layer): 32 rows 134.6 ms, 64 rows 161.9, 128 rows 220.2 — larger expert blocks
  lose (the per-pair cost grows with M; padding grows too). Kept 32. Index-build constants (ones row, positions, zero templates)
  are now cached per prefill length (no host tensor creation per call). Runtime fallback audit extended with a 640-token
  sorted-dispatch prefill. Final evidence chain started (03-final-chain.sh): full suite, Tracy fused vs optimized (prefill 128,
  prefill 2048 in 4x512 chunks, traced decode), watcher, long parity.
- (host 05:50) final suite 23/23 PASS (logs/03-final-suite.log): every PCC equals the stage-1/2 value within 1e-4; sorted vs masked
  prefill 0.99999999 (T=37/1023 GDN, 333 attention). Final Tracy (doc/optimized_decoder/tracy, PERF_TABLES.md): device time fused ->
  optimized: GDN prefill128 26.75 -> 16.15 ms, prefill2048 400.6 -> 125.2 ms, decode 1.288 -> 0.840 ms; attention 27.77 -> 15.09,
  472.2 -> 91.4, 1.128 -> 0.700 ms. The fused GDN baseline (1.288) is below the committed stage-2 value (1.464) because the shared
  recurrent-step rewrite applies to the fused graph too. Roofline: 152 / 138 MB per token -> 0.297 / 0.269 ms = 35% / 38% of
  roofline reached. Watcher and long parity running.
- (host 05:48) final chain complete: watcher 6/6 PASS (23,047 lines, 0 bad; incl. the sorted-dispatch chunked prefill and GDN traced
  decode), long parity attention 16421 0.99998 / decode 0.99999, GDN 65573 0.99741 / decode 0.99790 (same as stages 1/2).
  Stage-review #1 launched on the working tree.
- (host 06:00) stage-review #1 (working tree): more-work-needed. P1: DRAM-sharded decode matmuls never reached a decode
  measurement (all dense_ds rows failed in prefill / harness) and the README deferred them to stage 4/5 -> the hook now runs
  prefill on a one-time interleaved copy of the sharded weight (measurement path) and picks a legal reader count per role
  (bank-shard tiles divisible by the reader count), dense_ds re-measured with 1/2/3 readers on both layers. P2: sorted dispatch
  admitted T up to 8191 while the index math is exact only to Rd <= 65536 (T <= 7196) -> hard bound in moe_dispatch
  (MAX_DISPATCH_ROWS) + `sorted_max_tokens` = 4096 cap, dispatch check extended to 4096. P2: decode advanced only the per-tap
  conv buffers, so a chunk continuation after decode read a stale fused window -> the continuation rebuilds the fused window from
  the taps; new test `test_opt_decode_then_chunk_continuation` (prefill 128, 128 decode steps, 100-token continuation vs HF,
  both kinds). P3s: dead duplicate split branch removed, constant cache bounded to 8 lengths, budget change flagged for the 9B
  owners (fp32 state, B > 12 -> DRAM temporary), fidelity note, .refpt files excluded from the commit.
- (host 06:10) DRAM-sharded decode matmuls MEASURED (dense_ds re-run with prefill on a one-time interleaved copy of the sharded
  weight, plus per-role family dense_ds2; whole-layer traced decode, PCC unchanged): attention qkv+o sharded 1 reader 0.731 /
  0.735 ms vs 1d 0.777 (-42..-46 us; 2 readers 0.834, 3 readers 0.769); attention o only 0.760; GDN mega+o: 1 and 2 readers fail
  in the kernel (81 tiles/bank -> in1 block CB 1.70 MB > L1; 2 readers illegal by divisibility), 3 readers 0.944 vs 1d 0.884;
  GDN o only 1 reader 0.869 (-15 us); GDN mega alone 3 readers 0.928. SELECTED dense_decode_mode = dram_sharded with roles
  (qkv, o) (mega stays 1d), 1 reader. Cost recorded: the single-device prefill path runs these projections on an interleaved copy
  (+84 MB per attention layer, +27 MB per GDN layer; ~2 GB on one chip for 48 layers) because the 2D prefill config overflows L1
  at the full-chip widths; the (1,4) TP modules already hold DRAM-sharded weights with fitting prefill configs, so the copy does
  not carry into the production path. Sorted dispatch check extended to 4096 tokens: PASS. Decode -> chunk-continuation test:
  GDN 0.99808 / attention 0.99993 vs HF (128 prefill + 128 decode steps + 100-token continuation). GDN traced-test prefill
  determinism failed once (bit-level) in the review-fix run; reruns queued with a diff printout. Final evidence chain re-queued
  with the final policy.
- (host 06:11) GDN prefill bit-determinism: the review-fix run failed `torch.equal(p1, p2)` once (T=40, values equal to the
  printed 4 digits); two standalone reruns PASS (0 differing elements), as did every earlier suite run (stage 1, 2, 3 x3).
  Recorded as a 1-in-7 anomaly of the single-device GDN prefill; the final-chain suite re-runs it. Not root-caused; the candidate
  suspects are the masked sparse matmul with runtime `nnz` inference and `fast_reduce_nc` (data-independent code paths otherwise).
- (host 06:27) final evidence chain #2 (final policy): suite 24/25 — the one failure is `test_opt_stress_repeated_runs[attn]`
  (bit-equality of two identical T=37 cycles; the 20 cycles were all finite; passed in chain #1 and in every earlier run);
  Tracy fused -> optimized: GDN decode 1.288 -> 0.828 ms, attention 1.125 -> 0.659 ms, prefill 2048 400.9 -> 125.2 / 471.9 -> 91.3 ms,
  prefill 128 26.77 -> 16.19 / 27.75 -> 15.05 ms; watcher 6/6 PASS, 19,377 lines, 0 bad; long parity unchanged (attention 16421
  0.99998, GDN 65573 0.99741). Bit-determinism probe (tools/check_determinism.py: 8 identical prefill+decode runs per policy
  variant, both kinds) running to localize the two flakes seen today.
- (host 06:35) bit-determinism root cause (tools/check_determinism.py, identical T=37 prefill + decode repeated, real weights,
  attention layer): decode is bit-identical in every variant (0 differing of 29 runs each); prefill differs at bf16-ulp level
  (max 7.8e-3, up to ~11k elements) in 5/29 runs with the SELECTED policy, 3/29 with the sorted path forced, 1/29 for the stage-2
  fused graph, and 0/29 with the functional router (softmax + topk) in the otherwise identical optimized graph; GDN layer 0/35.
  The common factor of every flaky variant is the TTMoEGate op with more than 32 rows (it iterates over row blocks with sliced
  persistent buffers); its single-tile decode output is deterministic. Fix: `Qwen35MoEFused._route` uses the gate op for the
  single-tile decode rows only and the functional router for prefill rows (router weight kept, +1.5 MB/layer; prefill cost of the
  functional router is ~1% of the layer). Candidate tt-metal issue: TTMoEGate multi-row-block nondeterminism (repro: this tool
  with `DET_VARIANTS=focus`). Post-fix: 30-run probe both kinds, full suite and Tracy re-collection running (03-chain18).
- (host 06:45) post-fix evidence: determinism probe 0/29 differing runs (both kinds, prefill and decode); full suite 25/25;
  Tracy re-collected with the final policy (PERF_TABLES.md): GDN decode 1.287 -> 0.826 ms, attention 1.121 -> 0.660 ms,
  prefill 2048 399.3 -> 124.1 / 471.4 -> 90.1 ms, prefill 128 26.82 -> 16.16 / 27.66 -> 15.12 ms (functional router in prefill:
  -15 ops per kind, no measurable time change). Stage-review #2 launched.
- (host 06:55) stage-review #2: review-#1 findings verified closed, numbers reconcile; more-work-needed with two P2s:
  (a) KV cache bfp8 is faster at 16K context (1.100 -> 1.063 ms with k_chunk 256) and PCC-neutral on real weights, so
  keeping bf16 "for the datatype sweep" is a deferral (OPT-002) -> family kv_final at 16384 context, then select and re-validate
  (K/V content windows, 16421 parity, context contract capacity); (b) the GDN mega in-projection was rejected for the DRAM-sharded
  kernel on the first L1 error -> adapted candidate `mega_split_ds`: DS on the qkv columns [3072 x 12288] (48 tiles/bank) + 1D on
  a|b|z, concatenated (family mega_split, readers 1/2/3). P3s: watcher must be re-run on the final code (the last run predates the
  router fix), B=32 decode determinism probe (full 32-row tile through the gate op), a warm 4096-token prefill point for the sorted
  cap, a 65K GDN parity row with the bfp4 mega weight to close that candidate with evidence, DS activation grid 16/8 cores
  (family ds_cores), the norm->DS reshard pair (~10 us, documented), the T=40 GDN flake note in README section 3.
- (host 07:02) review-#2 families: mega_split (GDN, DS on the qkv columns + 1D on a|b|z + concat): 1 reader 0.880 ms, 2 readers
  0.903, 3 readers 0.895 vs the current 0.869 (1D mega + DS out-proj) -> REJECTED with measurements (the extra concat and the
  narrower DS matmul do not pay back). ds_cores (attention DS activation grid): 32 cores 0.732 ms; 16 and 8 cores fail
  ("Statically allocated circular buffers ... beyond max L1": wider K shards per core grow the in0/in1 blocks) -> 32 kept.
  kv_final (attention, decode at 16384 context, real weights): bf16 0.846 ms; bfp8 k_chunk 64 0.824; bfp8 k_chunk 256 0.808
  (-4.5%), PCC prefill/decode 0.99962 / 0.99985 vs 0.99959 / 0.99983 for bf16. SELECTED kv_dtype = bfloat8_b with SDPA decode
  chunks 32/256 (fill cast lives in the attention kernel). Re-validation (K/V content windows, 16421 parity, suite, Tracy, watcher
  on the final code) queued; doc/context_contract.json KV capacity doubles.
- (host 07:04) GDN bfp4 in-projection weights, 65573-token HF parity (test variant gdn_65573_bfp4_mega): last-2048 PCC 0.99387
  < 0.995 (bfp8: 0.99732) -> the candidate is rejected on long-context correctness, not by prose; the variant stays in the suite
  as a strict xfail so the evidence re-runs with the stage. Warm 4096-token prefill (attention layer): masked 517.6 ms vs sorted
  244.7 ms -> the sorted cap at 4096 is supported by measurement.
- (host 07:05) B=32 determinism probe (12 identical batched decodes, 32 users = one full tile through the gate op): attention
  decode 4/11 runs differ (bf16 ulp), prefill 0/11 -> the gate op is nondeterministic whenever several valid rows are present,
  not only above 32 rows. `_route` now uses the gate op for the single-user decode row only (x4.shape[2] == 1); batched decode
  and prefill use the functional router (batched decode is not the primary latency target; cost ~+100 us at B=32). B=32 probe
  re-queued after the final chain.
- (host 07:21) final chain #3 (bfp8 KV + SDPA 32/256, DS weights, gate op single-user): suite 25/25 (K/V content windows pass
  with the bfp8 cache), Tracy (PERF_TABLES.md) GDN decode 1.291 -> 0.828 ms, attention 1.125 -> 0.659 ms, prefill 2048
  399.0 -> 124.2 / 471.3 -> 88.9 ms; watcher on the final code 6/6, 20,562 lines, 0 bad; long parity attention 16421 0.99999 /
  decode 0.99999 with the bfp8 cache, GDN unchanged 0.99741; the bfp4-mega parity variant xfails as intended (0.99387).
  The suite in this chain imported the router before the single-user restriction landed (07:05), so chain 20 re-runs the suite
  and the B=32 probe on the final code.
- (host 07:29) final code: suite 25/25; B=32 determinism probe 0/11 differing runs for both kinds (prefill and decode).
  Stage-review #3 launched.
- (host 07:40) stage-review #3: more-work-needed. P2: the 65K-token parity runs use SYNTHETIC weights (`_opt` without real=True),
  so the bfp4 GDN in-projection rejection (0.99387) was not real-weight evidence while the real-weight T=128 row passes the bar
  -> real-weight long-parity variants added (attention 16421, GDN 65573 bfp8 and bfp4) and running; decision follows the real
  numbers. P3s: README section-3 attention rows came from an earlier suite log (deltas <= 1.4e-4) -> table regenerated from
  03-final-suite.log by script; norm->DS reshard pair not attempted (~13-17 us/attention replay) -> attempted below;
  T=1 prefill goes through the gate op / DS hook like a decode row (one valid row: deterministic) -> noted.
- (host 07:50) REAL-weight long parity (logs/03-long-real.log): attention 16421 0.99990 (all 0.99988), K/V windows 0.99983;
  GDN 65573 with the selected bfp8 in-projection 0.99552 (last 2048) / 0.99584 (all) / decode 0.99847; GDN 65573 with the bfp4
  in-projection 0.99304 (last 2048) < 0.995 -> the bfp4 candidate is rejected on REAL weights (both parity variants kept as
  strict xfails). Note: real-weight GDN long parity is lower than the synthetic 0.9974 (the state accumulates real activation
  statistics); the bar holds. norm->DS hand-off attempt (`attn_norm_out_sharded`: attention norm output stays width-sharded, the
  DS qkv hook adopts the norm's shard grid when it divides K and N): family norm_ds running.
- (host 07:45) family norm_ds (attention layer, whole-layer traced decode): norm output handed sharded to the DS qkv matmul
  0.725 ms vs 0.731 with the sharded->interleaved->sharded pair (-6 us, PCC identical). Small but repeatable in direction; SELECTED
  attn_norm_out_sharded = True. Final evidence chain #4 re-run with this policy (suite, Tracy, watcher, long parity).
- (host 08:02) final evidence chain #4 (final policy incl. attn_norm_out_sharded, final code): suite 25/25; Tracy (PERF_TABLES.md)
  GDN decode 1.288 -> 0.827 ms, attention 1.125 -> 0.656 ms (92 ops: the norm->DS reshard pair is gone), prefill 2048 398.9 -> 124.2 /
  471.4 -> 88.9 ms, prefill 128 26.74 -> 16.20 / 27.71 -> 15.14 ms; watcher 6/6, 19,354 lines, 0 bad; long parity 4 passed + 2
  strict xfails (real weights: attention 0.99990 / 0.99988 / decode 0.99968; GDN 0.99552 / 0.99584 / decode 0.99847; bfp4 mega
  0.99304 real, 0.99387 synthetic). Stage-review #4 launched on the staged tree.
- (host 08:12) stage-review #4: clean-pass (P3 doc nits applied: long-parity header 4 passed / 2 xfails, rounded real-weight
  digits, DS qkv row 129 us / 81%, B=32 probe timestamp wording, tracy_dispatch breakdown dated; guard added so
  `attn_norm_out_sharded` requires the DS hook and fused qkv; 9B B=4 check and the non-aligned GDN host selector recorded as
  follow-ups). Committing stage 3.
