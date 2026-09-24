# Fused decoder — work log (stage 2)

Baseline: stage-1 functional decoder (doc/functional_decoder/tracy). Device: one Blackhole die, 1x1 mesh.

## 2026-09-24
- 02:45 Op sequence tables written from the stage-1 perf CSVs (decode: 115 ops GDN layer / 84 ops attention layer per step, complete profiler windows;
  prefill 128: 305 / 132 ops). Candidates classified (graph-fusing skill priority order):
  1. Dedicated op: router chain (linear, softmax, fill_pad, topk, 2 slices, fill_pad, reduce, div, typecast, untilize = ~130 us,
     11 ops) -> `TTMoEGate` (`generalized_moe_gate` kernel, k=8, softmax-over-selected == softmax+renormalize exactly).
  2. Graph rewrite (shared-LHS): shared expert gate/up + sigmoid-gate scalar (3 matmuls, 44+44+68 us; the [3072x32] scalar
     matmul ran on ONE core) -> one packed [3072 x 2049] matmul + 3 slices, SiLU folded into the multiply.
  3. Graph rewrite (algebraic): masked expert groups apply routing weights to the [E,C,1024] intermediate before the linear
     down projection instead of the [E,C,3072] output (exact; 3x fewer multiply elements per chunk).
  4. Graph rewrite (shared-LHS): attention q(+gate)/k/v projections (144+39+39 us) -> one [3072 x 17408] matmul + slices
     (opt-in `AttentionConfig.fuse_qkv`, default off so the dense Qwen3.5-9B path is unchanged).
  Assessed and not applicable / deferred: SDPA, RoPE, concat-heads, RMSNorm, topk already dedicated ops in qwen36; the
  single-device GDN decode conv/recurrent chain (tilize/untilize/slice/reshape, ~40 ops) belongs to the qwen36 single-device
  module that the (1,4) TP stage replaces with `TPGatedDeltaNet`; not rewritten here (documented in README).
- 02:55 tt/fused_decoder.py: `FusedDecoder(FunctionalDecoder)` with `FusionFlags`, `Qwen35MoEFused`, `Qwen35MoeFusedDecoderLayer`.
  Shared code: `AttentionConfig.fuse_qkv`, `load_attention_weights(..., fuse_qkv)`, `gated_attention_forward_ttnn(qkv_proj_weight=)`.
- 03:00 first run: router gate op failed at T=333 (`num_blocks_y <= grid.y`): TTMoEGate derives its matmul program config for
  one M tile; fixed with `gate_matmul_auto_program_config=False` (ttnn picks per shape).
- 03:04 tests/test_fused_decoder.py: fused vs HF T=5/129/333 both kinds PASS (PCC within 1e-5 of the functional values);
  fused vs functional on identical weights/inputs: GDN 0.99999999 / 0.99999999, attention 0.9999994 / 0.9999989.
- 03:10 wall-clock before/after (tests -m perf, real weights, one die): attention layer prefill T=128 30.1 -> 29.2 ms,
  traced decode 1.488 -> 1.280 ms/step (-14%). GDN fused perf cases failed at T=128 with the seq-kernel CB clash
  ("L1 buffer allocated at 1355776 and static circular buffer region ends at 1360896"): the fused gate keeps small sharded
  L1 buffers resident, which pushes the wide (8192) GDN's L1-staged prefill activations into the kernel's CB region. Fix in
  `ttnn_gated_deltanet._seq_memory_config(seq_len, width)`: GDN widths > 4096 stage prefill activations in DRAM (9B width 4096
  unchanged). Rerunning the GDN perf cases.
- stage-1 checkpoint commit: 530cc27a48c
- 03:55 full fused suite (-m "not perf"): GDN 8/8 PASS incl. real weights (PCC identical to functional within 1e-6); attention cases
  first failed on a test-helper import removed by autoflake at the stage-1 commit (fixed, stage-1 commit amended), then 6/7 PASS and
  `test_fused_batched_decode_against_hf[attn_b32]` failed: `concatenate_heads` (NLPConcatHeads, one core) static dataflow buffers clash
  with L1 buffers at B=32 once the fused MoE keeps its small gate buffers L1-resident. Rewrite (op merging / structural, stage-2 pattern):
  at T==1 the head concat is a transpose of the size-1 dim + reshape (same permutation) instead of the one-core NLPConcatHeads
  (~105 us at B=1 in the stage-1 decode table). Applied in the shared kernel for T==1 only (prefill keeps concatenate_heads).
- (host 03:55) fused attention regression after the stage-1 page-table fix + decode head-concat rewrite: 8/8 PASS incl. batch 32 and real
  weights (logs/02-fix2-fused-attn.log). Fused decoder full suite state: GDN 8/8, attention 8/8 (-m "not perf").
- (host 04:02) Tracy before/after collection started (scripts/02-perf.sh -> doc/fused_decoder/tracy/<kind>_<functional|fused>/).
- (host 04:20) Tracy before/after (doc/fused_decoder/tracy/<kind>_<mode>/): attention prefill128 28.69 -> 27.73 ms device (132 -> 148 ops),
  attention traced decode (3 complete replays each) 1.340 -> 1.128 ms/replay (85 -> 89 ops; Matmul 654 -> 486 us from the fused QKV);
  GDN prefill128 27.64 -> 26.72 ms (306 -> 321 ops). Note: the "functional" attention decode baseline here (1.340 ms) is already faster
  than the stage-1 table (1.439 ms) because the T==1 head-concat rewrite lives in the shared kernel and applies to both graphs
  (NLPConcatHeads 105 us -> transpose+reshape ~10 us). GDN decode windows were first collected with 10 replays (profiler
  truncation) and are being re-collected with 3.
- (host 04:07-04:10) fused audit + stress (logs/02-audit-stress.log): `test_fused_runtime_fallback_audit` 0 host transfers in aligned prefill,
  continuation and decode (both kinds); `test_fused_stress_repeated_runs` 20 prefill+decode cycles over lengths 5..1500 (both kinds), all
  finite, identical cycles bit-identical: 4/4 PASS. Watcher (TT_METAL_WATCHER=10, real-weight T=37 both kinds + fused attention B=32):
  pytest exit 0, watcher.log 10,886 lines, 0 exception/fatal/sanitize/overflow (doc/fused_decoder/watcher/, artifacts/02-watcher/).
- GDN decode re-collected with 3 replays (complete sessions 118/118/118): functional 1.610 -> fused 1.464 ms/replay (-9.1%).
- (host 04:25) stage-review #1 (1a4e5f255eb): more-work-needed, two P2: (a) no dense-9B decode evidence for the shared T==1 head-concat
  rewrite; (b) fused graph never run at the 2048-token prefill chunk. Actions: `test_fused_chunked_prefill_matches_hf` (2048+2048+333,
  K/V cache windows, decode at 4429) both kinds; a 9B layer-3 paged-decode PCC check against HF (tools/check_qwen36_9b_decode.py) plus
  the 9B unit suite on the amended tree. Other concerns applied: sigmoid folded into the shared-expert multiply (input activation),
  the functional router weight is deallocated when the gate op is used, README records the router compute policy change (TTMoEGate:
  HiFi2 + fp32 acc, exp over raw selected logits) and rewords the K=8 fold rejection as "not attempted" (no in-repo repro), dead test
  code removed.
- (host 04:28) review-#1 fix chain (logs/02-fix-chain.log): `test_fused_chunked_prefill_matches_hf` both kinds PASS (attention
  chunked prefill vs HF 0.99997, all nine 512-position K/V windows >= 0.99986, decode at 4429 0.99999); 9B layer-3 paged decode check
  first failed on a transformers-5 class rename (`Qwen3_5RotaryEmbedding` -> `Qwen3_5TextRotaryEmbedding`), rerun PASS (B=1 0.99676,
  B=4 distinct positions 0.99661..0.99739); 9B unit suite 5/5 PASS. Amending the stage-2 commit with these files and re-running the
  stage review.
- (host 04:52) stage-review #2 on bbda338fe56: clean-pass. P3 notes applied in the stage-3 tree: layout-op counts in the README
  audit section corrected from the committed CSVs (incl. the autoport's own scores tilize), the "router matmul ~1%" comment
  corrected (70 us = ~5%), window completeness now cited from the committed `*_perf_report.csv`, 305 -> 306 ops.
