# Stage 6 — full model + readiness generator: work log

### 2026-09-24 14:05–14:50 — design, code, first smokes

- Design (from the qwen36 survey): `tt/model.py` `Qwen35MoeModel(Qwen36Model)` with `DECODER_LAYER_CLS =
  Qwen35MoeOptimizedDecoderLayer`, `_layer_state_dict` -> `StreamingStateDict.layer(i)` (the base `substate` would read
  the whole 250 GB checkpoint per layer), args built once with `MultichipDecoder._args_hook(MULTICHIP_SELECTED)`, one
  model-wide `TT_CCL` (the per-layer CCL of stage 4 goes away), qwen36 embeddings / final `DistributedNorm` / vocab-sharded
  bfp8 LM head / `SamplingGenerator` on-device sampling. Decode trace contract: persistent inputs (tokens uint32 [B,1],
  current_pos int32 [B], rot_idxs uint32 [1,B], page_table int32), RoPE cos/sin gathered ON DEVICE from resident tables
  (`ttnn.embedding`) and `ttnn.plus_one` on positions + rope indices inside the trace when the sampler consumes the logits
  (`_tt_vllm_always_refresh_decode_trace_inputs=False`, so the stock `Generator` binds `tt_out_tok` to the token input).
  `tt/generator.py` `Qwen35MoeGenerator(readiness Generator)` around the stock `models.tt_transformers.tt.generator.Generator`
  traced decode: prefill through the model's masked fixed-bucket path in 2048-token segments (`prefill_masked_bucket`,
  `chunk_start`), `prefill_all_logits` (padded to the 128-token GDN chunk, every position through norm + LM head) for the
  prefill check, `generate` greedy with `next_input` teacher forcing (host-authoritative token only when forced), host
  sampling as the explicit compatibility mode.
- Sampler choice: `models/common/sampling` `SamplingGenerator` (already integrated by qwen36: vocab-sharded logits, per-shard
  top-k candidates + all-gather, `tt_out_tok`, seeds, penalties, `set_trace_bucket`) over `Sampling1D` (single-device
  oriented, no vocab-sharded gather contract); recorded in README.
- Smoke 1 (layers [0,3], T=21): GDN prefill kernel refused an un-padded T (`chunk_gated_delta_rule.cpp:288 !flat_v || pad
  == 0`) -> `prefill_all_logits` pads to the 128-token chunk and masks with valid_len; `prefill_forward` uses the model's
  masked-bucket path.
- Runner mesh: the readiness CLIs open the mesh with the bare `ttnn.open_mesh_device(mesh_shape=...)` -> trace region 0 and
  small-L1 0 (`DEFAULT_TRACE_REGION_SIZE == 0`, `DEFAULT_L1_SMALL_SIZE == 0`), no env override in tt-metal. A generator-side
  close + re-open of the FABRIC_1D mesh HANGS in-process (smoke 3: 30 min at 0 % CPU after "re-opening the mesh"; kill,
  `tt-smi -r`). Resolution: `tools/readiness.py` opens the (1,4) mesh with `l1_small_size=24576` and a trace region and
  calls the runners' programmatic entry points (`run_prefill_check` / `run_teacher_forcing` / `run_autoregressive`)
  unchanged; `build_generator` fails fast with that instruction when handed a mesh without a trace region.
- Smoke 4 (wrapper mesh) spun 30 min at 100 % CPU inside `Qwen36Model.__init__:70` (framework `Embedding`). Isolated by
  `scripts/06-embd-probe.py`: `ttnn.get_memory_view` is harmless; `ttnn.as_tensor` WRITE of the 1.5 GB bf16 ROW_MAJOR
  replicated embedding with a cache file takes 0.9 s, its RELOAD from the cache file spins > 10 min (killed). Every TILE
  tensor (layers, LM head) reloads normally, so it is the ROW_MAJOR large-tensor reload path (ttnn backlog item). Fix:
  `_CacheRoot` path hook returns no cache file for `tok_embeddings*` (the embedding is converted from the streaming
  checkpoint each run, ~1 s); the pathological cache file was deleted.
- Smoke 5 spun again, now in `load_gdn_weights_tp` (`gdn/tp.py:116`, a TILE bfp8 `as_tensor` with a cache file). Raw probe
  (`scripts/06-reload-probe.py`): `ttnn.load_tensor` of the cached 67 MB `qkvzab.il_dtype_BFLOAT8_B_layout_TILE.tensorbin`
  (4-shard multi-device host tensor, flatbuffer format) does not return within 240 s on the host alone (no device involved).
  The only tensor cache on disk is the one the first smoke created at 14:12 — the stage-1..5 tests never read a cache file
  (they rebuilt each layer from the streaming shards in ~11 s), so the reload path was never exercised before. Decision:
  no ttnn tensor-file cache for this model by default (`create_model`, opt-in `QWEN35_TENSOR_CACHE=1`); 48 layers convert
  in ~9 min per process start (measured in the full-model run below); ttnn backlog item with the probe as repro.
- Smoke 7 (16:35, no tensor cache, layers [0,3]): build 24.6 s (two layers converted from the streaming shards), decode trace +
  sampler captured at position 0, `prefill_logits(34 tokens)` -> `[1, 34, 248320]` finite in 2.1 s. `generate` then failed
  reading the masked-bucket prefill logits: `pytensor.cpp:300 buffers.size() == 1` (a host multi-device tensor needs a mesh
  composer) -> `_host_logits` reads one replica through `ConcatMeshToTensor`. Re-queued.
- Smoke 8 (16:40): build 24.8 s, `prefill_logits` OK (0.2 s), `generate(8)` with on-device sampling = 8 tokens in 0.09 s
  (gibberish as expected from a 2-layer model). The SECOND `generate` hung on device inside the eager masked-bucket prefill
  (`prefill_masked_bucket` -> `ttnn.synchronize_device`, model.py:2175) after the traced decode had run; `tt-smi -r` needed
  a retry (first reset exit 1). Same class as the stage-5 traced-prefill hang (collectives from two contexts sharing the
  TT_CCL semaphore pools). Probe `scripts/06-prefill-after-decode-probe.py`: after a traced decode run + reset, run one
  of {eager all_gather, eager reduce_scatter, prefill attention_norm, full prefill} in its own process (reset after a hang).
- Prefill-after-traced-decode probes (`logs/06-pad-*.log`, 2-layer model, one process each): eager all_gather OK, eager
  reduce_scatter OK, prefill attention_norm OK, full prefill HANGS, GDN mixer prefill alone (`forward_prefill(capture_state=True)`
  -> fused fp32 `matmul_reduce_scatter_async` out-proj) HANGS. Together with the stage-5 trace bisect (mixer + MoE in one
  trace hangs, each alone replays) the fused MMRS is the common factor. The `tt-smi -r` after the mixer hang failed once
  (exit 1, "Read 0xffffffff over PCIe"), the MoE probe then ran on a dead board (invalid) — resets now retry 3x in the
  chains. Testing `gdn_prefill_mmrs=False` (`06-mmrs-off-chain.sh`).
- 17:55 `gdn_prefill_mmrs=False` does NOT fix prefill-after-traced-decode (still hangs), although it makes the stage-5
  `mixer+MoE` trace replay (the whole-layer trace still hangs — a second composition factor, stage-7 item). Re-reading the
  evidence: the generator captured the decode trace BEFORE any prefill had run; prefill A then allocated its persistent
  buffers (masked-bucket programs, expert-sorted dispatch constants, GDN conv carry, fused out-proj buffers) after the
  capture -> they land in the trace's freed scratch range -> generate A's decode replays overwrite them -> prefill B runs on
  corrupted index tensors and hangs (the mixer-alone probe hangs the same way; the eager single collectives, which allocate
  nothing persistent, pass). This is the ordering rule qwen36 documents (#48536): warm every prefill program/allocation
  before parking a trace. Fix: `Qwen35MoeGenerator._warm_prefill()` (`warmup_prefill_masked_buckets` over the 5 buckets,
  masked and full) runs before `_warm_decode_trace()`. Smoke 9 running.
- 18:01 Smoke 9 (`_warm_prefill` before `_warm_decode_trace`, 2-layer model): build 39 s, `prefill_logits` `[1,34,248320]`
  finite, `generate(8)` on-device sampling 0.08 s, the repeat `generate` is bit-identical (the hang is gone), teacher
  forcing 4 predictions with 4 `next_input` callbacks. Only the host-sampling compatibility mode failed: the stock
  `Generator.decode_forward` returns `(logits, log_probs)` on the host path -> unwrapped in `decode_forward`.
- 18:03 Smoke 10 started right after smoke 9's error exit without a reset and hung inside the prefill warm-up
  (`synchronize_device`, py-spy `FDMeshCommandQueue::finish`); killed, `tt-smi -r`, rerun (`06-smoke-then-05final2.sh`).
- 18:09 Smoke 10 after the reset: all steps pass (build 25.6 s; same tokens as smoke 9; deterministic repeat; teacher forcing
  4/4 callbacks). Host-sampling mode returns `[1146, 641, 10735, 3909]` while the on-device sampler returns
  `[1146, 8599, 43506, 3149]` for the same prompt: the first token (host argmax of the prefill logits in both modes) agrees,
  the first decode step disagrees. One of the two decode paths is wrong. Discriminator queued behind the stage-5 chain
  (`scripts/06-mode-probe.{py,sh}`): ground truth = last-position argmax of `prefill_all_logits(prompt + [pred0])`
  (the prefill path is the one validated against HF in stages 1-5), compared with the traced and eager decode step in
  both modes, in both capture orders.
- 19:18-19:21 Mode probe (`logs/06-mode-probe-{device_first,host_first}.log`, 2-layer model, prompt 24 tokens): truth for the
  step after `prompt + [1146]` (prefill path) = 8599. On-device sampler traced 8599 / eager 8599 (correct, both orders);
  host-logits mode eager 8599 (correct) but TRACED 641 (wrong, max |logit diff| 2.39 vs truth, both orders). Cause: the
  stock decode-trace capture runs the step twice on device (compile pass executes, capture records, first replay executes
  again), so a trace captured in the middle of a generation advances the GDN recurrent/conv state twice for that step.
  The on-device path never hits this because `_warm_decode_trace` captures on dummy inputs at build and resets the state;
  the host-logits variant was captured lazily at the first real host-mode step. Fix (`tt/generator.py`): the compatibility
  mode runs untraced unless its trace was captured at build (`warm_host_trace=True` / `QWEN35_WARM_HOST_TRACE=1`, dummy
  inputs + reset like the device variant). Verification queued (`06-host-mode-check.py`, default and warm builds).
- 19:21 Trace evidence run 1 (`logs/06-trace-evidence.log`, 2-layer model): token feedback (device token input == the token
  sampled at the previous step, no host write), positions and RoPE indices advance by 1 per replay on device, the trace
  input buffers keep their identity, an unchanged page table is not copied (`prev_page_table` object kept, device content
  equal), two identical generations are bit-identical and a different prompt differs — all PASS. The "changed page table"
  case was a no-op in the probe (it set the trailing block id to 63, which the arange page table already held) -> probe
  fixed to swap the two trailing unused block ids; rerun queued (`06-trace-evidence2.sh`). Token-out step on the 2-layer
  model: 5.13 ms per step (device work ~1.0 ms + sampler) -> ~4 ms of host-side per-step cost in the stock
  `Generator.decode_forward` path; recorded as the first target of the optimized-full-model stage.
- 19:30-19:55 First 48-layer run (`tools/readiness prefill-check`, AIME24 reference): 48 layers converted from the shards in
  7:11 (8.99 s/layer), generator init OK (KV pool (4096,1,64,256) bfp8 x 12 layers for 262144 x B=1), then the prefill
  warm-up HUNG on device in bucket 512 / actual_len 256 (the first sorted-dispatch bucket) — py-spy: host spinning in
  `tt::umd::memcpy_from_device` under `undispatch_rows` (`moe_dispatch.py:248`), 200 % CPU, no progress for 20 min; the
  same warm-up sequence passes on the 2-layer model (smokes 9/10) and every stage-4/5 T>=512 test runs the same path.
  Killed (`kill`, then the queued 2-layer runs + stage-5 review-2 runs continue). Added `QWEN35_WARM_LOG=1` (per bucket and
  per layer log with a device sync per layer) and queued `06-warm-repro.py` (48 layers, `warm_decode_trace=False`) to name
  the layer/bucket; candidate causes to test next: layer-dependent routing on the all-zero warm-up tokens (degenerate
  top-8 -> few groups) vs. an eager prefill-collective hang that only shows at 48 x 10 layer-prefills per warm-up.
- 19:52-19:54 Host-mode verification (`logs/06-host-mode-{default,warm}.log`, 2-layer model): the compatibility mode now
  returns the same 6 tokens as the on-device sampler ([1146, 8599, 43506, 3149, 79731, 95746]) in the default build (untraced
  host mode). The `warm` run did not exercise the warmed host trace (`build_generator` did not forward `warm_host_trace`;
  fixed: block_size / warm_decode_trace / warm_host_trace are forwarded) — rerun queued (`06-host-mode-warm2.sh`).
- 19:55 Trace evidence run 2 (`trace_evidence.log`): all checks pass including the real page-table change (device tail
  `[63, 62]` after the copy, token feedback intact). Token-out step 5.15 ms on the 2-layer model (host-path cost, stage 7).
- 20:08-20:38 Warm-up repro run 2 (`logs/06-warm-repro.log`, `QWEN35_WARM_LOG=1`, sync per layer): DRAM footprint before any
  prefill program = 2.543 GB/bank x 8 = 20.3 GB of 33.6 GB per device (weights + KV pool + GDN state), trace region 0
  allocated. Buckets 128 (64/128), 256 (128/256) and 512/256 completed for all 48 layers; the hang came in bucket 512 /
  actual_len 512 at layer 7 (the second full-attention layer) — run 1 hung in 512/256 (layer unknown), so the (bucket,
  layer) is NOT deterministic across runs: a timing/race class inside the T=512 prefill, not a static shape/nnz contract
  (`SortedDispatch.build` gives every group exactly one expert incl. padding groups -> expert 0, re-read). Host stack at the
  hang: `synchronize_device` after layer 7 (`FDMeshCommandQueue::finish`), idle. `tools/tt-triage.py --llm-output` captured
  to `triage/warm_hang_2_layer7_b512.txt` before the kill. Next: the same repro under the watcher (`06-warm-watcher.sh`,
  `artifacts/06-watcher-warm/`) to name the stuck kernel (fabric/CCL vs compute); candidates: the fused fp32
  `matmul_reduce_scatter_async` GDN prefill out-projection (a factor of the stage-5 traced-prefill hang) and the T=512
  sorted-dispatch MoE ops.
- 20:40 Host-mode `warm` rerun (`logs/06-host-mode-warm2.log`): with `warm_host_trace=True` forwarded, the host-logits decode
  trace is captured at build and the TRACED compatibility mode returns the same tokens as the on-device sampler
  ([1146, 8599, 43506, 3149, 79731, 95746]); the double-execution explanation is confirmed from both sides (untraced OK,
  build-time-captured trace OK, mid-generation-captured trace wrong).
- 20:49-20:59 Warm-up repro UNDER THE WATCHER (`artifacts/06-watcher-warm/`, TT_METAL_WATCHER=10): buckets 128, 256, 512 and
  1024 (masked and full) completed for ALL 48 layers without a hang — the sequence that hung twice without the watcher.
  The run then died at bucket 2048/1024, layer 0, with `TT_THROW: statically allocated circular buffers in program 1204
  clash with L1 buffers on core range [0-0 - 7-9]` — the watcher's L1 reservation plus the 2048-token GDN prefill program of
  the masked-bucket path (one 2048-token program, vs the 512-token chunks the stage-4/5 watcher prefill tests run); not
  the hang under study. Conclusion: the bucket-512 hang is timing-dependent (the watcher's device polling changes the
  timing and it does not reproduce) -> a race in a T>=512 prefill kernel/collective, not a shape or allocation contract.
  A/B running next: the same repro with `gdn_prefill_mmrs=False` (fused fp32 matmul_reduce_scatter_async off).
- 21:20 ROOT CAUSE of the 48-layer warm-up hang (both tt-triage captures, `triage/warm_hang_2_layer7_b512.txt` and
  `warm_hang_3_layer15_b512_mmrsoff.txt`): on ONE device (device 2 in hang 2, device 1 in hang 3) a 64-core program is
  stuck — `reader_cross_core_data_exchange` in `noc_semaphore_wait` inside `sort_barrier`, `writer_cross_core_data_exchange`
  and `sort_cross_core_data_exchange` in `cb_wait_front` — i.e. `ttnn.sort`'s multi-core cross-core-exchange kernel
  deadlocked; the other three devices sit in `ring_reduce_scatter_minimal_async` reader/writer (`noc_semaphore_wait_min` /
  `cb_wait_front`) waiting for that device, and the eth cores show mismatched NOC read counters. The sort is
  `SortedDispatch.build`'s `ttnn.sort(ids_flat, dim=-1)` on `[1,1,1,4096]` fp32 (the T=512 sorted-dispatch path); it is
  intermittent (hangs 2 and 3 in different layers, none under the watcher), the fused GDN out-projection is unrelated
  (`gdn_prefill_mmrs=False` hangs the same way). The stage-4/5 2-layer tests ran a few dozen T>=512 sorts; the 48-layer
  warm-up runs 48 x 6 of them per process.
- 21:25 Fix: `SortedDispatch.build` sorts the expert ids as UINT16 (`moe_dispatch.py`, `SORT_ON_UINT16`;
  `QWEN35_DISPATCH_SORT_FP32=1` restores the old path). `ttnn.sort` routes uint16 TILE rows wider than 32 tiles to
  `SortProgramFactorySingleRowMultiCore` (DRAM multi-core, no cross-core barrier); the fp32 row of width 4096..16384
  (Wt 128..512) took `SortProgramFactoryCrossCoreDataExchange`, whose `sort_barrier` (leader/follower semaphore
  handshake, `cross_core_data_exchange_common.hpp:175`) is the deadlocked kernel. The sorted values are cast back to
  fp32 for the rest of the build; the pipeline is otherwise unchanged. Evidence queued: op-level stress
  (`06-sort-stress.py`: 2000 sorts of [1,1,1,4096] uint16 then fp32, plus 500 x 16384 uint16), the 48-layer warm-up twice,
  then the readiness runs. ttnn backlog item: sort cross-core-exchange barrier deadlock (repro = the stress script, fp32).
- 21:26 The first uint16-sort version reshaped/padded the uint16 tensor (`reshape_view` accepts bf16/fp32/int32/uint32 only)
  -> the two prefill_2048 host-wall cases of the adopt chain failed at once (`logs/05-perf-plain2.log`, TT_FATAL). Fixed:
  reshape + pad on the fp32 copy, typecast the flat row to uint16 right before the sort. Queue re-linked: adopt chain
  (Tracy decode, watcher 7) -> host walls 3 + final suite 3 (`05-post-adopt.sh`) -> sort stress -> fixed warm-up x2 ->
  readiness x4 + autoregressive (`06-fixed-chain.sh`).
- 21:32-21:45 Host walls with the uint16 sort (`logs/05-perf-plain3.log`): decode 0.578 / 0.445 ms (the adopted persistent
  gathers) but prefill 2048 (4 x 512 chunks) 139.9 / 135.5 ms vs 122.7 / 117.9 — the DRAM multi-core sort factory costs
  ~4 ms per 512-token chunk (x48 layers = ~0.8 s per 2048 prompt tokens at the full stack): not acceptable as the fix.
  Replaced by a COUNTING SORT (`DISPATCH_MODE="counting"`, default): row i keeps its token-major position; destination =
  offsets[e_i] + rank_i, rank = cumsum of the one-hot down the rows (`ttnn.cumsum(dim=2)`, multi-core) picked through the
  one-hot with the values split into < 256 hi/lo parts (exact through the reduce datapath). No `ttnn.sort`, no gather of
  the scores, no perm scatter for the undispatch (d is already token-major). `sort_u16` / `sort_fp32` kept behind
  `QWEN35_DISPATCH_MODE` for A/B and the hang repro. Unit test `test_mc_dispatch_counting_matches_sort` (T = 512, 1000,
  2048; degenerate block): identical group masks, valid undispatch/weights/expert placement, same (expert, token)
  multiset as the fp32 sort build. Queue: suite 3 (uint16 code) -> sort stress -> host walls 4 + suite 4 (counting sort)
  -> fixed 48-layer warm-up x2 -> readiness.
- 21:50 Sort stress (`logs/06-sort-stress-*.log`, op alone on the (1,4) mesh, one sort per iteration): uint16 [1,1,1,4096]
  2000 sorts in 10.5 s (~5 ms each, DRAM multi-core factory) hang-free; fp32 [1,1,1,4096] 2000 sorts in 1.2 s (0.6 ms each,
  cross-core-exchange factory) ALSO hang-free in isolation. The deadlock therefore needs the model context (the sort
  dispatched while the mesh's other devices are inside/around the reduce-scatter and the fabric mux kernels on cores
  (0,0)/(3,0)/(6,0) are active — the triage shows `tt_fabric_mux_v2` on worker cores next to the stuck sort cores);
  it is not a property of the sort kernel alone. The counting-sort dispatch removes the op from the model regardless.
- 22:39-22:48 Fixed 48-layer warm-up run 1 (counting-sort dispatch, `logs/06-warm-fixed-1.log`): NO HANG — buckets 128, 256,
  512 and 1024 (masked and full) completed for all 48 layers (the sequence that hung 3/3 with `ttnn.sort`). It then failed
  at bucket 2048 / actual_len 1024, layer 0, in the GDN in-projection's unfused prefill matmul (`tp_common.
  sharded_decode_matmul` prefill branch, 2048 x 3072 x 5152 2D program): `TT_THROW: statically allocated circular buffers
  in program 1174 clash with L1 buffers on core range [0-0 - 7-9]; L1 buffer allocated at 1152768, static CB region ends
  at 1312256` — the same message the watcher run hit at the same bucket, so it is not the watcher's L1 reservation: the
  48-layer model holds 395,520 B of L1 per bank BEFORE any prefill program (MEMORY_REPORT; the 2-layer smoke has a
  fraction of that and runs the 2048 bucket), i.e. ~8 KB of persistent L1 per core per layer that the largest prefill
  program no longer fits beside. Chain stopped (`06-fixed-chain.sh`); `06-l1-probe.py` measures the per-layer L1 footprint
  for layer sets {0}, {3}, {0,3}, {0..7} (no warm-up) to find the allocation.
- 22:50 L1 probe (`logs/06-l1-probe-*.log`, model built with no warm-up): one GDN layer = 10,496 B of L1 per bank, one
  attention layer = 10,496 B — the same ~10.5 KB per core per layer for both kinds -> `TTMoEGate`'s four persistent
  L1-sharded buffers (score-correction bias, input indices, output scores, output indices: one 32x32 tile per core for
  `buffer_rows` = all 110 cores), allocated per layer instance. 48 layers = 395 KB per core, which the 2048-token 2D
  prefill matmul programs cannot fit their static circular buffers beside. Fix: `fused_decoder._share_gate_buffers`
  — one buffer set per mesh shared by every layer (the buffers are layer-independent constants / scratch outputs
  consumed right after the gate op, also inside the decode trace where the 48 gate ops run sequentially). Verification
  (8-layer L1 probe) and the fixed 48-layer chain queued.
- 22:56 L1 verification (`logs/06-l1-probe-shared8.log`): 8 layers with the shared gate buffers = 10,496 B of L1 per bank —
  identical to one layer (per-layer growth gone; the earlier 48-layer build reported 395,520 B). The 48-layer warm-up rerun
  follows the review-3 device runs.
- 23:13-23:21 Fixed 48-layer warm-up run 1 (`logs/06-warm-fixed-1.log`, counting-sort dispatch + shared gate buffers): all
  ten bucket forwards (128..2048, masked and full) complete for 48 layers — "prefill buckets warmed in 11.7 s"; no hang, no
  L1 clash. MEMORY_REPORT after the warm-up: DRAM 2,639,950,208 B/bank x 8 = 19.67 GiB per device (weights + KV pool for
  262144 x B=1 + GDN state + prefill persistent buffers), L1 10,496 B/bank (one gate buffer set), trace region empty
  (trace not yet captured in the repro). Run 2 and the readiness runs follow.
- 23:35 Stage 5 committed locally (81ad00bde5c, review #5 clean-pass). The stage-6 tree on top of it: tt/model.py,
  tt/generator.py, tools/readiness.py, doc/full_model/, and the `_share_gate_buffers` hunk in tt/fused_decoder.py.
- 23:21-23:30 Fixed 48-layer warm-up run 2 (`logs/06-warm-fixed-2.log`): same result — all buckets, 10.5 s, DRAM 19.67 GiB,
  L1 10,496 B per bank. Two clean passes of the sequence that hung 3/3 before the counting sort; readiness runs start
  (prefill check AIME24 -> book, teacher forcing AIME24 -> book, autoregressive).
- 23:30-23:38 Readiness prefill check, AIME24 chat reference (161 prompt tokens, 100 generated, k=100;
  `logs/06-prefill-check-qwen35_122b_aime24_chat_g100.log`, `readiness/prefill_check_qwen35_122b_aime24_chat_g100.json`):
  top1 0.980 (98/100), top5 1.000, top100 1.000 — the bar (top5 >= 98 %, top100 = 100 %) is met on the full 48-layer model.
  MEMORY_REPORT: DRAM 2,640,778,688 B/bank (19.68 GiB per device incl. the decode + sampler traces), L1 10,496 B/bank,
  trace region 3,579,904 B/bank = 27.3 MiB used of 512 MiB. Context-contract block `full_model_stage6` written.
- 23:39-23:47 Readiness prefill check, book reference (128 prompt tokens; `logs/06-prefill-check-qwen35_122b_book_p128_g100.log`):
  top1 0.970 (97/100), top5 0.990 (99/100), top100 1.000 — bar met (top5 >= 0.98). Same footprint as the AIME24 run.
- 23:48-23:57 Teacher forcing, AIME24 chat reference (`logs/06-teacher-forcing-qwen35_122b_aime24_chat_g100.log`, traced decode
  with the on-device sampler, next input forced from the reference): top1 0.980, top5 1.000, top100 1.000; TTFT 626.7 ms
  (161-token prefill through the 256 masked bucket + first token), decode 33.24 t/s/u (30.1 ms per token-out step: 48
  layers x ~0.45-0.58 ms device + LM head + sampler + the stock Generator's host path), e2e 27.51 t/s/u over 100 tokens.
- 23:57-00:06 Teacher forcing, book reference (`logs/06-teacher-forcing-qwen35_122b_book_p128_g100.log`): top1 0.970, top5
  0.990, top100 1.000; TTFT 389.5 ms (128 tokens, the 128 bucket), decode 33.05 t/s/u (30.3 ms per step), e2e 29.28 t/s/u.
  Both references clear the bar on prefill and on traced token-out decode. Autoregressive comparison running (HF greedy on
  the CPU first, 256 tokens, then TT).
- 00:05-00:2x Autoregressive (`logs/06-autoregressive.log`; plugin prompt "Once upon a time ... Elena ...", 59 prompt tokens,
  256 new tokens): HF greedy on the CPU and TT on-device greedy both produce 256 coherent tokens of the same story; the TT
  stream forks at token 20 ("a different tune" vs "in a rhythm she had never heard before") and stays coherent — no
  repetition, no language drift (read in full). 24 / 256 positions equal after the fork (informational). Qualitative
  verdict PASS; `06-full-model.check.sh` run next.
- 00:20 `06-full-model.check.sh` (MODEL_DIR + HF_MODEL set, `logs/06-full-model-check.log`): exit 0 — degenerate-output check
  on the TT completion: adjacent duplication 0.0, trigram loop fraction 0.030 (advisory threshold not reached), "No
  degenerate output detected"; context contract OK (target 262144, supported 262144, full HF context).
- 00:30 Stage-7 baseline profile on the 2-layer model (`logs/07-tokenout-profile.log`, `scripts/07-tokenout-profile.py`): warm
  token-out step 5.20 ms; the MODEL trace alone (execute_trace of the captured decode trace, no sampler, no readback) replays
  in 1.67 ms; under cProfile the step is 5.16 ms of which 4.7 ms is `Generator.read_decode_output` (the blocking to_torch of
  the sampled token = the device sync point) and ~0.4 ms the rest of the Python path. So the fixed per-step cost is the
  sampler trace + readback latency (~3.5 ms, device side), not host Python — README section 6 corrected (it had attributed
  the ~4 ms to the host path). At 48 layers: 22.6 ms model trace + 4-5 ms sampler/readback + 0.4 ms host ~= 30.1 ms.
  Stage-7 target #1 = the sampling contract (force-argmax vs top-k path, sampler placement, deferred readback).
- 00:45 Stage-6 review #1 = more-work-needed: P1 (a) the measured token-out path is the sampler's FORCE-ARGMAX strategy
  (qwen36 model_config sets allow_force_argmax=True; the logs print "Forcing argmax sampling") while the README described
  the top-k candidate path, and the alternative was never benchmarked; (b) cross-request stale-token merge: the stock
  Generator's reset path keeps the device token/position for slots not marked as freshly prefilled, and
  `Qwen35MoeGenerator.prefill_forward` never marked them; (c) B > 1 untested and its prefill went through the exact-length
  paged path (compiles after the trace is parked). P2 (d) qualitative suite for an instruct model must use the chat
  template (the Elena continuation is a raw prompt); (e) prompts > 2048 tokens through the generator untested. Fixes:
  `prefill_forward` marks `_slots_prefilled_since_decode`; B > 1 prefill = `_prefill_slots` (bind the model's B=1 GDN
  prefill scratch — allocated at warm-up — run each user through the segmented masked-bucket path against its page-table
  row, snapshot and `_write_gdn_slot`); `QWEN35_FORCE_ARGMAX` knob in `create_model`; `tests/test_generator.py` (cross-
  request token feedback, batched mixed prompts B = 2/8/32, long prompts 2381/4097 vs the all-positions path, both greedy
  sampler strategies with step timing); `06-qualitative.py` (6 shared prompts through `apply_chat_template`, HF CPU
  controls, per-prompt autoregressive_meta.json + qualitative_prompt_format.json); Tracy token-out profiles for both
  strategies (single-device readbacks replaced by mesh-composer reads for the profiler's op merge). Also: contract
  arithmetic (trace region already excluded from total_bytes_per_bank -> 12.75 GiB pool; GDN state per user in the
  per-user average), refpt sha256 + generation commands recorded, `_GATE_SHARED` keeps the mesh reference, README
  section-5 wording. Device chain `06-r1-chain.sh` running.
- 00:35-00:40 `tests/test_generator.py` run 1 (`logs/06-generator-tests.log`): cross-request token feedback PASSED (the
  first token of a request whose prompt length equals the previous prompt + generated length, and that minus one, equals
  the prefill truth; token feedback and determinism hold); long prompt T=2381 PASSED (segmented vs all-positions last-token
  logits PCC 0.99940, same token); greedy sampler force-argmax PASSED (5.13 ms per step, 2-layer). Failures, all in the
  test harness / new paths, fixed: (a) batched builds (B=2/8/32) failed inside the WARM-UP (`prefill_masked_bucket`
  reshape volume mismatch): the masked-bucket programs run B=1 GDN state, so on a batched model the warm-up and every
  prefill must run on the model's B=1 GDN prefill scratch (`_bind_gdn_prefill_scratch`, allocated at warm-up) — the
  generator now binds it for the warm-up and routes every prefill of a batched model through `_prefill_slots`; (b) T=4097
  failed in the TEST's truth path (`prefill_all_logits` runs the whole prompt as one program; above 4096 the GDN
  in-projection program does not fit) — the test now checks segment continuity through a second path (prefill of T-1
  tokens + one traced decode must give the segmented prefill's token) and keeps the single-pass comparison for T <= 4096;
  `prefill_all_logits` stays a <= 4096-token path (readiness prefill checks are short); (c) the top-k candidate sampler path
  asserts a rank-4 `[1,1,1,sampler_batch]` preallocated output (`sampling_device_operation.cpp:130`) — the model's `[B,1]`
  token input cannot be its output, so `_token_feedback_buffer` returns None on that path (sampler-owned output, no token
  feedback: measurement-only mode); the force-argmax path keeps writing into the token input. Tracy on the token-out step
  fails in the ops merge ("Device data missing ... device 1" for a single-device host op of the sampler stack), so the
  strategy comparison uses a 4-way wall breakdown instead (model trace replay / sampler trace / token readback / host).
- 00:44-00:52 Qualitative suite (`06-qualitative.py`, `qualitative/`): 6 shared prompts through the chat template, HF CPU bf16
  controls (6 x 128 tokens in ~1 min: the bf16 checkpoint is page-cache resident), TT 128 tokens each in 4.1-4.2 s (~30 ms
  per token incl. prefill). All six TT completions coherent, on-task reasoning traces mirroring HF's structure; forks at
  positions 78 / 9 / 60 / 29 / 36 / 3; equal positions 83 / 10 / 61 / 31 / 36 / 4 of 128. `06-full-model.check.sh` rerun over
  all seven autoregressive_meta.json: no degenerate output, contract OK (`logs/06-full-model-check2.log`). Verdict PASS.
- 01:03-01:07 `tests/test_generator.py` run 2 (`logs/06-generator-tests2.log`): cross-request PASSED; long prompts 2381 and 4097
  PASSED (4097: segmented prefill token == prefill(T-1) + traced decode token); sampler strategies: force-argmax 5.13 ms
  per step vs top-k candidates 2.47 ms = model trace 1.673 + sampler 0.541 + readback 0.041 + host 0.216 ms (2-layer) ->
  the full-vocab all-gather + argmax costs ~3.5 ms per token; the top-k candidate path (per-shard top-k, 32-candidate
  all-gather, ttnn.sampling k=1) 0.54 ms. DECISION: top-k candidate path = default (`allow_force_argmax=False` in
  `create_model`, env override for A/B). To wire its token feedback the persistent decode token buffer becomes the sampler's
  rank-4 `[1,1,1,32]` shape (host pads to 32; the model slices its B tokens out of a row-major view). Two remaining
  failures: (a) batched builds: `reset_state_inplace` on the batched [B,...] buffers copies the model's B=1 zero templates
  (qwen36 keeps `_zero_rec` at the prefill-scratch shape) -> the generator now zeroes a batched model from B-sized templates
  allocated at warm-up; (b) `prefill_logits(prompt + [t])` returned a different token AFTER traced decodes had run in the
  process (40527 vs 8599 before) -> new test `test_generator_all_logits_after_traced_decode` isolates it (run 3).
- 01:10-01:25 Generator tests run 3 (`logs/06-generator-tests3.log`): cross-request and long prompts PASS; batched builds now
  fail later, in the B=1 reference generation on the batched model (`copy_host_to_device` shape mismatch: the trace is
  captured at max_batch_size, the host inputs were B=1) -> `decode_forward` pads tokens / positions (-1) / page table to
  max_batch_size and returns the caller's B rows. The "token feedback 8599 != 40527" assertions and the new all-logits test
  point at a NEAR-TIE: for `prompt + [1146]` the top-2 logits are 8599 and 40527 (the mode probe's top-5:
  [8599, 40527, 467, 1146, 96333]); the mode probe run at 01:18 (gate-buffer sharing on AND off) gives 8599 from every path
  in its process, the pytest process got 40527 from both prefill paths and 8599 from decode. Prefill and decode take
  different matmul programs (bf16 roundings), so a near-tie can legitimately order the two differently; a token-equality
  test must allow the runner-up when the margin is small -> `_truth_set` (top-2 when the margin < 1 logit) in the tests,
  and the all-logits test compares logits (bit-identical before/after decode; PCC >= 0.999 vs the masked path). Whether
  the prefill itself is bit-reproducible ACROSS processes is measured next (`06-determinism-probe.py`: three identical
  processes + two trace-region sizes + single-layer variants, top-5 logit bit patterns).
- 01:22-01:29 Determinism probe (`logs/06-det-{a,b,c,l0,l3}.log`): three fresh processes (two trace-region sizes) give
  bit-identical top-5 logits for `prompt + [1146]` from BOTH prefill paths — [8599, 40527, 467, 1146, 96333] with values
  5.5938 / 5.2188 / 5.125 / 5.0625 / 4.9062 (a 0.375-logit near-tie between 8599 and 40527); the attention-only build is
  identical across its two evaluations too (the GDN-only build failed to construct — the readiness generator needs at least
  one attention layer for its KV pool, not a determinism finding). So prefill is reproducible across processes. The 40527
  seen in the pytest process came from the gate-buffer share: `_share_gate_buffers` was scoped to the MESH, and a pytest
  session builds several models on one mesh — every model after the first pointed its gates at the first model's (by then
  deallocated) L1 buffers. Fixed: the share lives on the model's `args` object (one per build). Run 4 of the generator tests
  will confirm; the 48-layer single-model processes were never affected (one model per process).
- 01:31-01:37 `tests/test_generator.py` run 5 (`logs/06-generator-tests5.log`, gate share scoped to the model): 9 passed —
  cross-request token feedback; batched mixed prompts B = 2 / 8 / 32 (every user's 4 tokens == its B=1 generation; the
  per-slot warmed prefill + padded decode inputs work; largest tested batch 32); long prompts 2381 / 4097; both sampler
  strategies with the step breakdown (force-argmax 5.119 = 1.680 model + 3.344 sampler + 0.051 readback + 0.044 host ms;
  top-k candidates 2.417 = 1.681 + 0.567 + 0.052 + 0.117 ms); all-positions prefill bit-identical before/after traced
  decodes and PCC 1.0 vs the masked path (same top-2 {8599, 40527}; margin 0.062 in the pytest mesh vs 0.375 in the
  readiness mesh — an environment difference, same top-2, recorded). 48-layer decode-side reruns on the top-k default next.
- 01:38-01:47 Teacher forcing rerun on the top-k default, AIME24 (`logs/06-teacher-forcing2-qwen35_122b_aime24_chat_g100.log`):
  top1 0.970 (97/100), top5 1.000, top100 1.000; TTFT 632.1 ms; decode 36.45 t/s/u = 27.4 ms per token-out step (was 30.1
  with force-argmax: -2.7 ms, as the 2-layer breakdown predicted); e2e 29.62 t/s/u. The 1-point top-1 change (98 -> 97) is a
  near-tie flip on one position (top-5 and top-100 unchanged).
- 01:47-01:56 Teacher forcing rerun, book (`logs/06-teacher-forcing2-qwen35_122b_book_p128_g100.log`): top1 0.970, top5
  0.990, top100 1.000 (unchanged); TTFT 394.9 ms; decode 36.49 t/s/u (27.4 ms per step); e2e 31.90 t/s/u. README headline,
  section 4 and section 6 carry the top-k-default numbers with the force-argmax first-run numbers alongside.
- 01:56-02:10 Autoregressive rerun on the top-k default (`logs/06-autoregressive2.log`, `autoregressive/` overwritten): 256/256
  tokens, fork from HF at token 20 again (24 equal positions), coherent continuation (a hermit on the western mountain, the
  mountains "breathe every hundred years"), no repetition, no drift. Qualitative suite rerun next.
- 02:08-02:20 Qualitative suite rerun on the top-k default (`logs/06-qualitative2.log`): all six TT completions coherent,
  on-task reasoning traces; forks from HF at 97 / 84 / 65 / 43 / 36 / 15 (equal positions 97 / 100 / 66 / 43 / 38 / 17 of
  128), 3.8 s per 128 tokens. `06-full-model.check.sh` over all seven meta files: exit 0, no degenerate output
  (`logs/06-full-model-check3.log`). Stage-6 evidence complete on the committed default; re-review requested.
- 02:40 Stage-6 review #2 = clean-pass (all five review-1 items verified closed from the artifacts). Pre-commit hygiene from
  its concerns: README section 7 first limitation updated to the committed 27.4 ms step; section 6's 2-layer rows labelled
  force-argmax era vs top-k default; section 3 no longer cites the failed Tracy directory (the ops-merge failure signature is
  recorded instead; `doc/full_model/tracy/` stays unstaged); hand-off notes added (row->slot prefill mapping, whole-batch
  reset, host round-trip of the per-user GDN state, `prefill_logits` is a max_batch_size == 1 path, the 98 -> 97 top-1
  tie-break, the fixture-vs-readiness mesh logit difference); `prefill_forward` asserts B <= max_batch_size. Trace evidence
  re-run on the committed token-buffer shape / sampler (`trace_evidence.log` refreshed) before the commit.
- 02:45 Trace evidence re-run (`trace_evidence.log`): token feedback, positions, unchanged/changed page table, determinism all
  PASS on the committed rank-4 token buffer + top-k sampler; token-out step 2.35 ms on the 2-layer model.
- 02:50 Stage 6 committed locally: 215075683d2 (38 files). Stage 7 begins (doc/optimized_full_model/).
