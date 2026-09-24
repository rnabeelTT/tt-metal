# Multichip decoder — Qwen/Qwen3.5-122B-A10B on the (1,4) Blackhole mesh (stage 4)

`tt/multichip_decoder.py` — `MultichipDecoder(OptimizedDecoder)` driven by `MultichipConfig` (`MULTICHIP_SELECTED`), one
decoder layer tensor-parallel over the 4 dies of the P300x2 (`MESH_DEVICE=P300x2`, FABRIC_1D, Ring). Baseline: the stage-3
optimized decoder (commit 19aa5d24777, `OptimizedDecoder` / `SELECTED`). Plan and rejected alternatives: `PLAN.md`.
Evidence: `tests/test_multichip_decoder.py`, `tools/baseline_dump.py` (single-die reference outputs, `baseline/*.pt`, gitignored),
`tracy/<kind>/*_perf_report.{txt,csv}`, `watcher/`, run logs `~/qwen35-122b-bringup/logs/04-*.log`.

## 1. Mesh strategy (see PLAN.md for the full derivation)

| what | strategy on the (1,4) mesh | per device |
|---|---|---|
| residual stream | FRACTURED along the hidden dim between layers (the stacking contract; tests shard once at entry, compose once at exit) | `[1,1,B or T,768]` bf16 |
| norms | qwen36 `DistributedNorm`: all-gather -> RMSNorm on the replicated activation (decode output L1 width-sharded on the 16-core `attn_input_grid`); prefill norm non-distributed (framework rule dim <= 4096 on a 1D mesh, option A) | — |
| attention | TP=4 via `TPAttention`: fused `[q|gate|k|v]` column-parallel, local heads, paged SDPA on the local KV head, `wo` row-parallel + reduce-scatter (`tt_all_reduce(dim=3)`); KV heads replicated per device pair (`kv_replication`) | wqkv `[3072,4608]` bfp8, wo `[2048,3072]` bfp8, 8 q heads, KV head `d // 2` |
| paged KV cache | one `[blocks, 1, 64, 256]` tensor per device (its local head), bfp8, page table / positions replicated | 6528 B/token/device over the 12 attention layers |
| GDN | TP=4 via `TPGatedDeltaNet`: fused `[qkv|z|a|b]` column-parallel, blocked depthwise conv1d (3 x 1024 channels), local recurrence (shared rewritten kernel), out-proj row-parallel + reduce-scatter | qkvzab `[3072,5152]` bfp8, out `[2048,3072]` bfp8, 4 k-heads / 16 v-heads, state `[B,16,128,128]` bf16 |
| routed experts | TP on the expert intermediate dim: all 256 experts on every device at width 256, routing replicated, gather-mode `sparse_matmul` at decode / masked or expert-sorted at prefill (stage-3 paths unchanged), ONE reduce-scatter of (routed + shared) partials per layer | `up_gate [1,256,3072,512]` bfp4, `down [1,256,256,3072]` bfp4 (324 MiB/layer) |
| shared expert | column/row parallel (`[3072,513]` packed, `[256,3072]`), summed into the same reduce-scatter | bfp8 |
| router | replicated (bf16 `[3072,256]`; gate op for the single-user decode row, functional router otherwise) | — |
| collectives per layer | attention-norm all-gather, mixer reduce-scatter, ffn-norm all-gather, MoE reduce-scatter (4 per layer, 192 per decode step; 4.6 KB per device each at B=1) | — |

Rejected alternative — EP=4 (64 experts per device): same 15.2 GiB of expert weights per device and the same single reduce-scatter,
but data-dependent load (expected max 3.5 of the 8 active experts on one device), no gather-mode `sparse_matmul` (the local active
count is data dependent, so only the masked mode with `nnz=None` and its EXPANDED output applies: ~46 MB of extra elementwise
traffic per decode step), and a dispatch rework of the sorted prefill. It wins only for large batches / long prefill (about
2-2.5x on the routed rows); recorded in PLAN.md as the optimize-stage candidate `routed_mode="ep"`.

## 2. Context contract on (1,4)

Per-device DRAM (31.9 GiB): routed experts bfp4 15.19 GiB; attention mixers 0.25; GDN mixers 0.83; shared experts 0.12;
routers 0.07 (bf16 `[3072,256]` x 48); embedding bf16 hidden-sharded `[248320,768]` 0.355 + LM head bfp8 vocab-sharded `[3072,62080]` 0.189 = 0.54
-> weights 17.00 GiB. GDN state at B=32: 0.59 GiB (bf16 recurrent state + bf16 conv shift register, 19.76 MB/user/device
over the 36 GDN layers). Reserve for trace, CCL buffers and prefill transients (the expert-sorted dispatch peaks ~330 MB at
2048 tokens): 3.0 GiB. KV pool 11.306 GiB at 6528 B/token/device (12 attention layers x 1 local KV head x 256 x (K,V) x
1.0625 B) -> 1.860 M tokens: 7 users at the full 262144 context (1.835 M tokens, thin headroom), or 32 users x 58 K tokens.
The advertised context stays 262144 (no hard limit reached). Block `multichip_decoder_stage4` in `doc/context_contract.json`
holds these figures (computed, not measured); the full-model stage measures the per-device DRAM after all 48 layers are
loaded and replaces them before the KV pool is trusted.

## 3. Correctness — mesh vs single-chip baseline and vs HF

Baseline: the stage-3 `OptimizedDecoder` with `SELECTED` on one die (commit 19aa5d24777), the same dtypes (bfp4 routed,
bfp8 mixers/KV, bf16 GDN state). Its outputs and states are dumped once per (layer kind, length) by `tools/baseline_dump.py`
(`baseline/*.pt`) and compared on the mesh (`test_mc_matches_single_chip_baseline`, bar 0.995). HF comparisons use the
per-layer HF reference (`tests/reference.py`). Final full suite with the final config (`MULTICHIP_SELECTED`, fixed
reduce-scatter writer): 35 passed, 0 failed (`~/qwen35-122b-bringup/logs/04-mc-final-suite.log`, 12.5 min); probe run
`04-mc-probe.log`.

| check (GDN layer 0 unless noted) | PCC |
|---|---|
| mesh vs single-chip prefill T=5 / 37 / 129 / 333 / 2048 | 0.99993 / 0.99964 / 0.99937 / 0.99921 / 0.99906 |
| attention: mesh vs single-chip prefill / decode T=5..2048, chunked 4429 | >= 0.99966 / >= 0.99977, chunks >= 0.99995 |
| mesh vs single-chip decode at pos 5 / 37 / 129 / 333 / 2048 | 0.99986 / 0.99939 / 0.99907 / 0.99906 / 0.99915 |
| mesh vs single-chip chunked prefill 2048+2048+333 (T=4429) and decode after it | 0.99904 / 0.99903 |
| GDN recurrent state vs single-chip (bar 0.995), T=5 .. 2048 and chunked 4429, after prefill and after decode | 0.99930 (T=5) down to 0.99648 (T>=2048); conv window after prefill T=5 0.99992 |
| mesh vs HF prefill T=5..2048 (GDN / attention) | >= 0.9984 / >= 0.99998 |
| mesh vs HF, real weights, T=37 prefill / decode (GDN) | 0.99991 / 0.99999 |
| chunked prefill T=4429 vs HF and decode after it (GDN) | 0.99840 / 0.99829 |
| 128 decode steps after prefill 128, then continuation T=100 at 256 (GDN / attention) | 0.99907, 0.99875 / 0.99995, 0.99997 |
| non-aligned prefill T=5 / 37 / 333 (GDN) | 0.99999 / 0.99983 / 0.99927 |
| non-aligned prefill T=5 / 37 / 333 (attention) | 0.99984 / 0.99988 / 0.99995 |
| batched decode B=32 (GDN, per-user min) / distinct positions 5, 37, 129, 300 | 0.99899 / >= 0.99905 |
| GDN state reset after prefill / prefill+decode / prefill+continuation history, fresh T=5 | 0.99999 (all three) |
| traced decode: replays 1-2 vs replay 0 (both kinds) | bit-identical replays; replay vs HF 0.99911 (GDN) |

Paged KV cache on the mesh: one `[blocks, 1, 64, 256]` bfp8 tensor per device (local head), page table and positions
replicated; `test_mc_batched_decode_distinct_positions` fills four users at different positions, `test_mc_chunked_prefill_matches_hf`
exercises `start_pos` chunks with the one-row page table. The stacking contract (fractured `[1,1,T|B,768]` in and out) is
what every test shards once at entry and composes once at exit (`_MeshHarness`).

## 4. Performance — single-chip baseline vs mesh (speedup, efficiency, tt-perf-report)

Baseline: the stage-3 `OptimizedDecoder`/`SELECTED` on one die, measured with the same timing helpers in a separate pytest
process (`-m perf_single`, real weights, warmed; `logs/04-perf-single.log`); mesh: `-m perf` (`logs/04-perf-plain.log`), no
profiler. Prefill 2048 = four 512-token `start_pos` chunks; decode = 3 warmed traced replays (`tools/perf_join.py` -> `HOST_WALLS.md`).

Host wall (ms):

| layer | case | single-chip ms | mesh (1x4) ms | speedup | efficiency |
|---|---|---|---|---|---|
| full_attention | prefill_128 | 16.097 | 15.356 | 1.05x | 26% |
| full_attention | prefill_2048 | 229.755 | 238.668 | 0.96x | 24% |
| full_attention | traced_decode | 0.727 | 0.458 | 1.59x | 40% |
| linear_attention | prefill_128 | 18.511 | 15.867 | 1.17x | 29% |
| linear_attention | prefill_2048 | 250.441 | 238.718 | 1.05x | 26% |
| linear_attention | traced_decode | 0.867 | 0.584 | 1.48x | 37% |

Sweeps on the mesh (`tools/sweep_multichip.py`, real weights, PCC vs HF identical for every legal candidate; decode = traced,
3 replays; noise about +-6 us, see `state_bf16` = the baseline dtype at +6 us; `sweeps/SUMMARY.md`):

| family | candidates | best delta (GDN / attention) | decision |
|---|---|---|---|
| `shared` (shared-expert decode cores) | auto(17), 8, 16, 17 (1D) | 16 cores: -26 / -26 us | SELECTED (`shared_decode_cores=16`); the remeasured tables below use it |
| `dense_1d` (attention qkv / wo cores) | auto, 32/32, 64/64, 88/33, 44/44 | 88/33 or 44/44: -9 us | not adopted in stage 4 (inside 2x noise); stage-5 candidate |
| `gdn_1d` (GDN qkvz / out cores) | auto, 32/32, 64/64, 88/33, 44/44 | -2 us | no change |
| `gate_up` (routed gate/up cores / in0) | 16/32, 16/96, 8/32, 16/8 | 16/96: -5 us; 8 cores +19, in0 8 +45 | keep 16/32 |
| `down` (routed down cores / in0) | 48, 96, 32, 24 (in0 8) | 48 best; 96 illegal (`not on_dispatch_core`, 96-core grid touches the dispatch cores) | keep 48 |
| `norm_mode` | option A, option B | B fails (section 5) | A |
| `gdn_state` | bf16 (selected), fp32 + high-precision decode | fp32 +15 us | bf16 |
| `sdpa` (16384-token context) | chunks 0/0 (auto), 32/256 | 32/256: -7 us | keep 32/256 |
| `conv1d` (prefill conv block channels) | 1024, 1536, 3072 | all equal (decode unaffected; prefill wall within noise; a single 3072-channel call works at T=128) | keep 1024 (the block size proven on the 9B TP path, whose 4096 channels per device did overflow the conv1d CB) |

Device time from Tracy (`scripts/04-perf.sh`, `tt-perf-report --no-merge-devices` per device, `tracy/<kind>/*_mesh_perf_report.*`;
rendered by `tools/perf_tables_mc.py` -> `PERF_TABLES.md`; single-chip = the committed stage-3 tables). The per-device
`prefill_2048` report files (4 devices x 948 / 768 ops) exceed the repo's 500 KB hook limit: the commit carries the device-0
rows as `prefill_2048_mesh_perf_report_dev0.csv`; the full `.csv`/`.txt` and the `_stacked` plots stay in the workspace copy
(`~/qwen35-122b-bringup` runs regenerate them with `scripts/04-perf.sh`):

| layer | case | single-chip device ms | mesh device ms (max over 4 devices) | speedup | efficiency | CCL ms per device |
|---|---|---|---|---|---|---|
| GDN (L0) | prefill_128 | 16.197 | 8.213 | 1.97x | 49% | 0.14-0.19 |
| GDN (L0) | prefill_2048 | 124.241 | 49.963 | 2.49x | 62% | 1.26-1.37 |
| GDN (L0) | traced decode | 0.827 | 0.474 | 1.75x | 44% | 0.045-0.054 |
| attention (L3) | prefill_128 | 15.138 | 9.035 | 1.68x | 42% | 0.07-0.09 |
| attention (L3) | prefill_2048 | 88.948 | 47.198 | 1.88x | 47% | 0.69-0.78 |
| attention (L3) | traced decode | 0.656 | 0.419 | 1.57x | 39% | 0.046-0.061 |

Findings (device 0, one traced decode replay, GDN layer; attention differs only in the mixer rows):

- Communication: the four collectives per layer (2 `AllGatherAsync` for the norms, 2 `ReduceScatterMinimalAsync` for the
  mixer and the MoE) cost 46-50 us per step on 12 cores, 9-11 % of the device time; they are latency bound at B=1
  (4.6 KB payload per device). Device times are balanced within 1 % over the 4 devices.
- DRAM: the routed gate/up `sparse_matmul` (32 x 3072 x 512 per expert, BFP4 LoFi, 16 cores) takes 62 us for 7.1 MB of
  expert weight = 114 GB/s (22 % of the 512 GB/s die peak); the down projection (48 cores) 18 us. The GDN in-proj
  (32 x 3072 x 5152 bfp8, 41 cores) reads 16.8 MB in 51 us (330 GB/s, 61 %); attention qkv (48 cores) 54 us; `wo`
  (32 x 2048 x 3072, 32 cores) 22 us (290 GB/s). The shared expert (32 x 3072 x 544 bfp8) took 39 us for 1.8 MB
  (45 GB/s) with the ttnn auto program; the 16-core 1D config selected from the `shared` sweep brings it to 15 us. The
  routed rows and the mixers are the `gate_up`, `down`, `dense_1d`/`gdn_1d` sweep families (`sweeps/`, no change adopted).
- Compute: nothing is math bound; the largest FLOP utilisation is 25 % (`wo`).
- Data movement: 15 `BinaryNg` elementwise ops (58 us), 4 `ReshapeView` (23 us), 4 `Ternary`, tilize/untilize pairs around
  the GDN recurrent step and slices/copies of the fused projection make up ~35 % of the GDN step; the attention step has
  the same pattern with 10 `BinaryNg` (36 us) and the RoPE / head-creation chain (~25 us).
- Prefill: device time halves or better (2.5x at 2048 tokens for GDN) but the host wall does not (1.0-1.2x): the wall is
  host bound. Under the profiler the op-to-op gaps on device 0 sum to 540-610 ms per 2048-token window (948 / 768 ops
  per device); without the profiler the wall is 241 ms against 50 ms of device time. The routed `sparse_matmul` rows
  (expert-sorted dispatch, 377 active groups of 32 rows per 512-token chunk) are 48-50 % of the prefill device time; the
  replicated dispatch (embeddings gather, sorts, scatters, accumulation) is 20-25 %.

## 5. Shared-code changes (commit 23d2bca06fa) and stage-3 knobs that do not transfer

Imported from the Ornith branch only after reading each hunk (`~/qwen35-122b-bringup/artifacts/stage4/ornith_hunks.md`);
nothing else of Ornith's TP path (per-die replication, `max_num_seqs=1`, host sampling, `fabric_config: DISABLED`) is used:

| file (`models/demos/blackhole/qwen36/tt/`) | change | why it is needed here |
|---|---|---|
| `tp_common.py` | `in0_block_w = largest divisor of k_tiles <= cap` in the prefill program config; `prefill_norm_agmm_enabled(args)` and `mlp_gateup_agmm_enabled(num_devices, args)` gates | the (1,4) fused norm+matmul all-gather gathered twice at hidden 3072 ("K=8192 vs K_w=2048"); the fused AGMM is off for dim <= 4096 unless `prefill_norm_distributed` (option B) |
| `attention/tp.py` | `_fuse_agmm` gate; unfused fractured-prefill fallback with `ttnn.all_gather` (not `tt_all_gather`, which deallocates its input); bfp8 SDPA when `kv_dtype == bfloat8_b`; SDPA decode chunks from args; `reset_cache` alias | option A prefill path; stage-3 KV/SDPA policy carried over |
| `gdn/tp.py` | blocked depthwise conv1d (`gdn_conv1d_max_channels`, default 1024, equal blocks asserted); `reset_state(batch_size)`, `recurrent_state`/`fused_conv_state` properties; `_fuse_agmm` gate + fallback all-gather; `_z_mc = DRAM` for S > 32; `gdn_state_dtype` / `gdn_decode_high_precision` from args | the 9B TP conv1d (4096 channels per device) overflowed its L1 circular buffer on main and passes with the import (13/13); for this model (3072 channels per device) one call also works (sweep `conv1d`), the 1024 block is kept as the proven configuration |
| `layer.py`, `mlp.py` | gates take `args` | same AGMM rule for the dense MLP path (unused by this model, kept consistent) |
| `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_writer.cpp` (ttnn kernel, stage 4) | scatter-write header `set_state` calls guarded by `if constexpr (num_tiles_to_write_per_packet > 1)` | with fp32 (4 KiB) pages the op packs one tile per packet and set up a 1-chunk scatter header, tripping the fabric `ASSERT(chunk_count >= 2)` under the watcher (work_log 10:31); one-tile packets always use the unicast path, so behaviour is unchanged |

Stage-3 knobs disabled on the mesh by `MultichipConfig.to_optimization_config()` (with the reason):

| knob | stage-3 value | on the mesh | reason |
|---|---|---|---|
| `prefill_norm_distributed` (option B: fractured norm output + fused all-gather matmul in the mixer) | n/a | off (option A) | option B fails on the mesh: the MoE block (router, experts) receives the fractured `[T,768]` norm output (`TT_FATAL a_shape[-1] == b_shape[-2]: width=768 height=3072`, `sweeps/norm_mode.jsonl`); it needs an explicit all-gather before the router, a stage-5 candidate |
| `dense_decode_mode` (attention qkv/o, GDN o) | `dram_sharded`, 1 reader | `default` (TP mixers' own `proj_1d_decode` matmuls) | the TP modules own their matmuls; per-device N is 4x smaller (4608 / 2048), so the DRAM-sharded win (-6 us) does not carry; measured in the `dense_1d` / `gdn_1d` sweep families |
| `decode_norm_sharded`, `attn_norm_out_sharded` | on | off | `DistributedNorm` already emits the width-sharded decode activation on the 16-core `attn_input_grid` |
| `mega_split_ds` | off | off | single-device candidate, rejected in stage 3 |
| routed / shared expert geometry | gather 64 cores / in0 32; shared 44 cores | gather 16 cores / in0 32, down 48 / in0 8, shared 16 cores (1D) | per-device widths 512 (gate/up) and 256 (down); swept in `gate_up`, `down`, `shared` (section 4) |
| router | TTMoEGate op for the single-user decode row | unchanged (replicated per device) | identical top-8 on every device is what makes TP-experts remap-free |

## 6. Watcher / stress / fallback audit

- Runtime fallback audit (`test_mc_runtime_fallback_audit`, after warming every length): attention prefill/decode issue no host
  transfers; GDN non-aligned prefill issues 3 `from_torch` (the `valid_len` selector and the fused-chunk mask, all inside the
  shared kernels: `fused_chunk.py:158/163`, `ttnn_gated_deltanet.py:205`), the same set as on one die (stage 3 accepted <= 4).
  Source audit (`test_mc_source_fallback_audit`): no `to_torch`/`from_torch` in the decoder forward paths.
- Routing identity across devices (`scripts/04-routing-probe.py`, `logs/04-routing-probe.log`, real weights, layer 0):
  the replicated router returns bit-identical top-8 ids AND scores on all 4 devices for the single-user decode row (gate
  op), a B=32 block and a T=128 prefill (functional router), 3 repeats each — the premise of the remap-free TP-experts
  path (ROUTING_RESULT PASS).
- Determinism: `test_mc_traced_decode` (replay bit-identical to eager) and `test_mc_stress_repeated_runs` (20 prefill+decode
  cycles over mixed lengths incl. the sorted-dispatch range; final T=37 pair bit-identical).
- Watcher runs (`TT_METAL_WATCHER=10`, all features, logger at Warning; `watcher/pytest_<i>.txt`, `watcher/watcher_<i>.tail.txt`, `watcher/TALLY.md`,
  `~/qwen35-122b-bringup/artifacts/04-watcher-<i>/`): one pytest process per test with `tt-smi -r` between them
  (`scripts/04-watcher-each.sh`): real-weight T=37 GDN and attention, batched decode B=32 (attention), chunked prefill
  to 4429 tokens (GDN and attention), traced GDN decode — 6/6 PASSED, watcher.log 2503-7599 lines each, 0 bad lines, no
  assert tripped in any model kernel.
  - First watcher run (before the fix) tripped a real assert in the ttnn ring reduce-scatter writer (1-chunk scatter
    header with fp32 4 KiB pages from the fused GDN out-proj `matmul_reduce_scatter_async`); fixed in the kernel
    (section 5, work_log 10:31); results were never affected (one-tile packets use the unicast path).
  - Every watcher-enabled process on this board fails at TEARDOWN of the fabric routers on device 0 eth core (0,9) /
    virtual (29,25), in one of two forms: (i) "Timed out while waiting for active ethernet core 29-25 to become active
    again" (exit 134 from `MetalContext::teardown`), or (ii) after the test PASSED, the watcher reports "subordinate_erisc
    detected invalid NOC command buffer state before starting the next kernel ... Current kernel: fabric_erisc_router.cpp"
    on that core. Tally over 14 watcher processes (`watcher/TALLY.md`): form (ii) in exactly the 2 processes whose 10 s
    watcher dump tick fell 0-4 s into teardown (at or after the PASSED report) (the aborted "Dump #4 at 33.188s" / "Dump #5 at 43.244s" lines in
    `watcher/watcher_2.tail.txt` / `watcher_5.tail.txt`; `MetalContext::teardown` detaches the watcher before the eth-core
    wait, so a dump must be in flight to observe (ii)); form (i) in the other 12, including 3 attention repeats whose
    teardown fell between ticks and every model-free control (20 plain all-gathers; the attention reduce-scatter pattern —
    bfp8/bf16 `[1,1,128,3072]` and bf16 `[1,1,1,3072]` via `tt_all_reduce` plus `ttnn.all_gather` — for 20 and 100
    rounds, `scripts/04-attn-ccl-control.py`). The erisc NOC command-buffer state that check reads is programmed only by
    the fabric router code on that erisc, never by model worker kernels; the check itself is
    `DebugAssertNCriscNOCPacketTagClearedTripped`, raised by the erisc firmware AFTER `kernel_main()` returns
    (`tt_metal/hw/firmware/src/tt-1xx/active_erisck.cc:51`, waypoint "KD" then "NKFW" — the dumps show "Last waypoint:
    SEW,NKFW"), and the router kernel only returns when the fabric is torn down at mesh close, so this check can only fire
    at teardown whatever the model did. Classification: both forms are the fabric router
    not exiting cleanly under eth-core watcher instrumentation at teardown, on this board — not attention-specific, not
    model-specific. `TT_METAL_WATCHER_DISABLE_ETH=1` (`artifacts/04-watcher-noeth`) makes the same test exit 0 with a
    clean teardown, and every non-watcher run (final suite 35/35, perf, sweeps) tears down cleanly. Every model kernel is
    watcher-clean in all 14 processes (0 bad watcher.log lines, no tripped assert after the reduce-scatter writer fix).
    A `tt-smi -r` after each watcher run is part of the scripts; to be reported upstream with `watcher/`.

## 7. Limitations and what stage 5 inherits

- Prefill at 2048 tokens (4 x 512 chunks) gains nothing over one die in host wall (section 4): the wall is host-dispatch
  bound (the stage-3 device time at 2048 was about half the wall), and the MoE dispatch (router, sort, gather/scatter) is
  replicated on every device. Stage 5 targets: chunk-outer traced prefill, fewer/cheaper dispatch ops, the `routed_mode="ep"`
  candidate for long prefill.
- Decode efficiency 39-44 % of linear (device time; 37-40 % host wall): the four collectives per layer and the latency-bound small ops dominate at B=1; the
  tt-perf-report tables (section 4) list the top ops. Stage 5 works on collective placement, fused CCL+matmul and persistent
  buffers under the fractured residual contract.
- The single-chip baseline cannot run on an in-session `(1,1)` submesh of the fabric mesh (parent-mesh hang at the first
  readback afterwards, p300x2, 2026-09-24); it runs in a separate process (`-m perf_single`).
- CCL dtypes (stage-5 item): the attention `wo` prefill partial reaches the reduce-scatter as bfp8 (four quantised partials
  are summed; attention PCC vs HF 0.99998, accepted), the GDN prefill out-projection uses the fused fp32
  `matmul_reduce_scatter_async`, MoE and decode partials are bf16 (`scripts/04-rs-dtype-probe.py`).
- Watcher runs need a `tt-smi -r` afterwards (eth-core 29-25 fabric-router teardown failure under the watcher, both forms
  explained in section 6 / `watcher/TALLY.md`; upstream report pending). Watcher evidence on this board can never show a
  clean teardown, so a future eth-core regression must be told apart by the message text and the dump timing.
- Non-aligned GDN prefill still issues 3 host `from_torch` per call inside the shared kernels (`valid_len` selector and fused-chunk mask; accepted in stage 3, bound <= 4);
  the full-model stage builds it on device.
- Batched decode (B > 1) uses the functional softmax/top-k router on every device (the gate op is bit-nondeterministic with
  several valid rows); the single-user row uses the gate op.
- Stage-5 inputs from the review of this stage: re-measure `dense_1d` 88/33 (-9 us on 4 of 4 candidates) with more
  replays; probe `output_tile [32,16]` for the routed gate/up `sparse_matmul` (N=512 caps the grid at 16 cores, 22 % of DRAM
  peak); option B as the per-norm variant (attention norm distributed, ffn norm replicated) instead of flipping both;
  one shared `TT_CCL` across layers before the full model (`from_state_dict` builds one per layer today).
- Vision tower: not ported (user scope "text now, vision later"); the `Qwen3_5MoeForConditionalGeneration` arch name is
  kept for serving with image/video limits 0.
