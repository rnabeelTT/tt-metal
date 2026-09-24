# Stage 5 — optimized multichip decoder: work log

Baseline = stage 4 commit 1b9770ef8e6 (`MultichipDecoder` / `MULTICHIP_SELECTED`, shared_decode_cores=16). Evidence
harness: `tests/test_multichip_decoder.py -m perf` (host walls), `scripts/04-perf.sh` (Tracy per device), `tools/sweep_multichip.py`.

### 2026-09-24 11:55 — operation-topology audit (device 0, one traced decode replay, from the stage-4 Tracy CSVs)

GDN layer (L0): 103 ops, 476 us device; attention layer (L3): 84 ops, 417 us. Matmuls 182 / 166 us, collectives 50 / 49 us,
everything else (~90 / ~70 small ops at 1-8 us each) 245 / 200 us. See README section 1 for the table and the candidate families.

### 2026-09-24 11:55–12:03 — candidate families implemented as `MultichipConfig` knobs (default path unchanged until selected)

- `fuse_out_mmrs_decode`: decode out-projection fused with the reduce-scatter (`ttnn.experimental.matmul_reduce_scatter_async`
  via the qwen36 helpers `build_mmrs_decode_state` / `matmul_reduce_scatter_decode`; persistent intermediate/output buffers
  per logical B, matmul on the (8,6) grid, RS workers at row offset 6). New `_out_mmrs_decode` in `gdn/tp.py` and
  `attention/tp.py`, gated by `args.fuse_out_mmrs_decode` (shared-code default off).
- `mixer_decode_mode`: "1d" (stage 4) / "ds_in" (DRAM-sharded in-projection weights, `proj_1d_decode` off ->
  `sharded_decode_matmul`) / "ds_all" (+ DRAM-sharded GDN out / attention wo; disables the fused prefill out-proj MMRS
  in GDN because `_out_sharded`). tt-perf-report advice on the in-proj rows ("Try a DRAM-sharded program config").
- `gate_up_output_tile`: `(32, 16)` output tiles for the gather-mode routed gate/up `sparse_matmul` (N = 512 -> 32
  16-wide tiles -> 32 cores); `sparse_matmul_program_config(tile_w=)`, `OptimizationConfig.gather_gate_up_output_tile`.
- `tools/sweep_multichip.py`: families `mmrs`, `mixer_ds`, `gu_tile`, `dense_1d_long` (88/33 and 44/44 twice each) and
  `--replays` (10-20 traced replays instead of 3 to shrink the +-6 us noise).
- `scripts/05-agmm-probe.py`: decode in-projection as all_gather -> 1D matmul vs fused `all_gather_minimal_matmul_async`
  (the qwen36 prefill helper) at M=32, K 768 -> 3072, N 5152 / 4608, traced 20 replays + PCC between the paths. Decides
  whether the per-norm option B (attention norm distributed at decode) is worth wiring.
- `test_perf_prefill_512_traced`: one aligned 512-token chunk captured as a trace and replayed vs eager — quantifies the
  prefill host gap on the default path (all inputs device-resident; a capture failure is reported, not hidden).
- Running: `scripts/05-chain1.sh` (sweeps mmrs, mixer_ds, gu_tile, dense_1d_long -> AGMM probe -> traced prefill chunk).

### chain 1 results (as they land; `sweeps/*.jsonl`, `logs/05-sweep-*.log`)

- `mixer_ds` (3 replays): DRAM-sharded in-projection weights are SLOWER than the tuned 1D interleaved matmuls at the
  per-device widths: GDN 0.582 -> 0.626 ms (`ds_in`) / 0.611 (`ds_all`); attention 0.458 -> 0.485 / 0.471. PCC moves
  in the 6th digit (different kernel), so the knob is live. Matches the qwen36 note that DRAM-sharding the narrow-K
  row-parallel projections loses to the interleaved 1D kernel. REJECTED with before/after; tt-perf-report's
  "try DRAM-sharded" advice does not hold for M=32 rows at N = 4608-5152 on this die.
- `mmrs` (3 replays, before the `--replays` fix): mmrs_on 0.604 vs 0.583 (GDN), 0.476 vs 0.457 (attention) with PCC
  bit-identical to mmrs_off — the identical PCC is suspicious (a different op should move the last digits); the op-site
  probe runs the GDN layer with the knob on to confirm the fused op is actually issued; re-measured with 10 replays in chain 3.
- `gu_tile` (10 replays): `(32,16)` output tile FAILS at op level: `matmul_utilities.cpp:229 out_tile_shape[1] % in1_tile_w
  == 0` — the override output tile must be a multiple of the WEIGHT tile width (32). Retry per the optimize rule (adapt
  the weights): `moe.py` now stores the gate/up expert tensor with `ttnn.Tile([32,16])` when the knob is set (uncached
  `from_torch`, as_tensor has no tile=); re-queued as chain 4.
- `dense_1d_long` (20 replays, each candidate twice): attention qkv 88 / wo 33 cores and 44 / 44 both 0.446 ms vs 0.453 ms
  auto (-7 us, reproducible in both repeats; bit-identical output — the 1D mcast kernel accumulates K in the same order on
  any grid). ADOPT `attn_qkv_decode_cores=88, attn_wo_decode_cores=33` (the wide-first grid, same shaping as the qwen36
  MLP tuning).
- AGMM decode probe (`logs/05-agmm-probe.log`): fused `all_gather_minimal_matmul_async` with the prefill helper's config
  (M_block 4, K_block, N_block 8, (8,9) grid, 2 links) at M=32: 0.153 ms (N=5152) / 0.140 ms (N=4608) vs separate
  all_gather + tuned 1D matmul 0.070 / 0.067 ms — the fused op is 2x slower at decode shapes; PCC 1.0 between the paths.
  Before rejecting the family, chain 5 tries decode-shaped MinimalMatmulConfigs (M_block 1, N_block 1-8, grids (8,9)/(8,5)/(4,9), 1-2 links).
- `test_perf_prefill_512_traced[gdn]`: eager 45.7 ms for one aligned 512-token chunk; trace capture FAILED inside the GDN
  prefill (`fd_mesh_command_queue.cpp:825: !trace_id_.has_value()` — a host boundary (readback/write) inside the chunk).
  The op-site probe (chain 2) now runs a host-boundary census of the same chunk to name the calls; the attention chunk
  result follows.
- 12:38 the attention traced-prefill case stalled 30 min at 100 % host CPU after the GDN capture failure: the failed
  `begin_trace_capture` was never closed, so the mesh command queue stayed in capture mode and the next op blocked
  (`timeout` SIGTERM ignored, SIGKILL needed). Test fixed: the except path ends/releases the capture and skips with the
  PERF line. Killed the chain, `tt-smi -r`, smoke PASS; probes re-queued as one sequential script (`05-chain6.sh`).
- Op-site census (`logs/05-opsite-probe.log`): the aligned 512-token prefill chunk issues 208 (GDN) / 173 (attention)
  python-level ttnn calls and ZERO host boundaries (no from_torch/to_torch), so the GDN capture failure is not a host
  transfer; a traceback probe is queued (chain 7). Decode: 112 (GDN) / 101 (attention) python-level calls (the Tracy
  device op counts 103 / 84 exclude views); with `fuse_out_mmrs_decode` the GDN step issues
  `experimental.matmul_reduce_scatter_async` (111 calls) — the fused op is live.
- `mmrs` (10 replays): fused out-proj+RS 0.601 vs 0.579 ms (GDN, +22 us) and 0.473 vs 0.454 ms (attention, +19 us); the
  fused op uses the helper's generic (8,6) 2D matmul config. Before rejecting: `scripts/05-mmrs-probe.py` (chain 8) times
  the fused op with 1D-mcast configs on grids disjoint from the RS workers against the tuned 1D matmul + separate RS.
- AGMM decode probe 2 (`logs/05-agmm-probe2.log`, N=5152): every decode-shaped `MinimalMatmulConfig` (M_block 1; N_block
  1-8; grids (8,9) / (8,5) / (4,9); 1-2 links) is slower than the separate path: best fused 0.157 ms (4x9 grid, 1 link)
  vs 0.070 ms separate. The fused op's matmul is a minimal-matmul kernel designed for prefill blocks; at M=32 it cannot
  match the tuned 1D mcast kernel, and the 4.6 KB gather it hides costs only ~13 us. REJECTED (family: all-gather+matmul
  fused; per-norm option B at decode would inherit this op). Prefill keeps option A for this model (dim <= 4096 rule).
- `gu_tile` retry with the 16-wide weight tile: now fails `matmul_utilities.cpp:241 out_tile_shape[0] <= FACE_HEIGHT`,
  which fires only when the output tile width differs from the in1 tile width -> the weight tile was still 32 wide.
  Chain 9 probes whether `from_torch(tile=[32,16])` applies to a bfp4 4D tensor with a mesh mapper and whether the
  sparse matmul accepts the pair; the constraints (`matmul_utilities.cpp:225-249`) require out width == in1 width for an
  interleaved output, so the only legal narrow configuration is weight tile [32,16] + output tile [32,16].
- `mmrs` probe (`logs/05-mmrs-probe.log`, decode out-proj [32,2048]x[2048,3072] per device): separate tuned 1D matmul +
  `reduce_scatter_minimal_async` 0.046-0.049 ms; fused `matmul_reduce_scatter_async` with the helper's (8,6) 2D config
  0.068 ms; 1D-mcast configs are refused by the fused op (`matmul_reduce_scatter_async_device_operation.cpp:45`). The fused
  op cannot use the kernel that makes the separate path fast, and the RS it hides is a 12 us latency-bound collective.
  REJECTED (family: matmul fused with reduce-scatter at decode). Prefill keeps the fused fp32 out-proj MMRS (stage 4).
- Traced prefill chunk: attention ALSO fails capture, same site — traceback (`logs/05-gdn-trace-probe.log`):
  `optimized_decoder._routed_sorted` -> `ttnn.zeros(..., device=)` for the sparse-matmul output buffers writes the buffer
  from the host (`fd_mesh_command_queue.cpp:825 !trace_id_.has_value()`); `moe._routing_matrix` has the same pattern.
  Fix: `_zeros_dev()` — a per-shape zero template created once by the untraced compile run, `ttnn.zeros_like` (device op)
  per call. This is the change that makes the MoE prefill traceable for the full-model stage. Re-test queued.
- `gu_tile` probe (`logs/05-gu-tile-probe.log`): `from_torch(tile=[32,16])` DOES apply to the bfp4 4D expert weight with a
  mesh mapper, and the pair weight [32,16] + output [32,16] passes the matmul tile checks (the probe's own sparsity
  layout was wrong, TILE instead of ROW_MAJOR). The sweep failure at :241 came from the PREFILL gate/up sparse matmuls
  (masked / sorted) that still asked for a 32-wide output tile against the narrow weight -> all gate/up calls now take the
  narrow tile and tile_w configs; retry queued.

### 2026-09-24 12:45 — selections and the small-op analysis

- ADOPTED into `MULTICHIP_SELECTED`: `attn_qkv_decode_cores=88`, `attn_wo_decode_cores=33` (-7 us, `dense_1d_long`).
  Kept: shared 16 cores (stage 4). Device-side zero buffers (`_zeros_dev`) replace `ttnn.zeros(device=)` in the MoE
  prefill paths (trace safety; same numerics).
- Op-site map of the GDN decode step (`logs/05-opsite-probe.log`): ops 5-9 five slices of the fused [qkv|z|a|b] row,
  10-18 conv shift register (4 copies + mul + 3 mac + silu), 19-24 three slice+reshape pairs into head rows, 25-29 GVA
  `repeat_interleave` x2 + reshapes, 30-34 a/b gates, 35-38 q/k norms, 39-44 gating + 5 reshapes, 45-54 recurrent step,
  55 state write-back copy, 56-58 out norm. No single op above 8 us; the chain is the stage-3 kernel design. Recorded in
  README section 5 as a backlog item (fused GDN head-split op), not deferred silently: every family from the optimize list
  has a measured row in README section 2.
- Final evidence chain queued (`05-final-chain.sh`): full mesh suite, host walls, Tracy per device, six watcher runs, reset.
- 12:55 traced prefill chunk re-test after the zeros fix: the GDN chunk now CAPTURES (no host boundary left), but the
  first `execute_trace(blocking=True)` of the captured 512-token chunk never returns (host spinning at
  test_multichip_decoder.py:906; py-spy). Killed (SIGKILL), `tt-smi -r`, smoke PASS. A traced MoE prefill chunk therefore
  needs a device-side investigation (candidates: the sorted-dispatch ops — `ttnn.sort` / RM uint32 scatters — or the
  fused fp32 out-proj MMRS persistent buffers replayed without their host-side reset); this belongs to the full-model
  generator stage where chunk-outer tracing is implemented (qwen36 pattern), with the host-gap numbers of this stage
  (eager 45.7 / 41.4 ms per 512 chunk vs ~12 ms device) as the target. The test keeps a 900 s timeout and reports capture
  failures as skips; it is excluded from the -m perf run of the final chain.

### 2026-09-24 13:09 — final chain: suite 35/35; the device-side zeros halve the prefill host wall

- Final mesh suite with the final defaults: 35 passed / 0 failed (`logs/05-mc-final-suite.log`, 12.0 min).
- Host walls (`logs/05-perf-plain.log`, `HOST_WALLS.md`): GDN 15.425 / 124.141 / 0.582 ms, attention 15.179 / 119.436 /
  0.451 ms. prefill_2048 went from 238.7 ms (stage 4) to 124.1 / 119.4 ms -> 2.02x / 1.92x vs one die: the `ttnn.zeros(device=)`
  buffers in the MoE prefill were blocking host->device writes of 74 + 12 MB per 512-token chunk (G=377 groups x 32 rows
  x 3072 / 512 bf16) — the "host dispatch" gap of stage 4 was mostly these writes. Decode 0.582 / 0.451 ms (attention -7 us
  from the 88/33 grids; GDN unchanged, noise).
- 13:11-13:50 Tracy step of the final chain: gdn prefill_128 / prefill_2048 / decode and attn prefill_128 collected; the
  attn prefill_2048 case blocked right after the `PERF_PREFILL_MESH` signpost (host asleep in a device wait, 0 % CPU, 36
  min; the same case passed under Tracy in stage 4 and passes without the profiler at 119 ms). Killed by PID (a `pkill -f`
  on the script name killed the operator shell instead — twice now; PID-only from here), `tt-smi -r`, smoke PASS.
  `05-final-chain2.sh`: retry attn prefill_2048 + attn decode under Tracy, then tables, then the six watcher runs.
- 13:50-13:53 Tracy retry: attn prefill_2048 collected on the first retry (486.6 ms profiled wall; the earlier block happened on the first
  attempt after the GDN Tracy series and cleared after a reset — a profiler-state pattern, see 19:05), attn decode 0.505 ms profiled. Per-device device time (`PERF_TABLES.md`): GDN 8.205 / 51.794 / 0.473 ms,
  attention 9.042 / 49.035 / 0.411 ms. Attention decode -8 us vs stage 4 (qkv matmul 54 -> 45 us on the 72-core grid the
  88-core request resolves to); prefill +1.8 ms device per 2048 tokens for the on-device zero fills (host wall -115 ms).
  Roofline accounting in README section 4: 39.2 / 36.5 MB per step per device -> 76.7 / 71.3 us bound; device 472 / 407 us
  (16 / 18 %); wall 582 / 451 us; op-count bound with four named ttnn limitations.
- 14:01 watcher: 6/6 PASSED, 0 bad watcher.log lines, all six processes ended in the eth-timeout teardown form (none of the
  NOC-state form this time); devices reset and smoke-tested. README complete; stage review next.

### 2026-09-24 14:05–14:25 — stage review #1: more-work-needed; fixes and re-measurements

Required work (reviewer, read-only, re-derived every number; kernel/PCC/perf tables confirmed):
- P1 narrow gate/up tile never reached the op: the retry stored 16-wide weight tiles but the masked and sorted gate/up calls
  still asked for 32-wide outputs (`matmul_utilities.cpp:241`) and the probe's own sparsity layout was wrong. Now every
  gate/up `sparse_matmul` (gather, masked, sorted) takes `tile_gu` with `tile_w` configs, the sorted path drops
  `optional_output_tensor` when narrow (the op refuses the pair) and the DOWN weight is stored with 16-high tiles
  (its in0 gets 16-wide K tiles). `gu_tile` re-runs with 10 replays (chain `05-review-chain.sh`).
- P1 mixer policy: the TP mixers run HiFi2 (qwen36 default) where stage 3 selected LoFi, and the DS candidate was measured
  only through `sharded_decode_matmul`'s reshards at 3 replays. New knobs `mixer_fidelity` (hifi2|lofi ->
  `args.mixer_compute_cfg`, read by attention/gdn tp), `attn_qkv_in0` / `gdn_qkvz_in0` (explicit in0_block_w on the 1D
  kernel, `create_matmul_1d_decode_progcfg(in0_block_w=)`); families `mixer_fidelity`, `mixer_in0`; Tracy of the ds_in
  candidate via the test override `QWEN35_MC_OVERRIDE` for the op breakdown.
- P1 `_zeros_dev` template cache (unbounded, ~261 MB per layer for the T=512+2048 shapes -> ~12.5 GB per device on 48
  layers; contract text wrong): replaced by `_alloc_dev` — `ttnn.empty` for the sparse-matmul outputs (every real group is
  written by the op, padded groups are never read by `undispatch_rows`; also removes the 1.75 ms/2048-token device fill)
  and `ttnn.fill` for the routing-matrix scatter target; nothing cached. Contract block corrected.
- P2 fused matmul+RS strawman: `05-mmrs-probe2.py` tries the widest legal single-tile-row 2D grids (the op only accepts
  2D configs and M is one tile, so at most one grid row of cores works — the cap the probe records).
- P2 persistent CCL buffers: `05-ccl-persistent-probe.py` times all_gather_async / reduce_scatter_minimal_async with and
  without persistent buffers at the decode payload.
- Other: the "88/33" wording -> qkv 72 cores (88 requested; wo 33 = qwen36 default), README labels of the two sparse
  buffers swapped, PERF_TABLES label, `perf_probe` excluded explicitly (`-m "... and not perf_probe"` + env skip),
  trace-replay hang bisect (`05-trace-bisect.py`: moe / mixer / layer under a hard timeout with reset).

### 2026-09-24 16:11– — review-#1 re-measurements (`05-review-chain.sh`, 10 replays)

- `gu_tile` retry 2 still failed at `matmul_utilities.cpp:241`: the FUSED decoder class overrides `_routed_masked`
  (`fused_decoder.py:183`) and still requested a 32-wide output tile against the narrow weight (the base-class patch was
  not the live path). Patched (`output_tile=self.tile_gu`, `tile_w` config); retry 3 queued after the chain.
- `mixer_fidelity`: LoFi vs HiFi2 on the TP mixers is a wash in time (GDN 0.579 / 0.579 ms, attention 0.446 / 0.446 ms —
  the rows are DRAM-bound) and costs PCC (prefill 0.99992 -> 0.99960 GDN, 0.99985 -> 0.99927 attention; decode 0.99996 ->
  0.99987 / 0.99993 -> 0.99983). KEEP HiFi2 (measured, not deferred).
- `mixer_in0` (in0_block_w 12/16/24/32/48 vs the helper's largest divisor 8 on K=96 tiles, both in-projections): no gain;
  monotonically slower with larger blocks (GDN 0.579 -> 0.585 ms at 48, attention 0.446 -> 0.449). KEEP the default.
- DS candidate Tracy (`tracy_ds_in/`, device 0, one replay): with `mixer_decode_mode=ds_in` the in-projection runs as the
  DRAM-sharded kernel on 8 cores (one reader per DRAM bank on Blackhole) — GDN 68.4 us at 45 % DRAM vs the 1D kernel's
  50.5 us on 41 cores at 61 %; attention 58.4 us (8 cores, 47 %) vs 45.1 us (72 cores, 63 %). The reshard cost is NOT the
  reason (5 / 12 reshard ops = 6.6 / 11.9 us in both variants; the norm output already sits in the width-sharded L1
  layout the DS kernel wants). Side effect: `proj_1d_decode=False` also drops the tuned 1D out-projection (23 -> 47 us on a
  96-core auto config). Step totals 513.9 vs 471.5 us (GDN), 447.7 vs 405.9 us (attention). REJECTED with the op-level
  cause: at N = 4608-5152 per device the 8-bank DS kernel cannot beat a 41-72-core 1D mcast read.
- `mmrs` probe 2 (`logs/05-mmrs-probe2.log`): widest legal single-tile-row 2D grids for `matmul_reduce_scatter_async`
  ([32,2048]x[2048,3072]): 8x1 in0 8 sub_w 2 = 0.063 ms, 6x1 = 0.070, 4x1 = 0.087 (8x1 with sub_w 4 refused:
  out_block_w % out_subblock_w; 11x1 not a divisor of 96 N tiles) vs separate tuned 1D (44 cores) + RS 0.047 ms. The
  op-contract cap: M is one tile, the fused op takes 2D configs only, so at most one grid row (<= 8 cores) computes
  where the separate path uses 44. REJECTED with the adapted attempts.
- CCL persistent buffers (`logs/05-ccl-persistent-probe.log`, decode payloads, 20 traced replays): all_gather_async
  22.8 -> 20.1 us with a persistent output buffer (-2.7 us), reduce_scatter_minimal_async 22.2 -> 22.6 us (no gain).
  Two norm all-gathers per layer -> -5 us per layer step (~1 %); the gathers live in the framework `DistributedNorm`
  (`models/tt_transformers/tt/distributed_norm.py`, no persistent-buffer option today), so the layer-level change is a
  framework edit; measured and carried to the full-model optimize stage where 96 gathers per token make it -0.26 ms.
- Trace-replay bisect (`logs/05-trace-bisect-*.log`): the MoE block alone (expert-sorted dispatch + sparse matmuls +
  reduce-scatter, T=512) captures and replays (12.0 ms per replay, 4 replays OK) — the sorted-dispatch ops are NOT the
  hang; the GDN mixer `forward_prefill(capture_state=False)` also replays (1.3 ms). Remaining suspects: the layer's
  `_stable_state` write-back (`capture_state=True`, in-place state copies) and the qwen36 layer composition (norm
  all-gathers) — `mixer_state` and `layerfwd` variants queued (`05-bisect2.sh`); the whole-layer case is running.
- Bisect part 2: `mixer_state` (GDN `forward_prefill(capture_state=True)`, the layer's in-place state write-back) also
  replays (1.4 ms x4 OK); the whole layer (`prefill_forward`) hangs at replay 1 again (exit 124, reset). Suspect now: the
  composition — norm all-gathers + MMRS + MoE RS in one trace (CCL semaphore cycling) — `05-bisect3.sh` isolates
  norms / norm+mixer / norm+MoE / mixer+MoE; `layerfwd` (qwen36 layer forward without the wrapper) is running.
- Bisect part 3 (`logs/05-trace-bisect-{norm_mixer,norm_moe}.log`): norm all-gather + GDN mixer (state write-back on)
  replays (1.5 ms x4), norm all-gather + MoE replays (11.3 ms x4); the qwen36 `layer.forward` (`layerfwd`) hangs like the
  wrapper. So the hang needs the mixer AND the MoE in one trace (the composition), not any block alone. Part 4 queued:
  `mixer_norm_moe` (norm -> mixer -> residual add -> ffn norm -> MoE, i.e. the layer minus its last residual add) and
  `mixer_moe` (probe fixed: the fp32 mixer partial is cast to bf16 before the MoE). Hypothesis under test: two
  reduce-scatters in one trace — the fused fp32 `matmul_reduce_scatter_async` (mixer out-proj) and the MoE's
  `reduce_scatter_minimal_async` — take the two alternating `TT_CCL` RS semaphore sets / barrier semaphores; if the MMRS
  leaves its (shared) barrier or RS semaphores at a non-zero value, the second op's replay waits forever.
- `gu_tile` retries 3/4: with every gate/up call on the narrow tile the sparse matmuls now RUN (the matmul tile checks
  pass); the path fails at `slice_device_operation.cpp:203` "slice does not currently support tiles other than 32x32";
  routing the decode gather path through `ttnn.swiglu` did not remove it (another slice on the narrow tensor). Retry 5
  records the failing frames; the exact op contract is then documented as the blocker.
- `gu_tile` retry 5 (frames in `sweeps/gu_tile.jsonl`): the failing op is the gate/up split after the narrow-tile sparse
  matmul — `fused_decoder.py:199` `ttnn.swiglu` (a slice + silu*mul composite) -> `slice_device_operation.cpp:203` "slice
  does not currently support tiles other than 32x32, got 32x16". The sparse matmuls themselves run on the narrow tiles.
  Final status: REJECTED as an exact op-contract blocker (no ttnn op splits a narrow-tiled tensor); README section 2/5.
- Bisect part 4 (`logs/05-trace-bisect-mixer_moe.log`): GDN mixer prefill (fused fp32 `matmul_reduce_scatter_async` out-proj)
  followed by the MoE prefill (its `reduce_scatter_minimal_async`) in ONE trace hangs at the first replay, while each block
  alone, norm+mixer and norm+MoE replay. The same pairing explains the stage-6 finding that an EAGER prefill hangs after a
  TRACED decode (eager all_gather / reduce_scatter / norm after the decode all pass, the full prefill hangs): the fused op
  is the one prefill-only collective. Candidate fix: `gdn_prefill_mmrs=False` (new MultichipConfig knob, plain matmul +
  `tt_all_reduce` for the GDN prefill out-projection); `06-mmrs-off-chain.sh` re-runs the `mixer_moe` / `layer` bisects
  and the prefill-after-decode probe with it.
- `mixer_norm_moe` (norm -> mixer -> residual add -> ffn norm -> MoE): the FIRST replay completes (12.7 ms), the second
  replay never returns — state left by the fused MMRS + MoE RS pair carries across replays (semaphore/barrier residue),
  consistent with the eager-prefill-after-traced-decode hang. Waiting for the `gdn_prefill_mmrs=False` runs.
- `gdn_prefill_mmrs=False` (`logs/05-trace-bisect-mixer_moe-mmrsoff.log`, `-layer-mmrsoff.log`): mixer + MoE in one trace
  now REPLAYS (12.6 ms x4), the whole layer still hangs at replay 1. So the fused fp32 MMRS is one factor; a second one
  lives in the layer composition with the two `DistributedNorm` all-gathers (they share the TT_CCL barrier pool with the
  reduce-scatters: AG B0, RS B1, AG B0, RS B1 — the same pattern the decode trace replays fine, so shapes/programs of the
  PREFILL collectives matter). Not resolved in this stage: the default keeps the fused MMRS (fastest measured eager
  prefill); traced prefill is the full-model optimize stage's item with this bisect as the starting point.
- 18:09-19:05 Final chain on the final code (`06-smoke-then-05final2.sh` -> `05-final-chain.sh`): final mesh suite 35 passed /
  0 failed (12:11), host walls regenerated (`HOST_WALLS.md`: prefill 2048 122.697 / 117.939 ms, traced decode 0.582 / 0.450
  ms), Tracy GDN x3 + attention prefill_128 collected; the attention prefill_2048 Tracy case blocked again at its first
  readback (`_compose` -> `to_torch`, 18:26 -> killed 19:04; third occurrence, always this case, only under the profiler; the
  plain run passes in the suite and in the host walls). Recovery `05-final-chain3.sh`: reset, the two remaining attention
  Tracy cases with a retry + reset per case, `PERF_TABLES.md`, watcher x6, mesh smoke.
- 19:05-19:08 Recovery chain: reset, attention prefill_2048 Tracy collected on the first retry (493.0 ms profiled wall),
  attention decode 0.506 ms profiled. `PERF_TABLES.md` regenerated from the final code: GDN 8.199 / 49.938 / 0.473 ms,
  attention 9.035 / 47.193 / 0.408 ms per device — prefill device time back to the stage-4 values (the `_alloc_dev`
  buffers are allocations, not fills), attention decode -11 us. README section 4 updated; device-0 prefill_2048 CSVs
  extracted (`*_dev0.csv`, the per-device reports exceed the 500 KB commit hook). Watcher x6 running.
- 19:25-19:45 Stage-5 review #2 = more-work-needed (4 required items). Work started: (1) Blackhole multi-reader DRAM-sharded
  candidates — `tp_common.create_dram_sharded_matmul_program_config(readers=)` sets `num_workers_per_dram_bank` when the
  per-bank N shard splits evenly; `MultichipConfig.mixer_ds_readers` (1..3) rebuilds the four mixer DS progcfgs;
  `mixer_ds_out_1d` decouples the out-projection (qwen36 `_wo_proj`/`_row_proj` honour `args.out_proj_1d_decode`); sweep
  family `mixer_ds_readers` (1d, ds_in+1D-out r=1/2/3, ds_all r=2/3, 10 replays). (2) `test_mc_prefill_after_traced_decode`
  (both kinds): prefill -> traced decode x4 -> release -> reset -> eager aligned + chunked prefill vs HF -> traced decode
  again. (3) `05-gu-tile-probe.py` fixed (ROW_MAJOR sparsity) and timed eager + traced at 16 vs 32 cores. (4) CCL probe
  extended: bfp8 reduce-scatter / all-gather payloads, bfp8-vs-bf16 numerics, fused `all_reduce_async` vs RS+AG. Also:
  persistent norm all-gather output buffers in-model (`DistributedNorm._persistent_ag_buffer`, `norm_ag_persistent` knob,
  family `norm_ag_persistent`), `PERF_CONFIG` line with the effective `MultichipConfig` in every PERF measurement, README
  wording (72 of 88 active cores / 32 of 33, 7th-digit PCC note, residual-layout reasoning, one-run dispatch accounting:
  device 473/408 + gaps 129/85 = 602/493 us vs profiled wall 614/506 us). Device runs queued behind the stage-6 chain
  (`05-review2-chain.sh`, `05-review2b-chain.sh`). The Tracy attention prefill_2048 block (13:11, 18:26): both times the
  first attempt after the GDN Tracy series, cleared by a reset — profiler-state related, not random.
- 19:58-20:04 Review-2 device runs (`05-review2-chain.sh`): (a) `mixer_ds_readers` — ds_in + 1D out r=1: GDN 0.599 /
  attention 0.461 ms vs 1D 0.579 / 0.446; r=2 on GDN = r=1 (21 tiles/bank, falls back); every r>1 config that reaches the
  kernel dies in `matmul_utilities.cpp:446` -> `get_worker_noc_hop_distance` "only supported on unit MeshDevice"
  (`device.cpp:20`): the Blackhole multi-reader DRAM-sharded matmul is unit-mesh only in this checkout. (b) gu-tile op
  timing (ROW_MAJOR sparsity fixed): 32x32/16 cores 65.7 us eager, 74.3 traced; 32x16/32 cores 64.6 / 73.6 us — no gain,
  rejection now measured. (c) CCL probe: AG persistent 24.3 -> 20.3 us, RS persistent none (22.2 -> 22.5); bfp8 payload RS
  20.8 us (-1.4), AG 21.7 us (-2.6); fused `all_reduce_async` 38.4 us vs RS+AG 35.9 us (slower). The probe's "bfp8 vs bf16
  numerics" line is INVALID (it compared two different random inputs) — numerics come from the in-model PCC of the
  `out_dtype` family instead. New knob `out_proj_decode_dtype=bfp8` (tp_common `matmul_1d_decode(dtype=)`), families
  `out_dtype` and `norm_ag_persistent` queued (10 replays, PCC vs HF). Regression test run 1: eager prefill after traced
  decode PASSES for both kinds (GDN 0.9994 / 0.9994, attention 0.99992); the two failures were test bugs (GDN state not
  rebuilt after the compile pass before the second capture; HF chunk reference) — fixed, rerun queued.
- 20:41 `test_mc_prefill_after_traced_decode` rerun (`logs/05-r2-regression2.log`): PASS for both kinds — GDN prefill 0.99975,
  eager T=128 after traced decode 0.99941, chunk start=128 0.99936, traced decode pos=228 after the eager prefill 0.99948;
  attention 0.99988 / 0.99992 / 0.99996 / 0.99996. The layer-level sequence prefill -> traced decode x4 -> release -> reset
  -> eager prefill -> traced decode is clean; the stage-6 hang was the stack-level ordering hazard (README section 3).
  Final mesh suite rerun on the review-2 code queued (`05-final-suite2.sh`, 37 tests).
- 20:44 `out_dtype` family (`sweeps/out_dtype.jsonl`, 10 replays): bfp8 out-projection partials into the decode reduce-scatter
  — GDN 0.580 vs 0.579 ms (PCC 0.99996 -> 0.99989), attention 0.444 vs 0.446 (PCC unchanged to 6 digits). No wall-clock
  gain (latency-bound collectives), a PCC cost on GDN -> rejected; the CCL dtype family is now measured in-model.
- 20:47 `norm_ag_persistent` rerun with the sharded persistent buffer (`logs/05-r2-sweep-norm-ag2.log`): GDN 0.574 vs 0.579,
  attention 0.441 vs 0.446 ms (10 replays), PCC identical -> ADOPTED in `MULTICHIP_SELECTED`. The first attempt
  (`sweeps/norm_ag_persistent_run1_noop.jsonl`) skipped sharded output configs (the decode gathers on the 1D mesh take the
  non-distributed branch whose output is the L1 width-sharded norm layout) and was a no-op — kept as the negative control.
  Decode-side default-path evidence to regenerate: host walls, Tracy decode rows, PERF_TABLES, watcher traced GDN decode
  (`05-adopt-normag.sh`, queued after the final suite rerun, which itself runs on the adopted default).
- 20:48 Adopting the knob left `MULTICHIP_SELECTED = MultichipConfig(,` (syntax error) for ~60 s; fixed before any queued
  process imported the module (the watcher repro was still in its `tt-smi -r`; verified by process list). `MULTICHIP_SELECTED`
  now = defaults + `norm_ag_persistent=True` (import check: attn_qkv_decode_cores 88, mixer_ds_readers 1, out dtype bf16).
- 21:48 Final suite 3 (uint16-sort code, `norm_ag_persistent=True`): 36 passed, 1 FAILED — `test_mc_traced_decode[attn]`:
  "traced decode replay 2 differs bit-wise from replay 0" (low-bit differences; GDN replays identical). The persistent
  all-gather output buffer is the only decode-path change since suite 2 (suite 2 was killed before completing so that it
  would include the uint16-sort change; it has no summary line) -> the knob is NOT adopted (default reverted, `MULTICHIP_SELECTED = MultichipConfig()`); README row
  rewritten; Tracy decode rows to be regenerated on the reverted default (`05-tracy-decode-rerun.sh`, queued after host
  walls 4 + suite 4). Note for stage 7: check `all_gather_async(persistent_output_buffer=...)` on the L1 width-sharded
  layout for a completion race before reusing it.
- 21:53 Host walls 4 (`logs/05-perf-plain4.log`, counting-sort dispatch, `norm_ag_persistent` reverted): prefill 128 15.5 /
  15.1 ms, prefill 2048 133.0 / 128.2 ms (fp32-sort baseline 122.7 / 117.9; uint16 sort 139.9 / 135.5), traced decode
  0.583 / 0.449 ms (unchanged). The counting sort costs ~2.6 ms per 512-token chunk more than the (deadlocking) fp32 sort;
  a Tracy prefill_2048 run is queued to attribute it (cumsum over the [Rp, E] one-hot vs the hi/lo selections).
- 21:56 Final mesh suite 4 (`logs/05-mc-final-suite4.log`, counting-sort dispatch, `norm_ag_persistent` reverted): 40 passed /
  0 failed (13:27) — the 35 stage-4 tests, `test_mc_prefill_after_traced_decode` x2, `test_mc_dispatch_counting_matches_sort`
  x3 (T=512: G=377, 4096 real rows; T=1000: G=499; T=2048: G=761 — counting == sort_fp32 dispatch). Traced decode replays
  bit-identical again for both kinds. Host walls 4 are the final wall numbers; Tracy decode rows are being regenerated.
- 22:11 Tracy decode rows regenerated on the committed default (`logs/05-tracy-decode-rerun-chain.log`): GDN 0.472 ms
  (103 ops, gaps 131 us, profiled wall 0.614), attention 0.406 ms (84 ops, gaps 87 us, profiled wall 0.508) per device;
  PERF_TABLES.md and README section 4 (device table + accounting) updated from these CSVs. Tracy prefill_2048 rerun for
  the counting-sort op attribution running next.
- 22:14 Tracy GDN prefill_2048 with the (row-major) counting sort (`tracy/gdn/prefill_2048_*`, device 0): 60.98 ms device per
  2048 tokens vs 49.94 with the fp32 sort: `AccumulationDeviceOperation` (cumsum) 8x = 9.33 ms and `PermuteDeviceOperation`
  16x = 5.34 ms — the dim-2 scan over the `[Rp, E]` one-hot permutes in and out and scans slowly (~2.7 ms per 512-token
  chunk, matching the +2.6 ms host wall). Rewritten as the TRANSPOSED counting sort (default `counting`): one-hot
  `[1,1,E,Rp]`, cumsum along the last dim, picks as H-reductions; the row-major variant stays as `counting_h`. The unit
  test now checks counting == counting_h == sort_fp32; validation (`05-counting-check.sh`: unit test + host walls 5)
  queued ahead of the 48-layer chain.
- 22:18 Transposed counting sort validated (unit test: counting == counting_h == sort_fp32 at T=512/1000/2048) but SLOWER:
  host walls 5 prefill 2048 140.8 / 136.2 ms (`logs/05-perf-plain5.log`) vs 133.0 / 128.2 for the row-major variant ->
  default set back to `counting_h` (the row-major counting sort validated by suite 4); host walls 4 stay the final wall
  numbers. Final suite 5 + Tracy prefill_2048 rerun on the committed default queued (`05-final5.sh`). Stage-7 item: the
  rank via per-32-row-block tril matmuls + one block-prefix matmul instead of the scan (all fast tile ops).
- 22:21 Final mesh suite 5 (`logs/05-mc-final-suite5.log`; the chain label says "transposed" but the module default had been
  set back to `counting_h` before the process started — the committed default): 40 passed / 0 failed (13:18). Tracy
  prefill_2048 rows on the same default running next; the section-4 prefill device-time rows will be updated from them
  (the fp32-sort rows 49.938 / 47.193 ms are superseded).
- 22:36-22:38 Final Tracy prefill_2048 rows on the committed default (`logs/05-final5-chain.log`): GDN 60.971 / attention
  58.231 ms per device (924 / 744 ops); `PERF_TABLES.md` regenerated, `*_dev0.csv` re-extracted. `scripts/05-final5.sh`
  line 9 had regenerated `HOST_WALLS.md` from host walls 5 (the transposed variant, 140.8 / 136.2); it was regenerated
  again from host walls 4 (`05-perf-plain4.log`, the committed default) at 22:21 and the README host-wall table re-synced
  — HOST_WALLS.md == plain4 PERF lines. The chain labels "(transposed counting sort)" in `05-counting-check.sh` /
  `05-final5.sh` predate the default flip; both suite 5 and the Tracy prefill rows ran `counting_h` (module default at
  process start, echoed by the dispatch test from review 3 on).
- 22:55 Review #3 = more-work-needed (2 P1: watcher runs on the counting-sort prefill path; sentinel-branch coverage; P2
  wording/provenance). Fixes: `test_mc_dispatch_counting_matches_sort` gains T = 513 and 1023 (T*k % 32 != 0, the
  Rp != R sentinel branch) and echoes `DISPATCH_MODE`; watcher runs 1, 2, 4, 5 re-queued on the committed code
  (`05-r3-chain.sh`); README section-4 provenance sentence, the removed-op accounting (~3.6 ms, not ~1 ms), the PCC
  digit note, the precision-row citation, the stale `counting` comment, `perf_tables_omc.py` docstring and the commit
  message (72/88 and 32/33 active cores, 16 / 18 %, the counting-sort statement) corrected.
- 22:58-23:04 Review-3 device runs (`logs/05-r3-chain.log`): dispatch test 5/5 (T = 512 / 513 / 1000 / 1023 / 2048; 513 and
  1023 carry 24 / 8 sentinel rows; module default echoed as `counting_h`). Watcher on the committed code: real-weight GDN
  and attention T=37 PASSED, 2503 lines, 0 bad; chunked GDN prefill to 4429 tokens (the counting-sort path) PASSED
  (PCC 0.9984 / decode after 0.9983), 3996 lines, 0 bad, then the known form-(ii) teardown trip (`fabric_erisc_router`,
  eth core 29-25 "invalid NOC command buffer state", `doc/multichip_decoder/watcher/TALLY.md`) AFTER "1 passed". The
  chain's slot counter was clobbered by its reset loop (the chunked-GDN run wrote slot 2): slots 2, 4, 5 are rerun by
  `05-r3b-chain.sh` with a fixed script; slots 1 and 2 (real-weight) results above stand on their chain-log lines.
- 23:07-23:11 Corrected watcher slots (`logs/05-r3b-chain.log`): real-weight attention T=37 (2) PASSED 2503 lines 0 bad;
  chunked prefill to 4429 tokens GDN (4) PASSED 3758 lines 0 bad; attention (5) PASSED 3758 lines 0 bad — all on the
  committed counting-sort code, each ending in the known fabric-router teardown form after "1 passed". `watcher/
  pytest_{2,4,5}.txt` and `artifacts/05-watcher-{2,4,5}/` regenerated; slots 1 (23:00) and 3 / 6 (19:05 chain, decode
  paths unchanged since) stand.
- 23:25 Review #4 = more-work-needed (doc-only). Provenance: (a) the r3 watcher runs (22:59-23:11) executed with the worktree
  copy of `tt/fused_decoder.py` already carrying the UNSTAGED stage-6 `_share_gate_buffers` hunk (written 22:52) — a no-op
  for the single gate instance of a one-layer test (the first instance only registers its buffers), so the evidence holds
  for the index, but index and tested tree differ by that hunk; (b) the 23:02 slot-2 run (chunked GDN, form-(ii) teardown
  trip after "1 passed" per the chain-log timing) had its pytest.log overwritten by the r3b rerun — slot 4 (23:08, PASSED,
  3758 lines, 0 bad, form (i)) is the surviving clean run; (c) README section 3 now cites `05-r2-regression2.log` (the
  passing run), section 4 counts the committed suite as 42 tests (40 in suite 5 + the two sentinel parameters from
  `05-r3-dispatch-test.log`), the fp32-sort host-wall row is labelled with its source (`05-perf-plain.log`, 18:22), and
  the context-contract prefill note carries the committed-path numbers. Sentinel rows' out-of-range source token (inert
  OOB read, pre-existing) recorded in section 5 with the one-line fix scheduled for the next dispatch change.
- 23:33 Review #5 (re-review of the index after the review-4 doc fixes) = clean-pass. Local commit of the INDEX follows
  (the unstaged `_share_gate_buffers` hunk in `tt/fused_decoder.py` belongs to stage 6).
