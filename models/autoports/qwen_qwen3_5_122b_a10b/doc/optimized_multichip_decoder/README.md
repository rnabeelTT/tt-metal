# Optimized multichip decoder — Qwen/Qwen3.5-122B-A10B on the (1,4) Blackhole mesh (stage 5)

Optimizes `tt/multichip_decoder.py` (`MultichipDecoder` / `MULTICHIP_SELECTED`) in place. Baseline = stage 4 commit
1b9770ef8e6. Evidence harness unchanged: `tests/test_multichip_decoder.py` (`-m perf` host walls, `-m perf_single` one-die
baseline), `scripts/04-perf.sh` (Tracy per device, `tools/perf_tables_mc.py`), `tools/sweep_multichip.py` (families).

## 1. Operation-topology audit (stage-4 default path, device 0, one traced decode replay)

Source: `doc/multichip_decoder/tracy/{gdn,attn}/decode_mesh_perf_report.csv` (device 0 rows, chronological).

| stage of the step | GDN layer (L0): ops / us | attention layer (L3): ops / us | material items | candidate family |
|---|---|---|---|---|
| attention-norm all-gather + RMSNorm + reshard | 3 / 20 | 3 / 19 | `AllGatherAsync` 13 us on 4.6 KB (latency bound); norm on 32 cores | fused all-gather+matmul (`all_gather_minimal_matmul_async`) at decode = per-norm option B; probe |
| in-projection (column-parallel, replicated input) | 1 / 51 (`[32,3072]x[3072,5152]` bfp8, 41 cores, 61 % DRAM) | 1 / 53 (`x[3072,4608]`, 48 cores) | one matmul reads 16.8 / 15.0 MB | DRAM-sharded weights + sharded activation (`mixer_decode_mode=ds_in`), tt-perf-report advice |
| projection split + layouts (slices, copies, tilize/untilize, reshapes) | ~30 / ~75 | ~14 / ~30 | GDN head-row layouts, conv shift register (1 binary + 3 ternary + 1 unary), 7 `ReshapeView` (2 x 8 us) | shared-kernel data movement; reshape probe |
| mixer core (GDN recurrent step / attention RoPE+KV+SDPA) | ~20 / ~60 | ~22 / ~50 | 2 small batched matmuls; SDPA 13 us; paged update 2 x 3 us | inherited from stage 3 (already rewritten) |
| out-projection + reduce-scatter + residual add | 3 / 37 | 3 / 36 | `[32,2048]x[2048,3072]` 22 us + `ReduceScatterMinimalAsync` 12.5 us + add | fused matmul+reduce-scatter (`matmul_reduce_scatter_async`, `fuse_out_mmrs_decode`) |
| ffn-norm all-gather + RMSNorm + reshard | 3 / 19 | 3 / 19 | same as the first row | option B does not apply (the MoE router/experts need the replicated activation); keep |
| router (matmul + gate op + reshards) | 13 / 21 | 13 / 21 | `[32,3072]x[3072,256]` 9 us, `GeneralizedMoeGate` 1.4 us, 4 slices / 2 I2S-S2I pairs / reshape 5 us | stage-3 router contract (gate op needs height-sharded rows); keep |
| routed experts (gather-mode sparse gate/up + swiglu + down) | 6 / 90 | 6 / 89 | gate/up `sparse_matmul` 62 us on 16 cores = 114 GB/s (N = 512 -> 16 tiles cap the grid); down 18 us | `gate_up_output_tile (32,16)` -> 32 cores probe; in0_block_w swept in stage 4 |
| expert scoring + reduce over k (mul, fillpad, fast_reduce_nc, slices) | 6 / 23 | 6 / 23 | `FillPad` 7 us + `BinaryNg` 6.4 us + reduce 3 us | stage-3 contract; keep |
| shared expert (packed gate/up + silu*mul + down) | 8 / 35 | 8 / 35 | 16-core 1D configs (stage-4 selection) | keep |
| MoE reduce-scatter + residual add | 2 / 14 | 2 / 15 | `ReduceScatterMinimalAsync` 11.8 us on 4.6 KB | residual stays fractured (lowest-movement contract); keep |
| total | 103 / 476 | 84 / 417 | matmuls 182 / 166 us, collectives 50 / 49 us, ~90 / ~70 small ops 245 / 200 us | |

Residual layout contract between layers: FRACTURED `[1,1,B,768]` bf16 per device (stage-4 contract; one reduce-scatter per
row-parallel boundary and one all-gather per norm — no gather/reshard/all-reduce between layers). The replicated-residual
alternative (an all-reduce at each row-parallel boundary, the next norm then runs replicated without a gather) moves the
same bytes through the same four collective phases per layer (RS+AG, RS+AG vs RS, AG, RS, AG) — the difference is launch
count, not payload: a fused `all_reduce_async` would be 2 launches per layer instead of 4 latency-bound (12-13 us) ops.
Measured at the decode payload (`logs/05-r2-ccl-probe.log`): `all_reduce_async` 38.4 us vs reduce-scatter + all-gather
35.9 us, so the replicated alternative has no lever left; the fractured contract stays.

Prefill (2048 tokens as 4 x 512 chunks): device time 50.0 / 47.2 ms per device (2.5x / 1.9x vs one die) but host wall 239 ms
at the start of this stage. Section 3: half of that wall was host->device writes of zero buffers inside the MoE prefill
(fixed here: 124 / 119 ms); the rest is host dispatch of ~950 / 770 mesh ops per chunk.

## 2. Families measured (before/after)

All rows: `tools/sweep_multichip.py` on the (1,4) mesh, real weights, PCC vs HF identical to the baseline unless noted, decode
= traced replays (count in the table), noise about +-6 us at 3 replays and +-2 us at 10-20 (`sweeps/*.jsonl`, `logs/05-*.log`).
Baseline = stage-4 `MULTICHIP_SELECTED` (GDN 0.579-0.583 ms, attention 0.453-0.458 ms host wall per traced step).

| family (skill list) | candidate | GDN decode ms | attention decode ms | evidence | decision |
|---|---|---|---|---|---|
| residual layout | fractured `[1,1,B,768]` (stage 4) vs replicated | – | – | same payload, same four collective phases per layer; the only lever is a fused `all_reduce_async` (2 launches instead of 4): measured at the decode payload (`logs/05-r2-ccl-probe.log`, 20 traced replays): `all_reduce_async` 38.4 us vs reduce-scatter + all-gather 35.9 us — the fused op is slower on this ring | keep fractured (measured; contract in section 4) |
| collective placement / fused CCL+matmul | out-proj `matmul_reduce_scatter_async` (`fuse_out_mmrs_decode`) | 0.601 vs 0.579 (10 replays) | 0.473 vs 0.454 | in-model op is live (op-site census); op-level probes `logs/05-mmrs-probe*.log`: helper (8,6) config 0.068 ms, widest legal single-tile-row 2D grids 8x1 / 6x1 / 4x1 = 0.063 / 0.070 / 0.087 ms vs separate tuned 1D (44 cores) + RS 0.047 ms; the op accepts 2D configs only and M is one tile, so at most one grid row (<= 8 cores) computes; 1D configs refused (`device_operation.cpp:45`) | REJECTED (adapted attempts recorded) |
| all-gather fused into the in-proj matmul (decode, per-norm option B) | `all_gather_minimal_matmul_async` M=32 | – | – | `logs/05-agmm-probe*.log`: 0.140-0.598 ms for every decode-shaped MinimalMatmulConfig vs 0.067-0.070 ms all_gather + tuned 1D matmul | REJECTED (op designed for prefill blocks) |
| DRAM-sharded decode matmuls | `mixer_decode_mode=ds_in` / `ds_all` | 0.626 / 0.611 vs 0.582 (3 replays); Tracy 513.9 vs 471.5 us | 0.485 / 0.471 vs 0.458; Tracy 447.7 vs 405.9 us | `sweeps/mixer_ds.jsonl` + Tracy of the candidate (`tracy_ds_in/`): the DS kernel runs on 8 cores (one reader per DRAM bank on Blackhole) at 45-47 % DRAM = 68.4 / 58.4 us vs the 1D kernel 50.5 us on 41 cores (61 %) / 45.1 us on 72 cores (63 %); reshards are identical in both variants (6.6 / 11.9 us), so the layout is not the cause; `proj_1d_decode=False` also loses the tuned 1D out-proj (23 -> 47 us) | REJECTED (op-level cause: bank-count-bound DS grid at this N) |
| DRAM-sharded decode matmuls, Blackhole multi-reader kernel (review-2) | `mixer_ds_readers` 1/2/3 with the in-projection DRAM-sharded and the tuned 1D out-projection kept (`mixer_ds_out_1d`), and `ds_all` r=2/3 | ds_in+1D-out r=1: 0.599 vs 0.579 (10 replays); r=2 = r=1 (GDN in N=5152 pads to 21 tiles/bank, not divisible by 2 -> 1 reader); r=3 and every `ds_all` r>1: TT_FATAL | ds_in+1D-out r=1: 0.461 vs 0.446; r=2/3: TT_FATAL | `sweeps/mixer_ds_readers.jsonl`, `logs/05-r2-sweep-ds-readers.log`: `num_workers_per_dram_bank > 1` runs `matmul_utilities.cpp:446` -> `experimental::Device::get_worker_noc_hop_distance`, which asserts `mesh->num_devices() == 1` (`tt_metal/impl/device/experimental/device.cpp:20`): the multi-reader DRAM-sharded matmul is unit-mesh only in this checkout; the 1-reader in-projection DS kernel with the 1D out-projection is still slower than all-1D | REJECTED (exact op contract for r>1; measured for r=1; ttnn backlog item) |
| MoE dispatch ordering (sorted-dispatch prefill, T >= 512) | `ttnn.sort` (fp32, cross-core-exchange factory) vs `ttnn.sort` on uint16 ids (DRAM multi-core factory) vs a counting sort from the one-hot cumsum (`DISPATCH_MODE`: `counting_h` = `[Rp, E]` one-hot, dim-2 scan — DEFAULT; `counting` = transposed `[E, Rp]` one-hot, last-dim scan) | prefill 2048 wall: 122.7 (fp32 sort) / 139.9 (uint16 sort) / 133.0 (counting_h) / 140.8 ms (counting, transposed) | 117.9 / 135.5 / 128.2 / 136.2 ms | the fp32 sort DEADLOCKED 3/3 times in the 48-layer prefill warm-up (one device stuck in `sort_barrier`, the others in the next reduce-scatter; `doc/full_model/triage/warm_hang_*.txt`) although 2000 isolated sorts pass (`logs/06-sort-stress-*.log`); the uint16 sort is hang-free but ~4 ms per 512-token chunk; the counting sorts have no sort op at all (`test_mc_dispatch_counting_matches_sort`, `logs/05-r3-dispatch-test.log`: counting == counting_h == sort_fp32 for T = 512 / 513 / 1000 / 1023 / 2048 — 513 and 1023 exercise the sentinel `Rp != R` branch with 24 / 8 padded rows) and cost ~2.6 ms per 512-token chunk more than the fp32 sort: Tracy (`tracy/gdn/prefill_2048_*`, device 0, 2048 tokens) 60.98 vs 49.94 ms device — `AccumulationDeviceOperation` (cumsum) 8x = 9.3 ms and `PermuteDeviceOperation` 16x = 5.3 ms (the dim-2 scan permutes in and out) against ~3.6 ms of removed sort-path ops (sort + score gather + perm scatter); the transposed last-dim scan is slower still on the host wall (`logs/05-perf-plain5.log`, not profiled) | counting_h ADOPTED (correctness over 2 % of the per-layer prefill wall); optimized-full-model item: replace the scan by per-32-row-block tril matmuls + a block-prefix matmul (all fast tile ops) |
| activation sharding / 1D grids | attention qkv: an (11,8) = 88-core grid with per_core_N = 2, i.e. 72 of 88 cores active for the 144 N tiles (Tracy row: 72 cores); `wo` 33 requested = the qwen36 default -> 32 active cores (Tracy row: 32) | – | 0.446 vs 0.453 (20 replays, x2) | `sweeps/dense_1d_long.jsonl`; PCC vs HF identical to 6 digits (three attention decode lines moved in the 7th-10th digit between the stage-4 and stage-5 final suites) | ADOPTED |
| 1D kernel K block (`mixer_in0`) | in0_block_w 12 / 16 / 24 / 32 / 48 vs the largest-divisor default (8 of 96 K tiles) | 0.579 -> 0.579..0.585 | 0.446 -> 0.446..0.449 | `sweeps/mixer_in0.jsonl` (10 replays): monotonically slower with larger blocks | keep default |
| packed vs separate projections | fused `[q|gate|k|v]`, `[qkv|z|a|b]`, `[up|gate]` (inherited) | – | – | one matmul per activation (topology audit) | keep |
| activation / CCL dtype | bf16 collectives vs bfp8 payloads (`out_proj_decode_dtype=bfp8`: the decode out-projection emits bfp8 partials straight into the reduce-scatter; all-gather input = the bf16 norm output, a bfp8 gather would need an extra typecast op) | 0.580 vs 0.579 (10 replays); PCC decode 0.99996 -> 0.99989 | 0.444 vs 0.446; PCC decode 0.999934 -> 0.999933 | op-level (`logs/05-r2-ccl-probe.log`, 20 traced replays, decode shapes): reduce_scatter_minimal_async bf16 22.2 us -> bfp8 20.8 us (-1.4 us), all_gather_async bf16 24.3 -> bfp8 21.7 us (-2.6 us, not reachable without a typecast op of ~2-3 us); in-model: `sweeps/out_dtype.jsonl` (family `out_dtype`, 10 replays, PCC vs HF) | REJECTED: within noise on the wall clock (the two reduce-scatters are latency-bound at this payload), and the GDN layer loses PCC; keep bf16 partials |
| persistent buffers | persistent output / intermediate buffers on the 4 collectives; in-model `norm_ag_persistent` (the two `DistributedNorm` decode gathers write into per-shape persistent output buffers, `DistributedNorm._persistent_ag_buffer`) | 0.574 vs 0.579 (10 replays), PCC identical | 0.441 vs 0.446, PCC identical | op-level (`logs/05-r2-ccl-probe.log`, same run as the dtype rows): all_gather_async 24.3 -> 20.3 us with a persistent output (-4.0 us), reduce_scatter_minimal_async 22.2 -> 22.5 us (none); in-model: `sweeps/norm_ag_persistent.jsonl` (10 replays; `norm_ag_persistent_run1_noop.jsonl` is the first attempt whose buffer was skipped for the sharded output config — identical times, kept as the negative control) | measured, NOT adopted: -5 us per step on both kinds and PCC identical, but with the persistent all-gather output buffer the attention traced-decode replays stopped being bit-identical (`logs/05-mc-final-suite3.log`, `test_mc_traced_decode[attn]`: replay 2 != replay 0 in the low bits) while the GDN replays stayed identical; determinism of the served path wins over 1 %. Kept as an opt-in knob for the optimized-full-model stage, where the persistent-output path of `all_gather_async` on the L1 width-sharded layout has to be checked for a completion race before use |
| precision / fidelity | mixers bfp8 HiFi2 vs LoFi (`mixer_fidelity`) | 0.579 vs 0.579 | 0.446 vs 0.446 | `sweeps/mixer_fidelity.jsonl` (10 replays, real weights): no time difference (rows are DRAM-bound), PCC drops (prefill 0.99992 -> 0.99960 GDN, 0.99985 -> 0.99927 attention) | keep HiFi2 (measured); routed bfp4 LoFi, KV bfp8, state bf16 from stages 3/4 |
| MoE active-expert execution | gather-mode `sparse_matmul` gate/up on 16 cores (N=512 caps the grid) | 62 us op | 62 us op | `gate_up_output_tile (32,16)` retried 5x (`sweeps/gu_tile.jsonl`, `logs/05-sweep-gu_tile*.log`): (1) needs the WEIGHT stored with 16-wide tiles (`from_torch(tile=)` works on the bfp4 4D expert tensor); (2) with the narrow weight + narrow output on every gate/up call and 16-high tiles on the down weight the sparse matmuls RUN; (3) the gate/up split that follows (`ttnn.swiglu` = slice + silu*mul, `fused_decoder.py:199`, and `ttnn.slice` in the decode gather path) refuses non-32x32 tiles (`slice_device_operation.cpp:203 "slice does not currently support tiles other than 32x32, got 32x16"`) — no ttnn op splits a narrow-tiled tensor; (4) review-2 op-level timing (`logs/05-r2-gu-tile-probe.log`, real gate/up geometry E=256, K=3072, 2Nl=512, k=8, ROW_MAJOR sparsity): 32x32 tiles on 16 cores 65.7 us eager / 74.3 us traced vs 32x16 tiles on 32 cores 64.6 / 73.6 us — the op is not core-bound at this width (bfp4 expert rows, one gather per expert), so the split adaptation would buy nothing | REJECTED, measured: the 32-core narrow-tile matmul is within 2 % of the 16-core one; the downstream split contract (`slice_device_operation.cpp:203`) stays a backlog note only |
| shared-expert grid | 16 cores (stage 4) | -26 us | -26 us | stage-4 `shared` sweep | kept |


## 3. Prefill host gap

Before this stage (stage-4 default path): 2048 tokens as 4 x 512 chunks = 239 ms host wall vs 50 / 47 ms device time
(GDN / attention), i.e. no speedup over one die (250 / 230 ms) although the device work was 2.5x / 1.9x faster.

Root cause found while making a chunk traceable: the MoE prefill paths created their `sparse_matmul` output buffers and the
routing matrix with `ttnn.zeros(..., device=)`, which WRITES the buffer from the host — per 512-token chunk that is
gate/up `[1,G,32,512]` bf16 (G = 377 expert groups -> 12 MB) + down `[1,G,32,3072]` (74 MB) + the routing matrix `[1,1,T,256]`, blocking host->device
transfers inside the prefill (`_routed_sorted`, `_routing_matrix`; also illegal in a trace capture,
`fd_mesh_command_queue.cpp:825`). Fix: `_alloc_dev()` — `ttnn.empty` (device allocation, no host write) for the two sparse-matmul
output buffers (every real group is written by the op and the padded groups are never read by `undispatch_rows`) and
`ttnn.fill` (device op) for the routing-matrix scatter target; nothing is cached across calls or layers. Same numerics
(final suite 40/40). The counting-sort dispatch (section 2, `MoE dispatch ordering`) gives back ~10 ms of the 2048-token
wall for a hang-free path; its attribution and recovery are an optimized-full-model item.

| 2048-token prefill (4 x 512 chunks), host wall | GDN layer | attention layer |
|---|---|---|
| one die (stage 3/4 `perf_single`) | 250.4 ms | 229.8 ms |
| mesh, stage-4 default | 238.7 ms (1.05x) | 238.7 ms (0.96x) |
| mesh, device-side buffers, fp32 `ttnn.sort` dispatch (post-review-1 final chain, `logs/05-perf-plain.log`, 18:22 UTC) | 122.7 ms (2.04x) | 117.9 ms (1.95x) |
| mesh, stage-5 default: device-side buffers + counting-sort dispatch (`logs/05-perf-plain4.log`) | 133.0 ms (1.88x) | 128.2 ms (1.79x) |
| mesh device time per device (Tracy, section 4) | see PERF_TABLES.md | see PERF_TABLES.md |

The remaining gap between wall and device time is host dispatch of ~950 / 770 mesh ops per chunk; the removal is a traced
chunk. Status of the traced chunk on the default path (`test_perf_prefill_512_traced`, `-m perf_probe`, `QWEN35_RUN_PROBES=1`):
both chunk kinds CAPTURE (no host boundary left: 0 from_torch/to_torch in the op-site census) but the replay hangs on device.
Bisected (`scripts/05-trace-bisect.py`, `logs/05-trace-bisect-*.log`, work_log 16:11-18:00): the MoE block alone, the GDN
mixer alone (with state write-back), norm + mixer and norm + MoE all replay; mixer + MoE in one trace hangs; with the GDN
out-projection UNFUSED (`gdn_prefill_mmrs=False`, new knob) mixer + MoE replays, but the whole layer (its two `DistributedNorm`
all-gathers plus the two reduce-scatters) still hangs at the first replay and the near-full layer replays once then hangs.
One factor is established (the fused fp32 `matmul_reduce_scatter_async`: unfusing it makes mixer + MoE replay); a second
factor in the whole-layer composition is a HYPOTHESIS (the norm all-gathers sharing the TT_CCL barrier pool with the
reduce-scatters — the decode trace replays the same AG/RS/AG/RS pattern fine, so the prefill-shaped programs matter) with no
confirming experiment yet (e.g. norm all-gathers on a separate semaphore set). Not resolved in this stage: the default keeps
the fused out-projection (fastest eager prefill); the full-model stage runs prefill eagerly through the masked buckets and
the full-model optimize stage owns traced prefill with this bisect as its starting point.

Related but DIFFERENT: the stage-6 two-layer stack hung in an EAGER prefill after a TRACED decode (`logs/06-pad-*.log`).
`gdn_prefill_mmrs=False` does NOT change that (`06-pad-prefill-mmrsoff.log` hangs the same way), so it is not a candidate fix
for it. Its root cause is trace-lifecycle ordering (prefill programs / persistent buffers first created AFTER the decode
trace was captured land in the trace's scratch range; the next replays corrupt them; qwen36 rule #48536): with the prefill
programs warmed before the capture the sequence passes, and the layer-level regression `test_mc_prefill_after_traced_decode`
(prefill -> traced decode x4 -> release -> reset -> eager aligned + chunked prefill -> traced decode again, both kinds;
`logs/05-r2-regression2.log`, 2 passed — `05-r2-regression.log` is the first run whose two failures were test bugs, see the work log 20:41) pins that ordering at the decoder level.

## 4. Final numbers, PCC, watcher, audit

Final default path = `MULTICHIP_SELECTED = MultichipConfig()` as committed (attention qkv on 72 of 88 cores, `wo` 32 active, shared
16 cores, device-side MoE buffers, counting-sort dispatch `counting_h`). All numbers below come from that path on the committed code:
`logs/05-mc-final-suite5.log` (suite), `05-perf-plain4.log` -> `HOST_WALLS.md` (host walls), `05-tracy-decode-rerun-chain.log` and
`05-final5-chain.log` -> `PERF_TABLES.md` (Tracy decode / prefill rows), `05-r3-chain.log` + `artifacts/05-watcher-{1,2,4,5}/` and the
19:05 chain (`05-final-chain3.log`, watchers 3 and 6) for the watcher runs.

**Correctness**: the committed suite is 42 tests: 40 passed / 0 failed in the final run (`logs/05-mc-final-suite5.log`, 22:21 UTC,
and `05-mc-final-suite4.log` before it: the 35 stage-4 tests + `test_mc_prefill_after_traced_decode` x2 +
`test_mc_dispatch_counting_matches_sort` x3) plus the two sentinel-length parameters (T = 513, 1023) added afterwards and
run in `logs/05-r3-dispatch-test.log` (5 passed, 22:58 UTC, together with the three lengths already in the suite); same PCC bars and values as
stage 4: mesh vs single-chip baseline >= 0.999 GDN / >= 0.9996 attention up to 4429 tokens, HF real-weight T=37 0.9999,
batched B=32, non-aligned lengths, GDN reset, traced decode replays bit-identical, runtime fallback audit unchanged:
attention 0, GDN non-aligned 3 host transfers).

**Host wall** (ms; single-chip = stage-3 `OptimizedDecoder` in its own process; prefill 2048 = 4 x 512 `start_pos` chunks;
decode = 3 warmed traced replays):

| layer | case | single-chip ms | mesh (1x4) ms | speedup | efficiency |
|---|---|---|---|---|---|
| full_attention | prefill_128 | 16.097 | 15.133 | 1.06x | 27% |
| full_attention | prefill_2048 | 229.755 | 128.171 | 1.79x | 45% |
| full_attention | traced_decode | 0.727 | 0.449 | 1.62x | 40% |
| linear_attention | prefill_128 | 18.511 | 15.530 | 1.19x | 30% |
| linear_attention | prefill_2048 | 250.441 | 132.990 | 1.88x | 47% |
| linear_attention | traced_decode | 0.867 | 0.583 | 1.49x | 37% |

**Device time** (Tracy, per device, `PERF_TABLES.md`; single-chip = committed stage-3 tables):

| layer | case | single-chip ms | stage-4 mesh ms | stage-5 mesh ms | speedup vs one die | efficiency |
|---|---|---|---|---|---|---|
| GDN (L0) | prefill_128 | 16.197 | 8.213 | 8.199 | 1.97x | 49% |
| GDN (L0) | prefill_2048 | 124.241 | 49.963 | 60.971 | 2.04x | 51% |
| GDN (L0) | traced decode | 0.827 | 0.474 | 0.472 | 1.74x | 44% |
| attention (L3) | prefill_128 | 15.138 | 9.035 | 9.035 | 1.67x | 42% |
| attention (L3) | prefill_2048 | 88.948 | 47.202 | 58.231 | 1.53x | 38% |
| attention (L3) | traced decode | 0.656 | 0.419 | 0.406 | 1.61x | 40% |

Prefill device time at 2048 tokens is +11 ms per layer vs stage 4: the `_alloc_dev` buffers are allocations, not fills (the device time was back at the stage-4 value with them), and the counting-sort dispatch that replaced the deadlocking `ttnn.sort` adds ~2.7 ms device per 512-token chunk (section 2, `MoE dispatch ordering`: cumsum 9.3 ms + permutes 5.3 ms per 2048 tokens on the GDN layer against ~3.6 ms of removed sort-path ops); the host wall still nearly halved (239 -> 133 / 128 ms, section 3). Decode: attention -13 us device / -7 us wall vs stage 4 from the wider qkv grid (48 -> 72 cores, 54 -> 45 us); GDN unchanged (its 1D grids were already at the optimum, `gdn_1d` sweep).  (Tracy decode rows regenerated on the committed default, `logs/05-tracy-decode-rerun-chain.log`, 22:11 UTC.)

**Performance accounting (one traced decode step, per device)**

| | GDN layer | attention layer |
|---|---|---|
| bytes the step must move per device (routed 8 experts bfp4 10.6 MB + mixers bfp8 23.5 / 21.7 MB + shared 2.5 MB + router 1.6 MB + GDN state r/w 1.0 MB / KV read 0.07 MB) | 39.2 MB | 36.5 MB |
| theoretical roofline at 512 GB/s per die | 76.7 us | 71.3 us |
| device time (Tracy window / 3 replays, device 0) | 472 us (16 %) | 406 us (18 %) |
| in-window op-to-op gaps (same run, device 0; 1.28 / 1.03 us per op) | 131 us | 87 us |
| profiled wall per replay (same Tracy run, `*_tracy.console.log`) | 614 us | 508 us |
| unprofiled traced wall (host-wall run, `05-perf-plain4.log`) | 583 us | 449 us |

The roofline fraction is bounded by op count, not by the DRAM-bound rows: matmuls + sparse matmuls are 172 / 158 us
(within 1.5-2.5x of their own rooflines), the 4 collectives 47 us (latency-bound at 4.6 KB), and ~90 / ~70 small ops
(layouts, gates, norms, scoring, reshards) 245 / 200 us at 1-8 us each. The per-op issue cost of the trace replay on the
mesh is the in-window op-to-op gap of the same run: 131 us over 103 ops (GDN) and 87 us over 84 ops (attention), 1.0-1.3 us
per op; device time + gaps = 603 / 493 us, within 2-3 % of the profiled wall (614 / 508 us). Named limitations feeding the ttnn
backlog: (1) gather-mode `sparse_matmul` capped at 16 cores by N=512 per device (62 us, 22 % DRAM); (2) no fused
"create heads" op for the GDN 4/4/16-head layout (~55 us of slices/reshapes/repeat_interleave); (3) 12-13 us floor for
4-device ring collectives on 4.6 KB; (4) per-op trace dispatch ~1 us on the mesh.

**Watcher** (`TT_METAL_WATCHER=10`, all features, one pytest process per test with `tt-smi -r` between runs — the stage-4
protocol; `watcher/pytest_<i>.txt`, `watcher/watcher_<i>.tail.txt`, `artifacts/05-watcher-<i>/`). On the committed code
(counting-sort dispatch, `logs/05-r3-chain.log` + `05-r3b-chain.log`, 22:59-23:10 UTC): real-weight T=37 GDN (1) and
attention (2), chunked prefill to 4429 tokens GDN (4) and attention (5) — the two runs that exercise the sorted-dispatch
path; from the 19:05 chain (`05-final-chain3.log`, decode paths unchanged since): batched decode B=32 attention (3) and
traced GDN decode (6); plus the traced GDN decode with the later-rejected `norm_ag_persistent=True` (7). Every test PASSED,
watcher.log 2503-3996 lines each, 0 bad lines, no tripped assert in any model kernel. Every process ended in the known
eth-core 29-25 fabric-router teardown failure (`doc/multichip_decoder/watcher/TALLY.md`, forms (i)/(ii)), never during
the test.

**Runtime fallback audit / stress**: unchanged from stage 4 (`test_mc_runtime_fallback_audit`, `test_mc_stress_repeated_runs`
in the final suite: attention 0 host transfers, GDN non-aligned 3 inside the shared kernels; 20 mixed-length cycles with
bit-identical repeats).


## 5. Rejected options and limitations

- Fused matmul+reduce-scatter and fused all-gather+matmul at decode are both slower than the separate tuned ops at these
  shapes (section 2); the fused ops' matmul kernels (2D mcast / minimal matmul) are prefill-oriented and the collectives
  they hide are 12-13 us latency-bound ops on 4.6 KB. Prefill keeps the fp32 `matmul_reduce_scatter_async` out-proj (GDN).
- DRAM-sharded decode weights lose to the interleaved 1D kernel at per-device N 4608-5152 / K 2048 (section 2); the
  Blackhole multi-reader DRAM-sharded kernel (`num_workers_per_dram_bank > 1`) is unit-mesh only in this checkout.
- Sentinel rows of the sorted dispatch (T*k % 32 != 0) carry a source-token index >= T into the trailing all-padding group
  (`tok_sorted = floor(i/k)` for the padded positions; pre-existing in the `ttnn.sort` build): `gather_rows` reads 1-3 rows
  past the activation buffer for them; inert (padding rows have weight 0 and are never un-dispatched) but an out-of-range
  read — one-line mask (`where(is_real, tok, 0)`) plus an `src_tok < T` assertion in the dispatch test, scheduled with the
  next dispatch change (optimized-full-model stage) so the committed tree stays the tested tree.
- `ttnn.sort` (fp32 row, cross-core-exchange factory) deadlocks intermittently inside the multi-layer prefill on the (1,4)
  mesh (one device in `sort_barrier`, the others in the next reduce-scatter) although it passes 2000 isolated iterations —
  the model no longer uses it (counting-sort dispatch); ttnn backlog item with the triage captures and the stress script.
- `all_gather_async(persistent_output_buffer=...)` on the L1 width-sharded decode norm output saves 5 us/step but made the
  attention traced-decode replays non-bit-identical (section 2) — opt-in knob, not the default; to be checked for a
  completion race in the optimized-full-model stage.
- Gather-mode gate/up `sparse_matmul` is capped at 16 cores by N = 512 per device (114 GB/s, 22 % of DRAM peak, 62 us =
  13 % of the step). The 16-wide-output-tile route was carried to the op that blocks it: the matmuls run with narrow
  weight/output tiles, but the gate/up split afterwards (`ttnn.swiglu` / `ttnn.slice`) supports 32x32 tiles only
  (`slice_device_operation.cpp:203`). Backlog: narrow-tile slice/swiglu or a sparse matmul with an in-kernel SwiGLU;
  repro = `tools/sweep_multichip.py --family gu_tile` (frames in `sweeps/gu_tile.jsonl`).
- The GDN decode step still spends ~55 us in head-layout ops (5 slices of the fused projection, 3 slice+reshape pairs into
  head rows, 2 `repeat_interleave` for the GVA expansion, 7 `ReshapeView` incl. two 8 us row->head-rows conversions, the
  conv shift-register copies) and ~44 us in the recurrent step (op-site map `logs/05-opsite-probe.log`, ops 5-58). These
  are shared-kernel data-movement ops already rewritten in stage 3; a fused "GDN create heads" op (the `nlp_create_qkv_heads_decode`
  analogue for 4 k / 4 q / 16 v heads) does not exist in ttnn — backlog item, estimated 30-40 us per GDN step (36 layers).
- Attention `wo` prefill partial is reduce-scattered as bfp8 (stage-4 note); real-weight T=37/333 PCC 0.9999; re-check at
  long context in the full-model stage before changing the dtype (a bf16 partial doubles the RS payload at prefill).
- Watcher runs keep the stage-4 protocol (one process per test, reset after each; eth-core teardown limitation).
