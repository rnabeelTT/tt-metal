# Optimized decoder — Qwen/Qwen3.5-122B-A10B (stage 3)

`tt/optimized_decoder.py` — `OptimizedDecoder` (same public contract as the functional/fused decoders: paged prefill with
`start_pos` chunking, traced decode, GDN state, batch) driven by `OptimizationConfig`; `SELECTED` is the policy chosen by the
sweeps below. One Blackhole die (p300c), real weights (layer 0 = GDN + MoE, layer 3 = gated attention + MoE), HF layer reference.
Evidence: `tests/test_optimized_decoder.py`, `tools/sweep_optimized.py` (+ `tools/summarize_sweeps.py` -> `sweeps/SUMMARY.md`),
`tracy/<kind>_<fused|optimized>/*_perf_report.{txt,csv}`, `watcher/`, run logs `~/qwen35-122b-bringup/logs/03-*.log`.

## 1. Operation-topology audit of the measured decode path (stage-2 fused tables, per traced replay)

| layer kind | ops | device ms | dominant rows (us) | structural rows (us) |
|---|---|---|---|---|
| GDN (L0) | 118 | 1.464 | mega in-proj `32x3072x20608` 190; out-proj `32x8192x3072` 183; sparse gate/up 203 + down 100 (8 experts); router `32x3072x256` 70 (8 cores, SLOW); packed shared 56 + down 25; 3 RMSNorm 111 (one core each) | ReshapeView x5 86; BinaryNg (GDN recurrent step, ~30 ops) 139; tilize/untilize 80; Slice 41; conv/recurrent small matmuls ~25 |
| attention (L3) | 89 | 1.128 | fused qkv `32x3072x17408` 153; o_proj `32x8192x3072` 182; sparse gate/up 203 + down 101; router 70; packed shared 56 + down 25; 2 RMSNorm 98 (one core each) | SDPA-decode 25; paged_update_cache 2x4; slices/reshapes/permutes ~60; BinaryNg ~50 |

Findings and actions: (a) same-input projections are already packed (fused qkv, GDN mega qkv|a|b|z, packed shared expert; stage 2);
(b) the router matmul, the shared-expert matmuls and both out-projections run far below DRAM bandwidth with ttnn's auto program
configs -> round-2 program-config families; (c) both RMSNorms run on one core -> sharded norm program on an L1 width-sharded copy of
the residual; (d) the routed experts are the largest rows -> dtype/fidelity + `sparse_matmul` geometry families; (e) the GDN
recurrent-step chain (~40 small ops) is the shared single-device kernel's structure — see section 8.

## 2. Selected policy (`SELECTED` in `tt/optimized_decoder.py`) and the cumulative contract

| contract | selected | evidence |
|---|---|---|
| routed experts gate/up + down | bfp4 / LoFi, `fp32_dest_acc_en=False` (tt-metal #49068), `ttnn.sparse_matmul` gather mode in decode (8 experts per token, 64 cores, in0_block_w 32 for both roles) | sweeps `precision`, `gather` |
| routed experts prefill | masked union per 64-token group (32 cores / in0 64 gate-up, 32 cores / in0 32 down) outside 512..4096 tokens; expert-sorted dispatch (`tt/moe_dispatch.py`, one compact sparse matmul per projection, every 32-row group = one expert) for 512 <= T <= 4096 (index math exact to 7196 dispatched-row-wise, checked to 4096) — see section 5 | sweeps `masked`, `masked2`, `prefill_mode_*`, `tools/check_sorted_dispatch.py` |
| shared expert | packed `[gate|up|sigmoid-gate]` matmul + down, bfp8 / HiFi2 (fp32 acc), 1D mcast_in0 program configs on 44 cores in decode | sweeps `shared`, `shared_pc` |
| router | single-user decode row: TTMoEGate op (bf16 weights, HiFi2 + fp32 acc) with the derived 2D-mcast config (in0_block_w 32, 8 N-cores); prefill and batched decode: functional softmax + top-k router (the gate op is bit-nondeterministic with several valid rows, section 3) | sweep `router`, determinism probe |
| attention projections | fused `[q+gate|k|v]` and `o` bfp8, decode LoFi (qwen36 decode compute config); decode: DRAM width-sharded weights + DRAM-sharded kernel, 1 reader per bank, activation width-sharded in L1 on 32 cores (qkv in0_block_w 3, o in0_block_w 8); prefill on an interleaved copy of the weight | sweeps `mixer_weights`, `dense_1d`, `dense_ds`, `dense_ds2` |
| GDN projections | mega `[qkv|a|b|z]` in-proj bfp8 with the 1D mcast_in0 config (64 cores, in0_block_w 8: its 81-tile bank shard needs 3 readers to fit L1 and then loses); out-proj bfp8 DRAM-sharded, 1 reader | sweeps `mixer_weights`, `gdn_dtype2`, `dense_1d`, `dense_ds2` |
| KV cache | **bfp8** (paged, block 64; the fill cast lives in the attention kernel): -4.5% attention decode at 16K context, PCC-neutral on real weights (0.99962 / 0.99985 vs 0.99959 / 0.99983 bf16); doubles the KV capacity of the context contract | sweeps `kv`, `sdpa_long`, `kv_final` |
| SDPA decode | q_chunk 32 / k_chunk 256 (with the bfp8 cache: 0.808 vs 0.824 ms at k_chunk 64) | sweeps `sdpa`, `sdpa_long`, `kv_final` |
| GDN state | bf16 recurrent state, split per-tap conv buffers; single fused `mac` update written in place; one-hot GVA matmul | sweep `gdn_state`, section 6 |
| residual / norms (decode) | residual L1 interleaved; both RMSNorms run the sharded program on a width-sharded L1 copy (tt_transformers `get_norm_config("attn", DECODE)` grid); the attention norm hands its width-sharded output straight to the DRAM-sharded qkv matmul (the hook adopts the norm's shard grid), the ff norm returns L1 interleaved for the MoE block | sweeps `norm`, `norm_ds` |
| logical batch | 1 user = one 32-row tile (primary target); batched decode [1,B,1,dim] (one tile per user) keeps the interleaved norm and the default matmul path; verified at B=32 | `test_opt_batched_decode_against_hf` |
| fusion flags | all stage-2 flags on | stage 2 |

Decode device time and dominant rows: section 4. All PCC values in section 3 were produced by this exact policy.

## 3. Correctness (SELECTED) — `tests/test_optimized_decoder.py` (logs `03-final-suite.log`, `03-final-long.log`)

25/25 PASS (`-k "not test_perf and not test_opt_long"`, 412 s, final policy and final code) plus the long-parity cases (4 passed, 2 strict xfails). PCC vs the HF layer (prefill / decode):

| case | GDN (L0) | attention (L3) |
|---|---|---|
| synthetic T=5 | 0.99990 / 0.99982 | 0.99963 / 0.99964 |
| synthetic T=129 | 0.99886 / 0.99847 | 0.99980 / 0.99988 |
| synthetic T=333 | 0.99854 / 0.99818 | 0.99987 / 0.99994 |
| real weights T=37 | 0.99823 / 0.99948 | 0.99960 / 0.99951 |
| real weights T=300 | 0.99882 / 0.99959 | 0.99971 / 0.99989 |
| batch 32 per-user prefill / batched decode | 0.99941 / 0.99911 | 0.99971 / 0.99979 |
| traced decode 3 steps (bit-identical replays / prefill) | 0.99880 / 0.99884 / 0.99859 | 0.99983 / 0.99982 / 0.99979 |
| chunked prefill 2048+2048+333 (`start_pos`), decode at 4429 | 0.99758 / 0.99743 | 0.99997 / 0.99999 (K/V windows pass, bfp8 cache) |
| expert-sorted vs masked prefill (real weights, same routing) | T=37 1.00000000, T=1023 1.00000000 | T=333 1.00000000 |
| expert-sorted vs HF | T=37 0.99881, T=1023 0.99924 | T=333 0.99972 |
| prefill 128, 128 decode steps, 100-token chunk continuation at 256 (state carried across decode) | decode steps 0.99860, continuation 0.99808 | 0.99991, 0.99994 |
| long HF parity, synthetic weights (SELECTED policy incl. bfp8 KV cache) | 65573 tokens 0.99741 (last 2048: 0.99732), decode 0.99790 | 16421 tokens 0.99999 (all 0.99998), decode 0.99999; K/V content windows pass with the bfp8 cache |
| long HF parity, REAL weights (SELECTED policy) | 65573 tokens 0.99584 (last 2048: 0.99552), decode 0.99847 | 16421 tokens 0.99990 (all 0.99988), decode 0.99968; K/V windows 0.99983 |
| rejected-candidate evidence (strict xfail variants) | bfp4 GDN in-projection at 65573 tokens: last-2048 PCC 0.99304 on REAL weights (0.99387 synthetic) < 0.995 | — |

Every value equals the stage-1/2 functional value to within 1e-4 (the routed bfp4 experts and the rewritten GDN step do not move
the layer output at these lengths); GDN long-context parity is unchanged from stage 1 (0.9974). Runtime fallback audit: 0
`from_torch`/`to_torch` in aligned prefill, chunk continuation, the 640-token expert-sorted prefill and decode (both kinds).
Stress: 20 prefill+decode cycles over lengths 5..1500 per kind, all finite, identical cycles bit-identical.

Bit-determinism (`tools/check_determinism.py`, identical inputs repeated on real weights): single-user decode is bit-identical
in every policy variant (0/29 x 5 variants). The TTMoEGate router is NOT bit-deterministic when it processes several valid rows:
5/29 attention-layer T=37 prefills and 4/11 B=32 batched decodes (both kinds) differed at bf16-ulp level, 0/29 prefills with the
functional softmax/top-k router in the otherwise identical graph. The optimized block therefore uses the gate op for the
single-user decode row only (`x4.shape[2] == 1`); prefill and batched decode use the functional router (prefill cost ~1% of the
layer; batched decode ~+100 us at B=32, not the latency target). Post-fix probes: single-user 0/29 differing runs for both kinds,
prefill and decode; B=32 (code as of 07:28, before the attention-norm hand-off which the batched layout bypasses): 0/11 differing runs for both kinds, prefill and decode. One unexplained bit-flake of a T=40 GDN prefill was seen once (1 of 7
runs, values equal to 4 digits) and never reproduced in 35 + 29 GDN runs.

## 4. Performance before/after — `tracy/<kind>_<fused|optimized>/*_perf_report.{txt,csv}` (real weights, one die)

Signposted `tt-perf-report` windows, real weights, one die; "fused" = the stage-2 graph (`FusedDecoder`) run on the CURRENT
shared kernels, "optimized" = `OptimizedDecoder` with `SELECTED`. Full tables (dominant rows with the measured dtype / fidelity /
in0_block_w, op-class totals, roofline) in `PERF_TABLES.md` (generated by `tools/perf_tables.py`); logs `03-final-perf.log`.

| layer | case | fused device ms (ops) | optimized device ms (ops) | change |
|---|---|---|---|---|
| GDN (L0) | prefill T=128 | 26.74 (315) | 16.20 (285) | -39.4% |
| GDN (L0) | prefill T=2048 (4 x 512 chunks) | 398.9 (1994) | 124.2 (1550) | -68.9% |
| GDN (L0) | traced decode (per replay) | 1.288 (91) | 0.827 (95) | -35.8% |
| attention (L3) | prefill T=128 | 27.71 (132) | 15.14 (104) | -45.4% |
| attention (L3) | prefill T=2048 (4 x 512 chunks) | 471.4 (1248) | 88.9 (812) | -81.1% |
| attention (L3) | traced decode (per replay) | 1.125 (88) | 0.656 (92) | -41.6% |

Note on the baseline: the stage-2 commit measured the fused GDN decode at 1.464 ms; the fused graph now runs 1.288 ms because
the GDN recurrent-step rewrite lives in the shared kernel and applies to every caller. Against the committed stage-2 tables the
optimized decoder is -43.5% (GDN) / -41.6% (attention) in decode (the decode window has 128 keys, where the bfp8 cache has no
time effect; its -4.5% is measured at 16K context in sweep `kv_final`). Host wall per traced step (3 replays, profiler
attached): GDN 1.362 -> 0.912 ms, attention 1.227 -> 0.772 ms; unprofiled sweep-tool walls 0.869 / 0.725 ms. (Prefill op counts
dropped by 15 per kind versus the earlier collection because prefill rows now use the functional router instead of the gate op;
the attention prefill has 8 extra typecast rows for the bfp8 cache fill.)

Dtype / fidelity proof (optimized decode rows): routed expert rows `SparseMatmul ... x 32 x 3072 x 2048` and `... x 1024 x 3072`
show weight dtype **bf4**, fidelity **LoFi**, in0_block_w 32; the attention qkv `32 x 3072 x 17408` runs the DRAM-sharded
kernel (**bf8 / LoFi**, in0_block_w 3, 129 us at 81% DRAM utilisation; was 160 us / 66% with the 1D config) and the out-projections
`32 x 8192 x 3072` likewise (DRAM-sharded, bf8 / LoFi); the GDN mega in-projection `32 x 3072 x 20608` keeps the 1D config
(bf8 / LoFi, in0_block_w 8, 66% DRAM); the shared expert rows bf8 / HiFi2; the router `32 x 3072 x 256` bf16 / HiFi2 9.1 us
(was 70 us).

Roofline accounting (decode, bytes moved per token at the stored dtypes over 512 GB/s):

| layer | bytes/token | roofline | device time | fraction | host wall |
|---|---|---|---|---|---|
| GDN | 152 MB | 0.297 ms | 0.827 ms | 36% | 0.912 ms (0.869 unprofiled) |
| attention | 138 MB | 0.269 ms | 0.656 ms | 41% | 0.772 ms (0.725 unprofiled) |

Where the remaining gap is (per replay): the GDN mega in-projection runs at 66% of DRAM bandwidth (186 us vs 131 at peak; the
DRAM-sharded kernel cannot hold its 81-tile bank shard in L1 with fewer than 3 readers), the DRAM-sharded rows at ~80%; the 8 active experts take 178 us for 42.5 MB (240 GB/s: the sparse kernel streams each expert
once per token group and adds a per-pair fixed cost of ~15 us); the remaining ~45 GDN structural ops sum to ~300 us
(ternary state updates 39 us, head reshapes, conv copies, norms); dispatch gap between replays is ~0.08 ms (host wall minus
device time). Named limitations: per-pair fixed cost of `sparse_matmul`, output-tile-parallel `reshape_tiled` (padded->dense
reshapes on 8-16 cores), single-core `scatter`; none has a model-side workaround left in this stage.

Host-wall history of the traced decode (sweep tool, 3 replays):

| GDN layer | attention layer | change |
|---|---|---|
| 1.610 | 1.340 | stage 1 functional |
| 1.464 | 1.128 | stage 2 fused |
| 1.370 | 1.076 | routed bfp4 / LoFi |
| 1.358 | 1.063 | gather geometry 64 cores / in0 32 |
| 1.306 | 1.002 | sharded decode norms |
| — | 0.924 | router derived program config |
| — | 0.937 (from 0.987 base) | shared expert 1D configs |
| 1.140 | 0.882 | dense projections 1D configs |
| 0.884 | 0.777 | GDN recurrent-step rewrite (GDN only) + all of the above |
| 0.869 | 0.735 | DRAM-sharded decode weights for attention qkv/o and GDN o |
| — | 0.808 at 16K context (from 0.846) | bfp8 KV cache + SDPA chunks 32/256 |
| — | 0.725 | attention norm output handed sharded to the DS qkv matmul (final; device 0.827 / 0.656) |

## 5. Sweeps (real weights; every row in `sweeps/*.jsonl`, rendered by `tools/summarize_sweeps.py` -> `sweeps/SUMMARY.md`)

| family | decision | key numbers (GDN / attention; prefill T=128 ms, traced decode ms; PCC prefill / decode) |
|---|---|---|
| precision (routed) | bfp4 / LoFi | bfp8 HiFi2 28.7 / 1.488 -> bfp4 LoFi 21.2 / 1.370 (L0), PCC unchanged to 4 decimals; bfp4 HiFi2 bit-identical to LoFi |
| shared | bfp8 / HiFi2 | LoFi and bfp4 within noise on time; HiFi2 best PCC |
| mixer_weights | attention bfp4 REJECTED (PCC prefill 0.99952 -> 0.99466 < 0.995); GDN: see gdn_dtype2 | attention bfp4 -4.7% decode but below the bar |
| gdn_dtype2 (mega weight really bfp4) | REJECTED on real weights | 1.284 -> 1.225 ms (-59 us); real-weight PCC 0.99882 -> 0.99527 / 0.99958 -> 0.99773 at 128 tokens, and 0.99304 on the last 2048 of a 65573-token real-weight prefill (bfp8: 0.99552) — below 0.995 (both the real and the synthetic variants stay in the suite as strict xfails) |
| kv, sdpa_long, kv_final | **bfp8 + SDPA chunks 32/256** | 128 context: PCC-neutral, no time signal (SDPA row 25 us); 16384 context (kv_final, real weights, full layer): bf16 0.846 ms -> bfp8 k64 0.824 -> bfp8 k256 0.808 (-4.5%), PCC 0.99962 / 0.99985 vs 0.99959 / 0.99983; k_chunk 512 invalid (static CB overflow) |
| gather (decode sparse geometry) | 64 cores, in0 32 both roles | 1.375 -> 1.358; 110-core grid illegal (dispatch cores) |
| masked / masked2 (prefill sparse geometry) | chunk 64, gate-up 32c/in0 64, down 32c/in0 32 | 21.2 -> 17.9 / 19.2 -> 15.8 ms; chunk 128 fails (CB clash / host-tensor builder bug) |
| norm | sharded decode norms | 1.369 -> 1.306 / 1.074 -> 1.002 |
| router | derived 2D config | 0.987 -> 0.924 ms |
| shared_pc | 44 cores | 0.987 -> 0.937 ms (16..88 cores within 5 us) |
| dense_1d | 1d, in 64 / out 96 cores, in0_block_w 8 (kept for the GDN mega in-proj) | GDN 1.306 -> 1.137, attention 0.987 -> 0.882; in0_block_w 16/32 slower |
| dense_ds / dense_ds2 (DRAM-sharded weights + DRAM-sharded decode kernel; 1/2/3 readers per bank) | attention qkv+o and GDN o DRAM-sharded, 1 reader; GDN mega stays 1d | attention 0.777 -> 0.731 (1 reader), 0.834 (2), 0.769 (3); attention o only 0.760; GDN o only 0.869 (from 0.884); GDN mega: 1 and 2 readers fail (in1 block CB 1.70 MB > L1 for the 81-tile bank shard; 2 readers illegal by divisibility), 3 readers 0.944 / mega-only 0.928 (slower). First attempt failed in prefill: the auto matmul rejects sharded weights and the qwen36 2D prefill config overflows L1 at full-chip N -> prefill runs on a one-time interleaved copy (see section 9) |
| gdn_state | bf16 | fp32 state + step +57 us, PCC no better at 128 context |
| mega_split (GDN mega in-proj: DS on the qkv columns [3072 x 12288] + 1D on a|b|z + concat) | REJECTED | 1 reader 0.880 ms, 2 readers 0.903, 3 readers 0.895 vs 0.869 with the 1D mega + DS out-proj |
| ds_cores (DS activation grid) | 32 cores | 16 and 8 cores: static CB overflow (wider K shards per core); 32: 0.732 ms |
| prefill_mode at 4096 (sorted cap, warm) | cap kept at 4096 | masked 517.6 ms -> sorted 244.7 ms (-53%) |
| norm_ds (attention norm output kept sharded into the DS qkv matmul) | selected | 0.731 -> 0.725 ms (-6 us, PCC identical) |
| mixer decode fidelity | LoFi (measured rows) | the qwen36 decode compute config already runs the bfp8 mixer projections at LoFi (the fastest legal fidelity; the rows are DRAM-bound at 66%), so no separate fidelity family was needed; the shared expert was swept (HiFi2 kept on PCC) |
| prefill_mode (masked vs expert-sorted dispatch) | sorted for 512..4096 tokens, masked outside | attention layer whole-layer wall, masked -> sorted: T=128 15.8 -> 38.5 ms, 256 30.7 -> 43.7, 512 62.2 -> 54.4, 1024 126.4 -> 80.6, 2048 259.0 -> 135.4 (GDN layer 2048: 244.6 -> 164.5); PCC identical to 1e-8 |
| sorted_geom (2048) | 64 cores / in0 32 both roles | 133.6-136.6 ms across the four legal geometries (noise); 96 cores illegal |
| sorted_group (rows per expert block) | 32 | 2048: 32 rows 134.6, 64 rows 161.9, 128 rows 220.2 ms; 512: 59.3 / 93.5 / 160.3 |

## 6. GDN decode structural rewrite (shared kernel; inherited by the TP path)

Op-to-code audit of the stage-2 GDN decode replay (per replay: ~426 us of structural ops next to 373 us of large matmuls):
conv step 153 us (row slices of the fused window: untilize/slice/tilize/concat + 7 tap mul/adds), head split + GVA 69 us
(three padded->dense reshapes on 4-8 cores, composite `repeat_interleave`), recurrence 145 us excluding its 3 matmuls
(85 us of padded->dense reshapes, 2 staging copies of the state, decay multiply, beta multiply, add), output gate 38 us.
Changes (`ttnn_delta_rule_ops.py`, `ttnn_gated_deltanet.py`, `qwen36/tt/gdn/{weights,decode,config}.py`, autoport state code):
* per-tap `[B,1,D]` conv buffers (`gdn_state["split"]`, refreshed from the fused window after every prefill) -> the decode
  conv is three `ttnn.copy` + `ttnn.mac` tap accumulation (also used by the mega in-projection branch, which previously
  never took the split path);
* recurrent step: all head-row tensors stay tile-padded `[B,H,1,*]`; decay applied to the read row; beta folded into the
  `[B,H,1,V]` operand; outer product as a column x row broadcast multiply (`transpose(k_row) x d_row`); the update is ONE
  `ttnn.mac(h, exp(g), outer)` written straight into the persistent buffer (`state_out`) — 2 state-sized passes instead of 6;
  the L1 budget for the state temporary counts real bytes (24 MiB);
* GVA q/k repeat as an exact one-hot expansion matmul (`gva_expand`, HiFi4 fp32-acc) at B=1;
* gated output norm on the `[B,H,1,V]` head rows with one dense reshape before the out-projection.
Result: GDN decode 1.140 -> 0.884 ms (-256 us) with decode PCC 0.99958 -> 0.99960; all GDN tests, the 9B unit suite and the
9B decode check pass. Bugs found on the way: `ttnn.mac(..., output_tensor=)` returns a new object aliasing the buffer (the
in-place wrapper must return the caller's tensor), and the fp32-byte state budget.

## 7. Rejected / blocked (evidence in `work_log.md` and the sweep rows)

* attention projections bfp4 (real weights): PCC 0.99466 < 0.995 (OPT-007 trial recorded).
* GDN mega in-projection bfp4: -59 us but real-weight PCC 0.99527 / 0.99773 at 128 tokens and 0.99304 at 65573 tokens
  (below the bar; the bfp8 policy gives 0.99552 on the same real-weight run).
* DRAM-sharded GDN mega in-projection: 1/2 readers overflow L1 (81-tile bank shard), 3 readers slower than the 1D config; the
  adapted split candidate (DS on the qkv columns + 1D on a|b|z + concat) is also slower (0.880-0.903 vs 0.869 ms).
* DS activation grids of 16 / 8 cores (wider K shards): static CB overflow.
* DRAM-sharded prefill matmul on the sharded weight itself: auto config rejects sharded in1; the 2D config overflows L1 at the
  single-chip widths (CB 1.70 MB > 1.5 MB) -> prefill uses an interleaved copy (memory cost in section 9).
* masked chunk 128: static CB clash at gu64/in96; smaller blocks hit a host-tensor builder bug at a single chunk (noted).
* SDPA-decode k_chunk 512: static CB overflow.
* 110-core sparse grid: dispatch cores inside the 11x10 grid.
* fp32 GDN state: +57 us, no PCC gain at 128 context (kept as a candidate for the long-context full-model checks).
* 1D in0_block_w 16 / 32 for the dense decode matmuls: slower than 8 (mcast_in0 streams the full K per core).

## 8. Watcher / stress / fallback audit

* Watcher (`TT_METAL_WATCHER=10`, `TT_METAL_WATCHER_APPEND=1`, no profiler): real-weight T=37 prefill+decode both kinds, attention
  batched decode B=32, chunked 2048+2048+333 prefill both kinds (exercises the expert-sorted dispatch), GDN traced decode +
  determinism: 6/6 PASS (final policy and code, 07:55), `watcher.log` 19,354 lines, 0 exception / fatal / sanitizer / overflow (`watcher/`,
  `~/qwen35-122b-bringup/artifacts/03-watcher/`).
* Stress: `test_opt_stress_repeated_runs` (20 prefill+decode cycles over lengths 5..1500 per kind, bit-identical repeats).
* Runtime fallback audit: `test_opt_runtime_fallback_audit` — 0 `from_torch`/`to_torch` in aligned prefill, chunk continuation,
  640-token expert-sorted prefill and decode, both kinds (the dispatch index math runs entirely on device).

## 9. Limitations and what the next stages inherit

* Single-device batched decode uses the qwen36 `[1,B,1,dim]` layout (one tile per user): batched matmuls carry 32x row padding
  and the sharded norm / 1D configs apply to batch 1 only. The (1,4) TP modules use `[1,1,B,dim]`; batched performance is a
  stage 4/5 topic.
* GDN chunked-prefill continuation is single-user (asserted); multi-user continuation belongs to the full-model stage. Decode
  advances the per-tap conv buffers only; a continuation rebuilds the fused conv window from them (tested: 128 prefill, 128
  decode steps, 100-token continuation vs HF).
* DRAM-sharded decode weights (attention qkv/o, GDN o) need an interleaved copy for prefill on this single-chip path: +84 MB per
  attention layer, +27 MB per GDN layer (~2 GB for 48 layers). The (1,4) TP modules already hold DRAM-sharded weights with
  prefill program configs that fit their N/4 widths, so the copy is a single-device-only cost.
* Shared-kernel budget change (`_DECODE_STATE_L1_BUDGET_BYTES` 96 MiB at 4 bytes/element -> 24 MiB at the real element size):
  for the dense 9B (fp32 state, 1 MiB/user) the outer-product temporary moves to DRAM above 12 users (was 48). 9B unit and
  decode checks pass at B=1/4; flagged for the qwen36 owners as a perf-only change at large batch.
* Non-aligned GDN prefill keeps the stage-1 host-built valid_len selector (2 `from_torch` per call, accepted boundary; every real
  prompt is non-aligned, so the full-model stage should build the selector on device).
* The 9B B=4 paged-decode check (`tools/check_qwen36_9b_decode.py`) last ran at 05:04, before the final GDN kernel edits; the 9B
  unit suite (incl. GDN decode) re-ran at 06:08 after them. Re-run the B=4 check before merging to main.
* Padded->dense tiled reshapes run on as many cores as output tiles (`reshape_tiled` splits over output pages): the remaining
  q/k/v head-split reshapes (~40 us) need a tt-metal kernel change.
* Prefill: the expert-sorted dispatch has a static per-expert padding upper bound (31 rows x 256 experts) and a ~70-op index
  build (10.7 ms device at 2048 tokens in the 05:26 profile — sort 3.6, gather 2.2, four single-core scatters 3.1; the later
  constant cache removed six host tensor creations per call, not device time), so it only wins above 512 tokens;
  below it the masked union path stays. The sparse kernel streams an expert's weights once per 32-row group (~19 us per
  (group, expert) pair at bfp4), so the cost follows the pair count; per-expert weight reuse across consecutive groups and a
  multi-core scatter are the next levers (tt-metal kernel work). Prefill chunks of 512 tokens carry ~60% padding groups; the
  full model should prefill in 2048-token chunks where possible.
