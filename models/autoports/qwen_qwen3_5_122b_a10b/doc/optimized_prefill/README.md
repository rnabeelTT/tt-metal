# Optimized prefill (stage 12) — group-parallel MoE experts for Qwen/Qwen3.5-122B-A10B on the (1,4) Blackhole mesh

Goal: the served prefill of the shipped autoport ran at ~1000 tokens/s (TTFT 132 s for a 131072-token prompt, 283 s at
262080), which makes long prompts impractical. This stage parallelizes the prefill-time MoE block, which dominated the
2048-token layer step, without changing the served numerics contract (same expert weights, same bf16 activations, same
routing). Accuracy evals are not rerun (user decision); correctness is guarded by the layer PCC suite against HF, the
run-to-run determinism test and the full-model readiness prefill check (top-k against the stage-6 HF references).

## 1. Diagnosis (Tracy, one die, one GDN layer, 2048-token prefill segment; `scripts/12-tracy-moe.sh`, TAG=before)

`doc/optimized_prefill/tracy/before/moe_2048_ops.csv` (raw `ops_perf_results`, second half = the profiled call).

| MoE phase (per layer, 2048 tokens, device 0) | device time | cores | what it was |
|---|---|---|---|
| expert-sorted dispatch bookkeeping (`SortedDispatch.build`) | ~35 ms | mostly 1-5 | 4 x `Permute` 14.2 ms, `cumsum` (Accumulation) 10.8 ms on ~4.5 cores, 40 x `BinaryNg` 4.8 ms, 3 x 1-core `Scatter` 2.4 ms |
| gather rows (`ttnn.embedding`) | 1.8 ms | | |
| gate/up `sparse_matmul` (masked, nnz = G) | 8.7 ms | 32 | multicast-1D factory: every core iterates ALL groups, cores only split N (N = 2 x 1024 per die = few tiles) |
| swiglu | 0.6 ms | | |
| down `sparse_matmul` (masked, nnz = G) | 3.8 ms | 32 | same factory, N = 3072 |
| undispatch rows | 2.5 ms | | |
| **MoE block** | **~53 ms of the ~50 ms device step** (the rest of the layer overlaps) | | layer wall 109.5 ms untraced (GDN), 106.8 ms (attention) |

Two structural problems, both in the expert-sorted prefill path (`OptimizedMoE._routed_sorted`, T >= 512):

1. `ttnn.sparse_matmul` has a single program factory (`SparseMatmulMultiCoreReuseMcast1DProgramFactory`): the activation
   is multicast to a row of cores that split the N dimension. With tensor-parallel experts the per-die N is 2048 (gate/up)
   or 3072 (down) -> 32 cores at most, and every core walks all G groups serially: the work is O(G) per core, the grid is
   ~30 % used.
2. The counting sort that groups the dispatched rows by expert used `ttnn.permute` (4 transposes of a [Rp, E] one-hot)
   and `ttnn.cumsum` along the token axis, which runs on a handful of cores.

## 2. Changes

### 2.1 Group-parallel sparse matmul factory (ttnn, C++)

New `SparseMatmulMultiCoreReuseGroupedProgramFactory` (`ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/
sparse_matmul_multicore_reuse_grouped.{hpp,cpp}`) plus the in1 reader / output writer kernel
`kernels/dataflow/reader_writer_bmm_tile_layout_in1_indexed.cpp`. Selected when `ttnn.sparse_matmul` receives a
`MatmulMultiCoreReuseProgramConfig` together with the `indices` operand (`SparseMatmulDeviceOperation::
select_program_factory`; the device-op framework needs `select_program_factory` once `program_factory_t` has two variants):

* inputs `a [1, G, M, K]` (compact activation groups), `b [1, E, K, N]` (expert weights), `indices [1, G] uint16`
  (expert id of every group, device resident: it is the routing of the very forward pass, read by the kernel at run time,
  so the program is trace safe), output compact `[1, G, M, N]`;
* `split_work_to_cores(grid, G, row_wise=true)`: WHOLE groups per core over the full compute grid (110 cores on p300);
  each core reads its groups' activation block (in0 reader `reader_bmm_tile_layout_in0.cpp`, batch = groups of the core)
  and the whole `[K, N]` block of each group's expert from DRAM (no multicast), K stepped in `in0_block_w` tiles, the
  standard `bmm_large_block_zm_fused_bias_activation.cpp` compute kernel with `batch` = groups per core;
* numerics: the same K-blocked bf16 accumulation (`packer_l1_acc`) as the multicast factory; the unit test measures the
  grouped result CLOSER to the fp32 torch reference than the multicast result at every shape (PCC vs torch 0.99996 vs
  0.99981 for bfp4 gate/up; 0.99987 vs 0.99972 for bfp8);
* the op's indexed mode previously meant "one activation block x num_active distinct experts" (decode design, output rank
  `a_batch + b_batch + 2`); the grouped mode maps group g to `indices[g]` (several groups may share an expert, G may exceed
  E) and returns the compact `[1, G, M, N]` spec (`compute_output_specs`, optional-output validation).

Validation: `tests/test_sparse_matmul_grouped.py` (one die + the (1,4) mesh; PCC vs torch and vs the multicast masked
path, odd group counts (core groups 1 and 2), bfp4 and bfp8 weights, G > E, perf vs multicast).

| unit case (one die, host wall of 5 iterations) | multicast masked (`nnz = G`) | grouped (`in0_block_w` 4 / 8 / 16) | speedup |
|---|---|---|---|
| gate/up: 512 groups x [32, 3072] x [3072, 1024] bfp4 | 12.02 ms | 3.61 / 3.64 / 3.67 ms | 3.3x |
| down: 512 groups x [32, 512] x [512, 3072] bfp4 | 37.86 ms | 2.52 / 2.53 ms (16: exceeds L1) | 15.0x |

Bug found and fixed on the way: `split_work_to_cores` orders its core groups column-major by default while the runtime
args were assigned row-major, so any core owning more than one group ran a compute kernel built for a different batch
and hung (bisected with `scripts/12-grouped-case.py`: 64 groups pass, 130 hang).

### 2.2 Matmul counting sort (`tt/moe_dispatch.py`, `DISPATCH_MODE = "counting_mm"`)

The running count of the one-hot `[Rp, E]` routing matrix along the token axis (what `ttnn.cumsum(dim=2)` computed on a
few cores) is now two small fp32 matmuls: within blocks of `Bk` rows `L_tri @ oh_block` (batched over the blocks, all
cores) plus the exclusive prefix of the block totals `P_excl @ totals` broadcast over the block. `Bk` is the smallest tile
multiple giving at most 32 blocks; `Rp` is zero-padded to `nb * Bk` when needed (T = 500: 4000 rows -> 4096) and sliced
back. Counts are small integers accumulated in fp32, so the result is exact (`test_dispatch_modes_equivalent`: bit-equal
index tensors, weights, group ids and permutation cover for T = 256 / 500 / 513 / 1000 / 1023 / 2048).

### 2.3 Layer wiring (`tt/optimized_decoder.py`)

`OptimizationConfig.sorted_grouped` (env `QWEN35_SORTED_GROUPED`, default on), `grouped_gate_up_in0` / `grouped_down_in0`
(K block, default 4; the sweep above is flat between 4 and 16) and `grouped_subblock_w`. `_routed_sorted` passes
`indices = group_ids` (uint16 `[1, G]`, produced by the dispatch) and the grouped configs (`_grouped_configs`: per_core_M =
group rows / 32, per_core_N = all N tiles, K block shrunk to the L1 budget). The masked path (`nnz = G`) remains one env
flip away for A/B runs.

## 3. Results

### 3.1 Layer level (Tracy, device 0 of the (1,4) mesh, one GDN layer, one 2048-token segment)

`tracy/moe_2048_ops_before_after.md` (from `tracy/{before,after}/moe_2048_ops_slim.csv`, `scripts/12-ops-compare.py`):

| MoE op (device 0, per layer at 2048 tokens) | before: count / ms (avg cores) | after: count / ms (avg cores) |
|---|---|---|
| `SparseMatmul` (gate/up + down) | 2 / 12.16 (32) | 2 / 4.68 (110) |
| `Permute` (one-hot transposes of the counting sort) | 4 / 14.21 (84) | 2 / 0.13 (59) |
| `Accumulation` (`cumsum`) | 2 / 10.77 (4) | 1 / 0.14 (1) |
| `BinaryNg` (dispatch arithmetic) | 40 / 4.82 (110) | 41 / 4.52 (110) |
| `Scatter` (inverse permutations, ROW_MAJOR, 1 core) | 3 / 2.40 (1) | 3 / 2.40 (1) |
| `Embeddings` (gather / undispatch rows) | 10 / 2.11 (69) | 10 / 2.12 (69) |
| `Matmul` (router, block-triangular counts) | 8 / 1.38 (87) | 11 / 2.69 (65) |
| everything else | 95 / 5.1 | 103 / 6.6 |
| **MoE block** | **164 ops / 52.98 ms** | **173 ops / 23.24 ms** (2.28x) |

Host wall of the MoE phases with a sync after each (`test_perf_moe_prefill`, `PERF_MOE_PHASE`): dispatch build 19.19 ->
6.66 ms, gather 1.75 -> 1.77, gate/up 8.68 -> 2.87, swiglu 0.55 -> 0.52, down 3.83 -> 2.75, undispatch 2.53 -> 2.52 ms.
Whole-layer 2048-token prefill host wall (untraced, `test_perf_prefill_2048_whole`): GDN 109.5 -> 89.6 ms, attention
106.8 -> 86.2 ms (the untraced wall is host-bound; the served path replays traces, see 3.2). Captured device time of the
whole-layer runs (all rows, device 0): GDN 115.3 -> 57.7 ms, attention 108.7 -> 51.0 ms.

What remains in the MoE block after this stage (candidates for a later pass, all in the dispatch bookkeeping): the 41
elementwise ops on the `[Rp, E]` fp32 one-hot (4.5 ms), the three 1-core ROW_MAJOR `ttnn.scatter` inverse permutations
(2.4 ms), the gather / undispatch embeddings (2.1 ms). The expert matmuls themselves are now 4.7 ms for 2 x 512 groups
(gate/up [32, 3072] x [3072, 2048] bfp4 and down [32, 1024] x [1024, 3072] bfp4 per group per die), 110 cores.

### 3.2 Full model (48 layers, (1,4) mesh)

Traced prefill of the 48-layer generator (`tools/prefill_bucket_bench --layers all --modes dev,traced`, 3 timed
repetitions, median of the traced replays; before = stage-7 `doc/optimized_full_model/prefill_bucket_bench_48layers.json`,
after = `prefill_bucket_bench_48layers_after.json`):

| prompt tokens | before (traced) | after (traced) | after, untraced dev path | prompt tokens / s after |
|---|---|---|---|---|
| 128 (masked path, < 512: unchanged by design) | 344.0 ms | 343.8 ms | 354.7 ms | 372 |
| 161 | 555.8 ms | 344.9 ms | 352.8 ms | 467 |
| 2048 (one sorted segment) | 1934.8 ms | **988.5 ms** | 992.2 ms | **2072** |
| 4097 (2 segments + 1-token tail) | - | 2208.7 ms | 2223.1 ms | 1855 |
| 8192 (4 segments) | - | 3977.5 ms | 3988.3 ms | 2060 |

Correctness through the grouped path on the full model (`tools/readiness prefill-check`, top-k agreement of the device
logits with the bf16 HF reference over 100 positions; bars top5 >= 0.98, top100 = 1.0; `readiness/prefill_check_*.json`):

| reference | prompt | top1 | top5 | top100 | stage-7 value (masked path) |
|---|---|---|---|---|---|
| `qwen35_122b_book_p2048_g100` | 2048 book tokens (sorted segment) | 0.99 | 1.00 | 1.00 | - (p128: 0.97 / 0.99 / 1.00) |
| `qwen35_122b_book_p8192_g100` | 8192 book tokens (4 sorted segments) | 0.95 | 1.00 | 1.00 | same-day control with `QWEN35_SORTED_GROUPED=0`: 0.95 / 1.00 / 1.00 (identical; `prefill_check_..._grouped0.json`) |
| `qwen35_122b_aime24_chat_g100` | AIME chat prompt (< 512 tokens: masked path, unchanged) | 0.97 | 1.00 | 1.00 | 0.97 / 1.00 / 1.00 |

Served (vLLM through the plugin, `run_vllm_server --stages serve` held server from the checkout, 32 slots, 262144
context, block 64, trace region 768 MiB; `scripts/12-serve-chain.sh`). "before" = the stage-11 TTI benchmark rows
(`doc/tti_release/RUN_NOTES.md`, same server settings) and, for the long prompts, the same-day A/B with
`QWEN35_SORTED_GROUPED=0` (`served_before/direct_long.json`); "after" = `served_after/` (readiness benchmark runner for
128 / 128, `scripts/12-direct.py` streaming greedy completions with 64 new tokens for the long prompts, best of 2):

| workload | stage 11 (masked matmuls, `counting_h`) | A/B: masked matmuls + `counting_mm` (`QWEN35_SORTED_GROUPED=0`) | after: grouped + `counting_mm` | prompt tok/s after |
|---|---|---|---|---|
| 128 / 128, 1 user: TTFT p50 / TPOT mean | 394-398 ms / 37.7 ms | 372 ms / 37.7 ms | 423 ms / 37.7 ms (26.5 t/s/u) | short prompts take the unchanged masked path (T < 512); 372 vs 423 ms is server-instance variance (4 requests, p99 405 / 470) |
| 128 / 128, 32 users: TTFT p50 / TPOT mean | 12.3 s / 127.0 ms | 12.05 s / - | 12.1 s / 132.1 ms (7.6 t/s/u) | unchanged |
| 8192-token prompt, 1 user: TTFT | 7.8 s | 5.33 s | **4.01 s** | 2044 |
| 32768-token prompt, 1 user: TTFT | 31.4 s | 21.6 s | **16.4 s** | 1999 |
| 131072-token prompt, 1 user: TTFT | 132.6 s | 93.5 s | **72.3 s** | 1812 |

The A/B column isolates the two changes: the matmul counting sort alone takes the 131072-token TTFT from 132.6 to
93.5 s, the group-parallel expert matmuls take it from 93.5 to 72.3 s (the dispatch mode is a module default,
`QWEN35_DISPATCH_MODE=counting_h` restores the stage-11 sort). The greedy completions of all three variants are
byte-identical at every length.

The 131072-token greedy completion is also byte-identical to the stage-10 completion of the same prompt
(`doc/optimized_vllm/direct_checks/direct_checks_stage10.json` `long_131072`: "..., and their heated faces were distorted
and convulsed. The teeth were bared, glaring like ..."), and the 8192 / 32768 completions are stable across repetitions.
Prefill throughput falls from ~2040 tok/s at 8K to ~1810 tok/s at 128K because attention over the growing KV (chunk-outer
SDPA per 2048-token segment) is now a visible share of the segment; the MoE block no longer dominates.
