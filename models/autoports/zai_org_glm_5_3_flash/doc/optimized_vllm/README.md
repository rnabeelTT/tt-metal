# GLM-5.3-Flash (E192) optimized vLLM serving (stage 10)

## Headline: primary single-user vLLM serving (default profile p300x2: 16 slots x 243,712)

Workload: `vllm bench serve`, random dataset, 128 input / 128 output tokens, 1 request, concurrency 1, temperature 0,
`--ignore-eos`, `sample_on_device_mode=all`, mesh (1, 4) P300x2, served profile p300x2 (max-num-seqs 16,
max-model-len 243,712, replicated DSA cache). Before: the stage-9 A6 readiness benchmark on p300x2
(`doc/vllm/evidence/bench_p300x2.log.txt`). After: the same command and profile on the stage-10 HEAD in the combined
stage-9 ticket, run on the code freeze 0a2663d0b6b (`doc/vllm/evidence/final/p300x2_vllm_benchmark.json`, server
command `doc/vllm/evidence/final/p300x2_servercmd.txt`; the p300x2 max-model-len after the G9 re-probe is 239,616).
Why these numbers also hold for the shipped 0bcf89f7fd6:
- 0bcf89f7fd6 changes the MoE bucket paths only where the build's own path is sorted, which is 32 slots. At 16 slots and
  below the family is pairs, the same paths as 0a2663d0b6b.
- Its dealloc fixes change only frees: `force=False` on the pairs ids loop and `del q_rm` in the sharded path.
- t19 ran the probe and the invariance checks on the fixed tree.
- Wide (32 slots) was re-measured on the fix (wide-r2, 3872b583092).
The slot-aware trace-region fit (044dbeaa3fd) picks the same bucket lists for p300x2 and long.

The after values below were measured with the multi-slot DSA decode buckets on. The shipped code turns them off at
S >= 2 for served batch invariance (see "Batch invariance of the DSA width buckets"). The p300x2 TPOT is estimated to
rise by about 4 to 5 ms (not measured); the final scorecard run measures it.

Settings that changed between before and after:
- Async decode is on (stage 9, 4678d8fccf4; the before runs had it off). It moves the TTFT, see "Async decode and
  TTFT" below.
- The p300x2 trace region is 96 MiB (was 64 MiB).
- The p300x2 max-model-len is 239,616 (was 243,712).

| Metric | Before | After |
|---|---|---|
| TTFT (median) | 263.0 ms | **265.3 ms** |
| TPOT (mean) | 119.27 ms | **69.91 ms** |
| ITL (median) | 119.04 ms | **69.90 ms** |
| Decode t/s/u (1000 / TPOT) | 8.38 (the A6 sweep's 128 / 128 row: 9.10; run-to-run range 8.38-9.10) | **14.30** |
| Output throughput | 8.31 tok/s | **14.0 tok/s** |

Secondary, the CI serving burst on p300x2 (100 input / 100 output tokens, 32 prompts; capacity and nightly parity, not
the decode headline). Before (`doc/vllm/evidence/bench_ci_p300x2.log.txt`): TTFT median 14,485 ms, TPOT mean 179.6 ms,
output 74.6 tok/s. After (`~/glm53-flash-bringup/logs/09/final/readiness-p300x2/vllm_ci_serving_benchmark.json`): TTFT
median 14,164 ms, TPOT mean 172.5 ms, output 76.8 tok/s.

Separate row, profile p300x2-1m (1 slot, 1,048,576, block-sharded DSA cache). The DSA and MoE buckets are off there by
design (sharded layout, 1 slot), so the after value reflects async decode / stage-9 changes only.
- Before (`readiness_vllm/vllm_benchmark.json`, async off): TPOT 73.54 ms, TTFT 234.5 ms.
- After (`doc/vllm/evidence/final/1m_vllm_benchmark.json`, server `1m_servercmd.txt`, async on): TPOT 71.52 ms
  (13.98 t/s/u), TTFT 290.0 ms. CI burst: TPOT 71.38 ms, TTFT median 113,622 ms.

Full-model comparison for comparable work (standalone traced decode, token-out):
- 1 slot: 50.10 ms (280,576);
- 16 slots, 1 active: 72.9 ms with the MoE row buckets (106.3 ms before; t15b).
The served TPOT of the same build should match these within the host overhead:

| Comparable work | Standalone traced token-out | Served TPOT, 1 user |
|---|---|---|
| p300x2 (16 slots, 243,712), 1 active | 72.9 ms (t15b, 16 slots at 16,384; DSA width bucket 4,096 at short positions) | **69.91 ms** (DSA buckets on; see above) |
| p300x2-long (1 slot, 325,632) | 50.10 ms (t5, 280,576, position 300) | **49.84 ms** (async decode on) |

Non-aligned prompt lengths stay supported. The combined serving ticket's smokes send NONALIGNED prompts of 1 / 3 / 2,381
/ 4,223 tokens (`~/glm53-flash-bringup/logs/09/s9_final.log`, NONALIGNED smoke). The reduced adapter validator (t16) passes on this code.

## Problem

Each of the 11 DSA layers selects the top-512 pools of every query. Before this change, the selection scored the
complete pool-slot range of a slot, `P_alloc = T/4 + 32` pools (`T` = `max_seq_len`):

- Decode: `dsa_decode_scores` wrote `[S, P_alloc]` scores (it already skipped the reads past each slot's limit).
  Then the iota mask, the `where` and the top-512 ran over `P_alloc` columns.
- Prefill: every sparse chunk (positions past 2,050) scored `P_alloc` keys per row and ranked them.

The vLLM adapter builds the generator with `max_seq_len = model_context(max_model_len)`, so serving paid the same cost.

Measured cost of the decode selection per DSA layer (synthetic keys, traced, one die, `tools/dsa_bucket_probe.py`,
`~/glm53-flash-bringup/logs/s10/probe1.log`):

| Scored width (pools) | 1,056 | 2,080 | 4,128 | 8,224 | 16,416 | 32,800 | 65,568 | 70,176 |
|---|---|---|---|---|---|---|---|---|
| 1 slot, ms | 0.047 | 0.061 | 0.084 | 0.105 | 0.170 | 0.277 | 0.484 | 0.537 |
| 8 slots, ms | 0.070 | 0.078 | 0.094 | 0.116 | 0.184 | 0.299 | 0.529 | 0.585 |

`dsa_decode_scores` itself is 0.01 to 0.05 ms of this. The mask and the top-512 over the width are the cost. At
`max_seq_len` 280,576, 11 x (0.537 - 0.084) ms = 5.0 ms per token. This is the measured 51.1 vs 56.2 ms gap.

## Design (`tt/dsa_buckets.py`)

Bucket contexts `T_b` = 4,096 x 2^k below `T`, then `T`. Bucket width `W_b = T_b/4 + 32` pools: the `P_alloc` of a
build with `max_seq_len = T_b`. A query at position `p` needs the pools `[0, (p+1)//4)`. The smallest `W_b` that holds
the largest need of the batch gives every row its complete candidate set.

Decode (traced, static shapes per trace):
- The decode step is captured as segments. At each DSA layer, the hook (`GlmDSA.decode_forward`, `decode_select`)
  ends the running main trace and captures the selection once per bucket width. The selection writes the top-512
  pool ids into a persistent per-layer buffer (`_sel_buf`, uint32). Then the next main trace begins.
- A replay runs `main_0, sel_0[b], main_1, ..., sel_10[b], main_11`. The host selects `b` from the positions mirror
  (`DeviceDecodeInputs.positions`). Only the selection varies with `b`.
- Trace region: 44.1 MiB for 12 segments + 11 x 8 selection traces at 280,576. Before: 40.4 MB for the single trace
  and the sampler traces, in the 64 MiB region. One selection trace is 8 to 12 KB.
- Every bucket program compiles in `install()`, before any capture: an eager decode step that runs every width, and
  one sparse prefill per chunk length at every forced width. The program cache stays constant after the capture.
- The step output (logits) is allocated while the earlier segments are live traces. It is marked corruptible for
  the trace-allocation tracker: it is read only inside its own step (sampler trace, host logits).
- `GLM53_DSA_BUCKET_MIN_T` (default 65,536): a smaller `max_seq_len` keeps the single decode trace. At 16,384 the
  segment replays cost 0.1 to 0.45 ms per token, more than the narrower selection saves (table below).

Prefill (eager): a sparse chunk takes the smallest `W_b` that holds the pools of its last position.
`indexer_score_dsa` gets `kv_len = W_b` and `chunk_start_idx = W_b - 32` (= `T_b/4`, the causal start of a `T_b`
build). The selection then ranks only the first `W_b` columns.

Switches:
- `GLM53_DSA_BUCKETS=auto` (default) | `0` (the full-width code, unchanged) | a comma list of bucket contexts;
- `GLM53_DSA_BUCKET_MIN_T`.

The block-sharded DSA layout (`GLM53_DSA_SHARD`, the 1m profile) takes no DSA decode buckets and no prefill views:
`bucket_contexts(T, sharded=True)` returns None, and the sharded pool allocation does not set the views (see the next
steps).

API change: with buckets on, `Glm53Generator.trace_id` is a `DsaBuckets` object. Replay through
`dsa_buckets.replay_decode(g)`. A direct `ttnn.execute_trace(mesh, g.trace_id)` fails. `tools/s7_probe.py` uses
`replay_decode`, and so does `tools/model_step_probe.py`.
`tools/dtype_sweep.py` counts every segment replay, so its `trace_verified` flag reads false (2,376 = 99 x 24).

## Results (full model, 1 slot, default precision, `tools/dsa_bucket_check.py`)

Token-out per step: model trace + sampler trace replays, 1 active slot, 20 steps. Ticket `t5`
(`~/glm53-flash-bringup/logs/s10/t5.log`, snapshot `s10_snap4` = 24be34c7cbe + the stage-10 hunks).

| Position | 280,576 before | 280,576 after | 16,384 single trace (t4) |
|---|---|---|---|
| 300 | 55.07 ms | **50.10 ms** (bucket 4,096) | 49.95 ms |
| 5,000 | 58.43 ms | **53.54 ms** (8,192) | 53.23 ms |
| 20,000 | 58.44 ms | **54.05 ms** (32,768) | - |
| 70,000 | 58.47 ms | **56.04 ms** (131,072) | - |
| 140,000 | 58.58 ms | 58.41 ms (262,144) | - |
| 270,000 | 58.74 ms | 59.17 ms (280,576) | - |

The token-out at 280,576 is now 0.15 to 0.3 ms from the 16,384 figure up to position 5,000. It grows with the position
as the bucket grows. At the full width the segment replays cost +0.43 ms.

TTFT (warm prefill to last-row device logits):

| Prompt | 280,576 before | 280,576 after | 16,384 (t4) |
|---|---|---|---|
| 128 | 229 ms | 229 ms | - |
| 2,048 | 1,029 ms | 1,026 ms | 1,025 ms |
| 8,192 | 6,456 ms | **5,826 ms** (-9.8 %) | 4,410 ms |
| 32,768 | 28,186 ms | **25,255 ms** (-10.4 %) | - |

The 8K TTFT gap that remains (5,826 vs 4,410 ms) comes from the prefill indexer. `indexer_score_dsa` compiles its
work split for the length of the pool-key view (`P_alloc`). `kv_len = W_b` limits the written columns but not the work:
5.34 vs 5.60 ms per 256 rows at a 262,144 view, and 1.38 ms at a 65,536 view (ticket t9). The indexer row block is not
the cause: the 16,384 build with the 280,576 row block (256 rows) gives 4,448 ms (ticket t8). Proposed follow-up: size
the pool-key cache to a smooth block count and give each chunk a view whose length is a divisor of it, near `W_b`.

16,384 build, single trace vs forced segments (ticket `t4`, `GLM53_DSA_BUCKET_MIN_T=0`):

| Position | single trace | segmented |
|---|---|---|
| 300 | 49.95 ms | 50.08 ms |
| 5,000 | 53.23 ms | 53.45 ms |
| 15,000 | 53.24 ms | 53.69 ms |

At 16,384 the segmented capture was bit-identical to the single trace in all 18 tokens and logits (prefill included).
The regression decided `GLM53_DSA_BUCKET_MIN_T` = 65,536.

## Correctness

| Gate | Result | Evidence |
|---|---|---|
| Teacher forcing, 3 references, 280,576, buckets on (tracker on) | 92/100/100, 95/100/100, 92/100/100 (prefill check 92, 95, 91): the same as buckets off | t5 `tf_on`, `tf_off` |
| Trace-allocation tracker (`TT_METAL_TRACE_ALLOC_TRACKING=1`) | clean: no unsafe allocation at any segment replay (reduced t2 and t3, full-model teacher forcing t5) | t2, t3, t5 logs |
| Program cache after capture | constant in every record (reduced and full model) | `RECORD DONE program cache stable: True` |
| vLLM adapter validator (`09_validate_adapter.py`, buckets forced on at 8,192) | PASS, PROGRAM_GUARD 874 = 874 | t6 |
| Pool-key alias (below), on vs off | bit-identical, 27/27 | t7 |
| Reduced probe, buckets on vs off at the same `T` (280,576 1 slot; 65,536 8 slots) | top-1 agreement 89/91 and 13/20 (random-token prompts); max logit delta 0.03 to 1.8 | t2 |
| Reduced probe, buckets on (every prefill at `W_b`, decode inside bucket b) vs off built at `max_seq_len = T_b`, one request per process | **bit-identical**: 1 slot buckets 4,096 / 8,192 / 16,384 / 65,536 (top-k fused and classic modes), 8 slots buckets 8,192 / 16,384 | t3, t10 |
| Same, three requests in sequence in one process (6,000 / 7,000 / 4,400 tokens, bucket 8,192) vs the same sequence off at 8,192 | **bit-identical** 27/27 | t11 |

The same top-1 and logits as off at the same `T` are not expected. The DSA selection of the current code already
depends on `max_seq_len`:
- `ttnn.topk` (the `topk_large_indices` composite) is not invariant to trailing masked columns. On the same scores
  and limit, the order of tied bf16 scores, and sometimes the set at the 512 cut, changes with the total width. It
  changes when the top-k changes between its fused mode (at most 32 chunks of 512) and its classic mode. Probe1,
  synthetic, pos 2,100: width 1,056 vs 70,176 gives a different order. So the current code built at 16,384 and the
  same code built at 280,576 already select differently for the same prompt.
- The indexer scores outside the causal band do not depend on the view length or on `kv_len` (ticket t9: bit-equal
  for views 65,536 / 262,144 against 4,096 to 16,384, 256 and 2,048 rows). The index row block does not change the
  selection (t3b).
- Ticket t3 ran several requests in one process against single-request references. Only its first request
  matched. `dsa_bucket_check record` draws every request's prompt from one seeded generator, so in a multi-request run
  a later request has a different prompt than the single-request reference (harness artifact, found by the request
  isolation agent). The t10 single-request runs and the t11 same-order multi-request run compare equal prompts.

## Pool-key alias (`tt/dsa.py`, replicated layout)

The prefill indexer views the pool keys as `[KB, 1, P_alloc, 128]`, so `KB = ceil(NB * 32 / P_alloc)`. A vLLM pool
at the context cap has `NB - 1 = MB` blocks (vLLM's null block 0 included) plus the model's dummy block. That one extra
block made `KB = 2`: a whole extra `P_alloc` of pool keys per layer. The measured 1-slot serving profile (2,546 blocks
at 325,632) hits this case: 20.8 MB per layer, 229 MB per die.

Fix, the same rule as the sharded layout (`tt/dsa_shard.py`): at the cap, the dummy block's pool-key tile aliases
null block 0. Block 0 is never a block of a request. The dummy only takes padding writes and overrun reads, and the
indexer limit masks both. The latent, the host page table and the decode-input table keep the dummy.
`GLM53_POOL_ALIAS=0` turns the alias off (A/B only).

Ticket t7: reduced model, vLLM-style pool (1 slot, 65,536, num_blocks = 513):
- `KB` 2 -> 1, pooled keys 8.40 -> 4.20 MB per layer;
- tokens and logits bit-identical to the alias off (27/27, prefill and traced decode, 3 prompts).

The capacity gain enters the serving profile through a G9 re-probe with this code (stage 9 owner).

## Evidence note

- Primary evidence for the served numbers is the benchmark JSON and logs in `doc/vllm/evidence/final/`, with the server
  commands in `*_servercmd.txt` (`p300x2_servercmd.txt`, `wide-r2_servercmd.txt`, `long-async{0,1}_servercmd.txt`,
  `1m_servercmd.txt`).
- The `wide_*` files without `-r2` (`wide_smoke.log.txt`, `wide_sweep.json`, `wide_vllm_benchmark.json`, ...) and their
  qualitative copy were measured BEFORE the batch-invariance fix. They are stale (4 of 32 burst identity). Only the
  `wide-r2_*` files are final evidence for wide.
- No device profiler ran on the serving path. `perf_summary.json` records that (device-time fields null).

## Files

- `tt/dsa_buckets.py` (new): DSA width buckets, MoE row buckets (compaction, `fit_row_buckets`), warm-up, segmented
  capture and replay, `device_tensors()`, `release()`.
- `tt/dsa.py`: the decode selection hook, the prefill scored width and pool-key views (`PM`, `view_ds`), the pool-key
  alias, the shared pool iotas.
- `tt/serving_profiles.py`, `tools/serving_profiles.py`, `serving_profiles.json`, `tt-model.yaml`: the per-profile
  trace region.
- `tt/generator_vllm.py`: the trace-region check per slot count.
- `tt/generator.py`: `install` before the capture, `capture_decode`, `replay_decode`, `release_decode`.
- `tools/s7_probe.py`, `tools/model_step_probe.py`: replay through `replay_decode`.
- `tools/dsa_bucket_probe.py`, `tools/dsa_bucket_probe2.py`, `tools/dsa_bucket_check.py`, `tools/moe_bucket_check.py`
  (new): the evidence tools. `tests/test_dsa_buckets_host.py` (new): host tests.
- `work_log.md`: tickets, results, commits.
- Ticket scripts and logs: `~/glm53-flash-bringup/scripts/s10_*.sh`, `~/glm53-flash-bringup/logs/s10/`.
  Records: `~/glm53-flash-bringup/artifacts/s10/`. Summary: `doc/optimized_vllm/dsa_buckets_summary.json`.

## 16-slot decode cost (profile, ticket t12)

Deviation, recorded: the stage-10 goal forbids Tracy / device-profiler collection. t12 is one Tracy run on the
STANDALONE reduced generator (not the serving path). It only located where the 16-slot cost sits. Every stage-10
claim rests on benchmark / token-out JSON (t5, t13, t14, t15b and the combined serving ticket), not on this profile.

Served p300x2 (16 slots, 243,712) gives TPOT 110 to 119 ms at 1 active request. A 1-slot build gives 49.9 ms. Tracy
profile of one decode step: reduced model (layers 0, 3, 4 + head + sampler), max_seq_len 65,536, position 300, 1
active slot. Compared: a 1-slot build and a 16-slot build (`GLM53_SKIP_INACTIVE` on). Device time:
- 1-slot build: 4.20 ms (host wall 4.70);
- 16-slot build: 7.36 ms (host wall 7.82).

Evidence: `~/glm53-flash-bringup/artifacts/s10/tracy_s{1,16}/`.

| Source (reduced model) | 1 slot | 16 slots | Delta | Per layer | Full model (count) |
|---|---|---|---|---|---|
| Routed MoE: pairs path (sparse matmuls, tilize of 128 one-row pair tiles, fill-pad, k-sum, untilize) | 545 us | 2,560 us | +2,015 us | +1,008 us | 42 MoE layers: +42 ms |
| Element-wise / slice / reshape (mostly the pairs glue: weights, SwiGLU on padded pairs) | 695 us | 1,454 us | +759 us | about +330 us (MoE) | +14 ms |
| KDA recurrent update + conv over 16 slots | 199 us | 532 us | +333 us | +167 us | 34 KDA layers: +5.7 ms |
| DSA, head, sampler | - | - | < +30 us | - | < 1 ms |

The estimate (+61 ms) matches the served gap (110 - 50 ms). The pairs path makes one tile-padded row per (row, expert)
pair: 16 x 8 = 128 groups of 32 padded rows. `GLM53_SKIP_INACTIVE` skips the expert weight reads of inactive rows, but
the relayout and element-wise glue still runs on all 128 padded groups.

## MoE row buckets (`tt/dsa_buckets.py`, `GLM53_MOE_BUCKETS`)

The decode trace also takes one segment point per MoE layer, keyed by the number of active slots: buckets of 1, 2, 4
and 8 rows, then all S rows (`GLM53_MOE_BUCKETS=auto`, builds of 2 or more slots; `0` = off).

How a bucket below S works:
- It gathers the active rows: active slots first, and the padding rows repeat the first active slot.
- It runs the routed experts on those rows (the pairs path by default, `GLM53_MOE_BUCKET_PATH`).
- It scatters the result back to the slot rows; inactive rows get zeros.
- Every move is an exact row copy (`ttnn.embedding`). The routing weights and expert ids move as one bf16 table
  (ids < 256 are exact in bf16).
- Pairs padding rows take expert id 65535 and are skipped.
- The router and the shared expert stay on all S rows.
- The S bucket runs the build's own path.
- The host refreshes the three compaction inputs (gather index, scatter index, pad mask) only when the active set
  changes. There is no device sync.

Ticket t13 (reduced model, 16 slots, 16,384):
- The active rows are **bit-identical** to buckets off at 1 / 2 / 3 / 5 / 8 / 16 active (prefill + 6 traced steps).
- The tracker is clean, and the program cache is constant after capture.

| Token-out (reduced model, 2 MoE layers) | 1 active | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Buckets off | 7.01 ms | 7.46 | 7.90 | 8.62 | 10.22 |
| Pairs buckets (default) | **5.42 ms** | 5.63 | 6.37 | 7.79 | 10.30 |
| Sorted buckets (not exact) | 7.57 ms | 7.56 | 7.56 | 7.57 | 10.30 |

Ticket t15b (full model, 16 slots, max_seq_len 16,384, default precision):
- Active rows are bit-identical to buckets off at 1 / 5 / 16 active, with both a 64 MiB and a 96 MiB trace region.
- The program cache is constant.

| Token-out, 16-slot build | 1 active | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Buckets off | 106.3 ms | 115.8 | 125.0 | 140.1 | 173.8 |
| 64 MiB region: buckets 1, 16 | **72.9 ms** | 117.5 | 126.7 | 141.9 | 175.5 |
| 96 MiB region: buckets 1, 2, 4, 8, 16 | **72.9 ms** | **77.7** | **92.9** | **122.9** | 175.6 |
| 96 MiB, 8-row bucket on the sorted path (not exact) | 72.9 ms | 77.7 | 92.9 | 125.7 | 175.6 |

- The 1-slot build is 49.95 ms, so the 16-slot build at 1 active is 22.95 ms slower (72.9 ms). Of that gap:
  - the KDA update over 16 slots is about 5.7 ms (profile above);
  - segment replays are about 1.7 ms (this is also the cost at the S bucket, 16 active);
  - **15.5 ms is unattributed.** Likely sources, not measured separately: work that still runs on all 16 rows (router,
    shared expert, the DSA / KDA projections and the mHC mixing are tile-padded to 32 rows either way, but their
    element-wise and relayout glue grows with S); the MoE compaction ops (to_layout, embedding gather / scatter, 42
    layers); the 16-row sampler; and the inactive-skip ops.
- One row-bucket point trace is 254 KB of trace region, so each bucket below S costs about 10.7 MB for 42 MoE layers.
  `fit_row_buckets` keeps only the buckets the trace region holds: 64 MiB keeps the 1-row bucket, 96 MiB keeps 1 / 2 / 4 / 8.
- An explicit `GLM53_MOE_BUCKETS` list is taken as given.
- The multi-slot serving profiles take a larger trace region: p300x2 and p150x4 96 MiB, p300x2-wide 160 MiB (32 slots:
  six row buckets, 107.3 MiB captured): `trace_region_bytes` per
  profile in `serving_profiles.json`, read by `tt/serving_profiles.trace_region` and rendered into `tt-model.yaml`. The
  1-slot profiles keep 64 MiB.
- The standalone generator uses the region its mesh was opened with: 1-row bucket at 64 MiB.

## Prefill pool-key views (`tt/dsa.py`, `GLM53_DSA_VIEWS`)

The indexer cost follows the length of its pool-key view (ticket t9). The replicated pool-key cache is now one run of
`PM` blocks, where `PM` = 64 x a 5-smooth count >= the blocks needed. A sparse chunk scores the view
`[PM/d, 1, 32 d, 128]`, where `d` is the smallest of about 7 divisors of `PM` with `32 (d - 1)` >= the chunk's filled
pools. The causal start is `32 (d - 1)`. There is no `kv_len` and no slice.

The device page table is padded to the largest view with dummy columns. The host table and the decode-input table
keep `MB` columns. The block-sharded layout is unchanged; it also takes no DSA decode buckets
(`bucket_contexts(T, sharded)`).

Memory: the pool iotas are one tensor per mesh, not one per layer. Net change per die, computed on the host:

| Profile | Pool keys + iotas | Trace region | Net |
|---|---|---|---|
| p300x2 16 x 243,712 | -210 MiB | +32 MiB (96 MiB) | **-178 MiB** |
| p300x2-long 325,632 | -78 MiB | 0 (64 MiB) | **-78 MiB** |
| p300x2-wide 32 x 137,216 | -130 MiB | +96 MiB (160 MiB) | **-34 MiB** |
| p150x4 16 x 243,712 | -210 MiB | +32 MiB (96 MiB) | **-178 MiB** |

The block-sharded `p300x2-1m` profile is unchanged. Every multi-slot measurement in `serving_profiles.json` was taken
with a 64 MiB region. The G9 capacity re-probe of the combined stage-9 ticket replaces them.

### Capacity re-probe (review #1 P2-1, ticket t21)

The G9 probes (`doc/vllm/capacity/s{1,16,32}/`, 2,547 / 1,890 / 1,104 blocks) left 875 to 906 MB free after the
bind. `tools/memory_probe.py` now sizes the pool from the bytes per block measured by an earlier probe of the same
slot count (`--per-block-from`), to leave about 700 MiB free after the bind (`--free-after-bind-mib 700`). t21 ran on cf41be52370 + the P2-2 fallback (`tt/dsa_buckets.py`), workspace 1,024
MiB, the copy check on a non-aligned prompt near the limit (evidence: `doc/vllm/capacity/s10_s{1,16,32}/`).

| Profile | Trace region | Pool blocks | Free after the bind | Copy prompt | Copy accuracy | max_model_len before (G9) | after |
|---|---|---|---|---|---|---|---|
| p300x2 (16 slots), p150x4 (derived) | 96 MiB | 2,072 | 741 MB | 265,027 | 1.0 | 239,616 | **262,144** |
| p300x2-wide (32 slots) | 160 MiB | 1,282 | 752 MB | 163,907 | 1.0 | 137,216 | **159,744** |
| p300x2-long (1 slot) | 64 MiB | 2,704 | 761 MB | 345,923 | 1.0 | 325,632 | **344,064** |

`tools/serving_profiles.py ingest` / `refresh` / `render-yaml` wrote the new values to `serving_profiles.json` and
`tt-model.yaml`; `doc/context_contract.json` cites them (`replicated_default_layout.serving_profiles_stage10_t21`).
The served numbers in this README were measured at the earlier limits (239,616 / 137,216 / 325,632). p300x2-1m
(block-sharded, 1,048,576) is unchanged.

Ticket t14 (full model, 1 slot, 243,712):

| | views off | views on |
|---|---|---|
| TTFT 2,048 | 1,026 ms | 1,027 ms |
| TTFT 8,192 | 5,611 ms | **4,376 ms** (16K build: 4,410) |
| TTFT 32,768 | 24,171 ms | **18,486 ms** |
| Token-out, positions 300 / 5,000 / 70,000 / 200,000 | 50.10 / 53.54 / 56.01 / 58.20 ms | 50.10 / 53.52 / 55.99 / 58.20 ms |
| Teacher forcing, 3 references | 92 / 95 / 92 (prefill 92 / 95 / 91) | identical (tracker on: clean) |

- Prefill and decode logits are bit-identical views on vs off in the full-model check (4,221 tokens).
- On the reduced model, a 40,000-token prompt shows a 0.03 logit delta (top-k tie order across widths); top-1 21/21.

## Served before / after (vLLM, `vllm bench serve`, the stage-9 sweep workload)

"Before": the stage-9 A6 sweeps on code before stage 10 (DSA, MoE and view buckets absent; 64 MiB trace region),
`~/glm53-flash-bringup/artifacts/s10/sweep_<profile>.json`. "After": the same sweep, on the same profile, workload
and sampling, on the stage-10 HEAD (0a2663d0b6b + the slot-aware fit), in the combined stage-9 ticket
(`~/glm53-flash-bringup/artifacts/s10/sweep_final-<profile>.json`).

Settings that changed between the two columns:
- The trace region changes 64 -> 96 MiB for p300x2 and p150x4, and to 160 MiB for wide in this release. The MoE row
  buckets need it.
- The G9 re-probe set new max-model-len values: p300x2 239,616 (was 243,712) and wide 137,216 (was 151,552). The longest
  prompt row of each table uses the new limit.
- Every other server setting is the same.
- The long profile was not re-swept. Its 128/128 TPOT after is the stage-9 final run's 49.84 ms (async decode on).
- The decode gain at 1 to 2 users comes from the MoE row buckets (106 -> 73 ms standalone). The TTFT gain at long prompts
  comes from the prefill views. At 16 users (the S bucket) the time is unchanged.

### p300x2 (max_model_len 243,712)

| Users | Input / output tokens | TTFT before | TTFT after | TPOT before | TPOT after | t/s/u before | t/s/u after | Output t/s before | Output t/s after |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 / 128 | 245 ms | 252 ms | 109.8 ms | 69.9 ms | 9.10 | 14.30 | 9.0 | 14.0 |
| 1 | 1,024 / 256 | 766 ms | 770 ms | 111.5 ms | 71.8 ms | 9.00 | 13.90 | 8.8 | 13.4 |
| 1 | 4,096 / 256 | 2,986 ms | 2,402 ms | 113.3 ms | 73.5 ms | 8.80 | 13.60 | 8.0 | 12.1 |
| 1 | 16,384 / 256 | 13,934 ms | 9,956 ms | 113.7 ms | 73.9 ms | 8.80 | 13.50 | 6.0 | 8.9 |
| 1 | 65,536 / 256 | 57,759 ms | 42,888 ms | 114.5 ms | 75.8 ms | 8.70 | 13.20 | 2.9 | 4.1 |
| 1 | 131,072 / 256 | 116,170 ms | 98,371 ms | 114.1 ms | 77.9 ms | 8.80 | 12.80 | 1.8 | 2.2 |
| 1 | 243,456 / 256 (after: 239,360) | 216,555 ms | 194,768 ms | 114.0 ms | 79.2 ms | 8.80 | 12.60 | 1.0 | 1.2 |
| 2 | 128 / 128 | 365 ms | 491 ms | 122.4 ms | 74.8 ms | 8.20 | 13.40 | 16.1 | 25.6 |
| 4 | 128 / 128 | 964 ms | 984 ms | 125.8 ms | 90.0 ms | 7.90 | 11.10 | 30.3 | 41.3 |
| 8 | 128 / 128 | 1,916 ms | 1,928 ms | 143.6 ms | 119.9 ms | 7.00 | 8.30 | 50.7 | 59.6 |
| 16 | 128 / 128 | 3,822 ms | 3,859 ms | 178.7 ms | 172.6 ms | 5.60 | 5.80 | 77.2 | 79.4 |
| 2 | 1,024 / 256 | 1,146 ms | 1,534 ms | 124.7 ms | 76.5 ms | 8.00 | 13.10 | 15.5 | 24.3 |
| 4 | 1,024 / 256 | 3,046 ms | 3,054 ms | 128.4 ms | 91.8 ms | 7.80 | 10.90 | 28.7 | 38.7 |
| 8 | 1,024 / 256 | 6,089 ms | 6,091 ms | 144.8 ms | 121.7 ms | 6.90 | 8.20 | 47.6 | 55.2 |
| 16 | 1,024 / 256 | 12,170 ms | 12,167 ms | 180.5 ms | 174.4 ms | 5.50 | 5.70 | 70.4 | 72.3 |

### p300x2-wide (32 slots; before: max_model_len 151,552, pre-stage-10 A7 sweep; after: 137,216, 160 MiB trace region, all MoE buckets sorted, `doc/vllm/evidence/final/wide-r2_sweep.json`)

| Users | Input / output tokens | TTFT before | TTFT after | TPOT before | TPOT after | t/s/u before | t/s/u after | Output t/s before | Output t/s after |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 / 128 | 256 ms | 263 ms | 120.4 ms | 121.0 ms | 8.30 | 8.30 | 8.2 | 8.2 |
| 1 | 1,024 / 256 | 774 ms | 780 ms | 122.3 ms | 122.8 ms | 8.20 | 8.10 | 8.0 | 8.0 |
| 1 | 16,384 / 256 | 12,306 ms | - (not in the re-run sweep) | 124.6 ms | - (not in the re-run sweep) | 8.00 | - (not in the re-run sweep) | 5.8 | - (not in the re-run sweep) |
| 1 | 131,072 / 256 | 101,427 ms | - (not in the re-run sweep) | 125.4 ms | - (not in the re-run sweep) | 8.00 | - (not in the re-run sweep) | 1.9 | - (not in the re-run sweep) |
| 8 | 128 / 128 | 1,995 ms | 2,022 ms | 133.9 ms | 130.0 ms | 7.50 | 7.70 | 54.2 | 55.2 |
| 16 | 128 / 128 | 3,986 ms | 3,991 ms | 150.7 ms | 145.8 ms | 6.60 | 6.90 | 89.2 | 91.2 |
| 32 | 128 / 128 | 7,954 ms | 7,963 ms | 168.7 ms | 160.1 ms | 5.90 | 6.20 | 139.9 | 144.8 |
| 8 | 1,024 / 256 | 6,158 ms | 6,186 ms | 133.5 ms | 132.4 ms | 7.50 | 7.60 | 50.8 | 51.4 |
| 16 | 1,024 / 256 | 12,305 ms | 12,334 ms | 150.9 ms | 146.3 ms | 6.60 | 6.80 | 80.9 | 82.2 |
| 32 | 1,024 / 256 | 24,604 ms | 24,612 ms | 166.3 ms | 163.5 ms | 6.00 | 6.10 | 122.6 | 123.6 |

Wide final (36f3dd30b61, the batch-invariance fix): the served burst of 32 concurrent greedy requests matches solo
greedy on 32 of 32 streams (`wide-r2_smoke.log.txt`, SMOKE_BURST). Before the fix it was 4 of 32. Primary single user
(128 / 128, 1 request): TTFT 278 ms, TPOT 120.9 ms (8.27 t/s/u). Before stage 10, from the A7 sweep row: TTFT 256 ms,
TPOT 120.4 ms. CI burst (100 / 100 / 32): TTFT median 7,828 ms, TPOT 161.1 ms, 134.6 tok/s; before
(`bench_ci_p300x2-wide.log.txt`): 7,817 ms, 167.7 ms, 132.3 tok/s.
- At 1 user the sorted family runs at the speed of the code before stage 10. The pairs buckets were faster (76 ms) but
  not batch-invariant.
- From 8 users up, wide is faster than before: TPOT 133.9 -> 130.0 ms at 8 users, 150.7 -> 145.8 ms at 16, 168.7 ->
  160.1 ms at 32 (128 input). The sources of this gain are not separated.
- The standalone t18 build (32 slots, max_seq_len 2,048) gave 164.4 ms with buckets off at 1 active. The served wide
  build before stage 10 was already at 120.4 ms. The standalone "46 ms faster than off at 1 active" therefore does not
  carry over to the served wide profile: served single-user TPOT is unchanged (120.4 -> 121.0 ms).
- G9: 32 x 137,216 at 160 MiB, 107.3 MiB of trace region captured.

### p300x2-long (max_model_len 325,632)

| Users | Input / output tokens | TTFT before | TTFT after | TPOT before | TPOT after | t/s/u before | t/s/u after | Output t/s before | Output t/s after |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 / 128 | 236 ms | - | 56.5 ms | 49.84 ms (async on) | 17.70 | 20.06 | 17.3 | - |
| 1 | 4,096 / 256 | 3,201 ms | - | 60.2 ms | - | 16.60 | - | 13.8 | - |
| 1 | 65,536 / 256 | 64,592 ms | - | 60.0 ms | - | 16.70 | - | 3.2 | - |
| 1 | 262,144 / 256 | 260,888 ms | - | 60.1 ms | - | 16.60 | - | 0.9 | - |
| 1 | 325,376 / 256 | 324,233 ms | - | 60.0 ms | - | 16.70 | - | 0.8 | - |

Expected from the standalone measurements: single-user TPOT on p300x2 about 106 -> 73 ms (MoE row buckets, t15b);
long-prompt TTFT about 0.8x (views, t14: 8K 5.6 -> 4.4 s at 243,712). Long context and wide: DSA buckets plus views.

## Next step, measured but not landed (code freeze)

DSA decode buckets on the block-sharded layout (1m profile). They are off there today
(`bucket_contexts(T, sharded=True)`). At 1,048,576 the full-width selection runs over 262,176 pools in every DSA
layer. Extrapolating probe1 (0.537 ms per layer at 70,176 pools, about linear in the width) gives about 19-20 ms per
token at short positions. Needs its own measurement and exactness evidence on the sharded path.


KDA state compaction. In the 16-slot build at 1 active, the KDA recurrent update and the conv run over all 16 slots:
+167 us per KDA layer against the 1-slot build (t12: recurrent 172 vs 60 us, conv 93 vs 39 us per layer), 34 KDA
layers = about 5.7 ms per token. The fix: compact the active slots' KDA state in the same row buckets as the MoE
(gather the recurrent / conv state of the active slots, run the T_b-row update, scatter back), with its own
exactness evidence. Expected 16 slots / 1 active: about 73 -> 67 ms (1-slot build: 49.95 ms). Deferred to after
this release, so the v4b evidence and the combined serving ticket stay valid.

## Batch invariance: one routed-path family per build (wide regression, ticket t18)

At 32 slots the build's own MoE path for 32 rows is the expert-sorted path (QWEN35_DECODE_SORTED_MIN_ROWS 32). The row
buckets below 32 ran the pairs path. A request's routed math therefore depended on how many slots were active. The two
paths are each exact against itself, but they differ from each other far beyond rounding. Measured: max logit delta 1.63
(reduced model, t13, top-1 66 of 71) and 6.88 (full model, t15b 8-row bucket, top-1 32 of 34). Served wide burst: 4 of 32 greedy streams matched solo (32 of 32
before stage 10). After the fix (36f3dd30b61): 32 of 32 (`doc/vllm/evidence/final/wide-r2_smoke.log.txt`).

Rule (`moe_family`, `moe_bucket_paths`, `GLM53_MOE_BUCKET_PATH=family`, the default): every row bucket, the S bucket
included, runs the family of the S bucket's own path. That is sorted at 32 slots and pairs at 16 slots and below (no
change there). A host test pins the rule (`test_all_buckets_share_the_s_bucket_family`).

t18 measured a target request in slot 0 with 1 / 5 / 16 / 32 active slots, and at 32 active with two different filler
prompt sets. Bit-identical means the target row matched its A = 1 run exactly.

| 32-slot configuration | Target row bit-identical (reduced / full model) | Token-out, full model, 1 / 8 / 32 active | Teacher forcing (1 ref) |
|---|---|---|---|
| Buckets off | yes / yes | 164.4 / 164.3 / 145.9 ms | 93 / 100 / 100 |
| Before the fix (pairs below 32, sorted at 32) | **no at 32 active** (max delta 0.06 / 0.87) | 73.7 / 123.5 / 147.4 ms | - |
| **Family = sorted everywhere (shipped)** | yes / yes | 118.6 / 123.8 / 147.4 ms | **93 / 100 / 100** |
| Pairs everywhere (the 32 bucket on pairs too) | yes / yes | 73.7 / 123.5 / **284.3** ms | 92 / 100 / 100 |

Pairs everywhere is faster at 1 active but doubles the full-occupancy step (284 vs 147 ms). The wide profile exists for
high concurrency, so it ships the sorted family. It is invariant, matches buckets off at 32 active, and is 46 ms faster
than buckets off at 1 active.

## Async decode and TTFT (a changed setting)

Stage 9 turned async decode on (4678d8fccf4). It lowers TPOT slightly but adds TTFT. A new request's first token
waits for the decode step already in flight, and with two requests the first waits for the second's prefill:
- long profile, same code, async off / on (`doc/vllm/evidence/final/long-async{0,1}_vllm_benchmark.json`): TTFT 238.3
  -> 288.1 ms (+21 %), TPOT 50.89 -> 49.84 ms. CI burst TTFT median 82,318 -> 80,321 ms.
- p300x2, 2 users at the same time (sweep rows): TTFT 365 -> 491 ms (128 input) and 1,146 -> 1,534 ms (1,024 input),
  both +34 %. Part of that rise is async, part is the different code; they are not separated here.
- Stage 9 is running an async TTFT A/B with N >= 4 runs per arm; its result goes here when it lands.

## Batch invariance of the DSA width buckets (review #1 P2-2, tickets t20, t22, t23, t24)

The DSA decode bucket is keyed on the batch's largest position. A 4,000-token request decoded next to a 70,000-token
request runs its selection at the larger width. Below 16,384 pools the top-k runs its fused mode (at most 32 chunks of
512); above, it runs the classic mode. The two modes order tied scores differently.

t20 (reduced model, 2 slots, 131,072): the 4,000-token target, alone vs next to a 70,000-token filler (2 filler sets).

| DSA buckets | Target alone vs with the filler | Full model, 1 slot, 262,144: token-out at positions 300 / 5,000 / 20,000 |
|---|---|---|
| auto (1,056 ... 32,800) | **not identical**: max logit delta 1.73, tokens differ | 50.11 / 53.56 / 54.08 ms |
| classic only (16,416 / 32,800) | bit-identical, tokens equal | 51.45 / 54.76 / 54.76 ms |
| off | bit-identical | - |

The first fix (3993d5eecef) started the multi-slot buckets at 65,536 (width 16,416), so every bucket ran the classic
mode. t22 (reduced model, 3993d5eecef) passed: S = 2 and S = 16 (also 16 active), bit-identical, tokens equal. The
served check did not pass. t23 isolated the cause on the full model.

Served check (`tools/vllm_served_invariance.py`, profile p300x2, 16 slots, 239,616, greedy, temperature 0, no
logprobs): a 4,100-token request alone (solo 1), the same request while a 70,000-token request decodes 800 tokens
(concurrent), then alone again (solo 2). Two prompt kinds: natural text (Tale of Two Cities) and random token ids.
Evidence: `evidence/served_invariance/`.

| Ticket, code | Server settings | Text: solo 2 / concurrent vs solo 1 | Random ids: solo 2 / concurrent vs solo 1 |
|---|---|---|---|
| t22, 3993d5eecef | default (async on) | - | - / **differs at token 2** |
| t23 (i), 3993d5eecef | DSA buckets, MoE row buckets and prefill views off | equal / equal | equal / equal |
| t23 (iv), 3993d5eecef | DSA buckets off (MoE row buckets, views on) | equal / equal | equal / equal |
| t23 (ii), 3993d5eecef | default, async off | equal / **differs at token 17** | equal / **differs at token 2** |
| t23 (iii), 3993d5eecef | default | equal / **differs at token 17** | equal / **differs at token 2** |
| t24, fallback code | default | equal / equal | equal / equal |

Findings:
- The served path is deterministic: solo 2 equals solo 1 in every run.
- The DSA decode width buckets cause the served batch dependence. The MoE row buckets, the prefill views and async
  decode do not: (iv) is identical and (ii) still differs.
- Classic-only widths still select differently on the full model (42 layers, 11 DSA). The 3-layer reduced probe did
  not show it. The cause inside the classic top-k is not isolated.

Rule (shipped): with S >= 2 slots there are no auto DSA decode buckets. Every step selects over the full `P_alloc`.
- One slot (profiles long and p300x2-1m) keeps the buckets from 4,096: one request at a time, so there is no
  cross-request effect. p300x2-1m takes no DSA buckets (block-sharded layout).
- An explicit `GLM53_DSA_BUCKETS` list is taken as given (a diagnostic setting, not batch-invariant at S >= 2).
- The MoE row buckets and the prefill views stay on.
- Host test: `test_multi_slot_takes_no_auto_dsa_buckets`.
- Cost: the multi-slot profiles lose the DSA part of the decode gain. Estimated (not measured): about +4 to +5 ms
  per token at p300x2 short positions (the 1-slot full-width vs bucketed difference at 262,144 to 280,576: 55.07 vs
  50.10 ms). Measured in the final scorecard run.

The async validator with a bucket switch (`GLM53_DSA_BUCKETS=4096` at 8,192, a 4,100-token request crossing position
4,227) passes (t20, ASYNC_RESULT PASS).

## Trace region (part of the optimization)

The multi-slot profiles take a 96 MiB trace region (64 MiB before). The 2 / 4 / 8-row MoE buckets need it: one row-bucket
point trace per MoE layer is 254 KB. The code before stage 10 has no use for the larger region. Evidence (t15b, full
model, 16 slots): at 2 / 4 / 8 active, 117.5 / 126.7 / 141.9 ms with 64 MiB (1-row bucket only) vs 77.7 / 92.9 /
122.9 ms with 96 MiB. The 1-slot profiles keep 64 MiB (their MoE buckets are off).

## Trace, async and input decisions

- Decode is one segmented trace per build: main segments plus point traces (DSA selection by width, MoE routed experts
  by active rows). The replay is all `ttnn.execute_trace(..., blocking=False)` (`dsa_buckets.replay_decode`). No host
  read or sync is added. The plugin's async decode (stage 9, `supports_async_decode`, stale-input device test
  `tools/vllm_async_validate.py`, 4678d8fccf4) is unchanged.
- Persistent inputs change only with the scheduler state:
  - tokens / positions (stage 9);
  - the page table (copied on change);
  - the MoE compaction inputs (gather index, scatter index, pad mask), copied host to device only when the active
    set changes. Several active sets in sequence in one process are bit-exact against buckets off (t13, t15b), and so is
    the adapter remap check (t16).
- The bucket key is computed on the host from the positions mirror, without a device read.
- Async decode and stale inputs: the stage-9 evidence (`tools/vllm_async_validate.py`,
  `tests/test_vllm_async_validate.py`, 4678d8fccf4) covers changed tokens / positions and changed / unchanged page
  tables. Ticket t17 runs the same test on 0a2663d0b6b (4 slots), so the MoE row buckets and the views are active under
  async decode and the compaction inputs change with the active set: **PASS** (`~/glm53-flash-bringup/logs/s10/t17.log`,
  ASYNC_RESULT PASS).
- Tilize / untilize in the trace: the MoE compaction converts the S-row activation to row-major for the row gather
  (`ttnn.embedding`) and back. It is necessary for an exact row copy. Its cost is in the measured win (106.3 -> 72.9 ms
  at 1 active). Host tensors (`from_torch`) are built only when the active set changes.

## Rejected options

| Option | Why rejected |
|---|---|
| One full decode trace per DSA width bucket | each trace is about 40 MB in a 64 MiB region |
| `kv_len` alone on the prefill indexer | its cost follows the view length, not `kv_len` (t9: 5.34 vs 5.60 ms per 256 rows) |
| A deterministic top-k tie-break | changes the current code's numerics; the gate was moved to exactness against the `T_b` build |
| The sorted MoE path in the row buckets | slower than pairs at 1-4 rows, only 8 rows faster on the reduced model, slower on the full model (125.7 vs 122.9 ms), not exact |
| "Inactive rows copy an active row's routing" | removes no relayout glue |
| DSA buckets on the block-sharded layout | unmeasured; gated off (`bucket_contexts(T, sharded)`) |

## Limitations

- A trace-capture out-of-memory is fatal: there is no retry with fewer row buckets. `fit_row_buckets` uses a slot-aware
  estimate of the segmented decode (`decode_trace_estimate`, calibrated on the 16-slot t15b and G9 captures and on the
  32-slot G9 out-of-memory). It selects lists that fit for every measured configuration. The 0a2663d0b6b fixed estimate
  over-selected at 32 slots (wide at 96 MiB: out of memory). The retry is deferred: the mesh state after a failed
  `end_trace_capture` is undocumented, so it needs its own device test.
- The wide profile ships with a 160 MiB trace region in this release (MoE buckets 1/2/4/8/16/32). Both the old and the
  slot-aware rule choose that list at 160 MiB (estimate 108.9 MiB).
- Segment replays cost about 0.4 ms (1 slot, full DSA width) and about 1.7 ms (16 slots, the S bucket) per token where
  no bucket narrows the work.
- A 64 MiB trace region holds only the 1-row MoE bucket: the multi-slot profiles take 96 MiB.
- KDA over all slots is not compacted (next step above).
- The top-k tie order differs between the fused and classic top-k modes, so buckets change low-order logits in some
  tie cases against the same build without buckets (exact against the `T_b` build). Teacher forcing is unchanged.

## $optimize checklist (serving path)

| Item | Evidence |
|---|---|
| The hot op is identified by measurement | probe1 (DSA selection by width), t9 (indexer by view), t12 (16-slot ops) |
| Exactness for every change | t3 / t10 / t11 (DSA buckets against the `T_b` build), t13 / t15b (MoE rows), t7 (alias), t14 (views) |
| Trace safety | tracker clean (t2, t3, t5, t13, t14), program cache constant after capture, t16 PROGRAM_GUARD |
| Accuracy | teacher forcing on 3 references unchanged (t5, t14) |
| No host fallback on the decode path | replay is non-blocking traces only; device sampling unchanged |
| Memory | net per-die change per profile, computed (table above), G9 re-probe in the combined ticket |
