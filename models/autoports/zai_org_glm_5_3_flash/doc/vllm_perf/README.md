# GLM-5.3-Flash (E192) stage 10: DSA cost by filled positions

Stage 10 (optimized vLLM). This section covers the DSA indexer: decode and prefill cost now follow the batch's
filled positions, not `max_seq_len`. It also covers one memory fix of the replicated pool-key cache.

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

The sharded DSA layout (`GLM53_DSA_SHARD`) uses the same `GlmDSA` selection code, so the buckets apply there too.

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

## Files

- `tt/dsa_buckets.py` (new): buckets, warm-up, segmented capture and replay.
- `tt/dsa.py`: the decode selection hook, the prefill scored width, the pool-key alias.
- `tt/generator.py`: `install` before the capture, `capture_decode`, `replay_decode`, `release_decode`.
- `tools/s7_probe.py`, `tools/model_step_probe.py`: replay through `replay_decode`.
- `tools/dsa_bucket_probe.py`, `tools/dsa_bucket_probe2.py`, `tools/dsa_bucket_check.py` (new): the evidence tools.
- Ticket scripts and logs: `~/glm53-flash-bringup/scripts/s10_*.sh`, `~/glm53-flash-bringup/logs/s10/`.
  Records: `~/glm53-flash-bringup/artifacts/s10/`. Summary: `doc/vllm_perf/dsa_buckets_summary.json`.
