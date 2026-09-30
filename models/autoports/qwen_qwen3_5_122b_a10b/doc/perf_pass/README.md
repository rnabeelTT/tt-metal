# Perf pass (stage 13) — TTFT and tok/s/u across batch widths, Qwen/Qwen3.5-122B-A10B on the (1,4) Blackhole mesh

Post-release pass requested after stage 12: capture the low-hanging performance fruit turn by turn, TTFT and per-user
decode at every width including one user; first target the width-1 non-determinism that keeps a single user on the
width-2 decode trace. Baseline = stage-12 commit 6da58d94a83. Evidence: this folder + `work_log.md` (run list with
timestamps), workspace `~/qwen35-122b-bringup/{scripts/13-*,logs/13,artifacts/stage13}`.

## Turn 1 — width-1 non-determinism: root cause found and fixed (single user on its own trace)

| experiment | result |
|---|---|
| per-op `ttnn.synchronize_device` after every op / after collectives only / after all non-collectives (48 layers, eager, min width 1) | 4-5 unique streams of 6 in every variant, baseline 4 of 6: **not an inter-op race** (`artifacts/stage13/w1sync-b.json`) |
| op-level bisection: hash of every op output during the same decode step across 4 repetitions, state tensors excluded | first differing op = the gather-mode MoE `ttnn.sparse_matmul`, with bit-identical logical inputs; only the first half of its 512 output columns differs (`w1opbisect_findings.json`) |
| standalone repro of that op, 40 identical repetitions per variant (both factories, 1 / 32 rows, bfp4 / bfp8 / bf16, L1 acc on/off, fp32 acc, HiFi2, NaN / huge / zero padded rows) | 1 distinct output everywhere: the kernel is deterministic in isolation (`sm_determinism_repro.log`) |
| pairs path (grouped factory) at width 1 on 48 layers | still 5 unique of 6; widths 2 / 4 / 8 with the same one-row-per-group geometry: 1 unique of 3 (`w1pairs-8.json`) |
| in-situ replay: every op of a width-1 step re-executed 3x on its live inputs (`scripts/13-w1replay.py`) | the router gate op `generalized_moe_gate` differs on replay in EVERY layer (136-328 of 2048 output elements, other expert ids); `fused_decoder._route` only ever used it for the single-row decode (>= 2 rows: functional router), and stage 3 had measured it bit-non-deterministic with several rows |
| **fused gate op off** (`QWEN35_ROUTER_GATE_OP=0`, functional router everywhere), 48 layers | width 1 traced **1 unique of 6**, eager 1 of 4, async-overlap 1 of 3; widths 2 / 4 / 8 1 of 3 (`w1pairs-gate0.json`) |
| gate op on with per-layer (unshared) scratch buffers as in stage 3 | the 48-layer build fails with the stage-6 L1 clash (that is why the buffers were shared): the op has no deterministic configuration at 48 layers |
| two-op router (`ROUTER_TOPK_FIRST`: top-k on the logits, softmax over the k selected; the global partition function cancels in the renormalization) | same determinism at every width; 0.7-0.9 ms per step cheaper than softmax -> top-k -> sum -> div (`w1pairs-topk1.json`) |

Why the earlier evidence pointed at the matmul: the gate op's outputs (ids / scores) hashed identical across runs at the
layer where the matmul output first differed, so its effect there is not through its logical outputs; the op is a
one-token-per-core kernel over 110 cores with L1-sharded scratch shared by all 48 layers, and what it leaves in L1 or
in flight is not visible to an output hash. The mechanism inside the kernel is not established; the fix removes the op
from the served path (it was already unused for >= 2 rows).

Shipped defaults (2026-09-29): `FusionFlags.router_gate_op` off, `moe.ROUTER_TOPK_FIRST` on, `QWEN35_DECODE_MIN_WIDTH`
1 (a single live user runs the width-1 trace). Correctness with the new router: layer suite 28/28 (prefill vs HF up to
2048 and chunked 4429 both kinds, batched decode B = 32, distinct positions, real weights, stress bit-identity), full-model
prefill-check book_p2048 top1 0.98 / top5 1.00 / top100 1.00 (0.99 / 1.00 / 1.00 before: one near tie).

48-layer traced step (in-process, deferred readback): w1 32.69 ms (stage 10: 37.7 on the width-2 trace; the racy gate
op: 27.45), w2 36.46, w4 45.73, w8 67.11, w16 85.12, w32 90.94.

## Turn 2 — "pairs" decode MoE path for widths 1-8 (adopted)

`Qwen35MoEOptimized._routed_pairs` (`tt/optimized_decoder.py`, `OptimizationConfig.pairs_max_tokens`, env
`QWEN35_MOE_PAIRS_MAX_TOKENS`, default 8): the T x k (token, expert) pairs of a decode step become G = T*k groups of one
tile-padded activation row each (the token row repeated k times, `ttnn.repeat`), the flattened top-k ids the `indices`
operand, and the stage-12 group-parallel sparse matmul factory runs gate/up and down in ONE call each; `ttnn.swiglu`,
a `[1,G,1,1]` broadcast multiply with the routing weights and a batch-dim `ttnn.sum` over the k experts of each token
close the block. It replaces the per-token gather loop (widths 2-4) and the serial masked path (width 8). Widths > 8
keep the masked path (weight-bandwidth bound, see below). Two op quirks met on the way: the data-moving `ttnn.reshape`
rejects uint16 (ids go through uint32), and `ttnn.swiglu` reports the tile-padded 32 rows as its logical shape (kept
through the chain, one row sliced after the k-sum).

Layer level (traced decode step per layer, `test_perf_traced_decode_batched`, `logs/13/pairs-{0,8}-perf.log`):

| width | GDN layer before -> after | attention layer before -> after |
|---|---|---|
| 1 | 0.524 -> 0.588 ms | 0.431 -> 0.495 ms |
| 2 | 0.803 -> 0.793 | 0.710 -> 0.699 |
| 4 | 1.078 -> 0.985 | 0.968 -> 0.877 |
| 8 | 1.987 -> 1.430 | 1.906 -> 1.282 |
| 16 / 32 (masked, unchanged) | 2.350 / 2.786 | 2.272 / 2.657 |

48 layers, in-process traced step (deferred readback, `w1pairs-8.json`): w1 27.45 ms, w2 37.36, w4 46.56, w8 67.80,
w16 85.87, w32 91.56 (stage 10: w1 25.1, w2 37.7, w4 50.8, w8 79.8, w32 124).

Correctness (`logs/13/pairs-8-correctness-full.log`, 18/18): decode vs HF at positions 37 / 333 both layer kinds, real
weights, 4 users at distinct positions (per-user PCC >= 0.9990), non-aligned continuation, traced decode, prefill after
traced decode, repeated-run bit-identity; 48-layer streams bit-identical at widths 2 / 4 / 8 (3 runs each).

Served (`scripts/13-serve-chain-w1.sh`, held 48-layer server from the checkout, 32 slots, 262144 context, min width 1,
functional two-op router, pairs path; readiness benchmark runner 128 / 128 tokens (TTFT p50, TPOT mean) and the stage-10
per-width client (TPOT from stream timing); `served_w1/`; the intermediate run with the pairs path alone on the width-2
minimum is in `served_pairs/`):

| users | TTFT p50 stage 10 -> final (turn 4) | TPOT stage 10 -> pairs only -> final | t/s/u final | aggregate tok/s stage 10 -> final (client) |
|---|---|---|---|---|
| 1 (own width-1 trace) | 394-426 -> **371 ms** | 37.7 -> 37.3 -> **32.7 ms** | **30.6** | 24.6 -> 28.5 |
| 2 | 0.91 -> 0.85 s | 39.4 -> 39.1 -> 38.1 ms | 26.2 | 43.1 -> 45.1 |
| 4 | 1.79 -> 1.65 s | 51.8 -> 47.7 -> 46.7 ms | 21.4 | 60.8 -> 68.0 |
| 8 | 3.23 -> 3.00 s | 93.1 -> 68.5 -> 67.7 ms | 14.8 | 66.8 -> 88.4 |
| 16 | 6.27 -> 5.69 s | 107.6 -> 107.5 -> 107.0 ms | 9.3 | 100.6 -> 106.2 |
| 32 | 12.12 -> 11.05 s | 125.1 -> 124.9 -> 124.7 ms | 8.0 | 140.6 -> 152.0 |

Reproducibility on the served path with the width-1 trace (`served_w1/seedprobe.json`): seeded completions 1 unique of
6 / 12 / 24 / 24 / 48 at widths 1 / 2 / 4 / 8 / 16, greedy width-1 re-runs 1 unique of 6 (stage 10 at width 1: 6 of 6).

## Turn 3 — served-step host overhead and admission cost

Host profile of the vLLM EngineCore during served benchmarks (`hostprof/pyspy_c{8,32}.raw`, `scripts/13-pyspy-summary.py`):
at 32 users 31.8 % of the engine's host time re-applied unchanged sampling parameters every step (`apply_decode_state`:
penalty copies 16.6 %, k / p / temperature resets 15.8 %), and under continuous admissions 23 % was the per-admission
GDN slot write (36 layers x rebuild of the whole [B, ...] state buffer around one slot).

| change | result |
|---|---|
| sampling-params memo (`generator_vllm._memoize_sampling_params`, `QWEN35_SAMPLING_PARAMS_MEMO`, on): skip the sampler reset when the formatted params equal the last applied ones; prefill-side or direct sampler writes invalidate it | plugin sampling suite 73 passed / 4 skipped (= stage 9), qualitative coherent; served TPOT c1 / c8 / c32 32.7 / 67.7 / 130.5 ms with and without -> **no latency effect** (the host work was overlapped with the device step); kept as CPU hygiene |
| GDN slot write by in-place `ttnn.experimental.slice_write` of the recurrent state (`TPGatedDeltaNet._write_index`, `QWEN36_GDN_SLICE_WRITE_SLOTS`; conv taps keep the rebuild) | correctness 9 / 9 (batched decode B = 32, distinct positions, reset after history, prefill after traced decode); per-layer write_slot 0.80 -> 0.77 ms per admitted user (`test_perf_slot_write`): the write is op-launch bound, not traffic bound -> **neutral, off by default** |
| step-synchronization probe (`13-stepsync-probe.py`): traced step with deferred vs blocking vs plugin-style overlapped readback | < 0.3 ms difference at widths 1 / 8 / 32 |
| API-server vs engine CPU during the c32 benchmark (`hostprof/cpu_c32.txt`, `/metrics`) | api_server 4 %, engine 411 % (dispatch threads): the API server is not the bound |
| **resolution of the served-vs-trace gap** (107 / 130 served vs 85 / 91 ms in-process at 16 / 32 users) | the in-process probes gave every user the SAME prompt -> same experts -> a fraction of the expert-weight bytes; served users differ and touch ~all 256 experts. The layer perf test with distinct users predicts 2.786 x 48 = 134 ms. Widths 16-32 are at the weight-bandwidth floor; there is no host gap |

## Turn 4 — short-prompt TTFT: expert-sorted prefill from the 128-token bucket (adopted)

Stage 7 pinned the expert-sorted prefill path at bucket 256 because the masked chunks won at 128 tokens with the
multicast sparse-matmul factory (344 vs 456 ms). With the stage-12 group-parallel factory the sorted 128-token bucket
is the faster one. 48-layer traced prefill bench (`prefill_bucket_bench_48layers_min{256,128,64}.json`, best of 3):

| prompt tokens | threshold 256 (stage 7-12) | threshold 128 (adopted) | control (64 = 128, repeat) |
|---|---|---|---|
| 128 | 344.9 ms | **303.2 ms** (-12 %) | 302.2 ms |
| 161 (bucket 256) | 347.1 | 348.1 | 344.4 |
| 256 | 343.1 | 344.8 | 344.7 |
| 333 (bucket 512) | 436.3 | 436.8 | 440.5 |
| 512 | 437.5 | 436.9 | 440.2 |

Correctness at threshold 128: layer suite 13/13 (prefill vs HF at 129 and 333 tokens both kinds, non-aligned
continuation, dispatch vs sort, stress bit-identity); full-model prefill-check book_p128 top1 0.95 / top5 1.00 / top100 1.00 (stage-7 masked path 0.97 / 0.99 / 1.00; bars top5 >= 0.98, top100 = 1.0); served TTFT p50 (readiness
benchmark, 128-token prompts) 1 user 408 -> 371 ms, 8 users 3234 -> 3004 ms, 32 users 12120 -> 11052 ms (every serialized 128-token
prefill is ~35 ms shorter; TPOT unchanged; seeded / greedy completions reproduce at every width; `served_min128/`). `MULTICHIP_SELECTED.sorted_min_tokens` is 128 (env `QWEN35_SORTED_MIN_TOKENS` still overrides).

## Turn 5 — long-prompt prefill: the MoE dispatch as one kernel, no zero-fills, no padding groups (adopted)

The question "why is prefill still serialized?": within a request every op already spans all 440 cores; the serial
parts are the 2048-token segment loop and the scheduler's one-prefill-at-a-time. Per 2048-token segment a layer spent
~20 ms of which ~16.5 ms was the MoE block, and of that ~6 ms was dispatch bookkeeping in ~100 small ops on a
[T*k, E] fp32 one-hot (row-proportional, so larger segments did not help: 4096-token segments gave -13 % per token,
failed at whole-layer 4096 rows, and segfaulted the 48-layer bench; a device sort of the 16K keys alone cost 17 ms).
Three changes, all in the expert-sorted prefill path (T >= 128), all bit-identical or PCC-identical:

1. `ttnn.experimental.moe_sorted_dispatch` (new op, `ttnn/cpp/ttnn/operations/experimental/moe_sorted_dispatch/`): one
   single-core dataflow kernel reads the tiled `[1,1,T,k]` top-k ids / scores in place and does the counting sort in L1
   (histogram, padded per-expert offsets, group ids, source-token / weight / destination-row scatter) -> four sticks.
   Same row layout as the counting sort bit for bit (`test_dispatch_modes_equivalent`, T = 256 .. 2048 and identical
   routing); 1.26 ms device at 2048 tokens (+0.4 ms glue) instead of ~6 ms. `QWEN35_DISPATCH_MODE=kernel` (default).
2. The grouped sparse matmul no longer zero-fills its output (`SparseMatmulDeviceOperation::create_output_tensors`
   recognised only the `nnz` compact shape as fully written; the group-parallel factory writes every group): -0.88 ms
   per layer of `FILL` unary ops.
3. Padding groups skipped: the dispatch marks the static trailing padding groups (~16 % of the groups at 2048 tokens)
   with a sentinel expert id and the grouped factory (`MatmulMultiCoreReuseProgramConfig.skip_padding_groups`, the
   in0 reader reads the id list, the in1 writer signals validity to compute through the BRISC mailbox) skips their
   weight / activation reads and output writes: expert matmuls 4.34 -> 3.67 ms per layer; real groups bit-identical
   (`test_grouped_skip_padding_groups`, trailing and interleaved sentinels). `QWEN35_SKIP_PAD_GROUPS=1` (default).

MoE block device time per layer at 2048 tokens: 16.45 -> 11.24 -> 10.57 ms; whole layers 20.7 -> 15.5 ms (GDN) and
18.3 -> 13.1 ms (attention). 48-layer traced prefill (`prefill_bucket_bench_48layers_{min128,kernel,kernelskip}.json`):

| prompt tokens | turn 4 | kernel dispatch + no zero-fill | + padding skip (shipped) |
|---|---|---|---|
| 128 | 303.2 ms | 249.0 ms | **214.1 ms** (-29 %) |
| 2048 | 984.5 | 732.6 | **701.3** (-29 %) |
| 8192 | 3959.0 | 2950.9 | **2828.3** (-29 %) |
| 16384 | 7993.2 | 5980.6 | **5744.9** (-28 %) |

Correctness: layer suite 14/14 (kernel) and 12/12 (kernel + skip) with PCCs identical to the counting-sort path;
full-model prefill-check vs HF: book_p2048 top1 0.99 / top5 1.00 / top100 1.00 and book_p8192 0.97 / 1.00 / 1.00 (stage 12: 0.99 / 1.00 / 1.00 and 0.95 / 1.00 / 1.00; bars top5 >= 0.98, top100 = 1.0). Served (`served_kernelskip/`, same server settings as turn 4): 1 user TTFT p50 371 -> 258 ms, TPOT 32.6 -> 32.0 ms (31.3 t/s/u); 32 users TTFT p50 11.1 -> 8.2 s, TPOT 130 ms (the weight-bandwidth floor, unchanged); direct long prompts 8192 / 32768 / 131072 tokens TTFT 4.0 / 16.4 / 72.3 -> 2.89 / 11.8 / 54.1 s (2840 / 2780 / 2420 prompt tok/s), greedy completions byte-identical to the turn-4 server. Bundle rebuilt from c9b9139df6e (image e63ad8e67d71), pushed private (Hub revision 2ce007aa) and verified from the Hub: burst 155 tok/s (137), 8K prompt 3.3 s (4.5), 128K prompt 54.5 s (72.6).

Rejected in this turn: 64-row expert groups (fewer weight re-reads but +50 % padding rows: MoE 11.2 -> 11.8 ms);
4096-token segments (above); the `sort_u16` dispatch. Still open (measured, not done): the un-dispatch as one
multi-core gather-add kernel (`ttnn.experimental.moe_gather_add_rows`, written, off by default until its compute-kernel
init issue is closed; would take the 2.1 ms un-dispatch + 0.7 ms untilize to ~1 ms) and reading each expert's weights
once per layer instead of once per 32-row group (~1.0 GB vs 0.34 GB of expert reads per chip per layer; ~-1.9 ms).
The Qwen3.8 bundle study (`qwen38_bundle_study.md`) and the MTP plan (`mtp_plan.md`) are in this directory.

## Rejected in this pass

- Expert-sorted MoE at decode (`sorted_min_tokens` 256 -> 8 with the group-parallel factory): per layer w8 4.16 /
  w16 4.37 / w32 4.77 ms vs 1.99 / 2.35 / 2.78 masked: the dispatch bookkeeping (~170 ops) is op-launch bound at
  T <= 32. Correct (12/12) but slower; the knob `QWEN35_SORTED_MIN_TOKENS` stays for prefill experiments.

## Where the remaining time is

- Widths >= 16 are weight-bandwidth bound: a step touches ~all 256 experts -> ~55 GB of bfp4 expert weights per step
  over 48 layers (~108 ms at ~512 GB/s per die); served 107 / 130 ms at 16 / 32 users is that floor plus ~15 %.
  (The lower in-process numbers came from identical prompts for all users: fewer experts, fewer bytes.)
- Width 1 (32.7 ms): ~96 device ops per layer (4600 per step + 2400 deallocations, eager count): op-launch bound; the
  functional router costs ~5 ms per step more than the (racy) fused gate op (linear + topk + softmax on a 1-row tile, DRAM
  intermediates: an L1-resident router chain is the next small step); the largest
  op groups are the GDN decode mixer (~50 ops: conv shift register, head layouts, recurrence) and the attention block.
  `ttnn.experimental.kda` ships fused kernels (`qkv_causal_conv1d_silu`, `sigmoid_gated_rms_norm`, recurrent scans);
  evaluated on paper for the GDN decode shift register (3 copies + multiply + 3 mac + silu = 8 ops per layer): the
  kernel wants ROW_MAJOR DRAM [B, seq, channels] inputs and a [B, K-1, channels] state and returns split q / k / v, so
  the TILE / L1 decode path would spend ~6 layout and state ops around it -> no op-count win without a bespoke
  TILE-layout variant. A fused GDN decode step (conv + gate + recurrence + norm) is the real single-user lever and is
  kernel work beyond a low-hanging pass.

## Shipped

Commits 62010702d98 (pairs path), 4e7f349857d (width-1 fix, two-op router, min width 1), 29c5dfa2171 / 098fafa7216
(docs, host probes, neutral knobs) and 3ec334e38a3 (sorted prefill from 128 tokens) on the local branch
`qwen3_5_122b_a10b`. Bundle rebuilt from 3ec334e38a3 (image `tt-model/qwen3-5-122b-a10b-blackholex4:c6e9186de5d4`),
pushed private (`raahemnabeel/qwen3-5-122b-a10b-blackholex4`, Hub revision 024095e6) and verified from the Hub on reset
devices (`logs/13/verify-hub-t4.log`): healthy in 10.5 min, /v1/models 262144, chat, tool call, `tt-model curl` default /
thinking on (1134 reasoning chars) / thinking off, non-aligned 2381-token prompt, 32-request burst 137 tok/s (stage 12
image: 127), 8192-token prompt 4.5 s and 131072-token prompt 72.6 s wall for 16 new tokens. Greedy continuations at near
ties can differ from earlier images (router renormalization order, sorted-path accumulation order): the default greeting
changed twice across the stage-13 images, each time to a valid sentence.

Net served effect of the pass vs the stage-11 release (128-token prompts, 128 output tokens, same server settings):
1 user TTFT 394-426 -> 371 ms and TPOT 37.7 -> 32.6 ms (26.5 -> 30.6 t/s/u); 4 users TPOT 51.8 -> 46.7 ms; 8 users
93.1 -> 67.7 ms (aggregate 66.8 -> 88.4 tok/s); 16 / 32 users unchanged at the weight-bandwidth floor with TTFT
6.3 -> 5.7 s and 12.1 -> 11.1 s; seeded and greedy completions reproduce at every width.
