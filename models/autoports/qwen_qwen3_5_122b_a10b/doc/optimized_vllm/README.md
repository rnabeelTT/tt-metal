# Qwen3.5-122B-A10B — optimized vLLM serving (stage 10)

Served path: the stage-9 adapter (`tt/generator_vllm.py`, `TTQwen3_5MoeForConditionalGeneration`) on the (1,4) Blackhole
mesh at `--max-num-seqs 32 --max-model-len 262144 --block-size 64`, `sample_on_device_mode: all`, 48 layers, the stage-8
precision policy, vLLM 0.26.0 + TT plugin 5961e48 (+ the registration hunk), asynchronous scheduling on (vLLM default).
Before = stage 9 (commit 0e57369c7a6, `doc/vllm_integration/`); after = this stage. Same runner, stages, sampling profile,
lengths, mesh, sampling mode; the only TT-config change is the trace region (512 -> 768 MiB, needed by the extra decode traces).

**Primary single-user serving numbers** (`readiness_check.run_vllm_server`, `vllm bench serve` 128-token prompt / 128 output
tokens / 1 request / concurrency 1 / greedy, server at `--max-num-seqs 32`):

| metric | before (stage 9, `doc/vllm_integration/readiness_vllm_48l-final/`) | after (`readiness_vllm_48l-final/`) |
|---|---|---|
| TTFT p50 / p99 | 474.1 / 474.1 ms | **426.4 / 426.4 ms** |
| TPOT mean / p99 | 124.19 / 124.19 ms | **37.70 / 37.70 ms** |
| ITL p50 / p99 | 124.19 / 126.60 ms | 37.75 / 41.17 ms |
| TPOT-derived decode | 8.05 t/s/u | **26.52 t/s/u** (3.3x) |
| output throughput | 7.88 tok/s | 24.54 tok/s |

**Secondary CI serving burst** (100 / 100 / 32 requests, no concurrency cap, greedy; vLLM-nightly parity shape, not a decode headline):

| metric | before | after |
|---|---|---|
| TTFT p50 / p99 | 13.57 / 13.57 s | 11.50 / 11.79 s |
| TPOT mean / p99 | 130.85 / 132.88 ms | 133.83 / 143.85 ms |
| output throughput (32 concurrent) | 120.7 tok/s | 129.4 tok/s |

Serving status after (`readiness_vllm_48l-final/`, `~/qwen35-122b-bringup/logs/10/vllm-48l-final.log`): `run_vllm_server --stages
serve,sampling,qualitative,benchmark --sampling-profile full` PASS — sampling suite 73 passed / 4 skipped (11:05; the same 4 skips as
stage 9: 3 chunked-prefill tests, logprobs-all-vocab), qualitative 6 / 6 coherent (section 6), both benchmarks complete with no
missing tokens; `DECODE_BUCKETS` over the run {2: 3582, 4: 24, 8: 77, 16: 21, 32: 1183} steps, 46 switches, 649 full-width
steps forced by seeded rows; `DECODE_CONTRACT v1 steps 4875, host-authoritative reloads 498`; program cache 1591, `new: 0` after
every admission; trace region 578 of 768 MiB, DRAM 29.99 of 31.08 GiB. The decode headline (37.7 ms) is the width-2 trace:
the single-user step of the served production configuration, 3.3x faster than stage 9 and 1.58x the model's B=1 token-out
step (23.9 ms; the width-1 trace is 25.1 ms but is not served, section 3). The CI burst's aggregate rose 120.7 -> 129.4 tok/s
(TTFT p50 13.6 -> 11.5 s: the burst's admissions decode on the narrower traces while the batch fills); its TPOT rose slightly: mean 130.85 -> 133.83 ms
(+2.3%), p99 132.88 -> 143.85 ms (+8.3%). The width-32 trace itself is unchanged (held server, concurrency 32: 131.1 / 132.9 ms,
section 4); the extra is admission-time work during the burst's fill — every admission that condenses the batch runs one
full-width `slot_remap` step, and every width switch costs a forced host-authoritative reload plus the token / position
carry-over (two small host round trips) — 46 switches in the run. Full-model reference: the B=1 traced token-out step is
23.9 ms (41.8 t/s/u, stage 8; the stage-9 single-slot server matched it at 23.91 ms).

## 1. What changed

| item | before (stage 9) | after (stage 10) | evidence |
|---|---|---|---|
| decode traces | one trace at max_batch_size (32): a lone user paid the 32-wide MoE step, 124 ms | one trace per power-of-two width {2, 4, 8, 16, 32}, selected per step from the plugin's front-packed active prefix (`tt/decode_buckets.py`, `_select_decode_bucket`); device token / position carry-over on a width switch | `DECODE_BUCKETS` histogram lines in the server logs; in-process step times width 2 / 4 / 8: 37.7 / 50.8 / 79.8 ms (`artifacts/w1probe-fix2.json`) |
| decode embedding input | `[B, 1]` token view -> `[B, 1, dim]` embedding -> tiled re-pack reshape at 1 < B < 32 (hung the device at width 2: `triage/width2_reshape_hang_triage.txt`) | `[1, B]` token view -> `[1, B, dim]` -> free view (`Qwen35MoeModel._decode_tokens_view`); B=1 / B=32 numerics unchanged (adapter streams byte-identical to stage 9) | `logs/10/adapter-test2.log` |
| GDN slot write at admission (B > 1) | host round trip of every GDN layer's state (~80 MB, ~120 ms of TTFT per admission) | device clones of the B=1 scratch written into the slot on device (`_snapshot_gdn_scratch_device`, `_write_gdn_slot_device`, before any further trace replay) | primary TTFT 474 -> 426 ms (single-slot stage-9 server: 352 ms; traced prefill: 344 ms) |
| decode-input contract | legacy `reset_batch` (plugin warning "correctness is not guaranteed") | `decode_input_update_contract = 1`: `reload_inputs / reload_page_table / reload_sampling_params / reset_sampling_state` mapped onto the shared Generator; `DECODE_CONTRACT` line | server log (no legacy warning), `test_vllm_adapter_stale_inputs` contract-v1 arms |
| overlap | README said "off"; in fact vLLM async scheduling was on and the plugin's v0 steady-decode fast path overlapped step N+1 with N's readback | declared (`supports_async_decode_overlap: True`, a key this plugin revision does not read); the plugin gates overlapped steps on the v1 reload plan | "Asynchronous scheduling is enabled" in every server log; page-growth controls |
| KV pool | 1,861,248 tokens (512 MiB trace region) | 1,820,160 tokens (768 MiB trace region accounted for in `get_max_tokens_all_users`; `QWEN35_TRACE_REGION_MIB` must match the TT config, asserted at build) | server log `KV token pool` line |
| program guard evidence | `PROGRAM_GUARD` / `PROGRAM_CACHE` lines (stage 9 rework) | + `DECODE_BUCKETS`, `DECODE_CONTRACT` lines after every admission | server logs |

## 2. Decode bucketing

The vLLM plugin keeps live requests front-packed (`condense`) and marks idle rows with position -1. The adapter picks the
smallest captured width >= the active prefix, binds that width's trace store (trace ids, persistent inputs, output) into the
shared Generator and keys the sampler's trace namespace by the width (`sampling.set_trace_bucket`). The GDN recurrent /
conv state ([32, ...]) is shared: the width-W step runs rows [0:W) (`TPGatedDeltaNet.forward_decode`), paged attention and
the sampler address the full slot space through the page-table rows and the `[1,1,1,32]` token buffer. Host-logits
compatibility mode (host sampling) and `slot_remap` steps run at the full width. On a width switch the previous width's
device tokens / positions are copied into the selected width's inputs (32 integers, host round trip: a device copy is a
program) and, if the plugin did not flag the step, `reset_batch` is forced. Warm-up: every width's programs compile
eagerly in phase 1; phase 2 captures one decode trace per width (host-logits variant at 32 only), then the prefill bucket
traces; trace region 578 of 768 MiB with every trace parked (`MEMORY_REPORT` of the final run; 603 MiB when the width-1 trace was still captured).

Cross-width numerics: the width-W step reproduces the B=W build's numerics (MoE accumulation order follows the token
count), so streams at different widths agree only up to the first near tie (2-layer test: a 0.03-logit fork; 48 layers:
the width-2 and width-1 streams of the same prompt fork at token 2). The plugin's determinism tests compare seeded /
greedy streams within one server configuration.

## 3. Width 1: intermittent device non-determinism (why the minimum width is 2)

The first bucketed 48-layer run failed 3 sampling tests (`test_batch1_seed_reproducible[0,1]`, `test_mixed_params_batch`):
one seeded user re-run 10 times gave 10 different streams. Localisation (work_log 05:20-07:40, all artifacts under
`~/qwen35-122b-bringup/artifacts/`):

| probe | result |
|---|---|
| held bucketed server, N identical seeded requests x R rounds (`seedprobe-base.json`) | width 1: 6 unique / 6; widths 2, 4, 8, 16: 1 unique of 12 / 24 / 24 / 48; GREEDY 64-token re-runs at width 1: 6 unique / 6 (identical first ~160 chars, forks later) |
| 2-layer adapter test, seeded widths 1 / 2 / 4 | reproducible -> the seed manager / sampler mechanics are width-agnostic |
| in-process 48-layer, width 1: traced sync / traced async-overlapped / eager | all differ run to run -> device ops, not tracing or readback |
| masked MoE path forced at width 1 (`w1probe-masked`) | 6 / 6 identical, but 69.5 ms per step (masked costs ~70-80 ms at any T <= 8) |
| layer bisection at the forking step (`bisect-2.json`) | layers 0-3 bit-identical; LSB differences (1 bf16 LSB) start in layer 4's GDN mixer or MoE and cascade to a token flip at a near tie |
| gather rewrites at T = 1 (explicit copies, DRAM outputs, plain reshape/sum tail, duplicated-token T = 2 code) | all non-deterministic (4-6 unique / 6) |
| attention block padded to 2 rows, GDN block padded, both | all non-deterministic (5-6 unique / 6) |
| two REAL live rows (width 2), any configuration | bit-reproducible in every probe (> 150 streams) |

Conclusion: a timing-dependent race of a B = 1-specific device path that no block-level rewrite reachable from this
model's Python removes; it is the stage-7 "intermittent attention-layer traced-decode non-bit-identity at B=1" at full
scale. Isolating the kernel needs tt-metal-level tooling (open item for the tt-metal team; probe knobs left in the code:
`QWEN35_MOE_GATHER_T1_SHORTCUT`, `QWEN35_MOE_GATHER_MAX_TOKENS`, `Qwen35MoE.GATHER_KNOBS`, `QWEN35_SDPA_DECODE_GRID`).
Decision: `QWEN35_DECODE_MIN_WIDTH=2` — a single live user runs on the width-2 trace with one idle row (37.7 ms per step
in-process; 3.3x faster than stage 9's 124 ms, 1.5x slower than the model's 25 ms width-1 step). The standalone B=1
generator (readiness teacher forcing / token-out) is unchanged.

**Seeded requests run at the full width.** Per-row numerics differ between widths (gather-mode MoE at widths 2-4 vs the
masked path at 8+, SDPA-decode cores per head 16 / 16 / 13 / 6 / 3), so a request whose width history depends on the rest
of the batch (admission order, others finishing) can flip a sampled token at a near tie: the second bucketed run failed the
plugin's `test_mixed_params_batch` that way (`readiness_vllm_48l-minwidth2/NOTE.md`). A batch with any live seeded row is
therefore decoded at width 32 (the stage-9 numerics, composition-independent; `QWEN35_SEEDED_FULL_WIDTH=0` disables), the
`DECODE_BUCKETS` log line counts those steps. Unseeded requests (greedy, or sampled without a seed — the common serving
case) use the buckets; a greedy stream can still fork at a near tie when its batch composition differs between runs (the
same LSB-class effect documented since stage 7).

## 4. Concurrency and long context

Held 48-layer server (`~/qwen35-122b-bringup/scripts/10-hold-chain.sh`, same TT config; `readiness_vllm_48l-hold/server.log`):

| workload | runner | TTFT p50 / p99 | TPOT mean / p99 | ITL p50 / p99 | per-user decode | aggregate | artifact |
|---|---|---|---|---|---|---|---|
| 128 / 128, 16 requests, concurrency 8 | `run_vllm_server --stages benchmark --server-url` | 3.24 / 3.41 s | 93.8 / 97.3 ms | 92.8 / 98.1 ms | 10.7 t/s/u | 67.4 tok/s | `concurrency/vllm_benchmark_c8.json` |
| 128 / 128, 64 requests, concurrency 32 | same | 12.0 / 12.1 s | 131.1 / 132.9 ms | 132.9 / 136.4 ms | 7.6 t/s/u | 142.8 tok/s | `concurrency/vllm_benchmark_c32.json` |

Per-width served latency from the client (`scripts/10-direct.py`, N concurrent greedy 128 / 128 requests on distinct
128-token prompts, streaming; `direct_checks/direct_checks_stage10.json`):

| live users N | decode trace width | TTFT (first .. last request) | TPOT (per token after the first) | aggregate |
|---|---|---|---|---|
| 1 | 2 | 0.41 s | 37.7 ms (26.5 t/s/u) | 24.6 tok/s |
| 2 | 2 | 0.81 .. 1.08 s | 39.4 ms | 43.1 tok/s |
| 4 | 4 | 1.63 .. 1.91 s | 51.8 ms | 60.8 tok/s |
| 8 | 8 | 3.25 .. 3.53 s | 93.1 ms | 66.8 tok/s |
| 16 | 16 | 6.43 .. 6.71 s | 107.6 ms | 100.6 tok/s |
| 32 | 32 | 12.97 .. 13.25 s | 125.1 ms | 140.6 tok/s |

TTFT grows by ~0.4 s per concurrently admitted request (prefills are serialised); the step time follows the MoE path of the
width (gather at 2-4: ~12.6 ms per token; masked at 8+: 80-125 ms). The single-user TTFT of 410 ms (vs 474 ms in stage 9 at
32 slots and 352 ms on the stage-9 single-slot server) reflects the device-side GDN slot write.

Long context (`/v1/completions`, streaming, 64 new tokens, greedy; `direct_checks_stage10.json` `long_*`): 131,072 prompt tokens ->
TTFT 132.2 s, completion coherent ("..., and their heated faces were distorted and convulsed. The teeth were bared, glaring
like the teeth ..."); 262,080 prompt tokens (= max_model_len - 64) -> TTFT 283.4 s, 64 tokens, coherent. Both run within the
KV pool (4096 blocks for the 256K request); the prefill cost per 2048-token segment grows with the position (attention over
the earlier blocks), ~1 ms per prompt token at 128K-256K vs 0.17 ms per token for a 2048-token prompt. The 262144 context
contract is served end to end; TTFT at that length is a documented limitation (chunked prefill and prefix caching are not
supported).

Page growth on the bucketed traces (`page_growth_chat_buckets`: 3 distinct chat requests x 400 tokens concurrently on the
width-4 trace, each growing 1 -> 7 blocks, vs each request alone on the width-2 trace): completion_tokens 400 / 400 / 400,
coherent; the concurrent and solo texts fork at tokens 11 / 53 / 55 into equally coherent continuations ("...can keep your
milk cold and your ice cream frozen" vs "...can keep your food cold when the kitchen is warm") — per-row numerics differ
between decode widths (section 3), so a solo run is not a same-width control here. The same-composition control
(`direct_checks_stage10_part2.json`, `scripts/10-direct2.py`, chain stdout `~/qwen35-122b-bringup/logs/10-hold2-chain.log` and `logs/10/direct-checks-stage10-part2.log`; the hold2 server's own log was NOT saved — the chain script's copy step had a wrong `mkdir` path and its runner log name collided with the first hold chain's, both fixed afterwards — so the widths of that server are established by the script's fixed configuration (`QWEN35_DECODE_BUCKETING=1`, 768 MiB, min width 2 = the same code as the final run), not by a log line): the three chat requests
run twice concurrently on the width-4 trace -> byte-identical [True, True, True] (400 tokens each, 1 -> 7 blocks per
request); stage 9's raw 3 x 300-token book check run twice concurrently -> [True, True, True] (3 -> 8 blocks each); the
1000-token non-aligned prompt of stage 9 run three times alone on the width-2 trace -> 1 unique continuation, byte-identical
to the stage-9 served text ("apped out theirs. Mere messages in the earthly order, ..."). Automatic page growth across block
boundaries under concurrency is therefore reproducible on the bucketed traces; only the width changes numerics.

## 5. Contract, async and fallback evidence

- **Async decode**: the plugin calls `decode_forward(..., read_from_device=False)` (device handles back; "Decode trace replay
  active (... split sampling, deferred readback)" in every server log), then `read_decode_output(async_read=True)` /
  `process_decode_output_host`; the base Generator replays each width's trace with `ttnn.execute_trace(..., blocking=False)`.
  vLLM's asynchronous scheduling is on ("Asynchronous scheduling is enabled." in the API-server and EngineCore logs) and the
  plugin's steady-decode fast path submits step N+1 before step N is read whenever its reload plan is overlap-safe; the
  adapter declares `supports_async_decode` and `supports_async_decode_overlap` (the latter is not read by plugin 5961e48).
- **Decode-input contract v1**: the legacy warning ("does not advertise decode_input_update_contract >= 1") is gone from the
  stage-10 server logs (0 occurrences vs 1 in stage 9); `DECODE_CONTRACT v1 steps N, host-authoritative reloads M` lines
  count the explicit commands (second run: 1672 steps, 485 reloads = admissions / completions / sampling-mode changes).
  Persistent trace inputs (tokens, positions, RoPE indices, page table, sampler params / seeds) are refreshed only on those
  commands; the page table only when the host table changed (`_decode_forward_trace_text`). Stale-input evidence:
  `test_vllm_adapter_stale_inputs` (wrong host token / position on non-reset steps -> device inputs win; a page table changed
  from shared to private blocks before a block crossing -> picked up; unchanged tables copy nothing) through both the legacy
  and the v1 command shapes, B=1 and B=4 (`logs/10/adapter-test9.log`, 6 passed).
- **On-device sampling**: `sample_on_device_mode: all`; prefill returns device-sampled tokens; decode samples on device through
  the per-width sampler traces; host sampling only for the plugin's host-only parameters (min_p, logit bias, logprobs,
  structured outputs), at the full width through the warmed host-logits trace. No host argmax, no full-logits readback on
  the measured path (code: `tt/generator_vllm.py`, `tt/generator.py::decode_forward`).
- **Compile-after-capture guard**: `PROGRAM_GUARD active` at construction; `PROGRAM_CACHE 1591 entries ... (after the
  warm-up captures: 1591; new: 0)` after every admission and every 2000 decode steps, in every stage-10 server log (1591 =
  five decode widths x sampler variants + host-logits + bucket programs; stage 9 had 1186 with one width).
- **Memory**: trace region 578 of 768 MiB with every trace parked (`MEMORY_REPORT`, second run), DRAM 29.99 of 31.08 GiB
  (the allocator total shrank by the extra 256 MiB of trace region); KV pool 1,820,160 tokens (the budget accounts for the
  region; a mismatch between `QWEN35_TRACE_REGION_MIB` and the TT config raises at build).
- **Cleanup**: the chains SIGTERM `api_server` / `EngineCore` and reset the devices after every run; the API server's process
  manager still reports one force-killed process at shutdown (as in stage 9); `pgrep` after every run showed no leftover
  device holder. **No Tracy / tt-perf-report / device profiler** was collected on the served path (skill rule; the device-op
  picture is stage 7's; `perf_summary.json` marks the device-time fields null with the reason).

## 6. Qualitative verdict (final run)

`readiness_vllm_48l-final/vllm_qualitative_outputs.json`: the six shared prompts (`runtime/readiness_check/vllm_prompts.txt`)
through `/v1/chat/completions`, greedy and sampled, each 800-1200 characters read in full: every completion opens with the
model's thinking process as in stage 9 and the HF controls (`doc/full_model/qualitative/prompt_*/hf_completion.txt`) —
haiku planning, supervised-vs-unsupervised plan, story plan, thermodynamics recall, French translation analysis, Fibonacci
approaches — coherent, on topic, English, no repetition loops, no cross-prompt contamination; `10-optimized-vllm.check.sh`
(degenerate-output scan + context contract) exits 0 (`logs/10/check.log`). Against stage 9's greedy completions of the same
prompts the texts are identical for the first 134-360 characters and then fork into equally coherent continuations: the
qualitative requests are single users, so they decoded on the width-2 trace (stage 9: width 32) and the per-row numerics
differ between widths (section 3).

## 7. Rejected / deferred options

- Width-1 decode trace (25.1 ms): rejected until the device non-determinism is fixed (section 3).
- Masked MoE at narrow widths: deterministic but 70-80 ms per step.
- Per-token gather at widths 2-4 costs ~12.6 ms per extra token (op-count bound, ~8 ops per token per layer): a batched
  gather-mode sparse matmul over T tokens would remove it (kernel work, deferred).
- Untested cell: a width-2 trace whose MoE gather covers only the live row (T = 1 gather inside the B = 2 step) would recover
  the idle row's ~12.6 ms gather cost, IF the non-determinism is not in the MoE (the `dup2` variant and the bisection's first
  difference in a GDN layer point that way, but the padded-attention / padded-GDN probes did not settle it). Deferred: measure
  after the width-1 race is understood.
- Chunked prefill / prefix caching: unchanged (not supported).
- Not broken down: ~75 ms of single-user TTFT at 32 slots remain above the stage-9 single-slot server (426 vs 352 ms; traced
  prefill 344 ms): the 36 per-layer `write_slot` slice / concat / copy programs of the device-side slot write, the
  32-slot sampler-parameter upload at admission and the first decode's width binding are the candidates; a per-request
  engine-side timer is the next step (no profiler on the served path).
- Trace region 512 -> 768 MiB (the only TT-config change vs stage 9): the KV pool shrinks 1,861,248 -> 1,820,160 tokens
  (-2.2%; 56,880 vs 58,164 tokens per user at full occupancy).
- Every admission that condenses the batch (`slot_remap`) runs one full-width decode step (~124 ms) before bucketing resumes
  with a forced host-authoritative reload: a per-admission latency blip, counted in the `DECODE_BUCKETS` line
  (`slot_remap full-width steps`, `forced resets`; added after the final run, so its log carries only `switches 46`).

## 8. Artifacts

Repo hygiene: `bringup/` at the tt-metal root (the stage-6 HF reference tensors, `bringup/references/*.refpt`) is untracked
workspace input and is not part of any stage commit; the plugin registration hunk lives in the separate `~/vllm-tt-plugin`
checkout (committed locally there, not pushed). The width-1 device race has no tracked tt-metal issue yet: the repro is
`~/qwen35-122b-bringup/scripts/10-w1probe.py` + `10-bisect.py` with `artifacts/{w1probe*,bisect-2}.json` (open item).


`readiness_vllm_48l-buckets/` (first bucketed run, sampling suite 70 / 3 failed: the width-1 evidence), `readiness_vllm_48l-minwidth2/`
(second run, 72 / 1 failed: the width-history evidence), `readiness_vllm_48l-final/` (the evidence run: PASS), `concurrency/`, `direct_checks/`, `triage/width2_reshape_hang_triage.txt`, `work_log.md`; workspace
`~/qwen35-122b-bringup/{scripts/10-*.sh,10-*.py, logs/10/, artifacts/{w1probe*,bisect*,gatherprobe*,seedprobe*}.json}`.
