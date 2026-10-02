# Perf pass (stage 13) work log — TTFT and tok/s/u across widths, incl. width 1

User request (2026-09-29 ~16:40 UTC): after stage 12, capture the low-hanging perf fruit turn by turn; TTFT and tok/s/u at
every batch size including 1; first fix the width-1 non-determinism so one user runs its own decode trace.

## Width-1 non-determinism (stage 10 open item; single user served on the width-2 trace, 37.7 vs 25.1 ms per token)
- 16:48 probe 1 (`scripts/12-w1sync-probe.py`, chain5): ran at width 2 by mistake (QWEN35_DECODE_MIN_WIDTH default 2) and
  the ttnn post-op hooks were inert under fast runtime mode -> 1 unique of 6 everywhere, eager step 129 ms: no information.
- 17:33 probe 1b (chain5b: QWEN35_DECODE_MIN_WIDTH=1, TTNN_CONFIG_OVERRIDES fast_runtime_mode=false): baseline 4 unique
  of 6 (eager step 553 ms: the host is slow with hooks, the device idles between ops); `sync_all` (ttnn.synchronize_device
  after EVERY op) 5 unique of 6 -> NOT an inter-op race; an op is non-deterministic on identical inputs at the B = 1 shapes.
- next: op-level bisection (`scripts/13-w1opbisect.py`, chain6): hash every op output during decode steps across 4
  repetitions; first differing op with identical earlier outputs = the culprit.

## sorted_min_tokens sweep (does the group-parallel factory make the expert-sorted MoE win at decode widths 8-32?)
- 17:06 baseline (threshold pinned at 256 by MULTICHIP_SELECTED; masked path at decode): per layer traced decode b8 1.99 /
  b16 2.35 / b32 2.78 ms (GDN), 1.91 / 2.27 / 2.66 (attention); prefill_128 17.4 / 15.2 ms, prefill_256 18.2 / 16.8 ms.
- the env knob did not reach the model (MULTICHIP_SELECTED.sorted_min_tokens=256 overrides the OptimizationConfig
  default); fixed: QWEN35_SORTED_MIN_TOKENS now overrides the pinned value in `MultichipConfig.opt()`; rerun at 8 queued.
- 18:09 sweep at 8 (`logs/13/sorted-min-sweep-8.log`): traced decode per layer b8 4.16 / b16 4.37 / b32 4.77 ms (GDN) vs
  1.99 / 2.35 / 2.78 masked -> the sorted path LOSES at decode: its dispatch bookkeeping (~170 ops per layer) is op-launch
  bound at T <= 32 while the masked path is weight-bandwidth bound. prefill_128 13.5 / 12.1 ms vs 17.4 / 15.2 (untraced,
  host-bound, noisy); prefill_256 unchanged. Decision: keep the pinned 256 for prefill, masked path for decode widths >= 8.
- Roofline note: at width 32 a decode step touches ~all 256 experts -> 1.15 GB of bfp4 expert weights per layer per die,
  55 GB per step over 48 layers -> ~108 ms at ~512 GB/s; measured 125-133 ms. Widths >= 16 are within ~20 % of the DRAM
  roofline; the levers are width 1 (25.1 vs 37.7 ms: non-determinism fix) and widths 2-8 (per-token gather loop: 12.6 ms
  per extra token at widths 2-4, 80 ms masked at 8, vs a weight-read floor of ~0.3 ms per layer at width 4).
- 18:12-18:16 correctness at threshold 8 (`logs/13/sorted-min-8-correctness.log`): 12/12 pass (batched decode B=32,
  distinct positions, seq37/129/333 both kinds, stress determinism) -> the sorted path is correct at decode widths, only
  slower; not adopted for decode.

## Turn 2 design: decode "pairs" MoE path for widths 1-8 (`Qwen35MoEOptimized._routed_pairs`, QWEN35_MOE_PAIRS_MAX_TOKENS)
- G = T*k groups of one tile-padded activation row each (token row repeated k times), `indices` = the flattened top-k ids,
  ONE group-parallel sparse matmul for gate/up and one for down, swiglu in between, routing weights as a [1,G,1,1]
  broadcast multiply, sum over the k experts of each token as a batch-dim reduction. ~12 ops per layer instead of the
  per-token gather loop (~12 ops x T, 12.6 ms per extra token) or the serial masked path at width 8 (80 ms). Validation
  chain `13-chain7-pairs.sh`: layer perf b1..b32 with the knob off / on, then the decode correctness + determinism suite.
- 18:27-18:40 op bisection run 1 (`artifacts/stage13/w1opbisect.json`): 4 repetitions, streams bimodal (978,22235 vs
  8660,1141 at tokens 2-3); every op output up to the first attention layer's `paged_update_cache` is bit-identical
  across runs, incl. the K/V rows fed to it; the cache tensor itself differs only because it is persistent state
  (stale rows from the previous repetition) -> not evidence. Probe fixed to skip state tensors and list all differing
  indices; rerun queued (chain6b).
- Op mix of ONE width-1 decode step (hooked eager, 48 layers): 7053 top-level ops = 2448 deallocate + ~4600 device ops
  (832 reshape, 757 slice, 449 multiply, 326 to_memory_config, 229 rms_norm, 193 linear, 192 matmul, 192 view, 184 add,
  144 copy, 144 mac, 97 all_gather, 96 reduce_scatter, 96 sparse_matmul, 84 transpose ...) -> ~96 device ops per layer;
  at ~3-5 us of device dispatch per traced op the 25.1 ms width-1 step is largely op-count bound. Op-count reduction is
  the general lever for widths 1-8 (the pairs path is that lever for the MoE block).
- 18:36 pairs perf attempt 1 (chain7): every b1..b8 case failed inside `_routed_pairs` (the data-moving reshape rejects
  uint16 ids; at b1 a reshape volume mismatch), b16/b32 unchanged (masked path). Fixes: ids reshaped as uint32 then
  typecast to uint16 for the final view; activation repeat via `ttnn.repeat` + RM view; the k-sum's padded row count
  trimmed before the final reshape. Retry with full pytest logs (chain7b) queued behind the bisection rerun (chain6b).
- 18:50 op bisection rerun (`artifacts/stage13/w1opbisect-b.json`, 4 repetitions, streams identical this time but
  ~3640 of 7053 op outputs differ at step 0 from the first MoE block on): FIRST non-state difference = the gather-mode
  `ttnn.sparse_matmul` (gate/up, indexed, compact output [1,1,1,8,1,512]) at op #847, with bit-identical inputs (#841-#846:
  the routed ids slices/views/reshapes and everything before). Only the first half of its 512 output columns differs
  (#849 `slice` cols 0..255 != , #850 `slice` cols 256..511 ==) -> the multicast-1D sparse factory's gather mode at M = 1
  is non-deterministic in part of its N range (per-core), consistent with stage 10 (masked path deterministic, every
  gather variant not). The group-parallel factory (no multicast, plain bmm reader per core) is the natural replacement:
  the pairs path uses it at widths 1-8.
- 18:56 pairs attempt 2: all widths failed at `reshape(weighted, (T, k, 1, H))`: `ttnn.swiglu` returns the tile-padded
  32 rows as its logical shape ([1,G,32,Nl]), so down / multiply carry 32 rows. One-die shape probe
  (`scripts/13-pairs-shapes.py`): rows are zero padding through the chain; fix = keep the padded rows in the batch-dim
  split and slice one row after the k-sum. Probe PCC vs fp32 reference 0.99988 (bfp4 weights). chain7c (perf b1..b8 GDN
  full log -> all widths both kinds -> correctness) and chain8 (48-layer width-1 determinism + traced step times with
  pairs=8, min width 1) queued.
- 19:01 pairs perf (chain7c, GDN layer, traced, per step): b1 0.588 ms (gather loop 0.524), b2 0.793 (0.803), b4 0.985
  (1.078), b8 1.430 (masked 1.987, -28 %), b16/b32 unchanged (masked). The MoE is a small share of the width-1 step
  (~96 device ops per layer dominate), so at width 1 the pairs path matters for determinism, not speed; the width-8 gain
  is 95 -> ~69 ms per step at 48 layers. Tail trimmed for T = 1 (no row-major round trip).
- 19:03-19:12 chain7c: perf both kinds b1..b32 (attention: b1 0.495 vs 0.431, b2 0.699 vs 0.710, b4 0.877 vs 0.968,
  b8 1.282 vs 1.906 ms); correctness with pairs=8: 18/18 (`logs/13/pairs-8-correctness-full.log`: B=1 decode vs HF at
  37/333 both kinds, real weights, B=4 distinct positions PCC >= 0.9990 per user, non-aligned continuation, traced decode,
  prefill after traced decode, stress determinism). chain8 (48-layer width-1 determinism + traced step times) running.
- 19:14-19:35 chain8 (`artifacts/stage13/w1pairs-8.json`, 48 layers, pairs=8, min width 1): width 1 traced 5 unique of 6,
  eager 4 of 4, async-overlap 3 of 3 -> still non-deterministic with the grouped factory; widths 2 / 4 / 8 traced 1 of 3
  each (and equal to one of the two width-1 outcomes). Traced step (48 steps, deferred readback): w1 27.45 ms, w2 37.36,
  w4 46.56, w8 67.80, w16 85.87, w32 91.56 (stage 10 in-process: w1 25.1, w2 37.7, w4 50.8, w8 79.8). Since the pairs
  matmul at width 2 has exactly the width-1 geometry (one padded row per group) and IS deterministic, the sparse matmul
  kernel is not inherently non-deterministic at M = 1; the standalone repro (chain9) tests it in isolation incl. NaN /
  huge values in the padded input rows.
- 19:31-19:37 standalone repro (`scripts/13-sm-determinism.py`, one die, 40 identical repetitions per variant,
  `logs/13/smdet.log`): 1 distinct output everywhere: mcast gather / grouped factory x rows 1 / 32 x bfp4 / bfp8 / bf16
  weights x packer_l1_acc on/off x fp32 dest acc x HiFi2, and padded input rows 1..31 = NaN / 3e38 / 0 (row 0 hashed) ->
  the sparse matmul kernel is deterministic in isolation, also with garbage padding. The in-model difference needs the
  model's live memory state: in-situ replay probe (`scripts/13-w1replay.py`, chain10): every op of a width-1 decode
  step re-executed 3x on its live inputs, outputs compared bit-wise.
- 20:18-20:29 in-situ replay probe (`scripts/13-w1replay.py`, `artifacts/stage13/w1replay-c.json`): 3594 ops of a
  width-1 step re-executed 3x on their live inputs; 63 flagged: `generalized_moe_gate` in EVERY layer (136-328 of the
  2 x 1024 output elements differ, max |d| 5-12 = other expert ids) and `nlp_concat_heads_decode` in the attention layers
  (~90 % of elements: most likely an in-place / aliasing false positive). `fused_decoder._route` documents that the
  TTMoEGate op is used for the SINGLE-ROW decode only (>= 2 rows fall back to the softmax/topk router) and that it was
  measured bit-non-deterministic with several rows in stage 3 -> it is the width-1 / width-2 discriminator. Experiment:
  QWEN35_ROUTER_GATE_OP=0 (functional router everywhere) + width-1 stream probe (chain11).
- 20:30-20:50 chain11 (`artifacts/stage13/w1pairs-gate0.json`, QWEN35_ROUTER_GATE_OP=0 = functional router everywhere,
  gate op off): width 1 traced 1 unique of 6, eager 1 of 4, async-overlap 1 of 3, widths 2/4/8 1 of 3 -> ROOT CAUSE
  CONFIRMED: the fused router gate op (TTMoEGate / generalized_moe_gate, single-row decode only) is the width-1
  non-determinism. Width-1 traced step with the functional router 33.46 ms (gate op 27.45; served width-2 trace 37.4).
  The functional router (linear, softmax, topk, sum, div) costs ~125 us per layer at width 1. Next: does the gate op
  become deterministic with per-layer (unshared) scratch buffers as in stage 3 (chain12, QWEN35_SHARE_GATE_BUFFERS=0)?
- 20:48-20:57 chain12 (gate op on, QWEN35_SHARE_GATE_BUFFERS=0 = per-layer scratch as in stage 3): the 48-layer build
  fails with the stage-6 L1 clash ("statically allocated circular buffers ... clash with L1 buffers", program 1238) ->
  the gate op cannot run with per-layer buffers at 48 layers; it stays off at width 1 (it was already unused at >= 2 rows).
- Router cost: functional router 33.46 vs gate op 27.45 ms per width-1 step (~125 us per layer for linear, softmax,
  topk, sum, div). Optimization: `ROUTER_TOPK_FIRST` (QWEN35_ROUTER_TOPK_FIRST=1): top-k on the logits then softmax over
  the k selected values (identical math: the global partition function cancels in the renormalization), 2 ops instead of
  4. Queued: chain13 (width-1 probe with gate off + topk-first: determinism + step times), chain14 (layer correctness
  suite incl. prefill 2048 both kinds + full-model prefill-check p2048 with the new router), served run with the width-1
  trace (QWEN35_DECODE_MIN_WIDTH=1, gate off, topk-first, pairs).
- 20:58-21:20 chain13 (`artifacts/stage13/w1pairs-topk1.json`, gate off + topk-first + pairs, min width 1): width 1
  traced 1 unique of 6, eager 1 of 4, async 1 of 3, widths 2/4/8 1 of 3 (the near tie at token 2 now resolves to the
  other of the two earlier outcomes at every width: a numerics change of the renormalization, consistent across widths).
  Traced step: w1 32.69 ms (functional 4-op router 33.46, gate op 27.45), w2 36.46, w4 45.73, w8 67.11, w16 85.12,
  w32 90.94 (-0.7..0.9 ms at every width vs the 4-op router). Remaining width-1 gap to the gate op ~5 ms = the router
  linear + topk + softmax on a 1-row tile (~100 us per layer; L1-resident router chain is a later experiment).
- 21:16-21:26 chain14 layer correctness with gate off + topk-first (`logs/13/router-correctness-full.log`): 28/28
  (prefill vs HF 5..2048 both kinds, chunked 4429, batched decode B=32 PCC 0.9996 / 0.9999, distinct positions, real
  weights, traced decode, stress determinism). Defaults flipped: router_gate_op off, ROUTER_TOPK_FIRST on,
  QWEN35_DECODE_MIN_WIDTH 1. Full-model prefill-check p2048 and the served width-1 run follow.
- 21:27-21:37 prefill-check book_p2048 with gate off + topk-first (`doc/perf_pass/readiness/prefill_check_book_p2048_router.json`):
  top1 0.98 / top5 1.00 / top100 1.00 (stage-12 router: 0.99 / 1.00 / 1.00; bars top5 >= 0.98, top100 = 1.0). Served
  run with the width-1 trace (min width 1) running.
- 21:37-21:56 served run with the width-1 trace, gate off, topk-first, pairs (`13-serve-chain-w1.sh`, `served_w1/`):
  c1 TTFT p50 408 ms, TPOT 32.7 ms (30.6 t/s/u; stage 10: 37.7 / 26.5); c2 38.1; c4 46.7; c8 67.7; c16 110.3 (client
  107.0); c32 130.5 (client 125.2). Seeded 1 unique of 6/12/24/24/48 at widths 1..16, greedy width 1: 1 unique of 6.
  Per-width client aggregate: 28.0 / 44.0 / 65.5 / 84.5 / 100.4 / 140.7 tok/s at 1 / 2 / 4 / 8 / 16 / 32 users.
- 21:59-22:07 turn 3a, L1-resident router chain (`QWEN35_ROUTER_L1=1`, chain15): GDN layer b1 0.709 -> 0.690 ms, b4 0.992
  -> 0.972, b8 1.431 -> 1.409; attention b1 0.601 -> 0.629, b8 1.267 -> 1.263 -> net < 0.5 ms per 48-layer step: rejected
  (knob kept off). The functional router's ~120 us per layer at width 1 is op-launch cost of three small ops.
- 21:58-22:05 bundle rebuilt from 4e7f349857d (`12-package.sh`, warm cache) -> image f6b91cc768b9, pushed private (Hub
  revision 5fef4bd0). 22:11 Hub verification: `tt-model serve --refresh` moved 1e17833e -> 5fef4bd0, new image started.
- 22:11-22:25 Hub verification of the turn-1 bundle (rev 5fef4bd0, image f6b91cc768b9): all checks pass; 8K 4.5 s, 128K
  72.6 s; near-tie greedy texts differ from the stage-12 image (default greeting "Hello!"). Host-profiling run (chain16:
  py-spy on the EngineCore at c8 / c32) next.

## Turn 3b — host-side per-step overhead (served step 20-35 ms above the in-process trace at 16-32 users)
- 22:26-22:40 chain16: held server with the shipped defaults, py-spy (200 Hz, 30 s) on the vLLM EngineCore during the
  readiness benchmark at c8 and c32 (`doc/perf_pass/hostprof/pyspy_c{8,32}.raw`, `scripts/13-pyspy-summary.py`). c8
  window (32 requests at concurrency 8 = continuous admissions): 54 % of samples inside ttnn op calls; decode_forward 26 %
  of which `sample_decode_on_device -> apply_decode_state -> reset_sampling_params` 20 % (the common sampler re-applies
  the whole batch's k / p / temperature / log-prob / penalty params on EVERY step: 4 host->device copies + resets);
  prefill admissions 30 % of which `_write_gdn_slot_device` 21 % (36 per-layer slot writes); `read_decode_output` 9 %.
- Fix for the first item: `generator_vllm._memoize_sampling_params` (QWEN35_SAMPLING_PARAMS_MEMO, default on): skip
  `reset_sampling_params` when the formatted params equal the last applied ones; any prefill-side or direct sampler
  write invalidates the memo. Validation chain17: readiness `sampling` + `qualitative` stages on the served path, then
  benchmark c1 / c8 / c32 with the memo on and off.
- c32 window (`pyspy_c32.raw`, 4158 samples): 60 % inside ttnn op calls; `apply_decode_state` 31.8 % of the engine's
  host time (penalties `_copy_host_to_device` 16.6 %, `tt_sampling.reset_params` 15.8 %), prefill admissions 32.5 % of
  which `_write_gdn_slot_device` 23.2 % (`write_slot` 19.8 %, `_slice_along` 11.1 %), `finalize_decode` 6.2 %,
  `read_decode_output` 5.9 %. Served TPOT in this window: c8 67.5 ms, c32 131.5 ms.
- 22:43-23:06 chain17 part 1 (memo on): readiness `sampling` stage = the plugin's sampling test suite 73 passed / 4
  skipped (stage 9: 73 / 4), `qualitative` 6 prompts coherent (`served_memo/sampling_tests.log`, `vllm_qualitative_outputs.json`).
  Benchmarks c1 / c8 / c32 with the memo on, then off, follow.
- 23:06-23:20 chain17 part 2, memo ON: c1 TTFT p50 406 ms / TPOT 32.7 ms, c8 3270 ms / 67.7 ms, c32 12054 ms / 130.5 ms =
  identical to the memo-off served numbers (32.7 / 67.7 / 130.5): the sampler-reset host time is overlapped with the
  device step; removing it does not shorten the served step at any width. Kept on (validated harmless, less CPU) but
  no TPOT gain. The served - in-process gap (c32 130 vs 91 ms, c16 107 vs 85) therefore is not host CPU inside the step:
  probe queued (`13-stepsync-probe.py`): traced step at widths 1 / 8 / 32 with deferred readback (the in-process number),
  blocking per-step readback, and the plugin's overlapped async readback.
- 23:20-23:33 chain17 part 3, memo OFF controls: c1 32.6 ms, c8 67.7 ms, c32 130.5 ms TPOT (TTFT p50 398 / 3261 / 12006
  ms) = the memo-on values within noise. Memo: no served effect; stays on as validated, CPU-saving hygiene.
- 23:43-00:00 step-sync probe (`artifacts/stage13/stepsync.json`, 48 layers, 40 steps each): width 1 deferred 32.69 /
  blocking readback 32.86 / plugin-style overlapped 32.68 ms; width 8 67.10 / 67.37 / 67.10; width 32 90.94 / 91.21 /
  90.94 -> per-step readback synchronization costs < 0.3 ms at every width. The served step at 16-32 users (107 / 130 ms
  vs 85 / 91 in-process, greedy in both) is therefore neither host CPU nor sync: the served decode step does more device
  work. Candidates for the next turn: per-step page-table upload and its consumers (32 x 4096-block table), slot-remap /
  reset_batch full-width steps, the served sampler variant; needs per-step phase timestamps in the adapter.
- 23:56 chain18b `test_perf_slot_write` (GDN layer, B=32): write_slot alone 0.80 ms (rebuild) vs 0.77 ms (slice_write) per
  admitted user; prefill(37)+write 11.8 vs 10.6 ms. The slot write is host-launch bound (conv taps + slices), the traffic
  saving is invisible -> knob default off. Turn 3 summary: both "obvious" host fixes are latency-neutral; the served
  wide-batch gap needs the API-server / engine split (chain20 running).
- 00:09-00:15 chain20 (API-server vs engine at c32): api_server 4 % CPU (median), engine 411 % (ttnn dispatch threads);
  TPOT 132.9 ms; /metrics TTFT mean 12.04 s -> the API server is not the bound. RESOLUTION of the served-vs-trace gap:
  the in-process 48-layer probes fed all users the SAME prompt, so the router chose the same few experts for every row
  and the masked MoE read a fraction of the expert weights (w32 91 ms); served users differ and touch ~all 256 experts
  -> the weight-bandwidth floor (~108 ms per step) + overheads = 125-131 ms. The layer perf test with distinct users
  already showed it: 2.786 ms x 48 = 134 ms. Widths 16-32 are at the roofline; no host gap exists.
- Turn 4 plan: short-prompt TTFT. The masked prefill path at 128-256 tokens iterates ~all experts serially per chunk on
  32 cores (mcast factory); the expert-sorted path now has group-parallel matmuls -> sweep QWEN35_SORTED_MIN_TOKENS 256
  (shipped) vs 128 vs 64 on the 48-layer traced prefill bench at 128 / 161 / 256 / 333 / 512 tokens (chain21).
- 00:13-00:32 chain21 48-layer traced prefill bench (`prefill_bucket_bench_48layers_min{256,128,64}.json`): threshold 256
  (shipped): 128 tok 344.9 ms, 161 347.1, 256 343.1, 333 436.3, 512 437.5; threshold 128: 128 tok 303.2 ms (-12 %),
  161 348.1, 256 344.8, 333 436.8, 512 436.9 (unchanged: bucket 256 was already sorted). The 128-token bucket through the
  group-parallel sorted path beats the masked path; 64 = 128 (no smaller bucket), run as a repeat control. Adoption
  checks queued (chain22: layer PCC at 129 / 333 both kinds, non-aligned continuation, dispatch vs sort, stress;
  full-model prefill-check book_p128; served c1 TTFT).
- 00:42-00:46 chain22 layer correctness at threshold 128: 13/13 (prefill T=129 GDN PCC 0.99944, seq333, non-aligned
  continuation, dispatch vs sort, stress bit-identity). Default flipped: MULTICHIP_SELECTED.sorted_min_tokens 256 -> 128.
- 00:47-00:57 prefill-check book_p128 at threshold 128 (`readiness/prefill_check_book_p128_min128.json`): top1 0.95 / top5
  1.00 / top100 1.00 (stage 7, masked: 0.97 / 0.99 / 1.00). Served c1..c32 + seedprobe with threshold 128 running.
- 00:56-01:15 served with threshold 128 (`served_min128/`): TTFT p50 c1 371 ms (was 408), c2 854 (914), c4 1649 (1793),
  c8 3004 (3234), c16 5693 (6274), c32 11052 (12120); TPOT unchanged (32.6 / 38.1 / 46.7 / 67.7 / 109.9 / 130.0); seeded
  1 unique of 6..48 at widths 1-16, greedy width 1 1 of 6; client aggregate 28.5 / 45.1 / 68.0 / 88.4 / 106.2 / 152.0 tok/s.
  Adopted: MULTICHIP_SELECTED.sorted_min_tokens = 128.
- 01:25 width-1 follow-up sized: `ttnn.experimental.kda.qkv_causal_conv1d_silu` (ROW_MAJOR DRAM [B, seq, C] + [B, K-1, C]
  state, split q/k/v outputs) does not fit the TILE/L1 GDN decode shift register without ~6 surrounding layout/state ops
  -> not pursued; a fused GDN decode step is kernel work. Turn-4 bundle rebuild -> push -> Hub verify running.
- 01:16-01:37 turn-4 bundle: image c6e9186de5d4, Hub revision 024095e6, verified from the Hub (`logs/13/verify-hub-t4.log`):
  all checks pass; burst 137 tok/s (was 127), 8K 4.5 s, 128K 72.6 s; default greeting back to the long form (near tie).

## Turn 5 — long-prompt prefill: larger segments (user question: "why is prefill still serialized?")
- Within a request every prefill op already spans all 440 cores (TP 4); the serial parts are the segment loop (2048
  tokens per bucket program, one after another) and the scheduler's one-prefill-at-a-time across requests. Per 2048-token
  segment ~1.0 s over 48 layers (~20 ms per layer): expert matmuls ~5 ms, attention / GDN / projections ~7 ms, MoE
  dispatch bookkeeping ~8-10 ms in ~170 small ops (mostly fixed cost per op) -> the lever is fewer, fuller segments
  (then fusing the bookkeeping; batched prefill across requests only for concurrency fairness).
- Code (behind QWEN35_PREFILL_SEGMENT, default 2048): `Qwen35MoeModel.PREFILL_SEGMENT`, `_PREFILL_MASK_BUCKETS` gains 4096
  (and 8192 only when asked; the sorted dispatch index math is exact to 7196 tokens, so 8192 would need the masked path or
  a check), segmented prefill + `_prefill_single` use the knob, the warm-up page table covers the longest bucket.
- Queued: chain23 (layer Tracy of the MoE phases at 4096 tokens; whole-layer walls 2048 vs 4096), chain24 (48-layer traced
  prefill bench 2048 / 4096 / 8192 / 16384 tokens with segment 4096 vs 2048, 1 GiB trace region).
- 01:45 chain23 Tracy MoE at 4096 tokens (`artifacts/stage13/tracy_moe_4096`): device 40.54 ms vs 2 x 23.24 at 2048 ->
  -13 % per token; every major op scales ~linearly with rows (BinaryNg on the one-hot 4.52 -> 8.15 ms, sparse matmuls
  4.68 -> 7.82, counting matmuls 2.69 -> 4.94, 1-core scatters 2.40 -> 4.66, embeddings 2.12 -> 3.09, slices 1.66 -> 3.29).
  Host walls with syncs: dispatch 6.66 -> 11.55, gate/up 2.87 -> 4.71, down 2.75 -> 4.39, undispatch 2.52 -> 4.23.
  So larger segments amortize little; the dispatch bookkeeping (~11 ms device of the 23 at 2048) is row-proportional
  work on the [Rp, E] fp32 one-hot + single-core scatters -> a leaner dispatch (sort_u16 keys + O(Rp)/O(E) vectors, or a
  fused kernel) is the prefill lever. chain25 queued: Tracy of the MoE phases with QWEN35_DISPATCH_MODE=sort_u16.
- 01:46 chain23 whole-layer walls: 2048 GDN 87.9 / attention 85.2 ms; the 4096-in-one-call layer test failed at a program
  build (`program.cpp:2065`, L1 circular-buffer assert) in the non-MoE part of the layer -> some attention / GDN /
  projection program config does not fit 4096 rows; the first 48-layer bench attempt died on a class-scope NameError
  in the new bucket-list expression (fixed: the env is read inline). chain24 re-queued behind the sort_u16 Tracy.
- 01:55 chain25 Tracy of the MoE phases with `sort_u16`: the device sort of 16K uint16 keys alone costs 17.4 ms at 2048
  tokens (dispatch build 25.0 ms host wall vs 6.7 for counting_mm) -> sort-based dispatch is dead; the counting sort stays.
- 02:09 chain24: the 4096-token-segment 48-layer bench segfaulted (exit 139, after the whole-layer L1 CB assert at 4096
  rows in chain23) -> segment stays 2048 (`QWEN35_PREFILL_SEGMENT` kept as a knob, default 2048, documented as unsafe
  above 2048). Control run at segment 2048 (`prefill_bucket_bench_48layers_seg2048.json`): 2048 / 4096 / 8192 / 16384
  tokens = 984.5 / 1969.5 / 3959.0 / 7993.2 ms traced (0.48-0.49 ms per token, linear).

## Turn 5 (continued) — the dispatch bookkeeping as ONE kernel: `ttnn.experimental.moe_sorted_dispatch`
- New op (`ttnn/cpp/ttnn/operations/experimental/moe_sorted_dispatch/`, classic Program API, one core, one RISCV_0
  dataflow kernel): reads the tiled `[1,1,T,k]` top-k outputs in place (uint16 ids, bf16 scores; the tile/face index math
  is in the kernel, so no typecast / untilize / reshape in front), does the integer counting sort over the T*k pairs
  in L1 (histogram, padded per-expert offsets, group ids, scatter of source token / weight / destination row) and
  writes four sticks: `src_tok` uint32 [1,Rd], `weights` bf16 [1,Rd], `group_ids` uint16 [1,G], `undispatch` uint32
  [1,T*k]. Python: `SortedDispatch._build_kernel` (QWEN35_DISPATCH_MODE=kernel); the outputs keep the exact row layout
  of the counting sort. First attempt exposed only `ttnn._ttnn.operations.experimental...`; the nanobind name
  `moe_sorted_dispatch` is re-exported at `ttnn.moe_sorted_dispatch` (namespace renamed `moe_dispatch_nb` after a clash
  with the function name). Build note: `ninja -C build_Release ttnn` links to `build_Release/ttnn/_ttnn.so`
  (NOT `build_Release/lib/`, which still holds the 09-10 build) and Python loads `ttnn/ttnn/_ttnn.so`: copy + atomic
  `mv` (a `cp` over the loaded file would corrupt a running job).
- 02:10 `test_dispatch_modes_equivalent` (T = 256 / 500 / 513 / 1000 / 1023 / 2048): kernel == counting_mm == counting_h
  bit for bit on all five tensors, first run. Host wall per build (incl. sync): 5.98 -> 1.66 ms at 2048 (1.41 -> 0.41 at 256).
- 02:12-02:18 Tracy of the MoE block at 2048 (`artifacts/stage13/tracy_moe_kernel{,2}`, device time, 4 dies merged):
  16.45 ms (counting_mm) -> 13.75 (kernel, stick inputs: the two data-moving RM reshapes cost 1.66 ms) -> 12.11 ms
  (kernel reading the tiled top-k outputs). The kernel itself: 1.19-1.26 ms (single RISC over 16K pairs); the rest of
  the dispatch build is now 0.4 ms (group one-hot for the sparse-op signature, weights relayout). Remaining MoE time:
  sparse matmuls 4.32, gather (embedding) 1.44, un-dispatch 0.73 untilize + 8 x (0.075 gather + 0.10 add) = 2.1,
  swiglu/weights 0.4, zero-fills 0.13 + 0.75 (see next), router 0.25.
- 02:21 the two `UnaryDeviceOperation FILL 0` before each sparse matmul (0.88 ms per layer) are `SparseMatmulDeviceOperation::
  create_output_tensors` zero-filling the caller-provided output because only the `nnz` compact shape was recognised as
  "fully written"; the grouped factory writes every one of its G groups too -> `compact_output = grouped || ...`
  (sparse_matmul_device_operation.cpp). Installed before the 48-layer bench of chain26.
- 02:20-02:25 chain26 layer tests with QWEN35_DISPATCH_MODE=kernel: 14 passed (prefill/decode vs HF at 513 / 1000 /
  2048, chunked prefill, real-weight, prefill after traced decode), PCCs unchanged (GDN 0.9985-0.9999, attention 0.99996+).
  48-layer traced bench + prefill-check (2048 / 8192 book references) running.
- Study of the user's `sjettTT/qwen3.8-flash-next_p150x4` bundle (`qwen38_bundle_study.md`): its MoE is EP4 through a
  forked `moe_compute` (not portable), but its 2048-row expert stream (1.84 ms/layer, weights read once per expert) and
  0.14 ms fused combine bound what our sparse path should reach: our gate/up + down read ~1.0 GB of expert blocks per
  layer per chip (once per 32-row group, ~3 groups per expert) for 340 MB of unique weights -> "one weight pass per
  expert" (ragged M per expert in the grouped factory) is the next big lever (~-2.5 ms/layer), then the un-dispatch.
- Un-dispatch kernel written: `ttnn.experimental.moe_gather_add_rows(rows RM [Rd,H], undispatch RM [1,T*k], T, k)` ->
  TILE [1,1,T,H]; multi-core (work item = 32-token tile x column block, ~128 items at 2048 x 3072), reader gathers the k
  rows of a token tile as row-major blocks, compute tilizes + adds in slot order (same bf16 association as the k adds it
  replaces), writer stores tiles. Knob QWEN35_UNDISPATCH_KERNEL (off until `test_gather_add_rows_matches_reference` passes).
- 03:07 chain26's prefill-check (2048 tokens) hung in the T=128 all-zero-token program warm-up (blocked in a device read,
  py-spy) -> killed; bench + layer tests had passed the same warm-up minutes earlier. 03:25 the un-dispatch kernel's own
  unit test hangs the device on its first call (T=500) -> the prefill-check process had imported `moe_dispatch.py` while
  QWEN35_UNDISPATCH_KERNEL still defaulted to on (flipped to off two minutes later), so the hang is the new
  `moe_gather_add_rows` kernel, not the dispatch kernel. Watcher run queued to locate it; a warm-up reproducer (chain27,
  dispatch=kernel, un-dispatch kernel off) queued to confirm the kernel-dispatch path is clean.
- 03:30 64-row expert groups (QWEN35_SORTED_GROUP_ROWS=64, Tracy `tracy_moe_kernel_gs64`, device time at 2048 tokens):
  sparse matmuls 4.32 -> 3.58 ms (509 vs 761 groups) but the row gather 2.04 -> 3.07, untilize 0.81 -> 1.05, un-dispatch
  +0.24 (Rd 24352 -> 32576 rows): MoE block 11.23 (32-row groups, zero-fill skipped) -> 11.83 ms -> rejected; the
  activation traffic of the wider padding outweighs the fewer weight re-reads. Also shows the expert matmuls are not
  purely weight-bandwidth bound (-33 % groups -> -17 % time).
- 03:52 chain27: the 48-layer generator init (all-zero-token warm-up of every bucket) with QWEN35_DISPATCH_MODE=kernel and
  the un-dispatch kernel off passes twice (128-token dev 256 / 258 ms) -> the kernel-dispatch path is clean; the chain26
  hang was the un-dispatch kernel. Padding-group skip unit test (`test_grouped_skip_padding_groups`): real groups
  bit-identical with 126 trailing + interleaved sentinel groups skipped; gate/up 2.59 -> 2.16 ms, down 1.78 -> 1.49 ms
  per call at 761 groups (host wall).
- 03:55 Tracy MoE at 2048, kernel dispatch + zero-fill skip (`tracy_moe_kernel3`): 11.24 ms device per layer (was 16.45
  with counting_mm): sparse matmuls 4.34, gather 1.44, dispatch kernel 1.26 (+0.4 ms glue), un-dispatch 2.1 + untilize
  0.73, swiglu/weights 0.4, router 0.25.
- 03:58 whole-layer Tracy at 2048 tokens, kernel dispatch (`artifacts/stage12/tracy_kernel`): GDN layer 20.70 -> 15.47 ms
  device (GDN block 4.2 unchanged, MoE 16.5 -> 11.3), attention layer 18.32 -> 13.12 (attention block 1.9); 36 x 15.47 +
  12 x 13.12 = 714 ms vs the 733 ms traced 48-layer bench (the rest: embeddings, norm, LM head, host gaps).
- 04:17 chain28: padding-group skip at layer level: 12 layer tests pass with PCCs identical to the non-skip run (the
  padding groups' outputs were never read); Tracy MoE at 2048 (`tracy_moe_kernel_skip`): sparse matmuls
  4.34 -> ~3.6 ms (host wall gate/up 2.86 -> 2.33, down 2.68 -> 1.67). Defaults flipped in code: QWEN35_DISPATCH_MODE
  = kernel, QWEN35_SKIP_PAD_GROUPS = 1. chain29 (final validation: bench, prefill-check, served) queued.
- 04:18-04:36 chain29 (shipped config: kernel dispatch + zero-fill skip + padding skip): 48-layer traced prefill 128 / 161 /
  2048 / 4097 / 8192 / 16384 tokens = 214.1 / 249.3 / 701.3 / 1596.3 / 2828.3 / 5744.9 ms (turn 4: 303.2 / 348.1 / 984.5 /
  2208.7 / 3959.0 / 7993.2: -29 % at every length); prefill-check book_p2048 vs HF top1 0.99 / top5 1.00 / top100 1.00
  (unchanged). 8192-token check and the served chain follow.
- 04:48 chain29 prefill-check book_p8192 vs HF: top1 0.97 / top5 1.00 / top100 1.00 (stage 12: 0.95 / 1.00 / 1.00).
  Committed c9b9139df6e; bundle rebuilt (image e63ad8e67d71, ccache build, new ops + bindings present in the image) and
  pushed private (Hub revision 2ce007aa, 04:48); from-Hub verification queued behind the served chain.
- 04:56-05:1x turn-5 bundle verified FROM THE HUB (`logs/13/verify-hub-t5.log`): healthy in 10.3 min, tt-model curl
  default / thinking on / thinking off pass, 32-request burst 155.3 tok/s (turn 4: 137), 131072-token prompt 54.5 s wall
  for 16 new tokens (turn 4: 72.6 s), 8192-token prompt 3.3 s (turn 4: 4.5 s), tool call, non-aligned 2381-token prompt.
- 05:19-05:23 chain29 served (`served_kernelskip/`): 1 user TTFT p50 257.6 ms (turn 4: 371), TPOT 32.0 ms (31.3 t/s/u);
  32 users TTFT p50 8158 ms (11052), TPOT 130.1 ms (floor); direct 8192 / 32768 / 131072-token prompts TTFT 2.89 / 11.8 /
  54.1 s (4.0 / 16.4 / 72.3), completions byte-identical. Turn 5 shipped.

## MTP turn (started 2026-09-30 04:50) — plan in mtp_plan.md
- 05:25 `test_mtp_head_matches_reference` (reduced model layers 0 + 3 and the MTP head as virtual layer 48, 8 single-row
  decode steps at positions 0..7 vs the torch reference's causal pass): runs end to end first time; logits PCC 0.947-0.981
  vs the bf16 reference, argmax agreement 5 / 8 - the comparison includes bfp4 experts + bfp4 LM head vs bf16, so the
  test now also reports the post-norm hidden PCC and the device hidden through a bf16 head (rerun queued).
- 05:27 un-dispatch kernel fixed: the watcher showed 92 of 110 cores done and the 18 two-item cores stuck (unpack UABD /
  math MWDD / pack in-kernel, reader in cb_reserve_back) -> the writer waited on the SAME accumulator buffer the compute
  kernel used as its add intermediate (two consumers: it drained a partial sum and the pointers desynchronised; the
  one-item cores won the race). Fix: a dedicated output buffer for the last add (+ hardware re-init between the tilize
  and the add phases). `test_gather_add_rows_matches_reference[t500]` passes under the watcher: bit-identical to the
  k x (gather + add) reference, 4.88 -> 1.14 ms host wall. Full unit test + layer tests + Tracy (chain30) queued.
- 05:30 MTP head rerun with the post-norm hidden PCC: 0.953-0.988 (logits through a bf16 head: the same), so the gap is
  in the mix / layer, not the bfp4 LM head. Reference point: the shipped bfp4 policy gives the full-attention layer
  0.993 (prefill) / 0.995 (decode) vs HF (`SHIPPED_POLICY_PCC`). Next run isolates the mix (norms + fc) and the layer
  alone (reference mix in). Un-dispatch kernel unit test: 3 / 3 bit-identical, 2.32 -> 1.18 ms host wall at 2048 x 3072.
- 05:36 MTP head isolation: the mix (norms + fc) matches the reference at PCC 0.99987-0.99989; the layer alone, fed the
  reference's mix output on a fresh KV cache, gives 0.955-0.986 (same as the end-to-end hidden PCC) -> the MTP layer
  itself is more sensitive to the shipped bfp4 policy than the main attention layers (0.995 under the same policy).
  Probe queued: the head test under routed-experts-bfp8 / attention-bfp8 / both overrides (QWEN35_PRECISION_CONFIG).
  Layer tests with the un-dispatch kernel: 12 / 12, PCCs bit-identical to the run without it.
- 05:38 Tracy MoE at 2048 with the un-dispatch kernel (`tracy_moe_kernel_undispatch`): 10.57 -> 9.43 ms device per layer
  (un-dispatch phase 3.05 -> 1.93: MoeGatherAddRows 0.44 ms replaces 8 x (gather 0.075 + add 0.10); the untilize of the
  down output, 0.81 ms, remains), 89 -> 35 ops per MoE block. Default flipped to QWEN35_UNDISPATCH_KERNEL=1; chain31
  (48-layer bench, prefill-checks, served) queued.
- 05:43 MTP precision probe (QWEN35_PRECISION_CONFIG overrides on the reduced model + head): routed experts bfp8 ->
  hidden PCC unchanged (0.954); attention + GDN mixers bfp8 (HiFi2) + routed bfp8 -> 0.992 (layer-only 0.992, reference
  top-1 inside the device top-5 8 / 8, argmax 6 / 8). The MTP layer's attention projections are the bfp4-sensitive part
  -> `Qwen35MtpHead` builds its layer on an args copy with bfp8 attention weights at HiFi2 (~20 MB per die;
  QWEN35_MTP_ATTN_DTYPE=bfp4 restores the shipped policy). Head test requeued under the shipped global policy.
- 05:50 chain31 bench (kernel dispatch + padding skip + un-dispatch kernel): 128 / 161 / 2048 / 4097 / 8192 tokens =
  190.8 / 225.4 / 645.0 / 1461.9 / 2611.1 ms (turn 4: 303.2 / 348.1 / 984.5 / 2208.7 / 3959.0: -35 %).
- 06:12 chain31: 16384 tokens 5287.6 ms (turn 4: 7993.2); prefill-check book_p2048 0.99 / 1.00 / 1.00 and book_p8192
  0.97 / 1.00 / 1.00 with the un-dispatch kernel (unchanged). Served chain running.
- 06:16 harness control (`test_layer_decode_rows_harness`): the MTP layer built with bfp8 attention weights, driven row by
  row through the decode harness with a random fractured residual in, matches the HF layer at PCC 0.99879 (the main
  layer-3 control passed too; its PCC line was filtered out of the log). Head test under the shipped global policy queued.
- 06:23 `test_mtp_head_matches_reference` PASSES under the shipped global policy with the head's bfp8 attention
  (hidden PCC 0.990, logits 0.982, reference top-1 inside the device top-5 8 / 8); the global attention-bfp8 probe gave
  the same, both-bfp8 0.992 (experts add ~0.002; not worth ~0.6 GB per die for the MTP layer).
- 06:38 chain31 served (`served_undispatch/`): 1 user TTFT p50 274.3 ms (noise; 257.6 in chain29) TPOT 32.0; 32 users
  TTFT p50 7488 ms (8158) TPOT 131.1; direct 8192 / 32768 / 131072 tokens TTFT 2.64 / 10.91 / 50.61 s (2.89 / 11.81 / 54.14),
  completions byte-identical. Shipping: package -> push -> verify from the Hub (13-ship.sh).
- 06:46 turn-5b bundle: image 1a248468eae8 packaged from bab8b2c662b (no host recompilation needed: only kernel sources
  changed), pushed private (Hub revision a16fa483); from-Hub verification queued behind the MTP acceptance run.
- 07:40 turn-5b bundle (image 1a248468eae8, Hub revision a16fa483) verified FROM THE HUB (`logs/13/verify-hub-t5b.log`):
  healthy in 10.0 min, /v1/models 262144, chat, tool call, tt-model curl default / thinking on / off, non-aligned
  2381-token prompt, 32-request burst 160.2 tok/s (turn-5a image 155.3, turn 4 137), 131072-token prompt 51.0 s wall
  (54.5 / 72.6), 8192-token prompt 3.1 s (3.3 / 4.5). Turn 5b shipped.
- 07:59 `tools/mtp_acceptance.py`: the first version (greedy pass, then reset + re-prefill + replay) stalled in the first
  decode after the second prefill (both on the full and the 2-layer model; not investigated - the tool now drafts inline
  during ONE greedy pass and scores afterwards); the 2-layer smoke runs the whole loop in ~1 s per prompt. Full 48-layer
  run queued (k = 4, 64 new tokens, prose / code / json / chat). `tests/test_mtp_rows.py::test_attention_rows_mode`
  (R rows of one user in one batched decode call vs the HF causal prefill) written and queued.
- 08:30 the acceptance tool's SECOND prompt hangs the device in the first eager decode after `reset()` + prefill (main
  thread in a futex wait, dispatch threads spinning), on the 2-layer model as well; a single generate-style pass per
  process is fine. Not the vLLM path (the server never re-prefills a finished generator this way); worked around by one
  prompt per process; to investigate when the MTP verify path needs repeated prefill + eager decode in one process.
- 08:40 first full-model acceptance run: 0 / 64 first drafts agree (prose) -> not precision: the head drafted at position
  P+1 against an EMPTY MTP KV cache; the MTP layer must process the whole prompt first (vLLM runs the MTP layer over all
  prompt tokens: row t+1 = token t+1 with the model's final-normed hidden at t; the row at P uses the first generated
  token). Tool fixed (head prefill from the bucket hidden captured in `_run_bucket_hidden` + `norm`), cross-check rerun.
  Serving impact: the MTP layer runs in prefill mode over every bucket during prefill (1/48 extra layer of work) and its
  KV cache is one more attention layer's worth.
- 08:43 `test_attention_rows_mode` (R = 5 rows of one user in ONE batched decode call, fresh cache, positions 0..4 with
  3 dummy rows at 5..7): PCC 0.76-0.88 on every row including row 0 (which only attends to itself) -> not a causality
  problem; single-row control on the same layer queued to split harness vs batched-path.
- 08:46 single-row control (`test_layer_decode_rows_harness[main3]`): layer 3 through the same decode harness PCC
  0.994-0.997 (the shipped-policy figure) -> the batched R-rows call is the fault: B rows of ONE user that all fall into
  the same 32-position KV tile are written by B per-user paged updates that race on the tile (last writer wins), so rows
  read wiped K/V. Verify-pass consequence: write the R rows' K/V with one sequence-style (chunk) update, keep the batched
  decode SDPA for the reads; alternatively attention chunk-prefill over a 32-row padded chunk at offset P (validated
  path: "chunk prefill start=128 T=100" PCC 0.99996) if its cost at 32 rows is acceptable.
- 09:06 acceptance cross-check with the head prefill and the eager all-logits prefill: device first drafts equal the torch
  reference head's on 6 / 8 steps (' like', ' the', ' one', ' that', ' now', '.'; two near-ties differ), hidden RMS ~2.2;
  vs the model's own continuation (prose, 8 steps, k = 2): first-draft agreement 0.50, mean accepted 0.75 -> 1.75 tokens
  per pass. The head is functional end to end; full measurement (4 prompt kinds, k = 4, 64 tokens) queued.
- 09:16 MTP acceptance, full model, prose, k = 4, 64 steps: mean accepted 1.89 -> 2.89 tokens per pass, first-draft
  agreement 0.78, histogram [14, 15, 13, 8, 14] (14 / 64 passes accept all four). With a verify pass at ~1.7x a decode
  step this is ~1.7x single-user throughput; code / json / chat running.
- 09:26 acceptance, code prompt, k = 4: mean accepted 3.86 -> 4.86 tokens per pass, first-draft agreement 1.00, histogram
  [0, 1, 1, 4, 58] (k = 5 or 6 would pay on code).
- 09:36 acceptance, json prompt, k = 4: mean accepted 2.98 -> 3.98 tokens per pass, first-draft agreement 0.84,
  histogram [10, 4, 4, 5, 41].
- 09:46 acceptance, chat prompt, k = 4: mean accepted 2.03 -> 3.03 tokens per pass, first-draft agreement 0.84, histogram
  [10, 14, 17, 10, 13]. Summary (k = 4, 64 greedy steps each, `doc/perf_pass/mtp/acceptance_k4_*.json`): prose 2.89,
  code 4.86, json 3.98, chat 3.03 tokens per pass (Qwen3.8 bundle at its k = 4: chat 2.94, code 4.45, json 4.75,
  prose 2.09). Break-even for a verify pass costing c x a decode step is c < tokens/pass; target c ~ 1.5-1.7.
- 09:48 `test_attention_rows_mode[r5]` PASSES with `rows_kv_write` (min PCC 0.9954 over the 5 rows written in one call):
  the attention rows mode of the verify pass is validated at prefix 0; prefix-40 variants and the GDN cost test rerun.
- 09:49 `test_gdn_rows_cost` (GDN layer 0, real weights, eager host wall with sync, 10 iters): prefill chunk T = 32 6.48 ms,
  T = 128 9.39 ms, decode B = 1 2.11 ms -> the GDN prefill form on a 32-row chunk costs ~3x a GDN decode step per layer;
  over 36 layers a verify pass lands near ~1.8 decode steps (prose 1.6x, code 2.7x at the measured acceptance) ->
  build the rows mode with the prefill form first, fuse later.
- 09:51 rows-mode attention with a 40-position prefix: PCC 0.94-0.97 -> the TEST's prefix steps (one real row + 7
  dummies per call, batched write) raced too; per-row write for every call in the test; rerun queued. `tt/mtp_rows.py`
  (`layer_forward_rows`: attention rows via decode + rows_kv_write, GDN rows via the prefill form on a 128-row padded
  block with valid_len = R and the conv carry rebuilt from the taps, decode-mode ffn norm + pairs MoE) written;
  `test_gdn_rows_mode` (256-token prefill, 3 decode steps, then R rows) queued.
- 09:53 `test_gdn_rows_mode`: GDN layer 0 after a 256-token prefill + 3 decode steps, R rows in ONE rows-mode call
  (prefill form, valid_len = R, conv carry rebuilt from the taps) vs the HF causal prefill: R = 5 min PCC 0.99989,
  R = 8 0.99859. Both layer kinds now have a validated rows mode; `tt/mtp_verify.py` (whole-model verify pass over R
  rows with the GDN scratch bound + slot copy, host greedy accept) written; equivalence test vs sequential decode queued.
- 09:55 `test_attention_rows_mode` all variants pass with the per-row K/V write: R = 5 prefix 0 / 40 and R = 8 prefix 40
  min PCC 0.9954 / 0.9962 / 0.9962 (the shipped-policy layer figure).
- 09:57 `test_verify_pass_matches_sequential_decode` (2-layer generator): the whole-model verify pass over 5 rows vs the
  same 5 tokens decoded one per step: per-row logits PCC 0.9979-0.9995, argmax 4 / 5 (a near-tie on the 2-layer model's
  near-uniform logits). v1 MTP loop (`mtp_generate_eager`: draft k, verify pass, host greedy accept, exact GDN commit by
  re-running the accepted prefix, head KV over the accepted rows) written; reduced-model mechanics test and the
  full-model greedy-equivalence tool (`tools/mtp_generate_full.py`) queued.
- 10:08 v1 loop on the reduced generator: `TT_FATAL input_tensor.is_allocated()` in `copy_slot_into_scratch` (the batched
  rec_state reported deallocated at bind time). A probe (`scripts/14-gdn-alloc-probe.py`: build, head, greedy, reset,
  all-logits prefill, bind / unbind) shows every GDN buffer allocated at every step, so the failure was specific to the
  scratch-bind + slot-copy path; the loop now runs the pass IN PLACE on the B = 1 generator's bound state (the pass
  advances it; a device snapshot of rec + taps is taken before the pass and restored for the commit). Verify test passes
  again (PCC 0.9979-0.9995).
- 10:20 first full-model loop (`tools/mtp_generate_full.py`, prose, k = 4, N = 48): drafts `[0, 0, 0, 0]` on every pass,
  MTP text diverges from greedy at token 3. Diagnostics (`head draft dbg`): the head's input hidden is sane (rms 2.24)
  but the head's logits are ALL ZERO (max 0.000, argmax 0) and its hidden output is zero -> the head's weights / KV were
  overwritten. Cause: the tool builds the head AFTER `build_generator` captured the prefill bucket traces, then runs the
  greedy baseline through the traced prefill; a replay's baked intermediates alias the head's later allocations
  (the #48536 rule). Fix: the MTP tools build the generator with `warm_prefill_trace=False`; the generator gets an
  `mtp_head=True` option that builds the head and its KV before any warm-up / capture (the serving path).
- 10:56 k = 0 loop (pass of one row per step, no drafts): also diverges at token 3 -> the pass itself, not the head.
  `tools/mtp_pass_check.py` (no traces, no head: eager prefill, 15 eager decode steps with a state snapshot before each,
  then the verify pass from every snapshot over R greedy tokens vs the decode logits of the same steps), prose:
  R = 1: 9 / 10 argmax, PCC min 0.99775 mean 0.99905; R = 2: 18 / 20, 0.99701 / 0.99882; R = 5: 47 / 50, 0.99658 /
  0.99865. EVERY disagreement is the same token: decode ' the' vs pass ' this' at a 0.125 logit margin (bf16 near-tie),
  reproduced identically as row 0, 1 or 2 of a pass -> the verify pass is exact up to bf16 numerics on the full model;
  the token-3 divergence of the loops is that one near-tie.
- 11:40 full-model loop without prefill traces (prose, k = 4, N = 48, v1 commit by re-running the accepted prefix):
  23 passes for 48 tokens = 2.09 tokens per pass (acceptance histogram 0:10 1:7 2:1 3:2 4:3), greedy-equal up to token 8
  (then a near-tie divergence, text stays coherent). Eager host walls: greedy 5.9 s, MTP 8.6 s (untraced; the traced
  pass is the next step). GDN-only commit variant and code / json / chat prompts queued.
- 12:02 GDN-only commit (restore snapshot, replay the 36 GDN layers on the saved 128-row inputs, valid_len = a + 1) gives
  the IDENTICAL 48 tokens and passes as the v1 re-run commit on prose (2.09 tokens per pass); eager MTP wall 7.0 s vs
  8.6 s (v1) vs 5.8 s plain eager greedy. Code prompt: 10 passes for 48 tokens = 4.80 tokens per pass, all 48 tokens equal
  to greedy, eager wall 2.8 s vs 6.0 s greedy (already 2.1x faster untraced).
- 12:24 json prompt (GDN-only commit): 13 passes for 48 tokens = 3.69 tokens per pass (histogram 0:2 1:2 2:1 4:8), 47 / 48
  tokens equal to greedy (one near-tie at token 32), eager MTP wall 3.8 s vs 6.1 s greedy.
- 12:34 chat prompt (GDN-only commit): 15 passes for 48 tokens = 3.20 tokens per pass (histogram 0:1 1:5 2:3 3:1 4:5),
  48 / 48 tokens equal to greedy, eager MTP wall 4.4 s vs 5.9 s greedy. Summary of the v1 loop on the full model at k = 4:
  prose 2.09 / code 4.80 / json 3.69 / chat 3.20 tokens per pass (offline acceptance predicted 2.89 / 4.86 / 3.98 / 3.03).
- 12:35 traced step smoke (2-layer model): TT_FATAL "Cannot load new binaries during trace capture" at the first
  `ttnn.copy` into a persistent buffer - the warm run had adopted its outputs as the buffers, so the copy programs were
  first compiled inside the capture. Warm now runs the pass twice (second run compiles the copies). Rerun queued.
- 12:57 traced step smoke (2-layer model, k = 4, N = 24): captures and replays; program walls pass 13.5 ms, commit 0.83,
  draft 3.69, head rows 1.94 (host wall incl. read-back).
- 13:07 traced step on the FULL model (prose, k = 4, N = 64): 27 passes = 2.37 tokens per pass (hist 0:9 1:8 2:3 3:3 4:4);
  loop 132.9 ms per pass -> 17.8 tok/s, i.e. SLOWER than plain traced decode (30.6 tok/s). Program walls: pass 90.7 ms
  (2.8 decode steps; the estimate was 1.9), commit 28.6 ms (36 GDN prefill-form replays, ~0.8 ms each), draft 3.76 ms x 4,
  head rows 1.94 ms. Break-even needs pass + commit + drafts < 2.37 x 32.7 ms = 77 ms; today 134 ms. Next: per-op Tracy
  profile of the pass on the 2-layer model (the pass is op-count bound like decode: the GDN prefill form and 8 per-row
  K/V writes per attention layer add ops), then cut ops (single R-row K/V write, GDN glue), fewer GDN chunk rows.
- 14:31 Tracy of the traced programs (2-layer model, 5 replays, per-op device time, device 0), BASELINE (per-row slice /
  reshard K/V writes, 128-row GDN block, tile argmax): pass 291 ops / 12.9 ms device, of which `ArgMaxDeviceOperation` on
  the [8, 62080] TILE logits = 9.25 ms (single-core for TILE inputs; ROW_MAJOR is multi-core) and the draft step's argmax
  1.98 ms x 4. The two layers + LM head are ~3.3 ms: MoE on 8 rows (sparse gate_up 222 us + down 162 us per layer), LM head
  385 us, tilize-with-padding 13 ops 351 us, binary 26 ops 264 us, slice 71 ops 131 us, GDN prefill form (qkvzab matmul 94,
  out-proj RS 85, prep 68 + scan 65 us) is small. Commit (1 GDN layer): 75 ops / 0.72 ms. Fixes: argmax over a ROW_MAJOR
  view (multi-core), the masked K/V writes (already in), fewer glue ops next.
- 14:34 Tracy with the masked K/V writes (`rows_pos_masks`: one paged update per row on the full sharded K/V, users at -1
  skipped) and the ROW_MAJOR multi-core argmax: pass 252 ops / 3.73 ms device (was 291 / 12.9), argmax 118 us (was 9249),
  PagedUpdateCache 16 ops 50 us, slices 47 ops 100 us (was 71 / 131). Host wall per pass 4.35 ms (was 13.8) on the 2-layer
  model. Remaining pass time is the B = 8 decode-pairs MoE (sparse gate_up 223 + down 161 us per layer, tilize-with-padding
  13 ops 351 us, binary 26 ops 264 us) + LM head 384 + max 244 us: a rows pass costs about a B = 8 decode step (served
  67.7 ms) plus the GDN prefill form. Draft step 416 ops / 5.9 ms per 4 steps (1.5 ms each: head layer + LM head 384 us);
  head rows 170 ops / 1.9 ms. Commit (whole GDN layer replay) 75 ops / 0.72 ms per layer -> replaced by `commit_scan`
  (conv FIR + chunk scan on the pass's saved q/k/v/beta/g and pre-conv qkv; expected ~0.2 ms per layer).
- 14:44 scan-only commit (`TPGatedDeltaNet.commit_scan`: conv FIR for the taps + chunk scan on the pass's saved
  q/k/v/beta/g and pre-conv qkv, from the restored snapshot) reproduces the whole-layer re-run commit token for token on
  the 2-layer generator (forced-greedy drafts, 16 / 16). Tracy: commit 58 ops / 0.54 ms per GDN layer (was 75 / 0.72);
  what remains is the fused-chunk adapter's relayouts (untilize 10 ops 118 us, tilize 3 ops 82 us) around prep 68 +
  scan 65 us.
- 15:02 full-model traced step v3 (masked K/V writes, multi-core argmax + gather, scan-only commit; prose k = 4, N = 64):
  2.37 tokens per pass; per pass 107 ms = pass 81.0 (was 90.7) + commit 22.4 (was 28.6; 23 of 27 passes) + drafts 4 x 1.45
  (was 3.76) + head rows 1.95 -> 21.8 tok/s (plain traced decode 30.6). The pass is ~1.5 ms per layer, i.e. a B = 8 decode
  step (served 67.7 ms): the decode-pairs MoE at 8 rows (64 padded pairs) is the cost. Levers queued: rows width = the
  smallest decode width holding k + 1 rows (k = 3 -> 4 rows: half the pairs), the expert-sorted MoE path for the rows
  (dispatch kernel + skip-padding grouped matmuls + gather-add: ~5 ops instead of ~18 per layer).
- 15:06 Tracy with a 32-row GDN block (`QWEN35_MTP_GDN_ROWS=32`; the fused chunk kernel's native chunk): pass device
  3.18 ms (128-row block: 3.49) -> ~0.3 ms less per GDN layer (~11 ms per full-model pass); scan-only commit 0.295 ms
  per layer (was 0.542: prep 46 + scan 18 us, relayouts halved) -> ~11 ms instead of 20 for the 36 layers. Correctness
  test (verify pass vs sequential decode) and full runs with the 32-row block queued; k = 3 (4-row width) runs queued.
- 15:19 expert-sorted MoE for the rows (`QWEN35_MTP_MOE_ROWS=sorted`): correct (verify PCC 0.99791) but SLOWER: pass
  5.19 ms vs 3.49 on the 2-layer model. The sparse matmuls do get cheaper (341 + 257 us vs 446 + 322 per 2 layers) and the
  dispatch / gather-add kernels are small (158 + 93 us), but the 32-row expert groups pad the 64 pairs to ~2048 rows: the
  row gather (embedding, 1215 us) and the untilize before the un-dispatch (492 us) dominate. Rejected for rows mode; the
  knob stays (default pairs).
- 15:10 the 32-row GDN block was WRONG in the verify test (PCC 0.22): `forward_prefill` takes a K-sharded input only when
  T > TILE (fused norm-AG + in-proj matmul); at T = 32 it expects a replicated input, but the rows mode fed it the
  prefill-form distributed norm output (post-AG skipped). Fix: at a 32-row block the attention norm runs in decode form
  (gather-then-norm, replicated). Test + full runs (prose, code, k = 4) requeued behind the k = 3 runs.
- 15:33 k = 3 (4-row width, 128-row GDN block), prose: 2.29 tokens per pass (k = 4: 2.37); pass 63.7 ms (k = 4, 8 rows:
  81.0), commit 22.4, drafts 3 x 1.43, rows 1.54 -> 87 ms per pass = 26.2 tok/s (k = 4: 21.8; plain decode 30.6). Half the
  pairs (32 vs 64) took 17 ms off the pass; the 32-row GDN block should take ~10 ms off the pass and ~10 off the commit.
- 15:42 k = 3 code: 16 passes for 64 tokens = 4.00 tokens per pass (15 of 16 passes accepted all 3 drafts), 64 / 64 equal
  to greedy, 71.9 ms per pass -> 55.6 tok/s = 1.8x plain traced decode (30.6). Only one commit in 16 passes (all
  accepted -> no commit), so the pass (63.7 ms) is the whole cost there.
- 15:43 the 32-row GDN block still fails (PCC 0.22, every row's argmax identical) with the decode-form norm -> the
  failure is inside forward_prefill at T = 32 (candidates: out-proj matmul_reduce_scatter at M = 32, fused-out relayout,
  one-chunk fused scan); bisecting probe (`scripts/14-gdn32-probe.py`) queued.
- 15:58 probe (`scripts/14-gdn32-probe.py`, GDN layer 0 of the 2-layer generator): `forward_prefill` on a 32-row block
  with a REPLICATED input matches the 128-row block at PCC 1.00000 (output and state; also with mmrs / fused-out off), so
  the layer path is fine at T = 32 (this generator has `_fuse_agmm` off: replicated input at every T; a K-sharded input
  fails at T = 32 as expected). The rows mode passed `chunk_size = 32` to forward_prefill (the probe passed 128): the
  fused adapter's cached chunk masks are the model's 128 chunk -> keep `args.gdn_chunk_size`. Test requeued.
- 16:50 probe 2 (rows-mode GDN layer 0 at 32 vs 128 rows, same input): decode-form norm vs prefill-form norm PCC 0.99999,
  GDN output PCC 1.00000 / state 0.99997 -> both pieces are right; the whole rows call then died with
  `input_tensor_a.is_allocated()` at the residual add. Root cause of the PCC 0.22: a [1,1,8,H/TP] TILE tensor already
  spans one 32-row tile, so `ttnn.pad` to 32 rows ALIASES the input, and the decode-form norm frees its input -> the
  residual x itself was freed (the eager test read a freed buffer: deterministic garbage, identical PCCs across every
  variant). Fix: clone before padding at a 32-row block. Test + full runs requeued.
- 17:10 32-row GDN block verify test PASSES (PCC 0.99791, same as the 128-row block). Full model, prose, k = 3, 32-row
  block: pass 51.9 ms (128-row: 63.7), commit 14.2 (22.4), drafts 3 x 1.41, head rows 1.53 -> 71.5 ms per pass; this run's
  text took the other side of the token-3 near-tie and accepted 2.00 tokens per pass -> 28.0 tok/s (at the earlier 2.29 it
  is 32 tok/s: prose is break-even with plain decode at 30.6). Code at k = 3 accepts 4.00 tokens per pass with almost no
  commits -> ~57 ms per pass -> ~70 tok/s (2.3x). Measurement stops here (per the user: no further full runs unless a
  decision needs one); defaults set: 32-row GDN block, k = 3, masked K/V writes, gather argmax, scan-only commit. Known
  remaining lever: overlap the commit (14 ms) with the next pass's drafts (4 ms) on a second command queue (~6 %).
- 17:50 serving hook written. `tt/mtp_plugin_shim.py` (host-only, 20 tests pass; the plugin's own 76 runner / scheduler
  tests pass with the shim not applied): variable-width AR output through the vllm-tt-plugin - `_coerce_output_block`
  and `_get_output_tokens` accept 1..k+1 ids per row (-1 trailing pads), `_apply_sampled_tokens_to_state` appends the
  non-pad ids and advances num_tokens by that many (clips at max_model_len), `_build_runner_output` emits ragged lists,
  `TTScheduler._update_request_with_output` uses upstream's per-token check_stop (EOS / stop / max_tokens trim inside a
  row) and consumes one placeholder per frame, `num_lookahead_tokens = k` so the KV slots cover the verify window,
  `_is_still_prefilling` compares the gap against k+1. `tt/mtp_serving.py` + generator_vllm `QWEN35_MTP=k`: single-user
  (--max-num-seqs 1); head built at model construction, its KV with the model's; eager masked-bucket prefill (no bucket
  traces) with the `_run_bucket_hidden` hook feeding the head's prompt rows in 32-row groups (rows selected by a one-hot
  matmul: fixed programs), the four MTP traces captured in the trace phase instead of the decode / bucket traces;
  decode_forward -> `MtpServing.step` -> host int32 [1, k+1] row. Reduced-model serve test queued.
- 18:54 MTP SERVED end to end on the reduced 2-layer model through vLLM (`scripts/14-serve-mtp.sh`: QWEN35_MTP=3,
  --max-num-seqs 1): two 64-token greedy completions (24 tok/s incl. TTFT on the 2-layer model, ~1.2 tokens per pass
  on its garbage text). Four attempts fixed on the way: 23 programs compiling at the first request (the eager served
  prefill with the head hook and the sampler variants are now warmed before the captures), the plugin's warm-up dummy
  decode reaching the step before any prefill / capture (eager fallback + initialised state), the generator's own eager
  warm-up decode routed into the step (bypass flag), and the plugin's read / host-format hooks receiving the host
  [1, k+1] row (pass-through overrides). The plugin's steady async path hands a host token that can lag the device: the
  model owns the continuation and only checks the plugin's token when `reload_inputs` / `reset_batch` is set.
  Full-model served run (prose + code, 128 tokens) launched.
- 19:12 first full-model MTP server: out of DRAM allocating the head's paged K/V (two 1017 MB buffers) after vLLM had
  taken the pool sized for 12 KV layers (`get_max_tokens_all_users`). MTP mode now budgets the head as a 13th KV layer
  (kv bytes per token x 13/12) plus 1.5 GiB for its weights, the step buffers and slack; the adapter logs the head K/V
  shape / dtype at allocation. Relaunched.
- 19:34 MTP SERVED on the FULL model through vLLM (QWEN35_MTP=3, --max-num-seqs 1): prose 128 tokens, 55 passes, 2.36 tokens
  per pass, 70.0 ms per pass inside the step (33.7 tok/s of device steps); code 3.94 tokens per pass, 59.7 ms per pass
  (66 tok/s); text coherent and equal to the standalone tool's. The client-side numbers of that run were polluted by the
  request builder (a ttnn import inside the timed window); a properly timed run with the readiness benchmark
  (streaming TTFT / TPOT, concurrency 1) queued. KV pool with the head budgeted: 1.66 M tokens (was 1.82 M).
- 20:07 SERVED MTP, full model, k = 3, single user (timed requests + the readiness streaming benchmark, c = 1, 4 requests
  of 128 random prompt tokens / 128 out): prose 128 tokens in 4.10 s incl. TTFT = 31.2 tok/s (plain serving 29.4 at
  32.0 ms TPOT + 0.26 s TTFT: 1.06x); code 2.28 s = 56.2 tok/s (1.9x); benchmark TPOT mean 19.4 ms (P99 24.9) = 51.5
  t/s/u vs 31.3 plain (1.65x), ITL p50 60.1 ms (one pass emits ~3 tokens at once), TTFT p50 329 ms (plain 258-274: the
  eager bucket prefill + the head's prompt rows). Commit fee950b4098. MTP turn closed for single user; next: multi-user.
- 20:42 MULTI-USER turn, block 1: `tt/mtp_multi.py::verify_pass_multi` - B users x R rows user-major in one decode-layout
  block; attention rows as the batched decode step with per-row positions / page tables / K/V masks (already row-keyed),
  GDN rows through `forward_prefill_batched` (users = independent sequences, state written in place; conv left-context
  rebuilt from the taps; the per-user [B, T, dim] block built by a one-hot placement matmul and gathered back by its
  transpose - `ttnn.pad` on a [B, R, dim] tensor hits "FillPad N_slices > 1 not supported", and the decode-form norm's
  L1 width-sharded output has to be interleaved before the matmul). Test (2-layer generator, B = 2, R = 4) vs the batched
  sequential decode of both users: PCC 0.99903-0.99943, argmax 8 / 8.
- 21:02 MULTI-USER block 2: eager loop `mtp_generate_multi` (batched head drafts over B users, one batched verify pass,
  per-user accept, ONE batched commit = restore + batched replay with per-row accepted lengths, batched head K/V rows
  padded to a served width with -1 masks, next-draft hidden per user by a one-hot selection). Test (2-layer generator,
  B = 2, k = 3, N = 12, forced-greedy drafts): user 0 equal to greedy 12 / 12, user 1 9 / 12 (a repeated token ended one
  step early: the 2-layer model's near-tie), 5 passes for 24 tokens.
- 21:11 MULTI-USER block 3: `RowsLen` (device-resident per-user row counts: [B,T,1] masks + [B,K-1,T+K-1] conv one-hots,
  staged by host copies) accepted by the fused chunk adapter (B > 1 device masks) and the conv FIR (already batched):
  the batched verify pass with RowsLen matches the host-length path exactly (PCC 0.99903, 8 / 8) - the traced multi-user
  pass and commit can now stage their lengths without host allocations.
- 21:52 MULTI-USER block 4: `tt/mtp_traced_multi.py::MtpTracedMulti` - the traced step over ALL B slots: fixed-width
  pass (B x R rows, RowsLen lengths, persistent saved blocks, batched snapshot), batched commit (restore + one batched
  replay with the accepted lengths), head drafts over the B slots with masked per-row K/V writes (a dead slot's stale
  page-table row must never be written), head rows for the accepted tokens (hidden gathered from the pass block by a
  one-hot), next-draft hidden selection, per-slot host state with slot remap. The batched GDN form's shifted-out zero
  tap was a per-call `from_torch` (a host write inside the capture: TT_FATAL !trace_id_ and a hung process) -> it now
  uses the persistent zero source. Reduced model, 2 users, k = 3, N = 16: traced tokens equal to the eager multi loop
  32 / 32, same 15 passes; 9.1 ms per pass (pass 3.8, commit 0.9, 3 drafts x 1.47). Full-model runs (2 and 4 users) queued.
- 22:10 MULTI-USER block 5 (code, device test queued): `MtpServing` generalised to B = --max-num-seqs slots on top of
  `MtpTracedMulti` (B == 1 keeps the single-user runner): queued per-request prefills (the served multi-slot prefill runs
  them in order; the hook pops the next request at chunk_start 0 and keeps that slot's page-table row and first-token
  hidden), `end_prefill` hands every slot to the runner, `step_multi` maps the plugin's [B_pad] positions / page tables
  / slot_remap onto the slots (a slot the plugin considers live without a prefill here = its warm-up dummy step: run
  as a phantom slot), the adapter emits the per-slot rows. Serve script: SEQS (max-num-seqs), CONC=1 fires prose + code
  together.
- 22:30 FULL-MODEL multi-user traced step (k = 3, N = 32 per user, real head drafts; traced == eager token for token):
    2 users (8-row pass):  pass 82.0 ms, commit 31.6 (12 / 15 passes), drafts 3 x 1.56, rows 1.9 -> 116 ms per pass,
                           4.27 tokens per pass -> 36.8 tok/s aggregate; plain 2-user decode 38.1 ms/step = 52.5 tok/s (0.70x)
    4 users (16-row pass): pass 143.6 ms, commit 49.5 (16 / 16), drafts 3 x 1.72, rows 2.7 -> 203 ms per pass,
                           8.00 tokens per pass -> 39.4 tok/s; plain 4-user decode 46.7 ms/step = 85.6 tok/s (0.46x)
  Reading: the pass costs like a served decode step of B x 4 rows (8 rows ~ 67.7 ms + GDN form + K/V rows; 16 rows ~
  107 ms + ...), and the batched whole-layer commit adds 30-50 ms, while prose-like acceptance (~2.1 tokens per user per
  pass) only buys ~2.1 plain steps of width B (76 / 93 ms). The weight-amortisation argument holds only where the
  decode step is bandwidth-bound and nearly flat in rows: 8 users -> 32-row pass (~125 ms + commit) vs 2.1 x 67.7 =
  142 ms: break-even at best for prose, ~1.6x for code (4 tokens per pass). Multi-user MTP pays for high-acceptance
  workloads (code / structured output) at <= 8 users and loses elsewhere until the 8-32-row decode MoE gets cheaper
  (the pairs / masked glue that also caps the served 4-16-user decode) and the commit is scan-only (~1/3 of today's).
- 22:41 MULTI-SLOT SERVING works on the reduced model (QWEN35_MTP=3, --max-num-seqs 2): sequential requests reproduce the
  single-slot texts, and a concurrent prose + code pair (both slots live: two prefills into slots 0 / 1, then 32 batched
  passes) returns the same two texts as the sequential requests. The warm-up path needed the GDN prefill scratch bound
  around its eager prefills (a batched model's [1, ...] state copy into the [B, ...] buffers failed otherwise). The
  serve script's concurrent block waited on the server process (a plain `wait`) - fixed to wait on the two curls.
  Multi-user MTP therefore serves end to end, but stays opt-in (the full-model measurements above).
- 01:02 (turn 6, decode MoE at 8-32 rows) Tracy per layer at width 8 (reduced model, traced batched step): GDN layer
  884 us device / 1.354 ms step (60 ops: ~35 % op-launch gaps), attention layer 1099 us / 1.226 ms. MoE share at B = 8:
  sparse gate_up 189-223 + down 119-161 us (~35 %, ~64 distinct experts = 75 MB of bfp4 weights per layer: near the
  bandwidth floor), pairs glue: tilize-with-padding 82-140 us (x_pairs + scores + acc), binary 15 ops ~155, reduce 37,
  slices / reshapes ~55. FOUND: the ROUTER matmul [32 x 3072] x [3072 x 256] runs the stock linear on 7-8 cores at
  HiFi4 bf16 and costs 71 us per layer at width 8 (also 71 us in the MTP rows pass) - 3.4 ms of a 48-layer step if
  every layer pays it. Fix: DRAM-sharded decode matmul for the router (like the other decode projections),
  `QWEN35_ROUTER_DRAM_SHARDED=1` default; numerics (batched decode B = 32 vs HF) + width-8 Tracy queued.
  Width 16 (attention layer, 1856 us device): the MASKED path's dense E = 256 dimension is the waste - sparse gate_up
  641 + down 318 us (~110 active experts, bandwidth-bound) but swiglu / weights-multiply / fast-reduce run over all
  256 x 32 rows (unary 327 us, reduce 138 us): the compact (sorted) layout with cheap gather / un-dispatch kernels is
  the structural fix (same work as the prefill levers "row gather kernel" + "untilize before un-dispatch"); parked.
- 01:06 router as a DRAM-sharded decode matmul: batched decode B = 32 vs HF unchanged (per-user PCC min 0.99988 attention
  layer, GDN layer passes), width-8 attention-layer router 71 -> 25 us, layer step 1.226 -> 1.186 ms (traced). Kept on
  by default (`QWEN35_ROUTER_DRAM_SHARDED=1`). Item 1 closes here: the remaining 8-32-row MoE cost is the weight-bound
  sparse matmuls plus the dense-E masked-path glue, whose fix is the compact layout with gather / un-dispatch kernels
  (parked with the prefill levers). Moving to item 2 (chunked prefill): generator `prefill_serving(start_pos, final)`
  with per-request carried GDN state keyed by the first KV block, adapter + plugin shim for the intermediate mask,
  `QWEN35_CHUNKED_PREFILL=1` opt-in; equivalence test (4 chunking patterns) queued.
- 01:11 (turn 7) chunked serving prefill equivalence on the 2-layer generator (2 slots, bound-scratch path): four
  chunking patterns of a 2500-token prompt ([2048, 2500], [1500, 2500], [2100, 2500], [700, 2200, 2500]) give logits
  PCC 1.000000 and a decode-slot GDN state (rec + conv taps) PCC 1.000000 vs the whole-prompt prefill; intermediate
  chunks run only whole 2048-token segments from the request's own processed end and return zero logits; the carried
  state is keyed by the request's first KV block. Next: the same with another request's prefill interleaved between the
  chunks (the conv left-context is rebuilt from the taps on restore), then the plugin's needle test on the full model.
- 01:26 chunked prefill with ANOTHER request's whole prefill interleaved between the chunks (the scratch reused in
  between): all four patterns still PCC 1.000000 for logits and the slot state - the carried state (rec + taps, conv
  carry rebuilt on restore) is correct under interleaving. Full-model served run with the plugin's needle test queued
  (`scripts/15-serve-chunked.sh`: --max-num-batched-tokens 2048, 4 concurrent ~1-budget prompts with passphrases).
- 01:38 served chunked prefill, full model, budget 2048: the plugin split a 3105-token prompt into [0, 2048) (intermediate,
  mask delivered by the shim) + [2048, 3105) (final) and the model ran both; the program guard then fired on the
  continuation chunk (1 program compiled after the captures: the carried-state restore / conv-carry rebuild ops are not
  in the warm-up) and the engine died, so the needle test only saw connection errors. Fix: warm a dummy two-chunk
  prefill in `_warm_prefill` when chunked prefill is on.
- 01:52 SERVED chunked prefill VALIDATED on the full model: `--max-num-batched-tokens 2048 --enable-chunked-prefill`,
  4 concurrent ~2048-token prompts with embedded passphrases (the plugin's tests/tt/test_chunked_prefill.py): 3 / 3
  passed; the server log shows ragged chunk starts (949, 985, 2004, 2997, 3033) and steps mixing a final and an
  intermediate row, all handled from each request's own processed end; the program guard stayed quiet after the
  continuation path was warmed at start-up. Kept opt-in (`QWEN35_CHUNKED_PREFILL=1`): the plugin never runs a decode
  in a prefill step, so chunking bounds step length and lets prompts share steps rather than cutting the running
  users' gap; default-on needs a served TTFT A/B on long prompts (per-chunk overhead ~ a bucket call + a state clone).
- 02:24 (MTP follow-up 1) exact speculative sampling in the single-user served step: drafts sampled from the head's
  request-processed distribution (temperature / top-k / top-p, seeded RNG per request), the verify rows read back as
  ROW_MAJOR logits (unpadded: 0.5 MB per row per mesh; the TILE tensor would have moved its 32-row padding, ~16 MB per
  read), accept by rejection sampling with the residual draw (`tt/mtp_sampling.py`; host Monte Carlo test: first-token
  KL(target || empirical) 0.001 over 40 000 trials). Temperature 0 keeps the argmax path. Penalties are not applied in
  the step (documented). Reduced-model served run with temperature 0.7: works end to end; the first version spent
  ~60 ms per pass in padded read-backs + host softmaxes (72.5 ms per pass on the 2-layer model) -> unpadded rows, lazy
  target rows, inverse-CDF draws: host sampler 13.5 -> 1.7 ms per pass. Full-model check (T = 0 unchanged, T = 0.7) next.
- 02:48 (MTP follow-up 2) the head's prompt rows now run INSIDE the prefill bucket program (`MtpServing._bucket_head`,
  hooked into `_run_bucket_dev` after the layers; host staging of the first-draft one-hot and the zeroed previous hidden
  per segment via `_mtp_stage_hook`): one extra layer per segment, row j = (token at cs + j, hidden of cs + j - 1) at the
  bucket's own positions / page tables (the hidden rows shifted down by one through a ROW_MAJOR concat with the previous
  segment's last hidden carried in `mtp_prev`), the prompt's last hidden picked by a one-hot into `hid_out`. With that,
  the prefill bucket TRACES are captured again in MTP mode (the eager-bucket path cost ~60 ms of TTFT). Reduced model:
  server starts (program guard quiet with the head inside the traces), 32-token requests 0.29 s (eager-bucket path:
  0.38 s). `QWEN35_MTP_TRACED_PREFILL=0` keeps the eager path. A full-model run with the earlier edits half-imported was
  invalid (queue not staged) - rerun queued with T = 0 / 0.7 and the c = 1 benchmark.
- 03:03 (MTP follow-up 2, full model) SERVED with the head's prompt rows inside the traced prefill bucket: temperature 0
  texts identical to the earlier served greedy MTP (prose " a storm like this. The waves crashed against the rocks...",
  code the merge loop); temperature 0.7 coherent (prose 22.1 / code 37.6 tok/s incl. TTFT; greedy 29.5 / 52.9).
  Benchmark c = 1: TTFT p50 314 ms (eager-bucket MTP prefill 329; plain 258-274), TPOT 18.6 ms = 53.8 t/s/u (51.5),
  ITL p50 57.3. Per-request MTP_STATS: greedy 58-68 ms per pass, SAMPLED 93-100 ms per pass (same acceptance
  histograms) - the sampling path costs ~33 ms per pass on the full model, far more than the host sampler (probs 0.12,
  sample 0.27, accept 0.64 ms on 248 320-wide rows, timed on the host): the suspect is the four logits read-backs per
  pass (3 draft rows + the pass rows; to_layout already unpads, so 0.5 MB per row per mesh). Instrumented the runner
  (`timing["ids_read"]` / `timing["lg_read"]`) and the traced tool (`--temperature`: a sampled loop with the same step
  as serving) - reduced-model timing run queued. Default-on MTP wants the sampled path at the greedy cost.
- 03:07 (MTP follow-up 3) scan-only BATCHED commit for multi-user MTP: `forward_prefill_batched(scan_sink=...)` saves
  the pre-conv qkv rows and q/k/v/beta/g of the rows pass per GDN layer; `commit_scan_batched(saved, valid_lens)`
  re-runs only the conv FIR (taps) + the fused chunk scan from the restored snapshot and rewrites rec_state / taps /
  the batched conv carry; `commit_multi` uses it (the whole-layer batched replay is gone). Device test
  `test_multi_user_scan_only_commit_matches_shorter_pass` (2 layers, B = 2, accepted [2, 1] and [4, 3]): rec_state and
  all 4 conv taps PCC 1.000000, max |diff| 0 vs a full pass over only the accepted rows; forced-greedy loop still
  3.43 tokens per pass. Traced multi-user run (reduced model) queued to confirm the commit trace captures with it.
- 03:14 traced multi-user step with the scan-only batched commit (reduced model, B = 2, k = 3, N = 24): traced tokens
  equal to the eager loop 48 / 48, commit 0.59 ms per pass (one GDN layer; the full model's 36 layers put the commit at
  ~11 ms instead of the 30-50 ms whole-layer replay measured 2026-09-30). Multi-user MTP stays opt-in (its pass cost
  is the width-8 / 16 decode step); the lever is in and validated, re-measure served when the compact decode MoE lands.
- 03:14 sampled-path diagnosis (reduced model, `--temperature 0.7`, no top-k / top-p): per read-back 'ids_read' (argmax
  + max, 2 small tensors) 1.67 ms, 'lg_read' (the ROW_MAJOR logits rows, 0.5 MB per draft row / 2 MB per pass) 2.45 ms
  -> ~10 ms of reads per sampled pass on top of greedy. The served run's extra 33 ms came from the host sampler after
  all: vLLM applied the checkpoint's generation_config (temperature 0.6, top_k 20, top_p 0.95; the request only set
  the temperature), and SpecSampler.probs did the top-p by SORTING THE WHOLE 248K ROW (2.2 ms per row with top_k set,
  ~10 ms without) for 3 draft rows + 1-2 target rows per pass; my host timing had used top_p = 1.0.
  Fixes: (1) top-p over the top_k survivors only (top_k set: 0.24 ms per row), nucleus from the top 1024 / 16384 when
  only top_p is set (full sort only if they do not reach top_p); (2) device per-shard top-k CANDIDATES: separate
  'cands_pass' / 'cands_draft' traces (pad 62080 -> 65536, split in two 32768 halves, ttnn.topk k = 32 each, as the
  shared 1D sampler does) over the persistent TILE logits of the pass / draft traces, read back as 4 x 2 x 32 values +
  ids per row (a few KB) instead of the rows; a request with 0 < top_k <= 32 is exactly reproducible from them
  (`SpecSampler.probs_topk`, host tests vs the dense processing); greedy passes never run them. top_k > 32 or unset
  keeps the full-row read. Ties at the k-th value: exactly k survivors (as the device top-k; vLLM's host sampler keeps
  ties) - documented, any fixed processing is a valid target. Reduced-model run with top_k 20 / top_p 0.95 queued.
- 03:48 SERVED, full model, sampled path from device top-k candidates (requests at T = 0.7 with the checkpoint's
  top_k 20 / top_p 0.95 defaults): per-request MTP_STATS prose 77.2 ms per pass (was 100.1; greedy 67.9), code 65.6
  (was 93.2; greedy 58.8) - the sampled pass now costs +7-9 ms over greedy (two candidate traces + a 0.4 ms read per
  step) instead of +33. End to end: T = 0.7 prose 28.2 tok/s (was 22.1; greedy 29.5), code 51.6 (was 37.6; greedy
  52.5). Greedy texts unchanged; sampled texts coherent. Benchmark c = 1: TTFT p50 306 ms, TPOT 18.6 ms = 53.7 t/s/u,
  ITL p50 57.5. Commit 2210fe18163 (+ this log). MTP follow-ups closed; next: adaptive default-on (item 1 of the
  production list in mtp_plan.md).
- 04:46 (production list item 1) ADAPTIVE MTP built (`doc/perf_pass/adaptive_mtp_plan.md`): with `--max-num-seqs` > 1 the
  server runs the single-user MTP step whenever exactly one request is live and the plain batched traced decode
  otherwise. State handed over at every switch: the slot's GDN state copied into a dedicated persistent B = 1 "MTP
  scratch" (bound onto the GDN layers for the phase; `write_slot` back on leave), the head hidden of every slot's last
  token kept on device in `hid_prev` [1,1,B,H] (written INSIDE every plain decode step by a post-norm hook, read by
  `enter`, written by `leave`), the head's paged K/V rows of every plain step written by a pre-layer hook (mixer + norm +
  QKV + RoPE + paged update, `TPAttention.kv_only`), the first plain step after a phase forced host-authoritative with
  the model's own last token / position for that slot. Reduced model (2 layers, 4 slots): server up, phases entered
  and left across single / staggered (code admitted 0.12 s into prose: phase end 31 passes -> plain -> survivor phase
  10 passes) / concurrent requests, no program compiled after the captures, single-request texts identical to the
  single-user served MTP; the pair texts differ from the single ones only where the 2-layer model's bf16 near-ties
  flip across decode widths (the same alternates the earlier plain width-2 runs produced). One failed attempt first:
  `ttnn.to_memory_config` handed back a new handle on the LM head's input buffer and the hook freed it -> clone.
  Full-model run queued (8 slots: texts, c = 1 MTP benchmark, c = 8 plain + hooks benchmark vs 68.5 ms TPOT).
- 05:50 adaptive MTP, full model (8 slots, first run): single requests = the single-user MTP texts and numbers (c = 1
  TPOT 18.7 ms = 53.6 t/s/u; TTFT p50 376 ms vs 306 in the 1-slot server: the enter / leave state copies around every
  request), c = 8 TPOT 60.0 ms (plain 8-slot baseline run right after: 59.6 ms; c = 1 plain 29.9 ms / TTFT 261 ms).
  The seeded pair requests ran at full width 8 in both servers (DECODE_BUCKETS: 783 width-8 steps) and the adaptive
  server's plain width-8 texts differ from the plain server's (prose from token 4, code '<=' vs '<'), while on the
  reduced model the adaptive and plain pair texts are identical at full width -> the hooks are not the cause there;
  full-model diagnosis queued (hooks off vs plain8; the new deferred-leave code). Deferred leave: a prefill only
  SUSPENDS the phase (unbind, keep the scratch), the next decode drops it (request finished, no write-back), resumes
  it (still alone) or leaves (write-back). One failed attempt: my own end_prefill assertion still demanded "no
  phase" during a prefill (now: no BOUND phase).
- 06:22 deferred-leave runs (hooks off / on): the SECOND single request decoded the first request's continuation -
  vLLM reused the finished request's slot for the new one, the suspended phase saw "its" slot still live at the next
  decode and RESUMED the old scratch state for the new request. Fix: `end_prefill` drops a suspended phase whose slot
  is among the freshly prefilled slots; the decode-side resume also requires the host position to continue the phase
  (a lower position = a new request). The pair texts of those two runs are void (polluted). Run 3 queued.
- 06:36 ADAPTIVE MTP VALIDATED on the full model (run 3, 8 slots): single requests = the single-user MTP texts and
  speed (code 51.6 tok/s incl. TTFT, prose 29.1); the concurrent pair's texts IDENTICAL to the plain 8-slot server's
  pair (both at the seeded full width 8) -> the plain path under adaptive MTP is exact, the earlier divergence was the
  leave-at-prefill flow; the staggered pair is MTP-exact up to the admission and continues on the plain path (its text
  diverges from the MTP continuation there, as the plain width-8 numerics do). c = 1 benchmark: TPOT 18.7 ms = 53.6
  t/s/u (plain 29.9), TTFT p50 344 ms (plain 261; 1-slot MTP 306): enter costs ~10 ms (36 slot slices + copies +
  carry rebuilds), the deferred leave drops a finished request's phase for free (7 of 7 phases here), a write-back
  leave 16-30 ms. c = 8 (run 1): TPOT 60.0 ms with the head hooks vs 59.6 plain. Requests with penalties or logprobs
  stay on the plain decode (`_mtp_eligible`).
- 08:10 served crash chasing (run 4b, full model): the request AFTER a finished MTP request died with
  `NotImplementedError: chunked / cached-prefix prefill` - the plugin dispatched a mid-sequence "prefill" of a RUNNING
  request. Mechanism: the plugin's `_is_still_prefilling` treats `num_computed + output_width < num_tokens` as a
  chunked continuation; the shim widened `output_width` to k + 1, but under vLLM's async scheduling (on: the model
  declares supports_async_decode) two in-flight multi-token steps leave up to 2k + 1 emitted tokens outstanding ->
  misclassified -> start_pos > 0 -> our prefill refuses (chunked prefill off). Fix in the shim: with chunked prefill
  OFF the outstanding-width test is disabled (a cached non-resumed row is never prompt work; a preemption resume
  resets num_computed to 0 and trips the closure's first test), with it ON the k + 1 bound stays. 22 host tests.
  Reduced-model runs queued: old shim + async (reproduction), sync scheduling, fixed shim + async.
- 08:40 ROOT CAUSE of the logprobs crash (and a latent plain-mode bug since turn 7): `prefill_forward` derived the
  per-row "final chunk" flag, when the chunked-prefill shim's mask is absent, from "device sampling params present" -
  every HOST-sampled prefill (logprobs / min_p / structured outputs: the plugin sends no TTSamplingParams) was treated
  as an intermediate chunk (zero logits in plain mode; in MTP mode the "MTP with chunked prefill" NotImplementedError
  the shim debug finally pinned: the plugin's own mask was False, computed 0 / prompt 37 / tokens 37). Fix: without
  the mask every row is final (the plugin never chunks when chunked prefill is off). Async scheduling was NOT the
  cause (the sync run failed identically); the shim's outstanding-width change stays (correct by construction).
- 08:41 reduced model with the fix: logprobs (32 logprob entries, host path), repetition penalty (plain decode; the
  text changes as expected), min_p (host path) all served under MTP; the concurrent pair = the plain texts; single
  requests = MTP texts. Full-model run 5 queued (extra + staggered + concurrent + c = 1 benchmark).
- 08:55 ADAPTIVE MTP, full model run 5 (8 slots) - CLEAN: single requests = the single-user MTP texts (code 52.1 /
  prose 29.4 tok/s incl. TTFT); logprobs (96 entries), repetition penalty and min_p requests served through the plain
  / host paths (12-15 tok/s: host sampling, and seeded requests run at the full bucket width - the stage-10 rule);
  concurrent pair IDENTICAL to the plain 8-slot server's texts; staggered pair MTP-exact until the admission; c = 1
  benchmark TPOT 18.6 ms = 53.7 t/s/u, TTFT p50 313 ms (344 before the deferred leave; plain 261). `QWEN35_MTP` now
  defaults to 3 (commit 961fab593a9); production-config run (32 slots, 262144 context, 768 MiB trace region, c = 1 /
  8 / 32 benchmarks) queued as the default-on gate.
- 09:12 PRODUCTION-CONFIG GATE (32 slots, --max-model-len 262144, 768 MiB trace region, MTP default k = 3): server up
  with the head's KV at full context; single requests = MTP texts (code 47.6 / prose 27.6 tok/s incl. TTFT);
  logprobs / penalty / min_p served (7-8 tok/s: host sampling + the seeded-full-width rule at 32 slots - the stage-10
  rule, not MTP); c = 1 TPOT 20.1 ms = 49.6 t/s/u (18.6 at 8 slots: the step rewrites the 4096-block page tables
  every pass -> cache the row), TTFT p50 351 ms (plain 32-slot 258-274); c = 8 TPOT 62.6 ms (stage 13: 68.5), c = 32
  128.4 ms (130.1): the head hooks cost nothing measurable at 8-32 users. MTP default-on holds at production scale.
- 09:21 (production list item 5, decode MoE at 16-32 rows) traced per-layer step time on the reduced model
  (`scripts/16-pairs-width-probe.sh`, test_perf_traced_decode_batched): width 16 masked path attn 2.212 / GDN 2.288
  ms vs PAIRS path (QWEN35_MOE_PAIRS_MAX_TOKENS=32) 1.843 / 2.078 (-17 % / -9 %; ~12 ms of a 48-layer step at 16
  users); width 32 masked 2.597 / 2.726 vs pairs 3.171 / 3.592 (+22 % / +32 %: 256 one-row pair groups re-read shared
  experts and pay 256 group launches). Default `pairs_max_tokens` 8 -> 16 (pairs at widths <= 16, masked at 32).
  The structural compact layout for 32 rows stays parked (the 32-row dense glue is ~25 % of the attention-layer step
  there; the sparse matmuls at ~110-220 active experts are the bandwidth floor).
- 09:25 width-16 batched decode vs HF on the pairs path (test_mc_batched_decode_b32[gdn_b16 / attn_b16], 16 users
  prefilled one at a time then one batched step): GDN layer PCC 0.99957 (per-user min 0.99896), attention layer
  0.99990 (per-user min 0.99988) - the same band as the width-32 masked path. Item 5 quick win closed: pairs up to 16.
  The 32-row compact layout and the prefill MoE levers share one kernel design (below); next: item 6 (width-1 op
  count) from a width-1 Tracy profile, then item 7 (weights once per expert in the grouped sparse matmul).
- 09:33 (item 6, width-1 decode) Tracy per op at width 1 (reduced model, traced): GDN layer 0.669 ms per step, ~90 ops,
  DEVICE time ~530 us (80 %: not launch-bound as assumed) - sparse matmuls 98 us (8 experts through the gather path:
  12.6 MB of bfp4 at ~130 GB/s), binary glue 62 (15 ops), data-moving reshapes 41 (10: the [B,1,Nv,Dk] <-> [B,Nv,1,Dk]
  head relayouts around the recurrence), router top-k 40 (single-core below a 1024-wide input), slices 18, qkv matmul
  46, reduce-scatter 32, norms 23; attention layer 0.598 ms / 85 ops / 505 us (sparse 99, qkv 42, top-k 40, binary
  38, 15 slices 21, FillPad 14). Variants probed at width 1: the pairs MoE path is SLOWER (GDN 0.876 vs 0.634 ms:
  the gather path stays for <= 4 rows); padded multi-core top-k pending. The head relayouts and the ~60 tiny ops are
  the fused-GDN-kernel territory (parked); the actionable width-1 items are the top-k and the 1-token sparse matmul.
- 09:40 (item 7, prefill MoE lever "weights once per expert") implemented in the grouped sparse matmul factory:
  RESIDENT in1 - cb_in1 holds an expert's WHOLE [K, N] block; the indexed in1 reader finds runs of consecutive
  same-expert groups on its core (the sorted dispatch lays a token's groups of one expert side by side: ~2.3 groups
  per active expert at 2048 tokens) and reads the block once per run, signalling the run position through the BRISC
  mailbox (0 skip / 1 single / 2 first / 3 middle / 4 last); a dedicated compute kernel
  (`bmm_grouped_resident_in1.cpp`, derived from the shared bmm kernel) indexes the K-blocks inside the resident set
  and pops it after the run's last group; the factory falls back to the streamed double buffer when the block does
  not fit the 1.3 MB per-core CB budget (bfp8 gate/up blocks do), `TT_SPARSE_GROUPED_RESIDENT_IN1=0` opts out; the
  layer's program config shrinks the K block so the resident block fits (gate/up ib 48, down ib 4). Cost: the next
  expert's read cannot start until the compute kernel releases the current block (reads dominate: ~3 %). Tests:
  sorted-run cases added to tests/test_sparse_matmul_grouped.py. Build + device tests + Tracy MoE 2048 A/B next.
- 09:37 width-1 top-k probe: padding the router logits to 1024 columns for the multi-core ttnn.topk is a LOSS (GDN layer
  step 0.634 -> 1.554 ms, attention 0.542 -> 0.562): the pad + wide top-k cost more than the single-core [32, 256]
  top-k. Knob `QWEN35_ROUTER_TOPK_PAD` kept at 0 (rejected). Item 6 conclusion: at width 1 the GDN layer is device-
  bound by ~60 tiny ops (3-5 us each) and the head relayouts; the pairs MoE path and the padded top-k both lose, so
  the remaining single-user lever is a fused GDN decode kernel (conv taps + gates + recurrence + norm in one program),
  parked as kernel work. Moving on with item 7 (prefill MoE: resident weights, built; validation chain running).
- 09:47 RESIDENT WEIGHTS A/B (Tracy MoE block, 2048 tokens, one GDN-layer MoE, host walls incl. sync per phase):
  gate/up sparse matmul 2.365 -> 1.430 ms (-40 %), down 1.652 -> 1.202 (-27 %); gather / dispatch / swiglu /
  un-dispatch unchanged (1.8 / 1.9 / 0.53 / 1.38) -> MoE block ~9.7 -> ~8.3 ms per layer at 2048 (-1.4 ms x 48
  layers = ~68 ms per 2048-token segment, ~15 % of the prefill time). Unit tests 14 / 14 (sorted runs per-group PCC
  >= 0.99996), layer PCC vs HF at 2048 unchanged (GDN 0.9985, attention 0.99998). Default on. The remaining prefill
  MoE levers (row gather 1.8 ms, dispatch build 1.9, un-dispatch 1.4 incl. the 0.8 ms untilize) are TILE-aware
  gather / scatter kernels; parked behind vision. Commit 6df93b62a7f (kernels) + this log.
- 09:58 (item 9, VISION) groundwork + V1-V4 code: the in-tree qwen36 vision path (tower port, trace-safe splice,
  M-RoPE) is the port source (the qwen3.8 bundle's vision code, extracted to the references dir, is tied to Qwen3.8
  and its own server). New `tt/vision.py`: the HF `Qwen3_5MoeVisionModel` built standalone from the 333 streamed
  `model.visual.*` tensors (no 250 GB from_pretrained), vision args over our args class (the stock VisionModelArgs
  died on the MoE text config), `attach_vision`. Model: multimodal segments now run on the persistent bucket path
  (the splice where-select is in the bucket traces; buffers allocated before capture), per-slot decode rope deltas
  (`set_slot_rope_delta`, used by prepare_decode_inputs_host and the MTP step's rot idxs), `rope_staged` so the
  M-RoPE table built over the whole prompt survives the segment loop. Generator: `prefill_serving(vision=[(rows,
  grid) | None])`. Adapter: `QWEN35_VISION=1` (limits image <= 4, tower attached + splice buffers + 448/896 px tower
  warm-up before the captures, per-user image rows from the prefill kwargs). Tower unit test
  (`tests/test_vision_tower.py`, 448x448 and 896x672 images vs the HF visual) running.
- 10:35 vision tower vs HF: bfp8 AND bf16 weights both give PCC 0.88 (448 px, rows aligned: sampled-row min 0.79), so
  not weight precision. Read the port: `prepare_input` zero-pads the patch sequence to a 2048 multiple and the blocks'
  SDPA is non-causal WITHOUT a mask -> every real patch attends to the padding keys (bias-only K/V; 62 % of the keys
  at 784 patches; the in-tree Qwen3.6 test tolerates 0.91 at 12 % padding). Fix in the shared qwen36 vision port:
  `cu_window_seqlens` (the SDPA op's on-device block-diagonal windows) = [0, real, padded] plumbed wrapper -> block ->
  attention; the HF semantics (block-diagonal per image) fall out for multi-image sequences too. Depth-1 / depth-4
  comparisons and the windowed full-depth test queued.
- 11:20 windowed SDPA probe (one device, [1,16,2048,96], real length 784, padding keys biased): unmasked PCC 0.983 vs
  torch, `cu_window_seqlens=[0,784,2048]` 0.99980 (exact boundary, no tile alignment needed, no hang), tile-aligned
  [0,800,2048] 0.99981, explicit [1,1,S,S] attn_mask 0.99980. The op needs an explicit `is_causal=False` with windows
  (its default is causal: the first tower chain died on that TT_FATAL inside pytest and sat until the timeout).
  Windowed tower comparison re-queued (depth 1, full depth bf16 / bfp8, 896x672).
- 11:54 VISION TOWER FIXED: with the SDPA windows (real patches attend only among themselves) the bfp8 tower matches
  the HF visual at PCC 0.9857 (448x448, 196 tokens) and 0.9868 (896x672, 588 tokens), sampled-row min 0.96 (was 0.88
  / 0.79 with the padding leak). Shared-port change (`qwen36/tt/vision/{model,vision_block,vision_attention}.py`):
  `cu_window_seqlens=[0, real, padded]` built once per image in the wrapper and passed to every block's SDPA with an
  explicit is_causal=False. bf16 tower weights: the construction stalls for minutes inside the patch merger's
  as_tensor (cache files intact) - probe queued; the served default is bfp8 (`QWEN35_VISION_DTYPE`).
- 11:58 bf16 tower probe (`scripts/16-vision-build-probe.py bf16`): the 27 blocks' cached tensors load in 0.5 s, then
  the main thread sits at 100 % CPU (single thread, no device wait) with the bf16 patch-merger fc1 cache file open
  (`visual.merger.linear_fc1_w_col ... BFLOAT16`, 42.5 MB = the expected [4608, 1152] x 4 dies, not truncated) for
  > 3 min; the bfp8 variant of the same tensor loads in milliseconds. A host-side load path cost on the bf16 TILE
  mesh tensor, not a device hang. Not pursued: bf16 bought 0.001 PCC over bfp8 (0.987 vs 0.986) and the tower is
  bfp8 by default. Probe killed so the reduced-model served image smoke (`IMAGE=1`, layers 0,3) could take the device.
- 13:10 SERVED WARM-UP HANG (found + fixed). Both reduced-model image smokes stalled after the MTP warm-up (EngineCore
  at 100 % CPU for 10+ min = `synchronize_device` spinning on a device hang, not a host loop; the adapter now registers
  `faulthandler` on SIGUSR1 so `kill -USR1 <EngineCore>` dumps the stacks into server.log). Standalone adapter repro
  (`scripts/16-vision-warm-bisect.py`): tower-only, buffers-only and both all hang at the first MTP `warm_prefill_path`
  bucket; `TT_SPARSE_GROUPED_RESIDENT_IN1=0` passes everything incl. the vision warm-up (448 px 15 s, 896 px 14 s,
  first eager run) and the captures -> the resident-weights grouped sparse matmul (commit 6df93b62a7f, never through a
  served warm-up: the prod32 run predates it). Isolated op repros with the layer's exact configs / mesh / id layouts
  (`16-resident-repro{,2}.py`) all pass, so every `ttnn.sparse_matmul` of the MTP warm prefill was traced with a
  device sync and its id list saved (`16-hang-trace.py`): the hanging call is the MTP HEAD's gate/up (in `_bucket_head`
  -> head layer -> `_routed_sorted`), program config `skip_padding_groups=0` with sentinel padding ids 0xFFFF.
  Root cause (two halves): (1) C++ - the resident in1 writer classifies id >= E as padding (run code 0, compute skips)
  for ANY config, but without SKIP_PADDING_GROUPS the plain in0 reader still pushes those groups' activations ->
  cb_in0 fills, reader blocks, program never ends. Fix: the factory runs the id-aware in0 reader (and its id CB /
  defines) whenever the resident path is on (`skip_groups = skip_padding_groups || resident_in1`). (2) Python - the
  head's block cached its grouped configs (`_grouped_configs`) BEFORE its dispatch existed, so `_skip_pad_groups` read
  False while the dispatch padded with 0xFFFF; pre-resident this "worked" by reading out-of-range weight tiles for the
  padding groups. Fix: one `_skip_pad_groups_rule()` for the dispatch and the configs. Unit test
  `test_grouped_resident_sentinel_ids_without_skip[noskip|skip]` (the T=128 layout: 186 singleton-ish groups + 95
  sentinel groups) + the 11 existing grouped tests: 13 / 13 pass after the rebuild (ninja 20 s). Also fixed: the
  faulthandler must be registered with chain=False (chain=True ran SIGUSR1's default disposition and killed the
  engine). Re-running the reduced warm-up (resident on) chained into the served image smoke.
- 13:31 VISION SERVED ON THE FULL MODEL (production config: 32 slots, 262144 ctx, 768 MiB traces, MTP k=3 adaptive,
  QWEN35_VISION=1 bfp8 tower): healthy in 10 min (tower + splice buffers before the captures; vision warm-up 448 /
  896 px 0.2 / 0.3 s with cached kernels); text requests unchanged (prose/code 64 tokens, logprobs / penalty / min_p
  served). Image chat request (448x448 synthetic PNG: red square, yellow circle on blue; 222 prompt tokens incl. 196
  image tokens, thinking off): MTP path 28 tokens in 0.99 s, text IDENTICAL to the HF bf16 CPU reference ("The image
  features a red square in the upper left and a yellow circle in the lower right, both set against a solid blue
  background."); the host-sampled plain path (logprobs request) gives the same facts in another order ("...a solid
  blue background with a red square in the upper left and a yellow circle in the lower right") - a greedy near-tie at
  token 5, so the free-running top-1 agreement (4/25) only measures the divergence point, not the tower. KV pool with
  the MTP head + tower resident: 22725 blocks x 64 = 1.45 M tokens (stage-13 text-only server: 28472 = 1.82 M); one
  request still holds the full 262144 context. Bundle manifest: QWEN35_VISION=1, QWEN35_VISION_MAX_IMAGES=4,
  `--limit-mm-per-prompt {"image": 4, "video": 0}`, card text. HANDOFF §5 rewritten (served vision). Item 10
  (readiness re-validation: serve / sampling / qualitative / benchmark, MTP + vision) launched (logs/10/vllm-48l-mtp-vision.log).
- 13:40 item 10 (readiness re-validation) in progress: host-side checks re-run on the current tree - 06-full-model,
  07-optimized-full-model, 11-tti-release all exit 0 (context contract 262144 supported; TTI artifacts name the
  autoport). The served suite (serve / sampling full profile / qualitative / benchmark; MTP k=3 adaptive + vision on,
  32 slots, 262144 ctx) is running (logs/10/vllm-48l-mtp-vision.log); 09 / 10 checks follow it. Item 11 bundle rebuild
  (run 3: MTP default-on, vision on, resident MoE) building in the background (logs/11/package-run3.log). Hygiene:
  readiness_vllm/ untracked (6a95e66978b), the Tracy env-gated patch committed (e2d56cd7328); p150x4 has no hardware
  here (profile stays "same (1,4) mesh, untested on this host").
- 13:46 readiness suite launch #1 died at startup: the API-server process exited 1 right after the image-processor
  init with NO traceback in server.log (stderr is captured there), before spawning the EngineCore. The exact server
  command relaunched by hand 12 min later went straight through to the engine init, so the failure was transient
  (the previous server's teardown / reset had completed 2.5 min earlier; the port was free at the retry). Relaunched
  the suite (10-serve.sh, same env: MTP k=3 adaptive, vision on, 32 slots, 262144 ctx).
- 14:02 readiness relaunch #2 crashed at the MTP warm-up: `KeyError: 'mtp_sel'` - the model allocates the MTP bucket
  inputs behind the RAW env `QWEN35_MTP` (default "0") while the adapter's default-on rule (k = 3) lived only in
  `_mtp_k_env`; every earlier served run exported QWEN35_MTP=3 explicitly (scripts/14-serve-mtp.sh), the readiness
  script and the BUNDLE do not -> the default-on server was broken at start. Fix (62c96f3ea1b): the adapter pins
  `os.environ["QWEN35_MTP"] = str(k)` at import (one source of truth); test `test_adapter_import_pins_mtp_env`
  (4 cases, subprocess import with a clean env) passes. The container build started at 13:35 had the unpinned tree:
  killed and restarted (run 3b). Readiness suite relaunched (#3).
- 14:15 item 11: container bundle run 3b packaged in 8 min on the pinned tree (62c96f3ea1b; docker layers cached,
  image 2.0 GB as OCI blobs) -> `~/qwen35-122b-bringup/pkgbuild/build/qwen3-5-122b-a10b-blackholex4`. The bundle
  verification (`scripts/11-serve-bundle-run3.sh`: fresh device, /v1/models, chat, tool call, tt-model curl,
  non-aligned prompt, 32-request burst, 128K + 262080-token prompts, IMAGE request vs the HF text) is queued behind
  the readiness suite on the device lock; the private push follows a manual review of its output.
- 14:24 ITEM 10 DONE: readiness suite on the full model, MTP k=3 adaptive + vision on, 32 slots, 262144 ctx
  (`doc/optimized_vllm/readiness_vllm_48l-mtp-vision/`): serve PASS, sampling full profile 73 passed / 4 skipped,
  6 qualitative completions, benchmarks: single_user_decode TTFT 396 ms, TPOT 16.6 ms = 60.3 t/s/u (stage-10 final:
  426 ms / 37.7 ms / 26.5); ci_serving_burst 100/100/32: TTFT p50 7.69 s (was 11.5), TPOT 128.9 ms (was 133.8),
  158 tok/s aggregate. Checks 09-vllm and 10-optimized-vllm exit 0 (06 / 07 / 11 earlier). Bundle verification
  (run 3b) now holds the device.
- 14:41 ITEM 11 BUNDLE VERIFIED (run 3b, staged package, fresh device, profile p300x2; `logs/11/serve-bundle-run3-chain.log`):
  healthy in 11.5 min; /v1/models max_model_len 262144; chat "The capital of France is Paris." (finish stop, no
  reasoning leak); tool call `get_weather(city=Paris)` through qwen3_xml; tt-model curl OK; non-aligned 2381-token
  prompt; 32-request burst 3200 tokens in 20.0 s = 160 tok/s; 131072-token prompt 48.8 s; the whole book as a prompt
  (194119 tokens, the source text ends there) 79.6 s; IMAGE request (448 px, thinking off) 222 prompt tokens, 28
  completion tokens in 0.97 s, text == the HF CPU reference. Clean shutdown. Pushing the bundle (private) and then
  the deployed-model perf sweep (`scripts/17-sweep-chain.sh` serves the Hub bundle and runs `17-perf-sweep.py`).
- 15:15 item 13 (deployed-model perf sweep, Hub bundle 668d1f0b, modelled on the gpt-oss-120b p150x4 sweep): pass 1
  running (`artifacts/stage17/perf_sweep_hub.json`, 16 / 48 rows at 15:10). Harness defect found in the single-user
  rows: the stream-chunk count undercounted MTP outputs (one chunk can carry several tokens: 47 of 128 counted ->
  TPOT inflated 4x); multi-user rows (plain decode, one token per chunk) are exact (1024 / 1024). Fix:
  `stream_options.include_usage` (exact completion count in the final chunk) + a `short_outputs` guard; the users=1
  rows and the sampled rows re-run in a second pass (`17-sweep-b1-chain.sh`, queued behind the thinking-default chain).
  Valid pass-1 rows so far: 128/128 at 2 / 4 / 8 / 16 / 32 users TPOT 37.0 / 44.5 / 63.7 / 102.7 / 115.9 ms, total
  48 / 74 / 100 / 120 / 185 tok/s; 1024/256 at 32 users TTFT 13.9 s, 175 tok/s.
- 17:18 OTHER-MACHINE SERVE FAILURE (user's log `serve-raahemnabeel--...-20261001-164257.log`, Hub bundle, profile
  p300x2 on a 4 x p150 host): model loaded (8 min), then the first fused-norm all-gather of the prefill warm-up died
  with `TT_FATAL fabric.cpp:174: Could not find any forwarding direction from src (M0, D0) to dst (M0, D3)`. The
  fabric log shows the box is cabled as a LINE (intra-mesh degree histogram `{1:2, 2:2}`: two chips with one
  ethernet neighbour) while the stock `ModelArgs.ccl_topology()` returns Ring for every P300_X2 / P150_X4 cluster,
  so the ring's D0 -> D3 hop has no route. Our (1,4) p300x2 box is a true ring (every die degree 2), which is why
  nothing here ever saw it. Fix: `Qwen35MoeModelArgs.ccl_topology()` (every collective of the model, the MTP head
  and the vision tower reads it) asks `ttnn.get_usable_topology(probe, Ring)` once and demotes to Linear when the
  mesh has no wraparound; `QWEN35_CCL_TOPOLOGY=ring|linear|auto` forces it. Validation queued: reduced-model served
  smoke with Linear forced on this box (`scripts/16-linear-topology-smoke.sh`: text, logprobs, image) and the auto
  rule logging Ring here. Also in that log: `--enable-auto-tool-choice` duplicated (tt-model adds it too; harmless),
  tt_transformers' "Unknown model ... MAX_PREFILL_CHUNK_SIZE 4" (unused by this model's prefill path).
- 17:20 deployed-model sweep PASS 1 complete (48 rows, 14:53 -> 17:14, `artifacts/stage17/perf_sweep_hub.json`;
  single-user TPOT column invalid - pass 2 pending). Shape of the results: prefill is serialised (no chunked prefill
  in the shipped config), so TTFT grows linearly with the closed-loop queue (4096 tokens: 1.3 s at 1 user, 39.7 s
  at 32 users; 130816 tokens: 48.5 s at 1 user = 2 700 tok/s). Decode at 2 / 4 / 8 / 16 / 32 users: 37 / 44 / 64 /
  103 / 116-131 ms TPOT for short contexts (matches the per-width traces); total output 184 tok/s at 32 users /
  128-token prompts, 175 at 1024, 112 at 4096. Long contexts at high occupancy are the weak spot: 32768 x 32 users
  425 ms TPOT, 65536 x 16 375 ms, 130816 x 8 / 16 / 32 178 / 265 / 512 ms (paged-attention reads over long K/V plus
  the serialised prefills interleaving with decode); total output falls to 5 tok/s at 130816 whatever the user count
  because the row is prefill-bound (32 prompts x 48 s).
- 17:25 CORRECTION to the 17:18 entry: the failing log is from THIS box, not another machine. Timeline: the user's
  `tt-model serve` started at 16:42:57 while the deployed-model sweep's Hub container (started 14:43, port 20000) was
  live and prefilling 130816-token prompts (its server log: prefills at 16:42:54, 16:44:30, ... 16:53:07 - it kept
  working). The second engine opened the same four chips, ran fabric auto-discovery while the first fabric owned
  the ethernet links, got a LINE adjacency `{1:2, 2:2}` instead of this box's ring, and died at the first all-gather
  (`no forwarding direction D0 -> D3`). Root cause: two engines on one device set; tt-model's "host ready: tt devices"
  check does not detect a running engine (docker hides the host's /dev/tenstorrent holders from the container).
  The cabling-aware topology rule (0b48a2312d2) stays as a robustness measure (harmless on a ring: logs Ring) but is
  not the fix for this incident - the fix is operational: one server per box (`docker ps`, `fuser
  /dev/tenstorrent/*` before `tt-model serve`), noted in HANDOFF. Linear-topology smoke stays queued (validates the
  demotion path).
- 17:35 SECOND CORRECTION (the user clarified: the failing box is ANOTHER p300x2, not this one, and not a p150x4):
  compare the fabric discovery lines - this box: `intra-mesh degree histograms mesh0 {2:4}` (every die has two
  ethernet neighbours: the two dies of each p300 plus both board-to-board links = a ring); the other box: `{1:2, 2:2}`
  (a LINE: one of the two board-to-board links is not up - unplugged, bad cable / QSFP, or a link that did not train;
  it runs firmware bundle 19.15.0 vs 19.13.1 here, same B850M-C motherboard, same KMD 2.10.0). The stock
  `ccl_topology()` returns Ring for every P300_X2 cluster, so the first all-gather asked the fabric for the D0 -> D3
  wraparound that box does not have. Fix on our side (0b48a2312d2, to ship in the next bundle): the model asks
  `ttnn.get_usable_topology` and runs Linear when the fabric has no wraparound (slower collectives, correct
  results); the remedy for the box is to restore the second inter-board link (`{2:4}` in the discovery log is the
  pass criterion). The concurrent-engine theory in the 17:25 entry was wrong for that machine (it would still apply
  to a second engine on one box, which tt-model does not detect).
- 18:02 user decision: the other box's missing-link failure is resolved on their side; the Linear-topology served
  smoke is dropped from the queue (not worth device time). The cabling-aware topology rule (0b48a2312d2) stays in the
  tree as a robustness measure but its Linear branch is UNTESTED here (this box is a ring: the rule logs Ring). Device
  queue now: thinking-default chain (running) -> sweep pass 2 -> decode-width Tracy (improvement item 2 measurement);
  bundle rebuild + push after the thinking verdict.
- 18:35 THINKING-DEFAULT DECISION TEST (production server with the model's template default = thinking on;
  `logs/11/think-default-chain.log`, `artifacts/stage11/think_probe_default_final.json`,
  tt-inference-server report 2026-10-01_18-29-35): (a) short budgets with the server-default sampling (T 0.6 /
  top_p 0.95 / top_k 20): max_tokens 256 -> empty `content` for all three simple prompts ("Say hello in one sentence"
  thinks for 1 657 tokens); 1024 -> 2 of 3 answer; 4096 -> 3 of 3. (b) thinking decode 39.8 tok/s at 1 user (MTP).
  (c) 24-prompt probe (16 IFEval + 4 GPQA + hello, 16K budget): 20 / 24 finish (stage 11 under the same preset:
  16 / 24), finished median 2 708 tokens, no degenerate repetition, every finished answer has content; 4 run to the
  16 384 cap (2 constraint-heavy IFEval, 2 GPQA; 36 min each). (d) TTI spec_tests: Acceptance FAIL, 11 / 22 tests
  fail (coherence_verbatim_echo: `content` None -> TypeError in the harness; non_uniform_seeding; 8 / 9 penalties;
  1 / 2 stop) - every failure is the think block consuming the test's small budget so `content` is empty; the same
  suite is 22 / 22 with thinking off (run 4). DECISION: the bundle keeps thinking OFF by default (the release gate);
  thinking stays available per request (`chat_template_kwargs: {"enable_thinking": true}`, card: general preset, large
  max_tokens). Shipping thinking on needs either a TTI harness that reads `reasoning` + `content` with a thinking-aware
  budget, or a reasoning-budget control the serving stack does not have (vLLM 0.26).
- 18:45 improvement item 1 scoping (prefill scheduling): the plugin itself forbids chunked prefill for a model with
  `output_tokens_per_step > 1` (`vllm_tt_plugin/platform.py::_apply_chunked_prefill_policy`, reason "block-output"), so
  MTP (k+1 tokens per step) and chunked prefill are mutually exclusive at the PLUGIN level, not only in our wiring.
  Interleaving prefill with decode under MTP therefore needs a plugin change (chunked output builder vs the
  variable-width block output) or an adapter-side scheme. Second fact from the sweep server log: the plugin hands us
  up to 31 requests in ONE prefill_forward call ("Prefilling 31 request(s)") and we run them one by one through the
  B=1 bucket path (~0.23 s each at 128 tokens, p10 of the back-to-back interval) -> the burst TTFT is per-request
  fixed cost x queue length. Measurements queued: Tracy of the 128-token bucket per layer kind + a host/device split
  of a served 128-token prefill (reduced + full model).
- 18:55 PREFILL FIXED COST, 128-token served prefill (`scripts/18-prefill-fixed-cost.py`, production adapter, MTP on,
  20 reps): full model 238 ms per request = logits readback 193 ms (blocks on the traced bucket: ~17 ms embedding /
  head rows / norm / LM head + all-gather (the 2-layer model's 24.8 ms minus its layers) + ~3.7 ms x 48 layers,
  op-launch bound like width-1 decode) + GDN slot write 28.8 ms (36 layers' rec_state + conv taps copied scratch ->
  slot, one small op per tensor) + snapshot 2.6 + two state resets 4 ms + staging 0.8 ms. Tracy of the eager
  128-token layer: 12.1 ms (GDN) / 11.8 ms (attn) per layer eager vs ~3.7 ms traced -> the traced bucket is the
  right baseline. Levers, cheapest first: (a) one fused write of the 36 GDN states into the slot (-29 ms, 12 %);
  (b) a head-rows-free bucket trace variant for prefills admitted while other requests are live (adaptive MTP only
  uses the head when exactly one request is live; the rows cost ~45 ms of the 193) -> 32-user burst TTFT -19 %;
  (c) scheduling: the plugin admits every waiting prefill in one call (max_num_batched_tokens bumped to
  max_model_len when chunked prefill is off) and a step is either prefill or decode, so 31 prefills block decode for
  7.4 s - a per-step prefill admission cap (plugin knob) would interleave decode steps like gpt-oss's "8 decode steps
  between chunks"; (d) batched short-prompt prefill (several 128-token prompts per bucket pass) = the batched GDN
  chunk form the MTP verify pass already uses (B users x 32 rows) generalised to B x 128 + batched prefill attention:
  the biggest win (~2.5x on short bursts) and the most work.
- 19:00 plugin scheduler read (`vllm_tt_plugin/scheduler.py::TTScheduler`): default mode "prefer prefill whenever
  prefill work is pending" and a step is prefill-only or decode-only, so waiting requests are admitted together (the
  base token budget = max_model_len once chunked prefill is off) and decode starves until the last of them is
  prefilled (the sweep: 31 prefills of 128 tokens = 7.4 s with no decode step). A gpt-oss-style "prefill step, then
  D decode steps" alternation is a scheduler policy change in the PLUGIN (forced modes exist for the lane
  coordinator, no public knob); the bundle pins the plugin revision, so a local plugin patch does not ship without
  vendoring -> adapter-side levers first (fused GDN slot write, head-rows-free bucket variant, batched short-prompt
  prefill), scheduler alternation raised with the plugin team.
- 19:02 DECODE WIDTH PROFILE (reduced model, traced batched decode step per layer kind, `artifacts/stage15/tracy_decode_*-oct1`):
  per-layer step wall GDN 1.35 / 2.16 / 2.75 ms and attention 1.22 / 1.94 / 2.63 ms at 8 / 16 / 32 users ->
  36 x GDN + 12 x attn = 63 / 101 / 130 ms, which is the served TPOT (64 / 103 / 129): the step is all layer device
  time, host overhead nil. Per-op (device 0): at 8 users the GDN layer is 61 ops / 934 us device (+0.4 ms of
  op-to-op gaps): the two MoE sparse matmuls 338 us (36 %), 15 binary ops 154 us, tilize-with-padding 82 us; the
  attention layer 85 ops / 1086 us: sparse matmuls 417 us (38 %), tilize 139 us (13 %), 10 binary ops 112 us. At 16
  users (pairs path) the attention layer is 73 ops / 1676 us: sparse matmuls 795 us (47 %, "active=?/32768"
  pair-groups), tilize 272 us (16 %), binary 185 us -> the MoE is ~65 % of the step and its weight traffic is one
  expert read per (user, expert) PAIR (128 reads at 16 users) with ~50 % bandwidth efficiency; sorting rows by expert
  (the grouped / resident path) would read each distinct expert once (~80 at 16 users) and drop the pairs path's
  relayout glue (tilize + slices). The b32 Tracy windows are empty (profiler buffer limit at 32 rows, known); the
  PERF wall numbers stand. NEXT (item 2): route decode rows >= N through `_routed_sorted` (the expert-sorted
  dispatch + grouped resident matmuls the MTP rows mode and prefill use) behind a knob and A/B the per-width step
  at 8 / 16 / 32 on the reduced model.
- 19:11 ITEM 2 A/B (decode rows >= 8 through the expert-sorted grouped path, `QWEN35_DECODE_SORTED_MIN_ROWS=8`;
  `logs/18/decode-sorted-ab.log`): SLOWER at every width - GDN layer step 2.26 / 2.51 / 2.91 ms vs 1.32 / 2.13 /
  2.72 (pairs / masked) at 8 / 16 / 32 users, attention 2.12 / 2.39 / 2.68 vs 1.18 / 1.90 / 2.60; correct (B=16 /
  32 per-user PCC vs HF >= 0.9998). The sorted path's fixed cost (dispatch kernel, row gather, un-dispatch, G ~ 257
  mostly-padding groups) outweighs the saved expert re-reads at these row counts. Knob stays 0 (off); a Tracy of the
  sorted path at 16 rows is queued to see which of its ops carry the fixed cost before deciding on a small-T variant.
- 19:23 MTP PREFILL TAX measured (same 128-token split, MTP off, full model): 231 ms vs 238 with MTP on -> the head
  rows + hooks cost ~7 ms (3 %); the ~45 ms TTFT difference seen between served configurations is phase entry and
  harness, not the bucket. Plan item 2 closed (not worth work). The bucket is 187 ms without MTP: ~3.5 ms per layer
  of op launches for 128 tokens - the lever is fewer ops per layer or several prompts per pass, not MTP.
- 19:23 SORTED PATH AT 16 ROWS, Tracy (`artifacts/stage15/tracy_decode_attn-b16-sorted8`): 74 ops / 2044 us device.
  The grouped resident sparse matmuls are FASTER than the pairs path's (381 + 224 = 605 us vs 795: each distinct
  expert read once), but the path pays 601 us (29 %) in the row gather (`gather_rows` = the embedding op over the
  dispatched rows: 4 128 mostly-padding rows x 3072 bf16 = 25 MB at ~42 GB/s, effectively single-core) and 248 us
  (12 %) in the un-dispatch's untilize (`UntilizeCodegen`) before the multi-core gather-add kernel (34 us). Same
  glue the prefill MoE pays (this morning: gather 1.8 ms, un-dispatch 1.4 ms incl. 0.8 ms untilize of ~8.3 ms per
  layer at 2048 tokens). ONE kernel pair fixes both: a multi-core TILE-layout row gather (like moe_gather_add_rows,
  skipping padding groups) and a tile-aware gather-add (no untilize). Projected: sorted path at 16 rows ~1.25 ms vs
  pairs 1.68 (-25 % on the attention layer step; 16-user TPOT ~103 -> ~85 ms), prefill MoE per layer -2.5 ms at
  2048 tokens (-30 %). This becomes plan item 1.
- 19:29 NEW OP `ttnn.moe_gather_rows` (ttnn/cpp/ttnn/operations/experimental/moe_gather_rows, sibling of
  moe_gather_add_rows): work item = (group, column block) over all cores; the reader gathers a REAL group's 32 source
  rows (src_tok stick) straight into a row-major L1 block, the compute kernel tilizes it, the writer stores the tiles
  at the group's offset; a padding group (group_ids[g] >= E) costs a zero block and no DRAM traffic (flag CB reader
  -> writer, tiles dropped). Replaces the embedding-op gather over every dispatched row (`gather_rows`, now routed
  to the kernel behind `QWEN35_GATHER_KERNEL`, default on, when the grouped skip-aware matmul consumes the output).
  Built (ninja 33 s); validation chain running: unit tests (bit-exact vs torch and vs the embedding gather, perf
  vs embedding) gate the layer PCC tests, the sorted-decode A/B at 8 / 16 / 32 and the 2048-token prefill Tracy.
- 19:35 moe_gather_rows unit tests 5 / 5: bit-exact vs torch and vs the embedding gather at T=16 (G=129, 98 real
  groups), T=2048 (G=761, 631 real), T=37 (G=281, both core groups); host wall incl. sync: 0.527 -> 0.392 ms at
  T=16, 1.507 -> 1.022 ms at T=2048 (device op time from the Tracy below). Note the `ninja ttnn` build does not
  refresh `ttnn/ttnn/_ttnn.so` (the nanobind module; stale since 04:56 - the C++ op library resolves to
  build_Release via rpath, which is why the resident fix was live but the new binding was not): copied both .so.
- 20:24 GATHER KERNEL RESULTS: (a) prefill MoE block at 2048 tokens (`12-tracy-moe.sh`, phase host walls incl. sync,
  `artifacts/stage12/tracy_moe_gather_kernel`): gather_rows 1.80 -> 1.27 ms (the rest unchanged: dispatch 1.85,
  sparse matmuls 1.43 + 1.20, swiglu 0.53, un-dispatch 1.38) -> MoE ~8.3 -> ~7.7 ms per layer (-0.6 ms x 48 =
  -29 ms per 2048-token segment, ~-6 % of prefill). The gather now moves 124 MB of real rows in + out (near
  bandwidth) plus the input untilize; the remaining un-dispatch cost is its 0.74 ms untilize. (b) decode, expert-
  sorted path with the kernel: per-layer step 1.98 / 2.24 / 2.63 ms (GDN) and 1.85 / 2.11 / 2.40 (attn) at 8 / 16 /
  32 rows vs the shipped pairs / masked 1.32 / 2.13 / 2.72 and 1.18 / 1.90 / 2.60 -> wins at 32 rows (-3 % GDN, -8 %
  attn), loses at 8 / 16. Knob `QWEN35_DECODE_SORTED_MIN_ROWS` stays off pending the next step. NEXT: the down
  projection's output untilize (0.74 ms at 2048 tokens, 0.25 ms at 16 rows, ahead of the multi-core gather-add) ->
  have the grouped sparse matmul factory emit ROW_MAJOR output (`untilize_out`, the bmm compute kernel's
  pack-untilize path) so the un-dispatch reads it directly.
- 20:36 DOWN-PROJECTION `untilize_out` (grouped sparse matmul factory): new `MatmulMultiCoreReuseProgramConfig.
  untilize_out` -> the op's output spec / optional output are ROW_MAJOR [1, G, 32, N]; the resident compute kernel
  takes the bmm pack-untilize path (compile arg 15) and the in1 writer stores each real group as 32 sticks. Finding:
  pack-untilize is exact only up to 16 tiles of row width on Blackhole (N=512 bit-exact, N >= 768 wrong, independent
  of K), so the kernel untilizes the [32, N] block in chunks of UNTILIZE_CHUNK_W (largest multiple of the subblock
  width dividing N, <= 16) and the writer stores partial sticks (page offset = chunk column). Needs the resident
  kernel (TT_FATAL otherwise; the Python knob `QWEN35_DOWN_UNTILIZE_OUT`, default on, is gated on the resident
  budget + skip-aware path + bf16 activations). Unit test `test_grouped_untilize_out_matches`: ROW_MAJOR == TILE
  output bit for bit on the real groups for N = 512 / 768 / 1024 / 2048 / 3072 (down: K=256, N=3072). `undispatch_rows`
  feeds the row-major output straight to the gather-add kernel (no untilize op). Validation chain running (layer
  PCC, sorted decode PCC + per-width PERF, MoE-block Tracy at 2048).
- 20:56 UNTILIZE_OUT RESULTS: layer PCC vs HF unchanged (0.9985 / 0.99998 prefill 2048; sorted decode B=16 / 32
  per-user >= 0.99896); prefill MoE block at 2048 tokens (phase walls): un-dispatch 1.38 -> 0.64 ms -> MoE ~8.3 (this
  morning) -> ~6.9 ms per layer (-1.4 ms x 48 = ~-67 ms per 2048-token segment, ~-10 % of a long prefill); decode
  per-layer step with the expert-sorted path (gather kernel + untilize_out) 1.74 / 1.99 / 2.38 ms GDN and 1.61 /
  1.87 / 2.16 attn at 8 / 16 / 32 rows vs the shipped pairs / masked 1.32 / 2.13 / 2.72 and 1.18 / 1.90 / 2.60:
  -7 % at 16 users, -13 / -17 % at 32, still +32 % at 8. DECISION: `QWEN35_DECODE_SORTED_MIN_ROWS` defaults to 16
  (projected served TPOT 16 users 103 -> ~94 ms, 32 users 130 -> ~112 ms). Full-model served check (production
  config, EXTRA + IMAGE + benchmark c=16 / 32) launched; then bundle rebuild / verify / push.
- 21:14 SERVED CHECK, production config, sorted decode from 16 rows + both kernels (`logs/13/serve-sorted16-prod.log`,
  readiness streaming benchmark 128/128, 4 x c requests): c=16 TPOT 95.3 ms (was 103; 133.6 tok/s aggregate), c=32
  TPOT 110.2 ms (was 115.9 in the sweep / 128.9 in the readiness burst; 201 tok/s aggregate vs 184 / 158), TTFT
  c=32 6.36 s (was 7.75). Image request: MTP and plain texts now both identical to the HF reference (28 / 28 top-1).
  REGRESSION at c=1: TPOT 26.5 ms (was 18.6-20.1 with this benchmark): the MTP verify pass runs its MoE on a 32-row
  block (4 real candidate rows padded to the GDN block), and the new `rows >= 16 -> sorted` rule intercepted it (the
  MTP rows mode had chosen the pairs path deliberately: `QWEN35_MTP_MOE_ROWS=pairs`). Fix: the threshold rule
  applies to plain decode only; the MTP rows mode keeps its own choice. Bundle run 4 (packaged 21:04) carries the
  regression -> rebuild after the fix.
- 21:32 c=1 still 26.4 ms after the MTP-rows guard -> MTP_STATS show the pass itself at 86-89 ms per device step
  (71.5 this morning) with the same acceptance (2.7-2.9 tokens per pass). Cause: `_grouped_configs` set `untilize_out`
  on the SHARED down-projection config, so the pairs path (the MTP pass's 32-row block and plain decode at 2-16 rows)
  also produced row-major output and paid layout conversions afterwards. Fix: a separate config object
  (`_cfg_grouped_dn_rm`) used only by `_routed_sorted`; the pairs path keeps TILE output. Served re-check at c = 1 /
  8 / 32 launched; the MTP-rows guard stays. Bundle run 5 (21:22) carries this regression -> rebuild after the check.
- 21:47 SERVED RE-CHECK with the separated configs (`logs/13/serve-sorted16-cfgfix.log`, production config, readiness
  streaming benchmark 128 / 128): c=1 TTFT 320 ms, TPOT 20.6 ms = 48.4 t/s/u (MTP pass back to 67-71 ms per device
  step, 2.7-2.9 tokens per pass); c=8 TTFT 1.83 s, TPOT 63.9 ms (unchanged: pairs path), 103 tok/s; c=32 TTFT 6.34 s,
  TPOT 110.2 ms, 201 tok/s. Against the stage-13 release numbers (37.7 / 93.1 / 125-131 ms) and this morning's
  (16.6-18.6 / 64 / 116-129): the 16-32-user win stands (-7 % / -14 %), 1 and 8 users unchanged within run-to-run
  noise of the MTP acceptance. Bundle run 6 packaging; verify + push next.
- 22:08 BUNDLE RUN 6 VERIFIED (staged package from 769bf6fa1d6-era tree HEAD 4c757328bca docs; code = eb45675cccd +
  f4920e34075 + d6d5337269c; fresh device, profile p300x2; `doc/tti_release/bundle_run6/verify.log`): healthy in
  ~11 min; chat / tool call / tt-model curl OK; non-aligned 2381-token prompt; 32-request burst 3200 tokens in 17.0 s
  = 188 tok/s (run 3b: 160); 131072-token prompt 45.0 s (48.8); 194119-token book 73.8 s (79.6); image request 0.92 s,
  text == HF reference. Pushing (private).
- 22:09 PUSHED: Hub revision 832a9834a5f0 (private; image e096e047ba10; tt-metal aa559ead231). Report republished
  (v28) with the revision and the run-6 verification. Next: batched short-prompt prefill (design in
  production_improvements_plan.md), then the deferred sweep rows on this bundle.
- 22:18 BATCHED SHORT-PROMPT PREFILL, step 1 committed (c8b026f34f9): `TPAttention.forward_prefill_paged_batched`
  (qwen36 attention/tp.py; one QKV / q-k norm / rope / head concat / gate / output projection over B*T rows, per-user
  `paged_fill_cache` + causal chunked SDPA against the user's own page-table row) + `tests/test_batched_prefill_attention.py`
  (B = 4 vs 4 single-user prefills, T = 128 / 512, paged K/V identical). Queued on the device behind the measurement
  chain (`scripts/20-batched-attn-test.sh`).
- 22:45 Step 2-4 written while the device measures (no device run yet): `tt/prefill_batched.py` = `BatchedPrefill`
  (persistent per-bucket inputs for B = 4: tokens [B, T], rope tables over B*T rows, B one-row page tables + chunk
  tables, `RowsLen`, one-hot logits selector [32, B*T], hidden-rows buffer; a dedicated B-row GDN scratch bound like the
  B = 1 prefill scratch with its own zero sources; GDN layers through `forward_prefill_batched(carry=True)` after
  `rebuild_batched_conv_carry` (device-only, trace-safe), attention layers through the batched attention, norms / MoE
  on B*T rows, LM head on the B selected rows; `write_slots` = scratch row u -> decode slot; `warm` / `capture_traces`
  (one trace per bucket 128 / 256 / 512) / `release_traces`). Serving: `Qwen35MoeGenerator(QWEN35_BATCHED_PREFILL=1)`
  builds the driver before the warm-up, warms it with the other prefill programs (before the decode capture) and
  captures its traces after the bucket traces; `prefill_serving` groups the call's final text requests <= 512 tokens by
  bucket in fours (`_batched_groups`), runs each group first, samples row i of the group's sharded logits
  (`sampler(sh, u, row=i)`), then runs the rest per request as before. MTP: `MtpServing.note_batched` takes the request
  out of the prefill queue, uses the pass's final-normed last-position row as the slot's first-draft hidden and marks
  the slot head-less (`no_head_slots`: no MTP phase until a new request takes the slot; follows slot remaps).
  Test `tests/test_prefill_batched.py` (reduced 2-layer generator, B = 4 slots): groups [20, 57, 128, 100] and
  [300, 129, 512] (one idle row) vs the production per-user path - logits PCC / argmax, GDN slot states, paged K/V
  identical, the sharded row == the gathered row; then traced replay == eager and no program compiled after capture.
  Queued behind the attention test, conditional on it passing (`scripts/20b-batched-model-test.sh`).
- 22:36 SWEEP PASS 2 DONE (Hub revision 832a9834 served from the Hub on a fresh device; `logs/17/sweep-b1-chain.log`,
  `artifacts/stage17/perf_sweep_hub_b1.json`, 13 rows): single-user greedy TPOT 24.2-25.5 ms (39-41 tok/s/user) at
  128 .. 65536 input tokens by the fixed harness (random-token prompts; the natural-text readiness benchmark gives
  20.6 ms on the same bundle), 18.4 ms (54 tok/s) after the 130816-token prompt; TTFT within 1 % of pass 1. Sampled
  rows (T = 1.0): 1 user 36.0 / 35.5 ms, 8 users 63.7 ms (== greedy), 32 users 129.7 / 131.5 ms vs 110.2 ms greedy
  -> the on-device sampler costs ~20 ms per step at 32 rows (new improvement item: sampler cost at full occupancy).
  `deployed_sweep.md` updated (pass-2 columns, pass-2 sampled table, reading notes). Multi-user MTP A/B running next.
- 22:50 SERVED IMAGE FAILURE (user report, another p300x2, bundle 832a9834): an image + text request killed the engine
  with `programs compiled after the trace captures (1894 -> 1923) (prefill_serving)` (29 programs); the MTP abort path
  then crashed on `ttnn.deallocate(torch.Tensor)` (`_drop_hid_rows` on the host-harvested hidden rows), masking the
  cause. Root cause: the tower pads each image to the next 2048-patch multiple and its programs are keyed by that
  padded length; the warm-up compiled 2048 / 4096 only (448 / 896 px squares) while the checkpoint's processor allows
  16 MP images (a 4032x3024 photo = 47,376 patches -> padded 49,152; even 1920x1080 = 8,160 -> 10,240). Fix:
  (1) served resolution cap `QWEN35_VISION_MAX_PIXELS` (default 4 MP = 16,384 patches = 4,096 image tokens per image)
  enforced in `TT_Qwen3_5MoeProcessingInfo.get_hf_processor` (clamps the image processor's `longest_edge`; a smaller
  client `max_pixels` is kept); (2) `_warm_vision` compiles every padded bucket 2048 .. 18432 with synthetic patches
  (`tt/vision.py: warm_tower_sizes`, nine eager passes); (3) `_vision_for_user` raises a clear error if an image still
  exceeds the cap; (4) `_drop_hid_rows` / `end_prefill` skip `ttnn.deallocate` for host rows. tt-model.yaml carries the
  knob; HANDOFF §5 updated. Test `tests/test_vision_sizes.py` (bfp8 tower: warm the buckets, then 1080p / 2048x2048 /
  4032x3024 / two images through the capped processor must compile nothing and match the HF visual) queued before the
  served A/B, whose extra check now sends those image sizes to the server (`scripts/22-image-sizes.py`).
- 22:49 FAILED ATTEMPT (mine): the multi-user MTP A/B server died at warm-up with `AttributeError: batched_prefill` -
  the driver attribute was set only on the standalone constructor path, not on the external-KV (vLLM) path that
  returns early. Fixed (`_init_batched_prefill` from both paths, attribute initialised first; 46ac95cbc02); the A/B is
  re-queued after the batched-prefill served A/B.
- 22:58 VISION SIZE TEST, attempt 1: the nine-bucket warm-up compiled (2.3 s / 2.7 s for 2048 / 4096, ~13 s each from
  6144 to 18432: 95 s total) but a 1080p image (8160 patches, bucket 10240) still compiled 9 programs: the tower sliced
  the block output to the REAL length before the patch merger (`x[:, :, :unpadded_seq_len, :]`), so every merger
  program was keyed on the exact patch count. Fix in the shared port (`qwen36/tt/vision/model.py`): the merger runs
  on the padded sequence; the real rows (and the concat of several images) are taken on host and re-uploaded
  hidden-fractured (the prefill splice reads the rows to host anyway). Retest queued after the batched tests.
- 23:00 BATCHED ATTENTION TEST PASSED (attempt 2, with the fabric device params): B = 4 vs 4 single-user prefills at
  T = 128 and 512, per-user PCC 1.000000, paged K/V identical. Model-level batched prefill test running.
- 23:03 VISION SIZE TEST, attempt 2 (merger on the padded sequence): warm-up 17.6 s for the nine buckets (was 95 s:
  the merger at the real length was the slow part), the 1080p image compiled 1 program (was 9). Remaining candidate:
  the tiled host->device upload of the real rows (a non-tile-multiple row count may tilize on device) -> the rows now
  go up ROW_MAJOR (the splice reads them to host anyway); the test now reports per-size program deltas (1080p twice,
  another length in the same bucket, cap-max, phone photo, two images) and asserts at the end.
- 23:05 BATCHED MODEL-LEVEL TEST, attempt 3: logits PCC 0.9994-1.0 (argmax equal for 6 of 7 users, the seventh inside
  the reference top-5), GDN slot states match, K/V identical for the full group; the 3-request group differed only in
  the reserved KV scratch block (the idle 4th row's pad tokens vs the single path's pad rows: never read) -> the test
  compares the users' blocks only. All remaining device work now runs from ONE sequential chain
  (`scripts/23-master-chain.sh`: vision test -> batched test -> served A/B (if passed) -> multi-user MTP A/B); the
  per-step waiters raced each other and twice started the MTP A/B early.
- 23:09 VISION SIZE TEST PASSED (attempt 3, `logs/18/vision-sizes-test3.log`): the nine-bucket warm-up takes 4.9 s;
  1080p (8160 patches, twice, plus 1088 px), 2048x2048 (16384 patches = the cap), a 4032x3024 photo resized by the
  cap to 16060 patches and a two-image request all compile 0 programs; tower PCC vs the HF visual 0.9875 (1080p) and
  0.9872 (two images). The served image failure is fixed in the tree (cap + full-bucket warm-up + padded merger +
  row-major rows); ships with the next bundle.
- 23:44 MULTI-USER MTP A/B, run 2: both servers (sorted / pairs rows) came up and died at the FIRST request:
  `TypeError: 'tuple' object does not support item assignment` in `prefill_serving` - the batched-prefill integration's
  results dict `res` was shadowed by the per-request `res = self._prefill_single(...)` tuple (every served prefill from
  this tree since c0baac21ffd; no unit test drives `prefill_serving`, the served checks do). Fixed (`outs_by_u`,
  74ed98b4065) before the batched A/B server started (23:45:17 vs its launch); the MTP A/B is re-queued after the k
  sweep. Lesson: a served smoke (one request) belongs in the test chain before any long served measurement.
- 23:45 BATCHED MODEL-LEVEL TEST PASSED (attempt 5, `logs/18/batched-tests5.log`): eager groups [20, 57, 128, 100] and
  [300, 129, 512] vs the per-user path - logits PCC >= 0.9994 (argmax equal 6 / 7, one inside the top-5), GDN slot
  states match, users' K/V blocks identical; warm + 3 bucket traces, traced replay == eager, no program compiled after
  the captures. Served A/B (32 slots, batching on, c = 32 / 8 / 1, greedy text check, large-image requests) running.
- 00:00 (10-02) SERVED A/B, BATCHED SHORT-PROMPT PREFILL (`logs/18/serve-batched-ab.log`, checkout server, production
  config 32 slots / 262144 ctx / MTP k=3 / vision on / program guard, `QWEN35_BATCHED_PREFILL=1`; readiness streaming
  benchmark 128 / 128, 4c requests): c=32 TTFT 3.76 s (was 6.34), TPOT 110.5 ms (110.2); c=8 TTFT 1.27 s (1.83), TPOT
  63.9 (63.9); c=1 TTFT 272 ms (320), TPOT 19.6 ms (20.6). The server log shows the admissions (groups of 4 and 3 at
  128 tokens, a [20, 26, 24] group), no program compiled after the captures. Greedy texts single vs batched admission:
  3 / 4 identical, the fourth a fluent paraphrase diverging at char 54 (the batched GDN form's PCC 0.9995 numerics).
  Image sizes on the same server: 1080p 2.0 s (2064 prompt tokens), 4032x3024 photo 3.3 s (4039), two images 2.8 s -
  the served image fix works end to end. DECISION: batched prefill ON for serving (tt-model.yaml env + adapter default;
  the standalone generator keeps it off). Bundle run 8 carries it; its verification adds the readiness sampling stage
  against the bundle server (a default-on decision is exercised with the bundle's exact env).
- 00:26 (10-02) MTP DRAFT-LENGTH SWEEP, served (one user, production config, readiness streaming benchmark 128 / 128;
  `logs/18/serve-ksweep-k{2,4}.log`): k=2 TPOT 23.9 ms (41.8 tok/s; 2.35-2.47 accepted per 64-ms pass), k=4 TPOT
  24.0 ms (41.7 tok/s; 2.9-3.4 accepted per 88-92 ms pass), k=3 (shipped) 19.6-20.6 ms (48-51 tok/s). Timed requests:
  prose k=2 28.5 / k=3 26.4 / k=4 20.8 tok/s incl. TTFT; code k=2 39.3 / k=3 48.2 / k=4 46.7. DECISION: k = 3 stays
  (the best of the three at one user on both prompt kinds and the benchmark).
- 00:55 (10-02) MULTI-USER MTP A/B, run 3 (`logs/13/serve-mtpmulti-{sorted,pairs}.log`; QWEN35_MTP_ADAPTIVE=0, 4 slots,
  expert-sorted vs pairs rows, readiness streaming benchmark c = 1 / 2 / 4): TPOT c=1 48.5 ms, c=2 46.7 ms, c=4 58.7 ms
  (both rows paths identical within noise; 6.2-6.6 accepted tokens per 152-ms 4-user pass). Against the shipped
  adaptive mode (c=1 19.6-20.6 ms MTP step; c=2 37 ms, c=4 44.5 ms plain decode) the batched multi-user step loses at
  every concurrency. DECISION: adaptive stays (`QWEN35_MTP_ADAPTIVE=1`, the default). Settings final for bundle run 8:
  MTP k=3 adaptive, batched short-prompt prefill on, vision on with the 4 MP cap, expert-sorted decode MoE from 16
  rows, thinking off. Package + fresh-device verification (bundle smoke, large images, readiness sampling stage
  against the bundle server) running.
- 01:34 (10-02) BUNDLE RUN 8 VERIFIED EXCEPT CONFORMANCE (packaged from 1d90836a5b4, batched prefill ON; fresh device,
  `logs/11/serve-bundle-run8-chain.log`): healthy in 12 min; chat / tool call / tt-model curl OK; 2381-token prompt;
  32-request burst 3200 tokens in 14.2 s = 225 tok/s (run 6: 188); 131072-token prompt 44.9 s; 194119-token book
  74.0 s; 448 px image == HF reference; 1080p 2.0 s / 4032x3024 3.2 s / two images 2.8 s (the served image fix,
  verified on the bundle). Readiness SAMPLING stage against the bundle server: 61 passed, 12 FAILED
  (`logs/11/bundle8-sampling_tests.saved.log`): seeded reproducibility across batches / uniform seed / same seeds
  across batches, greedy consistency across positions and runs, mixed-param batch isolation, mixed-batch penalties,
  bad_words. Cause: a batched admission's first token comes from the batched pass (GDN batched form, PCC 0.9995 vs the
  single path -> near-tie flips: "1.3: The" vs "1.3.2") sampled from row i of the group's logits (a different RNG
  row than the single path's row 0), so the same request yields different tokens depending on what arrived with it.
  DECISION: batched short-prompt prefill back to OPT-IN (`QWEN35_BATCHED_PREFILL=1`; yaml "0", adapter default off;
  a063e347809) - conformance (73 / 73) outranks the TTFT win for the shipped default. Future scope: make the first-token
  numerics admission-independent (same kernels for single and batched, row-0 RNG semantics via a row select) and add
  the MTP head rows for batched admissions, then flip the default. Bundle run 9 (opt-in off) packaging + verifying.
- 01:50 (10-02) WRAP-UP (owner): bundle run 10 = serving PROFILES (`tt-model.yaml serve_profiles`: `p300x2` interactive
  default; `p300x2-burst` = QWEN35_BATCHED_PREFILL=1; `p300x2-longctx` = QWEN35_CHUNKED_PREFILL=1 + QWEN35_MTP=0 +
  `--enable-chunked-prefill --max-num-batched-tokens 2048`; `p300x2-text` = QWEN35_VISION=0 + image limit 0; `p150x4`)
  and THINKING ON by default (`--default-chat-template-kwargs '{"enable_thinking": true}'`, per-request opt-out) by
  owner decision - the gates and the latency numbers stay those of the thinking-off configuration, documented in the
  card, HANDOFF and report. Chained after run 9's verification (staged build dir shared): package -> fresh-device
  verification (smoke, large images, readiness sampling stage with the thinking-on default; failures there are
  reported as such) -> push.
