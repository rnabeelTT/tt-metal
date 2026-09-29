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
