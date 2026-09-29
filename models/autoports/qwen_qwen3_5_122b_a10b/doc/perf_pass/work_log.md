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
