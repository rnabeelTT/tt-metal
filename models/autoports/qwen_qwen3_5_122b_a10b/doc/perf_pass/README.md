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

| users | TTFT p50 | TPOT stage 10 -> pairs only -> final | t/s/u final | aggregate tok/s stage 10 -> final |
|---|---|---|---|---|
| 1 (own width-1 trace) | 408 ms | 37.7 -> 37.3 -> **32.7 ms** | **30.6** | 24.6 -> 28.0 |
| 2 | 0.82-1.13 s | 39.4 -> 39.1 -> 38.1 ms | 26.2 | 43.1 -> 44.0 |
| 4 | 1.62-1.96 s | 51.8 -> 47.7 -> 46.7 ms | 21.4 | 60.8 -> 65.5 |
| 8 | 3.24-3.57 s | 93.1 -> 68.5 -> 67.7 ms | 14.8 | 66.8 -> 84.5 |
| 16 | 6.5-6.8 s | 107.6 -> 107.5 -> 107.0 ms | 9.3 | 100.6 -> 100.4 |
| 32 | 12.9-13.3 s | 125.1 -> 124.9 -> 125.2 ms | 8.0 | 140.6 -> 140.7 |

Reproducibility on the served path with the width-1 trace (`served_w1/seedprobe.json`): seeded completions 1 unique of
6 / 12 / 24 / 24 / 48 at widths 1 / 2 / 4 / 8 / 16, greedy width-1 re-runs 1 unique of 6 (stage 10 at width 1: 6 of 6).

## Rejected in this pass

- Expert-sorted MoE at decode (`sorted_min_tokens` 256 -> 8 with the group-parallel factory): per layer w8 4.16 /
  w16 4.37 / w32 4.77 ms vs 1.99 / 2.35 / 2.78 masked: the dispatch bookkeeping (~170 ops) is op-launch bound at
  T <= 32. Correct (12/12) but slower; the knob `QWEN35_SORTED_MIN_TOKENS` stays for prefill experiments.

## Where the remaining time is

- Widths >= 16 are weight-bandwidth bound: a step touches ~all 256 experts -> ~55 GB of bfp4 expert weights per step
  over 48 layers (~108 ms at ~512 GB/s per die); measured device step 86-92 ms in-process, served 107-125 ms -> the
  ~20-35 ms gap at wide batches is host-side serving overhead per step (next lever for aggregate tok/s).
- Width 1 (32.7 ms): ~96 device ops per layer (4600 per step + 2400 deallocations, eager count): op-launch bound; the
  functional router costs ~5 ms per step more than the (racy) fused gate op (linear + topk + softmax on a 1-row tile, DRAM
  intermediates: an L1-resident router chain is the next small step); the largest
  op groups are the GDN decode mixer (~50 ops: conv shift register, head layouts, recurrence) and the attention block.
  `ttnn.experimental.kda` ships fused kernels (`qkv_causal_conv1d_silu`, `sigmoid_gated_rms_norm`, recurrent scans)
  that could replace several of them.
