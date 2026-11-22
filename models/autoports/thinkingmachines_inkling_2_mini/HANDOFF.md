# Inkling-2-Mini on Blackhole: bring-up log

thinkingmachines/Inkling-2-Mini (26.4B looped MoE, 3.4B active / 5.4B effective per token, NVFP4/MXFP8 experts) on
p150 / p300 (p150x2) / p300x2 (p150x4), everything on device. Goal: served through vLLM + the TT plugin and published
with tt-model-manager, like the other ports. Branch `raahem/inkling-2-mini`.

## Model, as ported

24 layers, hidden 2304; layers 6..17 run 3 passes (48 applications, each with its own KV / conv state; passes 2-3
restart from the pre-loop state and get messages from the previous pass through per-layer gates). Global attention at
layers 5, 11, 17, 23 (8 applications), sliding window 1024 elsewhere. No RoPE: a hidden-state-conditioned relative bias
(r @ proj over 1024 distances), per-token sink logits, sigmoid-gated o_proj, L2-clipped q/k, unweighted-RMS v; short
convs on K, V (dynamic taps on the global layers), attention output and MoE output; block attention residuals over the
inputs of layers 0, 6, 12, 18. MoE: 128 experts top-8 (sigmoid + selection bias, route_scale 8),
act = 4 tanh(g/4) sigmoid(g) 25 tanh(u/25). Logit softcap 30.

## Code

- `tt/weights.py`: checkpoint reader (original names), NVFP4 / MXFP8 dequant bit-exact vs transformers 5.24.
- `tt/attn/`: Inkling-Small's custom SDPA (relative bias in-band, sliding ring, causal rows) + per-row sinks and
  per-item page tables (multi-user). `tests/test_sdpa_device.py` (TP1 / TP4 shapes).
- `tt/attention.py`, `tt/sconv.py` (static fused kernels from Inkling-Small; `DynShortConv`), `tt/moe.py`
  (sparse_matmul gather / masked paths), `tt/model.py` (block residuals, message gates, the 48-application schedule),
  `tt/ccl.py` (replicated-residual all-reduce), `tt/wload.py` (weight cache).
- Reference: `reference/golden.py` (per-application fixtures), `reference/ref_texts.py` (quality-gate reference).
- Tests: `tests/test_apps_device.py` (teacher-forced per application, prefill + decode), `tools/teacher_force.py`.

## Quality (teacher-forced vs the CPU reference, 8 held-out 1000-token texts)

| configuration | top-1 | top-1 (confident) | KL (top-32) | NLL (ref 2.022) |
|---|---|---|---|---|
| noise floor: fp32 vs bf16 CPU reference | 0.975 | 0.998 | 0.010 | 2.020 |
| bfp4 experts, bfp8 KV, 1 die | 0.938 | 0.989 | 0.040 | 2.029 |
| bfp8 experts, bfp8 KV, 1 die | 0.967 | 0.997 | 0.016 | 2.029 |
| **bfp8 experts, bf16 KV, 1 die** | **0.970** | **0.996** | **0.010** | 2.025 |
| bf16 experts, bf16 KV, TP4 | 0.971 | 0.996 | 0.013 | 2.027 |

bfp8 experts + bf16 KV is at the noise floor (bf16 experts add nothing). bfp4 is a measurable loss. Per-application
test (bf16 experts): worst pcc 0.99888 (prefill and decode), head logits pcc 1.00000.

## Host / ttnn issues found (keep)

- Host -> device writes from file-backed memory (safetensors mmap, ttnn .tensorbin via as_tensor) crawl on this host
  (kernel 7.0, page cache in large folios: get_user_pages splitting huge PMDs; 0.2 s -> tens of minutes,
  intermittent). Clone checkpoint tensors; stage cache files through /dev/shm (`tt/wload.py`).
- `ttnn.sparse_matmul` masked mode with fp32 dest accumulation mis-accumulates across in0 blocks (pcc 0.85): use one
  in0 block over all of K.
- `models/tt_transformers` `tt_all_reduce` on a (1, N) mesh only reduce-scatters, and returns its input unreduced for
  cluster_axis=1: use stock `ttnn.all_reduce(cluster_axis=1, Linear)`.
- transformers' last `hidden_states` entry is not the last layer's output: fixtures hook the layers.
- Dev runner mounts the prefetcher fix (`cq_prefetch.cpp`) like the other ports' scripts.

## Serving (vLLM, dev)

`tools/serve_dev.sh` (MESH_DEVICE, SEQS, CTX): vLLM 0.26 + the TT plugin in the dev image, the port's bundle via
EXTRA_MODELS_DIR, `--hf-config-path hf_config --trust-remote-code` (vLLM 0.26 / transformers 5.12 do not know
inkling2_loop; the shim's arch is `TTInkling2LoopForConditionalGeneration`), `--reasoning-parser inkling
--tool-call-parser inkling` (same message tokens as Inkling v1; tool calls parse). `tt/generator_vllm.py`: engine-owned
slot caches, a row -> slot map for the plugin's condense remaps, host sampling (v0). `tools/smoke_serve.py`.

## Decode performance (P150, one die, bfp8 experts, bf16 KV; device time per traced step)

| change | B=1 | B=4 | quality (prefill path) |
|---|---|---|---|
| first batched decode | 126 ms | 197 ms | 0.970 / 0.996 / 0.010 |
| fused rms_norm, stacked block mix, merged gates, folded q/r scales | 113 ms | | 0.969 / 0.996 / 0.012 |
| bfp8 attention + lm_head, tile-layout conv states | 103 ms | 171 ms | 0.971 / 0.997 / 0.014 |
| fused MoE decode kernels (qwen36 moe_decode + ACT_INKLING), one fused-layout weight copy | 102 ms | 133 ms | 0.968 / 0.996 / 0.014 |
| route once per layer | 97 ms | 126 ms | decode path: 0.967 / 0.994 / 0.018 (sparse 0.970 / 0.997 / 0.023) |
| activations in L1, one-program batched conv step | 83.0 ms | 101.3 ms | |
| fused block-mix + message-gate-tail kernels | 72.6 ms | 90.6 ms | decode 0.968 / 0.996 / 0.018, prefill 0.968 / 0.996 / 0.014 |
| explicit 1D mcast matmul configs for every decode matmul (tt/dmm.py) | 57.8 ms | | |
| one-row rms_norm width-sharded on 8 cores | 51.0 ms | 67.5 ms | decode 0.970 / 0.996 / 0.015 |
| attention prep in one program (tt/aprep.py: head gather, l2 clips, rms, tau, sink, KV-write layout) | 42.5 ms | 58.8 ms | see below |
| message gate in one program (tt/gate.py: two cross-core rounds through a root core) | 41.1 ms | | decode (8 x 736) 0.9669 / 0.9965 / 0.0191 |

TP (same code, B = 4 traced): TP4 38.1 ms, TP2 43.5 ms; TP4 B = 1 device 34.9 ms (replicated glue + 96 all-reduces at
39 us + the replicated LM head dominate: next, the fabric all-reduce boundary and a vocab-sharded head).

Fused gate zones (tools/zones_gate.py, us): squares 6, root reduce #1 11.5, norm + partial down proj 6, root reduce #2
9, forget / gain 10, tail 10 (fp32 unpack-to-dest copies with per-copy re-inits ~300 cycles each). Next step for it:
one round (scale the partial projections by the rms factors after the reduce) and a tree reduce over the cores.

At one row most ops are bound by the 32-row tile, not by bytes (attention projections ~60 us each vs a 32 us
bandwidth floor; DRAM-sharded only wins at LoFi): wider batches are nearly free, single-user latency needs fused
kernels (layer glue: norms / mixes / gates / convs ~50 ms of the 97).

Fused MoE notes: the kernels' single-row sigb path routes wrongly (pcc 0.5): one row runs the 2-row path. Expert slices
must be 128..768 columns: sub-shards 3 x 384 (1 die), 384 + 256 (2), 384 padded (4). Down rows op: 3 readers per bank
(H = 2304 is 9 columns per bank) -> 16 rows max; wider batches use the sparse union path.

Decode-quality noise: the teacher-forced decode eval (8 texts x 256 steps) moves by ~0.5 pt top-1 and ~0.005 KL
between numerically equivalent variants (ulp-level rounding differences); code_mbpp alone swings KL 0.03 - 0.10.
The attention-prep kernel at 8 x 736 steps: top-1 0.9671 = ops path 0.9671, top-1 (conf) 0.9954 vs 0.9951, KL 0.0171
vs 0.0150 (all of the difference in code_mbpp / chat_french; with the ops path's r instead, KL 0.0108). The relative-bias
query r is the sensitive input: its per-head sums of squares must not go through a plain TF32 matmul (truncation
bias +1e-4 on the clip factor measurably moved quality); the kernel splits x^2 into hi + lo (exact).

Bandwidth floor (one die, bfp8): every application re-reads its layer's weights (attention ~30 MB, 8 experts ~67 MB),
~5.7 GB per token -> ~11 ms at 512 GB/s; 51 ms is ~22 %. Matmul notes (tools/bench_proj.py, traced, M = 32): the
default config is 2-5x slow at one tile row (k / v 34 -> 13 us, o 60 -> 24, gate down 104 -> 20 with a 1D mcast_in0
config on ~48 cores); DRAM-sharded is slower than 1D here. Small-N matmuls (mix logits N <= 4 tiles, gate down 8 tiles,
router) still run on a few cores (~20 us each): split-K next. One-row rms_norm: 29 us interleaved, 10 us width-sharded
on 8 cores incl. both reshards (tools/bench_norm.py).

## Serving (P150, vLLM, greedy, 256 output tokens, tools/bench_serve.py)

| concurrency | TTFT mean | decode tok/s / user | total tok/s |
|---|---|---|---|
| 1 | 1.71 s | 20.8 | 18.3 |
| 2 | 2.52 s | 20.5 | 34.1 |
| 4 | 5.47 s | 16.8 | 49.1 |
| 8 | 11.97 s | 13.4 | 65.0 |

(before the prefill MoE fix below.) After it, with decode sampling on device (SAMPLE_DEVICE=1: the qwen36 RowsTopK fused
top-32 twice inside the decode trace, the second pass on the rows shifted by the first's max so bf16 resolves the
candidates; host draw with vLLM semantics over the 32, tt/sampling.py; 0.5 ms at 1 row, 1.3 ms at 8):

| concurrency | TTFT mean | decode tok/s / user | total tok/s |
|---|---|---|---|
| 1 | 0.79 s | 20.9 (24.7 at temperature 1) | 19.7 |
| 2 | 1.18 s | 21.2 | 38.7 |
| 4 | 2.56 s | 18.2 | 61.6 |
| 8 | 5.63 s | 14.1 | 85.7 |

Single-user device time depends on the text (greedy repetitive text 47 ms / step, sampled 40 ms): expert placement?
(open). Decode width buckets (one trace per width, the narrowest covering the highest live
slot), logits untilized on device (readback 8.3 -> 1.1 ms). Prefill: the MoE's masked union path was 84 % of a chunk;
the sorted dispatch (MiniMax / Qwen3.5 moe_dispatch, grouped sparse matmuls) takes a 128-row chunk 1832 -> 733 ms,
256 rows 1031 ms (prefill quality 0.969 / 0.996 / 0.016). Next for prefill: the multi-row block mix / message sums
(~165 ms of the 128-row chunk), then batching prefills across users.

## Speculative decoding: DSpark (the checkpoint's mtp.safetensors)

Not in transformers. vLLM serves it from 0.35 nightly (`"method": "dspark", "num_speculative_tokens": 7`), SGLang from
PR #51926 (`--speculative-algorithm DSPARK`, 8 drafts verified). Reference code (SGLang PR head
zcnrex/sglang@chunan/inkling-2-mini-day0) saved under ~/inkling2-work/ref/sglang: srt/models/dspark.py (RNNHead Markov
head = markov_head_type rnn_hidden; context KV from the target's hidden states), srt/models/dflash.py (DFlash base),
kernels/ops/speculative/dspark/markov_walk.py / markov_sample.py, srt/models/inkling2.py (taps at visits 5, 23, 41 +
the final normed state, 48).

DSpark acceptance (reference/dspark_ref.py, CPU, 8 held-out texts x 40 anchors; accepted drafts of 7 per pass):
teacher-forced greedy 1.20 (base logits without the Markov head 0.70), sampled at temperature 1 (the checkpoint's
default; expected acceptance sum_k prod alpha) 1.21; code / problems 1.6 - 2.1, human chat 0.4 - 0.6. Exact greedy
(the target's own continuation) on math: 2.05 vs 1.30 teacher-forced.

## Next

1. TP2 / TP4 validation of the fused MoE + batched decode; then per-profile step times at B = 1 .. 32.
2. Fused glue kernels for single-user latency; DSpark speculative decoding (CPU reference + acceptance first).
3. p150 memory plan at bfp8 experts; device sampling; profiles, card, package, publish.
