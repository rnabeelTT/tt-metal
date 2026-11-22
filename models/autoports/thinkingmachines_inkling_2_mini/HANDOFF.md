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

At one row most ops are bound by the 32-row tile, not by bytes (attention projections ~60 us each vs a 32 us
bandwidth floor; DRAM-sharded only wins at LoFi): wider batches are nearly free, single-user latency needs fused
kernels (layer glue: norms / mixes / gates / convs ~50 ms of the 97).

Fused MoE notes: the kernels' single-row sigb path routes wrongly (pcc 0.5): one row runs the 2-row path. Expert slices
must be 128..768 columns: sub-shards 3 x 384 (1 die), 384 + 256 (2), 384 padded (4). Down rows op: 3 readers per bank
(H = 2304 is 9 columns per bank) -> 16 rows max; wider batches use the sparse union path.

## Speculative decoding: DSpark (the checkpoint's mtp.safetensors)

Not in transformers. vLLM serves it from 0.35 nightly (`"method": "dspark", "num_speculative_tokens": 7`), SGLang from
PR #51926 (`--speculative-algorithm DSPARK`, 8 drafts verified). Reference code (SGLang PR head
zcnrex/sglang@chunan/inkling-2-mini-day0) saved under ~/inkling2-work/ref/sglang: srt/models/dspark.py (RNNHead Markov
head = markov_head_type rnn_hidden; context KV from the target's hidden states), srt/models/dflash.py (DFlash base),
kernels/ops/speculative/dspark/markov_walk.py / markov_sample.py, srt/models/inkling2.py (taps at visits 5, 23, 41 +
the final normed state, 48).

## Next

1. TP2 / TP4 validation of the fused MoE + batched decode; then per-profile step times at B = 1 .. 32.
2. Fused glue kernels for single-user latency; DSpark speculative decoding (CPU reference + acceptance first).
3. p150 memory plan at bfp8 experts; device sampling; profiles, card, package, publish.
