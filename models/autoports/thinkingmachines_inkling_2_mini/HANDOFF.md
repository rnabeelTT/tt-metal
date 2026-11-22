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

## Next

1. p150 memory plan at bfp8 experts (~26 GB): embedding on host, lm_head / attention bfp8 (each gated by the eval),
   bfp4 only in the most tolerant expert layers if KV room is still short.
2. Serving engine: multi-user decode (per-item SDPA, batched conv states), traced decode, device sampling, MTP drafter
   (`mtp.safetensors`), vLLM generator + plugin bundle, profiles, packaging, publish.
