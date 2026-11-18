# Hy4-flash on one QB2: running log

Model: [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash), official BF16 checkpoint (440 GB, 148 shards).
220B parameters (155.5B backbone + 63.5B Engram + 0.43B ViT + 0.85B DFly2 drafter), ~11.3B active per text token.
Reference implementation: transformers PR 50903 at `ManaEstras/transformers@c2ec850510a37978ea83df226589623d5f2be776`
(`src/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py`; a copy lives in `~/m27-work/hy4/`). The DFly2 drafter,
the `hy_v4` reasoning / tool parsers and the serving recipe exist only in vLLM PR 68296.

Branch `raahem/hy4-flash-hybrid` (fork rnabeelTT/tt-metal), off `hybrid-moe-starter` (a76f430eeea). Port:
`models/autoports/tencent_hy4_flash/` (layout follows the Inkling port: own engine + vLLM adapter).

## Decisions (2026-11-18, approved)

- Day 0: text only, one user, no drafter (SPEC=0); DFly2 is the fast follow-up. Every device op is written for B rows
  so B = 2..4 users can be turned on once they pass the same quality gate. Vision later.
- Context: the maximum (1,048,576) as long as short sessions lose nothing (speed or quality).
- Quality is judged against the CPU BF16 reference (teacher forced: top-1, KL), never against an earlier TT run.

## Why this is not a hybrid-expert model

41 MoE layers x 512 routed experts, each in a 1536-wide latent space: gate/up `[3072, 1536]` (2 x 1536 out, 1536 in)
and down `[1536, 1536]` = 7.08M weights. All 20,992 experts in bfp4 (0.5625 B/w) = 83.6 GB = 77.8 GiB: they ALL fit on
the 4 dies with the rest of the model (~8 GiB at 8-bit) and the KV cache (MLA latent 576 + indexer key 128 per token on
the 12 sparse layers: ~9 KB/token, 1.2 GB at 128K, ~9 GB at 1M; the 30 SWA layers keep 512 positions). ~25 GiB is left
for higher-precision experts (e.g. bfp8 downs) and buffers. So: no CPU experts, no per-layer host round trip; decode
can be one trace per step.

The host holds only the Engram tables (layers 1, 11, 21, 31; 16 hash heads x 96 dims each; 165M rows per layer;
63.4B params, 118 GB in BF16). The hash ids depend only on token ids (`HYV4FlashEngramLayer.hash_ngrams`), so the
host computes them and the per-head gathered rows ahead of each pass and uploads `[T, 16 x 96]` per Engram layer
(~3 KB per token per layer). The model reads the rows through a per-vector FP8 e4m3 round trip
(`_fp8_round_trip`, scale = amax / 448 per 96-wide head vector), so storing each row as FP8 + its bf16 scale is
lossless: ~62 GB of host RAM. The rest of the Engram layer (per-head RMS norms, value / key projections, the
per-stream gates) runs on the device.

## Architecture notes (from the reference code)

- Residual: 4 hyper-connection streams of 3072 (`hc_mult` 4). Each sublayer: `HyperConnection` (fp32): `mixes =
  (flat @ fn.T) * rsqrt(mean(flat^2) + eps)` over the flattened 12288; `pre = sigmoid(mix_pre * s0 + b_pre) + hc_eps`,
  `post = hc_magnitude(2.0) * sigmoid(mix_post * s1 + b_post) + hc_eps`; sublayer input = sum_i pre_i * stream_i;
  output streams = post_i * sublayer_out + stream_i. Final: `HyperHead` (4 mixes, sigmoid + eps, weighted sum), then
  `model.norm`, LM head (config asks for an fp32 LM head).
- Layer 0 dense MLP (13824) with `post_fc2_norm`; layers 1-41 MoE: sigmoid router + `e_score_correction_bias`, top-16
  of 512 (n_group 1), weights renormalised then x 3.998; experts get `routed_expert_down_proj(x)` (3072 -> 1536), SwiGLU
  with clamps (gate <= 10, |up| <= 10), outputs summed then `routed_expert_up_proj` (1536 -> 3072); plus one shared
  expert (3072 -> 1536 -> 3072, no clamp) on the full-width input; `combine_output_norm` closes the block.
- Attention: gated MLA, q_lora 1024, kv_lora 512, qk 192 nope + 64 rope, v 256; `gate_proj` (sigmoid, per head x 256)
  multiplies the attention output; learned sinks per head; `post_proj_norm` after `o_proj`. Sparse layers (0, 1, 5, 9,
  ..., 41: 12 of them): 32 heads, rope theta 1e7, compressed indexer. SWA layers (30): 64 heads, theta 1e4, window 512.
  RoPE is GLM-style interleaved on the last 64 dims; for text the 3 MRoPE sections carry the same position, so it is
  plain 1-D RoPE.
- Compressed indexer (sparse layers): token key = RMSNorm(W_tk x) (128); groups of 4 tokens pooled with an fp32 gated
  softmax (+ `intra_group_ape`), RMSNorm; group keys carry the RoPE phase of their first token. Query = W_q q_resid
  (16 heads x 128, RoPE on the last 64). score(g) = sum_h w_h * relu(q_h . k_g), w = W_hw x * 16^-0.5 * 128^-0.5.
  A query keeps the top 511 complete earlier groups + its own group (2048 tokens). Dense below 2048 total tokens
  (`kv_length <= candidate_topk`): short sessions never run the indexer.
- Engram: added to the 4 streams before the attention HC of layers 1, 11, 21, 31 (`engram_layer_ids` 2, 12, 22, 32
  are 1-based).

## Base and device port (2026-11-18)

- The branch merges upstream main (0fe57024bca) into the starter: it brings the upstream Qwen3.8-Flash-Next port
  (`models/demos/blackhole/qwen38_flash_next`: QSA = the same 4-token-block / 2048-token sparse attention, a 4-stream
  gated residual, an n-gram table read by the host, 512-expert BF4 MoE via `moe_compute`, fused decode kernels) and the
  DeepSeek-V4 pieces (`sparse_sdpa`, `indexer_score_dsa`, `topk_large_indices`, `update_padded_kv_cache`). Conflicts:
  kda / tracy took upstream, qwen36 kept the starter's copy (the M2.7 port uses it), sparse matmul keeps both the
  grouped MoE factory and upstream's prefetcher factory. Host build: `build_metal_hy4.sh` (untracked copy of
  build_metal.sh that points CPM at tarballs in ~/m27-work/cpm_src: GitHub git clones ran at ~70 KB/s here).
- Port `models/autoports/tencent_hy4_flash/` (day 0 = eager, one user, no drafter):
  `tt/config.py`, `tt/weights.py` (lazy safetensors), `tt/common.py` (uploads, all-reduce, RMSNorm),
  `tt/hc.py` (hyper-connections, fp32), `tt/rope.py` (bf16 tables exactly as HF, fp32 rotation by a +-1 matmul),
  `tt/attention.py` (absorbed MLA; row-major 608-wide latent cache [c 512 | k_rope 64 | sink 32] written with
  `slice_write`; every attention goes through `sparse_sdpa` with host-built index lists; the learned sink is a
  reserved cache row scored against q's two sink columns s_hi + s_lo), `tt/indexer.py` (token keys read back, groups
  pooled on the host in fp32 exactly as the reference, mirrored to a device cache; device scoring + top-512),
  `tt/moe.py` (latent MoE via sparse_matmul gather / chunk-masked paths, clamped SwiGLU, TP over the expert
  intermediate; dense MLP), `tt/engram.py` (device half), `tt/model.py`, `tt/engine.py`, `tt/generator_vllm.py`,
  `tools/generate.py`, `tests/test_primitives_device.py`, `tests/test_layer_device.py`.
- Layout: residual streams replicated on all 4 dies (the HC math is then local); one all-reduce per sublayer.
- Next after correctness: traces (position-dependent inputs from the host as one-hot / index tensors), fused HC
  (qwen38's gr_* kernels), device-side index lists, the DFly2 drafter, B = 2..4.

## Prefill (2026-11-18, branch raahem/hy4-prefill)

Borrowed from M2.7 (expert-sorted MoE) and tt-metal PR 61548 (rows split over chips, larger chunks):
- `tt/prefill_moe.py` SortedExperts: moe_sorted_dispatch + grouped sparse matmuls (padding groups skipped) + gather /
  gather-add rows, Hy4's clamped SwiGLU in the 1536 latent. MoE 327 -> 17 ms per layer at 2048 rows; sub-chunks of
  2048 rows (HY4_MOE_SUB: the dispatch admits ~3K rows at top-16 of 512).
- Row split (HY4_ROW_SPLIT_MIN=128): `Hy4Layer.forward_split` keeps each die's quarter of the rows for the HC / norms /
  Engram; sublayer inputs all-gathered, partials reduce-scattered (`Hy4Attention.forward_part`, `Hy4MoE.forward_part`,
  `Hy4DenseMLP.forward_part`). Prefill row counts are multiples of 128.
- Index lists: position-only lists (SWA, dense) uploaded once per chunk and shared by the layers
  (`Hy4Attention._idx_device`); selecting chunks (p0 >= 2048) build them on the device from the top-512
  (`Hy4Indexer._expand` + the per-chunk tail `_idx_tail`): no readback.
- Group pooling on the device for prefill (`Hy4Indexer._pool_prefill_device`, HY4_PF_DEVICE_POOL=1; decode's fp32
  math): no token-key readback. The host views decode's `init_stage` reads (`mirror`, `tk_hist`, `open_keys`) are
  properties now, read back lazily from the device.
- `Hy4Model.prefill(ids)`: the dense-regime chunk (to 2048) first, then HY4_PREFILL_CHUNK chunks (<= the model's
  `chunk`, which sizes the SWA rings): the engine should call it instead of looping forward() in fixed chunks, and build
  the model with chunk=8192.
- Head only on the rows asked for (`Hy4Model._head_split`, 32-row blocks): the full gather before the head OOMed
  (3 GiB) at 8K / 16K chunks.
- Traced first chunk (`Hy4Model._prefill_traced`, HY4_PF_TRACE=1, buckets 128..2048 at p0 = 0; needs
  HY4_TRACE_REGION ~1e9): last-row logits bit-equal to eager (PCC 1.0, same top-1 at 200 / 1000 / 2000) but no
  faster (0.389 vs 0.38 s, 0.94 vs 0.94, 1.68 vs 1.68): prefill is device-bound, so it is off by default. The time is
  data movement (repeat / slice / typecast / sum, the fp32 HC), not dispatch.
- Not touched (decode owner): `tt/hc.py` (in prefill `ttnn.sum(dim=1)` over the streams costs transpose + reduce +
  transpose on [1, 4, T/4, 3072] fp32; `ttnn.experimental.fast_reduce_nc(dims=[1])` is one op).

## Log

- 2026-11-18: plan approved; branch + worktree `~/hy4-flash`; BF16 download started (`~/m27-work/logs/hy4_download.log`).
- 2026-11-18 (serving worker): vLLM 0.26 front-end + DFly2 reference.
  - vLLM 0.26.0 / transformers 5.12.1 (image 2f9b96d1f881) know neither `hy_v4_flash` nor the `hy_v4` parsers.
    `models/autoports/tencent_hy4_flash/serving/` adds them: `sitecustomize.py` (post-import hooks in every process:
    config class into `AutoConfig`, `hy_v4` reasoning / tool parsers as lazy vLLM parsers; nothing imported
    eagerly; keeps the HYB_ENABLE overlay hook), `hy4_config.py`, `hy_v4_reasoning_parser.py` and
    `hy_v4_tool_parser.py` (vLLM PR 68296 @3f21671, imports adapted; the tool parser has no xgrammar structural tag
    on 0.26, so `tool_choice` required / named take vLLM's generic JSON-schema path; auto parses the native
    `<tool_calls:opensource>` tokens). Bundle: `vllm_bundle/hy4_flash/vllm_metadata.json`, arch
    `HYV4FlashForConditionalGeneration` (the checkpoint's own: no `--hf-overrides`), main class
    `models.autoports.tencent_hy4_flash.tt.generator_vllm:HYV4FlashForConditionalGeneration` (to be written: text
    only, `supports_multimodal = False`). vLLM derives max_model_len 1,048,576, vocab 120,832, eos 120025,
    sliding_window 512 (from text_config), head_size 64 (`head_dim` field; the engine owns the caches anyway),
    sampling defaults T 0.9 / top_p 1 from generation_config.json. Serve with `PYTHONPATH=<port>/serving:...`,
    `--reasoning-parser hy_v4 --tool-call-parser hy_v4 --enable-auto-tool-choice`.
  - Tests (CPU, in the image; the image has no pytest, mount a pure-python copy): `serving/tests/` 20 passed
    (registration, config, reasoning thinking / no_think, tool calls typed by schema, malformed -> content, streaming
    at 1 / 2-3 / 6 tokens per step equals non-streaming, chat-template structure).
  - Chat template: system turn always present, ends `<｜reasoning_mode:opensource｜>reasoning_effort:{high|max|no_think}`
    (anything else raises: OpenAI `reasoning_effort` low / medium requests would fail); generation prompt ends with
    `<think:opensource>` (thinking) or `<think:opensource></think:opensource>` (no_think); earlier assistant reasoning
    is dropped unless tools are present.
  - DFly2 (`models/experimental/hybrid_moe/dfly_hy4_ref.py`, tests `test_dfly_hy4_ref.py`: 10 passed incl. the real
    weights): one parallel pass of 6 rows [bonus, MASK x 5] at anchor..anchor+5 through 5 dense MLA layers (target
    sparse-layer attention: sinks, sigmoid gate, post-proj norm, theta 1e7) that read AND write the target's latent
    caches of layers 25/29/33/37/41 and attend to that layer's indexer recall for the last accepted row (+ the causal
    block); row 0 gets `mtp_fusion` with the target's final hidden; then 5 sequential `hidden_correction` + LM-head
    steps (greedy or temperature sampling). No prefill role. `confidence_head` unused (vLLM ignores it).
    Engine interface: per verify pass keep, for the last accepted row, the final normalized hidden and the recalled
    positions of layers 25/29/33/37/41 (<= 2048 each; all positions < anchor below 2048 tokens); the draft's block
    latents may go to a 6-row scratch instead of the target cache (equivalent: the next verify overwrites them).

### 2026-11-18: CPU BF16 reference + Engram host store (reference worker)

- venv `~/m27-work/venv-hy4` (torch 2.15 cpu, transformers 5.24.0.dev0 = PR 50903 @c2ec850).
- `engram_store.py`: `convert` writes the 4 tables as FP8 e4m3 rows + a bf16 scale per row to
  `~/.cache/hybrid-moe/hy4_engram/` (`L<li>_fp8.npy` 15.8 GB + `L<li>_scale.npy` 0.33 GB each, 64.7 GB total; ~1-4 min per
  table). `EngramRows(store_dir, ckpt_dir).rows(li, ids, p0)` returns the Engram layer's `embeddings` after the per-head
  norms and the flatten (bf16 [T, 1536], the value_proj / key_projs input); keeps the last 2 tokens per sequence itself.
  `python engram_store.py test`: hash ids, n-gram tokens (incl. incremental), rows after the fp8 round trip AND the
  normed rows bit-identical to the HF code on all 4 layers (4096 random tokens, 40 pads). Two traps found:
  (1) the reference returns `x + (q - x)`, which is +0 where q is -0: dequant adds +0.0; (2) `nn.RMSNorm(96, eps=None)`
  on bf16 input computes in fp32 with eps = finfo(**fp32**).eps = 1.19e-7 (not bf16's 7.8e-3) and applies the weight
  in fp32 before one rounding. Lookup: 0.4 ms (1 token, 4 layers, warm), EngramRows 0.9 ms (1 token) / 1.3 ms (3);
  8K tokens 0.29 s warm. Engram mask: only pad (120002) / image / video tokens zero the WHOLE Engram output.
- `ref_hy4.py`: runs HF's own `HYV4FlashTextModel.forward` (masks, indexer, caches) with the non-expert weights in RAM
  (14.8 GB anon) and the experts / Engram tables memory-mapped from the shards (page cache). `tf` (teacher forced,
  chunked through the HF cache, REF_GOLD streams) and `gen` (greedy). Attention: REF_ATTN=fp32 (default: HF's eager
  math with fp32 q/k/v; HF's own softmax runs in bf16) | hf. Router fp32 as HF; LM head fp32 as HF.
- `compare_hy4.py`: device `lp_<tag>_<name>.npy` [T, V] or `dev_<tag>_<name>.npz` (top1, lp_at_ref) vs ref_<name>.npz.
- Reference outputs `~/m27-work/hy4/ref/`: texts.json (8 held-out 1000-token texts, quality_texts.py with the Hy4
  tokenizer), prompts.json (4 chat prompts, reasoning_effort no_think), gen.json, tf_texts.json (the 8 texts + the 4
  prompts with their greedy continuations), ref_<name>.npz, routing_T<T>.npz, gold_<name>.pt (streams entering every
  layer [42, T, 4, 3072] + final hidden, for math_gsm8k and the 4 chat sequences).
- Greedy (no_think): "Hello! I'm Hunyuan, a large model developed by Tencent. ...", "72 km/h" (correct), a Fibonacci
  function, fluent Chinese on the Great Wall. Teacher forced on its own continuations: top-1 0.96-1.0, NLL 0.07-0.09.
  Held-out texts (nll / top-1): gsm8k 3.52/0.60, mbpp 5.04/0.47, en 5.55/0.37, zh 5.16/0.43, fr 4.40/0.49,
  py file 2.02/0.61, markdown 3.42/0.46, livecodebench 4.29/0.39: a very peaked model (top-1 p 0.84 on gsm8k), so the
  misses cost 10-19 nats each; top-1 / KL are the metrics to watch, not NLL.
- Timing (8 cores, warm page cache): 8 x 1000 tokens 407 s (9.7 s per layer); greedy 1.1-1.5 s/token warm (7.7 cold).
- Checks: layer 0 (indexed attention + compressed indexer) in float64, one 3000-token pass vs 1024-token chunks through
  the HF cache: bit-identical (0.0) incl. past 2048 where the indexer selects. Noise floor of the BF16 reference:
  fp32-vs-bf16 attention softmax + batch 1 vs 8: top-1 agreement 0.928 (gsm8k) / 0.96-1.0 (chat); full model single
  pass vs chunked over 3000 tokens: 0.97 / 0.91 / 0.86 (positions 0-1K / 1-2K / 2-3K). A TT port within ~0.9 top-1
  of this reference is at the reference's own noise level.
- Note: config.json spells the attention types `deepseek_sparse_attention` / `sliding_attention`; transformers
  remaps the first to `indexed_attention` (its rope_parameters key too). `mrope_interleaved` is ignored with a warning
  (the interleave is always applied); for text the 3 M-RoPE planes are equal, so it is plain 1-D RoPE.
