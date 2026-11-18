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
