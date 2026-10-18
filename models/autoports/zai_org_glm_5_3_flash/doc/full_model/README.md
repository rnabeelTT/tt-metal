# GLM-5.3-Flash (E192) full model

Stage 6 of the tt-model-bringup flow. `tt/model.py` (`Glm53Model`) runs the full text model on the (1, 4) mesh of a
p300x2. `tt/generator.py` (`Glm53Generator`, `build_generator`) is the readiness and serving contract.

## Structure

- Embedding: the table is hidden-sharded. Die d looks up columns `[d * C/4, (d + 1) * C/4)`. The 4 mHC streams start
  as copies of the embedding, in the stream-interleaved layout of `tt/multichip_decoder.py`.
- 45 `MultichipDecoder` layers (policy `opt`): 34 KDA layers (3 with a dense MLP) and 11 DSA layers.
- Head: stream mean, all-gather, final RMSNorm, LM head (bfp8, vocab-sharded). The logits stay sharded over the vocab.
- Sampling: `models.common.sampling.SamplingGenerator` on device. The sampler takes a multiple of 32 rows. With fewer
  slots (S), the model zero-pads the logits rows to `SR = ceil32(S)`.
- Prefill: one user per call, in chunks of the constructed lengths (128, 2048). KDA layers carry their state from chunk
  to chunk. DSA layers write the paged latent cache.
- Decode: all S slots in one traced step. The per-step DSA / KDA inputs come from the position tensor on the device
  (`tt/decode_inputs.py`), and the trace advances the positions. The sampler trace writes the token into the buffer that
  the model trace reads, so steady-state decode needs no host input.

## Warm-up order

`_warm` compiles everything before it captures a trace:

1. One prefill per constructed length. Prefill builds persistent constants lazily. When a prefill first ran after the
   capture, these constants landed in freed trace-intermediate memory. The next decode replay overwrote them, and the
   prefill after the replays hung (2026-10-16, `tools/model_step_probe.py` bisect).
2. One eager decode step with every slot inactive, and the sampler programs on the token buffer.
3. Model trace capture, then sampler trace capture on the model trace's logits buffer.

## Readiness results (full 45 layers, TP4, 4 slots, trace region 256 MiB)

Reference: `readiness_aime24_chat.refpt`. HF `Glm5NextTextModel` on CPU with the same E192 expert set and lazy FP8
experts (`tools/hf_full_ref.py`). The CPU model matches the per-layer reference bit for bit at all 45 layers. The
prompt is AIME24 #0 with the chat template: 161 prompt tokens, 100 generated.

| Check | top-1 | top-5 | top-100 | Gate | Report |
|---|---|---|---|---|---|
| Teacher forcing | 93 % | **100 %** | **100 %** | top-5 >= 98 %, top-100 = 100 % | `runs/teacher_forcing.json` |
| Prefill check | 92 % | **100 %** | **100 %** | same | `runs/prefill_check.json` |

HF's own greedy decode agrees with its teacher-forced top-1 at 96 % of the positions, so 93 % at bfp4 experts is
within the expected range.

Free-running checks (`autoregressive/VERDICT.md`, `qualitative/VERDICT.md`):

- Autoregressive (stock prompt, 256 tokens): no degenerate output (`check_degenerate_output.py` exit 0). Two runs give
  bit-identical tokens. TT leaves HF's path at token 1, which is a near-tie inside HF itself (15.31 / 15.06 / 14.19).
  HF scored over TT's tokens agrees at top-1 97.7 % and top-5 99.6 %. The later repetition is HF's own top-1 at every
  position (margin >= 1.75 logits).
- Qualitative (6 chat prompts, up to 2048 tokens): coherent, on topic, correct facts and code. TT follows HF for the
  first 10 to 75 tokens of each answer. The partial French translation (q5) is HF's behaviour too.

## Performance (same runs)

| Metric | Value |
|---|---|
| TTFT, 161-token prompt | 1.25 s |
| Decode, 4 slots, traced, on-device sampling | 10.5 t/s/u (95 ms per step) |
| Model build, no weight cache | 22 min |

Stages 5 and 7 optimize decode. Stage 7 adds traced prefill.

## Memory and context

DRAM is the limit on this model. Measured per die (`memory_budget.json`, `memory/`, `ttnn.get_memory_view`):

| Item | Bytes |
|---|---|
| DRAM before the trace region | 34,178,731,008 |
| Weights and model tensors after load | 32,123,120,128 (29.92 GiB) |
| Decode trace (45 layers + greedy sampler) | 47,972,352 (46 MiB); default region 64 MiB |
| KDA state per slot (34 layers) | 35.65 MB |
| DSA latent + pooled keys per token (11 layers, replicated) | 11,968 |
| 2048-row prefill peak above layer entry (MoE, after the row split) | 297 MiB (was 417 MiB) |

The MoE prefill runs its routed experts in row blocks of `GLM53_MOE_SORTED_ROWS` (default 1024). The DSA indexer
uses blocks of `GLM53_DSA_INDEX_BLOCK_MB` (default 64). Both are pure memory changes: last-row logits of a 6221-token
prefill are bit-identical over the settings.

Serving budget (trace region 64 MiB, reserve 128 MiB, prefill workspace 0.55 GB):

| Slots | Recommended max_model_len |
|---|---|
| 1 | 104,448 (verified) |
| 4 | 94,208 |
| 8 | 81,920 |
| 16 | 57,344 |
| 32 | 8,192 |

Capacity evidence: 1 slot, a 104,515-token prompt (51 x 2048 + 67) prefilled in 169.7 s. Then 24 traced decode steps
at positions 104,515 .. 104,538 copied the repeated passage 24 / 24. 183 MB stayed free per die at the prefill peak.
`doc/context_contract.json` records supported_context 104,448 against the advertised 1,048,576. The DSA cache is
replicated on each die. Sharding the latent pool by block over the 4 dies (about 420K tokens), plus a bfp8 latent and
a host embedding lookup (about 1.0M at 1 slot), is the documented path back to 1M.

## Commands

```
export GLM53_TP=4 GLM53_MAX_BATCH=4 GLM53_MAX_SEQ_LEN=4096   # trace region default 64 MiB
python -m models.autoports.zai_org_glm_5_3_flash.tools.readiness teacher-forcing \
  --reference models/autoports/zai_org_glm_5_3_flash/readiness_aime24_chat.refpt
python -m models.autoports.zai_org_glm_5_3_flash.tools.readiness prefill-check \
  --reference models/autoports/zai_org_glm_5_3_flash/readiness_aime24_chat.refpt
python -m models.autoports.zai_org_glm_5_3_flash.tools.full_model_checks run   # autoregressive + qualitative
pytest models/autoports/zai_org_glm_5_3_flash/tests/test_model_probe.py           # reduced model, 3 layers
```
