### What this is
[ornith-ai/Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B) has the Qwen3.5-122B-A10B architecture
(`Qwen3_5MoeForConditionalGeneration`: 256 experts top-8 + shared expert, 48 layers of Gated DeltaNet and gated
attention, vision tower, MTP head; every config shape and all 1949 tensor names match). It is served here by the code
of the Qwen3.5-122B-A10B Blackhole bundle ([raahemnabeel/qwen3-5-122b-a10b](https://huggingface.co/raahemnabeel/qwen3-5-122b-a10b)),
unchanged: same fused decode step, same profiles and serving recipe; only the checkpoint differs. The decode step:

1. **Fused MoE decode.** Router top-8 + softmax + shared-gate in one op; routed + shared experts in three ops that
   stream each selected expert's weights in place, for every decode width from 1 to 32 rows.
2. **Layer boundary.** The decode residual stays replicated on every die; for 1-4 users the all-reduce, residual add
   and RMSNorm run as one fabric op with the next layer's input projection (or the MoE router) inside its program.
3. **Gated DeltaNet decode mixer** as two programs (causal conv; gates + norms + recurrent state update + gated norm).
4. **Attention decode ops** around the QKV matmul, cache update and SDPA decode.
5. **Sampler top-k over the live rows only** (1-8 users).

MTP speculative decoding is off: its verify pass runs the unfused path, and the fused plain decode step is faster.

Numerics: the routed and shared expert weights are stored rounded to 5 significant bits (what the LoFi expert matmul
reads; bfp4 routed experts as in the Qwen3.5-122B bundle's precision policy), activations / routing weights rounded to
the 7 bits the matmul's other operand reads. Output quality is checked below.

### Use with your client
Point an OpenAI-compatible client at `http://127.0.0.1:20000` (or the port `tt-model serve` printed) with model
`ornith-ai/Ornith-1.5-122B`. Thinking is ON by default (the model is a reasoning model): the reasoning arrives in the
message's `reasoning` field (vLLM 0.26; older clients know it as `reasoning_content`) and the answer in `content`.
Give thinking requests a large `max_tokens` (thousands): when the budget ends inside the think block, `content` is
empty. Turn thinking off per request with `"chat_template_kwargs": {"enable_thinking": false}`. Without sampling
parameters a request uses the checkpoint's defaults (temperature 0.6, top_p 0.95, top_k 20); the model card recommends
temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5 for general tasks and temperature 0.6 for precise coding.
Tool calls use the `qwen3_xml` parser; images (up to 4 per request, resized to at most 4 MP) as `image_url` parts.

### Profiles - which one for which use case
`tt-model serve raahemnabeel/ornith-1.5-122b --profile <name>` (p300x2 names below; the same four exist as
`p150x4`, `p150x4-burst`, `p150x4-longctx`, `p150x4-text` for four p150 boards):
- `p300x2` (default) - interactive assistants and agents: a single user gets about 132 tok/s on chat text; images and
  tool calls work; thinking on.
- `p300x2-burst` - many short prompts arriving together: up to four prompts are prefilled per pass (faster first token
  for bursts, same decode speed); a request's first token can then depend on what arrived with it.
- `p300x2-longctx` - very long prompts mixed with short ones: prefill runs in 4096-token chunks shared between prompts
  (about 132 tok/s for a lone user); running users still pause while prefill chunks run.
- `p300x2-text` - text only: the vision tower is not loaded, giving a larger KV pool; image requests are rejected.
