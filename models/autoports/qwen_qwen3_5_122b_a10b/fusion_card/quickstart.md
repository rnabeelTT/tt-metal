### What blackhole-fast changes
This is [raahemnabeel/qwen3-5-122b-a10b](https://huggingface.co/raahemnabeel/qwen3-5-122b-a10b) with its decode step
rebuilt around fused kernels, ported from the Ornith-1.5-35B-A3B fusion
([raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast](https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast);
Ornith serves through this model's code). Same checkpoint, same vLLM plugin build, same profiles and serving recipe;
prefill is unchanged (TTFT equal or lower). The changes:

1. **Router.** The decode router matmul runs as one 1D matmul, and the 1-token top-8 + softmax + shared-gate sigmoid run
   on the RISC cores in one op instead of a 7-op chain.
2. **Fused MoE decode.** Routed + shared experts in three ops (route, gate|up + SwiGLU, down) that stream each selected
   expert's weights in place from the existing DRAM layout (no duplicate copy), for every decode width from 1 to 32
   rows, the MTP verify rows included (several users' experts as one union, each weight column read once for all rows).
3. **Replicated-residual decode boundary.** The decode residual stays replicated on every die: each mixer / MoE output
   is one all-reduce + add + local RMSNorm (two collectives per layer instead of four, no gathers).
4. **Gated DeltaNet decode mixer** as two programs (causal conv; gates + norms + recurrent state update + gated norm),
   each head's value dimension split across up to four cores.
5. **Attention decode ops** around the QKV matmul, cache update and SDPA decode (q / k norm + RoPE + gate in one prep op,
   output gating in one post op).
6. **Fabric layer boundary** (1-4 users): the all-reduce, residual add and RMSNorm as one op over the fabric, with the
   next layer's input projection (or the MoE router) inside the same program.
7. **Sampler top-k over the live rows only** (1-8 users), instead of all 32 slot rows.

MTP speculative decoding is off: its verify pass runs the unfused path, and the fused plain decode step is faster
(1 user on chat text: 132 tok/s, vs 45.9 for the original with MTP).

Numerics: the routed and shared expert weights are stored rounded to 5 significant bits (what the LoFi expert matmul
reads), activations / routing weights rounded to the 7 bits the matmul's other operand reads; the fused boundary and
DeltaNet ops sum in a different order than the op chain they replace. Output quality is checked below.

### Use with your client
Point an OpenAI-compatible client at `http://127.0.0.1:20000` (or the port `tt-model serve` printed) with model
`Qwen/Qwen3.5-122B-A10B`. Thinking is ON by default: the reasoning arrives in the message's `reasoning` field (vLLM
0.26; older clients know it as `reasoning_content`) and the answer in `content`. Give thinking requests a large
`max_tokens` (thousands): when the budget ends inside the think block, `content` is empty. Turn thinking off per
request with `"chat_template_kwargs": {"enable_thinking": false}` for short, direct answers (the configuration the
release gates and the published latency numbers were measured in). Without sampling parameters a request uses the
checkpoint's defaults (temperature 0.6, top_p 0.95, top_k 20); for thinking on constraint-heavy prompts the model
card's general preset (temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5) avoids repetition.
Tool calls use the `qwen3_xml` parser; images (up to 4 per request, resized to at most 4 MP) as `image_url` parts.

### Profiles - which one for which use case
`tt-model serve raahemnabeel/qwen3-5-122b-a10b-blackhole-fast --profile <name>` (p300x2 names below; the same four exist
as `p150x4`, `p150x4-burst`, `p150x4-longctx`, `p150x4-text` for four p150 boards):
- `p300x2` (default) - interactive assistants and agents: one or a few users at a time who want the fastest answer.
  A single user gets about 132 tok/s on chat text; images and tool calls work; thinking on.
- `p300x2-burst` - many short prompts arriving together (chat front-ends with dozens of users, batch classification,
  extraction jobs): up to four prompts are prefilled per pass, so a 32-request burst gets its first token in 3.8 s
  instead of 6.3 s with the same decode speed. Choose it when throughput matters more than bit-exact reproducibility
  of a request's first token across different batch compositions (seeded runs may differ by what arrived together).
- `p300x2-longctx` - document and codebase work where very long prompts (tens of thousands of tokens) mix with short
  ones: prefill runs in 4096-token chunks shared between prompts, so a short question is not stuck behind a 100K-token
  upload (about 132 tok/s for a lone user); running users still pause while prefill chunks run.
- `p300x2-text` - text-only deployments: the vision tower is not loaded, giving a 25 % larger KV pool (1.8 M pooled
  tokens for 32 users) for longer conversations at high occupancy; image requests are rejected at the front end.
