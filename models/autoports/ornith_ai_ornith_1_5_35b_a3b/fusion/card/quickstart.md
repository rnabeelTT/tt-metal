### What blackhole-fast changes
This is [raahemnabeel/ornith-1.5-35b-a3b](https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b) with its decode step
rebuilt around fused kernels. Same checkpoint, same vLLM plugin build, same profiles and serving recipe; prefill is the
original's (TTFT equal or lower). The changes, in the order they were made (each measured, kept only if it won):

1. **Router.** One 1D matmul (was DRAM-sharded on 8 cores); top-8 + softmax + shared-gate sigmoid on the RISC cores in
   one op instead of a 7-op chain.
2. **Fused MoE decode.** Routed + shared experts in three ops (route, gate|up + SwiGLU, down) that stream each selected
   expert's weights in place, for every decode width from 1 to 32 rows (several users' experts as one union, each weight
   column read once for all rows). On four dies the gate and up halves of an expert column are split across a pair of
   cores, with readers placed off the congested NOC rows.
3. **Replicated-residual layer boundary.** The decode residual stays replicated on every die: each mixer / MoE output is
   one all-reduce + add + local RMSNorm.
4. **Gated DeltaNet decode mixer** as two programs (causal conv; gates + norms + recurrent state update + gated norm),
   with each head's value dimension split across up to four cores.
5. **Attention decode ops** around the QKV matmul, cache update and SDPA decode (q / k norm + RoPE + gate in one prep op,
   output gating in one post op).
6. **Fabric layer boundary** (four dies, 1-4 users): the all-reduce, residual add and RMSNorm run as one op over the
   fabric, and the next layer's input projection (or the MoE router) runs inside the same program, streaming its weights
   while the boundary computes.
7. **Sampler top-k over the live rows only** (four dies, 1-8 users), instead of all 32 slot rows.

MTP speculative decoding is off on the multi-die profiles: its verify pass runs the unfused path, and the fused plain
decode step is faster on real text (four dies: 218 vs 86 tok/s).

Numerics: the routed and shared expert weights are stored rounded to 5 significant bits (exactly what the LoFi expert
matmul reads), activations / routing weights rounded to the 7 bits the matmul's other operand reads; the fused boundary
and DeltaNet ops sum in a different order than the op chain they replace. Output quality is checked below.

On one die (`p150`, `p100`) the full expert width does not fit the fused kernels' bank layout, so those profiles keep the
original MoE and gain only the fused router top-k (1-row steps), with MTP on as in the original.

### Use with your client
Point an OpenAI-compatible client at `http://127.0.0.1:20000` (or the port `tt-model serve` printed) with model
`ornith-ai/Ornith-1.5-35B-A3B`. Thinking is ON by default (reasoning in the message's `reasoning` field, answer in
`content`); turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`. Give thinking requests a
large `max_tokens`. Tool calls use the `qwen3_xml` parser. Images (up to 4 per request, resized to at most 4 MP) as
`image_url` parts on the multi-die profiles.

### Profiles
`tt-model serve raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast --profile <name>`:
- `p300` (default) / `p150x2` - two dies, bfp8 routed experts, text + images (no fabric boundary: its receive buffers do
  not fit next to the two-die prefill).
- `p300x2` / `p150x4` - four dies, bfp8 routed experts, text + images: the fastest profile.
- `p150` / `p100` - one die, bfp4 routed experts, text only (original MoE, MTP on).

Every profile: 32 concurrent sequences, 262144-token context.
