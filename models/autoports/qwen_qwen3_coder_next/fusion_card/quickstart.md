### What blackhole-fast changes
This is [raahemnabeel/qwen3-coder-next](https://huggingface.co/raahemnabeel/qwen3-coder-next) with its decode step
rebuilt around fused kernels, ported from the Ornith-1.5-35B-A3B fusion
([raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast](https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast)).
Same checkpoint, same vLLM plugin build, same profiles and serving recipe; prefill is unchanged (TTFT equal or
lower). The changes:

1. **Router.** The decode router matmul runs as one 1D matmul, and the 1-token top-10 + softmax + shared-gate sigmoid
   run on the RISC cores in one op instead of the top-k route -> slice -> softmax -> untilize chain.
2. **Fused MoE decode.** Routed + shared experts in three ops (route, gate|up + SwiGLU, down) that stream each selected
   expert's weights in place from the existing DRAM layout (no duplicate copy), for every decode width from 1 to 32
   rows (several users' experts as one union over the 512 experts, each weight column read once for all rows).
3. **Replicated-residual decode boundary.** The decode residual stays replicated on every die: each mixer / MoE output
   is one all-reduce + add + local RMSNorm (two collectives per layer instead of four, no gathers).
4. **Gated DeltaNet decode mixer** as two programs (short conv; gates + norms + recurrent state update + gated norm),
   each head's value dimension split across up to four cores.
5. **Attention decode ops** around the QKV matmul, cache update and SDPA decode (q / k norm + RoPE + gate in one prep op,
   output gating in one post op).
6. **Fabric layer boundary** (four dies, 1-4 users): the all-reduce, residual add and RMSNorm as one op over the fabric,
   with the next layer's input projection (or the MoE router) inside the same program; the MoE gate|up columns split
   across core pairs with readers off the congested NOC rows.
7. **Sampler top-k over the live rows only** (four dies, 1-8 users), instead of all 32 slot rows.

Numerics: the routed and shared expert weights are stored rounded to 5 significant bits (what the LoFi expert matmul
reads), activations / routing weights rounded to the 7 bits the matmul's other operand reads; the fused boundary and DeltaNet ops
sum in a different order than the op chain they replace. Output quality is checked below.

### Use with your client
Point an OpenAI-compatible client at `http://127.0.0.1:20000` (or the port `tt-model serve` printed) with model
`Qwen/Qwen3-Coder-Next`. The model does not think (no reasoning block); tool calls use vLLM's `qwen3_coder` parser
(`--enable-auto-tool-choice` is on). Without sampling parameters a request uses the checkpoint's defaults
(temperature 1.0, top_p 0.95, top_k 40).

### Profiles
`tt-model serve raahemnabeel/qwen3-coder-next-blackhole-fast --profile <name>`:
- `p300x2` (default) / `p150x4` - four dies: the faster profile.
- `p300` / `p150x2` - two dies: the same features on half the hardware, with bfp4 routed experts.

Every profile: 32 concurrent sequences, 262144-token context, chunked prefill (long prompts do not block short
requests). The checkpoint has no MTP head and no vision tower.
