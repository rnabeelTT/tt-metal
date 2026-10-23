### What blackhole-fast changes
This is [raahemnabeel/ornith-1.5-35b-a3b](https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b) (v2) with its decode
step rebuilt around fused kernels. Same checkpoint, same vLLM plugin build, same profiles and serving recipe; prefill is
unchanged. The changes, in the order they were made (each measured, kept only if it won):

1. **Router.** The router matmul runs as one 1D matmul (it was a DRAM-sharded matmul on 8 cores), and the top-8 +
   softmax + shared-gate sigmoid run on the RISC cores in one op instead of a 7-op chain (MoE 177 -> 137 us per layer
   on 4 dies).
2. **Fused MoE decode.** Routed + shared experts in three ops (route, gate|up + SwiGLU, down) that stream each
   selected expert's weights in place from the existing DRAM layout (no duplicate copy), for every decode width: 1 to
   32 rows, the MTP verify rows included (several users' experts as one union, each weight column read once for all
   rows). MoE 137 -> 38 us per layer at 1 user on 4 dies; 2 dies at 16 users 1291 -> 541 us.
3. **Replicated-residual decode boundary.** The decode residual stays replicated on every die: each mixer / MoE output
   is one all-reduce + add + local RMSNorm (two collectives per layer instead of four, no gathers); boundary 89 -> 52 us
   per layer on 2 dies.

Numerics: the routed and shared expert weights are stored rounded to 5 significant bits (exactly what the LoFi expert
matmul reads, so the rounding is no longer a bias), activations / routing weights rounded to the 7 bits the matmul's
other operand reads. Output quality is checked below.

On one die (`p150`, `p100`) the full expert width does not fit the fused kernels' bank layout yet, so those profiles keep
the v2 MoE and gain only the fused router top-k (1-row steps; the 1D router matmul needs a mesh).

### Use with your client
Point an OpenAI-compatible client at `http://127.0.0.1:20000` (or the port `tt-model serve` printed) with model
`ornith-ai/Ornith-1.5-35B-A3B`. Thinking is ON by default (reasoning in the message's `reasoning` field, answer in
`content`); turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`. Give thinking requests a
large `max_tokens`. Tool calls use the `qwen3_xml` parser. Images (up to 4 per request, resized to at most 4 MP) as
`image_url` parts on the multi-die profiles.

### Profiles
`tt-model serve raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast --profile <name>`:
- `p300` (default) / `p150x2` - two dies, bfp8 routed experts, text + images.
- `p300x2` / `p150x4` - four dies, bfp8 routed experts, text + images: the fastest profile.
- `p150` / `p100` - one die, bfp4 routed experts, text only (v2 MoE).

Every profile: 32 concurrent sequences, 262144-token context, MTP speculative decoding while one request is live.
