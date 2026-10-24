### What blackhole-fast-v2 changes
This is [raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast](https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast)
with the rest of the decode layer fused. Same checkpoint, same vLLM plugin build, same profiles; prefill is unchanged. The
changes, each measured and kept only if it won (4 dies, 1 user, per layer):

1. **Gated DeltaNet decode mixer.** Between the in- and out-projections, the short convolution is one program and the
   whole recurrent step (gates, q / k L2 norms, state read, delta rule, rank-1 state update, gated RMSNorm x silu(z)) is a
   second one, one unit per (user, value head), the state kept bf16 and updated in FP32: 46 ops -> 4, GDN mixer
   204 -> 66 us.
2. **Attention decode ops.** From the fused QKV projection, one program does the q / k RMSNorm (1 + w), RoPE, v and
   the sigmoid output gate and writes k / v straight into the cache-update layout; a second one gates the SDPA output
   into the o-proj input: 41 ops -> 7, attention mixer 156 -> 67 us.
3. **MTP off on the multi-die profiles.** The MTP verify pass still runs the unfused path, so with the fused step plain
   decode is about twice as fast on real text (4 dies: 169 vs 86 tok/s).

Numerics: the new kernels keep the reference precision (FP32 accumulation; state and vector operands split hi / lo for
the matmul unit); layer outputs match Hugging Face to the same PCC as before. Output quality is checked below.

One die (`p150`, `p100`) runs exactly the blackhole-fast decode path (the new ops have not run on one die there; MTP on).

### Use with your client
Point an OpenAI-compatible client at `http://127.0.0.1:20000` (or the port `tt-model serve` printed) with model
`ornith-ai/Ornith-1.5-35B-A3B`. Thinking is ON by default (reasoning in the message's `reasoning` field, answer in
`content`); turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`. Give thinking requests a
large `max_tokens`. Tool calls use the `qwen3_xml` parser. Images (up to 4 per request, resized to at most 4 MP) as
`image_url` parts on the multi-die profiles.

### Profiles
`tt-model serve raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast-v2 --profile <name>`:
- `p300` (default) / `p150x2` - two dies, bfp8 routed experts, text + images.
- `p300x2` / `p150x4` - four dies, bfp8 routed experts, text + images: the fastest profile.
- `p150` / `p100` - one die, bfp4 routed experts, text only (the blackhole-fast path, MTP on).

Every profile: 32 concurrent sequences, 262144-token context.
