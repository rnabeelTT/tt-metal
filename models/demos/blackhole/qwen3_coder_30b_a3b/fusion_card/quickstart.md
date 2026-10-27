### What changed since v1
This is the v1 bundle ([raahemnabeel/qwen3-coder-30b-a3b-v1](https://huggingface.co/raahemnabeel/qwen3-coder-30b-a3b-v1)) with
its MoE decode replaced by the fused kernels of the Ornith-1.5-35B-A3B fusion
([raahemnabeel/ornith-1.5-35b-a3b](https://huggingface.co/raahemnabeel/ornith-1.5-35b-a3b)),
in expert-parallel form. Same checkpoint, same vLLM plugin build, same profiles and serving recipe; prefill (the
expert-sorted MoE) is unchanged. Per decode layer and die, three ops replace the replicated router chain, the two
dynamic sparse matmuls, SwiGLU, the routing weights and the expert sum (and the expert-sorted path at 4+ rows):

1. **Route.** The router projection (FP32 accumulation) and the global top-8 + softmax run in one op; each die keeps
   the selected experts that fall in its own window (die d holds experts 32 d .. 32 d + 31 on four dies, 64 per die on
   two, all 128 on one).
2. **Gate|up + SwiGLU** and **down + weighted sum** stream each selected local expert's weights in place from the
   bundle's existing bfp4 DRAM layout (no copy, no re-rounding), for every decode width from 1 to 32 rows: several
   users' experts form one union per die, so each weight column is read once for all rows. A die whose window holds no
   selected expert writes a zero partial; the existing all-reduce sums the dies.

Numerics: the same bfp4 expert weights and FP32 router logits as v1. The fused route op accumulates
the logits in a different order and its expert matmuls accumulate in FP32, so a near-tied 8th expert can be picked
differently and the last bits of an expert's output can differ. Output quality is checked below.

### Use with your client
Point any OpenAI client at the address `tt-model serve` prints (port 20000 unless it was busy), with model id
`Qwen/Qwen3-Coder-30B-A3B-Instruct`. Tool calls use the `qwen3_coder` parser. The server's default generation config
is greedy (temperature 0); per-request sampling parameters apply.

### Profiles
`tt-model serve raahemnabeel/qwen3-coder-30b-a3b --profile <name>`:
- `p300` (default) / `p150x2` - two dies (64 experts per die), 256000-token context.
- `p300x2` / `p150x4` - four dies (32 experts per die), 256000-token context: the fastest profile.
- `p150` - one die (all 128 experts), 98304-token context (the bf16 KV cache is 96 KiB per token on one die).

All profiles: 32 concurrent sequences, prefix caching, bfp4 routed experts.
