### Accuracy
Benchmark evals through the served API (p300 profile, MTP on, thinking off, one sample per question, seed 42, temperature
0.6 / top_p 0.95 / top_k 20, lm-evaluation-harness), the same settings as the v2 baseline run on the same box:

| Eval | v2 (published) | blackhole-fast |
| --- | ---: | ---: |
| GPQA-Diamond (40 questions) | 0.775 | 0.85 |
| IFEval prompt-level strict (100) | 0.76 | 0.77 |
| LiveCodeBench v6 (50) | 0.36 | 0.40 |

All within the error bars of these subset sizes (GPQA +-0.06, LiveCodeBench +-0.07): no measurable quality change.
A long-context garble probe (non-word rate of sampled continuations, 6 runs) scores 7.4-9.0 per 1k words vs 7.2-10.7
for v2 on the same prompts. Layer-level PCC against Hugging Face (real weights) is unchanged: decode >= 0.9999 at
1 user, per-user minimum >= 0.9999 at 32 users. Teacher forcing (v2 numbers below) was not re-run.

### Performance
Measured 2026-10-23 on one QuietBox 2 (4 Blackhole p300c dies; p300 and p150 rows on a subset of it), both bundles
served exactly as shipped (`tt-model serve <bundle> --profile <p>`), one server at a time, the v2 card's method:
random-token prompts of exactly the input length, output pinned with ignore_eos, streamed; users are concurrent requests
in a closed loop; TTFT = time to first streamed token; TPOT = (E2EL - TTFT) / (output - 1); one unrecorded warm-up per
row. Single-user rows run MTP speculative decoding, whose gain depends on how predictable the text is: random-token
prompts understate it, so a real-text row (12 chat prompts, 384 output tokens, temperature 0) is added. Each cell is
`v2 -> blackhole-fast`.
