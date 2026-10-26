### Accuracy
Benchmark evals through the served API (p300 profile, thinking off, one sample per question, seed 42, temperature 0.6 /
top_p 0.95 / top_k 20, lm-evaluation-harness), the same settings and box as the original bundle's run:

| Eval | original | blackhole-fast |
| --- | ---: | ---: |
| GPQA-Diamond (40 questions) | 0.775 | 0.675 |
| IFEval prompt-level strict (100) | 0.76 | 0.82 |
| LiveCodeBench v6 (50) | 0.36 | 0.40 |

All within the error bars of these subset sizes (GPQA +-0.07, IFEval +-0.04, LiveCodeBench +-0.07): no measurable
quality change. The whole GPQA-Diamond set (198 questions, same settings) scores 0.707 +-0.032 on blackhole-fast; the
same build scored 0.75 on the first 40 questions in that run (0.675 in the subset run above), the spread one sample
per question gives at n = 40. Layer-level PCC against Hugging Face (real weights): decode >= 0.9999 at 1 user,
per-user minimum >= 0.9999 at 32 users.

### Performance
Measured 2026-10-25 on one QuietBox 2 (4 Blackhole p300c dies; p300 and p150 rows on a subset of it), both bundles
served exactly as shipped (`tt-model serve <bundle> --profile <p>`), one server at a time: random-token prompts of
exactly the input length, output pinned with ignore_eos, streamed; users are concurrent requests in a closed loop;
TTFT = time to first streamed token; TPOT = (E2EL - TTFT) / (output - 1); one unrecorded warm-up per row. The original
runs MTP speculative decoding while one request is live (its single-user rows include it); blackhole-fast runs plain
decode on the multi-die profiles. The real-text row (12 chat prompts, 384 output tokens, temperature 0) shows
single-user speed on predictable text, where MTP helps the original most. Each cell is `original -> blackhole-fast`.
