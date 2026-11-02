### Accuracy
Evaluated on the served bundle (p300x2, thinking off, one sample per question, seed 42, temperature 0.6 / top_p 0.95 /
top_k 20, lm-evaluation-harness through the chat API), next to the Qwen3.5-122B-A10B bundle on the same code, box and
settings:

| Eval | Ornith-1.5-122B | Qwen3.5-122B-A10B |
| --- | ---: | ---: |
| GPQA-Diamond (all 198 questions) | 0.712 ±0.032 | 0.697 |
| IFEval prompt-level strict (100) | 0.76 ±0.043 | 0.88 |
| IFEval prompt-level loose (100) | 0.79 ±0.041 | 0.93 |
| LiveCodeBench v6 (50) | 0.52 ±0.071 | 0.54 |

The IFEval misses are instruction-following misses in fluent text (exact sentence counts, keyword / letter frequencies,
all-caps or all-lowercase replies with an added Markdown heading), not corrupted output; the smaller Ornith-1.5-35B-A3B
scores 0.76-0.82 on the same harness and box. The model card's own benchmarks (agentic coding and tool use) were not
reproduced here.

Layer-level PCC against Hugging Face on the Ornith-1.5-122B weights (4 dies, shipped precision policy, fused decode):
Gated DeltaNet decode 0.99933 (prefill 0.99624), attention decode 0.99406 (prefill 0.99332); 32 concurrent users,
worst user 0.99717 (DeltaNet) / 0.99530 (attention) - the same as the Qwen3.5-122B-A10B checkpoint on this code
(0.99965 / 0.99471, worst users 0.99719 / 0.99532). Sampled long-context text (6 samples at temperature 0.6): coherent;
the probe's 8.9 flagged words per 1k are all real technical terms (quantize, outliers, dimensionality, embeddings).

### Performance
Measured on one QuietBox 2 (4 Blackhole p300c dies), 2026-10-27, the bundle served exactly as shipped
(`tt-model serve <bundle> --profile <p>`), one server at a time: random-token prompts of exactly the input length,
output pinned with ignore_eos, streamed; users are concurrent requests in a closed loop; TTFT = time to first streamed
token; TPOT = (E2EL - TTFT) / (output - 1); one unrecorded warm-up per row; total output tok/s includes prefill. The
right-hand column is the Qwen3.5-122B-A10B bundle on the same code (2026-10-26) for reference.
