### Accuracy
Both bundles evaluated side by side on the same box (p300x2, thinking off, one sample per question, seed 42,
temperature 0.6 / top_p 0.95 / top_k 20, lm-evaluation-harness through the served API):

| Eval | published | blackhole-fast |
| --- | ---: | ---: |
| GPQA-Diamond (all 198 questions) | 0.697 | 0.697 |
| IFEval prompt-level strict (100) | 0.85 | 0.88 |
| LiveCodeBench v6 (50) | 0.54 | 0.54 |

GPQA-Diamond is identical (138 of 198 both), and the IFEval and LiveCodeBench differences are within one standard
error (±0.033 over 100 prompts, ±0.071 over 50 problems).

Layer-level PCC against Hugging Face (real weights, 4 dies, shipped precision policy), published -> fused decode: Gated
DeltaNet layer 0.99967 -> 0.99965, attention layer 0.99477 -> 0.99471; 32 concurrent users, worst user 0.99717 ->
0.99719 / 0.99548 -> 0.99532. Prefill is unchanged (identical PCC).

### Performance
Measured on one QuietBox 2 (4 Blackhole p300c dies; published 2026-10-23, blackhole-fast 2026-10-26), both bundles
served exactly as shipped (`tt-model serve <bundle> --profile p300x2`), one server at a time: random-token prompts of exactly the input length, output
pinned with ignore_eos, streamed; users are concurrent requests in a closed loop; TTFT = time to first streamed token;
TPOT = (E2EL - TTFT) / (output - 1); one unrecorded warm-up per row. The published bundle's single-user rows run MTP
speculative decoding, whose gain depends on how predictable the text is (random-token prompts understate it), so a
real-text row (12 chat prompts, 384 output tokens, temperature 0) is added; blackhole-fast decodes without MTP. Each cell is `published -> blackhole-fast`.
