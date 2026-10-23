### Accuracy
Both bundles evaluated side by side on the same box (p300x2, thinking off, one sample per question, seed 42,
temperature 0.6 / top_p 0.95 / top_k 20, lm-evaluation-harness through the served API):

| Eval | published | blackhole-fast |
| --- | ---: | ---: |
| GPQA-Diamond (all 198 questions) | 0.697 | 0.677 |
| IFEval prompt-level strict (100) | 0.85 | 0.87 |
| LiveCodeBench v6 (50) | 0.54 | 0.54 |

GPQA: 12 questions right only with blackhole-fast, 16 only with the published bundle (McNemar p 0.57); the difference
is within one standard error (+-0.033). A 40-question subset had shown 0.65 vs 0.775, which is why the full set ran.

Layer-level PCC against Hugging Face (real weights, 4 dies, shipped precision policy), published -> fused decode: Gated
DeltaNet layer 0.99967 -> 0.99965, attention layer 0.99477 -> 0.99471; 32 concurrent users, worst user 0.99717 ->
0.99719 / 0.99548 -> 0.99532. Prefill is unchanged (identical PCC).

### Performance
Measured 2026-10-23 on one QuietBox 2 (4 Blackhole p300c dies), both bundles served exactly as shipped (`tt-model
serve <bundle> --profile p300x2`), one server at a time: random-token prompts of exactly the input length, output
pinned with ignore_eos, streamed; users are concurrent requests in a closed loop; TTFT = time to first streamed token;
TPOT = (E2EL - TTFT) / (output - 1); one unrecorded warm-up per row. Single-user rows run MTP speculative decoding,
whose gain depends on how predictable the text is: random-token prompts understate it, so a real-text row (12 chat
prompts, 384 output tokens, temperature 0) is added. Each cell is `published -> blackhole-fast`.
