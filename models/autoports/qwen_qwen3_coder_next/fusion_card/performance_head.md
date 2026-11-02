### Accuracy
Both bundles evaluated on the same box (v1 2026-10-23, current 2026-10-26), p300x2, through the served chat API with lm-evaluation-harness
0.4.9 at temperature 0 (full sets, 8 requests in flight, seed 42):

| Eval | v1 | current |
| --- | ---: | ---: |
| HumanEval pass@1 (164) | 0.939 | 0.933 |
| MBPP pass@1 (500) | 0.826 | 0.824 |
| GSM8K exact match, strict (1319) | 0.954 | 0.954 |
| IFEval prompt-level strict (541) | 0.799 | 0.791 |
| IFEval prompt-level loose (541) | 0.841 | 0.824 |

All differences are within one standard error (HumanEval +-0.020, IFEval +-0.018, MBPP +-0.017, GSM8K +-0.006): no
measurable quality change. The
harness's stock chat variants do not work through a chat API (`humaneval_instruct` and `mbpp_instruct` pre-fill the
assistant turn, which the API does not continue: both score 0), so HumanEval uses a chat prompt that asks for the whole
function in one code block, and MBPP answers are scored from their first code block; `gsm8k_cot_llama` is the stock
8-shot task with the dataset's current id. The task files are in `fusion/tools/lm_tasks` on the source branch. These
numbers are not comparable to the v1 card's (68.9 / 77.2 / 92.4 / 75.8 %), which used the stock tasks at an
unrecorded configuration; compare the two columns above. Greedy outputs of the p300 profile (bfp4 experts on two dies)
were checked against the v1 bundle's.

### Performance
Measured on one QuietBox 2 (4 Blackhole p300c dies; p300 rows on two of them; v1 2026-10-23, current
2026-10-26), both bundles served exactly
as shipped (`tt-model serve <bundle> --profile <p>`), one server at a time: random-token prompts of exactly the input
length, output pinned with ignore_eos, streamed; users are concurrent requests in a closed loop; TTFT = time to first
streamed token; TPOT = (E2EL - TTFT) / (output - 1); one unrecorded warm-up per row. A real-text single-user row (12 chat
prompts, 384 output tokens, temperature 0) is added. Each cell is `v1 -> current`.
