### Accuracy
Both bundles evaluated side by side on the same box, p300x2, through the served chat API with lm-evaluation-harness
0.4.9 at temperature 0 (full sets, 8 requests in flight, seed 42):

| Eval | published | blackhole-fast |
| --- | ---: | ---: |
| HumanEval pass@1 (164) | 0.939 | 0.945 |
| MBPP pass@1 (500) | 0.826 | 0.820 |
| GSM8K exact match, strict (1319) | 0.954 | 0.955 |
| IFEval prompt-level strict (541) | 0.799 | 0.782 |
| IFEval prompt-level loose (541) | 0.841 | 0.826 |

All differences are within one standard error (IFEval +-0.018, MBPP +-0.017, GSM8K +-0.006): no measurable quality change. The
harness's stock chat variants do not work through a chat API (`humaneval_instruct` and `mbpp_instruct` pre-fill the
assistant turn, which the API does not continue: both score 0), so HumanEval uses a chat prompt that asks for the whole
function in one code block, and MBPP answers are scored from their first code block; `gsm8k_cot_llama` is the stock
8-shot task with the dataset's current id. The task files are in `fusion/tools/lm_tasks` on the source branch. These
numbers are not comparable to the published card's (68.9 / 77.2 / 92.4 / 75.8 %), which used the stock tasks at an
unrecorded configuration; compare the two columns above. Greedy outputs of the p300 profile (bfp4 experts on two dies)
were checked against the published bundle's.

### Performance
Measured 2026-10-23 on one QuietBox 2 (4 Blackhole p300c dies; p300 rows on two of them), both bundles served exactly
as shipped (`tt-model serve <bundle> --profile <p>`), one server at a time: random-token prompts of exactly the input
length, output pinned with ignore_eos, streamed; users are concurrent requests in a closed loop; TTFT = time to first
streamed token; TPOT = (E2EL - TTFT) / (output - 1); one unrecorded warm-up per row. A real-text single-user row (12 chat
prompts, 384 output tokens, temperature 0) is added. Each cell is `published -> blackhole-fast`.
