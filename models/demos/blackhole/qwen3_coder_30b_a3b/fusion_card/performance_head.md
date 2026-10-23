### Accuracy
Both bundles evaluated side by side on the same box, p300x2, through the served chat API with lm-evaluation-harness
0.4.9 at temperature 0 (full sets, 8 requests in flight, seed 42):

| Eval | published | blackhole-fast |
| --- | ---: | ---: |
| HumanEval pass@1 (164) | {HE_PUB} | {HE_NEW} |
| MBPP pass@1 (500) | {MBPP_PUB} | {MBPP_NEW} |
| GSM8K exact match, strict (1319) | {GSM_PUB} | {GSM_NEW} |
| IFEval prompt-level strict (541) | {IFS_PUB} | {IFS_NEW} |
| IFEval prompt-level loose (541) | {IFL_PUB} | {IFL_NEW} |

{EVAL_NOTE} The harness's stock chat variants do not work through a chat API (`humaneval_instruct` and `mbpp_instruct`
pre-fill the assistant turn, which the API does not continue), so HumanEval uses a chat prompt that asks for the whole
function in one code block, and MBPP answers are scored from their first code block; `gsm8k_cot_llama` is the stock
8-shot task with the dataset's current id. The task files are in `fusion/tools/lm_tasks` on the source branch. Greedy
outputs of the p300 and p150 profiles were checked against the published bundle's.

### Performance
Measured 2026-10-24 on one QuietBox 2 (4 Blackhole p300c dies; p300 rows on two of them, p150 on one), both bundles
served exactly as shipped (`tt-model serve <bundle> --profile <p>`), one server at a time: random-token prompts of
exactly the input length, output pinned with ignore_eos, streamed; users are concurrent requests in a closed loop;
TTFT = time to first streamed token; TPOT = (E2EL - TTFT) / (output - 1); one unrecorded warm-up per row. A real-text
single-user row (12 chat prompts, 384 output tokens, temperature 0) is added. Each cell is `published -> blackhole-fast`.
