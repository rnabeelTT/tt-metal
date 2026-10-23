#!/bin/bash
# eval_coder.sh <label> [tasks...]: the Qwen3-Coder-Next card's eval set through the served chat API (port PORT, default
# 20000): humaneval_chat (lm_tasks/; humaneval_instruct's gen_prefix is not continued by the chat API, it scores 0),
# mbpp_instruct, gsm8k_cot_llama, ifeval at temperature 0, 8 in flight, full sets unless LIMIT is set.
# Run on both bundles with the same settings (the card's exact task variants were not recorded).
set -e
L=$1; shift
TD=$(cd "$(dirname "$0")" && pwd)
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-Coder-Next}
OUT=${RUNS:-$HOME/ornith-fusion/runs}/$L/evals
mkdir -p $OUT
TASKS=("$@"); [ ${#TASKS[@]} -eq 0 ] && TASKS=(humaneval_chat mbpp_instruct gsm8k_cot_llama ifeval)
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_ALLOW_CODE_EVAL=1 NLTK_DATA=${NLTK_DATA:-$HOME/ornith-fusion/nltk_data}
cd ${LMEVAL:-$HOME/lm-eval-poc}
for T in "${TASKS[@]}"; do
  echo "== $L $T $(date +%T)"
  .venv/bin/lm_eval --model local-chat-completions \
    --model_args "model=$MODEL_ID,base_url=http://localhost:${PORT:-20000}/v1/chat/completions,num_concurrent=${CONC:-8},max_retries=3,tokenized_requests=False,timeout=3600" \
    --tasks $T --include_path $TD/lm_tasks ${LIMIT:+--limit $LIMIT} --apply_chat_template --seed 42 --confirm_run_unsafe_code \
    --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=2048" \
    --output_path $OUT/$T --log_samples 2>&1 | tail -12
done
