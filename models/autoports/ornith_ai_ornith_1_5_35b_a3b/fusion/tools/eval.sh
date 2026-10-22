#!/bin/bash
# eval.sh <label> [tasks...]   The scorecard's abbreviated evals against the server on :20000 (thinking off at the server):
# r1_gpqa_diamond 40, livecodebench_v6 50, ifeval 100; one sample per question, seed 42, temperature 0.6 / top_p 0.95 /
# top_k 20, 8 in flight (CONC overrides: 1 exercises the single-user decode path); max gen 4096 (ifeval 2048).
set -e
L=$1; shift
OUT=${RUNS:-$HOME/ornith-fusion/runs}/$L/evals
mkdir -p $OUT
TASKS=("$@"); [ ${#TASKS[@]} -eq 0 ] && TASKS=(r1_gpqa_diamond ifeval livecodebench_v6)
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_ALLOW_CODE_EVAL=1 NLTK_DATA=${NLTK_DATA:-$HOME/ornith-fusion/nltk_data}
cd ${LMEVAL:-$HOME/lm-eval-poc}
for T in "${TASKS[@]}"; do
  case $T in
    r1_gpqa_diamond) LIM=40; MAXG=4096;;
    livecodebench_v6) LIM=50; MAXG=4096;;
    ifeval) LIM=100; MAXG=2048;;
  esac
  echo "== $L $T limit $LIM $(date +%T)"
  .venv/bin/lm_eval --model local-chat-completions \
    --model_args "model=ornith-ai/Ornith-1.5-35B-A3B,base_url=http://localhost:20000/v1/chat/completions,num_concurrent=${CONC:-8},max_retries=3,tokenized_requests=False,timeout=3600" \
    --tasks $T --limit $LIM --apply_chat_template --seed 42 --confirm_run_unsafe_code \
    --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,max_gen_toks=$MAXG" \
    --output_path $OUT/$T --log_samples 2>&1 | tail -15
done
