#!/bin/bash
# ref_evals.sh [keys...]: the original model cards' own evals that run on this box, measured the way the card does
# (thinking on as shipped, the checkpoint's sampling defaults, full sets), to sit beside the card's figures
# (user request 2026-10-28). lm-evaluation-harness through the served chat API, p300x2 as shipped from the hub,
# 32 requests in flight, max 32768 generated tokens. Only the answer (`content`) is scored; reasoning is separate.
#   orn : GPQA-Diamond x5 runs (Ornith's card averages 5 runs)          card 89.2
#   q122: GPQA-Diamond, IFEval (541), LiveCodeBench v6 (test6, 175)      card 86.6 / 93.4 / 78.9
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_ALLOW_CODE_EVAL=1 NLTK_DATA=$HOME/ornith-fusion/nltk_data
LCB_YAML=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools/livecodebench_v6.yaml

serve() {  # <label> <repo> <name>; sets C
  local L=$1 B=$2; C=tt-model-$3-p300x2
  docker rm -f $C >/dev/null 2>&1
  tt-model serve "$B" --profile p300x2 --detach --port 20000 --no-update-check > $R/$L/serve.log 2>&1
  local t0=$(date +%s)
  until curl -sf localhost:20000/v1/models >/dev/null; do
    docker ps --format '{{.Names}}' | grep -qx $C || { echo "$L: server died"; docker logs $C 2>&1 | grep -v DEBUG | tail -n 300 > $R/$L/boot_fail.log; return 1; }
    sleep 10
  done
  echo "$L: up after $(( $(date +%s) - t0 )) s; image $(docker inspect -f '{{.Config.Image}}' $C)"
}

ev() {  # <label> <model id> <task> <seed> <sampling "T,p,k">
  local L=$1 M=$2 TK=$3 S=$4; IFS=, read TT TP TKK <<< "$5"
  local OUT=$R/$L/evals/${TK}_s$S; mkdir -p $OUT
  echo "== $L $TK seed $S $(date +%T)"
  (cd $HOME/lm-eval-poc && .venv/bin/lm_eval --model local-chat-completions \
    --model_args "model=$M,base_url=http://localhost:20000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,timeout=7200" \
    --tasks $TK --apply_chat_template --seed $S --confirm_run_unsafe_code \
    --gen_kwargs "do_sample=True,temperature=$TT,top_p=$TP,top_k=$TKK,max_gen_toks=32768" \
    --output_path $OUT --log_samples > $OUT/lm_eval.log 2>&1)
  echo "$L $TK seed $S exit $? $(date +%T)"
  grep -E "^\|.*(exact_match|prompt_level|pass@1|acc)" $OUT/lm_eval.log | head -6
}

while systemctl --user is-active -q cards-release || chips_busy; do sleep 60; done
echo "ref evals start $(date +%T)"
for k in ${@:-orn q122}; do
  case $k in
    orn) L=ref_orn; mkdir -p $R/$L; serve $L raahemnabeel/ornith-1.5-35b-a3b ornith-1.5-35b-a3b || continue
         for s in 42 43 44 45 46; do ev $L ornith-ai/Ornith-1.5-35B-A3B r1_gpqa_diamond $s 1.0,0.95,20; done
         docker rm -f $C >/dev/null ;;
    q122) L=ref_q122; mkdir -p $R/$L; serve $L raahemnabeel/qwen3-5-122b-a10b qwen3-5-122b-a10b || continue
         ev $L Qwen/Qwen3.5-122B-A10B r1_gpqa_diamond 42 0.6,0.95,20
         ev $L Qwen/Qwen3.5-122B-A10B ifeval 42 0.6,0.95,20
         (cd $HOME/lm-eval-poc && ls lm_eval/tasks/livecodebench/livecodebench_v6.yaml >/dev/null) && ev $L Qwen/Qwen3.5-122B-A10B livecodebench_v6 42 0.6,0.95,20
         docker rm -f $C >/dev/null ;;
  esac
done
echo "ref evals done $(date +%T)"
