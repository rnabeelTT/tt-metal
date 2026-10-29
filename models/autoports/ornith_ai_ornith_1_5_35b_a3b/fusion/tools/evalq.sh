#!/bin/bash
# evalq.sh: queue runner for the original-card evals (fusion/REF_EVALS.md). Reads tasks from evalq.txt, one per line:
#   <model> <runner> <bench> [KEY=VAL ...]
#   model : orn | o122 | q122 | cn          (served from the hub repo as shipped, p300x2)
#   runner: lmeval | text2 | qtext | vision | scicode | aider | swebench | tbench
# Picks the next task for the model already being served (one boot per model), else the first line. Lines can be
# appended while it runs. Finished tasks go to evalq.done with their RESULT lines; it stops the server when the
# queue is empty and keeps polling. `# ...` lines and blank lines are ignored.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs
Q=$R/evalq.txt; DONE=$R/evalq.done
touch $Q $DONE
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_ALLOW_CODE_EVAL=1 NLTK_DATA=$HOME/ornith-fusion/nltk_data
BASE=http://localhost:20000/v1
declare -A REPO=([orn]=raahemnabeel/ornith-1.5-35b-a3b [o122]=raahemnabeel/ornith-1.5-122b
                 [q122]=raahemnabeel/qwen3-5-122b-a10b [cn]=raahemnabeel/qwen3-coder-next)
declare -A NAME=([orn]=ornith-1.5-35b-a3b [o122]=ornith-1.5-122b [q122]=qwen3-5-122b-a10b [cn]=qwen3-coder-next)
declare -A MID=([orn]=ornith-ai/Ornith-1.5-35B-A3B [o122]=ornith-ai/Ornith-1.5-122B
                [q122]=Qwen/Qwen3.5-122B-A10B [cn]=Qwen/Qwen3-Coder-Next)
declare -A SAMP=([orn]=1.0,0.95,20 [o122]=0.6,0.95,20 [q122]=0.6,0.95,20 [cn]=1.0,0.95,40)
CUR=""; C=""

stop_server() { [ -n "$C" ] && docker rm -f $C >/dev/null 2>&1; CUR=""; C=""; }

serve() {  # <model>
  local M=$1 L=$R/evalq_serve_$1; mkdir -p $L
  stop_server
  C=tt-model-${NAME[$M]}-p300x2
  docker rm -f $C >/dev/null 2>&1
  tt-model serve ${REPO[$M]} --profile p300x2 --detach --port 20000 --no-update-check > $L/serve.log 2>&1
  local t0=$(date +%s)
  until curl -sf localhost:20000/v1/models >/dev/null; do
    docker ps --format '{{.Names}}' | grep -qx $C || { echo "serve $M: server died $(date +%T)"; docker logs $C 2>&1 | grep -v DEBUG | tail -n 300 > $L/boot_fail.log; C=""; return 1; }
    sleep 10
  done
  CUR=$M
  echo "serve $M: up after $(( $(date +%s) - t0 )) s; image $(docker inspect -f '{{.Config.Image}}' $C) $(date +%T)"
}

pick() {  # prints and removes the next task line (prefers model $1)
  python3 - "$Q" "$1" <<'EOF'
import sys, os, fcntl
q, cur = sys.argv[1], sys.argv[2]
with open(q, "r+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    lines = f.read().splitlines()
    idx = [i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#")]
    if not idx:
        sys.exit(1)
    pick = next((i for i in idx if lines[i].split()[0] == cur), idx[0])
    print(lines[pick])
    del lines[pick]
    f.seek(0); f.truncate(); f.write("\n".join(lines) + ("\n" if lines else ""))
EOF
}

run_task() {  # <model> <runner> <bench> [KEY=VAL ...]
  local M=$1 RN=$2 B=$3; shift 3
  local TAG="" kv; for kv in "$@"; do [[ $kv == TAG=* ]] && TAG=_${kv#TAG=}; done
  local L=ref_${M}_${B}${TAG} LOG=$R/ref_${M}_${B}${TAG}.log
  IFS=, read TT TP TK <<< "${SAMP[$M]}"
  echo "== $M $RN $B $* $(date +%T)"
  (
    [ $# -gt 0 ] && export "$@"
    export RUNS_ROOT=$R
    # The checkpoint's default sampling for the single-turn harnesses; the agentic harnesses (swebench, tbench,
    # sweatlas, deepswe, claweval, toolathlon) set each card's own settings from the model id, so only explicit
    # KEY=VAL overrides on the queue line reach them.
    case $RN in lmeval|text2|qtext|vision|scicode|extend|aider)
      export TEMPERATURE=${TEMPERATURE:-$TT} TOP_P=${TOP_P:-$TP} TOP_K=${TOP_K:-$TK} ;; esac
    case $RN in
      lmeval) OUT=$R/$L/evals/$B; mkdir -p $OUT
              cd $HOME/lm-eval-poc && .venv/bin/lm_eval --model local-chat-completions \
                --model_args "model=${MID[$M]},base_url=$BASE/chat/completions,num_concurrent=${CONC:-32},max_retries=3,tokenized_requests=False,timeout=7200" \
                --tasks $B --apply_chat_template --seed ${SEED:-42} --confirm_run_unsafe_code \
                --gen_kwargs "do_sample=True,temperature=$TEMPERATURE,top_p=$TOP_P,top_k=$TOP_K,max_gen_toks=${MAX_TOKENS:-32768}" \
                --output_path $OUT --log_samples
              python3 - $OUT <<'EOF'
import json, sys, glob
f = sorted(glob.glob(sys.argv[1] + "/**/results_*.json", recursive=True))[-1]
r = json.load(open(f))["results"]
for task, m in r.items():
    print("RESULT", task, " ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))
EOF
              ;;
      text2)  ~/evals/text2_run.sh $B $L $BASE ${MID[$M]} ${CONC:-32} ;;
      qtext)  ~/evals/qwen_text_run.sh $B $L $BASE ${MID[$M]} ${CONC:-32} ;;
      vision) ~/evals/vision_run.sh $B $L $BASE ${MID[$M]} ${CONC:-32} ;;
      swebench) ~/evals/swebench_run.sh $B $L $BASE ${MID[$M]} ${CONC:-8} ;;
      tbench) ~/evals/tbench_run.sh $B $L $BASE ${MID[$M]} ${CONC:-4} ;;     # 2.0 | 2.1 | frontier | sweatlas-qna
      sweatlas) ~/evals/sweatlas_run.sh $B $L $BASE ${MID[$M]} ${CONC:-8} ;;  # qna
      deepswe) ~/evals/deepswe_run.sh $B $L $BASE ${MID[$M]} ${CONC:-8} ;;    # v1.1
      claweval) ~/evals/claweval_run.sh $B $L $BASE ${MID[$M]} ${CONC:-8} ;;  # all | general | multimodal | multi_turn
      toolathlon) ~/evals/toolathlon_run.sh $B $L $BASE ${MID[$M]} ${CONC:-1} ;;  # verified (official service)
      scicode) ~/evals/scicode_run.sh $L $BASE ${MID[$M]} ${CONC:-32} ${MAX_TOKENS:-32768} ;;
      aider)  ~/evals/aider_run.sh $L $BASE ${MID[$M]} ${CONC:-32} $B ;;
      extend) # continue a finished text2 run's truncated answers to BUDGET tokens (exact; see extend_truncated.py)
              cd $HOME/evals/text2 && .venv/bin/python extend_truncated.py $R/ref_${M}_${B}/$B ref_${M}_${B}_b${BUDGET} \
                --budget $BUDGET --conc ${CONC:-16} ;;
      *) echo "unknown runner $RN" ;;
    esac
  ) > $LOG 2>&1
  local rc=$?
  local res=$(grep -E "^(RESULT|JUDGE_PENDING|SCICODE|AIDER|EXTENDED)" $LOG | tr '\n' ' ')
  echo "$M $RN $B $* | exit $rc | $res | $(date '+%F %T')" >> $DONE
  echo "$M $B exit $rc: $res $(date +%T)"
}

# adopt a server that is already up (e.g. after a restart of this runner) instead of waiting for it to go away
for m in "${!NAME[@]}"; do
  if docker ps --format '{{.Names}}' | grep -qx "tt-model-${NAME[$m]}-p300x2" && curl -sf localhost:20000/v1/models >/dev/null; then
    CUR=$m; C=tt-model-${NAME[$m]}-p300x2; echo "adopted running server $m $(date +%T)"
  fi
done
while systemctl --user is-active -q cards-release || { [ -z "$CUR" ] && chips_busy; }; do sleep 60; done
echo "evalq start $(date +%T)"
SELF_MTIME=$(stat -c %Y "$0")
while true; do
  # pick up edits to this script at a task boundary (the running server is adopted again on restart)
  [ "$(stat -c %Y "$0")" != "$SELF_MTIME" ] && { echo "evalq.sh changed, restarting $(date +%T)"; exec "$0"; }
  if T=$(pick "$CUR"); then
    set -- $T
    if [ "$1" != "$CUR" ] || ! curl -sf localhost:20000/v1/models >/dev/null; then
      serve $1 || { echo "$T | serve failed" >> $DONE; continue; }
    fi
    run_task $T
  else
    [ -n "$CUR" ] && { echo "queue empty, stopping $CUR $(date +%T)"; stop_server; }
    sleep 60
  fi
done
