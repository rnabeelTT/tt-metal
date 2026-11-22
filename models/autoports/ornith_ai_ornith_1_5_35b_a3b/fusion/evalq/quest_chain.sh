#!/bin/bash
# quest_chain.sh [profiles...]: QUEST-35B-RL bring-up on the local bundle, each profile served exactly as shipped
# (tt-model serve --profile): smoke.py (thinking, no-thinking, tool call, image), greedy.py, bench rows, real text,
# the 128-request image burst (vision profiles); then thinking-off evals on p300 (Ornith's card settings).
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=osunlp/QUEST-35B-RL
R=$HOME/ornith-fusion/runs; T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
Q=$HOME/tt-metal-bringups/models/autoports/osunlp_quest_35b_rl
B=${B:-$HOME/tt-metal-bringups/build/quest-35b-rl/tt_kernel_manifest.json}
PROFILES=("$@"); [ ${#PROFILES[@]} -eq 0 ] && PROFILES=(p300x2 p300 p150 evals)

serve() {  # <label> <profile> [vllm args...]; sets C
  local L=$1 P=$2; shift 2
  C=tt-model-quest-35b-rl-$P; mkdir -p $R/$L
  docker rm -f $C >/dev/null 2>&1
  tt-model serve "$B" --profile $P --detach --port 20000 --no-update-check ${1:+-- "$@"} > $R/$L/serve.log 2>&1
  (docker logs -f $C 2>&1 | grep --line-buffered -v DEBUG > $R/$L/server.log &)
  local t0=$(date +%s)
  until curl -sf localhost:20000/v1/models >/dev/null; do
    docker ps --format '{{.Names}}' | grep -qx $C || { echo "$L: server died $(date +%T)"; return 1; }
    sleep 10
  done
  echo "$L: up after $(( $(date +%s) - t0 )) s $(date +%T)"
}

for P in "${PROFILES[@]}"; do
  if [ $P = evals ]; then
    L=quest_evals_p300
    serve $L p300 --default-chat-template-kwargs '{"enable_thinking": false}' || continue
    GPQA_LIMIT=198 RUNS=$R $T/eval.sh $L 2>&1 | grep -E "^==|\|.*(exact_match|strict|loose|pass)" | tee $R/$L/evals.txt
    docker rm -f $C >/dev/null; continue
  fi
  L=quest_$P
  serve $L $P || continue
  NOIMG=""; [ $P = p150 ] && NOIMG=--no-image
  python3 $Q/smoke.py $NOIMG 2>&1 | tee $R/$L/smoke.txt
  URL=http://localhost:20000 RUNS=$R python3 $T/greedy.py run $L > $R/$L/greedy.log 2>&1 && echo "$L: greedy done"
  ROWS=(128:128:1:4 1024:256:1:4 4096:256:1:4 32768:256:1:2 128:128:8:32 1024:256:8:32 1024:256:32:128)
  RUNS=$R $T/bench.sh $C $L "${ROWS[@]}" 2>&1 | grep -E "==|TTFT|TPOT|ROW FAILED"
  URL=http://localhost:20000 python3 $T/realtext.py 384 > $R/$L/realtext.txt 2>&1; tail -n 2 $R/$L/realtext.txt
  if [ $P != p150 ]; then
    ~/evals/text2/.venv/bin/python $R/mm_burst_test.py $MODEL_ID 128 32 > $R/$L/mm_burst.txt 2>&1; tail -n 1 $R/$L/mm_burst.txt
    curl -sf localhost:20000/v1/models >/dev/null || echo "$L: server down after the burst"
  fi
  grep -E "TT_FATAL|EngineCore encountered|Traceback" $R/$L/server.log | head -3
  docker rm -f $C >/dev/null
done
echo "chain done $(date +%T)"
