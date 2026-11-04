#!/bin/bash
# mira_chain.sh [profiles...]: MIRA-SFT-35B-A3B bring-up on the local bundle after the Tongyi chain frees the chips; each
# profile served exactly as shipped (tt-model serve --profile): smoke.py (thinking, no-thinking, tool call, image),
# greedy.py, bench rows, real text, sampled real text with / without presence_penalty 1.5 (MIRA's runtime value), the
# 128-request image burst (vision profiles), prefix caching (repeat + chat).
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=MIRA-MMDeepResearch/MIRA-SFT-35B-A3B
R=$HOME/ornith-fusion/runs; T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
SMOKE=$HOME/tt-metal-bringups/models/autoports/osunlp_quest_35b_rl/smoke.py
B=${B:-$HOME/tt-metal-bringups/build/mira-sft-35b-a3b/tt_kernel_manifest.json}
PROFILES=("$@"); [ ${#PROFILES[@]} -eq 0 ] && PROFILES=(p300 p300x2 p150)

while systemctl --user is-active -q mira-dl || systemctl --user is-active -q tongyi-chain; do sleep 30; done
grep -qiE "error|Traceback" $R/mira/dl.log && { echo "download failed"; tail -3 $R/mira/dl.log; exit 1; }
[ -f "$B" ] || { echo "bundle missing: $B"; exit 1; }
echo "weights ready, chips free $(date +%T)"

serve() {  # <label> <profile>; sets L, C
  L=$1; local P=$2
  C=tt-model-mira-sft-35b-a3b-$P; mkdir -p $R/$L
  docker rm -f $C >/dev/null 2>&1
  tt-model serve "$B" --profile $P --detach --port 20000 --no-update-check > $R/$L/serve.log 2>&1
  (docker logs -f $C 2>&1 | grep --line-buffered -v DEBUG > $R/$L/server.log &)
  local t0=$(date +%s)
  until curl -sf localhost:20000/v1/models >/dev/null; do
    docker ps --format '{{.Names}}' | grep -qx $C || { echo "$L: server died $(date +%T)"; grep -E "TT_FATAL|Error|Traceback" $R/$L/server.log | tail -5; return 1; }
    sleep 10
  done
  echo "$L: up after $(( $(date +%s) - t0 )) s $(date +%T)"
}

for P in "${PROFILES[@]}"; do
  serve mira_$P $P || continue
  NOIMG=""; [ $P = p150 ] && NOIMG=--no-image
  /usr/bin/python3 $SMOKE $NOIMG 2>&1 | tee $R/$L/smoke.txt
  URL=http://localhost:20000 RUNS=$R python3 $T/greedy.py run $L > $R/$L/greedy.log 2>&1 && echo "$L: greedy done"
  ROWS=(128:128:1:4 1024:256:1:4 4096:256:1:4 32768:256:1:2 128:128:8:32 1024:256:8:32 1024:256:32:128)
  RUNS=$R $T/bench.sh $C $L "${ROWS[@]}" 2>&1 | grep -E "==|TTFT|TPOT|ROW FAILED"
  URL=http://localhost:20000 python3 $T/realtext.py 384 > $R/$L/realtext.txt 2>&1; tail -n 2 $R/$L/realtext.txt
  for PP in "" 1.5; do for U in 1 8; do
    PRESENCE=$PP SAMPLE=1.0,20,0.95 CONC=$U URL=http://localhost:20000 python3 $T/realtext.py 384 > $R/$L/realtext_s_pp${PP:-0}_c$U.txt 2>&1
    echo "presence=${PP:-0} users=$U: $(tail -n 1 $R/$L/realtext_s_pp${PP:-0}_c$U.txt)"
  done; done
  if [ $P != p150 ]; then
    ~/evals/text2/.venv/bin/python $R/mm_burst_test.py $MODEL_ID 128 32 > $R/$L/mm_burst.txt 2>&1; tail -n 1 $R/$L/mm_burst.txt
    curl -sf localhost:20000/v1/models >/dev/null || { echo "$L: server down after the burst"; continue; }
    URL=http://localhost:20000 TAG=$P python3 $T/prefix_bench.py repeat 2048 8192 32768 2>&1 | tee $R/$L/prefix_repeat.txt | tail -n 6
    URL=http://localhost:20000 TAG=$P python3 $T/prefix_bench.py chat 8000 4 2>&1 | tee $R/$L/prefix_chat.txt | tail -n 7
    grep -E "Prefix cache hit rate" $R/$L/server.log | tail -n 1 | grep -oE "Prefix cache hit rate: [0-9.]+%"
  fi
  grep -E "TT_FATAL|EngineCore encountered|Traceback" $R/$L/server.log | head -3
  docker rm -f $C >/dev/null
done
echo "chain done $(date +%T)"
