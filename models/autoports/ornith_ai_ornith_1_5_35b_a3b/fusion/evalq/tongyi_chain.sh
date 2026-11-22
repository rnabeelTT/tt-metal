#!/bin/bash
# tongyi_chain.sh [profiles...]: Tongyi-DeepResearch-30B-A3B bring-up on the local bundle, each profile served exactly as
# shipped (tt-model serve --profile): smoke (thinking, no-thinking, tool call), greedy.py, bench rows, real text,
# real text sampled at Tongyi's settings with and without presence_penalty 1.1 (1 and 8 users); then prefix caching.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B
R=$HOME/ornith-fusion/runs; T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
SMOKE=$HOME/tt-metal-bringups/models/autoports/osunlp_quest_35b_rl/smoke.py
B=${B:-$HOME/tt-metal-c30/build/tongyi-deepresearch-30b-a3b/tt_kernel_manifest.json}
PROFILES=("$@"); [ ${#PROFILES[@]} -eq 0 ] && PROFILES=(p300x2 p300 p150)

while systemctl --user is-active -q tongyi-dl; do sleep 30; done
grep -qiE "error|Traceback" $R/tongyi/dl.log && { echo "download failed"; tail -3 $R/tongyi/dl.log; exit 1; }
echo "weights ready $(date +%T)"

serve() {  # serve <label> <profile>; returns 1 if the server dies
  L=$1; P=$2; C=tt-model-tongyi-deepresearch-30b-a3b-$P; mkdir -p $R/$L
  docker rm -f $C >/dev/null 2>&1
  tt-model serve "$B" --profile $P --detach --port 20000 --no-update-check > $R/$L/serve.log 2>&1
  (docker logs -f $C 2>&1 | grep --line-buffered -v DEBUG > $R/$L/server.log &)
  t0=$(date +%s)
  until curl -sf localhost:20000/v1/models >/dev/null; do
    docker ps --format '{{.Names}}' | grep -qx $C || { echo "$L: server died $(date +%T)"; grep -E "TT_FATAL|Error|Traceback" $R/$L/server.log | tail -5; return 1; }
    sleep 10
  done
  echo "$L: up after $(( $(date +%s) - t0 )) s $(date +%T)"
}

for P in "${PROFILES[@]}"; do
  serve tongyi_$P $P || continue
  /usr/bin/python3 $SMOKE --no-image 2>&1 | tee $R/$L/smoke.txt
  URL=http://localhost:20000 RUNS=$R python3 $T/greedy.py run $L > $R/$L/greedy.log 2>&1 && echo "$L: greedy done"
  ROWS=(128:128:1:4 1024:256:1:4 4096:256:1:4 32768:256:1:2 128:128:8:32 1024:256:8:32 1024:256:32:128)
  RUNS=$R $T/bench.sh $C $L "${ROWS[@]}" 2>&1 | grep -E "==|TTFT|TPOT|ROW FAILED"
  URL=http://localhost:20000 python3 $T/realtext.py 384 > $R/$L/realtext.txt 2>&1; tail -n 2 $R/$L/realtext.txt
  for PP in "" 1.1; do for U in 1 8; do
    PRESENCE=$PP SAMPLE=0.6,-1,0.95 CONC=$U URL=http://localhost:20000 python3 $T/realtext.py 384 > $R/$L/realtext_s_pp${PP:-0}_c$U.txt 2>&1
    echo "presence=${PP:-0} users=$U: $(tail -n 1 $R/$L/realtext_s_pp${PP:-0}_c$U.txt)"
  done; done
  grep -E "TT_FATAL|EngineCore encountered|Traceback" $R/$L/server.log | head -3
  docker rm -f $C >/dev/null
done
echo "chain done $(date +%T)"

for P in "${PROFILES[@]}"; do
  serve tongyi_prefix_$P $P || continue
  URL=http://localhost:20000 TAG=$P python3 $T/prefix_bench.py repeat 2048 8192 32768 2>&1 | tee $R/$L/repeat.txt | tail -n 6
  URL=http://localhost:20000 TAG=$P python3 $T/prefix_bench.py chat 8000 4 2>&1 | tee $R/$L/chat.txt | tail -n 7
  grep -E "Prefix cache hit rate" $R/$L/server.log | tail -n 1 | grep -oE "Prefix cache hit rate: [0-9.]+%"
  grep -E "TT_FATAL|EngineCore encountered|Traceback" $R/$L/server.log | head -3
  docker rm -f $C >/dev/null
done
echo "prefix done $(date +%T)"
