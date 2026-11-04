#!/bin/bash
# miro_chain.sh [profiles...]: MiroThinker-1.7-mini bring-up on the local bundle, each profile served exactly as
# shipped (tt-model serve --profile): smoke (thinking, no-thinking, tool call), greedy.py, bench rows, real text.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=miromind-ai/MiroThinker-1.7-mini
R=$HOME/ornith-fusion/runs; T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
SMOKE=$HOME/tt-metal-bringups/models/autoports/osunlp_quest_35b_rl/smoke.py
B=${B:-$HOME/tt-metal-c30/build/mirothinker-1.7-mini/tt_kernel_manifest.json}
PROFILES=("$@"); [ ${#PROFILES[@]} -eq 0 ] && PROFILES=(p300x2 p300 p150)

while systemctl --user is-active -q miro-dl; do sleep 30; done
grep -qiE "error|Traceback" $R/miro_dl.log && { echo "download failed"; tail -3 $R/miro_dl.log; exit 1; }
echo "weights ready $(date +%T)"

for P in "${PROFILES[@]}"; do
  L=miro_$P; C=tt-model-mirothinker-1.7-mini-$P; mkdir -p $R/$L
  docker rm -f $C >/dev/null 2>&1
  tt-model serve "$B" --profile $P --detach --port 20000 --no-update-check > $R/$L/serve.log 2>&1
  (docker logs -f $C 2>&1 | grep --line-buffered -v DEBUG > $R/$L/server.log &)
  t0=$(date +%s); ok=1
  until curl -sf localhost:20000/v1/models >/dev/null; do
    docker ps --format '{{.Names}}' | grep -qx $C || { echo "$L: server died $(date +%T)"; ok=0; break; }
    sleep 10
  done
  [ $ok = 1 ] || { grep -E "TT_FATAL|Error|Traceback" $R/$L/server.log | tail -5; continue; }
  echo "$L: up after $(( $(date +%s) - t0 )) s $(date +%T)"
  /usr/bin/python3 $SMOKE --no-image 2>&1 | tee $R/$L/smoke.txt
  URL=http://localhost:20000 RUNS=$R python3 $T/greedy.py run $L > $R/$L/greedy.log 2>&1 && echo "$L: greedy done"
  ROWS=(128:128:1:4 1024:256:1:4 4096:256:1:4 32768:256:1:2 128:128:8:32 1024:256:8:32 1024:256:32:128)
  RUNS=$R $T/bench.sh $C $L "${ROWS[@]}" 2>&1 | grep -E "==|TTFT|TPOT|ROW FAILED"
  URL=http://localhost:20000 python3 $T/realtext.py 384 > $R/$L/realtext.txt 2>&1; tail -n 2 $R/$L/realtext.txt
  grep -E "TT_FATAL|EngineCore encountered|Traceback" $R/$L/server.log | head -3
  docker rm -f $C >/dev/null
done
echo "chain done $(date +%T)"
