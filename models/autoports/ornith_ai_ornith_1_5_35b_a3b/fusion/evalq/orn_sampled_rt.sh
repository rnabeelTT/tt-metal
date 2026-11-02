#!/bin/bash
# Ornith-35B v1 real text, greedy vs Ornith's own sampling (generation_config: T 1.0, top-k 20, top-p 0.95),
# p300x2 as shipped (question from a scorecard reader, 2026-10-28). Runs after the prefix-caching release chain.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs
T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
while systemctl --user is-active -q pc-release || chips_busy; do sleep 60; done
echo "start $(date +%T)"
tt-model pull raahemnabeel/ornith-1.5-35b-a3b-v1 > $R/rtsamp_pull.log 2>&1; echo "pull v1 exit $?"
run() {  # <label> <repo>
  local L=$1 B=$2; mkdir -p $R/$L
  local NAME=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" $HOME/.cache/tt-model/pulled/${B/\//__}/tt_kernel_manifest.json)
  local C=tt-model-$NAME-p300x2
  docker rm -f $C >/dev/null 2>&1
  tt-model serve $B --profile p300x2 --detach --port 20000 --no-update-check > $R/$L/serve.log 2>&1
  until curl -sf localhost:20000/v1/models >/dev/null; do
    docker ps --format '{{.Names}}' | grep -qx $C || { echo "$L: server died"; return 1; }
    sleep 10
  done
  echo "$L: up, image $(docker inspect -f '{{.Config.Image}}' $C)"
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $R/$L/realtext_greedy.txt | grep REALTEXT
  SAMPLE=1.0,20,0.95 URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $R/$L/realtext_sampled.txt | grep REALTEXT
  docker logs $C 2>&1 | grep -i "MTP_STATS\|accept" | tail -n 3 > $R/$L/mtp_tail.txt
  docker rm -f $C >/dev/null
}
run rtsamp_v1 raahemnabeel/ornith-1.5-35b-a3b-v1
# current build: measured 2026-10-27 (runs/mtp_sampled_off.sh): greedy 217.6, sampled 217.6 tok/s
echo "done $(date +%T)"
