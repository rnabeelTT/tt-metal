#!/bin/bash
# miro_prefix.sh: after miro_chain.sh, serve each profile again and check prefix caching (cold vs warm TTFT, identical
# greedy continuations; multi-turn chat over one document).
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs; T=$HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
B=$HOME/tt-metal-c30/build/mirothinker-1.7-mini/tt_kernel_manifest.json
while systemctl --user is-active -q miro-chain; do sleep 30; done
for P in ${@:-p300x2 p300 p150}; do
  L=miro_prefix_$P; C=tt-model-mirothinker-1.7-mini-$P; mkdir -p $R/$L
  docker rm -f $C >/dev/null 2>&1
  tt-model serve "$B" --profile $P --detach --port 20000 --no-update-check > $R/$L/serve.log 2>&1
  (docker logs -f $C 2>&1 | grep --line-buffered -v DEBUG > $R/$L/server.log &)
  until curl -sf localhost:20000/v1/models >/dev/null; do
    docker ps --format '{{.Names}}' | grep -qx $C || { echo "$L: server died"; continue 2; }; sleep 10; done
  echo "$L: up $(date +%T)"
  URL=http://localhost:20000 TAG=$P python3 $T/prefix_bench.py repeat 2048 8192 32768 2>&1 | tee $R/$L/repeat.txt | tail -n 6
  URL=http://localhost:20000 TAG=$P python3 $T/prefix_bench.py chat 8000 4 2>&1 | tee $R/$L/chat.txt | tail -n 7
  grep -E "Prefix cache hit rate" $R/$L/server.log | tail -n 1 | grep -oE "Prefix cache hit rate: [0-9.]+%"
  grep -E "TT_FATAL|EngineCore encountered|Traceback" $R/$L/server.log | head -3
  docker rm -f $C >/dev/null
done
echo "prefix done $(date +%T)"
