#!/bin/bash
# Hub smoke test of raahemnabeel/qwen3-coder-30b-a3b-blackhole-fast: pull + serve p300x2 as a consumer would, greedy,
# compare with the card run of the locally built bundle (same models code: expect identical text).
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct CACHE_NAME=qwen3-coder-30b-a3b
T=$HOME/tt-metal-c30/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
while chips_busy; do sleep 60; done
echo "c30 hub smoke start $(date +%T)"
GREEDY=1 $T/card_bench.sh c30_hub_smoke raahemnabeel/qwen3-coder-30b-a3b-blackhole-fast p300x2 128:128:1:4 128:128:32:128
python3 $T/greedy.py compare card_c30new_p300x2 c30_hub_smoke 2>&1 | tail -n 3
echo "c30 hub smoke done $(date +%T)"
