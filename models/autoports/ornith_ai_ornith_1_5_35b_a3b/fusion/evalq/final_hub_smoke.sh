#!/bin/bash
# Final hub smoke of every -blackhole-fast repo (user: once, at the very end): pull + serve p300x2 as a consumer would,
# greedy (shipped default) vs the card run of the same build: expect identical.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH
R=$HOME/ornith-fusion/runs
while chips_busy; do sleep 60; done
echo "final hub smoke start $(date +%T)"
smoke() {  # <name> <tree> <MODEL_ID> <CACHE_NAME|-> <repo> <reference run>
  local T=$2/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
  export MODEL_ID=$3; if [ "$4" = - ]; then unset CACHE_NAME; else export CACHE_NAME=$4; fi
  rm -rf $R/$1; mkdir -p $R/$1
  # the box had older revisions installed and card_bench serves with --no-update-check: pull the current one first
  tt-model pull raahemnabeel/$5 > $R/$1/pull.log 2>&1; echo "$1: pull exit $? $(tail -n 3 $R/$1/pull.log | tr -s ' \n' ' ')"
  GREEDY=1 $T/card_bench.sh $1 raahemnabeel/$5 p300x2 128:128:1:4 128:128:32:128
  grep -h -m2 "image\|pull" $R/$1/serve.log
  python3 $T/greedy.py compare $6 $1 2>&1 | tail -n 3
}
smoke fin_smoke_orn $HOME/tt-metal-bringups ornith-ai/Ornith-1.5-35B-A3B - ornith-1.5-35b-a3b-blackhole-fast card_fast3_p300x2
smoke fin_smoke_cn $HOME/tt-metal-codernext Qwen/Qwen3-Coder-Next - qwen3-coder-next-blackhole-fast card_cnfast3_p300x2
smoke fin_smoke_q122 $HOME/tt-metal-q122 Qwen/Qwen3.5-122B-A10B qwen3-5-122b-a10b qwen3-5-122b-a10b-blackhole-fast card_q122fast_p300x2
smoke fin_smoke_c30 $HOME/tt-metal-c30 Qwen/Qwen3-Coder-30B-A3B-Instruct qwen3-coder-30b-a3b qwen3-coder-30b-a3b-blackhole-fast card_c30new_p300x2
echo "final hub smoke done $(date +%T)"
