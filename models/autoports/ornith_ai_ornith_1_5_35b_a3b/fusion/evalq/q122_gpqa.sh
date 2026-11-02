#!/bin/bash
# 122B GPQA diamond, all 198 questions, fused vs published (the 40-question subset gave 0.65 vs 0.775: 3 vs 8 discordant,
# McNemar p 0.23; settle it before publishing). Same servers and sampling as q122_card_chain's evals, 16 in flight.
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
Q=$HOME/tt-metal-q122
T=$Q/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
NEW=$Q/build/qwen3-5-122b-a10b-blackhole-fast/tt_kernel_manifest.json
IMG=$(python3 -c "import json; print(json.load(open('$NEW'))['container']['built']['image'])")
for L in q122gp_new q122gp_pub; do
  if [ $L = q122gp_new ]; then IM=$IMG; E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1"
  else IM=tt-model/qwen3-5-122b-a10b-blackholex4:cb33207c2577; E=""; fi
  GARBLE=0 KEEP=1 ROWS="128:128:1:4" $T/milestone.sh $L q122-ev $IM p300x2 --env VLLM_ENGINE_READY_TIMEOUT_S=14400 $E || continue
  GPQA_LIMIT=198 CONC=16 PORT=20000 RUNS=$R $T/eval.sh $L r1_gpqa_diamond 2>&1 | grep -E "^==|\|.*exact"
  docker rm -f q122-ev >/dev/null
done
echo "q122 gpqa done $(date +%T)"
