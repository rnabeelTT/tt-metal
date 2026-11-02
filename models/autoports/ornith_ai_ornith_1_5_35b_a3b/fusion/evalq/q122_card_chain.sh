#!/bin/bash
# Qwen3.5-122B-A10B blackhole-fast card: package the bundle (q122_fusion), then both bundles served exactly as shipped
# (p300x2: the card rows + real text + greedy; p300x2-longctx: the MTP-off single-user rows), then the scorecard subsets
# (GPQA 40 / IFEval 100 / LCB 50, thinking off: serve.sh servers, the fast image with the bundle's env vs the published
# image). One thing on the chips at a time.
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
Q=$HOME/tt-metal-q122
T=$Q/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
while pgrep -fx "bash $R/q122_chain.sh" >/dev/null || chips_busy; do sleep 60; done
echo "q122 card chain start $(date +%T)"
(cd $Q && tt-model package --container models/autoports/qwen_qwen3_5_122b_a10b/tt-model-fast.yaml > $R/package_q122.log 2>&1; echo "package exit $?")
NEW=$Q/build/qwen3-5-122b-a10b-blackhole-fast/tt_kernel_manifest.json
PUB=raahemnabeel/qwen3-5-122b-a10b
[ -f $NEW ] || { echo "no package"; exit 1; }
GREEDY=1 $T/card_bench.sh card_q122new_p300x2 $NEW p300x2
GREEDY=1 $T/card_bench.sh card_q122pub_p300x2 $PUB p300x2
python3 $T/greedy.py compare card_q122pub_p300x2 card_q122new_p300x2 2>&1 | tail -n 3
$T/card_bench.sh card_q122new_longctx $NEW p300x2-longctx 128:128:1:4 2048:128:1:4
$T/card_bench.sh card_q122pub_longctx $PUB p300x2-longctx 128:128:1:4 2048:128:1:4
IMG=$(python3 -c "import json; print(json.load(open('$NEW'))['container']['built']['image'])")
for cfg in "q122ev_new:$IMG:--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1" "q122ev_pub:tt-model/qwen3-5-122b-a10b-blackholex4:cb33207c2577:"; do
  L=${cfg%%:*}; rest=${cfg#*:}; IM=${rest%:*}; E=${rest##*:}
  [ "$L" = q122ev_pub ] && { IM=tt-model/qwen3-5-122b-a10b-blackholex4:cb33207c2577; E=""; }
  GARBLE=0 KEEP=1 ROWS="128:128:1:4" $T/milestone.sh $L q122-ev $IM p300x2 --env VLLM_ENGINE_READY_TIMEOUT_S=14400 $E || continue
  PORT=20000 RUNS=$R $T/eval.sh $L 2>&1 | grep -E "^==|\|.*(acc|exact|strict|loose|pass)"
  docker rm -f q122-ev >/dev/null
done
echo "q122 card chain done $(date +%T)"
