#!/bin/bash
# Qwen3.5-122B-A10B blackhole-fast-v2 (= blackhole-fast + GDN / attention decode ops, MTP off): package (q122_fusion,
# tt-model-fast-v2.yaml), card rows p300x2 + p300x2-longctx served as shipped (vs runs/card_q122new_*, blackhole-fast
# at its release), then the scorecard subsets (vs runs/q122ev_new).
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
Q=$HOME/tt-metal-q122
T=$Q/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
while chips_busy; do sleep 60; done
echo "q122 v2 chain start $(date +%T)"
(cd $Q && tt-model package --container models/autoports/qwen_qwen3_5_122b_a10b/tt-model-fast-v2.yaml > $R/package_q122v2.log 2>&1; echo "package exit $?")
NEW=$Q/build/qwen3-5-122b-a10b-blackhole-fast-v2/tt_kernel_manifest.json
[ -f $NEW ] || { echo "no package"; tail -n 20 $R/package_q122v2.log; exit 1; }
GREEDY=1 $T/card_bench.sh card_q122v2_p300x2 $NEW p300x2
python3 $T/greedy.py compare card_q122new_p300x2 card_q122v2_p300x2 2>&1 | tail -n 3
$T/card_bench.sh card_q122v2_longctx $NEW p300x2-longctx 128:128:1:4 2048:128:1:4
IMG=$(python3 -c "import json; print(json.load(open('$NEW'))['container']['built']['image'])")
E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_MTP=0"
GARBLE=1 KEEP=1 ROWS="128:128:1:4" $T/milestone.sh q122ev_v2 q122-ev $IMG p300x2 --env VLLM_ENGINE_READY_TIMEOUT_S=14400 $E && {
  PORT=20000 RUNS=$R $T/eval.sh q122ev_v2 2>&1 | grep -E "^==|\|.*(acc|exact|strict|loose|pass)"
  docker rm -f q122-ev >/dev/null; }
echo "q122 v2 chain done $(date +%T)"
