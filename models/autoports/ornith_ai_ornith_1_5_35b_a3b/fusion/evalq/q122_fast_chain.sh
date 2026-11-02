#!/bin/bash
# Qwen3.5-122B-A10B blackhole-fast (one fast repo: steps 1-5 + 7, package name qwen3-5-122b-a10b): package (q122_fusion,
# tt-model-fast.yaml), card rows p300x2 + p300x2-longctx served as shipped (vs runs/card_q122pub_*, the original), then
# the scorecard subsets + garble on p300x2 (vs runs/q122ev_pub).
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
Q=$HOME/tt-metal-q122
T=$Q/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
while systemctl --user is-active -q cn-fast || chips_busy; do sleep 60; done
echo "q122 fast chain start $(date +%T)"
(cd $Q && tt-model package --container models/autoports/qwen_qwen3_5_122b_a10b/tt-model-fast.yaml > $R/package_q122fast.log 2>&1; echo "package exit $?")
NEW=$Q/build/qwen3-5-122b-a10b/tt_kernel_manifest.json
[ -f $NEW ] || { echo "no package"; tail -n 20 $R/package_q122fast.log; exit 1; }
IMG=$(python3 -c "import json; print(json.load(open('$NEW'))['container']['built']['image'])"); echo "image $IMG"
rm -rf $R/card_q122fast_p300x2 $R/card_q122fast_longctx
GREEDY=1 $T/card_bench.sh card_q122fast_p300x2 $NEW p300x2
python3 $T/greedy.py compare card_q122pub_p300x2 card_q122fast_p300x2 2>&1 | tail -n 3
$T/card_bench.sh card_q122fast_longctx $NEW p300x2-longctx 128:128:1:4 2048:128:1:4
E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_MTP=0 --env QWEN35_FABRIC_BOUNDARY=1 --env QWEN35_FB_MAX_ROWS=4 --env QWEN35_FB_CONSUMER=1 --env QWEN35_FB_ROUTER=1 --env QWEN35_ROWS_TOPK=1"
rm -rf $R/q122ev_fast
GARBLE=1 KEEP=1 ROWS="128:128:1:4" $T/milestone.sh q122ev_fast q122-ev $IMG p300x2 --env VLLM_ENGINE_READY_TIMEOUT_S=14400 $E && {
  GPQA_LIMIT=198 PORT=20000 RUNS=$R $T/eval.sh q122ev_fast 2>&1 | grep -E "^==|\|.*(acc|exact|strict|loose|pass)"
  docker rm -f q122-ev >/dev/null; }
echo "q122 fast chain done $(date +%T)"
