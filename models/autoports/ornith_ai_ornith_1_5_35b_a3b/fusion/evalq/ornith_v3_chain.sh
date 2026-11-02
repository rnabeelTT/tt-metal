#!/bin/bash
# Ornith-1.5-35B-A3B blackhole-fast-v2 (= v2 + step 7, see tt-model-fast-v2.yaml; MTP off on the
# multi-die profiles): package (ornith_fusion, tt-model-fast-v2.yaml), card rows p300x2 / p300 served exactly as shipped
# (vs runs/card_new_<profile>, the release-time measurement of blackhole-fast), then the scorecard subsets on p300
# (GPQA 40 / IFEval 100 / LCB 50, thinking off; vs runs/fast_p300, blackhole-fast measured the same way).
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B
C=$HOME/tt-metal-bringups
T=$C/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
while chips_busy; do sleep 60; done
echo "ornith v3 chain start $(date +%T)"
(cd $C && tt-model package --container models/autoports/ornith_ai_ornith_1_5_35b_a3b/tt-model-fast-v2.yaml > $R/package_ov3.log 2>&1; echo "package exit $?")
NEW=$C/build/ornith-1.5-35b-a3b-blackhole-fast-v2/tt_kernel_manifest.json
[ -f $NEW ] || { echo "no package"; tail -n 20 $R/package_ov3.log; exit 1; }
for P in p300x2 p300; do
  GREEDY=1 $T/card_bench.sh card_ov3_$P $NEW $P
done
IMG=$(python3 -c "import json; print(json.load(open('$NEW'))['container']['built']['image'])")
E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_MTP=0 --env QWEN35_ROWS_TOPK=1"
GARBLE=1 KEEP=1 ROWS="128:128:1:4" $T/milestone.sh ov3_p300 orn-ev $IMG p300 $E && {
  PORT=20000 RUNS=$R $T/eval.sh ov3_p300 2>&1 | grep -E "^==|\|.*(acc|exact|strict|loose|pass)"
  docker rm -f orn-ev >/dev/null; }
echo "ornith v3 chain done $(date +%T)"
