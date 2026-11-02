#!/bin/bash
# After the step-7 ports: (1) rows top-k at the 2-die width (124160) vs torch, guard lifted (confirms the p300 garble
# junk); (2) repackage Ornith fast with the 16-bit guard; (3) p300 served as shipped: greedy + garble + 1-user row,
# then the p300 card rows (rows top-k now off on 2 dies) -> card_fast_p300.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B
C=$HOME/tt-metal-bringups
T=$C/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
while systemctl --user is-active -q ports-step7 || chips_busy; do sleep 60; done
echo "p300 fix start $(date +%T)"
IMAGE=tt-model/ornith-1.5-35b-a3b-blackhole-fast:d88d5b036f6d MESH=P300 DEVS=0,1 TNAME=rtkw timeout 1200 $T/trun.sh "python generated/rtk_wide.py 2>&1 | grep -E 'W=|Error|assert'"
docker rm -f rtkw >/dev/null 2>&1
(cd $C && tt-model package --container models/autoports/ornith_ai_ornith_1_5_35b_a3b/tt-model-fast.yaml > $R/package_ofast2.log 2>&1; echo "package exit $?")
NEW=$C/build/ornith-1.5-35b-a3b-blackhole-fast/tt_kernel_manifest.json
IMG=$(python3 -c "import json; print(json.load(open('$NEW'))['container']['built']['image'])"); echo "image $IMG"
E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_MTP=0 --env QWEN35_ROWS_TOPK=1"
rm -rf $R/ofix_p300
GARBLE=1 KEEP=1 ROWS="128:128:1:4" $T/milestone.sh ofix_p300 orn-fx $IMG p300 $E && {
  docker logs orn-fx 2>&1 | grep -E "rows top-k" | sort -u | head -3; docker rm -f orn-fx >/dev/null; }
python3 -c "
import json; d=json.load(open('$R/ofix_p300/garble.json')); print('garble', d['summary'])
print([ (r['n_words'], r['text'][:40]) for r in d['rows']][:3])"
rm -rf $R/card_fast2_p300
GREEDY=1 $T/card_bench.sh card_fast2_p300 $NEW p300
echo "p300 fix done $(date +%T)"
