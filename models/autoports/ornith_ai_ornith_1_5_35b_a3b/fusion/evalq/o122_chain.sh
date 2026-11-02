#!/bin/bash
# Ornith-1.5-122B bring-up on the Qwen3.5-122B-A10B fast code (branch ornith122_fusion): layer PCC vs HF on the real
# weights (shipped policy, fused decode), package tt-model-ornith122.yaml, card rows p300x2 + longctx profile (served as
# shipped), then evals (GPQA-198 / IFEval-100 / LCB-50) + garble gate on the packaged image.
export PATH=$HOME/ornith-fusion/bin:$HOME/.local/bin:$PATH MODEL_ID=ornith-ai/Ornith-1.5-122B CACHE_NAME=ornith-1.5-122b
Q=$HOME/tt-metal-q122; A=$Q/models/autoports/qwen_qwen3_5_122b_a10b
T=$Q/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs
set -o pipefail
while systemctl --user is-active -q o122-download || chips_busy; do sleep 60; done
grep -q "download exit 0" $R/o122_download.log || { echo "download failed"; exit 1; }
echo "o122 chain start $(date +%T)"
F="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_LAYER_TESTS_POLICY=shipped"
mkdir -p $R/o122
(cd $T && IMAGE=ornith-dev MESH=P300x2 TNAME=o122-pcc TENV="$F" timeout 7200 ./trun.sh \
  "python -m pytest -q -s models/autoports/qwen_qwen3_5_122b_a10b/tests/test_multichip_decoder.py -k '(test_mc_real_weight_against_hf and 37) or (test_mc_batched_decode_b32 and b32)' 2>&1 | grep -E 'PCC|passed|failed|Error|assert' | grep -v 'decode user' | tail -n 30" > $R/o122/pcc.log 2>&1)
echo "== pcc $(date +%T)"; cat $R/o122/pcc.log
(cd $Q && tt-model package --container models/autoports/qwen_qwen3_5_122b_a10b/tt-model-ornith122.yaml > $R/package_o122.log 2>&1; echo "package exit $?")
NEW=$Q/build/ornith-1.5-122b/tt_kernel_manifest.json
[ -f $NEW ] || { echo "no package"; tail -n 20 $R/package_o122.log; exit 1; }
IMG=$(python3 -c "import json; print(json.load(open('$NEW'))['container']['built']['image'])"); echo "image $IMG"
rm -rf $R/card_o122_p300x2 $R/card_o122_longctx
GREEDY=1 $T/card_bench.sh card_o122_p300x2 $NEW p300x2
$T/card_bench.sh card_o122_longctx $NEW p300x2-longctx 128:128:1:4 2048:128:1:4
E="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_MTP=0 --env QWEN35_FABRIC_BOUNDARY=1 --env QWEN35_FB_MAX_ROWS=4 --env QWEN35_FB_CONSUMER=1 --env QWEN35_FB_ROUTER=1 --env QWEN35_ROWS_TOPK=1"
rm -rf $R/o122ev
GARBLE=1 KEEP=1 ROWS="128:128:1:4" $T/milestone.sh o122ev o122-ev $IMG p300x2 --env VLLM_ENGINE_READY_TIMEOUT_S=14400 $E && {
  GPQA_LIMIT=198 PORT=20000 RUNS=$R $T/eval.sh o122ev 2>&1 | grep -E "^==|\|.*(acc|exact|strict|loose|pass)"
  docker rm -f o122-ev >/dev/null; }
echo "o122 chain done $(date +%T)"
