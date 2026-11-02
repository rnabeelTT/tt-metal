#!/bin/bash
# Fused attention decode ops: Ornith attention layer tests ops on / off (shipped policy, 4 dies), from the gdn_wip tree
export PATH=$HOME/ornith-fusion/bin:$PATH
T=$HOME/tt-metal-gdnwip/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs/attnlayer; mkdir -p $R
cd $T
SHIP="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_LAYER_TESTS_POLICY=shipped --env QWEN35_GDN_DECODE_OPS=1"
K="attn and (real_weight or batched_decode or traced_decode or nonaligned or runtime_fallback or prefill_after_traced)"
for F in ${FS:-1 0}; do
  IMAGE=ornith-dev MESH=P300x2 TNAME=attn-pcc TENV="$SHIP --env QWEN35_ATTN_DECODE_OPS=$F" timeout 3600 ./trun.sh \
    "python -m pytest -q -s models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_multichip_decoder.py -k '$K' 2>&1 | grep -E 'PCC|pcc|passed|failed|FAILED|Error|assert|tests/' | grep -v 'decode user' | tail -n 60" > $R/pcc_f$F.log 2>&1
  echo "== attn layer tests ops=$F $(date +%T)"; cat $R/pcc_f$F.log
done
echo "attn layer done $(date +%T)"
