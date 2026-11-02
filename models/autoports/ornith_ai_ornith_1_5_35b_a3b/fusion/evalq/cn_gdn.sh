#!/bin/bash
# Step 4 on Qwen3-Coder-Next (coder_next_fusion, the gdn_decode ops cherry-picked; same GDN shapes as Ornith, 36 GDN
# layers, no MTP head, so single-user decode runs the fused mixer directly): layer tests ops off / on, per-layer profile
# b1 / b8, served A/B on p300x2 (published image + this tree, fused env, ops off vs on): rows + real text + greedy.
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3-Coder-Next CACHE_NAME=qwen3-coder-next-blackhole
T=$HOME/tt-metal-codernext/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs/cngdn; mkdir -p $R
while chips_busy; do sleep 60; done
echo "cn gdn start $(date +%T)"
cd $T
SHIP="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_LAYER_TESTS_POLICY=shipped"
K="gdn and (real_weight or batched_decode or traced_decode or nonaligned or runtime_fallback or prefill_after_traced)"
for F in 0 1; do
  IMAGE=ornith-dev MESH=P300x2 TNAME=cngdn-pcc TENV="$SHIP --env QWEN35_GDN_DECODE_OPS=$F" timeout 3600 ./trun.sh \
    "python -m pytest -q -s models/autoports/qwen_qwen3_coder_next/tests/test_multichip_decoder.py -k '$K' 2>&1 | grep -E 'PCC|pcc|passed|failed|FAILED|Error|assert' | grep -v 'decode user' | tail -n 40" > $R/pcc_f$F.log 2>&1
  echo "== cn gdn layer tests ops=$F $(date +%T)"; cat $R/pcc_f$F.log
done
for F in 0 1; do
  PORT_DIR=qwen_qwen3_coder_next PROF_ENV="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=$F" \
    timeout 2400 ./run_layer_prof.sh "gdn and traced_decode_batched and (b1- or b8-)" cngdn_f$F
  docker rm -f ornith-prof-cngdn_f$F >/dev/null 2>&1
  f=$(find $HOME/ornith-fusion/runs/prof/cngdn_f$F -name "ops_perf_results*.csv" | sort | tail -1)
  echo "== cn gdn prof ops=$F"; [ -n "$f" ] && python3 ./segments.py "$f" 2>&1 | grep -E "==|mixer|layer"
done
IMG=tt-model/qwen3-coder-next-blackhole:bd0ca927a46b
for F in 0 1; do
  L=cngdn_f$F
  MODELS=1 GARBLE=0 KEEP=1 ROWS="128:128:1:8 2048:128:1:4 8192:64:1:2 128:128:8:32 128:128:32:128 1024:256:32:128" $T/milestone.sh $L cn-gdn $IMG p300x2 \
    --env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=$F || continue
  URL=http://localhost:20000 MODEL_ID=$MODEL_ID python3 $T/realtext.py 384 | tee $HOME/ornith-fusion/runs/$L/realtext.txt | tail -n 1
  docker rm -f cn-gdn >/dev/null
done
echo "== greedy off vs on"; python3 $T/greedy.py compare cngdn_f0 cngdn_f1 2>&1 | tail -n 3
echo "cn gdn done $(date +%T)"
