#!/bin/bash
# Step 4 on Qwen3.5-122B-A10B (q122_fusion, the gdn_decode ops cherry-picked; Nv 16 per die at TP4): GDN layer tests ops
# off / on (4 dies, shipped policy, fused MoE + replicated boundary), per-layer profile b1 / b8 ops off / on, then the
# served A/B on p300x2 (fast image + this tree, fused env; ops off vs on; MTP on rows + real text, MTP off 1 user).
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
T=$HOME/tt-metal-q122/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs/q122gdn; mkdir -p $R
while chips_busy; do sleep 60; done
echo "q122 gdn start $(date +%T)"
cd $T
SHIP="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_LAYER_TESTS_POLICY=shipped"
K="gdn and (real_weight or batched_decode or traced_decode or nonaligned or runtime_fallback or prefill_after_traced or gdn_reset)"
for F in 0 1; do
  IMAGE=ornith-dev MESH=P300x2 TNAME=q122gdn-pcc TENV="$SHIP --env QWEN35_GDN_DECODE_OPS=$F" timeout 3600 ./trun.sh \
    "python -m pytest -q -s models/autoports/qwen_qwen3_5_122b_a10b/tests/test_multichip_decoder.py -k '$K' 2>&1 | grep -E 'PCC|pcc|passed|failed|FAILED|Error|assert' | grep -v 'decode user' | tail -n 40" > $R/pcc_f$F.log 2>&1
  echo "== q122 gdn layer tests ops=$F $(date +%T)"; cat $R/pcc_f$F.log
done
for F in 0 1; do
  PORT_DIR=qwen_qwen3_5_122b_a10b PROF_ENV="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=$F" \
    timeout 2400 ./run_layer_prof.sh "gdn and traced_decode_batched and (b1- or b8-)" q122gdn_f$F
  docker rm -f ornith-prof-q122gdn_f$F >/dev/null 2>&1
  f=$(find $HOME/ornith-fusion/runs/prof/q122gdn_f$F -name "ops_perf_results*.csv" | sort | tail -1)
  echo "== q122 gdn prof ops=$F"; [ -n "$f" ] && python3 ./segments.py "$f" 2>&1 | grep -E "==|mixer|layer"
done
IMG=tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80
FAST="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env VLLM_ENGINE_READY_TIMEOUT_S=14400"
for F in 0 1; do
  L=q122gdn_mtp_f$F
  MODELS=1 KEEP=1 ROWS="128:128:1:8 128:128:8:32 128:128:32:128 2048:128:1:4 8192:64:1:2" $T/milestone.sh $L q122-gdn $IMG p300x2 \
    $FAST --env QWEN35_GDN_DECODE_OPS=$F || continue
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $HOME/ornith-fusion/runs/$L/realtext.txt | tail -n 1
  docker rm -f q122-gdn >/dev/null
  L=q122gdn_nomtp_f$F
  MODELS=1 KEEP=1 ROWS="128:128:1:8" $T/milestone.sh $L q122-gdn $IMG p300x2 $FAST --env QWEN35_MTP=0 --env QWEN35_GDN_DECODE_OPS=$F || continue
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $HOME/ornith-fusion/runs/$L/realtext.txt | tail -n 1
  docker rm -f q122-gdn >/dev/null
done
for m in mtp nomtp; do echo "== greedy $m off vs on"; python3 $T/greedy.py compare q122gdn_${m}_f0 q122gdn_${m}_f1 2>&1 | tail -n 3; done
echo "q122 gdn done $(date +%T)"
