#!/bin/bash
# Step 5 on the ports (attention decode ops cherry-picked to coder_next_fusion / q122_fusion): attention layer tests ops
# off / on, per-layer profile b1 / b8 off / on (GDN ops on in both), then the served A/B (fused env + GDN ops; MTP off on
# the 122B): rows + real text + greedy.
export PATH=$HOME/ornith-fusion/bin:$PATH
R=$HOME/ornith-fusion/runs
while pgrep -fx "bash $R/attn_ms.sh" >/dev/null || chips_busy; do sleep 60; done
echo "ports attn start $(date +%T)"
SHIP="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_LAYER_TESTS_POLICY=shipped --env QWEN35_GDN_DECODE_OPS=1"
K="attn and (real_weight or batched_decode or traced_decode or nonaligned or runtime_fallback or prefill_after_traced)"
for M in cn q122; do
  if [ $M = cn ]; then
    export MODEL_ID=Qwen/Qwen3-Coder-Next CACHE_NAME=qwen3-coder-next-blackhole; TREE=tt-metal-codernext; PD=qwen_qwen3_coder_next
  else
    export MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b; TREE=tt-metal-q122; PD=qwen_qwen3_5_122b_a10b
  fi
  T=$HOME/$TREE/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools; cd $T
  mkdir -p $R/${M}attn
  for F in 1 0; do
    IMAGE=ornith-dev MESH=P300x2 TNAME=${M}attn-pcc TENV="$SHIP --env QWEN35_ATTN_DECODE_OPS=$F" timeout 3600 ./trun.sh \
      "python -m pytest -q -s models/autoports/$PD/tests/test_multichip_decoder.py -k '$K' 2>&1 | grep -E 'PCC|pcc|passed|failed|FAILED|Error|assert' | grep -v 'decode user' | tail -n 40" > $R/${M}attn/pcc_f$F.log 2>&1
    echo "== $M attn layer tests ops=$F $(date +%T)"; cat $R/${M}attn/pcc_f$F.log
  done
  for F in 0 1; do
    rm -rf $R/prof/${M}attn_f$F
    PORT_DIR=$PD PROF_ENV="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=$F" \
      timeout 2400 ./run_layer_prof.sh "attn and traced_decode_batched and (b1- or b8-)" ${M}attn_f$F
    docker rm -f ornith-prof-${M}attn_f$F >/dev/null 2>&1
    f=$(find $R/prof/${M}attn_f$F -name "ops_perf_results*.csv" | sort | tail -1)
    echo "== $M attn prof ops=$F"; [ -n "$f" ] && python3 ./segments.py "$f" --labels=b1,b8 2>&1 | grep -E "==|mixer|layer"
  done
done
# served A/B
export MODEL_ID=Qwen/Qwen3-Coder-Next CACHE_NAME=qwen3-coder-next-blackhole
T=$HOME/tt-metal-codernext/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
for F in 0 1; do
  L=cnattn_f$F
  MODELS=1 GARBLE=0 KEEP=1 ROWS="128:128:1:8 2048:128:1:4 8192:64:1:2 128:128:8:32 128:128:32:128 1024:256:32:128" $T/milestone.sh $L cn-attn tt-model/qwen3-coder-next-blackhole:bd0ca927a46b p300x2 \
    --env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=$F || continue
  URL=http://localhost:20000 MODEL_ID=$MODEL_ID python3 $T/realtext.py 384 | tee $R/$L/realtext.txt | tail -n 1
  docker rm -f cn-attn >/dev/null
done
echo "== cn greedy off vs on"; python3 $T/greedy.py compare cnattn_f0 cnattn_f1 2>&1 | tail -n 3
export MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
T=$HOME/tt-metal-q122/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
for F in 0 1; do
  L=q122attn_f$F
  MODELS=1 KEEP=1 ROWS="128:128:1:8 128:128:8:32 128:128:32:128 2048:128:1:4 8192:64:1:2" $T/milestone.sh $L q122-attn tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80 p300x2 \
    --env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env VLLM_ENGINE_READY_TIMEOUT_S=14400 --env QWEN35_MTP=0 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=$F || continue
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $R/$L/realtext.txt | tail -n 1
  docker rm -f q122-attn >/dev/null
done
echo "== q122 greedy off vs on"; python3 $T/greedy.py compare q122attn_f0 q122attn_f1 2>&1 | tail -n 3
echo "ports attn done $(date +%T)"
