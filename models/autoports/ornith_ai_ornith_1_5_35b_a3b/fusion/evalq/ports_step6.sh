#!/bin/bash
# Step 6 on the ports (addnorm + GDN head fix, applied to coder_next_fusion / q122_fusion): layer tests (GDN + attention
# layers) with QWEN35_ADDNORM_DECODE=1, per-layer profile b1 / b8, then served with everything on; compared with the
# attention-ops runs (cnattn_f1 / q122attn_f1).
export PATH=$HOME/ornith-fusion/bin:$PATH
R=$HOME/ornith-fusion/runs
while pgrep -f "step6_ms.sh" >/dev/null || chips_busy; do sleep 60; done
echo "ports step6 start $(date +%T)"
OPS="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1 --env QWEN35_GDN_DECODE_OPS=1 --env QWEN35_ATTN_DECODE_OPS=1 --env QWEN35_ADDNORM_DECODE=1"
K="real_weight or batched_decode or traced_decode or nonaligned or runtime_fallback or prefill_after_traced"
for M in cn q122; do
  if [ $M = cn ]; then
    export MODEL_ID=Qwen/Qwen3-Coder-Next CACHE_NAME=qwen3-coder-next-blackhole; TREE=tt-metal-codernext; PD=qwen_qwen3_coder_next
  else
    export MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b; TREE=tt-metal-q122; PD=qwen_qwen3_5_122b_a10b
  fi
  T=$HOME/$TREE/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools; cd $T
  mkdir -p $R/${M}s6
  IMAGE=ornith-dev MESH=P300x2 TNAME=${M}s6-pcc TENV="$OPS --env QWEN35_LAYER_TESTS_POLICY=shipped" timeout 3600 ./trun.sh \
    "python -m pytest -q -s models/autoports/$PD/tests/test_multichip_decoder.py -k '$K' 2>&1 | grep -E 'PCC|pcc|passed|failed|FAILED|Error|assert' | grep -v 'decode user' | tail -n 40" > $R/${M}s6/pcc.log 2>&1
  docker rm -f ${M}s6-pcc >/dev/null 2>&1
  echo "== $M step6 layer tests $(date +%T)"; grep -E "passed|failed|FAILED|Error" $R/${M}s6/pcc.log | tail -3
  for Y in gdn attn; do
    rm -rf $R/prof/${M}s6_$Y
    PORT_DIR=$PD PROF_ENV="$OPS" timeout 2400 ./run_layer_prof.sh "$Y and traced_decode_batched and (b1- or b8-)" ${M}s6_$Y
    docker rm -f ornith-prof-${M}s6_$Y >/dev/null 2>&1
    f=$(find $R/prof/${M}s6_$Y -name "ops_perf_results*.csv" | sort | tail -1)
    echo "== $M $Y prof step6"; [ -n "$f" ] && python3 ./segments.py "$f" --labels=b1,b8 2>&1 | grep -E "==|mixer|layer"
  done
done
export MODEL_ID=Qwen/Qwen3-Coder-Next CACHE_NAME=qwen3-coder-next-blackhole
T=$HOME/tt-metal-codernext/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
L=cns6; rm -rf $R/$L
MODELS=1 GARBLE=0 KEEP=1 ROWS="128:128:1:8 2048:128:1:4 8192:64:1:2 128:128:8:32 128:128:32:128 1024:256:32:128" $T/milestone.sh $L cn-s6 tt-model/qwen3-coder-next-blackhole:bd0ca927a46b p300x2 \
  $OPS && { URL=http://localhost:20000 MODEL_ID=$MODEL_ID python3 $T/realtext.py 384 | tee $R/$L/realtext.txt | tail -n 1; docker rm -f cn-s6 >/dev/null; }
echo "== cn greedy attn vs step6"; python3 $T/greedy.py compare cnattn_f1 cns6 2>&1 | tail -n 3
export MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
T=$HOME/tt-metal-q122/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
L=q122s6; rm -rf $R/$L
MODELS=1 KEEP=1 ROWS="128:128:1:8 128:128:8:32 128:128:32:128 2048:128:1:4 8192:64:1:2" $T/milestone.sh $L q122-s6 tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80 p300x2 \
  $OPS --env VLLM_ENGINE_READY_TIMEOUT_S=14400 --env QWEN35_MTP=0 && { URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $R/$L/realtext.txt | tail -n 1; docker rm -f q122-s6 >/dev/null; }
echo "== q122 greedy attn vs step6"; python3 $T/greedy.py compare q122attn_f1 q122s6 2>&1 | tail -n 3
echo "ports step6 done $(date +%T)"
