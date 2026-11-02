#!/bin/bash
# Qwen3.5-122B-A10B fused port (branch q122_fusion, worktree ~/tt-metal-q122): op test at its shapes, layer PCC fused vs
# unfused (real weights, 4 dies), then served p300x2 new (Ornith fast image's tt-metal + this checkout's models/) vs the
# published image: greedy, garble gate, bench rows, real text. One thing on the chips at a time.
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=Qwen/Qwen3.5-122B-A10B CACHE_NAME=qwen3-5-122b-a10b
T=$HOME/tt-metal-q122/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
R=$HOME/ornith-fusion/runs/q122; mkdir -p $R
while pgrep -f cn_chain.sh >/dev/null || pgrep -f "download Qwen/Qwen3.5-122B" >/dev/null || $HOME/ornith-fusion/bin/chips_busy; do sleep 60; done
while pgrep -fx "bash $HOME/ornith-fusion/runs/cn_gsm8k.sh" >/dev/null; do sleep 60; done
echo "q122 chain start $(date +%T)"
# general-m gate|up kernels (ornith_fusion working tree): m = 4 / 3 / 6 + the m = 1 / 2 regressions, one die of a pair
(cd $HOME/tt-metal-bringups/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
 IMAGE=ornith-dev MESH=P300 DEVS=0,1 TNAME=mcols-op timeout 2400 ./trun.sh \
   "python models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_moe_decode_ops.py mcols experts rows tp2" > $HOME/ornith-fusion/runs/mcols_op.log 2>&1
 echo "mcols op test exit $? $(date +%T)"; grep -E "^==|pcc|Error|error|assert" $HOME/ornith-fusion/runs/mcols_op.log | tail -n 40)
# the published Ornith blackhole-fast bundle from the Hub (its image was rebuilt from the benchmarked sources): boot + greedy
(MODEL_ID=ornith-ai/Ornith-1.5-35B-A3B; S=$HOME/ornith-fusion/runs/hub_smoke; mkdir -p $S
 tt-model serve raahemnabeel/ornith-1.5-35b-a3b-blackhole-fast --profile p300x2 --detach --port 20000 --no-update-check > $S/serve.log 2>&1
 t0=$(date +%s); until curl -sf localhost:20000/v1/models >/dev/null || [ $(( $(date +%s) - t0 )) -gt 3600 ]; do sleep 10; done
 echo "hub smoke: up after $(( $(date +%s) - t0 )) s"
 URL=http://localhost:20000 MODEL_ID=$MODEL_ID RUNS=$HOME/ornith-fusion/runs python3 $T/greedy.py run hub_smoke > /dev/null 2>&1 && echo "hub smoke: greedy done"
 docker rm -f tt-model-ornith-1.5-35b-a3b-blackhole-fast-p300x2 > /dev/null 2>&1)
D=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B  # HF_HUB_OFFLINE resolves refs/main
[ -f $D/refs/main ] || { mkdir -p $D/refs; printf dc4d348443bc740c68e2d77492492c11606384d5 > $D/refs/main; }
cd $T
IMAGE=ornith-dev MESH=P300 DEVS=0,1 TNAME=q122-op timeout 1800 ./trun.sh \
  "python models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_moe_decode_ops.py q122" > $R/op.log 2>&1
echo "op test exit $? $(date +%T)"; grep -iE "pcc|assert|error" $R/op.log | tail -n 12
for F in 1 0; do
  IMAGE=ornith-dev MESH=P300x2 TNAME=q122-pcc TENV="--env QWEN35_MOE_FUSED_DECODE=$F --env QWEN35_DECODE_REPLICATED=$F --env QWEN35_LAYER_TESTS_POLICY=shipped" \
    timeout 7200 ./trun.sh "python -m pytest -q -s models/autoports/qwen_qwen3_5_122b_a10b/tests/test_multichip_decoder.py -k '(test_mc_real_weight_against_hf and 37) or (test_mc_batched_decode_b32 and b32)' 2>&1 | grep -E 'PCC|passed|failed|Error|assert' | grep -v 'decode user' | tail -n 30" > $R/pcc_f$F.log 2>&1
  echo "== pcc fused=$F $(date +%T)"; cat $R/pcc_f$F.log
done
ROWS="128:128:1:8 2048:128:1:4 128:128:8:32 128:128:16:64 128:128:32:128 1024:256:32:128"
for cfg in "q122_new:1:tt-model/ornith-1.5-35b-a3b-blackhole-fast:e7d45a33dd80" "q122_pub:0:tt-model/qwen3-5-122b-a10b-blackholex4:cb33207c2577"; do
  L=${cfg%%:*}; rest=${cfg#*:}; F=${rest%%:*}; IMG=${rest#*:}
  E="--env VLLM_ENGINE_READY_TIMEOUT_S=14400"  # first boot converts the 122B checkpoint into the tensor cache
  [ $F = 1 ] && E+=" --env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_DECODE_REPLICATED=1"
  MODELS=$F PORT=20000 KEEP=1 ROWS="$ROWS" $T/milestone.sh $L q122-srv $IMG p300x2 $E || continue
  URL=http://localhost:20000 python3 $T/realtext.py 384 | tee $HOME/ornith-fusion/runs/$L/realtext.txt
  docker rm -f q122-srv >/dev/null
done
echo "q122 chain done $(date +%T)"
