#!/bin/bash
# GLM-4.5-Air fusion (branch glm_fusion, worktree ~/tt-metal-glm), after the 122B chain:
#  1. router op test (sigb) on the GLM image;
#  2. layer PCC vs HF (real weights, 4 dies): published path / fused router / fused experts (I padded 352 -> 384);
#  3. p300x2 served (greedy, garble gate, bench rows incl. TTFT, real text): glm_rt (fused router), glm_fx (fused routed
#     experts + router, padded), glm_pub (published image as shipped). One thing on the chips at a time.
export PATH=$HOME/ornith-fusion/bin:$PATH MODEL_ID=zai-org/GLM-4.5-Air CACHE_NAME=glm-4.5-air
T=$HOME/tt-metal-glm/models/autoports/ornith_ai_ornith_1_5_35b_a3b/fusion/tools
IMG=tt-model/glm-4.5-air-blackhole:ca924a294812
R=$HOME/ornith-fusion/runs/glm; mkdir -p $R
while pgrep -fx "bash $HOME/ornith-fusion/runs/q122_chain.sh" >/dev/null || $HOME/ornith-fusion/bin/chips_busy; do sleep 60; done
echo "glm chain start $(date +%T)"
cd $T
IMAGE=glm-dev MESH=P300 DEVS=0,1 TNAME=glm-op timeout 1800 ./trun.sh \
  "python models/autoports/ornith_ai_ornith_1_5_35b_a3b/tests/test_moe_decode_ops.py sigb" > $R/op.log 2>&1
echo "glm op test exit $? $(date +%T)"; grep -E "route sigb|Error|error:|assert" $R/op.log | tail -n 12
FX="--env QWEN35_MOE_FUSED_DECODE=1 --env QWEN35_MOE_PAD_INTER=384 --env QWEN35_ROUTER_FUSED_SIGB=1"
for cfg in "base:" "rt:--env QWEN35_ROUTER_FUSED_SIGB=1" "fx:$FX"; do
  N=${cfg%%:*}; E=${cfg#*:}
  IMAGE=glm-dev MESH=P300x2 TNAME=glm-pcc TENV="$E" timeout 7200 ./trun.sh \
    "python -m pytest -q -s models/autoports/zai_org_glm_4_5_air/tests/test_multichip_decoder.py -k '(test_mc_real_weight_against_hf and 37) or (test_mc_batched_decode_b32 and b32)' 2>&1 | grep -E 'PCC|passed|failed|Error|assert' | grep -v 'decode user' | tail -n 30" > $R/pcc_$N.log 2>&1
  echo "== pcc $N $(date +%T)"; cat $R/pcc_$N.log
done
ROWS="128:128:1:8 128:128:8:32 128:128:16:64 128:128:32:128 1024:256:32:128 2048:128:1:8 8192:64:1:4"
for cfg in "glm_rt:1:--env QWEN35_ROUTER_FUSED_SIGB=1" "glm_fx:1:$FX" "glm_pub:0:"; do
  L=${cfg%%:*}; rest=${cfg#*:}; M=${rest%%:*}; E=${rest#*:}
  MODELS=$M PORT=20000 KEEP=1 ROWS="$ROWS" $T/milestone.sh $L glm-srv $IMG p300x2 --env VLLM_ENGINE_READY_TIMEOUT_S=14400 $E || continue
  docker logs glm-srv 2>&1 | grep -iE "kv cache|blocks|GiB" | grep -v DEBUG | tail -n 3 > $HOME/ornith-fusion/runs/$L/kv.log
  URL=http://localhost:20000 MODEL_ID=$MODEL_ID python3 $T/realtext.py 384 | tee $HOME/ornith-fusion/runs/$L/realtext.txt
  docker rm -f glm-srv >/dev/null
done
python3 $T/greedy.py compare glm_pub glm_rt 2>&1 | tail -n 3
python3 $T/greedy.py compare glm_pub glm_fx 2>&1 | tail -n 3
echo "glm chain done $(date +%T)"
